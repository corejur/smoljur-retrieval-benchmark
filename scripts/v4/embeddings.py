"""The trusted remote vLLM embedding adapter for the benchmarked models.

Indexing sends every published chunk's full cleaned text, and retrieval every
question, to one operator-trusted vLLM server (`POST {base_url}/embeddings`).
This module is the only code that talks to it, so it owns every guarantee
about that exchange:

* Only a model registered in `MODELS` is sent text, each under its own
  versioned query instruction and passage prefix (its `EmbeddingModel`).

* FR-026: the endpoint is checked before any text leaves the process. It must
  be `https://` with certificate verification, or a loopback address (which
  is also how an operator-established SSH/VPN tunnel is reached).
* Credentials come from the environment (`V4_EMBEDDING_API_KEY`) only, and
  never appear in an error, log line, or repr.
* A response is used only when it has exactly one result per input, indices
  exactly 0..N-1, and 1,024 finite numeric values with nonzero norm per
  vector. Vectors are then L2-normalized to float32.
* Timeouts, connection failures (including a response cut off mid-body),
  408, 429, and 5xx are retried within a bound; every other failure is permanent and fails at once.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

import numpy as np

__all__ = [
    "API_KEY_ENV",
    "TUNNEL_HOSTS_ENV",
    "CHUNK_TEXT_PROFILE",
    "BGE_M3",
    "BGE_M3_MODEL_ID",
    "DIMENSION",
    "JINA_V5_TEXT_SMALL",
    "JINA_V5_TEXT_SMALL_MODEL_ID",
    "MODELS",
    "QUERY_INSTRUCTION",
    "QWEN",
    "QWEN3_4B",
    "QWEN3_4B_MODEL_ID",
    "QWEN_MODEL_ID",
    "EmbeddingModel",
    "EmbeddingServiceError",
    "EmbeddingUsage",
    "EndpointConfigurationError",
    "RemoteEmbedder",
    "instructed_query",
    "model_profile",
    "validate_endpoint",
]

QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_4B_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
JINA_V5_TEXT_SMALL_MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
BGE_M3_MODEL_ID = "BAAI/bge-m3"
#: The default model output size. Every model is stored at its own full
#: (non-Matryoshka) size, `EmbeddingModel.dimension`; this is the common one.
DIMENSION = 1024
API_KEY_ENV = "V4_EMBEDDING_API_KEY"

#: Comma-separated hosts the operator declares as tunnel-local: the local end
#: of an encrypted tunnel they established (a WireGuard/VPN interface
#: address, say). Plain http to these is allowed; to any other non-loopback
#: host it is not. The declaration is the operator's statement that the
#: channel is encrypted — the code cannot observe the tunnel itself.
TUNNEL_HOSTS_ENV = "V4_EMBEDDING_TUNNEL_HOSTS"

#: Versioned query instruction, in the Qwen model-card format. Changing it
#: makes every existing index build incompatible, by design.
QUERY_INSTRUCTION = (
    "Instruct: Retrieve the passage from the same legal document that answers "
    "the question.\nQuery:"
)

#: How chunk text is composed before embedding: the corpus `text` field
#: exactly as published — no title, no instruction. Part of build identity.
CHUNK_TEXT_PROFILE = "corpus-text:no-title:no-instruction:v1"



@dataclass(frozen=True)
class EmbeddingModel:
    """How one model's text is composed before embedding.

    `query_instruction` is prepended to every question and `passage_prefix`
    to every chunk's text; `chunk_text_profile` names the passage composition.
    `dimension` is the model's full output size, which every vector must have
    (no reduced Matryoshka size is ever requested). All four are part of build
    identity: changing one makes existing index builds for the model
    incompatible, by design.
    """

    model_id: str
    query_instruction: str
    passage_prefix: str
    chunk_text_profile: str
    dimension: int = DIMENSION

    def query_text(self, question: str) -> str:
        return self.query_instruction + question

    def passage_text(self, text: str) -> str:
        return self.passage_prefix + text


QWEN = EmbeddingModel(QWEN_MODEL_ID, QUERY_INSTRUCTION, "", CHUNK_TEXT_PROFILE)

#: The larger Qwen3-Embedding: same instruction format, 2,560 dimensions.
QWEN3_4B = EmbeddingModel(
    QWEN3_4B_MODEL_ID, QUERY_INSTRUCTION, "", CHUNK_TEXT_PROFILE, dimension=2560
)

#: vLLM (0.30+) merges the model's `retrieval` LoRA adapter at load time but
#: does not add its prompts, so the `Query: `/`Document: ` prefixes from the
#: model card are sent with the text.
JINA_V5_TEXT_SMALL = EmbeddingModel(
    JINA_V5_TEXT_SMALL_MODEL_ID,
    query_instruction="Query: ",
    passage_prefix="Document: ",
    chunk_text_profile="corpus-text:no-title:document-prefix:v1",
)

#: BGE-M3 dense retrieval takes queries and passages without any instruction.
BGE_M3 = EmbeddingModel(BGE_M3_MODEL_ID, "", "", CHUNK_TEXT_PROFILE)

#: The models this pipeline may index or query, by the ID sent to vLLM.
MODELS = {model.model_id: model for model in (QWEN, QWEN3_4B, JINA_V5_TEXT_SMALL, BGE_M3)}


def model_profile(model_id: str) -> EmbeddingModel:
    """The registered text profile for `model_id`, or ValueError."""
    try:
        return MODELS[model_id]
    except KeyError:
        raise ValueError(
            f"{model_id!r} is not a benchmarked model; use one of {', '.join(MODELS)}"
        ) from None


_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain"})


class EndpointConfigurationError(ValueError):
    """The configured endpoint would send text over an unprotected channel."""


class EmbeddingServiceError(RuntimeError):
    """The service failed or answered with something unusable.

    `retryable` says whether the failure was transient (and was retried up
    to the bound) or permanent. The message never carries request text,
    response bodies, or credentials.
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class EmbeddingUsage:
    """What the embedding service was asked for so far — the run's cost.

    `attempts` exceeds `requests` by the retries. `prompt_tokens` is the sum
    of the server's own `usage.prompt_tokens`, or None when any response did
    not report it (an unknown is never passed off as a partial total).
    `seconds` is time spent waiting on the service, retries included.
    """

    requests: int = 0
    attempts: int = 0
    texts: int = 0
    prompt_tokens: int | None = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "attempts": self.attempts,
            "texts": self.texts,
            "prompt_tokens": self.prompt_tokens,
            "seconds": round(self.seconds, 3),
        }


def _reported_tokens(payload: Any) -> int | None:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        return None
    return tokens


def instructed_query(question: str, model: str = QWEN_MODEL_ID) -> str:
    """The exact text embedded for one question under `model`."""
    return model_profile(model).query_text(question)


def _is_loopback(host: str) -> bool:
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _declared_tunnel_hosts() -> frozenset[str]:
    raw = os.environ.get(TUNNEL_HOSTS_ENV, "")
    return frozenset(host.strip().lower() for host in raw.split(",") if host.strip())


def validate_endpoint(base_url: str, *, verify_tls: bool = True) -> str:
    """Return `base_url` if it satisfies FR-026, else raise.

    Accepted: `https://` with certificate verification, or plain `http://`
    to a loopback address or to a host the operator declared tunnel-local in
    `V4_EMBEDDING_TUNNEL_HOSTS`. Everything else — plaintext to any other
    host, disabled verification, embedded credentials — is refused.
    """
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise EndpointConfigurationError(
            "embedding endpoint must be an http(s) URL with a host, "
            f"such as https://host/v1 (got scheme {parts.scheme or 'none'!r})"
        )
    if parts.username or parts.password:
        raise EndpointConfigurationError(
            "embedding endpoint must not embed credentials in the URL; "
            f"set {API_KEY_ENV} in the environment instead"
        )
    host = parts.hostname
    if parts.scheme == "https":
        if not verify_tls:
            raise EndpointConfigurationError(
                "certificate verification must stay enabled for an https "
                "embedding endpoint"
            )
        return base_url
    if not _is_loopback(host) and host.lower() not in _declared_tunnel_hosts():
        raise EndpointConfigurationError(
            f"plaintext http to {host} would send source text unencrypted; use "
            "https, reach the server through a loopback tunnel "
            "(e.g. ssh -L 8000:localhost:8000 host, then http://127.0.0.1:8000/v1), "
            f"or declare {host} tunnel-local in {TUNNEL_HOSTS_ENV}"
        )
    return base_url


class RemoteEmbedder:
    """Embeddings from the trusted vLLM server, validated and normalized."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = QWEN_MODEL_ID,
        batch_size: int = 32,
        timeout: float = 120.0,
        max_attempts: int = 4,
        verify_tls: bool = True,
        api_key_env: str = API_KEY_ENV,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        profile = model_profile(model)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.base_url = validate_endpoint(base_url, verify_tls=verify_tls)
        self.model = model
        self.profile = profile
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._url = base_url.rstrip("/") + "/embeddings"
        self._api_key = os.environ.get(api_key_env) or None
        self._sleep = sleep
        self._ssl = (
            ssl.create_default_context() if urlsplit(base_url).scheme == "https" else None
        )
        self._usage = EmbeddingUsage()

    @property
    def usage(self) -> EmbeddingUsage:
        """A snapshot of the cumulative usage of this embedder."""
        return self._usage

    def __repr__(self) -> str:
        return f"RemoteEmbedder({self.base_url!r}, model={self.model!r})"

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Embed chunk text under the model's passage prefix (none for Qwen)."""
        return self._embed([self.profile.passage_text(text) for text in texts])

    def embed_queries(self, questions: Sequence[str]) -> np.ndarray:
        """Embed questions under the model's versioned query instruction."""
        return self._embed([self.profile.query_text(question) for question in questions])

    def _embed(self, texts: list[str]) -> np.ndarray:
        batches = [
            self._embed_batch(texts[start : start + self.batch_size])
            for start in range(0, len(texts), self.batch_size)
        ]
        if not batches:
            return np.zeros((0, self.profile.dimension), dtype=np.float32)
        return np.concatenate(batches)

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        payload = self._post({"model": self.model, "input": batch})
        vectors = _validated(payload, len(batch), self._url, self.profile.dimension)
        tokens = _reported_tokens(payload)
        before = self._usage
        self._usage = EmbeddingUsage(
            requests=before.requests + 1,
            attempts=before.attempts,
            texts=before.texts + len(batch),
            prompt_tokens=(
                before.prompt_tokens + tokens
                if before.prompt_tokens is not None and tokens is not None
                else None
            ),
            seconds=before.seconds,
        )
        return vectors

    def _count_attempt(self, seconds: float) -> None:
        before = self._usage
        self._usage = EmbeddingUsage(
            before.requests,
            before.attempts + 1,
            before.texts,
            before.prompt_tokens,
            before.seconds + seconds,
        )

    def _post(self, payload: dict[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        failure: EmbeddingServiceError | None = None
        for attempt in range(self.max_attempts):
            if attempt:
                self._sleep(min(2.0**attempt, 30.0))
            request = urllib.request.Request(self._url, data=body, headers=headers)
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout, context=self._ssl
                ) as response:
                    raw = response.read()
            except urllib.error.HTTPError as error:
                self._count_attempt(time.perf_counter() - started)
                # The body is deliberately dropped: servers echo the input.
                retryable = error.code in _RETRYABLE_STATUS
                failure = EmbeddingServiceError(
                    f"embedding service {self._url} answered HTTP {error.code}",
                    retryable=retryable,
                )
                if not retryable:
                    raise failure from None
                continue
            except (
                urllib.error.URLError,
                http.client.HTTPException,  # e.g. IncompleteRead: the connection dropped mid-body
                TimeoutError,
                ConnectionError,
                OSError,
            ) as error:
                self._count_attempt(time.perf_counter() - started)
                reason = getattr(error, "reason", error)
                failure = EmbeddingServiceError(
                    f"embedding service {self._url} unreachable or disconnected: "
                    f"{type(reason).__name__}",
                    retryable=True,
                )
                continue
            self._count_attempt(time.perf_counter() - started)
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise EmbeddingServiceError(
                    f"embedding service {self._url} returned a body without a "
                    "JSON data list",
                    retryable=False,
                ) from None
        assert failure is not None
        raise EmbeddingServiceError(
            f"{failure} (after {self.max_attempts} attempts)", retryable=True
        )


def _validated(payload: Any, expected: int, url: str, dimension: int = DIMENSION) -> np.ndarray:
    """Check one response against the contract and return unit vectors."""

    def reject(reason: str) -> EmbeddingServiceError:
        return EmbeddingServiceError(f"embedding service {url}: {reason}", retryable=False)

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise reject("response has no data list")
    if len(data) != expected:
        raise reject(f"{len(data)} results for {expected} inputs")

    by_index: dict[int, Any] = {}
    for item in data:
        index = item.get("index") if isinstance(item, dict) else None
        if isinstance(index, bool) or not isinstance(index, int):
            raise reject(f"result index {index!r} is not an integer")
        if index in by_index:
            raise reject(f"duplicate result index {index}")
        by_index[index] = item
    if sorted(by_index) != list(range(expected)):
        raise reject(
            f"result indices {sorted(by_index)} are not exactly 0..{expected - 1}"
        )

    vectors = np.empty((expected, dimension), dtype=np.float32)
    for index in range(expected):
        embedding = by_index[index].get("embedding")
        if not isinstance(embedding, list):
            raise reject(f"result {index} has no embedding list")
        if len(embedding) != dimension:
            raise reject(
                f"result {index} has {len(embedding)} dimensions, expected {dimension}"
            )
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in embedding):
            raise reject(f"result {index} holds a non-numeric value")
        values = np.asarray(embedding, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise reject(f"result {index} holds a value that is not finite")
        norm = float(np.linalg.norm(values))
        if not math.isfinite(norm) or norm == 0.0:
            raise reject(f"result {index} is a zero vector")
        vectors[index] = values / norm
    return vectors
