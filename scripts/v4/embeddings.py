"""The trusted remote vLLM embedding adapter, pinned to Qwen3-Embedding-0.6B.

Indexing sends every published chunk's full cleaned text, and retrieval every
question, to one operator-trusted vLLM server (`POST {base_url}/embeddings`).
This module is the only code that talks to it, so it owns every guarantee
about that exchange:

* FR-026: the endpoint is checked before any text leaves the process. It must
  be `https://` with certificate verification, or a loopback address (which
  is also how an operator-established SSH/VPN tunnel is reached).
* Credentials come from the environment (`V4_EMBEDDING_API_KEY`) only, and
  never appear in an error, log line, or repr.
* A response is used only when it has exactly one result per input, indices
  exactly 0..N-1, and 1,024 finite numeric values with nonzero norm per
  vector. Vectors are then L2-normalized to float32.
* Timeouts, connection failures, 408, 429, and 5xx are retried within a
  bound; every other failure is permanent and fails at once.
"""

from __future__ import annotations

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
    "CHUNK_TEXT_PROFILE",
    "DIMENSION",
    "QUERY_INSTRUCTION",
    "QWEN_MODEL_ID",
    "EmbeddingServiceError",
    "EmbeddingUsage",
    "EndpointConfigurationError",
    "RemoteEmbedder",
    "instructed_query",
    "validate_endpoint",
]

#: The only model this feature indexes or queries.
QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DIMENSION = 1024
API_KEY_ENV = "V4_EMBEDDING_API_KEY"

#: Versioned query instruction, in the Qwen model-card format. Changing it
#: makes every existing index build incompatible, by design.
QUERY_INSTRUCTION = (
    "Instruct: Retrieve the passage from the same legal document that answers "
    "the question.\nQuery:"
)

#: How chunk text is composed before embedding: the corpus `text` field
#: exactly as published — no title, no instruction. Part of build identity.
CHUNK_TEXT_PROFILE = "corpus-text:no-title:no-instruction:v1"

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


def instructed_query(question: str) -> str:
    """The exact text embedded for one question."""
    return QUERY_INSTRUCTION + question


def _is_loopback(host: str) -> bool:
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_endpoint(base_url: str, *, verify_tls: bool = True) -> str:
    """Return `base_url` if it satisfies FR-026, else raise.

    Accepted: `https://` with certificate verification, or plain `http://`
    to a loopback address. Everything else — plaintext to any other host,
    disabled verification, embedded credentials — is refused.
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
    if not _is_loopback(host):
        raise EndpointConfigurationError(
            f"plaintext http to {host} would send source text unencrypted; use "
            "https, or reach the server through a loopback tunnel "
            "(e.g. ssh -L 8000:localhost:8000 host, then http://127.0.0.1:8000/v1)"
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
        if model != QWEN_MODEL_ID:
            raise ValueError(
                f"only {QWEN_MODEL_ID} may be indexed or queried in this feature; "
                f"got {model!r}"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.base_url = validate_endpoint(base_url, verify_tls=verify_tls)
        self.model = model
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
        """Embed chunk text exactly as given (no instruction)."""
        return self._embed(list(texts))

    def embed_queries(self, questions: Sequence[str]) -> np.ndarray:
        """Embed questions under the versioned Qwen query instruction."""
        return self._embed([instructed_query(question) for question in questions])

    def _embed(self, texts: list[str]) -> np.ndarray:
        batches = [
            self._embed_batch(texts[start : start + self.batch_size])
            for start in range(0, len(texts), self.batch_size)
        ]
        if not batches:
            return np.zeros((0, DIMENSION), dtype=np.float32)
        return np.concatenate(batches)

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        payload = self._post({"model": self.model, "input": batch})
        vectors = _validated(payload, len(batch), self._url)
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
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                self._count_attempt(time.perf_counter() - started)
                reason = getattr(error, "reason", error)
                failure = EmbeddingServiceError(
                    f"embedding service {self._url} unreachable: "
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


def _validated(payload: Any, expected: int, url: str) -> np.ndarray:
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

    vectors = np.empty((expected, DIMENSION), dtype=np.float32)
    for index in range(expected):
        embedding = by_index[index].get("embedding")
        if not isinstance(embedding, list):
            raise reject(f"result {index} has no embedding list")
        if len(embedding) != DIMENSION:
            raise reject(
                f"result {index} has {len(embedding)} dimensions, expected {DIMENSION}"
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
