"""The trusted-remote Qwen embedding adapter (T021).

Every response is validated before a vector is used; the endpoint is checked
before any text leaves the process (FR-026); credentials come only from the
environment and never appear in errors.
"""

from __future__ import annotations

import math
import socket

import numpy as np
import pytest

from scripts.v4.embeddings import (
    DIMENSION,
    QUERY_INSTRUCTION,
    QWEN_MODEL_ID,
    EmbeddingServiceError,
    EndpointConfigurationError,
    RemoteEmbedder,
    instructed_query,
    validate_endpoint,
)
from .fake_vllm import embed_text, ok_response


def _embedder(fake, **kwargs) -> RemoteEmbedder:
    kwargs.setdefault("sleep", lambda _seconds: None)
    return RemoteEmbedder(fake.base_url, **kwargs)


def _vector(value: float = 1.0, dimension: int = DIMENSION) -> list:
    return [value] + [0.0] * (dimension - 1)


def _reply(data):
    return lambda _inputs: (200, {"data": data})


# --- model pin ----------------------------------------------------------------


def test_only_the_qwen_model_is_accepted() -> None:
    with pytest.raises(ValueError, match="Qwen/Qwen3-Embedding-0.6B"):
        RemoteEmbedder("http://127.0.0.1:8000/v1", model="BAAI/bge-m3")


def test_requests_name_qwen_and_never_ask_for_reduced_dimensions(fake_vllm) -> None:
    _embedder(fake_vllm).embed_passages(["um texto"])

    (request,) = fake_vllm.requests
    assert request["model"] == QWEN_MODEL_ID
    assert "dimensions" not in request


# --- FR-026: the endpoint is checked before any text leaves ------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://vllm.example.org/v1",
        "http://127.0.0.1:8000/v1",
        "http://127.8.9.10:8000/v1",
        "http://localhost:8000/v1",
        "http://[::1]:8000/v1",
    ],
)
def test_https_or_loopback_endpoints_are_accepted(url: str) -> None:
    assert validate_endpoint(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://vllm.example.org/v1",
        "http://10.0.0.5:8000/v1",
        "http://192.168.1.20:8000/v1",
        "ftp://127.0.0.1/v1",
        "127.0.0.1:8000/v1",
        "https://user:secret@vllm.example.org/v1",
    ],
)
def test_plaintext_remote_and_malformed_endpoints_are_rejected(url: str) -> None:
    with pytest.raises(EndpointConfigurationError) as caught:
        validate_endpoint(url)
    assert "secret" not in str(caught.value)


def test_disabling_certificate_verification_is_rejected() -> None:
    with pytest.raises(EndpointConfigurationError, match="certificate"):
        validate_endpoint("https://vllm.example.org/v1", verify_tls=False)


def test_a_rejected_endpoint_fails_before_any_request(fake_vllm) -> None:
    host, port = fake_vllm.server.server_address[:2]
    # Same server, but named by a non-loopback-looking address it would
    # still answer on: the adapter must refuse before sending anything.
    with pytest.raises(EndpointConfigurationError):
        RemoteEmbedder(f"http://0.0.0.0:{port}/v1")
    assert fake_vllm.requests == []


# --- the query instruction and passage text ----------------------------------


def test_queries_carry_the_exact_versioned_instruction(fake_vllm) -> None:
    _embedder(fake_vllm).embed_queries(["Quem e o autor?"])

    assert fake_vllm.texts == [
        "Instruct: Retrieve the passage from the same legal document that answers "
        "the question.\nQuery:Quem e o autor?"
    ]
    assert instructed_query("Quem?") == QUERY_INSTRUCTION + "Quem?"


def test_passages_are_sent_unchanged(fake_vllm) -> None:
    texts = ["Primeiro trecho.\nCom quebra.", "  espacos preservados  "]

    _embedder(fake_vllm).embed_passages(texts)

    assert fake_vllm.texts == texts


# --- happy path ---------------------------------------------------------------


def test_vectors_are_float32_unit_length_and_in_input_order(fake_vllm) -> None:
    texts = ["autor joao", "valor causa", "recurso tribunal"]
    fake_vllm.script.append(
        # Out-of-order indices are fine: the adapter sorts by index.
        lambda inputs: (
            200,
            {"data": [{"index": i, "embedding": embed_text(inputs[i])} for i in (2, 0, 1)]},
        )
    )

    vectors = _embedder(fake_vllm).embed_passages(texts)

    assert vectors.shape == (3, DIMENSION) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    for row, text in zip(vectors, texts):
        expected = np.asarray(embed_text(text), dtype=np.float32)
        assert np.allclose(row, expected / np.linalg.norm(expected), atol=1e-6)


def test_requests_are_bounded_batches(fake_vllm) -> None:
    _embedder(fake_vllm, batch_size=2).embed_passages([f"texto {i}" for i in range(5)])

    assert [len(request["input"]) for request in fake_vllm.requests] == [2, 2, 1]


def test_an_empty_input_sends_nothing(fake_vllm) -> None:
    assert _embedder(fake_vllm).embed_passages([]).shape == (0, DIMENSION)
    assert fake_vllm.requests == []


# --- malformed responses fail immediately, without retry ---------------------


@pytest.mark.parametrize(
    "data, reason",
    [
        ([{"index": 0, "embedding": _vector()}], "1 results for 2 inputs"),
        ([{"index": 0, "embedding": _vector()}] * 2, "duplicate result index 0"),
        ([{"index": 0, "embedding": _vector()}, {"index": 2, "embedding": _vector()}], "indices"),
        ([{"index": 0, "embedding": _vector()}, {"index": "1", "embedding": _vector()}], "index"),
        ([{"index": 0, "embedding": _vector()}, {"index": 1, "embedding": _vector(dimension=1023)}], "1023"),
        ([{"index": 0, "embedding": _vector()}, {"index": 1, "embedding": _vector(float("nan"))}], "finite"),
        ([{"index": 0, "embedding": _vector()}, {"index": 1, "embedding": _vector(float("inf"))}], "finite"),
        ([{"index": 0, "embedding": _vector()}, {"index": 1, "embedding": ["x"] * DIMENSION}], "numeric"),
        ([{"index": 0, "embedding": _vector()}, {"index": 1, "embedding": [True] * DIMENSION}], "numeric"),
        ([{"index": 0, "embedding": _vector()}, {"index": 1, "embedding": [0.0] * DIMENSION}], "zero"),
        ([{"index": 0, "embedding": _vector()}, {"index": 1}], "embedding"),
    ],
)
def test_a_malformed_response_is_rejected_without_retry(fake_vllm, data, reason) -> None:
    fake_vllm.script.append(_reply(data))

    with pytest.raises(EmbeddingServiceError, match=reason) as caught:
        _embedder(fake_vllm).embed_passages(["primeiro", "segundo"])

    assert caught.value.retryable is False
    assert len(fake_vllm.requests) == 1


@pytest.mark.parametrize("body", [{"object": "list"}, {"data": "nope"}, b"<html>502</html>"])
def test_a_response_without_a_data_list_is_rejected(fake_vllm, body) -> None:
    fake_vllm.script.append(lambda _inputs: (200, body))

    with pytest.raises(EmbeddingServiceError, match="data"):
        _embedder(fake_vllm).embed_passages(["primeiro"])


# --- only transient failures are retried -------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_transient_failure_is_retried(fake_vllm, status) -> None:
    slept: list[float] = []
    fake_vllm.script.append(lambda _inputs: (status, {"error": "busy"}))

    vectors = _embedder(fake_vllm, sleep=slept.append).embed_passages(["texto"])

    assert vectors.shape == (1, DIMENSION)
    assert len(fake_vllm.requests) == 2
    assert len(slept) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_a_permanent_client_error_fails_at_once(fake_vllm, status) -> None:
    fake_vllm.script.append(lambda _inputs: (status, {"error": "no"}))

    with pytest.raises(EmbeddingServiceError, match=str(status)) as caught:
        _embedder(fake_vllm).embed_passages(["texto"])

    assert caught.value.retryable is False
    assert len(fake_vllm.requests) == 1


def test_retries_are_bounded(fake_vllm) -> None:
    fake_vllm.script.extend([lambda _inputs: (503, {})] * 10)

    with pytest.raises(EmbeddingServiceError, match="503") as caught:
        _embedder(fake_vllm, max_attempts=3).embed_passages(["texto"])

    assert caught.value.retryable is True
    assert len(fake_vllm.requests) == 3


def test_an_unreachable_service_is_retried_then_reported() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    slept: list[float] = []

    with pytest.raises(EmbeddingServiceError, match="127.0.0.1") as caught:
        RemoteEmbedder(
            f"http://127.0.0.1:{port}/v1", max_attempts=2, sleep=slept.append
        ).embed_passages(["texto"])

    assert caught.value.retryable is True
    assert len(slept) == 1


# --- credentials and source text never leak ----------------------------------


def test_the_api_key_comes_from_the_environment(fake_vllm, monkeypatch) -> None:
    monkeypatch.setenv("V4_EMBEDDING_API_KEY", "sk-test-123")

    _embedder(fake_vllm).embed_passages(["texto"])

    assert fake_vllm.headers[0]["Authorization"] == "Bearer sk-test-123"


def test_no_authorization_header_without_a_key(fake_vllm, monkeypatch) -> None:
    monkeypatch.delenv("V4_EMBEDDING_API_KEY", raising=False)

    _embedder(fake_vllm).embed_passages(["texto"])

    assert "Authorization" not in fake_vllm.headers[0]


def test_errors_carry_neither_the_key_nor_the_source_text(fake_vllm, monkeypatch) -> None:
    monkeypatch.setenv("V4_EMBEDDING_API_KEY", "sk-test-123")
    secret_text = "segredo processual de Joao da Silva"
    # A server that echoes its input back in the error body.
    fake_vllm.script.append(lambda inputs: (400, {"error": f"bad input {inputs}"}))
    embedder = _embedder(fake_vllm)

    with pytest.raises(EmbeddingServiceError) as caught:
        embedder.embed_passages([secret_text])

    message = str(caught.value)
    assert "sk-test-123" not in message and "segredo" not in message
    assert "sk-test-123" not in repr(embedder)


def test_the_key_is_not_accepted_as_an_argument() -> None:
    with pytest.raises(TypeError):
        RemoteEmbedder("http://127.0.0.1:8000/v1", api_key="sk-test-123")  # type: ignore[call-arg]


def test_ok_response_helper_is_well_formed() -> None:
    body = ok_response(["a", "b"])
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert all(math.isfinite(v) for v in body["data"][0]["embedding"])


# --- usage metering (retrieval cost) -----------------------------------------


def test_usage_counts_requests_texts_tokens_and_time(fake_vllm) -> None:
    from .fake_vllm import token_count

    texts = [f"texto numero {i}" for i in range(5)]
    embedder = _embedder(fake_vllm, batch_size=2)

    embedder.embed_passages(texts)

    usage = embedder.usage
    assert (usage.requests, usage.attempts, usage.texts) == (3, 3, 5)
    assert usage.prompt_tokens == sum(token_count(t) for t in texts)
    assert usage.seconds >= 0


def test_a_retry_is_counted_as_an_extra_attempt(fake_vllm) -> None:
    fake_vllm.script.append(lambda _inputs: (503, {}))
    embedder = _embedder(fake_vllm)

    embedder.embed_passages(["texto"])

    assert (embedder.usage.requests, embedder.usage.attempts) == (1, 2)


@pytest.mark.parametrize("usage", [None, {"prompt_tokens": "many"}, {"prompt_tokens": True}])
def test_tokens_are_unknown_when_the_server_does_not_report_them(fake_vllm, usage) -> None:
    def reply(inputs):
        body = ok_response(inputs)
        if usage is None:
            del body["usage"]
        else:
            body["usage"] = usage
        return 200, body

    fake_vllm.script.extend([lambda inputs: (200, ok_response(inputs)), reply])
    embedder = _embedder(fake_vllm, batch_size=1)

    embedder.embed_passages(["um", "dois"])

    # One batch reported tokens and one did not: the total is unknown, not partial.
    assert embedder.usage.prompt_tokens is None
    assert embedder.usage.requests == 2


def test_usage_is_a_snapshot(fake_vllm) -> None:
    embedder = _embedder(fake_vllm)
    before = embedder.usage

    embedder.embed_passages(["texto"])

    assert before.requests == 0 and embedder.usage.requests == 1
