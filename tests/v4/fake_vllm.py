"""A stand-in for the trusted vLLM server, served over real HTTP on loopback.

The embedding is a bag of hashed words, so texts sharing words get similar
vectors and rankings in tests are predictable. Behavior can be scripted per
request to exercise every failure the adapter must reject.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

DIMENSION = 1024
MODEL = "Qwen/Qwen3-Embedding-0.6B"


def embed_text(text: str, dimension: int = DIMENSION) -> list[float]:
    vector = [0.0] * dimension
    for word in re.findall(r"\w+", text.lower()):
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:4], "big") % dimension] += 1.0
    if not any(vector):
        vector[0] = 1.0
    return vector


def token_count(text: str) -> int:
    """The fake server's token count: one per word."""
    return len(re.findall(r"\w+", text))


def ok_response(inputs: list[str]) -> dict[str, Any]:
    tokens = sum(token_count(text) for text in inputs)
    return {
        "object": "list",
        "model": MODEL,
        # Same shape as vLLM's OpenAI-compatible usage block.
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens, "completion_tokens": 0},
        "data": [
            {"object": "embedding", "index": i, "embedding": embed_text(text)}
            for i, text in enumerate(inputs)
        ],
    }


#: A scripted reply: (status, JSON body) for the given request inputs.
Reply = Callable[[list[str]], "tuple[int, Any]"]


@dataclass
class FakeVllm:
    server: ThreadingHTTPServer
    thread: threading.Thread
    requests: list[dict[str, Any]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    #: Replies consumed one per request; the default answers correctly.
    script: list[Reply] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def texts(self) -> list[str]:
        return [text for request in self.requests for text in request["input"]]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def start_fake_vllm() -> FakeVllm:
    state: dict[str, FakeVllm] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:  # keep test output clean
            pass

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            fake = state["fake"]
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            fake.requests.append(payload)
            fake.headers.append(dict(self.headers.items()))
            if self.path != "/v1/embeddings":
                status, body = 404, {"error": "not found"}
            elif fake.script:
                status, body = fake.script.pop(0)(payload["input"])
            else:
                status, body = 200, ok_response(payload["input"])
            encoded = (
                body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    fake = FakeVllm(server, thread)
    state["fake"] = fake
    thread.start()
    return fake


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))
