"""Document-scoped retrieval that writes a run file and computes nothing.

Scoring lives in `metrics.py` alone, so every model is measured by identical
code. This module's only job is to produce a ranking: encode the queries and
the chunks, score each query against its own document's chunks, and write the
result. Compare models by collecting run files, not by re-running evaluation.

The default encoder talks to any OpenAI-compatible `/v1/embeddings` endpoint,
which is what a vLLM server exposes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

__all__ = [
    "Encoder",
    "HttpEmbeddingEncoder",
    "SentenceTransformerEncoder",
    "load_corpus",
    "load_candidates",
    "retrieve",
    "write_run",
]


class Encoder(Protocol):
    """Turns text into vectors. `kind` is "query" or "passage"."""

    def encode(self, texts: Sequence[str], *, kind: str) -> list[list[float]]:
        ...


class HttpEmbeddingEncoder:
    """Embeddings from an OpenAI-compatible endpoint, such as vLLM's."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        batch_size: int = 64,
        timeout: float = 120.0,
        retries: int = 3,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = retries
        self.prefixes = {"query": query_prefix, "passage": passage_prefix}

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.retries):
            request = urllib.request.Request(self.url, data=body, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last = error
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"embedding request to {self.url} failed: {last}")

    def encode(self, texts: Sequence[str], *, kind: str) -> list[list[float]]:
        prefix = self.prefixes.get(kind, "")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [prefix + text for text in texts[start : start + self.batch_size]]
            payload = self._post({"model": self.model, "input": batch})
            data = sorted(payload["data"], key=lambda item: item["index"])
            if len(data) != len(batch):
                raise RuntimeError(
                    f"endpoint returned {len(data)} embeddings for {len(batch)} inputs"
                )
            vectors.extend(item["embedding"] for item in data)
        return vectors


class SentenceTransformerEncoder:
    """Local encoder, for when the model runs in-process rather than served."""

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 32,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.batch_size = batch_size
        self.prefixes = {"query": query_prefix, "passage": passage_prefix}

    def encode(self, texts: Sequence[str], *, kind: str) -> list[list[float]]:
        prefix = self.prefixes.get(kind, "")
        return self.model.encode(
            [prefix + text for text in texts],
            batch_size=self.batch_size,
            show_progress_bar=True,
        ).tolist()


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_corpus(path: str | Path, *, use_title: bool = True) -> dict[str, str]:
    """Return {chunk_id: text}, prefixing the section title when present."""
    corpus: dict[str, str] = {}
    for record in _read_jsonl(path):
        title = str(record.get("title") or "").strip() if use_title else ""
        text = str(record.get("text") or "")
        corpus[str(record["_id"])] = f"{title}\n{text}".strip() if title else text
    if not corpus:
        raise ValueError(f"no corpus records in {path}")
    return corpus


def load_queries(path: str | Path) -> dict[str, str]:
    queries = {str(r["_id"]): str(r["text"]) for r in _read_jsonl(path)}
    if not queries:
        raise ValueError(f"no queries in {path}")
    return queries


def load_candidates(path: str | Path) -> dict[str, list[str]]:
    """Return {query_id: [chunk_id]} - the pool each query may be scored on."""
    candidates: dict[str, list[str]] = {}
    for record in _read_jsonl(path):
        query_id = str(record["query_id"])
        if query_id in candidates:
            raise ValueError(f"duplicate query {query_id!r} in {path}")
        candidates[query_id] = [str(c) for c in record["candidate_ids"]]
    if not candidates:
        raise ValueError(f"no candidate records in {path}")
    return candidates


def retrieve(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    candidates: Mapping[str, Sequence[str]],
    encoder: Encoder,
    *,
    top_k: int = 100,
    normalize: bool = True,
) -> dict[str, dict[str, float]]:
    """Rank each query against its own document's chunks only."""
    import numpy as np

    missing = set(queries) - set(candidates)
    if missing:
        raise ValueError(f"{len(missing)} queries have no candidate pool")
    pool_ids = sorted({chunk_id for ids in candidates.values() for chunk_id in ids})
    absent = [chunk_id for chunk_id in pool_ids if chunk_id not in corpus]
    if absent:
        raise ValueError(
            f"{len(absent)} candidate chunks are absent from the corpus; "
            f"example: {absent[0]}"
        )

    query_ids = sorted(queries)
    print(f"encoding {len(query_ids):,} queries", file=sys.stderr)
    query_vectors = np.asarray(
        encoder.encode([queries[q] for q in query_ids], kind="query"), dtype="float32"
    )
    print(f"encoding {len(pool_ids):,} chunks", file=sys.stderr)
    chunk_vectors = np.asarray(
        encoder.encode([corpus[c] for c in pool_ids], kind="passage"), dtype="float32"
    )
    if normalize:
        query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-12
        chunk_vectors /= np.linalg.norm(chunk_vectors, axis=1, keepdims=True) + 1e-12

    index = {chunk_id: position for position, chunk_id in enumerate(pool_ids)}
    run: dict[str, dict[str, float]] = {}
    for position, query_id in enumerate(query_ids):
        pool = candidates[query_id]
        rows = np.fromiter((index[c] for c in pool), dtype="int64", count=len(pool))
        scores = chunk_vectors[rows] @ query_vectors[position]
        order = np.argsort(-scores)[:top_k]
        run[query_id] = {pool[int(i)]: float(scores[int(i)]) for i in order}
    return run


def write_run(
    run: Mapping[str, Mapping[str, float]],
    destination: str | Path,
    *,
    fmt: str = "json",
    tag: str = "v4",
) -> None:
    """Write the ranking for metrics.py to score."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(
            json.dumps({q: dict(r) for q, r in run.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
        return
    lines: list[str] = []
    for query_id, ranked in run.items():
        ordered = sorted(ranked.items(), key=lambda item: -item[1])
        lines.extend(
            f"{query_id} Q0 {chunk_id} {rank} {score:.6f} {tag}"
            for rank, (chunk_id, score) in enumerate(ordered, start=1)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("beir_directory", type=Path, help="<out>/beir")
    parser.add_argument("output", type=Path, help="Run file for metrics.py")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--encoder",
        choices=("http", "sentence-transformers"),
        default="http",
        help="http talks to a vLLM / OpenAI-compatible server",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--passage-prefix", default="")
    parser.add_argument("--no-title", action="store_true", help="Ignore chunk titles")
    parser.add_argument("--score", choices=("cos_sim", "dot"), default="cos_sim")
    parser.add_argument("--format", choices=("json", "trec"), default="json")
    args = parser.parse_args()

    beir = args.beir_directory
    corpus = load_corpus(beir / "corpus.jsonl", use_title=not args.no_title)
    queries = load_queries(beir / "queries.jsonl")
    candidates = load_candidates(beir / "candidates" / f"{args.split}.jsonl")

    if args.encoder == "http":
        encoder: Encoder = HttpEmbeddingEncoder(
            args.base_url,
            args.model,
            api_key=args.api_key,
            batch_size=args.batch_size,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
        )
    else:
        encoder = SentenceTransformerEncoder(
            args.model,
            batch_size=args.batch_size,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
        )

    run = retrieve(
        corpus,
        queries,
        candidates,
        encoder,
        top_k=args.top_k,
        normalize=args.score == "cos_sim",
    )
    write_run(run, args.output, fmt=args.format)
    ranked = sum(len(r) for r in run.values())
    print(
        f"wrote {args.output}: {len(run):,} queries, {ranked:,} ranked chunks "
        f"(model={args.model}, score={args.score}, top_k={args.top_k})"
    )


if __name__ == "__main__":
    main()
