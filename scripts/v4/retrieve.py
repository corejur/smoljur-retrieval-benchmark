"""Document-scoped retrieval from the stored Qwen pgvector build.

    python -m scripts.v4.retrieve DATASET_BEIR RUN_FILE --base-url https://host/v1
        [--model Qwen/Qwen3-Embedding-0.6B] [--model-revision REV]
        [--postgres-dsn-env V4_POSTGRES_DSN] [--split test] [--top-k N]
        [--batch-size N] [--format json|trec] [--provenance PATH]

This module ranks and writes a run; it computes no metric (see metrics.py).
Chunk vectors come from the active index build in PostgreSQL; only questions
are sent to the trusted vLLM server, under the versioned Qwen instruction.
Each question is ranked exactly against its own document's published
candidates, filtered before cosine-distance ordering.

`DATASET_BEIR` is resolved once to an immutable generation, and one active,
compatible build is pinned before any question leaves the process. The run
file and its provenance record are each replaced atomically, only after the
whole ranking succeeded; a failure leaves both untouched. Provenance lives
outside the runs directory so metrics.py never reads it as a run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.v4.embeddings import (
    CHUNK_TEXT_PROFILE,
    DIMENSION,
    QUERY_INSTRUCTION,
    QWEN_MODEL_ID,
    EmbeddingUsage,
    RemoteEmbedder,
)
from scripts.v4.generation import manifest_sha256, pin_generation_path, read_manifest
from scripts.v4.pgvector_store import (
    IndexBuild,
    IndexBuildError,
    active_build,
    chunk_documents,
    connect,
    search,
)

__all__ = [
    "IndexCompatibilityError",
    "RankedRun",
    "RetrievalInputError",
    "build_parser",
    "check_compatible",
    "default_provenance_path",
    "load_candidates",
    "load_queries",
    "main",
    "rank",
    "rank_queries",
    "write_run",
]

DEFAULT_DSN_ENV = "V4_POSTGRES_DSN"
BASE_URL_ENV = "V4_EMBEDDING_BASE_URL"
PRODUCTION_SPLIT = "test"


class RetrievalInputError(ValueError):
    """The questions or candidates cannot be ranked as published."""


class IndexCompatibilityError(IndexBuildError):
    """The active build does not match this generation or text profile."""


@dataclass(frozen=True)
class RankedRun:
    run: dict[str, dict[str, float]]
    provenance: dict[str, Any] = field(default_factory=dict)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_queries(path: str | Path) -> dict[str, str]:
    queries = {str(r["_id"]): str(r["text"]) for r in _read_jsonl(Path(path))}
    if not queries:
        raise RetrievalInputError(f"no queries in {path}")
    return queries


def load_candidates(path: str | Path) -> dict[str, tuple[str, list[str]]]:
    """Return {query_id: (document_id, [chunk_id])} from the candidate manifest."""
    candidates: dict[str, tuple[str, list[str]]] = {}
    for record in _read_jsonl(Path(path)):
        query_id = str(record["query_id"])
        if query_id in candidates:
            raise RetrievalInputError(f"duplicate query {query_id!r} in {path}")
        candidates[query_id] = (
            str(record["document_id"]),
            [str(c) for c in record["candidate_ids"]],
        )
    if not candidates:
        raise RetrievalInputError(f"no candidate records in {path}")
    return candidates


def check_compatible(
    build: IndexBuild,
    *,
    generation_id: str,
    generation_manifest_sha256: str,
    chunk_count: int,
    model_revision: str,
) -> None:
    """Refuse a build that was made for other data or other text handling."""
    provenance = build.provenance
    expected = {
        "generation_id": generation_id,
        "generation manifest_sha256": generation_manifest_sha256,
        "model revision": model_revision,
        "query_instruction": QUERY_INSTRUCTION,
        "chunk_text_profile": CHUNK_TEXT_PROFILE,
        "dimension": DIMENSION,
        "metric": "cosine",
        "normalized": True,
    }
    actual = {
        "generation_id": provenance.generation_id,
        "generation manifest_sha256": provenance.generation_manifest_sha256,
        "model revision": provenance.model_revision,
        "query_instruction": provenance.query_instruction,
        "chunk_text_profile": provenance.chunk_text_profile,
        "dimension": provenance.dimension,
        "metric": provenance.metric,
        "normalized": provenance.normalized,
    }
    for name, value in expected.items():
        if actual[name] != value:
            raise IndexCompatibilityError(
                f"active build {build.index_build_id} has {name} {actual[name]!r}, "
                f"but this retrieval needs {value!r}; rebuild the index"
            )
    if not build.row_count == provenance.chunk_count == chunk_count:
        raise IndexCompatibilityError(
            f"active build {build.index_build_id} holds {build.row_count} rows; "
            f"the build declares {provenance.chunk_count} and the generation "
            f"publishes {chunk_count} chunks"
        )


def rank_queries(
    conn,
    build_id: str,
    queries: Mapping[str, str],
    candidates: Mapping[str, tuple[str, Sequence[str]]],
    embedder: RemoteEmbedder,
    *,
    top_k: int,
    latencies: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    """Rank every query within its own document's candidates.

    Candidates are checked against the pinned build before any question is
    sent: each must be indexed, and indexed under the query's own document.
    When `latencies` is given, each query's search time (seconds) is
    appended to it.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    indexed = chunk_documents(conn, build_id)
    for query_id in queries:
        if query_id not in candidates or not candidates[query_id][1]:
            raise RetrievalInputError(f"query {query_id} has no published candidates")
        document_id, candidate_ids = candidates[query_id]
        for chunk_id in candidate_ids:
            if chunk_id not in indexed:
                raise RetrievalInputError(
                    f"query {query_id} lists candidate {chunk_id}, which build "
                    f"{build_id} does not index"
                )
            if indexed[chunk_id] != document_id:
                raise RetrievalInputError(
                    f"query {query_id} lists candidate {chunk_id} from document "
                    f"{indexed[chunk_id]}, not its own document {document_id}"
                )

    query_ids = sorted(queries)
    vectors = embedder.embed_queries([queries[q] for q in query_ids])
    run: dict[str, dict[str, float]] = {}
    for query_id, vector in zip(query_ids, vectors):
        document_id, candidate_ids = candidates[query_id]
        started = time.perf_counter()
        run[query_id] = dict(
            search(conn, build_id, document_id, candidate_ids, vector, top_k=top_k)
        )
        if latencies is not None:
            latencies.append(time.perf_counter() - started)
    return run


def _usage_since(before: EmbeddingUsage, after: EmbeddingUsage) -> dict[str, Any]:
    tokens = (
        after.prompt_tokens - before.prompt_tokens
        if after.prompt_tokens is not None and before.prompt_tokens is not None
        else None
    )
    return EmbeddingUsage(
        requests=after.requests - before.requests,
        attempts=after.attempts - before.attempts,
        texts=after.texts - before.texts,
        prompt_tokens=tokens,
        seconds=after.seconds - before.seconds,
    ).to_dict()


def _latency_summary(latencies: Sequence[float]) -> dict[str, float]:
    """Mean and nearest-rank percentiles, in milliseconds."""
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(latencies)

    def percentile(p: float) -> float:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]

    return {
        name: round(value * 1000, 3)
        for name, value in (
            ("mean", sum(ordered) / len(ordered)),
            ("p50", percentile(0.50)),
            ("p95", percentile(0.95)),
            ("max", ordered[-1]),
        )
    }


def rank(
    beir_directory: str | Path,
    *,
    embedder: RemoteEmbedder,
    dsn_env: str = DEFAULT_DSN_ENV,
    top_k: int = 100,
    model_revision: str = "unpinned",
    split: str = PRODUCTION_SPLIT,
) -> RankedRun:
    """Pin one generation and one compatible build, then rank every question."""
    if split != PRODUCTION_SPLIT:
        raise RetrievalInputError(f"only the {PRODUCTION_SPLIT!r} split is published; got {split!r}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    started = time.perf_counter()
    usage_before = embedder.usage
    latencies: list[float] = []

    # Resolve `current` once; every file below comes from this snapshot.
    beir = pin_generation_path(beir_directory)
    generation = beir.parent
    manifest = read_manifest(generation)
    manifest_digest = manifest_sha256(generation)

    conn = connect(dsn_env)
    try:
        build = active_build(conn, manifest.generation_id, embedder.model)
        check_compatible(
            build,
            generation_id=manifest.generation_id,
            generation_manifest_sha256=manifest_digest,
            chunk_count=manifest.chunks,
            model_revision=model_revision,
        )
        queries = load_queries(beir / "queries.jsonl")
        candidates = load_candidates(beir / "candidates" / f"{split}.jsonl")
        run = rank_queries(
            conn,
            build.index_build_id,
            queries,
            candidates,
            embedder,
            top_k=top_k,
            latencies=latencies,
        )
    finally:
        conn.close()

    provenance = {
        "generation_id": manifest.generation_id,
        "generation_manifest_sha256": manifest_digest,
        "index_build_id": build.index_build_id,
        "model_id": build.provenance.model_id,
        "model_revision": build.provenance.model_revision,
        "query_instruction": build.provenance.query_instruction,
        "chunk_text_profile": build.provenance.chunk_text_profile,
        "normalization": "l2",
        "score": "cosine_similarity",
        "split": split,
        "top_k": top_k,
        "queries": len(run),
        # What this run consumed: the service's own token count, requests and
        # retries, time waiting on it, and the exact-search cost per query.
        "cost": {
            "wall_seconds": round(time.perf_counter() - started, 3),
            "embedding": _usage_since(usage_before, embedder.usage),
            "search": {
                "queries": len(latencies),
                "seconds": round(sum(latencies), 3),
                "latency_ms": _latency_summary(latencies),
            },
        },
    }
    return RankedRun(run, provenance)


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_run(
    run: Mapping[str, Mapping[str, float]],
    destination: str | Path,
    *,
    fmt: str = "json",
    tag: str = "v4",
) -> None:
    """Replace `destination` with the complete run, or leave it untouched."""
    if fmt == "json":
        text = json.dumps(
            {q: dict(r) for q, r in run.items()}, ensure_ascii=False, allow_nan=False
        )
    else:
        lines = [
            f"{query_id} Q0 {chunk_id} {position} {score:.6f} {tag}"
            for query_id, ranked in run.items()
            for position, (chunk_id, score) in enumerate(
                sorted(ranked.items(), key=lambda item: (-item[1], item[0])), start=1
            )
        ]
        text = "\n".join(lines) + "\n"
    _write_atomically(Path(destination), text)


def default_provenance_path(run_file: str | Path) -> Path:
    """`<runs>.provenance/<run>.json`, beside — never inside — the runs directory."""
    run_path = Path(run_file)
    runs = run_path.parent
    return runs.parent / f"{runs.name}.provenance" / f"{run_path.stem}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("beir_directory", type=Path, help="<dataset>/current/beir")
    parser.add_argument("output", type=Path, help="Run file for metrics.py")
    parser.add_argument("--model", default=QWEN_MODEL_ID, help=f"Only {QWEN_MODEL_ID} is accepted")
    parser.add_argument("--model-revision", default="unpinned", help="Must match the index build")
    parser.add_argument(
        "--base-url",
        default=os.environ.get(BASE_URL_ENV),
        help=f"Trusted vLLM base URL ending in /v1 (default: ${BASE_URL_ENV})",
    )
    parser.add_argument(
        "--postgres-dsn-env",
        default=DEFAULT_DSN_ENV,
        help="Name of the environment variable holding the PostgreSQL DSN",
    )
    parser.add_argument("--split", default=PRODUCTION_SPLIT)
    parser.add_argument("--batch-size", type=int, default=64, help="Questions per request")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--format", choices=("json", "trec"), default="json")
    parser.add_argument(
        "--provenance",
        type=Path,
        help="Provenance record (default: <runs>.provenance/<run>.json)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Seconds per request")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.base_url:
        parser.error(f"--base-url is required (or set {BASE_URL_ENV})")
    if args.top_k <= 0 or args.batch_size <= 0:
        parser.error("--top-k and --batch-size must be positive")
    provenance_path = args.provenance or default_provenance_path(args.output)
    runs = args.output.resolve().parent
    if runs == provenance_path.resolve().parent or runs in provenance_path.resolve().parents:
        parser.error(
            f"--provenance {provenance_path} is inside the runs directory {runs}; "
            "metrics.py would read it as a competing run"
        )

    embedder = RemoteEmbedder(
        args.base_url, model=args.model, batch_size=args.batch_size, timeout=args.timeout
    )
    ranked = rank(
        args.beir_directory,
        embedder=embedder,
        dsn_env=args.postgres_dsn_env,
        top_k=args.top_k,
        model_revision=args.model_revision,
        split=args.split,
    )
    write_run(ranked.run, args.output, fmt=args.format)
    _write_atomically(
        provenance_path,
        json.dumps({"run_file": str(args.output), **ranked.provenance}, indent=2) + "\n",
    )
    total = sum(len(r) for r in ranked.run.values())
    cost = ranked.provenance["cost"]
    tokens = cost["embedding"]["prompt_tokens"]
    print(
        f"cost: {cost['wall_seconds']:.1f}s total, "
        f"{cost['embedding']['requests']:,} embedding requests "
        f"({cost['embedding']['attempts'] - cost['embedding']['requests']} retries, "
        f"{'unknown' if tokens is None else f'{tokens:,}'} prompt tokens, "
        f"{cost['embedding']['seconds']:.1f}s), search p50 "
        f"{cost['search']['latency_ms']['p50']:.1f} ms / p95 "
        f"{cost['search']['latency_ms']['p95']:.1f} ms",
        file=sys.stderr,
    )
    print(
        f"wrote {args.output}: {len(ranked.run):,} queries, {total:,} ranked chunks "
        f"from build {ranked.provenance['index_build_id']} "
        f"(generation {ranked.provenance['generation_id']}, top_k={args.top_k}); "
        f"provenance {provenance_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
