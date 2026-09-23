"""Build the Qwen pgvector index for the current published generation.

    python -m scripts.v4.index DATASET_ROOT --base-url https://host/v1
        [--model Qwen/Qwen3-Embedding-0.6B] [--model-revision REV]
        [--postgres-dsn-env V4_POSTGRES_DSN] [--batch-size N]

`DATASET_ROOT/current` is resolved once. Every published chunk's text is sent
to the trusted vLLM server exactly once, in `(document_id, chunk_index,
chunk_id)` order and bounded batches, and stored as one L2-normalized
`vector(1024)` row under a new build ID. The build is checked against the
corpus and the generation manifest's `summary.chunks`, then activated in one
PostgreSQL transaction. Any failure marks the new build failed and leaves the
previously active build selected.

Credentials are read from the environment only: the DSN from the variable
named by `--postgres-dsn-env`, the API key from `V4_EMBEDDING_API_KEY`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from scripts.v4.embeddings import (
    CHUNK_TEXT_PROFILE,
    QUERY_INSTRUCTION,
    QWEN_MODEL_ID,
    RemoteEmbedder,
)
from scripts.v4.generation import load_generation, manifest_sha256
from scripts.v4.pgvector_store import (
    BuildProvenance,
    IndexedChunk,
    activate_build,
    connect,
    create_build,
    fail_build,
    insert_chunks,
    migrate,
    validate_build,
)

__all__ = ["IndexResult", "build_index", "build_parser", "main"]

DEFAULT_DSN_ENV = "V4_POSTGRES_DSN"
BASE_URL_ENV = "V4_EMBEDDING_BASE_URL"
UNPINNED_REVISION = "unpinned"


@dataclass(frozen=True)
class IndexResult:
    index_build_id: str
    generation_id: str
    generation_manifest_sha256: str
    rows: int
    previous_index_build_id: str | None


@dataclass(frozen=True)
class _Entry:
    document_id: str
    chunk_index: int
    chunk_id: str
    offset: int


def _corpus_entries(corpus: Path) -> list[_Entry]:
    """One pass recording each chunk's identity and byte offset — not its text."""
    entries: list[_Entry] = []
    with corpus.open("rb") as handle:
        offset = 0
        for line in iter(handle.readline, b""):
            if line.strip():
                record = json.loads(line)
                metadata = record["metadata"]
                entries.append(
                    _Entry(
                        str(metadata["document_id"]),
                        int(metadata["chunk_index"]),
                        str(record["_id"]),
                        offset,
                    )
                )
            offset += len(line)
    entries.sort(key=lambda e: (e.document_id, e.chunk_index, e.chunk_id))
    return entries


def _texts(corpus: Path, entries: list[_Entry]) -> Iterator[str]:
    with corpus.open("rb") as handle:
        for entry in entries:
            handle.seek(entry.offset)
            yield str(json.loads(handle.readline())["text"])


def _batches(entries: list[_Entry], size: int) -> Iterator[list[_Entry]]:
    for start in range(0, len(entries), size):
        yield entries[start : start + size]


def build_index(
    dataset_root: str | Path,
    *,
    embedder: RemoteEmbedder,
    dsn_env: str = DEFAULT_DSN_ENV,
    model_revision: str = UNPINNED_REVISION,
    progress: bool = False,
) -> IndexResult:
    """Index the generation `dataset_root/current` names, then activate it."""
    generation, manifest = load_generation(dataset_root)
    manifest_digest = manifest_sha256(generation)
    corpus_path = generation / "beir" / "corpus.jsonl"
    entries = _corpus_entries(corpus_path)
    corpus = {e.chunk_id: (e.document_id, e.chunk_index) for e in entries}

    provenance = BuildProvenance(
        model_id=embedder.model,
        model_revision=model_revision,
        generation_id=manifest.generation_id,
        generation_manifest_sha256=manifest_digest,
        source_sha256=manifest.source_sha256,
        split=manifest.split,
        query_instruction=QUERY_INSTRUCTION,
        chunk_text_profile=CHUNK_TEXT_PROFILE,
        chunk_count=manifest.chunks,
    )

    conn = connect(dsn_env)
    try:
        migrate(conn)
        build_id = create_build(conn, provenance)
        try:
            texts = _texts(corpus_path, entries)
            written = 0
            for batch in _batches(entries, embedder.batch_size):
                vectors = embedder.embed_passages([next(texts) for _ in batch])
                written += insert_chunks(
                    conn,
                    build_id,
                    (
                        IndexedChunk(e.chunk_id, e.document_id, e.chunk_index, vector)
                        for e, vector in zip(batch, vectors)
                    ),
                )
                if progress:
                    print(f"\r{written:,}/{len(entries):,} chunks", end="", file=sys.stderr)
            if progress:
                print(file=sys.stderr)
            validate_build(conn, build_id, corpus)
            previous = activate_build(conn, build_id)
        except BaseException:
            fail_build(conn, build_id)
            raise
    finally:
        conn.close()
    return IndexResult(build_id, manifest.generation_id, manifest_digest, written, previous)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset_root", type=Path, help="Directory holding current/ and generations/")
    parser.add_argument("--model", default=QWEN_MODEL_ID, help=f"Only {QWEN_MODEL_ID} is accepted")
    parser.add_argument(
        "--model-revision",
        default=UNPINNED_REVISION,
        help="Served model revision (e.g. the Hugging Face commit); part of build identity",
    )
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0, help="Seconds per request")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.base_url:
        parser.error(f"--base-url is required (or set {BASE_URL_ENV})")
    embedder = RemoteEmbedder(
        args.base_url, model=args.model, batch_size=args.batch_size, timeout=args.timeout
    )
    result = build_index(
        args.dataset_root,
        embedder=embedder,
        dsn_env=args.postgres_dsn_env,
        model_revision=args.model_revision,
        progress=True,
    )
    replaced = (
        f", superseding {result.previous_index_build_id}"
        if result.previous_index_build_id
        else ""
    )
    print(
        f"index build {result.index_build_id} is active for generation "
        f"{result.generation_id}: {result.rows:,} chunks{replaced}"
    )


if __name__ == "__main__":
    main()
