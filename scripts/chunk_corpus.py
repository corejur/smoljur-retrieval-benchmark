"""Run legal chunking and persist one CSV per source document."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from scripts.persistence import write_csv, write_json
from scripts.strategies.chunking.legal_recursive import (
    Chunk,
    ChunkingConfig,
    chunk_documents,
)

DEFAULT_INPUT = Path("data/documents/documents.cleaned.json")
DEFAULT_OUTPUT_DIRECTORY = Path("data/chunks")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_document_name(document_id: str) -> str:
    value = _SAFE_NAME.sub("_", document_id).strip("._")
    if not value:
        raise ValueError(f"document ID {document_id!r} cannot form a filename")
    return value


def write_chunk_csvs(
    chunks: Sequence[Chunk], output_directory: str | Path
) -> int:
    """Write one atomic plain-text chunk CSV per document."""
    output = Path(output_directory)
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, []).append(chunk)
    for document_id, document_chunks in grouped.items():
        destination = output / f"{_safe_document_name(document_id)}.csv"
        write_csv(
            destination,
            [chunk.to_dict() for chunk in document_chunks],
            list(document_chunks[0].to_dict()),
        )
    return len(grouped)
def write_chunk_dataset(
    chunks: Sequence[Chunk],
    output_directory: str | Path,
    config: ChunkingConfig,
) -> tuple[int, int]:
    """Persist one complete chunk generation and remove stale chunk CSVs."""
    output = Path(output_directory)
    expected = {
        f"{_safe_document_name(chunk.document_id)}.csv" for chunk in chunks
    }
    document_count = write_chunk_csvs(chunks, output)
    for chunk_path in output.glob("*.csv"):
        if chunk_path.name not in expected:
            chunk_path.unlink()
    write_json(
        output / "manifest.json",
        {
            "strategy": "legal_recursive",
            "strategy_version": config.strategy_version,
            "length_unit": "whitespace_delimited_words",
            "stores_token_ids": False,
            "stores_embeddings": False,
            "configuration": asdict(config),
            "document_count": document_count,
            "chunk_count": len(chunks),
            "chunk_files": sorted(expected),
        },
    )
    return document_count, len(chunks)




def chunk_corpus(
    input_json: str | Path,
    output_directory: str | Path,
    config: ChunkingConfig,
) -> tuple[int, int]:
    """Chunk a cleaned document corpus and atomically persist its outputs."""
    input_path = Path(input_json)
    output = Path(output_directory)
    with input_path.open("r", encoding="utf-8") as source:
        documents = json.load(source)
    if not isinstance(documents, list):
        raise ValueError("input JSON must contain a document array")

    chunks = chunk_documents(documents, config=config)
    return write_chunk_dataset(chunks, output, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument(
        "output_directory",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--max-words", type=int, default=400)
    parser.add_argument("--target-words", type=int, default=300)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument("--minimum-words", type=int, default=50)
    args = parser.parse_args()
    config = ChunkingConfig(
        max_words=args.max_words,
        target_words=args.target_words,
        overlap_words=args.overlap_words,
        minimum_words=args.minimum_words,
    )
    document_count, chunk_count = chunk_corpus(
        args.input_json, args.output_directory, config
    )
    print(
        f"Wrote {chunk_count} chunks for {document_count} documents "
        f"to {args.output_directory}"
    )


if __name__ == "__main__":
    main()
