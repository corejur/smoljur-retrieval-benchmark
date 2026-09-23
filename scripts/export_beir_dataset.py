"""Export the prepared Representative Redator data in BEIR benchmark format."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.persistence import write_jsonl, write_tsv


DEFAULT_QUERIES_FILE = "representative-redator-2k-v1-v4.csv"


@dataclass(frozen=True)
class BeirExportSummary:
    corpus_count: int
    query_count: int
    qrel_count: int
    skipped_query_count: int


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise ValueError(f"input CSV does not exist: {path}")
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"input CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _require_fields(path: Path, fields: list[str], required: set[str]) -> None:
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _parse_relevant_chunks(raw_value: str, *, query_id: str) -> list[str]:
    try:
        value: Any = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"query {query_id!r} has invalid relevant_chunks JSON"
        ) from error
    if not isinstance(value, list) or any(
        not isinstance(chunk_id, str) or not chunk_id for chunk_id in value
    ):
        raise ValueError(
            f"query {query_id!r} relevant_chunks must be a JSON list of strings"
        )
    return list(dict.fromkeys(value))


def _beir_query_id(row: dict[str, str]) -> str:
    source_id = row["source_id"].strip()
    query_number = row["source_query_number"].strip()
    if not source_id or not query_number:
        raise ValueError("source_id and source_query_number cannot be empty")
    return f"{source_id}:q{query_number}"


def export_beir_dataset(
    data_directory: str | Path = "data",
    output_directory: str | Path = "data/beir/representative-redator",
    *,
    split: str = "test",
    queries_file: str = DEFAULT_QUERIES_FILE,
) -> BeirExportSummary:
    """Convert prepared chunk and query CSVs into a loadable BEIR dataset.

    Queries without relevant chunks are omitted because BEIR evaluation requires
    at least one relevance judgment for every evaluated query.
    """
    data_path = Path(data_directory)
    destination = Path(output_directory)
    query_path = data_path / "queries" / queries_file
    chunks_directory = data_path / "chunks"

    query_fields, query_rows = _read_csv(query_path)
    _require_fields(
        query_path,
        query_fields,
        {"source_id", "source_query_number", "query", "relevant_chunks"},
    )
    if not chunks_directory.is_dir():
        raise ValueError(f"chunks directory does not exist: {chunks_directory}")

    corpus_rows: list[dict[str, str]] = []
    corpus_ids: set[str] = set()
    for chunk_path in sorted(chunks_directory.glob("*.csv")):
        fields, chunks = _read_csv(chunk_path)
        _require_fields(chunk_path, fields, {"chunk_id", "text", "section"})
        for chunk in chunks:
            chunk_id = chunk["chunk_id"].strip()
            if not chunk_id:
                raise ValueError(f"empty chunk_id in {chunk_path}")
            if chunk_id in corpus_ids:
                raise ValueError(f"duplicate chunk_id: {chunk_id}")
            corpus_ids.add(chunk_id)
            corpus_rows.append(
                {
                    "_id": chunk_id,
                    "title": chunk["section"].strip(),
                    "text": chunk["text"],
                }
            )

    beir_queries: list[dict[str, str]] = []
    qrels: list[dict[str, str | int]] = []
    query_ids: set[str] = set()
    skipped = 0
    for row in query_rows:
        query_id = _beir_query_id(row)
        relevant_chunks = _parse_relevant_chunks(
            row["relevant_chunks"], query_id=query_id
        )
        if not relevant_chunks:
            skipped += 1
            continue
        if query_id in query_ids:
            raise ValueError(f"duplicate BEIR query ID: {query_id}")
        missing_chunks = sorted(set(relevant_chunks) - corpus_ids)
        if missing_chunks:
            raise ValueError(
                f"query {query_id!r} references missing chunks: "
                + ", ".join(missing_chunks)
            )
        query_text = row["query"].strip()
        if not query_text:
            raise ValueError(f"query {query_id!r} has empty text")
        query_ids.add(query_id)
        beir_queries.append({"_id": query_id, "text": query_text})
        qrels.extend(
            {
                "query-id": query_id,
                "corpus-id": chunk_id,
                "score": 1,
            }
            for chunk_id in relevant_chunks
        )

    if not corpus_rows:
        raise ValueError("no chunks were found to export")
    if not beir_queries:
        raise ValueError("no queries with relevance judgments were found")

    write_jsonl(destination / "corpus.jsonl", corpus_rows)
    write_jsonl(destination / "queries.jsonl", beir_queries)
    write_tsv(
        destination / "qrels" / f"{split}.tsv",
        qrels,
        ("query-id", "corpus-id", "score"),
    )
    return BeirExportSummary(
        corpus_count=len(corpus_rows),
        query_count=len(beir_queries),
        qrel_count=len(qrels),
        skipped_query_count=skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path, nargs="?", default=Path("data"))
    parser.add_argument(
        "output_directory",
        type=Path,
        nargs="?",
        default=Path("data/beir/representative-redator"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--queries-file", default=DEFAULT_QUERIES_FILE)
    args = parser.parse_args()
    summary = export_beir_dataset(
        args.data_directory,
        args.output_directory,
        split=args.split,
        queries_file=args.queries_file,
    )
    print(
        f"Wrote {summary.corpus_count} corpus entries, "
        f"{summary.query_count} queries, and {summary.qrel_count} qrels to "
        f"{args.output_directory}; skipped "
        f"{summary.skipped_query_count} queries without relevance judgments"
    )


if __name__ == "__main__":
    main()
