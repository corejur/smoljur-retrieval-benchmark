"""Build queries, cleaned documents, chunks, and audits from one source CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scripts.artifact_publication import publish_generation
from scripts.chunk_corpus import write_chunk_dataset
from scripts.citation_ground_truth import (
    generate_ground_truth,
    remove_untraceable_documents,
)
from scripts.corpus_generator import extract_documents
from scripts.data_cleaner import clean_documents
from scripts.persistence import write_csv, write_json
from scripts.query_preparation import (
    filter_fully_answered_sources,
    prepare_query_rows,
)
from scripts.strategies.chunking.legal_recursive import (
    ChunkingConfig,
    chunk_documents,
)


@dataclass(frozen=True)
class PreparationSummary:
    query_count: int
    document_count: int
    chunk_count: int
    removed_document_count: int
    removed_untraceable_document_count: int


def _build_dataset(
    source_csv: str | Path,
    output_directory: str | Path = "data",
    *,
    chunking: ChunkingConfig = ChunkingConfig(),
) -> PreparationSummary:
    """Run the ground-truth-free dataset preparation pipeline."""
    source = Path(source_csv)
    output = Path(output_directory)
    queries_path = output / "queries" / "representative-redator-2k-v1-v4.csv"
    documents_path = output / "documents" / "documents.json"
    cleaned_path = output / "documents" / "documents.cleaned.json"
    audit_directory = output / "data_preparation_output"
    chunks_directory = output / "chunks"

    csv.field_size_limit(sys.maxsize)
    with source.open("r", encoding="utf-8", newline="") as source_file:
        source_reader = csv.DictReader(source_file)
        if source_reader.fieldnames is None:
            raise ValueError("source CSV has no header")
        source_rows = list(source_reader)

    selected = filter_fully_answered_sources(source_rows)
    existing_rows: list[dict[str, str]] = []
    if queries_path.exists() and queries_path.stat().st_size:
        with queries_path.open("r", encoding="utf-8", newline="") as query_file:
            existing_rows = list(csv.DictReader(query_file))
    prepared = prepare_query_rows(source_rows, existing_rows)
    selected_indexes = set(selected.source_indexes)
    query_rows = [
        row for row in prepared.rows if int(row["source_row"]) in selected_indexes
    ]
    write_csv(queries_path, query_rows, prepared.fieldnames)

    documents = extract_documents(selected.rows)
    write_json(documents_path, documents)
    write_json(
        audit_directory / "remove_incomplete_documents.json",
        [
            {
                "source_row": source_index,
                "reason": "answer_count_does_not_match_query_count",
                "row": dict(source_row),
            }
            for source_index, source_row in enumerate(source_rows)
            if source_index not in selected_indexes
        ],
    )

    cleaned = clean_documents(
        documents,
        queries_path,
        audit_directory,
        queries_output=queries_path,
    )
    write_json(cleaned_path, cleaned)
    with queries_path.open("r", encoding="utf-8", newline="") as query_file:
        query_reader = csv.DictReader(query_file)
        query_fieldnames = list(query_reader.fieldnames or [])
        final_query_rows = list(query_reader)
    chunks_by_document: defaultdict[str, list] = defaultdict(list)
    for chunk in chunk_documents(cleaned, config=chunking):
        chunks_by_document[chunk.document_id].append(chunk)
    ground_truth = generate_ground_truth(
        source_rows,
        final_query_rows,
        {document["document_id"]: document["text"] for document in cleaned},
        chunks_by_document,
    )
    traceable = remove_untraceable_documents(ground_truth, cleaned)
    untraceable_ids = set(traceable.removed_document_ids)
    traceable_raw_documents = [
        document
        for document in documents
        if str(document["document_id"]) not in untraceable_ids
    ]
    write_json(documents_path, traceable_raw_documents)
    write_json(cleaned_path, traceable.documents)

    relevant_fieldnames = [
        field for field in query_fieldnames if field != "relevant_chunks"
    ] + ["relevant_chunks"]
    write_csv(queries_path, traceable.query_rows, relevant_fieldnames)
    write_json(
        audit_directory / "unmapped_citations.json",
        ground_truth.unmapped_citations,
    )
    write_json(
        audit_directory / "remove_untraceable_citations.json",
        [
            {
                "source_row": source_index,
                "reason": "document_has_incomplete_ground_truth",
                "unmapped_citations": [
                    entry
                    for entry in ground_truth.unmapped_citations
                    if entry["document_id"] == source_row.get("id", "")
                ],
                "queries_without_relevant_chunks": [
                    {
                        "source_query_number": row["source_query_number"],
                        "query": row["query"],
                        "citations": row.get("citations", "[]"),
                    }
                    for row in ground_truth.rows
                    if row["source_id"] == source_row.get("id", "")
                    and not json.loads(
                        str(row.get("relevant_chunks", "[]"))
                    )
                ],
                "row": dict(source_row),
            }
            for source_index, source_row in enumerate(source_rows)
            if source_row.get("id", "") in untraceable_ids
        ],
    )

    retained_document_ids = {
        str(document["document_id"]) for document in traceable.documents
    }
    retained_chunks = [
        chunk
        for document_id, chunks in chunks_by_document.items()
        if document_id in retained_document_ids
        for chunk in chunks
    ]
    document_count, chunk_count = write_chunk_dataset(
        retained_chunks, chunks_directory, chunking
    )
    return PreparationSummary(
        query_count=len(traceable.query_rows),
        document_count=document_count,
        chunk_count=chunk_count,
        removed_document_count=len(selected.removed_document_ids),
        removed_untraceable_document_count=len(
            traceable.removed_document_ids
        ),
    )

def prepare_dataset(
    source_csv: str | Path,
    output_directory: str | Path = "data",

    *,
    chunking: ChunkingConfig = ChunkingConfig(),
) -> PreparationSummary:
    """Build and publish one complete dataset generation."""
    return publish_generation(
        output_directory,
        lambda staging: _build_dataset(source_csv, staging, chunking=chunking),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path)
    parser.add_argument(
        "output_directory", type=Path, nargs="?", default=Path("data")
    )
    parser.add_argument("--max-words", type=int, default=400)
    parser.add_argument("--target-words", type=int, default=300)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument("--minimum-words", type=int, default=50)
    args = parser.parse_args()
    summary = prepare_dataset(
        args.source_csv,
        args.output_directory,
        chunking=ChunkingConfig(
            max_words=args.max_words,
            target_words=args.target_words,
            overlap_words=args.overlap_words,
            minimum_words=args.minimum_words,
        ),
    )
    print(
        f"Wrote {summary.query_count} queries, "
        f"{summary.document_count} documents, and "
        f"{summary.chunk_count} chunks to {args.output_directory}; "
        f"removed {summary.removed_document_count} documents without complete answers; "
        f"removed {summary.removed_untraceable_document_count} documents with incomplete ground truth"
    )


if __name__ == "__main__":
    main()
