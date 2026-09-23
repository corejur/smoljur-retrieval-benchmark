"""Build the v4 retrieval dataset in one streaming pass.

Per source row: read the question list, normalize and clean the document while
retaining offsets, resolve each citation to an exact span, chunk the cleaned
text, and emit BEIR records. Nothing accumulates across rows except counters,
so train-scale input is bounded by the largest single document.

Policy difference from v1-v3: an unusable query is dropped, but its document
is kept. Rejecting whole documents discarded 26% of v4, most of them for
answers that correctly cite nothing ("Nao", "NENHUMA DAS TESES"). Keeping the
documents also keeps their chunks in the corpus as honest distractors.

A query is retained only when the source both answered it in text and cited at
least one passage, and those citations resolve to spans. Every drop is written
to audits/dropped_queries.jsonl with the reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from scripts.artifact_publication import publish_generation
from scripts.strategies.chunking.legal_recursive import ChunkingConfig, LengthCounter
from scripts.v4.chunking import ENCODING_NAME, V4_CHUNKING, chunk_cleaned_document, o200k_counter
from scripts.v4.cleaning import plain_text_violations
from scripts.v4.generation import validate_generation
from scripts.v4.ground_truth import map_citations_to_chunks, prepare_document
from scripts.v4.questions import read_questions
from scripts.v4.source_contract import (
    PRODUCTION_SPLIT,
    V4_TEST_ROW_COUNT,
    V4_TEST_SHA256,
    SourceFingerprint,
    iter_source_rows,
    verify_source_file,
)
from scripts.v4.streams import JsonlWriter, TsvWriter

__all__ = [
    "RunSummary",
    "build_dataset",
    "new_generation_id",
    "run",
    "iter_csv_rows",
]

MANAGED_DIRECTORIES = ("beir", "documents", "evidence", "audits", "meta")


def new_generation_id() -> str:
    """A sortable, unique identity for one published generation."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


@dataclass(frozen=True)
class RunSummary:
    source_rows: int
    rejected_sources: int
    documents: int
    chunks: int
    queries: int
    queries_retained: int
    qrels: int
    unmapped_citations: int
    plain_text_violations: int


def iter_csv_rows(path: str | Path) -> Iterator[dict[str, str]]:
    """Stream a source CSV without materializing it."""
    csv.field_size_limit(sys.maxsize)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"source CSV has no header: {path}")
        yield from reader


def build_dataset(
    source_rows: Iterable[Mapping[str, Any]],
    output_directory: str | Path,
    *,
    split: str = "test",
    config: ChunkingConfig = V4_CHUNKING,
    counter: LengthCounter | None = None,
    data_column: str = "data",
    require_answer_text: bool = True,
    max_citation_words: int | None = None,
    keep_documents_without_queries: bool = False,
    max_gold_chunks: int | None = 10,
    fingerprint: SourceFingerprint | None = None,
    generation_id: str | None = None,
) -> RunSummary:
    """Write one complete generation into `output_directory`.

    `fingerprint` carries the gated source's provenance. A production build
    always supplies one; a synthetic-fixture build does not, and records an
    empty source SHA-256 to say so.
    """
    output = Path(output_directory)
    citation_limit = (
        config.max_words if max_citation_words is None else max_citation_words
    )
    rows = documents = chunk_total = queries = evaluable = qrels = 0
    rejected = unmapped_total = impure = 0

    with (
        JsonlWriter(output / "beir" / "corpus.jsonl") as corpus,
        JsonlWriter(output / "beir" / "queries.jsonl") as query_file,
        TsvWriter(
            output / "beir" / "qrels" / f"{split}.tsv",
            ("query-id", "corpus-id", "score"),
        ) as qrel_file,
        JsonlWriter(output / "documents" / "documents.jsonl") as document_file,
        JsonlWriter(output / "evidence" / "evidence.jsonl") as evidence_file,
        JsonlWriter(output / "audits" / "rejected_sources.jsonl") as rejected_file,
        JsonlWriter(output / "audits" / "unmapped_citations.jsonl") as unmapped_file,
        JsonlWriter(output / "audits" / "dropped_queries.jsonl") as dropped_file,
        JsonlWriter(output / "audits" / "plain_text_violations.jsonl") as impure_file,
        JsonlWriter(output / "beir" / "candidates" / f"{split}.jsonl") as candidate_file,
        JsonlWriter(output / "audits" / "cleaning_events.jsonl") as cleaning_file,
    ):
        for index, row in enumerate(source_rows):
            rows += 1
            source = read_questions(row)
            if not source.valid:
                rejected += 1
                rejected_file.write(
                    {
                        "source_row": index,
                        "document_id": source.document_id,
                        "reason": source.rejected_reason,
                    }
                )
                continue

            prepared = prepare_document(
                str(row.get(data_column, "") or ""),
                source.records,
                document_id=source.document_id,
                max_citation_words=citation_limit,
            )
            if not prepared.text.strip():
                rejected += 1
                rejected_file.write(
                    {
                        "source_row": index,
                        "document_id": source.document_id,
                        "reason": "empty_document_after_cleaning",
                    }
                )
                continue

            chunks = chunk_cleaned_document(
                source.document_id, prepared.text, config=config, counter=counter
            )
            if not chunks:
                rejected += 1
                rejected_file.write(
                    {
                        "source_row": index,
                        "document_id": source.document_id,
                        "reason": "no_chunks_produced",
                    }
                )
                continue

            chunk_records = [chunk.to_dict() for chunk in chunks]
            document_chunk_ids = [chunk.chunk_id for chunk in chunks]

            # Decide which queries survive before committing the document:
            # under document-scoped retrieval a document with no query is
            # never in any candidate pool, so its chunks are dead weight.
            kept: list = []
            for item in prepared.evidence:
                queries += 1
                if require_answer_text and not item.query.has_answer_text:
                    reason: str | None = "answer_has_no_text"
                elif not item.query.citations:
                    reason = "answer_has_no_citations"
                elif not item.has_evidence:
                    reason = "all_citations_unmapped"
                else:
                    reason = None
                # One chunk per golden citation: the qrel set is a direct image
                # of the citations, not a fan-out over every touched chunk.
                citation_chunks = (
                    map_citations_to_chunks(item.spans, chunk_records)
                    if reason is None
                    else []
                )
                if reason is None and not citation_chunks:
                    # Evidence that lands in no chunk would leave the query
                    # with no qrel, scoring every model zero on it.
                    reason = "no_chunk_holds_the_evidence"
                elif (
                    reason is None
                    and max_gold_chunks is not None
                    and len({entry.chunk_id for entry in citation_chunks})
                    > max_gold_chunks
                ):
                    # A query needing more gold chunks than any practical k can
                    # return is capped below recall 1.0 for every model, so it
                    # measures the label distribution rather than the model.
                    reason = "too_many_gold_chunks"
                if reason is None:
                    kept.append((item, citation_chunks))
                    continue
                dropped_file.write(
                    {
                        "query_id": item.query.query_id,
                        "document_id": source.document_id,
                        "reference_answer": item.query.reference_answer,
                        "citations": list(item.query.citations),
                        "gold_chunks": len({e.chunk_id for e in citation_chunks}),
                        "reason": reason,
                    }
                )
            if not kept and not keep_documents_without_queries:
                rejected += 1
                rejected_file.write(
                    {
                        "source_row": index,
                        "document_id": source.document_id,
                        "reason": "no_retained_queries",
                        "discarded_chunks": len(chunks),
                    }
                )
                continue

            documents += 1
            chunk_total += len(chunks)
            document_file.write(
                {"document_id": source.document_id, "text": prepared.text}
            )
            for event in prepared.cleaning.events:
                cleaning_file.write(
                    {"document_id": source.document_id, **asdict(event)}
                )
            for citation in prepared.unmapped:
                unmapped_total += 1
                unmapped_file.write(asdict(citation))

            violations = plain_text_violations(prepared.text)
            if violations:
                impure += 1
                impure_file.write(
                    {"document_id": source.document_id, "violations": violations}
                )

            for chunk in chunks:
                corpus.write(
                    {
                        "_id": chunk.chunk_id,
                        "title": chunk.section,
                        "text": chunk.text,
                        "metadata": {
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.chunk_index,
                            "start_offset": chunk.start_offset,
                            "end_offset": chunk.end_offset,
                            "chunk_type": chunk.chunk_type,
                        },
                    }
                )

            for item, citation_chunks in kept:
                evaluable += 1
                by_citation = {
                    entry.citation_id: entry for entry in citation_chunks
                }
                evidence_file.write(
                    {
                        "query_id": item.query.query_id,
                        "document_id": source.document_id,
                        "spans": [
                            {
                                **asdict(span),
                                "chunk_id": (
                                    by_citation[span.citation_id].chunk_id
                                    if span.citation_id in by_citation
                                    else None
                                ),
                            }
                            for span in item.spans
                        ],
                    }
                )
                query_file.write(
                    {
                        "_id": item.query.query_id,
                        "text": item.query.text,
                        "reference_answer": item.query.reference_answer,
                        "citations": list(item.query.citations),
                    }
                )
                candidate_file.write(
                    {
                        "query_id": item.query.query_id,
                        "document_id": source.document_id,
                        "candidate_ids": document_chunk_ids,
                    }
                )
                for chunk_id in dict.fromkeys(
                    entry.chunk_id
                    for entry in sorted(citation_chunks, key=lambda e: e.chunk_index)
                ):
                    qrels += 1
                    qrel_file.write(
                        {
                            "query-id": item.query.query_id,
                            "corpus-id": chunk_id,
                            "score": 1,
                        }
                    )

    summary = RunSummary(
        source_rows=rows,
        rejected_sources=rejected,
        documents=documents,
        chunks=chunk_total,
        queries=queries,
        queries_retained=evaluable,
        qrels=qrels,
        unmapped_citations=unmapped_total,
        plain_text_violations=impure,
    )
    manifest_path = output / "meta" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "pipeline": "redator-v4",
                "split": split,
                "generation_id": generation_id or new_generation_id(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_sha256": fingerprint.sha256 if fingerprint else "",
                "source_row_count": fingerprint.row_count if fingerprint else rows,
                "chunking": asdict(config),
                "length_unit": "o200k_base_tokens",
                "relevance": "one_chunk_per_citation_by_maximum_overlap",
                "length_unit_encoding": ENCODING_NAME,
                "max_citation_words": citation_limit,
                "citation_filter": "trackable_and_within_max_citation_words",
                "corpus_text": "plain_text_verified_before_chunking",
                "max_gold_chunks": max_gold_chunks,
                "document_filter": (
                    "kept_all"
                    if keep_documents_without_queries
                    else "requires_at_least_one_retained_query"
                ),
                "query_filter": (
                    "answer_text_and_resolved_citation"
                    if require_answer_text
                    else "resolved_citation"
                ),
                "summary": asdict(summary),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def run(
    source_rows: Iterable[Mapping[str, Any]],
    output_directory: str | Path,
    **kwargs: Any,
) -> RunSummary:
    """Build off-path and publish the generation, rolling back on failure.

    The generation is validated against the dataset contract *before* it is
    considered successful, so a build that emits inconsistent artifacts rolls
    back instead of replacing a good generation with a broken one.
    """

    def build(staging: Path) -> RunSummary:
        summary = build_dataset(source_rows, staging, **kwargs)
        # Staged output is not yet at generations/<generation-id>/, so its
        # directory name is not the identity. T017 publishes it there.
        validate_generation(staging, check_directory_identity=False)
        return summary

    return publish_generation(
        output_directory,
        build,
        managed_directories=MANAGED_DIRECTORIES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("output_directory", type=Path)
    # No --split: this feature publishes only the test split, and accepting a
    # value here is how a train or validation build would slip through.
    parser.add_argument(
        "--source-sha256",
        default=V4_TEST_SHA256,
        help="Expected source fingerprint; override only for the synthetic fixture",
    )
    parser.add_argument(
        "--source-rows",
        type=int,
        default=V4_TEST_ROW_COUNT,
        help="Expected source row count; override only for the synthetic fixture",
    )
    parser.add_argument("--max-words", type=int, default=V4_CHUNKING.max_words)
    parser.add_argument("--target-words", type=int, default=V4_CHUNKING.target_words)
    parser.add_argument("--overlap-words", type=int, default=V4_CHUNKING.overlap_words)
    parser.add_argument("--minimum-words", type=int, default=V4_CHUNKING.minimum_words)
    parser.add_argument(
        "--max-citation-words",
        type=int,
        default=None,
        help="Cap on words per citation; defaults to the chunk max_words",
    )
    parser.add_argument(
        "--max-gold-chunks",
        type=int,
        default=10,
        help="Drop queries needing more gold chunks than this (0 = no cap)",
    )
    parser.add_argument(
        "--keep-query-less-documents",
        action="store_true",
        help="Keep documents no retained query points at (open-corpus setups)",
    )
    parser.add_argument(
        "--allow-missing-answer",
        action="store_true",
        help="Keep queries whose source answer is null, if they cite evidence",
    )
    args = parser.parse_args()

    # The gate runs before a single row is read: a train/validation name is
    # refused outright, and altered bytes fail the fingerprint.
    fingerprint = verify_source_file(
        args.source_csv,
        expected_sha256=args.source_sha256,
        expected_row_count=args.source_rows,
    )

    summary = run(
        iter_source_rows(fingerprint.path),
        args.output_directory,
        split=PRODUCTION_SPLIT,
        fingerprint=fingerprint,
        require_answer_text=not args.allow_missing_answer,
        max_citation_words=args.max_citation_words,
        keep_documents_without_queries=args.keep_query_less_documents,
        max_gold_chunks=args.max_gold_chunks or None,
        # replace() rather than a fresh ChunkingConfig: building one from
        # scratch silently drops any v4 default the CLI does not name.
        config=replace(
            V4_CHUNKING,
            max_words=args.max_words,
            target_words=args.target_words,
            overlap_words=args.overlap_words,
            minimum_words=args.minimum_words,
        ),
    )
    print(
        f"{summary.documents} documents, {summary.chunks} chunks, "
        f"{summary.queries_retained}/{summary.queries} queries retained, "
        f"{summary.qrels} qrels -> {args.output_directory} "
        f"({summary.rejected_sources} sources rejected, "
        f"{summary.unmapped_citations} citations unmapped, "
        f"{summary.plain_text_violations} documents with residual markup)"
    )


if __name__ == "__main__":
    main()
