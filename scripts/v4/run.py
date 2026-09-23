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
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from scripts.strategies.chunking.legal_recursive import ChunkingConfig, LengthCounter
from scripts.v4.chunking import ENCODING_NAME, V4_CHUNKING, chunk_cleaned_document, o200k_counter
from scripts.v4.cleaning import plain_text_violations
from scripts.v4.generation import new_generation_id
from scripts.v4.ground_truth import map_citations_to_chunks, prepare_document
from scripts.v4.publication import Published, publish
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
    "DEFAULT_MAX_QUERY_TOKENS",
    "RunSummary",
    "build_dataset",
    "new_generation_id",
    "run",
    "iter_csv_rows",
    "query_token_overflow",
]

#: Longest question kept, in o200k_base tokens: the same unit and cap as a
#: passage. Longer "questions" in the source are embedded task prompts (lists
#: of hundreds of lawyers or categories to match against), not retrieval
#: queries, and overflow the embedding server's context window.
DEFAULT_MAX_QUERY_TOKENS = 512


@dataclass(frozen=True)
class RunSummary:
    source_rows: int
    rejected_sources: int
    documents: int
    chunks: int
    queries: int
    queries_retained: int
    dropped_queries: int
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


def query_token_overflow(
    text: str, max_tokens: int | None, counter: LengthCounter
) -> int | None:
    """The question's token count when it exceeds `max_tokens`, else None.

    `max_tokens=None` disables the limit.
    """
    if max_tokens is None:
        return None
    tokens = counter.count(text)
    return tokens if tokens > max_tokens else None


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
    max_query_tokens: int | None = DEFAULT_MAX_QUERY_TOKENS,
    fingerprint: SourceFingerprint | None = None,
    generation_id: str | None = None,
) -> RunSummary:
    """Write one complete generation into `output_directory`.

    `fingerprint` carries the gated source's provenance. A production build
    always supplies one; a synthetic-fixture build does not, and records an
    empty source SHA-256 to say so.
    """
    output = Path(output_directory)
    length = counter or o200k_counter()
    citation_limit = (
        config.max_words if max_citation_words is None else max_citation_words
    )
    rows = documents = chunk_total = queries = evaluable = qrels = 0
    rejected = dropped = unmapped_total = impure = 0
    seen_documents: set[str] = set()

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
            # A repeated ID would collide with the first row's query and
            # passage IDs, so the later row is rejected whole. Its questions
            # are not audited by ID: those IDs belong to the first row.
            if source.document_id in seen_documents:
                rejected += 1
                rejected_file.write(
                    {
                        "source_row": index,
                        "document_id": source.document_id,
                        "reason": "duplicate_document_id",
                    }
                )
                continue
            if source.document_id:
                seen_documents.add(source.document_id)

            for blank in source.dropped:
                queries += 1
                dropped += 1
                dropped_file.write(
                    {
                        "query_id": blank.query_id,
                        "document_id": blank.document_id,
                        "question_index": blank.question_index,
                        "reason": blank.reason,
                    }
                )

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
                source.document_id, prepared.text, config=config, counter=length
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
                query_tokens = query_token_overflow(
                    item.query.text, max_query_tokens, length
                )
                if query_tokens is not None:
                    reason: str | None = "question_exceeds_max_tokens"
                elif require_answer_text and not item.query.has_answer_text:
                    reason = "answer_has_no_text"
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
                dropped += 1
                dropped_file.write(
                    {
                        "query_id": item.query.query_id,
                        "document_id": source.document_id,
                        "question_index": item.query.question_index,
                        "reference_answer": item.query.reference_answer,
                        "citations": list(item.query.citations),
                        "gold_chunks": len({e.chunk_id for e in citation_chunks}),
                        **({"query_tokens": query_tokens} if query_tokens else {}),
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
        dropped_queries=dropped,
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
                "max_query_tokens": max_query_tokens,
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
) -> Published[RunSummary]:
    """Build a new immutable generation and make it `current` atomically.

    The v4 publisher validates the generation against the dataset contract
    before switching `current`, so a failed or inconsistent build leaves the
    active generation untouched. The shared v1-v3 publisher is not used.
    """
    generation_id = kwargs.pop("generation_id", None)

    def build(staging: Path, identity: str) -> RunSummary:
        return build_dataset(source_rows, staging, generation_id=identity, **kwargs)

    return publish(output_directory, build, generation_id=generation_id)


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
        "--max-query-tokens",
        type=int,
        default=DEFAULT_MAX_QUERY_TOKENS,
        help="Drop questions longer than this many o200k_base tokens (0 = no cap)",
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

    published = run(
        iter_source_rows(fingerprint.path),
        args.output_directory,
        split=PRODUCTION_SPLIT,
        fingerprint=fingerprint,
        require_answer_text=not args.allow_missing_answer,
        max_citation_words=args.max_citation_words,
        keep_documents_without_queries=args.keep_query_less_documents,
        max_gold_chunks=args.max_gold_chunks or None,
        max_query_tokens=args.max_query_tokens or None,
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
    summary = published.summary
    superseded = (
        f", superseding {published.previous_generation_id}"
        if published.previous_generation_id
        else ""
    )
    print(
        f"generation {published.generation_id} is current{superseded}: "
        f"{summary.documents} documents, {summary.chunks} chunks, "
        f"{summary.queries_retained}/{summary.queries} queries retained, "
        f"{summary.qrels} qrels -> {published.path} "
        f"({summary.rejected_sources} sources rejected, "
        f"{summary.unmapped_citations} citations unmapped, "
        f"{summary.plain_text_violations} documents with residual markup)"
    )


if __name__ == "__main__":
    main()
