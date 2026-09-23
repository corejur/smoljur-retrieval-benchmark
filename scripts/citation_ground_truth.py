"""Map source-answer HTML citations to deterministic relevant chunk IDs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scripts.source_normalization import CitationSpan, normalize_source_html
from scripts.strategies.chunking.legal_recursive import Chunk

@dataclass(frozen=True)
class TraceableDataset:
    query_rows: list[dict[str, Any]]
    documents: list[Mapping[str, Any]]
    removed_document_ids: tuple[str, ...]



@dataclass(frozen=True)
class GroundTruthResult:
    rows: list[dict[str, Any]]
    unmapped_citations: list[dict[str, Any]]


def citation_spans(document_html: str) -> tuple[str, dict[str, CitationSpan]]:
    """Return normalized document text and source spans for HTML element IDs."""
    normalized = normalize_source_html(document_html)
    return normalized.text, normalized.citation_spans


def _answer_citations(
    source: Mapping[str, str], answer_index: int, source_row: int
) -> list[str]:
    value = source.get("output", "").strip()
    answers = json.loads(value) if value else []
    if not isinstance(answers, list) or answer_index >= len(answers):
        raise ValueError(
            f"source row {source_row} has no answer at index {answer_index}"
        )
    answer = answers[answer_index]
    if not isinstance(answer, dict):
        raise ValueError(
            f"answer {answer_index} in source row {source_row} is not an object"
        )
    citations = answer.get("citations", [])
    if not isinstance(citations, list) or not all(
        isinstance(citation, str) for citation in citations
    ):
        raise ValueError(
            f"citations for answer {answer_index} in source row "
            f"{source_row} must be a list of strings"
        )
    return citations


def _occurrences(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    return [match.start() for match in re.finditer(re.escape(needle), text)]


def _anchored_location(
    needle: str, cleaned_text: str, expected: float
) -> tuple[int, int] | None:
    anchor_size = min(64, len(needle))
    if anchor_size < 24:
        return None
    maximum_offset = len(needle) - anchor_size
    offsets = sorted(
        {round(maximum_offset * index / 15) for index in range(16)}
    )
    matches: list[tuple[int, int]] = []
    for offset in offsets:
        anchor = needle[offset : offset + anchor_size]
        positions = _occurrences(cleaned_text, anchor)
        if positions:
            position = min(
                positions, key=lambda value: abs(value - (expected + offset))
            )
            matches.append((offset, position))
    if not matches:
        return None
    first_offset, first_position = matches[0]
    last_offset, last_position = matches[-1]
    start = max(0, first_position - first_offset)
    end = min(
        len(cleaned_text), last_position + len(needle) - last_offset
    )
    return start, max(start + 1, end)



def _locate_span(
    citation: CitationSpan, original_text: str, cleaned_text: str
) -> tuple[int, int] | None:
    needle = citation.text.strip()
    expected = (
        citation.start * len(cleaned_text) / len(original_text)
        if original_text
        else 0
    )
    variants = [
        needle,
        re.sub(r"(?<![\w/@.-])www\.", "https://www.", needle),
    ]
    for variant in dict.fromkeys(variants):
        matches = _occurrences(cleaned_text, variant)
        if matches:
            start = min(matches, key=lambda value: abs(value - expected))
            return start, start + len(variant)
    return _anchored_location(variants[-1], cleaned_text, expected)


def generate_ground_truth(
    source_rows: Sequence[Mapping[str, str]],
    query_rows: Sequence[Mapping[str, str]],
    cleaned_documents: Mapping[str, str],
    chunks_by_document: Mapping[str, Sequence[Chunk]],
) -> GroundTruthResult:
    """Add relevant chunk IDs by tracing every answer citation to chunk spans."""
    parsed_sources: dict[int, tuple[str, dict[str, CitationSpan]]] = {}
    output_rows: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []

    for query_row in query_rows:
        source_row = int(query_row["source_row"])
        answer_index = int(query_row["source_answer_index"])
        document_id = str(query_row["source_id"])
        if not 0 <= source_row < len(source_rows):
            raise ValueError(f"query references source row {source_row} outside source CSV")
        if document_id not in cleaned_documents:
            raise ValueError(f"query document {document_id!r} is absent from cleaned documents")
        if document_id not in chunks_by_document:
            raise ValueError(f"query document {document_id!r} has no chunks")

        source = source_rows[source_row]
        if source_row not in parsed_sources:
            parsed_sources[source_row] = citation_spans(source.get("data", ""))
        original_text, spans = parsed_sources[source_row]
        cleaned_text = cleaned_documents[document_id]
        relevant_ids: set[str] = set()
        for citation_id in _answer_citations(source, answer_index, source_row):
            citation = spans.get(citation_id)
            location = (
                _locate_span(citation, original_text, cleaned_text)
                if citation is not None
                else None
            )
            if location is None:
                unmapped.append(
                    {
                        "source_row": source_row,
                        "document_id": document_id,
                        "source_query_number": query_row["source_query_number"],
                        "citation": citation_id,
                        "reason": (
                            "citation_element_not_found"
                            if citation is None
                            else "citation_text_not_found_in_cleaned_document"
                        ),
                    }
                )
                continue
            start, end = location
            for chunk in chunks_by_document[document_id]:
                if chunk.start_offset < end and start < chunk.end_offset:
                    relevant_ids.add(chunk.chunk_id)

        row = dict(query_row)
        row["relevant_chunks"] = json.dumps(
            sorted(
                relevant_ids,
                key=lambda chunk_id: next(
                    chunk.chunk_index
                    for chunk in chunks_by_document[document_id]
                    if chunk.chunk_id == chunk_id
                ),
            ),
            ensure_ascii=False,
        )
        output_rows.append(row)
    return GroundTruthResult(rows=output_rows, unmapped_citations=unmapped)


def remove_untraceable_documents(
    ground_truth: GroundTruthResult,
    documents: Sequence[Mapping[str, Any]],
) -> TraceableDataset:
    """Remove documents with untraceable or missing query ground truth.

    A document is rejected as a unit when any of its queries has an unmapped
    citation or has no relevant chunk. This keeps every retained document's
    complete query group evaluable by retrieval benchmarks.
    """
    documents_without_relevant_chunks = {
        str(row["source_id"])
        for row in ground_truth.rows
        if not json.loads(str(row.get("relevant_chunks", "[]")))
    }
    removed_ids = tuple(
        sorted(
            {
                str(entry["document_id"])
                for entry in ground_truth.unmapped_citations
            }
            | documents_without_relevant_chunks
        )
    )
    removed = set(removed_ids)
    return TraceableDataset(
        query_rows=[
            row
            for row in ground_truth.rows
            if str(row["source_id"]) not in removed
        ],
        documents=[
            document
            for document in documents
            if str(document["document_id"]) not in removed
        ],
        removed_document_ids=removed_ids,
    )
