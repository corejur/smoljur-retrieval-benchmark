"""Resolve answer citations to exact character spans in the cleaned document.

Relevance is stored as spans, so the labels stay independent of the chunking
that produced them and any chunker can be scored against the same evidence.

Each golden citation then maps to exactly one chunk - the one holding most of
it - so a qrel is a direct image of a citation rather than a fan-out over every
chunk the span happens to touch. Chunks overlap by design, so without that rule
one citation would mark two or three chunks relevant and inflate recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from scripts.source_normalization import normalize_source_html
from scripts.v4.cleaning import CleanedText, clean_text
from scripts.v4.questions import QueryRecord

__all__ = [
    "EvidenceSpan",
    "UnmappedCitation",
    "PreparedDocument",
    "CitationChunk",
    "prepare_document",
    "map_citations_to_chunks",
]

_WHOLE_DOCUMENT_RATIO = 0.9
DEFAULT_MAX_CITATION_WORDS = 300


@dataclass(frozen=True)
class EvidenceSpan:
    """One citation, located in both the source and the cleaned document.

    `start`/`end` index the cleaned text the chunks are cut from;
    `source_start`/`source_end` index the normalized source document, so every
    retained citation stays traceable back to where it came from.
    """

    citation_id: str
    start: int
    end: int
    source_start: int
    source_end: int
    word_count: int


@dataclass(frozen=True)
class UnmappedCitation:
    document_id: str
    query_id: str
    citation_id: str
    reason: str


@dataclass(frozen=True)
class QueryEvidence:
    query: QueryRecord
    spans: tuple[EvidenceSpan, ...]

    @property
    def has_evidence(self) -> bool:
        return bool(self.spans)


@dataclass(frozen=True)
class PreparedDocument:
    document_id: str
    text: str
    cleaning: CleanedText
    evidence: tuple[QueryEvidence, ...]
    unmapped: tuple[UnmappedCitation, ...]

    @property
    def evaluable(self) -> tuple[QueryEvidence, ...]:
        return tuple(item for item in self.evidence if item.has_evidence)


def prepare_document(
    document_html: str,
    queries: Sequence[QueryRecord],
    *,
    document_id: str,
    max_citation_words: int | None = DEFAULT_MAX_CITATION_WORDS,
    drop_whole_document_citations: bool = True,
) -> PreparedDocument:
    """Normalize, clean, and resolve every citation to a cleaned-text span.

    A citation is kept only when it is trackable - its element exists and
    survives cleaning - and when it is at most `max_citation_words` long. A
    citation longer than the chunk cap can never sit inside a single chunk, so
    it cannot act as a precise retrieval target.
    """
    normalized = normalize_source_html(document_html)
    cleaned = clean_text(normalized.text)
    limit = len(cleaned.text) * _WHOLE_DOCUMENT_RATIO

    evidence: list[QueryEvidence] = []
    unmapped: list[UnmappedCitation] = []
    for query in queries:
        spans: list[EvidenceSpan] = []
        for citation_id in query.citations:
            source_span = normalized.citation_spans.get(citation_id)
            if source_span is None:
                unmapped.append(
                    UnmappedCitation(
                        document_id, query.query_id, citation_id, "element_not_found"
                    )
                )
                continue
            mapped = cleaned.map_span(source_span.start, source_span.end)
            if mapped is None:
                unmapped.append(
                    UnmappedCitation(
                        document_id, query.query_id, citation_id, "removed_by_cleaning"
                    )
                )
                continue
            words = len(cleaned.text[mapped.start : mapped.end].split())
            if max_citation_words is not None and words > max_citation_words:
                unmapped.append(
                    UnmappedCitation(
                        document_id, query.query_id, citation_id, "exceeds_max_words"
                    )
                )
                continue
            if drop_whole_document_citations and (mapped.end - mapped.start) > limit:
                unmapped.append(
                    UnmappedCitation(
                        document_id, query.query_id, citation_id, "covers_whole_document"
                    )
                )
                continue
            spans.append(
                EvidenceSpan(
                    citation_id=citation_id,
                    start=mapped.start,
                    end=mapped.end,
                    source_start=source_span.start,
                    source_end=source_span.end,
                    word_count=words,
                )
            )
        evidence.append(QueryEvidence(query, tuple(spans)))

    return PreparedDocument(
        document_id=document_id,
        text=cleaned.text,
        cleaning=cleaned,
        evidence=tuple(evidence),
        unmapped=tuple(unmapped),
    )


@dataclass(frozen=True)
class CitationChunk:
    """The single chunk a golden citation is labelled against."""

    citation_id: str
    chunk_id: str
    chunk_index: int
    overlap: int


def map_citations_to_chunks(
    spans: Iterable[EvidenceSpan],
    chunks: Sequence[Mapping[str, object]],
) -> list[CitationChunk]:
    """Map each citation span to the one chunk that holds most of it.

    Ties - a span split evenly across two chunks - go to the earlier chunk, so
    the mapping is deterministic and independent of chunk iteration order.
    """
    mapped: list[CitationChunk] = []
    for span in spans:
        best: CitationChunk | None = None
        for chunk in chunks:
            start = int(chunk["start_offset"])  # type: ignore[arg-type]
            end = int(chunk["end_offset"])  # type: ignore[arg-type]
            overlap = min(span.end, end) - max(span.start, start)
            if overlap <= 0:
                continue
            index = int(chunk["chunk_index"])  # type: ignore[arg-type]
            if best is None or (overlap, -index) > (best.overlap, -best.chunk_index):
                best = CitationChunk(
                    citation_id=span.citation_id,
                    chunk_id=str(chunk["chunk_id"]),
                    chunk_index=index,
                    overlap=overlap,
                )
        if best is not None:
            mapped.append(best)
    return mapped
