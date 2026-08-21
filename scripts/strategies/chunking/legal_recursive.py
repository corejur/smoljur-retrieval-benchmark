"""Structure-aware recursive chunking for Portuguese legal documents."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence


_HEADING = re.compile(
    r"^(?:"
    r"[IVXLCDM]+\s*[–—.:-]\s*.+"
    r"|\d+(?:\.\d+)*\s*[–—.:-]\s*.+"
    r"|(?:DOS?|DAS?|DO|DA)\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ].+"
    r"|[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇ\s\d./()_–—:-]{5,}"
    r")$",
)
_HARD_BOUNDARY = re.compile(
    r"^(?:"
    r"PÁGINA DE SEPARAÇÃO"
    r"|Evento\s+\d+"
    r"|Tipo documento:"
    r"|PROCESSO(?:\s|$)"
    r"|DOCUMENTO(?:\s|$)"
    r")",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9(\[])")
_TABLE_SEPARATOR = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+$")


class LengthCounter(Protocol):
    """Model-neutral text-length interface used by chunking strategies."""

    def count(self, text: str) -> int:
        ...


class WordCounter:
    """Count deterministic whitespace-delimited words without model tokenization."""

    def count(self, text: str) -> int:
        return len(text.split())


@dataclass(frozen=True)
class ChunkingConfig:
    max_words: int = 400
    target_words: int = 300
    overlap_words: int = 50
    minimum_words: int = 50
    strategy_version: str = "2"

    def validate(self) -> None:
        if self.max_words < 1:
            raise ValueError("max_words must be positive")
        if not 1 <= self.target_words <= self.max_words:
            raise ValueError("target_words must be between 1 and max_words")
        if not 0 <= self.overlap_words < self.target_words:
            raise ValueError("overlap_words must be smaller than target_words")
        if not 0 <= self.minimum_words <= self.target_words:
            raise ValueError("minimum_words must not exceed target_words")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    word_count: int
    start_offset: int
    end_offset: int
    section: str
    chunk_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Span:
    text: str
    start: int
    end: int
    kind: str = "prose"


@dataclass
class _DraftChunk:
    text: str
    start: int
    end: int
    section: str
    kind: str
    word_count: int


def _unit_count(counter: LengthCounter, text: str) -> int:
    return counter.count(text)


def _nonempty_lines(text: str) -> list[_Span]:
    spans: list[_Span] = []
    for match in re.finditer(r"[^\r\n]+", text):
        value = match.group(0).strip()
        if not value:
            continue
        leading = len(match.group(0)) - len(match.group(0).lstrip())
        start = match.start() + leading
        spans.append(
            _Span(
                text=value,
                start=start,
                end=start + len(value),
                kind="table" if value.startswith("|") else "prose",
            )
        )
    return spans


def _is_heading(text: str) -> bool:
    return len(text) <= 180 and bool(_HEADING.fullmatch(text.strip()))


def _segments(text: str) -> list[tuple[int, str, list[_Span]]]:
    """Split at headings while identifying true document/event boundaries."""
    result: list[tuple[int, str, list[_Span]]] = []
    current: list[_Span] = []
    section = ""
    region = 0

    def flush() -> None:
        nonlocal current
        if current:
            result.append((region, section, current))
            current = []

    for line in _nonempty_lines(text):
        if line.text.upper().startswith("PÁGINA DE SEPARAÇÃO"):
            flush()
            section = ""
            region += 1
            continue
        if _HARD_BOUNDARY.match(line.text) and current:
            flush()
            section = ""
            region += 1
        if _is_heading(line.text):
            flush()
            section = line.text
        current.append(line)
    flush()
    return result


def _largest_prefix(
    text: str,
    start: int,
    counter: LengthCounter,
    word_limit: int,
) -> int:
    """Return the largest character end whose text fits word_limit."""
    low, high, best = start + 1, len(text), start
    while low <= high:
        middle = (low + high) // 2
        if _unit_count(counter, text[start:middle]) <= word_limit:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best == start:
        raise ValueError("counter cannot fit even one character within max_words")
    if best < len(text):
        boundary = max(
            text.rfind(" ", start + 1, best + 1),
            text.rfind("\n", start + 1, best + 1),
        )
        if boundary > start:
            best = boundary
    return best


def _overlap_start(
    text: str,
    piece_start: int,
    piece_end: int,
    counter: LengthCounter,
    overlap_words: int,
) -> int:
    if overlap_words == 0:
        return piece_end
    low, high, best = piece_start, piece_end, piece_end
    while low <= high:
        middle = (low + high) // 2
        if _unit_count(counter, text[middle:piece_end]) <= overlap_words:
            best = middle
            high = middle - 1
        else:
            low = middle + 1
    if best > piece_start:
        next_space = text.find(" ", best, piece_end)
        if next_space != -1:
            best = next_space + 1
    return best


def _word_window_fallback(
    span: _Span,
    counter: LengthCounter,
    config: ChunkingConfig,
) -> list[_Span]:
    pieces: list[_Span] = []
    local_start = 0
    while local_start < len(span.text):
        local_end = _largest_prefix(
            span.text, local_start, counter, config.max_words
        )
        value = span.text[local_start:local_end].strip()
        if value:
            leading = len(span.text[local_start:local_end]) - len(
                span.text[local_start:local_end].lstrip()
            )
            absolute_start = span.start + local_start + leading
            pieces.append(
                _Span(
                    value,
                    absolute_start,
                    absolute_start + len(value),
                    span.kind,
                )
            )
        if local_end >= len(span.text):
            break
        next_start = _overlap_start(
            span.text,
            local_start,
            local_end,
            counter,
            config.overlap_words,
        )
        local_start = next_start if next_start > local_start else local_end
    return pieces


def _split_oversized_span(
    span: _Span,
    counter: LengthCounter,
    config: ChunkingConfig,
) -> list[_Span]:
    if _unit_count(counter, span.text) <= config.max_words:
        return [span]

    sentences: list[_Span] = []
    cursor = 0
    for match in _SENTENCE_BOUNDARY.finditer(span.text):
        value = span.text[cursor:match.start()].strip()
        if value:
            leading = len(span.text[cursor:match.start()]) - len(
                span.text[cursor:match.start()].lstrip()
            )
            start = span.start + cursor + leading
            sentences.append(_Span(value, start, start + len(value), span.kind))
        cursor = match.end()
    value = span.text[cursor:].strip()
    if value:
        leading = len(span.text[cursor:]) - len(span.text[cursor:].lstrip())
        start = span.start + cursor + leading
        sentences.append(_Span(value, start, start + len(value), span.kind))

    if len(sentences) <= 1:
        return _word_window_fallback(span, counter, config)

    pieces: list[_Span] = []
    for sentence in sentences:
        if _unit_count(counter, sentence.text) <= config.max_words:
            pieces.append(sentence)
        else:
            pieces.extend(_word_window_fallback(sentence, counter, config))
    return pieces


def _join(spans: Sequence[_Span]) -> str:
    return "\n".join(span.text for span in spans)


def _trailing_overlap(
    spans: Sequence[_Span],
    counter: LengthCounter,
    config: ChunkingConfig,
) -> list[_Span]:
    if not spans or config.overlap_words == 0:
        return []
    selected: list[_Span] = []
    for span in reversed(spans):
        candidate = [span, *selected]
        if _unit_count(counter, _join(candidate)) > config.overlap_words:
            break
        selected = candidate
    return selected


def _draft(spans: Sequence[_Span], section: str, kind: str, counter: LengthCounter) -> _DraftChunk:
    text = _join(spans)
    return _DraftChunk(
        text=text,
        start=min(span.start for span in spans),
        end=max(span.end for span in spans),
        section=section,
        kind=kind,
        word_count=_unit_count(counter, text),
    )


def _pack_prose(
    spans: Sequence[_Span],
    section: str,
    counter: LengthCounter,
    config: ChunkingConfig,
) -> list[_DraftChunk]:
    units: list[_Span] = []
    for span in spans:
        units.extend(_split_oversized_span(span, counter, config))

    drafts: list[_DraftChunk] = []
    current: list[_Span] = []
    for unit in units:
        candidate = [*current, unit]
        candidate_count = _unit_count(counter, _join(candidate))
        if current and candidate_count > config.target_words:
            drafts.append(_draft(current, section, "prose", counter))
            overlap = _trailing_overlap(current, counter, config)
            current = overlap
            if current and _unit_count(counter, _join([*current, unit])) > config.max_words:
                current = []
        current.append(unit)
        if _unit_count(counter, _join(current)) > config.max_words:
            raise AssertionError("recursive prose split exceeded max_words")
    if current:
        drafts.append(_draft(current, section, "prose", counter))
    return _merge_small_tail(drafts, counter, config)


def _table_header(spans: Sequence[_Span]) -> tuple[list[_Span], list[_Span]]:
    if len(spans) >= 2 and _TABLE_SEPARATOR.fullmatch(spans[1].text):
        return list(spans[:2]), list(spans[2:])
    return list(spans[:1]), list(spans[1:])


def _pack_table(
    spans: Sequence[_Span],
    section: str,
    counter: LengthCounter,
    config: ChunkingConfig,
) -> list[_DraftChunk]:
    header, rows = _table_header(spans)
    if not rows:
        return [_draft(header, section, "table", counter)]

    drafts: list[_DraftChunk] = []
    current_rows: list[_Span] = []
    for row in rows:
        row_pieces = _split_oversized_span(row, counter, config)
        for piece in row_pieces:
            candidate = [*header, *current_rows, piece]
            if current_rows and _unit_count(counter, _join(candidate)) > config.target_words:
                drafts.append(_draft([*header, *current_rows], section, "table", counter))
                current_rows = []
                candidate = [*header, piece]
            if _unit_count(counter, _join(candidate)) > config.max_words:
                if current_rows:
                    drafts.append(_draft([*header, *current_rows], section, "table", counter))
                    current_rows = []
                drafts.append(_draft([piece], section, "table", counter))
            else:
                current_rows.append(piece)
    if current_rows:
        drafts.append(_draft([*header, *current_rows], section, "table", counter))
    return _merge_small_tail(drafts, counter, config)


def _merge_small_tail(
    drafts: list[_DraftChunk],
    counter: LengthCounter,
    config: ChunkingConfig,
) -> list[_DraftChunk]:
    if len(drafts) < 2 or drafts[-1].word_count >= config.minimum_words:
        return drafts
    previous, tail = drafts[-2], drafts[-1]
    if tail.start < previous.end:
        return drafts
    combined = f"{previous.text}\n{tail.text}"
    count = _unit_count(counter, combined)
    if count > config.max_words:
        return drafts
    drafts[-2:] = [
        _DraftChunk(
            combined,
            previous.start,
            tail.end,
            previous.section,
            previous.kind if previous.kind == tail.kind else "mixed",
            count,
        )
    ]
    return drafts


def _coalesce_adjacent_sections(
    drafts: list[_DraftChunk],
    incoming: list[_DraftChunk],
    counter: LengthCounter,
    config: ChunkingConfig,
) -> list[_DraftChunk]:
    """Pack adjacent prose sections without introducing cross-section overlap."""
    if not drafts or not incoming:
        return [*drafts, *incoming]
    previous, following = drafts[-1], incoming[0]
    if previous.kind != "prose" or following.kind != "prose":
        return [*drafts, *incoming]
    combined_text = f"{previous.text}\n{following.text}"
    count = _unit_count(counter, combined_text)
    if count > config.target_words:
        return [*drafts, *incoming]
    sections = [value for value in (previous.section, following.section) if value]
    combined_section = " | ".join(dict.fromkeys(sections))
    combined = _DraftChunk(
        text=combined_text,
        start=previous.start,
        end=following.end,
        section=combined_section,
        kind="prose",
        word_count=count,
    )
    return [*drafts[:-1], combined, *incoming[1:]]


def _chunk_segment(
    spans: Sequence[_Span],
    section: str,
    counter: LengthCounter,
    config: ChunkingConfig,
) -> list[_DraftChunk]:
    drafts: list[_DraftChunk] = []
    run: list[_Span] = []
    run_kind = ""

    def flush() -> None:
        nonlocal run
        if not run:
            return
        if run_kind == "table":
            drafts.extend(_pack_table(run, section, counter, config))
        else:
            drafts.extend(_pack_prose(run, section, counter, config))
        run = []

    for span in spans:
        if run and span.kind != run_kind:
            flush()
        if not run:
            run_kind = span.kind
        run.append(span)
    flush()
    # A format transition (for example, a table followed by a short issuance
    # timestamp) creates separate runs. Apply the minimum-size rule once more
    # across those adjacent runs so the tail retains the context that gives it
    # meaning, without crossing a heading or hard document boundary.
    return _merge_small_tail(drafts, counter, config)


def chunk_document(
    document: Mapping[str, Any],
    counter: LengthCounter | None = None,
    config: ChunkingConfig = ChunkingConfig(),
) -> list[Chunk]:
    """Chunk one cleaned document while preserving source character offsets."""
    config.validate()
    counter = counter or WordCounter()
    if "document_id" not in document or "text" not in document:
        raise ValueError("document must contain document_id and text")
    document_id, text = str(document["document_id"]), document["text"]
    if not isinstance(text, str):
        raise TypeError("document text must be a string")
    if not text.strip():
        return []

    drafts: list[_DraftChunk] = []
    previous_region: int | None = None
    for region, section, spans in _segments(text):
        incoming = _chunk_segment(spans, section, counter, config)
        if previous_region == region:
            drafts = _coalesce_adjacent_sections(
                drafts, incoming, counter, config
            )
        else:
            drafts.extend(incoming)
        previous_region = region

    chunks: list[Chunk] = []
    for index, draft in enumerate(drafts):
        if draft.word_count > config.max_words:
            raise AssertionError(
                f"chunk {document_id}:{index} has {draft.word_count} words"
            )
        chunks.append(
            Chunk(
                chunk_id=(
                    f"{document_id}:v{config.strategy_version}:"
                    f"{index:04d}:{draft.start}-{draft.end}"
                ),
                document_id=document_id,
                chunk_index=index,
                text=draft.text,
                word_count=draft.word_count,
                start_offset=draft.start,
                end_offset=draft.end,
                section=draft.section,
                chunk_type=draft.kind,
            )
        )
    return chunks


def chunk_documents(
    documents: Sequence[Mapping[str, Any]],
    counter: LengthCounter | None = None,
    config: ChunkingConfig = ChunkingConfig(),
) -> list[Chunk]:
    """Plug-in interface: chunk cleaned documents into retrieval records."""
    counter = counter or WordCounter()
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, counter, config))
    return chunks


