"""Text edits that stay traceable: map a source span onto edited text.

The v1-v3 pipeline cleaned document text and then searched for each citation's
text again, because cleaning destroyed the character offsets the citation spans
were expressed in. That search needed fuzzy anchoring and still failed.

Here every edit is recorded as a replacement over the source string, so a span
in the original maps onto the edited text arithmetically and exactly.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ["Replacement", "Span", "EditedText", "apply_replacements"]


@dataclass(frozen=True)
class Replacement:
    """Replace `source[start:end]` with `text`."""

    start: int
    end: int
    text: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid replacement range [{self.start}, {self.end})")


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    @property
    def empty(self) -> bool:
        return self.end <= self.start


@dataclass(frozen=True)
class _Segment:
    source_start: int
    source_end: int
    result_start: int
    result_end: int
    kept: bool


@dataclass(frozen=True)
class EditedText:
    """Edited text plus the mapping back to the string it was produced from."""

    text: str
    _segments: tuple[_Segment, ...]
    _starts: tuple[int, ...]

    def _segment_at(self, offset: int) -> _Segment:
        index = bisect_right(self._starts, offset) - 1
        if index < 0:
            return self._segments[0]
        return self._segments[index]

    def map_start(self, offset: int) -> int:
        """Map a source offset forward, snapping to the left of a removal."""
        segment = self._segment_at(offset)
        if not segment.kept:
            return segment.result_start
        return segment.result_start + (offset - segment.source_start)

    def map_end(self, offset: int) -> int:
        """Map an exclusive source offset, snapping to the right of a removal."""
        segment = self._segment_at(max(offset - 1, 0))
        if not segment.kept:
            return segment.result_end
        return min(
            segment.result_start + (offset - segment.source_start),
            segment.result_end,
        )

    def map_span(self, start: int, end: int) -> Span | None:
        """Map a source span, or None when the edits removed all of it."""
        if end <= start:
            return None
        mapped_start = max(0, min(self.map_start(start), len(self.text)))
        mapped_end = max(0, min(self.map_end(end), len(self.text)))
        if mapped_end <= mapped_start:
            return None
        return Span(mapped_start, mapped_end)


def apply_replacements(
    source: str, replacements: Iterable[Replacement]
) -> EditedText:
    """Apply non-overlapping replacements and retain the offset mapping."""
    ordered: Sequence[Replacement] = sorted(
        replacements, key=lambda item: (item.start, item.end)
    )
    segments: list[_Segment] = []
    parts: list[str] = []
    source_cursor = 0
    result_cursor = 0

    def emit(source_start: int, source_end: int, value: str, kept: bool) -> None:
        nonlocal result_cursor
        parts.append(value)
        segments.append(
            _Segment(
                source_start=source_start,
                source_end=source_end,
                result_start=result_cursor,
                result_end=result_cursor + len(value),
                kept=kept,
            )
        )
        result_cursor += len(value)

    for replacement in ordered:
        if replacement.start < source_cursor:
            raise ValueError(
                f"overlapping replacement at {replacement.start} "
                f"(previous edit ended at {source_cursor})"
            )
        if replacement.start > source_cursor:
            emit(
                source_cursor,
                replacement.start,
                source[source_cursor : replacement.start],
                True,
            )
        emit(replacement.start, replacement.end, replacement.text, False)
        source_cursor = replacement.end
    if source_cursor < len(source):
        emit(source_cursor, len(source), source[source_cursor:], True)
    if not segments:
        emit(0, 0, "", True)
    return EditedText(
        text="".join(parts),
        _segments=tuple(segments),
        _starts=tuple(segment.source_start for segment in segments),
    )
