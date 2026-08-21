"""Normalize source HTML and retain citation offsets from one parse."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Sequence

_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "div", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "li", "main", "nav",
    "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
}
_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}


@dataclass(frozen=True)
class CitationSpan:
    citation_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class NormalizedSource:
    text: str
    citation_spans: dict[str, CitationSpan]


class _SourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[tuple[str, tuple[str, ...]]] = []
        self.stack: list[tuple[str, str | None]] = []

    def _active_ids(self) -> tuple[str, ...]:
        return tuple(value for _, value in self.stack if value is not None)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append(("\n", self._active_ids()))
        citation_id = next((value for name, value in attrs if name == "id" and value), None)
        if tag not in _VOID_TAGS:
            self.stack.append((tag, citation_id))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append(("\n", self._active_ids()))

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append(("\n", self._active_ids()))
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        self.parts.append((data, self._active_ids()))


def _normalize_parts(parts: Sequence[tuple[str, tuple[str, ...]]]) -> tuple[str, dict[str, tuple[int, int]]]:
    output: list[str] = []
    ranges: dict[str, list[int]] = {}
    line: list[tuple[str, tuple[str, ...]]] = []

    def emit_line() -> None:
        nonlocal line
        normalized: list[tuple[str, tuple[str, ...]]] = []
        whitespace_ids: set[str] = set()
        for character, citation_ids in line:
            if character in " \t":
                whitespace_ids.update(citation_ids)
                continue
            if whitespace_ids and normalized:
                normalized.append((" ", tuple(sorted(whitespace_ids))))
            whitespace_ids.clear()
            normalized.append((character, citation_ids))
        while normalized and normalized[-1][0] == " ":
            normalized.pop()
        line = []
        if not normalized:
            return
        if output:
            output.append("\n")
        for character, citation_ids in normalized:
            position = len(output)
            output.append(character)
            for citation_id in citation_ids:
                current = ranges.setdefault(citation_id, [position, position + 1])
                current[0] = min(current[0], position)
                current[1] = max(current[1], position + 1)

    previous_was_cr = False
    for text, citation_ids in parts:
        for character in text:
            if character == "\r":
                emit_line()
                previous_was_cr = True
            elif character == "\n":
                if not previous_was_cr:
                    emit_line()
                previous_was_cr = False
            else:
                previous_was_cr = False
                line.append((character, citation_ids))
    emit_line()
    return "".join(output), {key: (value[0], value[1]) for key, value in ranges.items()}


def normalize_source_html(document_html: str) -> NormalizedSource:
    """Return normalized text and citation spans produced by the same parse."""
    parser = _SourceParser()
    parser.feed(document_html)
    parser.close()
    text, ranges = _normalize_parts(parser.parts)
    return NormalizedSource(
        text=text,
        citation_spans={
            citation_id: CitationSpan(citation_id, start, end, text[start:end])
            for citation_id, (start, end) in ranges.items()
        },
    )
