"""Document cleaning that preserves the mapping back to source offsets.

Each rule reports the edits it wants as `Replacement`s rather than rewriting
the string, so a citation span located in the source document can still be
located in the cleaned text afterwards. See `offsets.py` for why.
"""

from __future__ import annotations

import base64
import binascii
import html
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from scripts.v4.offsets import EditedText, Replacement, Span, apply_replacements

__all__ = ["CleaningEvent", "CleanedText", "clean_text", "CLEANING_RULES", "plain_text_violations", "PLAIN_TEXT_CHECKS"]

_PROMPT = re.compile(
    r"\A\s*(?:\*{1,2}|_{1,2})?\s*analise\s+(?:o|os|a|as)\s+documentos?\s+"
    r"e\s+responda\s+(?:a|à|as|às)\s+perguntas?\s*[.:]?\s*"
    r"(?:\*{1,2}|_{1,2})?\s*(?:\r?\n)+",
    re.IGNORECASE,
)
_DATA_IMAGE = re.compile(
    r"data:image/[\w.+-]+;base64,(?P<payload>[A-Za-z0-9+/\s]+={0,2})",
    re.IGNORECASE,
)
_BASE64 = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_IMAGE_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a",
    b"BM", b"II*\x00", b"MM\x00*",
)
_URL_ADDRESS = (
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?::\d+)?"
    r"(?:/[^\s<>\\\"']*)?(?:\?[^\s<>\\\"']*)?"
)
_SPACED_SCHEME = re.compile(
    rf"\b(?P<scheme>https?)(?:\s+:\s*/\s*/\s*|\s*:\s+/\s*/\s*|\s*:\s*/\s+/\s*)"
    rf"(?P<address>{_URL_ADDRESS})",
    re.IGNORECASE,
)
_SPACED_WWW = re.compile(
    rf"(?<![\w/@.-])www(?:\s+\.\s*|\s*\.\s+)(?P<address>{_URL_ADDRESS})",
    re.IGNORECASE,
)
_BARE_WWW = re.compile(
    rf"(?<![\w/@.-])www\.(?P<address>{_URL_ADDRESS})", re.IGNORECASE
)
_REPLACEMENT_CHARACTER = "�"
_PRIVATE_USE_BULLET = ""
#: C0/C1 controls that carry no text meaning. Newline is structure, so it stays.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
#: Zero-width marks and separators that survive PDF extraction.
_INVISIBLE = re.compile(r"[\u00ad\u200b-\u200f\u2028\u2029\u2060\ufeff]")
#: Exotic spaces normalized to U+0020 so word splitting behaves.
_UNUSUAL_SPACE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_PRIVATE_USE = re.compile(r"[\ue000-\uf8ff]")
#: Only real HTML tag names, so legal placeholders like <nome> survive.
_HTML_TAG = re.compile(
    r"</?(?:p|div|span|br|hr|b|i|u|em|strong|font|sup|sub|a|img|pre|code"
    r"|blockquote|table|thead|tbody|tfoot|tr|td|th|ul|ol|li|h[1-6])"
    r"\b[^<>]{0,300}?/?>",
    re.IGNORECASE,
)
_HTML_ENTITY = re.compile(
    r"&(?:[a-zA-Z][a-zA-Z0-9]{1,9}|#\d{2,6}|#[xX][0-9a-fA-F]{2,6});"
)


@dataclass(frozen=True)
class CleaningEvent:
    rule: str
    count: int
    removed_characters: int


@dataclass(frozen=True)
class CleanedText:
    """Cleaned document text plus the mapping from the source it came from."""

    text: str
    events: tuple[CleaningEvent, ...]
    _stages: tuple[EditedText, ...]

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    def map_span(self, start: int, end: int) -> Span | None:
        """Map a source span onto the cleaned text, or None if it was removed."""
        span = Span(start, end)
        for stage in self._stages:
            mapped = stage.map_span(span.start, span.end)
            if mapped is None:
                return None
            span = mapped
        return span


def _is_encoded_image(value: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    if len(compact) < 128 or len(compact) % 4 or not _BASE64.fullmatch(compact):
        return False
    try:
        prefix = base64.b64decode(compact[: min(128, len(compact))], validate=True)
    except (binascii.Error, ValueError):
        return False
    return any(prefix.startswith(item) for item in _IMAGE_SIGNATURES) or (
        prefix.startswith(b"RIFF") and b"WEBP" in prefix[:16]
    )


def _leading_prompt(text: str) -> list[Replacement]:
    match = _PROMPT.match(text)
    return [Replacement(match.start(), match.end(), "")] if match else []


def _encoded_images(text: str) -> list[Replacement]:
    replacements = [
        Replacement(match.start(), match.end(), "")
        for match in _DATA_IMAGE.finditer(text)
        if _is_encoded_image(match.group("payload"))
    ]
    if replacements:
        return replacements
    stripped = text.strip()
    if stripped and _is_encoded_image(stripped):
        return [Replacement(0, len(text), "")]
    return []


def _replacement_characters(text: str) -> list[Replacement]:
    return [
        Replacement(index, index + 1, "")
        for index, character in enumerate(text)
        if character == _REPLACEMENT_CHARACTER
    ]


def _private_use(text: str) -> list[Replacement]:
    """Map the Wingdings bullet to U+2022 and drop other private-use glyphs.

    Private-use codepoints render only with the font that produced them, so as
    text they are noise; U+F0B7 is the one with a plain-text equivalent.
    """
    return [
        Replacement(
            match.start(),
            match.end(),
            "\u2022" if match.group(0) == _PRIVATE_USE_BULLET else "",
        )
        for match in _PRIVATE_USE.finditer(text)
    ]


def _control_characters(text: str) -> list[Replacement]:
    return [Replacement(m.start(), m.end(), "") for m in _CONTROL.finditer(text)]


def _invisible_characters(text: str) -> list[Replacement]:
    return [Replacement(m.start(), m.end(), "") for m in _INVISIBLE.finditer(text)]


def _unusual_spaces(text: str) -> list[Replacement]:
    return [Replacement(m.start(), m.end(), " ") for m in _UNUSUAL_SPACE.finditer(text)]


def _html_entities(text: str) -> list[Replacement]:
    replacements: list[Replacement] = []
    for match in _HTML_ENTITY.finditer(text):
        decoded = html.unescape(match.group(0))
        if decoded != match.group(0):
            replacements.append(Replacement(match.start(), match.end(), decoded))
    return replacements


def _html_tags(text: str) -> list[Replacement]:
    return [Replacement(m.start(), m.end(), "") for m in _HTML_TAG.finditer(text)]


def _urls(text: str) -> list[Replacement]:
    replacements: list[Replacement] = []
    for match in _SPACED_SCHEME.finditer(text):
        target = f"{match.group('scheme').lower()}://{match.group('address')}"
        replacements.append(Replacement(match.start(), match.end(), target))
    covered = [(item.start, item.end) for item in replacements]

    def overlaps(start: int, end: int) -> bool:
        return any(start < other_end and other_start < end for other_start, other_end in covered)

    for pattern in (_SPACED_WWW, _BARE_WWW):
        for match in pattern.finditer(text):
            if overlaps(match.start(), match.end()):
                continue
            target = f"https://www.{match.group('address')}"
            if target == match.group(0):
                continue
            replacements.append(Replacement(match.start(), match.end(), target))
            covered.append((match.start(), match.end()))
    return replacements


#: Order matters: entities are decoded before tags are stripped, so a
#: double-escaped "&lt;p&gt;" becomes a tag and is removed rather than left as
#: visible markup in the corpus.
CLEANING_RULES: tuple[tuple[str, Callable[[str], list[Replacement]]], ...] = (
    ("leading_prompt", _leading_prompt),
    ("encoded_images", _encoded_images),
    ("html_entities", _html_entities),
    ("html_tags", _html_tags),
    ("control_characters", _control_characters),
    ("invisible_characters", _invisible_characters),
    ("unusual_spaces", _unusual_spaces),
    ("replacement_characters", _replacement_characters),
    ("private_use", _private_use),
    ("urls", _urls),
)

#: Checks asserting the corpus text really is plain text before chunking.
#: Each entry counts the artefacts of one kind still present in the text.
PLAIN_TEXT_CHECKS: tuple[tuple[str, Callable[[str], int]], ...] = (
    ("control_character", lambda text: len(_CONTROL.findall(text))),
    ("invisible_character", lambda text: len(_INVISIBLE.findall(text))),
    ("unusual_space", lambda text: len(_UNUSUAL_SPACE.findall(text))),
    ("private_use_character", lambda text: len(_PRIVATE_USE.findall(text))),
    ("replacement_character", lambda text: text.count(_REPLACEMENT_CHARACTER)),
    ("html_tag", lambda text: len(_HTML_TAG.findall(text))),
    ("html_entity", lambda text: _decodable_entity_count(text)),
)


def _decodable_entity_count(text: str) -> int:
    """Count only entities that actually decode.

    "BM&FBOVESPA;" and "&V89;" match the entity shape but are ordinary text,
    and html.unescape leaves them untouched. Counting them as markup would make
    the plain-text guarantee unsatisfiable for documents that merely mention an
    ampersand.
    """
    return sum(
        1
        for match in _HTML_ENTITY.finditer(text)
        if html.unescape(match.group(0)) != match.group(0)
    )


def plain_text_violations(text: str) -> dict[str, int]:
    """Return any non-plain-text artefact still present, by kind."""
    found: dict[str, int] = {}
    for name, count_artefacts in PLAIN_TEXT_CHECKS:
        count = count_artefacts(text)
        if count:
            found[name] = count
    return found


def clean_text(
    text: str,
    rules: Sequence[tuple[str, Callable[[str], list[Replacement]]]] = CLEANING_RULES,
) -> CleanedText:
    """Apply every cleaning rule in order, retaining a source offset mapping."""
    current = text
    stages: list[EditedText] = []
    events: list[CleaningEvent] = []
    for name, rule in rules:
        replacements = rule(current)
        if not replacements:
            continue
        removed = sum(
            (item.end - item.start) - len(item.text) for item in replacements
        )
        stage = apply_replacements(current, replacements)
        stages.append(stage)
        events.append(
            CleaningEvent(
                rule=name,
                count=len(replacements),
                removed_characters=removed,
            )
        )
        current = stage.text
    return CleanedText(
        text=current, events=tuple(events), _stages=tuple(stages)
    )
