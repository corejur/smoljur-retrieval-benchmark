"""Heading detection tuned for v4 documents.

The v1-v3 detector accepts any line that starts with a digit group and a
separator, or that is broadly uppercase. In v4 - contract statements, bank
records and petitions rather than PJe event logs - that matches CPFs
("767.505.313-34 Data de Nascimento: ..."), timestamps ("13:59:01 Tribunal de
Justica ..."), and account codes ("2.99.033"). Those become chunk sections and
are exported as the BEIR `title`, so the noise reaches the retrieval index.

This detector keeps the same shapes but requires a heading to actually read
like one: enough letters, and not a recognizable identifier.
"""

from __future__ import annotations

import re

__all__ = ["is_heading", "MAXIMUM_HEADING_LENGTH"]

MAXIMUM_HEADING_LENGTH = 180
_MINIMUM_LETTERS = 4
_MINIMUM_LETTER_RATIO = 0.5
_MINIMUM_WORD_LETTERS = 3
_WORD = re.compile(r"[^\W\d_]{%d,}" % _MINIMUM_WORD_LETTERS)

_UPPER = "A-ZÁÉÍÓÚÂÊÔÃÕÀÇ"
_LETTER = r"^\W\d_"

#: Identifier shapes that are data, never section headings.
_IDENTIFIER = re.compile(
    r"^(?:"
    r"\d{1,2}:\d{2}(?::\d{2})?\b"                 # 13:59:01
    r"|\d{3}\.\d{3}\.\d{3}-\d{2}\b"               # CPF
    r"|\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"         # CNPJ
    r"|\d{1,2}/\d{1,2}/\d{2,4}\b"                 # 04/04/1976
    r"|\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b"    # CNJ process number
    r"|[\d.\-/:]+\s*$"                            # bare numeric code
    r")"
)
#: A clause number: at most four short components, then a separator.
_NUMBERED = re.compile(
    rf"^\d{{1,3}}(?:\.\d{{1,3}}){{0,3}}\s*[–—.:)-]\s*(?P<rest>[{_LETTER}].*)$"
)
_ROMAN = re.compile(
    rf"^[IVXLCDM]{{1,7}}\s*[–—.:)-]\s*(?P<rest>[{_LETTER}].*)$"
)
_ARTICLE = re.compile(rf"^(?:DOS?|DAS?)\s+[{_UPPER}].+$")
_UPPERCASE = re.compile(rf"^[{_UPPER}][{_UPPER}\s\d./()_–—:-]{{5,}}$")


def _letters(text: str) -> int:
    return sum(1 for character in text if character.isalpha())


def _reads_like_prose(text: str) -> bool:
    """True when the line carries enough letters to be a title, not a record."""
    letters = _letters(text)
    if letters < _MINIMUM_LETTERS:
        return False
    dense = [character for character in text if not character.isspace()]
    if not dense:
        return False
    if letters / len(dense) < _MINIMUM_LETTER_RATIO:
        return False
    # At least one real word: rejects OCR debris like "F KO 4MQ33E", which
    # clears the ratio test on scattered single letters alone.
    return _WORD.search(text) is not None


def is_heading(text: str) -> bool:
    """True when the line is a section heading rather than data."""
    line = text.strip()
    if not line or len(line) > MAXIMUM_HEADING_LENGTH:
        return False
    if _IDENTIFIER.match(line):
        return False

    for pattern in (_NUMBERED, _ROMAN):
        match = pattern.match(line)
        if match:
            return _reads_like_prose(match.group("rest"))
    if _ARTICLE.match(line) or _UPPERCASE.match(line):
        return _reads_like_prose(line)
    return False
