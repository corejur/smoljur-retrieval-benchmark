from __future__ import annotations

import pytest

from scripts.v4.offsets import Replacement, apply_replacements


def test_maps_span_after_earlier_deletion() -> None:
    source = "REMOVE ME keep this"
    edited = apply_replacements(source, [Replacement(0, 10, "")])
    assert edited.text == "keep this"
    span = edited.map_span(source.index("keep"), source.index("keep") + 4)
    assert edited.text[span.start : span.end] == "keep"


def test_maps_span_after_length_changing_replacement() -> None:
    source = "see www.x.com then cite this"
    edited = apply_replacements(
        source, [Replacement(4, 13, "https://www.x.com")]
    )
    assert edited.text == "see https://www.x.com then cite this"
    start = source.index("cite this")
    span = edited.map_span(start, start + len("cite this"))
    assert edited.text[span.start : span.end] == "cite this"


def test_span_fully_inside_removal_returns_none() -> None:
    edited = apply_replacements("abcDELETEDxyz", [Replacement(3, 10, "")])
    assert edited.text == "abcxyz"
    assert edited.map_span(4, 9) is None


def test_span_overlapping_removal_keeps_surviving_part() -> None:
    source = "keep GONE tail"
    edited = apply_replacements(source, [Replacement(5, 10, "")])
    assert edited.text == "keep tail"
    span = edited.map_span(0, 10)
    assert edited.text[span.start : span.end] == "keep "


def test_multiple_replacements_compose() -> None:
    source = "0123456789abcdefghij"
    edited = apply_replacements(
        source, [Replacement(2, 4, ""), Replacement(10, 12, "XX")]
    )
    assert edited.text == "01456789XXcdefghij"
    start = source.index("efghij")
    span = edited.map_span(start, start + 6)
    assert edited.text[span.start : span.end] == "efghij"


def test_identity_when_no_replacements() -> None:
    edited = apply_replacements("unchanged text", [])
    assert edited.text == "unchanged text"
    span = edited.map_span(3, 8)
    assert (span.start, span.end) == (3, 8)


def test_span_covering_whole_source_survives() -> None:
    source = "alpha beta"
    edited = apply_replacements(source, [Replacement(0, 5, "A")])
    span = edited.map_span(0, len(source))
    assert (span.start, span.end) == (0, len(edited.text))


def test_rejects_overlapping_replacements() -> None:
    with pytest.raises(ValueError, match="overlapping"):
        apply_replacements("abcdef", [Replacement(0, 4, ""), Replacement(2, 5, "")])
