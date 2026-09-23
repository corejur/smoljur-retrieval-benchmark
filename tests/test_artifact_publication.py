from __future__ import annotations

from pathlib import Path

import pytest

from scripts.artifact_publication import publish_generation


def _build_generation(root: Path, marker: str) -> str:
    for name in (
        "queries",
        "documents",
        "data_preparation_output",
        "chunks",
    ):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "marker.txt").write_text(marker, encoding="utf-8")
    return marker


def test_publishes_one_complete_generation(tmp_path: Path) -> None:
    output = tmp_path / "data"
    _build_generation(output, "old")

    result = publish_generation(
        output, lambda staging: _build_generation(staging, "new")
    )

    assert result == "new"
    for name in (
        "queries",
        "documents",
        "data_preparation_output",
        "chunks",
    ):
        assert (output / name / "marker.txt").read_text() == "new"


def test_build_failure_preserves_published_generation(tmp_path: Path) -> None:
    output = tmp_path / "data"
    _build_generation(output, "published")

    def fail(staging: Path) -> None:
        _build_generation(staging, "partial")
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        publish_generation(output, fail)

    assert (output / "chunks" / "marker.txt").read_text() == "published"
