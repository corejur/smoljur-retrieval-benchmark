"""Production-source contract tests for the v4 test CSV.

No column carries the split. The filename is advisory and the SHA-256 is the
authoritative gate, so these tests prove both directions: a correctly named
file with altered bytes is rejected, and a train/validation-named file is
rejected even when its bytes are right.

Every test here is portable: the real 81MB CSV is ignored by Git, so the
expected fingerprint is injectable and the synthetic fixtures carry their own.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.v4.source_contract import (
    REQUIRED_COLUMNS,
    V4_TEST_ROW_COUNT,
    V4_TEST_SHA256,
    SourceContractError,
    advisory_split,
    file_sha256,
    iter_source_rows,
    verify_source_file,
)

REAL_SOURCE = Path("datasets/redator_v4_repr-test_1k-val_2k-train_rest/redator_v4_repr-test_1k-val_2k-train_rest-test.csv")


def _write_csv(path: Path, rows, columns=REQUIRED_COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _row(document_id="doc-a", questions=("Quem?",), answers=None):
    answers = answers if answers is not None else [{"answer": "Joao", "citations": ["p-0"]}]
    return {
        "id": document_id,
        "data": "<p id='p-0'>texto</p>",
        "question_texts": json.dumps(list(questions), ensure_ascii=False),
        "output": json.dumps(list(answers), ensure_ascii=False),
    }


def _fixture(tmp_path: Path, rows=None, name="fixture-test.csv", columns=REQUIRED_COLUMNS):
    rows = rows if rows is not None else [_row("doc-a"), _row("doc-b")]
    return _write_csv(tmp_path / name, rows, columns)


# --- header and row contract ----------------------------------------------


def test_accepts_a_wellformed_fixture(tmp_path: Path) -> None:
    path = _fixture(tmp_path)
    fingerprint = verify_source_file(
        path, expected_sha256=file_sha256(path), expected_row_count=2
    )
    assert fingerprint.row_count == 2
    assert fingerprint.sha256 == file_sha256(path)


def test_rejects_a_missing_required_column(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "fixture-test.csv",
        [{"id": "doc-a", "data": "x", "question_texts": "[]"}],
        columns=("id", "data", "question_texts"),
    )
    with pytest.raises(SourceContractError, match="output"):
        list(iter_source_rows(path))


def test_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    path = _fixture(tmp_path, rows=[_row("doc-a"), _row("doc-a")])
    with pytest.raises(SourceContractError, match="doc-a"):
        list(iter_source_rows(path))


def test_rejects_a_blank_document_id(tmp_path: Path) -> None:
    path = _fixture(tmp_path, rows=[_row("")])
    with pytest.raises(SourceContractError, match="empty"):
        list(iter_source_rows(path))


def test_rejects_unequal_question_and_answer_arrays(tmp_path: Path) -> None:
    path = _fixture(
        tmp_path,
        rows=[_row("doc-a", questions=("Q1?", "Q2?"), answers=[{"answer": "a", "citations": []}])],
    )
    with pytest.raises(SourceContractError, match="doc-a"):
        list(iter_source_rows(path))


def test_rejects_unreadable_json_arrays(tmp_path: Path) -> None:
    row = _row("doc-a")
    row["output"] = "{not json"
    path = _fixture(tmp_path, rows=[row])
    with pytest.raises(SourceContractError, match="output"):
        list(iter_source_rows(path))


def test_streams_validated_rows_in_order(tmp_path: Path) -> None:
    path = _fixture(tmp_path, rows=[_row("doc-a"), _row("doc-b"), _row("doc-c")])
    assert [r["id"] for r in iter_source_rows(path)] == ["doc-a", "doc-b", "doc-c"]


# --- the filename is advisory ---------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("redator_v4-test.csv", "test"),
        ("redator_v4-train.csv", "train"),
        ("redator_v4-val.csv", "validation"),
        ("redator_v4-validation.csv", "validation"),
        ("some-other-export.csv", None),
    ],
)
def test_advisory_split_reads_the_filename(name: str, expected) -> None:
    assert advisory_split(Path(name)) == expected


def test_rejects_a_train_named_file_before_reading_its_rows(tmp_path: Path) -> None:
    path = _fixture(tmp_path, name="redator_v4-train.csv")
    with pytest.raises(SourceContractError, match="train"):
        verify_source_file(path, expected_sha256=file_sha256(path), expected_row_count=2)


def test_rejects_a_validation_named_file(tmp_path: Path) -> None:
    path = _fixture(tmp_path, name="redator_v4-val.csv")
    with pytest.raises(SourceContractError, match="validation"):
        verify_source_file(path, expected_sha256=file_sha256(path), expected_row_count=2)


# --- the SHA-256 is authoritative -----------------------------------------


def test_rejects_a_test_named_file_whose_bytes_changed(tmp_path: Path) -> None:
    """Right name, wrong bytes: the fingerprint gate catches it."""
    path = _fixture(tmp_path, name="redator_v4-test.csv")
    with pytest.raises(SourceContractError, match="sha256"):
        verify_source_file(
            path,
            expected_sha256="0" * 64,
            expected_row_count=2,
        )


def test_rejects_a_train_named_file_even_with_the_expected_bytes(tmp_path: Path) -> None:
    """Wrong name, right bytes: still rejected, and on the name."""
    path = _fixture(tmp_path, name="redator_v4-train.csv")
    with pytest.raises(SourceContractError, match="train"):
        verify_source_file(path, expected_sha256=file_sha256(path), expected_row_count=2)


def test_accepts_a_byte_identical_copy_under_an_unrelated_name(tmp_path: Path) -> None:
    original = _fixture(tmp_path, name="redator_v4-test.csv")
    digest = file_sha256(original)
    copy = tmp_path / "operator-supplied-copy.csv"
    copy.write_bytes(original.read_bytes())
    fingerprint = verify_source_file(copy, expected_sha256=digest, expected_row_count=2)
    assert fingerprint.sha256 == digest


def test_rejects_a_row_count_that_does_not_match(tmp_path: Path) -> None:
    path = _fixture(tmp_path, name="redator_v4-test.csv")
    with pytest.raises(SourceContractError, match="row"):
        verify_source_file(path, expected_sha256=file_sha256(path), expected_row_count=999)


# --- the production constants ---------------------------------------------


def test_production_constants_match_the_dataset_contract() -> None:
    assert V4_TEST_SHA256 == "d10d21f2074e48576cb715bc65ef01a36f369bf837dd2c404280775cef56fff3"
    assert V4_TEST_ROW_COUNT == 1000
    assert REQUIRED_COLUMNS == ("id", "data", "question_texts", "output")


@pytest.mark.skipif(not REAL_SOURCE.is_file(), reason="real v4 test CSV is not present (ignored by Git)")
def test_the_real_test_source_matches_its_recorded_fingerprint() -> None:
    assert file_sha256(REAL_SOURCE) == V4_TEST_SHA256
    assert advisory_split(REAL_SOURCE) == "test"
