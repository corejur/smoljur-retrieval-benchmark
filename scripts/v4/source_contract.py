"""The production source gate: only the v4 test CSV may build this dataset.

No column in the source carries the split, so two independent checks stand
between an operator and a wrong dataset:

* the **filename is advisory** — a name whose final segment is `train` or
  `val`/`validation` is refused before a single row is read;
* the **SHA-256 is authoritative** — the bytes must match the recorded
  fingerprint, so a correctly named file with altered content is refused and a
  byte-identical copy under any other name is accepted.

The expected fingerprint is injectable so the synthetic fixture can exercise
the same gate; production callers take the module defaults.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "PRODUCTION_SPLIT",
    "REQUIRED_COLUMNS",
    "V4_TEST_ROW_COUNT",
    "V4_TEST_SHA256",
    "SourceContractError",
    "SourceFingerprint",
    "advisory_split",
    "file_sha256",
    "iter_source_rows",
    "verify_source_file",
]

#: SHA-256 of the 1,000-row v4 test CSV, per contracts/dataset.md.
V4_TEST_SHA256 = "d10d21f2074e48576cb715bc65ef01a36f369bf837dd2c404280775cef56fff3"

#: Records in that CSV.
V4_TEST_ROW_COUNT = 1000

#: Columns the build reads. Other columns are source metadata.
REQUIRED_COLUMNS: tuple[str, ...] = ("id", "data", "question_texts", "output")

PRODUCTION_SPLIT = "test"

#: Final filename segments that name a split. Anything else is unknown, which
#: is not by itself a rejection -- the fingerprint decides.
_SPLIT_NAMES = {
    "test": "test",
    "train": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
}

_REFUSED_SPLITS = {"train", "validation"}

_READ_BLOCK = 1024 * 1024


class SourceContractError(ValueError):
    """The source file is not the v4 test CSV, or violates its row contract."""


@dataclass(frozen=True)
class SourceFingerprint:
    """Provenance of an accepted production source."""

    path: Path
    sha256: str
    row_count: int
    split: str


def advisory_split(path: str | Path) -> str | None:
    """Read the split the filename *claims*, or None.

    Only the final hyphen-separated segment of the stem is considered. The real
    source is named `..._repr-test_1k-val_2k-train_rest-test.csv`, which
    contains `test`, `val` and `train` as substrings; naive matching would
    classify it three different ways.
    """
    stem = Path(path).stem
    return _SPLIT_NAMES.get(stem.rsplit("-", 1)[-1].lower())


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, read in bounded blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_rows(path: Path) -> int:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) is None:
            raise SourceContractError(f"source CSV has no header: {path}")
        return sum(1 for row in reader if row)


def verify_source_file(
    path: str | Path,
    *,
    expected_sha256: str = V4_TEST_SHA256,
    expected_row_count: int = V4_TEST_ROW_COUNT,
) -> SourceFingerprint:
    """Gate one file as a production source, or raise `SourceContractError`."""
    source = Path(path)
    if not source.is_file():
        raise SourceContractError(f"source file does not exist: {source}")

    claimed = advisory_split(source)
    if claimed in _REFUSED_SPLITS:
        raise SourceContractError(
            f"{source.name} is named as a {claimed} split; this feature "
            f"publishes only the {PRODUCTION_SPLIT} split"
        )

    digest = file_sha256(source)
    if digest != expected_sha256:
        raise SourceContractError(
            f"source sha256 is {digest}, expected {expected_sha256}; "
            f"{source.name} is not the recorded v4 test CSV"
        )

    rows = _count_rows(source)
    if rows != expected_row_count:
        raise SourceContractError(
            f"source holds {rows} rows, expected {expected_row_count}"
        )

    return SourceFingerprint(
        path=source, sha256=digest, row_count=rows, split=PRODUCTION_SPLIT
    )


def _decode_array(value: str, *, column: str, document_id: str) -> list[Any]:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise SourceContractError(
            f"document {document_id!r}: {column} is not valid JSON: {error}"
        ) from error
    if not isinstance(decoded, list):
        raise SourceContractError(
            f"document {document_id!r}: {column} must hold a JSON array"
        )
    return decoded


def iter_source_rows(path: str | Path) -> Iterator[dict[str, str]]:
    """Stream rows, validating the header and each row's identity contract.

    Nothing accumulates but the set of seen document IDs, so an 81MB source
    streams in bounded memory.
    """
    source = Path(path)
    csv.field_size_limit(sys.maxsize)
    seen: set[str] = set()
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SourceContractError(f"source CSV has no header: {source}")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise SourceContractError(
                f"source CSV is missing required column(s): {', '.join(missing)}"
            )

        for number, row in enumerate(reader, start=2):
            document_id = (row.get("id") or "").strip()
            if not document_id:
                raise SourceContractError(f"row {number} has an empty document id")
            if document_id in seen:
                raise SourceContractError(
                    f"duplicate document id {document_id!r} at row {number}; "
                    "identities would collide with question and passage IDs"
                )
            seen.add(document_id)

            questions = _decode_array(
                row.get("question_texts", ""),
                column="question_texts",
                document_id=document_id,
            )
            answers = _decode_array(
                row.get("output", ""), column="output", document_id=document_id
            )
            if len(questions) != len(answers):
                raise SourceContractError(
                    f"document {document_id!r}: question_texts has "
                    f"{len(questions)} entries but output has {len(answers)}"
                )
            yield row
