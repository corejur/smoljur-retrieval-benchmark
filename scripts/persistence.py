"""Atomic local-file persistence for dataset preparation outputs."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO


def _atomic_write(
    destination: str | Path,
    write: Callable[[TextIO], None],
    *,
    newline: str | None = None,
) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline=newline
        ) as target:
            write(target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(destination: str | Path, value: Any) -> None:
    """Atomically write indented UTF-8 JSON with a trailing newline."""

    def serialize(target: TextIO) -> None:
        json.dump(value, target, ensure_ascii=False, indent=2)
        target.write("\n")

    _atomic_write(destination, serialize)


def write_jsonl(
    destination: str | Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Atomically write one JSON object per UTF-8 line."""

    def serialize(target: TextIO) -> None:
        for row in rows:
            json.dump(row, target, ensure_ascii=False)
            target.write("\n")

    _atomic_write(destination, serialize)


def write_csv(
    destination: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Atomically write a UTF-8 CSV with the supplied column order."""

    def serialize(target: TextIO) -> None:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _atomic_write(destination, serialize, newline="")


def write_tsv(
    destination: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Atomically write a UTF-8 tab-separated file."""

    def serialize(target: TextIO) -> None:
        writer = csv.DictWriter(
            target, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    _atomic_write(destination, serialize, newline="")
