"""Incremental, atomic JSONL/TSV writers for corpus-sized output.

The v1-v3 pipeline held whole corpora in memory and wrote single JSON arrays.
At v4 train scale (35k documents, ~1M chunks) that is not viable, so output is
streamed record by record into a temporary file that is moved into place on
close.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["JsonlWriter", "TsvWriter"]


class _AtomicTextWriter:
    def __init__(self, destination: str | Path) -> None:
        self.path = Path(destination)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        self._temporary = name
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        self.count = 0

    def __enter__(self):  # noqa: ANN204 - context manager protocol
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            try:
                self._handle.flush()
                os.fsync(self._handle.fileno())
                self._handle.close()
                os.replace(self._temporary, self.path)
            except BaseException:
                self._discard()
                raise
            return
        self._discard()

    def _discard(self) -> None:
        """Close and remove the temporary file; the destination is untouched."""
        try:
            self._handle.close()
        finally:
            try:
                os.unlink(self._temporary)
            except FileNotFoundError:
                pass


class JsonlWriter(_AtomicTextWriter):
    """Write one JSON object per line, atomically on clean exit."""

    def write(self, record: Mapping[str, Any]) -> None:
        # Serialize the whole record before touching the file, so a record
        # that cannot be encoded leaves no partial line. NaN and Infinity are
        # not JSON and would make the line unreadable to a strict parser.
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self._handle.write(line + "\n")
        self.count += 1

    def extend(self, records: Iterable[Mapping[str, Any]]) -> None:
        for record in records:
            self.write(record)


class TsvWriter(_AtomicTextWriter):
    """Write a tab-separated file with a header, atomically on clean exit."""

    def __init__(self, destination: str | Path, fieldnames: Sequence[str]) -> None:
        super().__init__(destination)
        self.fieldnames = tuple(fieldnames)
        self._handle.write("\t".join(self.fieldnames) + "\n")

    def write(self, record: Mapping[str, Any]) -> None:
        values = [str(record[name]) for name in self.fieldnames]
        for name, value in zip(self.fieldnames, values):
            if any(separator in value for separator in "\t\r\n"):
                raise ValueError(
                    f"{self.path.name}: {name} value {value!r} contains a tab or "
                    "line break and would split the row"
                )
        self._handle.write("\t".join(values) + "\n")
        self.count += 1
