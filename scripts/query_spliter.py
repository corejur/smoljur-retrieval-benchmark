"""Prepare one retrieval query row per question, including source citations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from scripts.persistence import write_csv
from scripts.query_preparation import PreparedQueries, prepare_query_rows

DEFAULT_QUERY_OUTPUT = Path("data/queries/representative-redator-2k-v1-v4.csv")


def prepare_queries(
    input_csv: str | Path,
    output_csv: str | Path = DEFAULT_QUERY_OUTPUT,
) -> PreparedQueries:
    """Read a source dataset and atomically write fully prepared query rows."""
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_csv and output_csv must be different files")

    csv.field_size_limit(sys.maxsize)
    with input_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        source_rows = list(reader)

    existing_rows: list[dict[str, str]] = []
    if output_path.exists() and output_path.stat().st_size:
        with output_path.open("r", encoding="utf-8", newline="") as existing:
            reader = csv.DictReader(existing)
            if reader.fieldnames is None:
                raise ValueError("output CSV has no header")
            existing_rows = list(reader)

    prepared = prepare_query_rows(source_rows, existing_rows)
    write_csv(output_path, prepared.rows, prepared.fieldnames)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "output_csv", type=Path, nargs="?", default=DEFAULT_QUERY_OUTPUT
    )
    args = parser.parse_args()
    prepared = prepare_queries(args.input_csv, args.output_csv)
    print(f"Wrote {len(prepared.rows)} prepared query rows to {args.output_csv}")


if __name__ == "__main__":
    main()
