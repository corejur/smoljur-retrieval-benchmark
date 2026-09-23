"""Extract HTML documents from a CSV into one plain-text JSON corpus."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.persistence import write_json
from scripts.source_normalization import normalize_source_html

DEFAULT_DOCUMENT_DIRECTORY = Path("data/documents")
DEFAULT_DOCUMENT_JSON = DEFAULT_DOCUMENT_DIRECTORY / "documents.json"

def _html_to_plain_text(document_html: str) -> str:
    return normalize_source_html(document_html).text


def extract_documents(
    source_rows: Sequence[Mapping[str, str]],
    *,
    data_column: str = "data",
    id_column: str = "id",
) -> list[dict[str, str]]:
    """Convert source rows into normalized document records in memory."""
    documents: list[dict[str, str]] = []
    for source_index, row in enumerate(source_rows):
        missing = {
            column for column in (data_column, id_column) if column not in row
        }
        if missing:
            raise ValueError(
                f"source row {source_index} is missing columns: "
                + ", ".join(sorted(missing))
            )
        documents.append(
            {
                "document_id": row[id_column],
                "text": _html_to_plain_text(row[data_column]),
            }
        )
    return documents



def document_extractor(
    input_csv: str | Path,
    output_json: str | Path = DEFAULT_DOCUMENT_JSON,
    *,
    data_column: str = "data",
    id_column: str = "id",
) -> int:
    """Extract source documents as plain text and save them as a JSON array."""
    input_path = Path(input_csv)
    output_path = Path(output_json)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_csv and output_json must be different files")

    csv.field_size_limit(sys.maxsize)
    with input_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        documents = extract_documents(
            list(reader), data_column=data_column, id_column=id_column
        )

    write_json(output_path, documents)
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract HTML documents from a CSV into plain-text JSON."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "output_json",
        type=Path,
        nargs="?",
        default=DEFAULT_DOCUMENT_JSON,
    )
    args = parser.parse_args()
    count = document_extractor(args.input_csv, args.output_json)
    print(f"Wrote {count} plain-text documents to {args.output_json}")


if __name__ == "__main__":
    main()
