from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.prepare_dataset import prepare_dataset
from scripts.query_preparation import filter_fully_answered_sources
from scripts.strategies.chunking.legal_recursive import ChunkingConfig


def test_filters_document_when_answer_count_does_not_match_queries() -> None:
    rows = [
        {
            "id": "complete",
            "query": "1. Primeira?\n2. Segunda?",
            "output": json.dumps([{"citations": []}, {"citations": []}]),
        },
        {
            "id": "incomplete",
            "query": "1. Primeira?\n2. Segunda?",
            "output": json.dumps([{"citations": []}]),
        },
    ]

    selected = filter_fully_answered_sources(rows)

    assert [row["id"] for row in selected.rows] == ["complete"]
    assert selected.source_indexes == (0,)
    assert selected.removed_document_ids == ("incomplete",)


def test_pipeline_excludes_incomplete_document_and_all_its_queries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=["id", "query", "data", "output"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "id": "complete",
                    "query": "1. Respondida?",
                    "data": "<p id='p-1'>Documento completo.</p>",
                    "output": json.dumps([{"citations": ["p-1"]}]),
                },
                {
                    "id": "incomplete",
                    "query": "1. Respondida?\n2. Sem resposta?",
                    "data": "<p id='p-1'>Documento incompleto.</p>",
                    "output": json.dumps([{"citations": ["p-1"]}]),
                },
            ]
        )

    output = tmp_path / "output"
    summary = prepare_dataset(
        source,
        output,
        chunking=ChunkingConfig(
            max_words=20,
            target_words=15,
            overlap_words=2,
            minimum_words=1,
        ),
    )

    assert summary.document_count == 1
    assert summary.query_count == 1
    assert summary.removed_document_count == 1
    assert (output / "chunks/complete.csv").exists()
    assert not (output / "chunks/incomplete.csv").exists()

    with (
        output / "queries/representative-redator-2k-v1-v4.csv"
    ).open(encoding="utf-8", newline="") as query_file:
        query_rows = list(csv.DictReader(query_file))
    assert {row["source_id"] for row in query_rows} == {"complete"}

    with (output / "documents/documents.cleaned.json").open(
        encoding="utf-8"
    ) as document_file:
        documents = json.load(document_file)
    assert [document["document_id"] for document in documents] == ["complete"]
