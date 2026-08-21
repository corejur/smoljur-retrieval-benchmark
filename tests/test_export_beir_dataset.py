from __future__ import annotations

import csv
import json
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from scripts.export_beir_dataset import export_beir_dataset


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_export_creates_dataset_loadable_by_beir(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_csv(
        data / "chunks" / "doc-1.csv",
        ["chunk_id", "text", "section"],
        [
            {"chunk_id": "chunk-1", "text": "Texto relevante.", "section": "Fatos"},
            {"chunk_id": "chunk-2", "text": "Outro texto.", "section": "Pedidos"},
        ],
    )
    _write_csv(
        data / "queries" / "representative-redator-2k-v1-v4.csv",
        ["source_id", "source_query_number", "query", "relevant_chunks"],
        [
            {
                "source_id": "doc-1",
                "source_query_number": "1",
                "query": "Qual é o fato?",
                "relevant_chunks": json.dumps(["chunk-1", "chunk-1"]),
            },
            {
                "source_id": "doc-1",
                "source_query_number": "2",
                "query": "Sem citação",
                "relevant_chunks": "[]",
            },
        ],
    )

    output = tmp_path / "benchmark"
    summary = export_beir_dataset(data, output)

    assert summary.corpus_count == 2
    assert summary.query_count == 1
    assert summary.qrel_count == 1
    assert summary.skipped_query_count == 1
    corpus, queries, qrels = GenericDataLoader(str(output)).load(split="test")
    assert corpus["chunk-1"] == {"text": "Texto relevante.", "title": "Fatos"}
    assert queries == {"doc-1:q1": "Qual é o fato?"}
    assert qrels == {"doc-1:q1": {"chunk-1": 1}}
