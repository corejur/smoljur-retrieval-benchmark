from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.prepare_dataset import prepare_dataset
from scripts.strategies.chunking.legal_recursive import ChunkingConfig


def test_pipeline_builds_ground_truth_free_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["id", "query", "data", "output"])
        writer.writeheader()
        writer.writerow(
            {
                "id": "doc-1",
                "query": "1. Qual é o fato?",
                "data": "<h1>FATOS</h1><p id='p-1'>Um fato relevante.</p>",
                "output": json.dumps([{"citations": ["p-1"]}]),
            }
        )

    summary = prepare_dataset(
        source,
        tmp_path / "output",
        chunking=ChunkingConfig(
            max_words=20,
            target_words=15,
            overlap_words=2,
            minimum_words=1,
        ),
    )

    assert summary.query_count == 1
    assert summary.document_count == 1
    assert summary.chunk_count == 1
    assert (tmp_path / "output/chunks/doc-1.csv").exists()
    assert (tmp_path / "output/chunks/manifest.json").exists()
    assert not list((tmp_path / "output").glob("ground_truth*"))
