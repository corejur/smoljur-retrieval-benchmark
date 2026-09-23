from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.v4.retrieve import (
    load_candidates,
    load_corpus,
    load_queries,
    retrieve,
    write_run,
)


class StubEncoder:
    """Maps text to a vector by keyword, so ranking is predictable."""

    KEYS = ("alpha", "beta", "gamma")

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, texts, *, kind):
        self.calls.append(kind)
        return [
            [1.0 if key in text.lower() else 0.0 for key in self.KEYS] + [0.01]
            for text in texts
        ]


def _dataset(tmp_path: Path) -> Path:
    beir = tmp_path / "beir"
    (beir / "candidates").mkdir(parents=True)
    corpus = [
        {"_id": "d1:c0", "title": "SECAO", "text": "alpha content here"},
        {"_id": "d1:c1", "title": "", "text": "beta content here"},
        {"_id": "d2:c0", "title": "", "text": "gamma content here"},
    ]
    (beir / "corpus.jsonl").write_text(
        "\n".join(json.dumps(r) for r in corpus), encoding="utf-8"
    )
    (beir / "queries.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"_id": "d1:q0", "text": "find the beta thing"},
                {"_id": "d2:q0", "text": "find the gamma thing"},
            ]
        ),
        encoding="utf-8",
    )
    (beir / "candidates" / "test.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"query_id": "d1:q0", "document_id": "d1", "candidate_ids": ["d1:c0", "d1:c1"]},
                {"query_id": "d2:q0", "document_id": "d2", "candidate_ids": ["d2:c0"]},
            ]
        ),
        encoding="utf-8",
    )
    return beir


def _load(beir: Path):
    return (
        load_corpus(beir / "corpus.jsonl"),
        load_queries(beir / "queries.jsonl"),
        load_candidates(beir / "candidates" / "test.jsonl"),
    )


def test_ranks_only_within_the_query_document(tmp_path: Path) -> None:
    corpus, queries, candidates = _load(_dataset(tmp_path))
    run = retrieve(corpus, queries, candidates, StubEncoder())

    assert set(run) == {"d1:q0", "d2:q0"}
    # d2's chunk must never appear for a d1 query.
    assert set(run["d1:q0"]) == {"d1:c0", "d1:c1"}
    assert set(run["d2:q0"]) == {"d2:c0"}


def test_the_matching_chunk_ranks_first(tmp_path: Path) -> None:
    corpus, queries, candidates = _load(_dataset(tmp_path))
    run = retrieve(corpus, queries, candidates, StubEncoder())
    best = max(run["d1:q0"].items(), key=lambda item: item[1])[0]
    assert best == "d1:c1"


def test_encodes_queries_and_passages_separately(tmp_path: Path) -> None:
    corpus, queries, candidates = _load(_dataset(tmp_path))
    encoder = StubEncoder()
    retrieve(corpus, queries, candidates, encoder)
    assert encoder.calls == ["query", "passage"]


def test_top_k_truncates_the_ranking(tmp_path: Path) -> None:
    corpus, queries, candidates = _load(_dataset(tmp_path))
    run = retrieve(corpus, queries, candidates, StubEncoder(), top_k=1)
    assert all(len(ranked) == 1 for ranked in run.values())


def test_corpus_text_includes_the_section_title(tmp_path: Path) -> None:
    path = _dataset(tmp_path) / "corpus.jsonl"
    assert load_corpus(path)["d1:c0"].startswith("SECAO")
    assert load_corpus(path, use_title=False)["d1:c0"] == "alpha content here"


def test_rejects_a_query_with_no_candidate_pool(tmp_path: Path) -> None:
    corpus, queries, candidates = _load(_dataset(tmp_path))
    del candidates["d2:q0"]
    with pytest.raises(ValueError, match="no candidate pool"):
        retrieve(corpus, queries, candidates, StubEncoder())


def test_run_file_round_trips_through_metrics(tmp_path: Path) -> None:
    from scripts.v4.metrics import evaluate_run, load_run

    corpus, queries, candidates = _load(_dataset(tmp_path))
    run = retrieve(corpus, queries, candidates, StubEncoder())
    write_run(run, tmp_path / "runs" / "stub.json")

    reloaded = load_run(tmp_path / "runs" / "stub.json")
    qrels = {"d1:q0": {"d1:c1": 1}, "d2:q0": {"d2:c0": 1}}
    scores = evaluate_run(qrels, reloaded, model="stub", k_values=[10])
    assert scores.metrics["Recall@10"] == pytest.approx(1.0)


def test_trec_format_is_readable_by_metrics(tmp_path: Path) -> None:
    from scripts.v4.metrics import load_run

    corpus, queries, candidates = _load(_dataset(tmp_path))
    run = retrieve(corpus, queries, candidates, StubEncoder())
    write_run(run, tmp_path / "stub.run", fmt="trec")

    reloaded = load_run(tmp_path / "stub.run")
    assert set(reloaded) == set(run)
    assert set(reloaded["d1:q0"]) == set(run["d1:q0"])
