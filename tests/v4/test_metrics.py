from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.v4.metrics import (
    comparison_table,
    evaluate_directory,
    evaluate_run,
    load_qrels,
    load_run,
)


def _qrels_file(tmp_path: Path) -> Path:
    path = tmp_path / "test.tsv"
    path.write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\tc1\t1\n"
        "q1\tc2\t1\n"
        "q2\tc5\t1\n",
        encoding="utf-8",
    )
    return path


def test_loads_qrels_with_header(tmp_path: Path) -> None:
    qrels = load_qrels(_qrels_file(tmp_path))
    assert qrels == {"q1": {"c1": 1, "c2": 1}, "q2": {"c5": 1}}


def test_loads_json_and_trec_runs_identically(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(
        json.dumps({"q1": {"c1": 2.0, "c9": 1.0}}), encoding="utf-8"
    )
    (tmp_path / "b.run").write_text(
        "q1 Q0 c1 1 2.0 tag\nq1 Q0 c9 2 1.0 tag\n", encoding="utf-8"
    )
    assert load_run(tmp_path / "a.json") == load_run(tmp_path / "b.run")


def test_a_perfect_run_scores_one(tmp_path: Path) -> None:
    qrels = load_qrels(_qrels_file(tmp_path))
    run = {"q1": {"c1": 3.0, "c2": 2.0}, "q2": {"c5": 3.0}}
    scores = evaluate_run(qrels, run, model="perfect", k_values=[10])
    assert scores.metrics["NDCG@10"] == pytest.approx(1.0)
    assert scores.metrics["Recall@10"] == pytest.approx(1.0)
    assert scores.metrics["MRR@10"] == pytest.approx(1.0)
    assert scores.queries_missing == 0


def test_a_run_missing_a_query_is_scored_as_zero_not_skipped(tmp_path: Path) -> None:
    qrels = load_qrels(_qrels_file(tmp_path))
    partial = evaluate_run(qrels, {"q1": {"c1": 3.0, "c2": 2.0}}, model="partial", k_values=[10])

    assert partial.queries_missing == 1
    # q2 unanswered must drag the mean down rather than vanish from it.
    assert partial.metrics["Recall@10"] == pytest.approx(0.5)
    assert partial.metrics["MRR@10"] == pytest.approx(0.5)


def test_ranking_order_changes_the_score(tmp_path: Path) -> None:
    qrels = load_qrels(_qrels_file(tmp_path))
    good = evaluate_run(qrels, {"q1": {"c1": 9.0, "x": 1.0}, "q2": {"c5": 9.0}}, model="g", k_values=[10])
    bad = evaluate_run(qrels, {"q1": {"x": 9.0, "c1": 1.0}, "q2": {"y": 9.0, "c5": 1.0}}, model="b", k_values=[10])
    assert good.metrics["MRR@10"] > bad.metrics["MRR@10"]


def test_evaluates_every_run_in_a_directory(tmp_path: Path) -> None:
    qrels_path = _qrels_file(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "model_a.json").write_text(
        json.dumps({"q1": {"c1": 3.0, "c2": 2.0}, "q2": {"c5": 3.0}}), encoding="utf-8"
    )
    (runs / "model_b.json").write_text(
        json.dumps({"q1": {"zz": 3.0}, "q2": {"c5": 1.0}}), encoding="utf-8"
    )

    scores = evaluate_directory(qrels_path, runs, k_values=[10])

    assert {item.model for item in scores} == {"model_a", "model_b"}
    table = comparison_table(scores, ["NDCG@10", "Recall@10"])
    # Best model sorts first in the leaderboard.
    assert table.splitlines()[2].startswith("model_a")


def test_rejects_a_run_sharing_no_query(tmp_path: Path) -> None:
    qrels = load_qrels(_qrels_file(tmp_path))
    with pytest.raises(ValueError, match="shares no query"):
        evaluate_run(qrels, {"other": {"c1": 1.0}}, model="x", k_values=[10])


def test_skips_its_own_report_left_in_the_runs_directory(tmp_path: Path) -> None:
    qrels_path = _qrels_file(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "model_a.json").write_text(
        json.dumps({"q1": {"c1": 3.0}, "q2": {"c5": 3.0}}), encoding="utf-8"
    )
    # A previous --output written into the same directory must not be scored.
    (runs / "metrics.json").write_text(
        json.dumps({"qrels": "x", "k_values": [10], "models": []}), encoding="utf-8"
    )

    scores = evaluate_directory(qrels_path, runs, k_values=[10])
    assert [item.model for item in scores] == ["model_a"]


def test_names_the_file_when_a_run_is_malformed(tmp_path: Path) -> None:
    bad = tmp_path / "broken.json"
    bad.write_text(json.dumps({"q1": "not-a-mapping"}), encoding="utf-8")
    with pytest.raises(ValueError, match="broken.json"):
        load_run(bad)
