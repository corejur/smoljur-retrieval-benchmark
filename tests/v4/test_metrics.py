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


# --- User Story 5: fair comparison of two or more runs (T031) ----------------


def _runs_dir(tmp_path: Path, **runs) -> Path:
    directory = tmp_path / "runs"
    directory.mkdir(exist_ok=True)
    for name, run in runs.items():
        (directory / f"{name}.json").write_text(json.dumps(run), encoding="utf-8")
    return directory


def test_two_runs_share_one_qrels_set_and_the_same_cutoffs(tmp_path: Path) -> None:
    runs = _runs_dir(
        tmp_path,
        qwen_top3={"q1": {"c1": 3.0, "c2": 2.0, "c7": 1.0}, "q2": {"c5": 1.0}},
        qwen_top1={"q1": {"c7": 1.0}, "q2": {"c5": 1.0}},
    )

    scores = evaluate_directory(_qrels_file(tmp_path), runs, k_values=[1, 3])

    assert [item.model for item in scores] == ["qwen_top1", "qwen_top3"]
    for item in scores:
        assert set(item.metrics) == {
            f"{name}@{k}"
            for name in ("NDCG", "MAP", "Recall", "P", "MRR")
            for k in (1, 3)
        }
    by_model = {item.model: item.metrics for item in scores}
    assert by_model["qwen_top3"]["Recall@3"] == 1.0
    assert by_model["qwen_top1"]["Recall@3"] == 0.5


def test_a_missing_judged_query_is_zero_filled_and_counted(tmp_path: Path) -> None:
    runs = _runs_dir(
        tmp_path,
        complete={"q1": {"c1": 2.0, "c2": 1.0}, "q2": {"c5": 1.0}},
        partial={"q1": {"c1": 2.0, "c2": 1.0}},
    )

    scores = {s.model: s for s in evaluate_directory(_qrels_file(tmp_path), runs, k_values=[10])}

    assert (scores["partial"].queries_scored, scores["partial"].queries_missing) == (1, 1)
    assert scores["partial"].metrics["NDCG@10"] == pytest.approx(0.5)
    assert scores["complete"].metrics["NDCG@10"] == 1.0


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_json_score_is_rejected(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "run.json"
    path.write_text('{"q1": {"c1": %s}}' % bad, encoding="utf-8")

    with pytest.raises(ValueError, match="run.json.*finite"):
        load_run(path)


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_a_non_finite_trec_score_is_rejected(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "run.trec"
    path.write_text(f"q1 Q0 c1 1 {bad} tag\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run.trec:1.*finite"):
        load_run(path)


@pytest.mark.parametrize("bad", ['"high"', "true", "null", "[1]"])
def test_a_non_numeric_json_score_is_rejected(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "run.json"
    path.write_text('{"q1": {"c1": %s}}' % bad, encoding="utf-8")

    with pytest.raises(ValueError, match="run.json.*q1.*c1"):
        load_run(path)


def test_a_malformed_trec_score_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "run.trec"
    path.write_text("q1 Q0 c1 1 2.0 tag\nq1 Q0 c2 2 high tag\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run.trec:2"):
        load_run(path)


def test_a_duplicate_trec_row_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "run.trec"
    path.write_text("q1 Q0 c1 1 2.0 tag\nq1 Q0 c1 2 1.0 tag\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run.trec:2.*c1"):
        load_run(path)


def test_an_empty_json_run_is_rejected_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="empty run"):
        load_run(path)


def test_an_unrelated_run_in_a_directory_fails_the_comparison(tmp_path: Path) -> None:
    runs = _runs_dir(
        tmp_path,
        good={"q1": {"c1": 1.0}},
        unrelated={"other-q": {"c1": 1.0}},
    )

    with pytest.raises(ValueError, match="unrelated.*shares no query"):
        evaluate_directory(_qrels_file(tmp_path), runs, k_values=[10])


# --- the comparison report (T033) --------------------------------------------


def _report(tmp_path: Path, **kwargs):
    from scripts.v4.metrics import comparison_report

    runs = _runs_dir(
        tmp_path,
        qwen_top3={"q1": {"c1": 3.0, "c2": 2.0}, "q2": {"c5": 1.0}},
        qwen_top1={"q1": {"c1": 1.0}},
    )
    qrels = kwargs.pop("qrels", None) or _qrels_file(tmp_path)
    k_values = [1, 3]
    scores = evaluate_directory(qrels, runs, k_values=k_values)
    return comparison_report(qrels, scores, k_values=k_values)


def test_the_report_carries_judgment_provenance_and_common_cutoffs(tmp_path: Path) -> None:
    import hashlib

    report = _report(tmp_path)
    qrels = tmp_path / "test.tsv"

    assert report["k_values"] == [1, 3]
    assert report["judgments"] == {
        "qrels": str(qrels),
        "qrels_sha256": hashlib.sha256(qrels.read_bytes()).hexdigest(),
        "queries": 2,
        "judgments": 3,
    }
    # Supplied qrels outside a published generation carry no generation.
    assert report["generation"] is None


def test_the_report_lists_every_run_with_its_counts_and_measures(tmp_path: Path) -> None:
    report = _report(tmp_path)

    models = {entry["model"]: entry for entry in report["models"]}
    assert set(models) == {"qwen_top1", "qwen_top3"}
    assert (models["qwen_top1"]["queries_scored"], models["qwen_top1"]["queries_missing"]) == (1, 1)
    assert set(models["qwen_top3"]["metrics"]) >= {"NDCG@3", "MAP@3", "Recall@3", "P@3", "MRR@3"}


def test_the_report_names_the_generation_its_qrels_came_from(tmp_path: Path) -> None:
    import hashlib

    from scripts.v4.generation import manifest_sha256
    from scripts.v4.run import build_dataset

    generation_id = "20260923-000000-abcdef"
    generation = tmp_path / "dataset" / "generations" / generation_id
    build_dataset(
        [
            {
                "id": "doc-a",
                "question_texts": json.dumps(["Quem e o autor?"]),
                "output": json.dumps([{"answer": "Joao", "citations": ["p-0"]}]),
                "data": (
                    '<p id="p-0">O autor da acao e Joao da Silva residente na capital</p>'
                    '<p id="p-1">O valor da causa foi fixado pela parte interessada</p>'
                    '<p id="p-2">Terceiro paragrafo com texto suficiente para o documento</p>'
                ),
            }
        ],
        generation,
        generation_id=generation_id,
    )
    (tmp_path / "dataset" / "current").symlink_to(f"generations/{generation_id}")
    qrels = tmp_path / "dataset" / "current" / "beir" / "qrels" / "test.tsv"

    from scripts.v4.metrics import comparison_report

    pinned = generation / "beir" / "qrels" / "test.tsv"
    report = comparison_report(qrels, [], k_values=[10])

    assert report["generation"] == {
        "generation_id": generation_id,
        "manifest_sha256": manifest_sha256(generation),
        "split": "test",
    }
    # The report names the immutable path, never the moving `current`.
    assert report["judgments"]["qrels"] == str(pinned.resolve())
    assert report["judgments"]["qrels_sha256"] == hashlib.sha256(pinned.read_bytes()).hexdigest()


def test_the_report_is_still_recognised_and_skipped_as_a_run(tmp_path: Path) -> None:
    from scripts.v4.metrics import _is_metrics_report

    assert _is_metrics_report(_report(tmp_path))


def test_the_cli_refuses_to_write_the_report_into_the_runs_directory(
    tmp_path: Path, monkeypatch
) -> None:
    import sys

    from scripts.v4 import metrics

    runs = _runs_dir(tmp_path, qwen={"q1": {"c1": 1.0}})
    monkeypatch.setattr(
        sys,
        "argv",
        ["metrics", str(_qrels_file(tmp_path)), str(runs), "--output", str(runs / "report.json")],
    )

    with pytest.raises(SystemExit):
        metrics.main()
    assert not (runs / "report.json").exists()


def test_the_cli_writes_a_complete_report_outside_the_runs_directory(
    tmp_path: Path, monkeypatch
) -> None:
    import sys

    from scripts.v4 import metrics

    runs = _runs_dir(
        tmp_path,
        qwen_top3={"q1": {"c1": 3.0, "c2": 2.0}, "q2": {"c5": 1.0}},
        qwen_top1={"q1": {"c1": 1.0}},
    )
    output = tmp_path / "reports" / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["metrics", str(_qrels_file(tmp_path)), str(runs), "--k-values", "1", "3", "--output", str(output)],
    )

    metrics.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert [entry["model"] for entry in report["models"]] == ["qwen_top1", "qwen_top3"]
    assert report["k_values"] == [1, 3]
    assert not list(output.parent.glob(".*tmp"))
