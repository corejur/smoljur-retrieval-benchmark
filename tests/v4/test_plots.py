"""Mean-metric bar charts from a comparison report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts.v4.plots import metric_means, plot_metric_means  # noqa: E402

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _report(*models: tuple[str, float]) -> dict:
    """A comparison report whose every measure is `base` scaled by k."""
    return {
        "k_values": [1, 3, 10],
        "generation": {"generation_id": "gen-1", "manifest_sha256": "a" * 64, "split": "test"},
        "judgments": {"qrels": "q.tsv", "qrels_sha256": "b" * 64, "queries": 7, "judgments": 9},
        "models": [
            {
                "model": name,
                "queries_scored": 7,
                "queries_missing": 0,
                "metrics": {
                    f"{family}@{k}": round(base * k / 10, 5)
                    for family in ("NDCG", "MAP", "Recall", "P", "MRR")
                    for k in (1, 3, 10)
                },
            }
            for name, base in models
        ],
    }


def test_metric_means_groups_by_family_then_cutoff_then_run() -> None:
    means = metric_means(_report(("qwen", 0.5), ("baseline", 0.3)))

    assert list(means) == ["NDCG", "MAP", "Recall", "P", "MRR"]
    assert list(means["Recall"]) == [1, 3, 10]
    assert means["Recall"][10] == {"qwen": 0.5, "baseline": 0.3}
    assert means["NDCG"][1] == {"qwen": 0.05, "baseline": 0.03}


def test_metric_means_can_select_families_and_cutoffs() -> None:
    means = metric_means(_report(("qwen", 0.5)), families=["MRR", "NDCG"], k_values=[10])

    assert means == {"MRR": {10: {"qwen": 0.5}}, "NDCG": {10: {"qwen": 0.5}}}


def test_an_unknown_family_or_cutoff_is_rejected() -> None:
    with pytest.raises(ValueError, match="F1"):
        metric_means(_report(("qwen", 0.5)), families=["F1"])
    with pytest.raises(ValueError, match="100"):
        metric_means(_report(("qwen", 0.5)), k_values=[100])


def test_writes_a_png(tmp_path: Path) -> None:
    output = plot_metric_means(_report(("qwen", 0.5), ("baseline", 0.3)), tmp_path / "charts" / "m.png")

    assert output == tmp_path / "charts" / "m.png"
    assert output.read_bytes()[:8] == PNG_SIGNATURE


def test_reads_the_report_from_a_path(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report(("qwen", 0.5))), encoding="utf-8")

    output = plot_metric_means(report, tmp_path / "m.png")

    assert output.read_bytes()[:8] == PNG_SIGNATURE


def test_more_runs_than_categorical_colors_is_refused(tmp_path: Path) -> None:
    report = _report(*[(f"run{i}", 0.5) for i in range(9)])

    with pytest.raises(ValueError, match="8 runs"):
        plot_metric_means(report, tmp_path / "m.png")
    assert not (tmp_path / "m.png").exists()


def test_a_report_without_runs_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no runs"):
        plot_metric_means(_report(), tmp_path / "m.png")


def test_the_cli_writes_the_chart_and_prints_a_table(tmp_path: Path, monkeypatch, capsys) -> None:
    from scripts.v4 import plots

    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report(("qwen", 0.5), ("baseline", 0.3))), encoding="utf-8")
    output = tmp_path / "m.png"
    monkeypatch.setattr(sys, "argv", ["plots", str(report), str(output), "--k-values", "10"])

    plots.main()

    assert output.read_bytes()[:8] == PNG_SIGNATURE
    printed = capsys.readouterr().out
    assert "Recall@10" in printed and "0.5000" in printed and "baseline" in printed


def _cli(monkeypatch, *argv):
    from scripts.v4 import plots

    monkeypatch.setattr(sys, "argv", ["plots", *map(str, argv)])
    plots.main()


def test_a_missing_report_names_the_metrics_step(tmp_path: Path, monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit):
        _cli(monkeypatch, tmp_path / "absent.json", tmp_path / "m.png")

    error = capsys.readouterr().err
    assert "absent.json" in error and "scripts.v4.metrics" in error and "--output" in error
    assert "Traceback" not in error


def test_a_run_file_given_instead_of_a_report_is_explained(tmp_path: Path, monkeypatch, capsys) -> None:
    run = tmp_path / "qwen.json"
    run.write_text(json.dumps({"q1": {"c1": 0.9}}), encoding="utf-8")

    with pytest.raises(SystemExit):
        _cli(monkeypatch, run, tmp_path / "m.png")

    assert "run file" in capsys.readouterr().err
    assert not (tmp_path / "m.png").exists()


def _with_pass(report: dict) -> dict:
    for run in report["models"]:
        run["metrics"].update({f"Pass@{k}": 0.9 for k in report["k_values"]})
    return report


def test_pass_at_k_gets_its_own_panel_when_the_report_has_it() -> None:
    means = metric_means(_with_pass(_report(("qwen", 0.5))))

    assert list(means) == ["NDCG", "MAP", "Recall", "P", "MRR", "Pass"]
    assert means["Pass"][10] == {"qwen": 0.9}


def test_a_report_from_before_pass_at_k_still_plots(tmp_path: Path) -> None:
    report = _report(("qwen", 0.5))  # no Pass@k keys

    assert "Pass" not in metric_means(report)
    assert plot_metric_means(report, tmp_path / "m.png").read_bytes()[:8] == PNG_SIGNATURE


def test_asking_for_a_family_the_report_lacks_is_explained() -> None:
    with pytest.raises(ValueError, match="Pass.*not in the report"):
        metric_means(_report(("qwen", 0.5)), families=["Pass"])
