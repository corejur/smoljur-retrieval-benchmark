"""Mean-metric bar charts from a comparison report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts.v4.plots import metric_means, plot_metric_means, plot_pass_comparison  # noqa: E402

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


# --- Pass@k comparison --------------------------------------------------------


def _capture_figures(monkeypatch) -> list:
    """Record each figure the plots module saves, still writing the PNG."""
    from scripts.v4 import plots

    figures: list = []
    save = plots._save

    def capture(figure, output):
        figures.append(figure)
        return save(figure, output)

    monkeypatch.setattr(plots, "_save", capture)
    return figures


def test_the_pass_comparison_writes_one_panel_with_a_bar_per_run_and_cutoff(tmp_path: Path, monkeypatch) -> None:
    figures = _capture_figures(monkeypatch)
    report = _with_pass(_report(("qwen", 0.5), ("jina", 0.4), ("bge", 0.3)))

    path = plot_pass_comparison(report, tmp_path / "pass.png")

    assert path.read_bytes()[:8] == PNG_SIGNATURE
    (panel,) = figures[0].axes
    assert len(panel.patches) == 3 * 3
    assert [tick.get_text() for tick in panel.get_xticklabels()] == ["@1", "@3", "@10"]
    assert "Pass@k" in figures[0]._suptitle.get_text()


def test_every_pass_bar_is_labeled(tmp_path: Path, monkeypatch) -> None:
    figures = _capture_figures(monkeypatch)
    report = _with_pass(_report(("qwen", 0.5), ("jina", 0.4)))

    plot_pass_comparison(report, tmp_path / "pass.png")

    assert [text.get_text() for text in figures[0].axes[0].texts] == ["0.90"] * 6


def test_a_crowded_pass_chart_still_labels_every_bar_without_overlap(tmp_path: Path, monkeypatch) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figures = _capture_figures(monkeypatch)
    runs = [(f"run-{i}", 0.5) for i in range(8)]  # 8 runs x 3 cutoffs = 24 bars

    plot_pass_comparison(_with_pass(_report(*runs)), tmp_path / "pass.png")

    panel = figures[0].axes[0]
    renderer = FigureCanvasAgg(figures[0]).get_renderer()
    boxes = [text.get_window_extent(renderer) for text in panel.texts]
    assert len(boxes) == 8 * 3
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])


def test_the_pass_comparison_can_select_cutoffs(tmp_path: Path, monkeypatch) -> None:
    figures = _capture_figures(monkeypatch)

    plot_pass_comparison(_with_pass(_report(("qwen", 0.5))), tmp_path / "pass.png", k_values=[10])

    assert [tick.get_text() for tick in figures[0].axes[0].get_xticklabels()] == ["@10"]


def test_a_pass_comparison_of_a_report_without_pass_at_k_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Pass.*not in the report"):
        plot_pass_comparison(_report(("qwen", 0.5)), tmp_path / "pass.png")
    assert not (tmp_path / "pass.png").exists()


def test_the_cli_plots_the_pass_comparison(tmp_path: Path, monkeypatch, capsys) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_with_pass(_report(("qwen", 0.5), ("bge", 0.3)))), encoding="utf-8")
    output = tmp_path / "pass.png"

    _cli(monkeypatch, report, output, "--pass-comparison")

    assert output.read_bytes()[:8] == PNG_SIGNATURE
    printed = capsys.readouterr().out
    assert "Pass@10" in printed and "NDCG" not in printed and "bge" in printed


def test_the_cli_explains_a_report_without_pass_at_k(tmp_path: Path, monkeypatch, capsys) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report(("qwen", 0.5))), encoding="utf-8")

    with pytest.raises(SystemExit):
        _cli(monkeypatch, report, tmp_path / "pass.png", "--pass-comparison")

    error = capsys.readouterr().err
    assert "re-run metrics.py" in error and "Traceback" not in error


def test_the_cli_refuses_metrics_with_the_pass_comparison(tmp_path: Path, monkeypatch, capsys) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_with_pass(_report(("qwen", 0.5)))), encoding="utf-8")

    with pytest.raises(SystemExit):
        _cli(monkeypatch, report, tmp_path / "pass.png", "--pass-comparison", "--metrics", "NDCG")

    assert "--pass-comparison" in capsys.readouterr().err


# --- charts name each run by its model -----------------------------------------


def _with_ids(report: dict, **ids: str) -> dict:
    for run in report["models"]:
        if run["model"] in ids:
            run["model_id"] = ids[run["model"]]
    return report


def test_runs_are_labeled_with_their_models_hugging_face_id() -> None:
    from scripts.v4.plots import run_labels

    report = _with_ids(
        _report(("qwen-top10", 0.5), ("qwen-top100", 0.5), ("bge", 0.4), ("adhoc", 0.3)),
        **{
            "qwen-top10": "Qwen/Qwen3-Embedding-0.6B",
            "qwen-top100": "Qwen/Qwen3-Embedding-0.6B",
            "bge": "BAAI/bge-m3",
        },
    )

    assert run_labels(report) == {
        "qwen-top10": "Qwen/Qwen3-Embedding-0.6B (qwen-top10)",
        "qwen-top100": "Qwen/Qwen3-Embedding-0.6B (qwen-top100)",
        "bge": "BAAI/bge-m3",
        "adhoc": "adhoc",  # no model_id (e.g. a report from before model IDs): the run name
    }


def test_chart_legends_and_the_table_show_the_model_ids(tmp_path: Path, monkeypatch, capsys) -> None:
    figures = _capture_figures(monkeypatch)
    report = _with_pass(
        _with_ids(
            _report(("jina", 0.5), ("bge", 0.4)),
            jina="jinaai/jina-embeddings-v5-text-small",
            bge="BAAI/bge-m3",
        )
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    _cli(monkeypatch, path, tmp_path / "pass.png", "--pass-comparison")
    plot_metric_means(report, tmp_path / "metrics.png")

    for figure in figures:
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "jinaai/jina-embeddings-v5-text-small",
            "BAAI/bge-m3",
        ]
    assert "BAAI/bge-m3" in capsys.readouterr().out


@pytest.mark.parametrize("run_count", [2, 3, 4, 8])
def test_metric_chart_value_labels_never_overlap(tmp_path: Path, monkeypatch, run_count: int) -> None:
    figures = _capture_figures(monkeypatch)
    report = _with_pass(_report(*[(f"model-{i}", 0.5) for i in range(run_count)]))
    report["k_values"] = [1, 3, 10]

    plot_metric_means(report, tmp_path / "metrics.png")

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = figures[0]
    renderer = FigureCanvasAgg(figure).get_renderer()
    for panel in figure.axes:
        if not panel.get_visible():
            continue
        boxes = [text.get_window_extent(renderer) for text in panel.texts]
        assert len(boxes) == run_count * 3  # every bar at every cutoff
        for index, box in enumerate(boxes):
            for other in boxes[index + 1 :]:
                assert not box.overlaps(other), f"labels overlap in {panel.get_title(loc='left')!r}"
