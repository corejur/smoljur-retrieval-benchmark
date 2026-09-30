"""Charts of mean retrieval metrics from a metrics.py comparison report.

    python -m scripts.v4.plots REPORT_JSON OUTPUT_PNG
        [--metrics NDCG MAP Recall P MRR Pass] [--k-values 1 3 5 10]
        [--pass-comparison]

Every value in a comparison report is already a mean over the judged
questions (missing ones counted as zero), so the chart reads the report and
recomputes nothing: one panel per metric family, cutoffs along the x-axis,
one bar per run, all panels on the same 0-1 scale, every bar labeled with its
value. The numbers are also printed as a table, the chart's accessible text
view.

`--pass-comparison` draws only Pass@k instead: one wide panel comparing the
models' hit rates at each cutoff, every bar labeled with its value.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["FAMILIES", "metric_means", "plot_metric_means", "plot_pass_comparison", "run_labels", "main"]

#: Metric families in display order, as named in the report's metric keys.
FAMILIES: tuple[str, ...] = ("NDCG", "MAP", "Recall", "P", "MRR", "Pass")
_TITLES = {"P": "Precision", "Pass": "Pass@k (hit rate)"}

#: Categorical series colors in fixed order, validated for color-vision
#: deficiency on the light surface. A run is never given a generated hue.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_SURFACE = "#fcfcfb"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3de"

#: Narrowest bar, in inches, that fits its value label ("0.56" at 7 pt is
#: about 0.2 in) with room to spare, so labels on adjacent bars never touch.
_MIN_BAR_INCHES = 0.3
#: Widest the metric chart grows before its panels wrap onto more rows.
_MAX_FIGURE_INCHES = 20.0

#: family -> cutoff -> run -> mean value
Means = dict[str, dict[int, dict[str, float]]]


def _load(report: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(report, Mapping):
        return report
    return json.loads(Path(report).read_text(encoding="utf-8"))


def run_labels(report: Mapping[str, Any] | str | Path) -> dict[str, str]:
    """How each run is named on a chart: its model's Hugging Face ID.

    The ID comes from the report's `model_id` (recorded from the run's
    provenance). Runs of the same model are told apart by their run name,
    e.g. `Qwen/Qwen3-Embedding-0.6B (qwen-top10)`; a run without a model ID
    keeps its run name.
    """
    runs = _load(report).get("models") or []
    ids = {str(run["model"]): run.get("model_id") for run in runs}
    shared = {
        model_id for model_id in ids.values() if model_id and list(ids.values()).count(model_id) > 1
    }
    return {
        run: (f"{model_id} ({run})" if model_id in shared else model_id) if model_id else run
        for run, model_id in ids.items()
    }


def metric_means(
    report: Mapping[str, Any] | str | Path,
    *,
    families: Sequence[str] | None = None,
    k_values: Sequence[int] | None = None,
) -> Means:
    """Arrange the report's per-run means as family -> cutoff -> run -> value."""
    loaded = _load(report)
    runs = loaded.get("models") or []
    unknown = [f for f in families or () if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown metric families {unknown}; choose from {list(FAMILIES)}")
    # Families the report actually carries: reports written before a family
    # existed (Pass@k) still plot, just without that panel.
    reported = {
        name.split("@", 1)[0] for run in runs for name in run.get("metrics") or {}
    }
    if families:
        absent = [f for f in families if runs and f not in reported]
        if absent:
            raise ValueError(f"families {absent} are not in the report; re-run metrics.py")
        chosen_families = list(families)
    else:
        chosen_families = [f for f in FAMILIES if not runs or f in reported]
    available = [int(k) for k in loaded.get("k_values") or []]
    chosen_k = [int(k) for k in (k_values or available)]
    missing_k = [k for k in chosen_k if k not in available]
    if missing_k:
        raise ValueError(f"cutoffs {missing_k} are not in the report (it has {available})")

    means: Means = {}
    for family in chosen_families:
        means[family] = {}
        for k in chosen_k:
            means[family][k] = {
                str(run["model"]): float(run["metrics"][f"{family}@{k}"]) for run in runs
            }
    return means


def plot_metric_means(
    report: Mapping[str, Any] | str | Path,
    output: str | Path,
    *,
    families: Sequence[str] | None = None,
    k_values: Sequence[int] | None = None,
    title: str | None = None,
) -> Path:
    """Write a PNG of mean metrics per run and return its path."""
    from matplotlib.figure import Figure

    loaded = _load(report)
    runs = _runs(loaded)
    means = metric_means(loaded, families=families, k_values=k_values)
    cutoffs = list(next(iter(means.values())))

    labels = run_labels(loaded)
    # Each panel is wide enough that every bar fits its value label; panels
    # wrap onto more rows rather than let the figure grow without bound.
    panel_inches = max(2.6, _MIN_BAR_INCHES * len(runs) * len(cutoffs) / 0.8 + 0.6)
    columns = max(1, min(len(means), int(_MAX_FIGURE_INCHES // panel_inches)))
    rows = -(-len(means) // columns)
    figure = Figure(
        figsize=(panel_inches * columns + 0.6, 3.2 * rows + 0.4), dpi=150, facecolor=_SURFACE
    )
    grid = figure.subplots(rows, columns, sharey=True, squeeze=False)
    axes = list(grid.flat)
    for panel, (family, by_k) in zip(axes, means.items()):
        _draw_bars(panel, by_k, runs, cutoffs, labels=labels)
        panel.set_title(_TITLES.get(family, family), fontsize=10, color=_TEXT_PRIMARY, loc="left")
    for unused in axes[len(means):]:
        unused.set_visible(False)
    for row in grid:
        row[0].set_ylabel("mean over judged questions", fontsize=8, color=_TEXT_SECONDARY)

    subject = labels[runs[0]] if len(runs) == 1 else f"{len(runs)} runs"
    _finish(figure, axes[0], loaded, runs, f"Mean retrieval metrics: {subject}", title=title)
    return _save(figure, output)


def plot_pass_comparison(
    report: Mapping[str, Any] | str | Path,
    output: str | Path,
    *,
    k_values: Sequence[int] | None = None,
    title: str | None = None,
) -> Path:
    """Write a PNG comparing the runs' Pass@k at each cutoff and return its path.

    One panel, cutoffs along the x-axis, one bar per run in the report's run
    order, on a 0-1 scale, every bar labeled with its value. The figure
    widens with the bar count so each bar fits its label.
    """
    from matplotlib.figure import Figure

    loaded = _load(report)
    runs = _runs(loaded)
    by_k = metric_means(loaded, families=["Pass"], k_values=k_values)["Pass"]
    cutoffs = list(by_k)
    width = max(6.0, 0.5 * len(runs) * len(cutoffs) + 2.4)
    figure = Figure(figsize=(width, 4.2), dpi=150, facecolor=_SURFACE)
    panel = figure.subplots()
    labels = run_labels(loaded)
    _draw_bars(panel, by_k, runs, cutoffs, labels=labels)
    panel.set_ylabel(
        "share of judged questions with a\nrelevant passage in the top k",
        fontsize=8,
        color=_TEXT_SECONDARY,
    )
    subject = labels[runs[0]] if len(runs) == 1 else f"{len(runs)} runs"
    # The legend gets its own row under the title: model names run long.
    _finish(figure, panel, loaded, runs, f"Pass@k (hit rate): {subject}", title=title, legend_row=True)
    return _save(figure, output)


def _runs(loaded: Mapping[str, Any]) -> list[str]:
    runs = [str(run["model"]) for run in loaded.get("models") or []]
    if not runs:
        raise ValueError("the report has no runs to plot")
    if len(runs) > len(_SERIES):
        raise ValueError(
            f"at most {len(_SERIES)} runs fit one chart ({len(runs)} given); "
            "split them across reports"
        )
    return runs


def _draw_bars(
    panel,
    by_k: Mapping[int, Mapping[str, float]],
    runs: Sequence[str],
    cutoffs: Sequence[int],
    *,
    labels: Mapping[str, str] | None = None,
) -> None:
    """One bar per run at each cutoff, each with its value printed above it.

    `labels` names each run in the legend (default: the run name).
    """
    panel.set_facecolor(_SURFACE)
    slot = 0.8 / len(runs)
    for index, run in enumerate(runs):
        positions = [c + (index - (len(runs) - 1) / 2) * slot for c in range(len(cutoffs))]
        values = [by_k[k][run] for k in cutoffs]
        bars = panel.bar(
            positions,
            values,
            width=slot,
            color=_SERIES[index],
            edgecolor=_SURFACE,  # the 2px gap between adjacent bars
            linewidth=1.0,
            label=(labels or {}).get(run, run),
            zorder=3,
        )
        # Callers size the panel so every bar is wide enough for its label.
        for bar, value in zip(bars, values):
            panel.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=_TEXT_SECONDARY,
            )
    panel.set_xticks(range(len(cutoffs)), [f"@{k}" for k in cutoffs])
    panel.set_ylim(0, 1)
    panel.grid(axis="y", color=_GRID, linewidth=0.6, zorder=0)
    panel.tick_params(colors=_TEXT_SECONDARY, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        panel.spines[side].set_visible(False)
    panel.spines["bottom"].set_color(_TEXT_SECONDARY)


def _finish(
    figure,
    legend_axes,
    loaded: Mapping[str, Any],
    runs: Sequence[str],
    subject: str,
    *,
    title: str | None = None,
    legend_row: bool = False,
) -> None:
    """Title the figure (`title` replaces the whole heading) and add the legend.

    The legend sits at the top right beside the title, or with `legend_row`
    on its own row below the title.
    """
    generation = (loaded.get("generation") or {}).get("generation_id")
    queries = (loaded.get("judgments") or {}).get("queries")
    heading = title or " · ".join(
        part
        for part in (
            subject,
            f"generation {generation}" if generation else "",
            f"{queries:,} judged questions" if queries else "",
        )
        if part
    )
    figure.suptitle(heading, fontsize=11, color=_TEXT_PRIMARY, x=0.01, ha="left")
    if len(runs) > 1:
        handles, labels = legend_axes.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper left" if legend_row else "upper right",
            bbox_to_anchor=(0.01, 0.93) if legend_row else None,
            ncol=min(len(runs), 4),
            frameon=False,
            fontsize=8,
            labelcolor=_TEXT_PRIMARY,
        )
    figure.tight_layout(rect=(0, 0, 1, 0.91 if legend_row and len(runs) > 1 else 0.96))


def _save(figure, output: str | Path) -> Path:
    """Replace `output` with the rendered PNG, or leave it untouched."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    try:
        figure.savefig(temporary, format="png", facecolor=_SURFACE)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def _table(means: Means, labels: Mapping[str, str] | None = None) -> str:
    runs = list(next(iter(next(iter(means.values())).values())))
    names = {run: (labels or {}).get(run, run) for run in runs}
    columns = [f"{family}@{k}" for family, by_k in means.items() for k in by_k]
    width = max(len(name) for name in names.values()) + 2
    lines = [f"{'model':<{width}}" + "".join(f"{c:>12}" for c in columns)]
    for run in runs:
        lines.append(
            f"{names[run]:<{width}}"
            + "".join(f"{means[family][k][run]:>12.4f}" for family, by_k in means.items() for k in by_k)
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("report", type=Path, help="Comparison report from metrics.py --output")
    parser.add_argument("output", type=Path, help="PNG to write")
    parser.add_argument(
        "--metrics", nargs="+", choices=FAMILIES, help="Families to plot (default: all in the report)"
    )
    parser.add_argument("--k-values", type=int, nargs="+", help="Cutoffs to plot (default: the report's)")
    parser.add_argument("--title", help="Override the chart title")
    parser.add_argument(
        "--pass-comparison",
        action="store_true",
        help="Plot only Pass@k, one wide panel comparing the runs at each cutoff",
    )
    args = parser.parse_args()
    if args.pass_comparison and args.metrics:
        parser.error("--pass-comparison plots Pass@k only; drop --metrics")

    if not args.report.is_file():
        parser.error(
            f"no report at {args.report}; write one first with "
            "python -m scripts.v4.metrics QRELS_TSV RUNS_DIRECTORY --output "
            f"{args.report}"
        )
    try:
        report = _load(args.report)
    except json.JSONDecodeError as error:
        parser.error(f"{args.report} is not JSON: {error}")
    if not isinstance(report, Mapping) or not {"models", "k_values"} <= set(report):
        parser.error(
            f"{args.report} looks like a run file, not a comparison report; score "
            "the runs with python -m scripts.v4.metrics ... --output REPORT_JSON "
            "and plot that report"
        )
    if args.pass_comparison:
        try:
            means = metric_means(report, families=["Pass"], k_values=args.k_values)
        except ValueError as error:
            parser.error(f"{args.report}: {error}")
        path = plot_pass_comparison(report, args.output, k_values=args.k_values, title=args.title)
    else:
        means = metric_means(report, families=args.metrics, k_values=args.k_values)
        path = plot_metric_means(
            report, args.output, families=args.metrics, k_values=args.k_values, title=args.title
        )
    print(_table(means, run_labels(report)))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
