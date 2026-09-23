"""Charts of mean retrieval metrics from a metrics.py comparison report.

    python -m scripts.v4.plots REPORT_JSON OUTPUT_PNG
        [--metrics NDCG MAP Recall P MRR Pass] [--k-values 1 3 5 10]

Every value in a comparison report is already a mean over the judged
questions (missing ones counted as zero), so the chart reads the report and
recomputes nothing: one panel per metric family, cutoffs along the x-axis,
one bar per run, all panels on the same 0-1 scale. The numbers are also
printed as a table, the chart's accessible text view.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["FAMILIES", "metric_means", "plot_metric_means", "main"]

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

#: family -> cutoff -> run -> mean value
Means = dict[str, dict[int, dict[str, float]]]


def _load(report: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(report, Mapping):
        return report
    return json.loads(Path(report).read_text(encoding="utf-8"))


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
    runs = [str(run["model"]) for run in loaded.get("models") or []]
    if not runs:
        raise ValueError("the report has no runs to plot")
    if len(runs) > len(_SERIES):
        raise ValueError(
            f"at most {len(_SERIES)} runs fit one chart ({len(runs)} given); "
            "split them across reports"
        )
    means = metric_means(loaded, families=families, k_values=k_values)
    cutoffs = list(next(iter(means.values())))

    figure = Figure(figsize=(2.6 * len(means) + 0.6, 3.6), dpi=150, facecolor=_SURFACE)
    axes = figure.subplots(1, len(means), sharey=True, squeeze=False)[0]
    slot = 0.8 / len(runs)
    for panel, (family, by_k) in zip(axes, means.items()):
        panel.set_facecolor(_SURFACE)
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
                label=run,
                zorder=3,
            )
            # Label only the largest cutoff: selective, not a number per bar.
            last = bars[-1]
            panel.annotate(
                f"{values[-1]:.2f}",
                (last.get_x() + last.get_width() / 2, last.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=_TEXT_SECONDARY,
            )
        panel.set_title(_TITLES.get(family, family), fontsize=10, color=_TEXT_PRIMARY, loc="left")
        panel.set_xticks(range(len(cutoffs)), [f"@{k}" for k in cutoffs])
        panel.set_ylim(0, 1)
        panel.grid(axis="y", color=_GRID, linewidth=0.6, zorder=0)
        panel.tick_params(colors=_TEXT_SECONDARY, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            panel.spines[side].set_visible(False)
        panel.spines["bottom"].set_color(_TEXT_SECONDARY)
    axes[0].set_ylabel("mean over judged questions", fontsize=8, color=_TEXT_SECONDARY)

    generation = (loaded.get("generation") or {}).get("generation_id")
    queries = (loaded.get("judgments") or {}).get("queries")
    subject = runs[0] if len(runs) == 1 else f"{len(runs)} runs"
    heading = title or " · ".join(
        part
        for part in (
            f"Mean retrieval metrics: {subject}",
            f"generation {generation}" if generation else "",
            f"{queries:,} judged questions" if queries else "",
        )
        if part
    )
    figure.suptitle(heading, fontsize=11, color=_TEXT_PRIMARY, x=0.01, ha="left")
    if len(runs) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper right",
            ncol=min(len(runs), 4),
            frameon=False,
            fontsize=8,
            labelcolor=_TEXT_PRIMARY,
        )
    figure.tight_layout(rect=(0, 0, 1, 0.96))

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


def _table(means: Means) -> str:
    runs = list(next(iter(next(iter(means.values())).values())))
    columns = [f"{family}@{k}" for family, by_k in means.items() for k in by_k]
    width = max(len(run) for run in runs) + 2
    lines = [f"{'run':<{width}}" + "".join(f"{c:>12}" for c in columns)]
    for run in runs:
        lines.append(
            f"{run:<{width}}"
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
    args = parser.parse_args()

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
    means = metric_means(report, families=args.metrics, k_values=args.k_values)
    path = plot_metric_means(
        report, args.output, families=args.metrics, k_values=args.k_values, title=args.title
    )
    print(_table(means))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
