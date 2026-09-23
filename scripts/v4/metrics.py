"""Centralized retrieval metrics for the v4 benchmark.

Each model is run separately and writes a run file; this module is the single
place that turns run files into numbers, so every model is scored by identical
code against identical qrels. Comparing models is then a matter of dropping
another run file into the directory, not of re-implementing evaluation.

A run file is JSON mapping query id to {chunk id: score}, or a TREC run
(`query_id Q0 chunk_id rank score tag` per line). Scores are ranked descending;
only the ordering matters.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_K_VALUES",
    "ModelScores",
    "load_qrels",
    "load_run",
    "evaluate_run",
    "evaluate_directory",
]

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10, 20, 100)


@dataclass(frozen=True)
class ModelScores:
    model: str
    queries_scored: int
    queries_missing: int
    metrics: dict[str, float] = field(default_factory=dict)


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Read a BEIR qrels TSV into {query_id: {chunk_id: relevance}}."""
    qrels: dict[str, dict[str, int]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        header = next(handle, "")
        if not header.lower().startswith("query-id"):
            handle.seek(0)
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 tab-separated fields")
            query_id, chunk_id, score = parts
            qrels.setdefault(query_id, {})[chunk_id] = int(score)
    if not qrels:
        raise ValueError(f"no qrels found in {path}")
    return qrels


def load_run(path: str | Path) -> dict[str, dict[str, float]]:
    """Read a retrieval run, accepting either JSON or TREC format."""
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty run file: {source}")
    if text[0] in "{[":
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"{source}: JSON run must be an object keyed by query id")
        if _is_metrics_report(loaded):
            raise ValueError(
                f"{source} is a metrics report, not a run file. Write --output "
                f"outside the runs directory so it is not scored as a model."
            )
        run: dict[str, dict[str, float]] = {}
        for query_id, ranked in loaded.items():
            if not isinstance(ranked, dict):
                raise ValueError(
                    f"{source}: query {query_id!r} must map to "
                    f"{{chunk_id: score}}, found {type(ranked).__name__}"
                )
            run[str(query_id)] = {str(k): float(v) for k, v in ranked.items()}
        return run
    run: dict[str, dict[str, float]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"{source}:{line_number}: expected a TREC run line")
        query_id, _, chunk_id, _, score = parts[:5]
        run.setdefault(query_id, {})[chunk_id] = float(score)
    return run


def _is_metrics_report(loaded: Mapping[str, Any]) -> bool:
    """True for this module's own --output, which lives beside run files."""
    return "models" in loaded and "k_values" in loaded


def _reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for position, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / position
        
    return 0.0


def evaluate_run(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    *,
    model: str,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> ModelScores:
    """Score one run against the qrels with BEIR's evaluator plus MRR."""
    from beir.retrieval.evaluation import EvaluateRetrieval

    ks = sorted(set(k_values))
    scored = {
        query_id: dict(ranked)
        for query_id, ranked in run.items()
        if query_id in qrels
    }
    if not scored:
        raise ValueError(f"run {model!r} shares no query with the qrels")
    missing = [query_id for query_id in qrels if query_id not in run]
    # A query the model returned nothing for still counts, as a zero, so an
    # incomplete run cannot score better by simply answering fewer queries.
    for query_id in missing:
        scored[query_id] = {}

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        dict(qrels), scored, ks
    )
    metrics: dict[str, float] = {}
    for group in (ndcg, _map, recall, precision):
        metrics.update({key: round(float(value), 5) for key, value in group.items()})

    for k in ks:
        total = 0.0
        for query_id, relevant in qrels.items():
            ranked = sorted(
                scored.get(query_id, {}).items(), key=lambda item: -item[1]
            )[:k]
            total += _reciprocal_rank(
                [chunk_id for chunk_id, _ in ranked],
                {c for c, score in relevant.items() if score > 0},
            )
        metrics[f"MRR@{k}"] = round(total / len(qrels), 5)

    return ModelScores(
        model=model,
        queries_scored=len(qrels) - len(missing),
        queries_missing=len(missing),
        metrics=metrics,
    )


def evaluate_directory(
    qrels_path: str | Path,
    runs_directory: str | Path,
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> list[ModelScores]:
    """Score every run file in a directory against one qrels file."""
    qrels = load_qrels(qrels_path)
    runs = sorted(
        path
        for path in Path(runs_directory).iterdir()
        if path.is_file() and path.suffix in {".json", ".jsonl", ".run", ".txt", ".tsv"}
    )
    if not runs:
        raise ValueError(f"no run files in {runs_directory}")
    scored: list[ModelScores] = []
    for path in runs:
        try:
            run = load_run(path)
        except ValueError as error:
            if "metrics report" in str(error):
                continue  # this module's own output, not a competitor
            raise
        scored.append(
            evaluate_run(qrels, run, model=path.stem, k_values=k_values)
        )
    if not scored:
        raise ValueError(f"no run files in {runs_directory}")
    return scored


def comparison_table(scores: Sequence[ModelScores], metrics: Sequence[str]) -> str:
    """Render a fixed-width leaderboard, best value first."""
    if not scores:
        return "no runs scored"
    width = max(len(item.model) for item in scores) + 2
    header = f"{'model':<{width}}" + "".join(f"{name:>12}" for name in metrics)
    lines = [header, "-" * len(header)]
    primary = metrics[0]
    for item in sorted(scores, key=lambda s: -s.metrics.get(primary, 0.0)):
        row = f"{item.model:<{width}}" + "".join(
            f"{item.metrics.get(name, float('nan')):>12.4f}" for name in metrics
        )
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qrels", type=Path, help="beir/qrels/<split>.tsv")
    parser.add_argument("runs", type=Path, help="Directory of per-model run files")
    parser.add_argument("--output", type=Path, help="Write the full metric set as JSON")
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument(
        "--report",
        nargs="+",
        default=["NDCG@10", "Recall@10", "MRR@10", "MAP@10"],
        help="Metrics shown in the printed table",
    )
    args = parser.parse_args()

    scores = evaluate_directory(args.qrels, args.runs, k_values=args.k_values)
    print(comparison_table(scores, args.report))
    for item in scores:
        if item.queries_missing:
            print(
                f"note: {item.model} returned nothing for "
                f"{item.queries_missing} of "
                f"{item.queries_missing + item.queries_scored} queries"
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "qrels": str(args.qrels),
                    "k_values": args.k_values,
                    "models": [
                        {
                            "model": item.model,
                            "queries_scored": item.queries_scored,
                            "queries_missing": item.queries_missing,
                            "metrics": item.metrics,
                        }
                        for item in scores
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
