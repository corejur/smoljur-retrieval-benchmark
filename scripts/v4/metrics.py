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
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.v4.generation import locate_generation, manifest_sha256, read_manifest

__all__ = [
    "DEFAULT_K_VALUES",
    "ModelScores",
    "load_qrels",
    "load_run",
    "evaluate_run",
    "evaluate_directory",
    "comparison_report",
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
            try:
                relevance = int(score)
            except ValueError:
                raise ValueError(
                    f"{path}:{line_number}: relevance {score!r} is not an integer"
                ) from None
            qrels.setdefault(query_id, {})[chunk_id] = relevance
    if not qrels:
        raise ValueError(f"no qrels found in {path}")
    return qrels


def _finite(score: float, where: str) -> float:
    # NaN sorts arbitrarily and infinities dominate every ranking: either
    # would decide a comparison by accident rather than by the model.
    if not math.isfinite(score):
        raise ValueError(f"{where}: score {score!r} is not finite")
    return score


def _json_score(value: Any, where: str) -> float:
    # bool is an int subclass, and a string like "1.5" would float() fine:
    # both are malformed run files, not scores.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: score must be a number, found {value!r}")
    return _finite(float(value), where)


def load_run(path: str | Path) -> dict[str, dict[str, float]]:
    """Read a retrieval run, accepting either JSON or TREC format.

    Every score must be a finite number; a malformed, empty, or duplicated
    entry is rejected with the file (and line, for TREC) that holds it.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty run file: {source}")
    if text[0] in "{[":
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}: not valid JSON ({error})") from None
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
            run[str(query_id)] = {
                str(chunk_id): _json_score(
                    score, f"{source}: query {query_id!r} chunk {chunk_id!r}"
                )
                for chunk_id, score in ranked.items()
            }
        if not run:
            raise ValueError(f"empty run file: {source} ranks no query")
        return run
    run: dict[str, dict[str, float]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"{source}:{line_number}: expected a TREC run line")
        query_id, _, chunk_id, _, score = parts[:5]
        where = f"{source}:{line_number}"
        try:
            value = float(score)
        except ValueError:
            raise ValueError(f"{where}: score {score!r} is not a number") from None
        ranked = run.setdefault(query_id, {})
        if chunk_id in ranked:
            raise ValueError(
                f"{where}: duplicate row for query {query_id!r} chunk {chunk_id!r}"
            )
        ranked[chunk_id] = _finite(value, where)
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
    """Score one run against the qrels with BEIR's evaluator plus MRR and Pass@k.

    Pass@k is the share of judged questions with at least one relevant
    passage in the top k — a hit rate. Missing questions count as misses.
    """
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

    # Rank once per query, breaking score ties as trec_eval does (higher ID
    # first), so MRR and Pass@k agree with NDCG/MAP/Recall/P on every tie.
    rankings = {
        query_id: [
            chunk_id
            for chunk_id, _ in sorted(
                scored.get(query_id, {}).items(), key=lambda item: (item[1], item[0]), reverse=True
            )
        ]
        for query_id in qrels
    }
    relevant_sets = {
        query_id: {c for c, score in judged.items() if score > 0}
        for query_id, judged in qrels.items()
    }
    for k in ks:
        reciprocal = hits = 0.0
        for query_id, ranked in rankings.items():
            top = ranked[:k]
            relevant = relevant_sets[query_id]
            reciprocal += _reciprocal_rank(top, relevant)
            # Pass@k: did any relevant passage make the top k?
            hits += any(chunk_id in relevant for chunk_id in top)
        metrics[f"MRR@{k}"] = round(reciprocal / len(qrels), 5)
        metrics[f"Pass@{k}"] = round(hits / len(qrels), 5)

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


def comparison_report(
    qrels_path: str | Path,
    scores: Sequence[ModelScores],
    *,
    k_values: Iterable[int],
) -> dict[str, Any]:
    """The comparison as a record: what was judged, against what, and how.

    `generation` names the published generation the qrels came from, pinned
    past `current`, or is None for qrels supplied from outside a generation.
    `judgments` fingerprints the exact qrels bytes every run was scored
    against, so two reports can be checked for comparability.
    """
    located = locate_generation(qrels_path)
    generation: dict[str, Any] | None = None
    qrels_file = Path(qrels_path)
    if located is not None:
        root, qrels_file = located
        manifest = read_manifest(root)
        generation = {
            "generation_id": manifest.generation_id,
            "manifest_sha256": manifest_sha256(root),
            "split": manifest.split,
        }
    qrels = load_qrels(qrels_file)
    return {
        "k_values": sorted(set(k_values)),
        "generation": generation,
        "judgments": {
            "qrels": str(qrels_file),
            "qrels_sha256": hashlib.sha256(qrels_file.read_bytes()).hexdigest(),
            "queries": len(qrels),
            "judgments": sum(len(judged) for judged in qrels.values()),
        },
        "models": [
            {
                "model": item.model,
                "queries_scored": item.queries_scored,
                "queries_missing": item.queries_missing,
                "metrics": item.metrics,
            }
            for item in scores
        ],
    }


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


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

    runs = args.runs.resolve()
    if args.output and runs in args.output.resolve().parents:
        parser.error(
            f"--output {args.output} is inside the runs directory {args.runs}; "
            "write the report elsewhere so it is not read as a competing run"
        )

    # Resolve `current` once: every run is scored against the same qrels
    # even if a new generation is published while this is running.
    located = locate_generation(args.qrels)
    qrels = located[1] if located else args.qrels
    scores = evaluate_directory(qrels, args.runs, k_values=args.k_values)
    print(comparison_table(scores, args.report))
    for item in scores:
        if item.queries_missing:
            print(
                f"note: {item.model} returned nothing for "
                f"{item.queries_missing} of "
                f"{item.queries_missing + item.queries_scored} queries"
            )
    if args.output:
        report = comparison_report(qrels, scores, k_values=args.k_values)
        _write_atomically(
            args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
