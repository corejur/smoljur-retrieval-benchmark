"""Evaluate COREJUR retrieval with candidates restricted to each query's document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=Path("datasets/corejur-v3"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="msmarco-distilbert-base-v3")
    parser.add_argument("--score-function", choices=("cos_sim", "dot"), default="cos_sim")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10, 100])
    return parser.parse_args()


def load_candidates(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    candidates: dict[str, list[str]] = {}
    documents: dict[str, str] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            record = json.loads(line)
            query_id = str(record["query_id"])
            if query_id in candidates:
                raise ValueError(f"Duplicate query {query_id!r} in {path}:{line_number}")
            candidates[query_id] = [str(chunk_id) for chunk_id in record["candidate_ids"]]
            documents[query_id] = str(record["document_id"])
    return candidates, documents


def retrieve_document_scoped(
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    candidates: dict[str, list[str]],
    model_name: str,
    score_function: str,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    missing_queries = set(queries) - set(candidates)
    extra_queries = set(candidates) - set(queries)
    if missing_queries or extra_queries:
        raise ValueError(
            f"Candidate/query mismatch: {len(missing_queries)} missing and {len(extra_queries)} extra candidate records"
        )

    candidate_ids = sorted({chunk_id for ids in candidates.values() for chunk_id in ids})
    missing_chunks = set(candidate_ids) - set(corpus)
    if missing_chunks:
        example = next(iter(missing_chunks))
        raise ValueError(f"{len(missing_chunks)} candidate chunks are absent from the corpus; example: {example}")

    query_ids = sorted(queries)
    model = SentenceTransformer(model_name)
    query_embeddings = model.encode(
        [queries[query_id] for query_id in query_ids],
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=score_function == "cos_sim",
        show_progress_bar=True,
    )
    corpus_embeddings = model.encode(
        [(corpus[chunk_id].get("title") + " " + corpus[chunk_id]["text"]).strip() for chunk_id in candidate_ids],
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=score_function == "cos_sim",
        show_progress_bar=True,
    )
    candidate_index = {chunk_id: index for index, chunk_id in enumerate(candidate_ids)}

    results: dict[str, dict[str, float]] = {}
    for query_index, query_id in enumerate(query_ids):
        scoped_ids = candidates[query_id]
        scoped_indices = torch.tensor(
            [candidate_index[chunk_id] for chunk_id in scoped_ids],
            device=corpus_embeddings.device,
        )
        scores = torch.mv(corpus_embeddings.index_select(0, scoped_indices), query_embeddings[query_index])
        results[query_id] = {
            chunk_id: float(score) for chunk_id, score in zip(scoped_ids, scores.detach().cpu().tolist())
        }
    return results


def main() -> None:
    args = parse_args()
    corpus, queries, qrels = GenericDataLoader(str(args.data_folder)).load(split=args.split)
    candidate_path = args.data_folder / "candidates" / f"{args.split}.jsonl"
    candidates, documents = load_candidates(candidate_path)

    results = retrieve_document_scoped(
        corpus,
        queries,
        candidates,
        args.model,
        args.score_function,
        args.batch_size,
    )
    for query_id, ranked_chunks in results.items():
        if set(ranked_chunks) != set(candidates[query_id]):
            raise AssertionError(f"Candidate restriction failed for query {query_id!r}")

    ndcg, mean_ap, recall, precision = EvaluateRetrieval.evaluate(qrels, results, args.k_values)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, args.k_values, metric="mrr")
    print(json.dumps({"ndcg": ndcg, "map": mean_ap, "recall": recall, "precision": precision, "mrr": mrr}, indent=2))
    print(f"Evaluated {len(results):,} queries with document-scoped candidates from {len(set(documents.values())):,} documents")


if __name__ == "__main__":
    main()
