"""Convert the COREJUR question/chunk pairs on Hugging Face to BEIR files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

DEFAULT_DATASET = "CEIA-COREJUR/question_chunk_pair_train_from_v3"
QUESTION_COLUMNS = ("question", "query", "question_text", "query_text")
CHUNK_COLUMNS = ("positive_chunks", "chunk", "chunk_text", "passage", "passage_text", "text")
NEGATIVE_CHUNK_COLUMNS = ("hard_negative_chunks", "negative_chunks")
QUERY_ID_COLUMNS = ("question_idx", "question_id", "query_id", "qid")
CHUNK_ID_COLUMNS = ("chunk_id", "passage_id", "corpus_id")
DOCUMENT_ID_COLUMNS = ("document_id", "doc_id", "source_id")
SPLIT_NAMES = {"validation": "dev", "valid": "dev", "val": "dev"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset repository")
    parser.add_argument("--output", type=Path, default=Path("datasets/corejur-v3"))
    parser.add_argument("--question-column", help="Question text column (auto-detected by default)")
    parser.add_argument("--chunk-column", help="Relevant chunk text column (auto-detected by default)")
    parser.add_argument("--negative-chunk-column", help="Optional non-relevant chunks to include in the corpus")
    parser.add_argument("--query-id-column", help="Stable query ID column")
    parser.add_argument("--chunk-id-column", help="Stable chunk ID column")
    parser.add_argument("--document-id-column", help="Document ID used when generating chunk IDs")
    parser.add_argument("--title-column", help="Optional corpus title column")
    parser.add_argument("--relevance-column", help="Optional integer relevance column; rows scoring <= 0 are omitted")
    parser.add_argument("--revision", help="Optional Hugging Face revision/tag/commit")
    return parser.parse_args()


def detect_column(
    columns: Iterable[str], requested: str | None, candidates: tuple[str, ...], kind: str, required: bool = True
) -> str | None:
    available = set(columns)
    if requested:
        if requested not in available:
            raise ValueError(f"{kind} column {requested!r} not found. Available columns: {sorted(available)}")
        return requested
    for candidate in candidates:
        if candidate in available:
            return candidate
    if required:
        raise ValueError(
            f"Could not detect the {kind} column. Available columns: {sorted(available)}. "
            f"Pass --{kind.replace('_', '-')}-column explicitly."
        )
    return None


def stable_id(prefix: str, *values: str) -> str:
    value = "\x1f".join(values)
    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def text_value(row: Mapping[str, Any], column: str) -> str:
    value = row.get(column)
    if value is None:
        return ""
    return str(value).strip()


def chunk_values(value: Any) -> Iterable[str]:
    """Yield chunk text from either a scalar string or a list of strings."""
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, Iterable):
        for item in value:
            text = str(item).strip()
            if text:
                yield text
        return
    text = str(value).strip()
    if text:
        yield text


def store_consistently(items: dict[str, Any], item_id: str, value: Any, kind: str) -> None:
    previous = items.get(item_id)
    if previous is not None and previous != value:
        raise ValueError(f"The same {kind} ID {item_id!r} refers to different content")
    items[item_id] = value


def convert_dataset(
    dataset: Mapping[str, Iterable[Mapping[str, Any]]],
    output: Path,
    *,
    question_column: str | None = None,
    chunk_column: str | None = None,
    negative_chunk_column: str | None = None,
    query_id_column: str | None = None,
    chunk_id_column: str | None = None,
    document_id_column: str | None = None,
    title_column: str | None = None,
    relevance_column: str | None = None,
) -> tuple[int, int, dict[str, int]]:
    if not dataset:
        raise ValueError("The Hugging Face dataset contains no splits")

    first_split = next(iter(dataset.values()))
    columns = getattr(first_split, "column_names", None)
    if columns is None:
        first_row = next(iter(first_split), None)
        if first_row is None:
            raise ValueError("The first dataset split is empty")
        columns = list(first_row)

    question_column = detect_column(columns, question_column, QUESTION_COLUMNS, "question")
    chunk_column = detect_column(columns, chunk_column, CHUNK_COLUMNS, "chunk")
    negative_chunk_column = detect_column(
        columns, negative_chunk_column, NEGATIVE_CHUNK_COLUMNS, "negative_chunk", required=False)
    query_id_column = detect_column(columns, query_id_column, QUERY_ID_COLUMNS, "query_id", required=False)
    chunk_id_column = detect_column(columns, chunk_id_column, CHUNK_ID_COLUMNS, "chunk_id", required=False)
    document_id_column = detect_column(
        columns, document_id_column, DOCUMENT_ID_COLUMNS, "document_id", required=False
    )
    if title_column:
        detect_column(columns, title_column, (), "title")
    if relevance_column:
        detect_column(columns, relevance_column, (), "relevance")

    corpus: dict[str, dict[str, str]] = {}
    queries: dict[str, str] = {}
    query_splits: dict[str, str] = {}
    query_documents: dict[str, str] = {}
    document_chunks: dict[str, set[str]] = {}
    document_splits: dict[str, str] = {}
    split_qrels: dict[str, set[tuple[str, str, int]]] = {}

    for source_split, rows in dataset.items():
        target_split = SPLIT_NAMES.get(source_split, source_split)
        qrels = split_qrels.setdefault(target_split, set())
        for row in rows:
            question = text_value(row, question_column)
            chunks = list(chunk_values(row.get(chunk_column)))
            negative_chunks = list(chunk_values(row.get(negative_chunk_column))) if negative_chunk_column else []
            if not question or not chunks:
                continue

            relevance = int(row[relevance_column]) if relevance_column else 1
            if relevance <= 0:
                continue

            document_id = text_value(row, document_id_column) if document_id_column else ""
            supplied_query_id = text_value(row, query_id_column) if query_id_column else ""
            if supplied_query_id and document_id:
                query_id = f"{document_id}:{supplied_query_id}"
            else:
                query_id = supplied_query_id or stable_id("q", question)

            title = text_value(row, title_column) if title_column else ""

            previous_document_split = document_splits.get(document_id)
            if previous_document_split is not None and previous_document_split != target_split:
                raise ValueError(
                    f"Document {document_id!r} occurs in both {previous_document_split!r} and {target_split!r}; "
                    "document-scoped candidates would leak across splits"
                )
            document_splits[document_id] = target_split

            previous_split = query_splits.get(query_id)
            if previous_split is not None and previous_split != target_split:
                raise ValueError(
                    f"Query {query_id!r} occurs in both {previous_split!r} and {target_split!r}; "
                    "fix the source split leakage before benchmarking"
                )
            query_splits[query_id] = target_split
            store_consistently(query_documents, query_id, document_id, "query document")
            store_consistently(queries, query_id, question, "query")
            candidate_ids = document_chunks.setdefault(document_id, set())
            supplied_chunk_id = text_value(row, chunk_id_column) if chunk_id_column else ""
            for negative_chunk in negative_chunks:
                negative_id = stable_id("c", document_id, negative_chunk)
                value = {"_id": negative_id, "title": title, "text": negative_chunk}
                store_consistently(corpus, negative_id, value, "chunk")
                candidate_ids.add(negative_id)
            if supplied_chunk_id and len(chunks) != 1:
                raise ValueError(
                    f"Row-level chunk ID {supplied_chunk_id!r} cannot identify {len(chunks)} chunks; "
                    "omit --chunk-id-column when the chunk column contains a list"
                )
            for chunk in chunks:
                chunk_id = supplied_chunk_id or stable_id("c", document_id, chunk)
                store_consistently(corpus, chunk_id, {"_id": chunk_id, "title": title, "text": chunk}, "chunk")
                candidate_ids.add(chunk_id)
                qrels.add((query_id, chunk_id, relevance))

    output.mkdir(parents=True, exist_ok=True)
    (output / "qrels").mkdir(exist_ok=True)
    (output / "candidates").mkdir(exist_ok=True)

    with (output / "corpus.jsonl").open("w", encoding="utf-8") as file:
        for chunk_id in sorted(corpus):
            file.write(json.dumps(corpus[chunk_id], ensure_ascii=False) + "\n")

    with (output / "queries.jsonl").open("w", encoding="utf-8") as file:
        for query_id in sorted(queries):
            file.write(json.dumps({"_id": query_id, "text": queries[query_id]}, ensure_ascii=False) + "\n")

    qrel_counts = {}
    for split, qrels in sorted(split_qrels.items()):
        relevant_by_query: dict[str, set[str]] = {}
        for query_id, chunk_id, _ in qrels:
            relevant_by_query.setdefault(query_id, set()).add(chunk_id)
        with (output / "qrels" / f"{split}.tsv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, delimiter="\t", lineterminator="\n")
            writer.writerow(["query-id", "corpus-id", "score"])
            writer.writerows(sorted(qrels))

        with (output / "candidates" / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            for query_id in sorted(relevant_by_query):
                document_id = query_documents[query_id]
                candidate_ids = sorted(document_chunks[document_id])
                relevant_ids = relevant_by_query[query_id]
                if not relevant_ids.issubset(candidate_ids):
                    raise ValueError(f"Candidates for query {query_id!r} omit a relevant chunk")
                record = {
                    "query_id": query_id,
                    "document_id": document_id,
                    "candidate_ids": candidate_ids,
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        qrel_counts[split] = len(qrels)

    return len(corpus), len(queries), qrel_counts


def main() -> None:
    args = parse_args()
    loaded = load_dataset(args.dataset, revision=args.revision)
    if not isinstance(loaded, DatasetDict):
        loaded = DatasetDict({"train": loaded})

    corpus_count, query_count, qrel_counts = convert_dataset(
        loaded,
        args.output,
        question_column=args.question_column,
        chunk_column=args.chunk_column,
        negative_chunk_column=args.negative_chunk_column,
        query_id_column=args.query_id_column,
        chunk_id_column=args.chunk_id_column,
        document_id_column=args.document_id_column,
        title_column=args.title_column,
        relevance_column=args.relevance_column,
    )
    print(f"Wrote {corpus_count:,} chunks and {query_count:,} queries to {args.output}")
    for split, count in qrel_counts.items():
        print(f"  {split}: {count:,} relevant query/chunk pairs")


if __name__ == "__main__":
    main()
