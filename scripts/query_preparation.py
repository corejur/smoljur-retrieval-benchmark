"""Derive retrieval query rows and source citations in one in-memory pass."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_QUESTIONS_MARKER = re.compile(
    r"Analise\s+o\s+documento\s+e\s+responda\s+as\s+perguntas\s+abaixo[^\n:]*:?\s*(?:\n|$)",
    flags=re.IGNORECASE,
)
_NUMBERED_QUESTION = re.compile(
    r"(?ms)^\s*(\d+)\s*[.)-]\s*(.*?)(?=^\s*\d+\s*[.)-]\s|\Z)"
)
_OUTPUT_CLAUSE = re.compile(
    r"(?is)\s*(?:"
    r"Responda\s+(?:apenas|somente|diretamente)[^.?!]*(?:[.?!]|$)"
    r"|N[aã]o\s+(?:adicione|acrescente|crie)[^.?!]*(?:[.?!]|$)"
    r"|Caso\s+(?:n[aã]o\s+)?(?:encontre|localize)[^.?!]*,?\s*responda[^.?!]*(?:[.?!]|$)"
    r"|Em\s+caso\s+(?:positivo|negativo)[^.?!]*,?\s*responda[^.?!]*(?:[.?!]|$)"
    r"|Se\s+sim,?\s*responda[^.?!]*(?:[.?!]|$)"
    r"|Op[cç][oõ]es?\s*:[^\n]*(?:\n|$)"
    r")"
)
_FORMAT_BLOCK = re.compile(
    r"(?is)\n\s*(?:Responda apenas[^\n]*|Siglas?[^\n]*|Caso encontre[^\n]*)"
    r"(?:\n(?!\s*\d+\s*[.)-]\s).*)*\Z"
)
_BASE_FIELDS = (
    "source_row",
    "source_id",
    "source_query_number",
    "source_answer_index",
    "query",
)


@dataclass(frozen=True)
class PreparedQueries:
    rows: list[dict[str, str | int]]
    fieldnames: list[str]

@dataclass(frozen=True)
class FullyAnsweredSources:
    rows: list[Mapping[str, str]]
    source_indexes: tuple[int, ...]
    removed_document_ids: tuple[str, ...]


def _natural_language_queries(
    prompt: str, expected_count: int | None
) -> list[tuple[int, str]]:
    marker = _QUESTIONS_MARKER.search(prompt)
    question_section = prompt[marker.end() :] if marker else prompt
    candidates = list(_NUMBERED_QUESTION.finditer(question_section))
    if not candidates:
        return []

    top_level: list[re.Match[str]] = []
    expected_number: int | None = None
    for candidate in candidates:
        number = int(candidate.group(1))
        if expected_number is None or number == expected_number:
            top_level.append(candidate)
            expected_number = number + 1
        if expected_count is not None and len(top_level) >= expected_count:
            break

    queries: list[tuple[int, str]] = []
    for index, candidate in enumerate(top_level):
        end = (
            top_level[index + 1].start()
            if index + 1 < len(top_level)
            else len(question_section)
        )
        text = question_section[candidate.start(2) : end].strip()
        text = _FORMAT_BLOCK.sub("", text)
        text = _OUTPUT_CLAUSE.sub(" ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*\n\s*", " ", text).strip(" \t\n.;:")
        if text:
            queries.append((int(candidate.group(1)), text))
    return queries


def _answers(
    row: Mapping[str, str], source_row: int, output_column: str
) -> list[object]:
    value = row.get(output_column, "").strip()
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{output_column!r} in source row {source_row} is not a list")
    return parsed


def filter_fully_answered_sources(
    source_rows: Sequence[Mapping[str, str]],
    *,
    query_column: str = "query",
    id_column: str = "id",
    output_column: str = "output",
) -> FullyAnsweredSources:
    """Keep documents only when answer count equals uncapped query count."""
    kept: list[Mapping[str, str]] = []
    indexes: list[int] = []
    removed_ids: list[str] = []
    for source_index, source in enumerate(source_rows):
        if query_column not in source:
            raise ValueError(
                f"source row {source_index} is missing required column "
                f"{query_column!r}"
            )
        query_count = len(_natural_language_queries(source[query_column], None))
        answer_count = len(_answers(source, source_index, output_column))
        if query_count == answer_count:
            kept.append(source)
            indexes.append(source_index)
        else:
            removed_ids.append(source.get(id_column, ""))
    return FullyAnsweredSources(
        rows=kept,
        source_indexes=tuple(indexes),
        removed_document_ids=tuple(removed_ids),
    )



def _citations(
    answers: Sequence[object], answer_index: int, source_row: int
) -> list[str]:
    if answer_index >= len(answers):
        return []
    answer = answers[answer_index]
    if not isinstance(answer, dict):
        raise ValueError(f"answer {answer_index} in source row {source_row} is not an object")
    citations = answer.get("citations", [])
    if not isinstance(citations, list) or not all(
        isinstance(citation, str) for citation in citations
    ):
        raise ValueError(
            f"citations for answer {answer_index} in source row "
            f"{source_row} must be a list of strings"
        )
    return citations


def prepare_query_rows(
    source_rows: Sequence[Mapping[str, str]],
    existing_rows: Sequence[Mapping[str, str]] = (),
    *,
    query_column: str = "query",
    id_column: str = "id",
    output_column: str = "output",
) -> PreparedQueries:
    """Prepare query identity, normalized text, and citations in one pass."""
    existing_by_key: dict[tuple[str, str], dict[str, str]] = {}
    existing_fields: list[str] = []
    for row in existing_rows:
        missing = {"source_row", "source_query_number"}.difference(row)
        if missing:
            raise ValueError(
                "existing query row is missing key columns: "
                + ", ".join(sorted(missing))
            )
        if not existing_fields:
            existing_fields = list(row)
        key = (str(row["source_row"]), str(row["source_query_number"]))
        existing_by_key[key] = dict(row)

    fieldnames = [
        *_BASE_FIELDS,
        *[field for field in existing_fields if field not in _BASE_FIELDS and field != "citations"],
        "citations",
    ]
    rows: list[dict[str, str | int]] = []
    for source_index, source in enumerate(source_rows):
        if query_column not in source:
            raise ValueError(
                f"source row {source_index} is missing required column {query_column!r}"
            )
        answers = _answers(source, source_index, output_column)
        expected_count = len(answers) or None
        for query_number, query in _natural_language_queries(
            source[query_column], expected_count
        ):
            answer_index = query_number - 1
            key = (str(source_index), str(query_number))
            row: dict[str, str | int] = dict(existing_by_key.get(key, {}))
            row.update(
                {
                    "source_row": source_index,
                    "source_id": source.get(id_column, ""),
                    "source_query_number": query_number,
                    "source_answer_index": answer_index,
                    "query": query,
                    "citations": json.dumps(
                        _citations(answers, answer_index, source_index),
                        ensure_ascii=False,
                    ),
                }
            )
            rows.append(row)
    return PreparedQueries(rows=rows, fieldnames=fieldnames)

