"""Build query records from the source `question_texts` list.

v4 ships the questions as an explicit ordered list whose position is the index
the `output` array answers. That list is authoritative: the v1-v3 approach of
recovering questions by regex from the prompt mis-parsed prompt rules as
questions and drifted out of step with the answers whenever a number was
skipped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = ["QueryRecord", "SourceQuestions", "read_questions"]

_TRAILING_BOILERPLATE = re.compile(
    r"(?is)[\s.;:]*(?:"
    r"Caso\s+(?:n[aã]o\s+)?(?:encontre|localize)[^.?!]*?responda[^.?!]*?[.?!]?"
    r"|N[aã]o\s+(?:adicione|acrescente|crie)\s+(?:nada|informa[cç][oõ]es)[^.?!]*?[.?!]?"
    r")\s*\Z"
)
_MINIMUM_RETAINED = 20


@dataclass(frozen=True)
class QueryRecord:
    document_id: str
    question_index: int
    query_id: str
    text: str
    citations: tuple[str, ...]
    #: The source answer for this question, kept as written: None where the
    #: source answered null, which 16% of v4 answers do.
    reference_answer: str | None = None

    @property
    def has_answer_text(self) -> bool:
        """True when the source actually answered, not null and not blank."""
        return bool(self.reference_answer and self.reference_answer.strip())


@dataclass(frozen=True)
class SourceQuestions:
    document_id: str
    records: tuple[QueryRecord, ...]
    rejected_reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.rejected_reason is None


def _question_list(value: Any) -> list[str] | None:
    if isinstance(value, (list, tuple)):
        parsed: Any = list(value)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
    else:
        return None
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        return None
    return list(parsed)


def _answer_list(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple)):
        parsed: Any = list(value)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
    else:
        return None
    return parsed if isinstance(parsed, list) else None


def normalize_query_text(text: str) -> str:
    """Collapse whitespace and drop trailing answer-protocol boilerplate.

    Only clearly instructional suffixes are removed, and only when enough of
    the question survives. The v1-v3 cleaner stripped far more aggressively and
    mangled questions whose answer format carried the actual content.
    """
    collapsed = re.sub(r"\s+", " ", text).strip()
    stripped = _TRAILING_BOILERPLATE.sub("", collapsed).strip(" \t.;:")
    if len(stripped) >= _MINIMUM_RETAINED:
        return stripped
    return collapsed


def _reference_answer(answer: Any) -> str | None:
    """Return the source answer for one question, preserving a null answer."""
    if not isinstance(answer, Mapping):
        return None
    value = answer.get("answer")
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _citations(answer: Any) -> tuple[str, ...]:
    if not isinstance(answer, Mapping):
        return ()
    value = answer.get("citations") or []
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def read_questions(
    row: Mapping[str, Any],
    *,
    id_column: str = "id",
    questions_column: str = "question_texts",
    output_column: str = "output",
) -> SourceQuestions:
    """Return one query record per question, or a reason the source is invalid."""
    document_id = str(row.get(id_column, "") or "")
    if not document_id:
        return SourceQuestions("", (), "missing_document_id")

    questions = _question_list(row.get(questions_column))
    if questions is None:
        return SourceQuestions(document_id, (), "question_texts_unreadable")
    answers = _answer_list(row.get(output_column))
    if answers is None:
        return SourceQuestions(document_id, (), "output_unreadable")
    if len(questions) != len(answers):
        return SourceQuestions(
            document_id,
            (),
            f"question_answer_count_mismatch:{len(questions)}!={len(answers)}",
        )

    records: list[QueryRecord] = []
    for index, (question, answer) in enumerate(zip(questions, answers)):
        text = normalize_query_text(question)
        if not text:
            continue
        records.append(
            QueryRecord(
                document_id=document_id,
                question_index=index,
                query_id=f"{document_id}:q{index}",
                text=text,
                citations=_citations(answer),
                reference_answer=_reference_answer(answer),
            )
        )
    if not records:
        return SourceQuestions(document_id, (), "no_usable_questions")
    return SourceQuestions(document_id, tuple(records))
