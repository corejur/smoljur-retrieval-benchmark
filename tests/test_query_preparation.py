from __future__ import annotations

import json

import pytest

from scripts.query_preparation import prepare_query_rows


def test_prepares_identity_text_and_citations_in_one_pass() -> None:
    prompt = (
        "Analise o documento e responda as perguntas abaixo:\n"
        "1. Primeira pergunta?\n"
        "2. Segunda pergunta?"
    )
    output = json.dumps(
        [
            {"citations": ["p-1"]},
            {"citations": ["p-2", "p-3"]},
        ]
    )

    prepared = prepare_query_rows(
        [{"id": "doc-1", "query": prompt, "output": output}]
    )

    assert len(prepared.rows) == 2
    assert prepared.rows[0] == {
        "source_row": 0,
        "source_id": "doc-1",
        "source_query_number": 1,
        "source_answer_index": 0,
        "query": "Primeira pergunta?",
        "citations": '["p-1"]',
    }
    assert prepared.rows[1]["citations"] == '["p-2", "p-3"]'


def test_rejects_invalid_source_citations() -> None:
    output = json.dumps([{"citations": "p-1"}])
    with pytest.raises(ValueError, match="list of strings"):
        prepare_query_rows(
            [{"id": "doc", "query": "1. Pergunta?", "output": output}]
        )


def test_preserves_unowned_existing_columns() -> None:
    prepared = prepare_query_rows(
        [{"id": "doc", "query": "1. Pergunta?", "output": "[]"}],
        [
            {
                "source_row": "0",
                "source_query_number": "1",
                "review_status": "approved",
            }
        ],
    )

    assert prepared.rows[0]["review_status"] == "approved"
    assert prepared.fieldnames[-2:] == ["review_status", "citations"]
