from __future__ import annotations

import json

from scripts.v4.questions import read_questions


def _row(questions, answers, document_id="doc-1"):
    return {
        "id": document_id,
        "question_texts": json.dumps(questions, ensure_ascii=False),
        "output": json.dumps(answers, ensure_ascii=False),
    }


def test_emits_one_record_per_question_aligned_to_its_answer() -> None:
    source = read_questions(
        _row(
            ["Primeira pergunta?", "Segunda pergunta?"],
            [{"answer": "a", "citations": ["p-1"]}, {"answer": "b", "citations": ["p-2"]}],
        )
    )
    assert source.valid
    assert [record.question_index for record in source.records] == [0, 1]
    assert source.records[0].text == "Primeira pergunta?"
    assert source.records[1].citations == ("p-2",)
    assert source.records[1].query_id == "doc-1:q1"


def test_rejects_count_mismatch_instead_of_guessing() -> None:
    source = read_questions(_row(["Uma?", "Duas?"], [{"citations": []}]))
    assert not source.valid
    assert source.rejected_reason.startswith("question_answer_count_mismatch")


def test_rejects_unreadable_question_texts() -> None:
    row = {
        "id": "doc",
        "question_texts": "['nao e json'\n 'numpy repr']",
        "output": "[]",
    }
    assert read_questions(row).rejected_reason == "question_texts_unreadable"


def test_accepts_a_real_list_from_datasets() -> None:
    source = read_questions(
        {"id": "doc", "question_texts": ["Pergunta unica?"], "output": [{"citations": []}]}
    )
    assert source.valid
    assert source.records[0].text == "Pergunta unica?"


def test_strips_trailing_protocol_boilerplate_only_when_enough_remains() -> None:
    source = read_questions(
        _row(
            ["Qual e o nome do autor da acao? Caso nao encontre, responda NULL."],
            [{"citations": []}],
        )
    )
    assert source.records[0].text == "Qual e o nome do autor da acao?"


def test_keeps_short_questions_whole_rather_than_mangling_them() -> None:
    source = read_questions(
        _row(["Achou? Caso nao encontre, responda NULL."], [{"citations": []}])
    )
    assert "Caso nao encontre" in source.records[0].text


def test_carries_the_reference_answer_beside_its_citations() -> None:
    source = read_questions(
        _row(
            ["Quem e o autor?"],
            [{"answer": "Joao da Silva", "citations": ["p-3", "p-4"]}],
        )
    )
    record = source.records[0]
    assert record.reference_answer == "Joao da Silva"
    assert record.citations == ("p-3", "p-4")


def test_preserves_a_null_reference_answer_that_still_has_citations() -> None:
    source = read_questions(
        _row(["Quem e o autor?"], [{"answer": None, "citations": ["p-0"]}])
    )
    record = source.records[0]
    assert record.reference_answer is None
    assert record.citations == ("p-0",)


def test_reference_answer_is_none_when_absent() -> None:
    source = read_questions(_row(["Pergunta?"], [{"citations": []}]))
    assert source.records[0].reference_answer is None


def test_a_blank_question_is_dropped_at_its_original_position() -> None:
    source = read_questions(
        _row(
            ["Quem e o autor?", "   ", "Qual o valor?"],
            [
                {"answer": "Joao", "citations": ["p-0"]},
                {"answer": "x", "citations": ["p-1"]},
                {"answer": "10", "citations": ["p-2"]},
            ],
        )
    )

    assert source.valid
    assert [record.query_id for record in source.records] == ["doc-1:q0", "doc-1:q2"]
    assert [(drop.query_id, drop.question_index, drop.reason) for drop in source.dropped] == [
        ("doc-1:q1", 1, "blank_question")
    ]


def test_a_source_of_only_blank_questions_is_rejected_with_its_drops() -> None:
    source = read_questions(_row(["", "  \n "], [{"citations": []}, {"citations": []}]))

    assert source.rejected_reason == "no_usable_questions"
    assert [drop.query_id for drop in source.dropped] == ["doc-1:q0", "doc-1:q1"]
    assert {drop.reason for drop in source.dropped} == {"blank_question"}


def test_rejects_unreadable_output() -> None:
    row = _row(["Quem?"], [])
    row["output"] = "{not json"

    source = read_questions(row)

    assert source.rejected_reason == "output_unreadable"
    assert source.dropped == ()


def test_unequal_arrays_reject_the_source_without_question_drops() -> None:
    source = read_questions(_row(["Uma?", "", "Tres?"], [{"citations": []}]))

    assert source.rejected_reason == "question_answer_count_mismatch:3!=1"
    assert source.records == () and source.dropped == ()
