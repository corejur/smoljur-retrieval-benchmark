"""Every exclusion is attributable: a source or question identity and a reason.

User Story 2. A maintainer reading `audits/` must be able to account for every
source row and every question that did not reach the published generation,
and the manifest counters must agree with those audit files exactly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import scripts.v4.run as run_module
from scripts.v4.generation import GenerationError, validate_generation
from scripts.v4.run import build_dataset
from scripts.v4.streams import JsonlWriter, TsvWriter

_PARAGRAPHS = (
    '<p id="p-0">O autor da acao e Joao da Silva residente na capital</p>'
    '<p id="p-1">O valor da causa foi fixado pela parte interessada</p>'
    '<p id="p-2">Terceiro paragrafo com texto suficiente para o documento</p>'
)


def _row(document_id, questions, answers, html=_PARAGRAPHS):
    return {
        "id": document_id,
        "question_texts": json.dumps(questions, ensure_ascii=False),
        "output": json.dumps(answers, ensure_ascii=False),
        "data": html,
    }


def _ok(citation="p-0"):
    return {"answer": "Sim", "citations": [citation]}


def _read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _audit(root: Path, name: str):
    return _read(root / "audits" / f"{name}.jsonl")


def _manifest(root: Path):
    return json.loads((root / "meta" / "manifest.json").read_text(encoding="utf-8"))


def _mixed_rows():
    """One row per exclusion kind, plus valid siblings that must survive."""
    return [
        # q1 blank, q2 unmapped citation, q3 no citation, q4 null answer.
        _row(
            "doc-a",
            ["Quem e o autor?", "  ", "Onde?", "Qual o valor?", "Houve?"],
            [
                _ok("p-0"),
                _ok("p-1"),
                _ok("p-99"),
                {"answer": "10", "citations": []},
                {"answer": None, "citations": ["p-1"]},
            ],
        ),
        _row("doc-dup", ["Primeira?"], [_ok("p-1")]),
        _row("doc-dup", ["Segunda?"], [_ok("p-2")]),
        _row("doc-empty", ["Algo?"], [_ok("p-0")], html=""),
        _row("doc-mismatch", ["Uma?", "Duas?"], [_ok()]),
        {**_row("doc-bad-json", ["Uma?"], [_ok()]), "question_texts": "[nao json"},
        _row("doc-blank", ["", " "], [_ok(), _ok()]),
        _row("doc-none", ["Sem resposta?"], [{"answer": "", "citations": ["p-0"]}]),
    ]


# --- source-level exclusions -------------------------------------------------


def test_a_duplicate_document_id_is_rejected_and_the_first_is_kept(tmp_path: Path) -> None:
    build_dataset(_mixed_rows(), tmp_path)

    rejected = [r for r in _audit(tmp_path, "rejected_sources") if r["document_id"] == "doc-dup"]
    assert [(r["source_row"], r["reason"]) for r in rejected] == [(2, "duplicate_document_id")]

    published = [r["document_id"] for r in _read(tmp_path / "documents" / "documents.jsonl")]
    assert published.count("doc-dup") == 1
    queries = {r["_id"]: r["text"] for r in _read(tmp_path / "beir" / "queries.jsonl")}
    assert queries["doc-dup:q0"] == "Primeira?"


def test_each_rejected_source_names_its_row_and_reason(tmp_path: Path) -> None:
    build_dataset(_mixed_rows(), tmp_path)

    reasons = {r["document_id"]: r["reason"] for r in _audit(tmp_path, "rejected_sources")}
    assert reasons["doc-empty"] == "empty_document_after_cleaning"
    assert reasons["doc-mismatch"] == "question_answer_count_mismatch:2!=1"
    assert reasons["doc-bad-json"] == "question_texts_unreadable"
    assert reasons["doc-blank"] == "no_usable_questions"
    assert reasons["doc-none"] == "no_retained_queries"
    for record in _audit(tmp_path, "rejected_sources"):
        assert isinstance(record["source_row"], int) and record["reason"]


def test_a_document_that_yields_no_passages_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_module, "chunk_cleaned_document", lambda *a, **k: [])

    summary = build_dataset([_row("doc-a", ["Quem?"], [_ok()])], tmp_path)

    assert summary.documents == 0
    assert [r["reason"] for r in _audit(tmp_path, "rejected_sources")] == ["no_chunks_produced"]


# --- question-level exclusions -----------------------------------------------


def test_each_dropped_question_names_its_id_and_reason(tmp_path: Path) -> None:
    build_dataset(_mixed_rows(), tmp_path)

    dropped = {r["query_id"]: r["reason"] for r in _audit(tmp_path, "dropped_queries")}
    assert dropped["doc-a:q1"] == "blank_question"
    assert dropped["doc-a:q2"] == "all_citations_unmapped"
    assert dropped["doc-a:q3"] == "answer_has_no_citations"
    assert dropped["doc-a:q4"] == "answer_has_no_text"
    assert dropped["doc-blank:q0"] == dropped["doc-blank:q1"] == "blank_question"
    assert dropped["doc-none:q0"] == "answer_has_no_text"
    for record in _audit(tmp_path, "dropped_queries"):
        assert record["document_id"] and record["reason"]


def test_valid_siblings_survive_and_keep_their_answer_position(tmp_path: Path) -> None:
    build_dataset(_mixed_rows(), tmp_path)

    retained = [r["_id"] for r in _read(tmp_path / "beir" / "queries.jsonl")]
    assert retained == ["doc-a:q0", "doc-dup:q0"]


def test_a_blank_question_does_not_renumber_later_questions(tmp_path: Path) -> None:
    rows = [_row("doc-a", ["", "Quem e o autor?"], [_ok(), _ok("p-0")])]

    build_dataset(rows, tmp_path)

    assert [r["_id"] for r in _read(tmp_path / "beir" / "queries.jsonl")] == ["doc-a:q1"]
    assert [r["query_id"] for r in _audit(tmp_path, "dropped_queries")] == ["doc-a:q0"]


def test_an_unmapped_citation_is_audited_with_its_identity(tmp_path: Path) -> None:
    build_dataset(_mixed_rows(), tmp_path)

    unmapped = [r for r in _audit(tmp_path, "unmapped_citations") if r["query_id"] == "doc-a:q2"]
    assert [(r["document_id"], r["citation_id"]) for r in unmapped] == [("doc-a", "p-99")]
    assert unmapped[0]["reason"]


def test_the_gold_passage_cap_defaults_to_ten(tmp_path: Path) -> None:
    build_dataset([_row("doc-a", ["Quem?"], [_ok()])], tmp_path)

    assert _manifest(tmp_path)["max_gold_chunks"] == 10


def test_a_document_is_excluded_only_when_no_question_remains(tmp_path: Path) -> None:
    rows = [
        _row("doc-keep", ["Boa?", "Ruim?"], [_ok(), {"answer": "x", "citations": []}]),
        _row("doc-drop", ["Ruim?"], [{"answer": "x", "citations": []}]),
    ]

    build_dataset(rows, tmp_path)

    published = {r["document_id"] for r in _read(tmp_path / "documents" / "documents.jsonl")}
    assert published == {"doc-keep"}
    rejected = {r["document_id"]: r["reason"] for r in _audit(tmp_path, "rejected_sources")}
    assert rejected == {"doc-drop": "no_retained_queries"}


# --- counters reconcile with the audits --------------------------------------


def test_exclusion_counters_match_the_audit_files(tmp_path: Path) -> None:
    summary = build_dataset(_mixed_rows(), tmp_path)

    assert summary.rejected_sources == len(_audit(tmp_path, "rejected_sources"))
    assert summary.dropped_queries == len(_audit(tmp_path, "dropped_queries"))
    assert summary.unmapped_citations == len(_audit(tmp_path, "unmapped_citations"))
    assert summary.source_rows == summary.documents + summary.rejected_sources
    assert summary.queries == summary.queries_retained + summary.dropped_queries
    assert _manifest(tmp_path)["summary"]["dropped_queries"] == summary.dropped_queries


def test_the_mixed_build_passes_generation_validation(tmp_path: Path) -> None:
    build_dataset(_mixed_rows(), tmp_path)

    validate_generation(tmp_path, check_directory_identity=False)


@pytest.mark.parametrize(
    "counter",
    ["rejected_sources", "dropped_queries", "unmapped_citations", "plain_text_violations", "source_rows", "queries"],
)
def test_validation_rejects_a_counter_that_disagrees_with_its_audit(
    tmp_path: Path, counter: str
) -> None:
    build_dataset(_mixed_rows(), tmp_path)
    manifest_path = tmp_path / "meta" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary"][counter] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(GenerationError, match=counter):
        validate_generation(tmp_path, check_directory_identity=False)


# --- audit writers keep one valid record per line ----------------------------


def test_an_unserializable_record_leaves_no_partial_line(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with JsonlWriter(path) as writer:
        writer.write({"reason": "first"})
        with pytest.raises(TypeError):
            writer.write({"reason": "bad", "value": object()})
        writer.write({"reason": "last"})

    assert [r["reason"] for r in _read(path)] == ["first", "last"]
    assert writer.count == 2


def test_a_non_finite_number_is_refused_rather_than_written_as_invalid_json(
    tmp_path: Path,
) -> None:
    with JsonlWriter(tmp_path / "audit.jsonl") as writer:
        with pytest.raises(ValueError):
            writer.write({"score": math.nan})


@pytest.mark.parametrize("value", ["tab\there", "line\nbreak", "cr\rhere"])
def test_a_tsv_value_that_would_split_a_row_is_refused(tmp_path: Path, value: str) -> None:
    with TsvWriter(tmp_path / "qrels.tsv", ("query-id", "corpus-id", "score")) as writer:
        with pytest.raises(ValueError):
            writer.write({"query-id": value, "corpus-id": "c", "score": 1})


def test_a_failed_close_removes_the_temporary_file(tmp_path: Path, monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise OSError("disk full")

    writer = JsonlWriter(tmp_path / "audit.jsonl")
    writer.write({"reason": "x"})
    monkeypatch.setattr("scripts.v4.streams.os.replace", refuse)

    with pytest.raises(OSError, match="disk full"):
        writer.__exit__(None, None, None)

    assert writer._handle.closed
    assert list(tmp_path.iterdir()) == []


def test_a_failed_build_leaves_no_audit_file_behind(tmp_path: Path, monkeypatch) -> None:
    def explode(*_args, **_kwargs):
        raise RuntimeError("chunker failed")

    monkeypatch.setattr(run_module, "chunk_cleaned_document", explode)

    with pytest.raises(RuntimeError):
        build_dataset([_row("doc-a", ["Quem?"], [_ok()])], tmp_path)

    assert not any(path.is_file() for path in tmp_path.rglob("*"))


# --- over-long questions -----------------------------------------------------


def _long_question(words: int) -> str:
    return " ".join(["palavra"] * (words - 1)) + " fim?"


def test_a_question_over_the_token_limit_is_dropped_and_audited(tmp_path: Path) -> None:
    from scripts.strategies.chunking.legal_recursive import WordCounter

    rows = [_row("doc-a", ["Quem e o autor?", _long_question(40)], [_ok(), _ok("p-1")])]

    summary = build_dataset(rows, tmp_path, counter=WordCounter(), max_query_tokens=30)

    assert [r["_id"] for r in _read(tmp_path / "beir" / "queries.jsonl")] == ["doc-a:q0"]
    (dropped,) = _audit(tmp_path, "dropped_queries")
    assert dropped["query_id"] == "doc-a:q1"
    assert dropped["reason"] == "question_exceeds_max_tokens"
    assert dropped["query_tokens"] == 40
    assert summary.dropped_queries == 1


def test_a_question_exactly_at_the_limit_is_kept(tmp_path: Path) -> None:
    from scripts.strategies.chunking.legal_recursive import WordCounter

    rows = [_row("doc-a", [_long_question(30)], [_ok()])]

    build_dataset(rows, tmp_path, counter=WordCounter(), max_query_tokens=30)

    assert [r["_id"] for r in _read(tmp_path / "beir" / "queries.jsonl")] == ["doc-a:q0"]


def test_the_default_limit_is_512_o200k_tokens(tmp_path: Path) -> None:
    import tiktoken

    long_text = " ".join(f"termo{i}" for i in range(600)) + "?"
    assert len(tiktoken.get_encoding("o200k_base").encode(long_text)) > 512
    rows = [_row("doc-a", ["Quem e o autor?", long_text], [_ok(), _ok("p-1")])]

    build_dataset(rows, tmp_path)

    assert _manifest(tmp_path)["max_query_tokens"] == 512
    assert {r["query_id"]: r["reason"] for r in _audit(tmp_path, "dropped_queries")} == {
        "doc-a:q1": "question_exceeds_max_tokens"
    }


def test_the_query_token_limit_can_be_disabled(tmp_path: Path) -> None:
    long_text = " ".join(f"termo{i}" for i in range(600)) + "?"
    rows = [_row("doc-a", [long_text], [_ok()])]

    build_dataset(rows, tmp_path, max_query_tokens=None)

    assert len(_read(tmp_path / "beir" / "queries.jsonl")) == 1
    assert _manifest(tmp_path)["max_query_tokens"] is None


def test_a_document_whose_only_question_is_too_long_is_rejected(tmp_path: Path) -> None:
    from scripts.strategies.chunking.legal_recursive import WordCounter

    rows = [_row("doc-a", [_long_question(40)], [_ok()])]

    build_dataset(rows, tmp_path, counter=WordCounter(), max_query_tokens=30)

    assert {r["document_id"]: r["reason"] for r in _audit(tmp_path, "rejected_sources")} == {
        "doc-a": "no_retained_queries"
    }
    validate_generation(tmp_path, check_directory_identity=False)
