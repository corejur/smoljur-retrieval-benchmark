from __future__ import annotations

import json
from pathlib import Path

from scripts.v4.run import build_dataset, run

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


def _rows():
    return [
        # q0 is usable; q1 answered but cites nothing.
        _row(
            "doc-a",
            ["Quem e o autor?", "Qual o valor?"],
            [
                {"answer": "Joao da Silva", "citations": ["p-0"]},
                {"answer": "Nao", "citations": []},
            ],
        ),
        # Cites evidence but the source never answered.
        _row("doc-b", ["Pergunta sem resposta?"], [{"answer": None, "citations": ["p-0"]}]),
        _row("doc-c", ["Ha valor da causa?"], [{"answer": "Sim", "citations": ["p-1"]}]),
    ]


def _read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_builds_a_complete_beir_generation(tmp_path: Path) -> None:
    summary = build_dataset(_rows(), tmp_path, split="test")

    # doc-b loses its only query, so the document is dropped with it.
    assert summary.documents == 2
    assert summary.queries == 4
    assert summary.queries_retained == 2
    assert summary.rejected_sources == 1

    corpus = _read(tmp_path / "beir" / "corpus.jsonl")
    queries = _read(tmp_path / "beir" / "queries.jsonl")
    assert len(corpus) == summary.chunks
    assert {item["_id"] for item in queries} == {"doc-a:q0", "doc-c:q0"}

    qrels = (tmp_path / "beir" / "qrels" / "test.tsv").read_text().splitlines()
    assert qrels[0] == "query-id\tcorpus-id\tscore"
    assert len(qrels) - 1 == summary.qrels
    assert all(row.split("\t")[1] in {c["_id"] for c in corpus} for row in qrels[1:])


def test_retained_queries_all_have_answer_text_and_a_citation(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")

    for item in _read(tmp_path / "beir" / "queries.jsonl"):
        assert item["reference_answer"] and item["reference_answer"].strip()
        assert item["citations"]


def test_each_drop_is_audited_with_its_reason(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")

    dropped = {
        item["query_id"]: item["reason"]
        for item in _read(tmp_path / "audits" / "dropped_queries.jsonl")
    }
    assert dropped == {
        "doc-a:q1": "answer_has_no_citations",
        "doc-b:q0": "answer_has_no_text",
    }


def test_a_partly_dropped_document_is_still_kept(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")

    documents = {
        item["document_id"]
        for item in _read(tmp_path / "documents" / "documents.jsonl")
    }
    # doc-a keeps q0 although q1 was dropped; doc-b lost every query.
    assert documents == {"doc-a", "doc-c"}

    dropped = {
        item["query_id"]: item for item in _read(tmp_path / "audits" / "dropped_queries.jsonl")
    }
    assert dropped["doc-a:q1"]["reference_answer"] == "Nao"
    assert dropped["doc-b:q0"]["reference_answer"] is None


def test_query_less_document_is_rejected_with_its_chunks(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")

    rejected = {
        item["document_id"]: item
        for item in _read(tmp_path / "audits" / "rejected_sources.jsonl")
    }
    assert rejected["doc-b"]["reason"] == "no_retained_queries"
    assert rejected["doc-b"]["discarded_chunks"] > 0

    corpus_documents = {
        item["metadata"]["document_id"]
        for item in _read(tmp_path / "beir" / "corpus.jsonl")
    }
    assert "doc-b" not in corpus_documents


def test_every_corpus_document_is_reachable_from_some_query(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")

    corpus_documents = {
        item["metadata"]["document_id"]
        for item in _read(tmp_path / "beir" / "corpus.jsonl")
    }
    queried = {
        item["document_id"]
        for item in _read(tmp_path / "beir" / "candidates" / "test.jsonl")
    }
    assert corpus_documents == queried


def test_query_less_documents_can_be_kept_for_open_corpus(tmp_path: Path) -> None:
    summary = build_dataset(
        _rows(), tmp_path, split="test", keep_documents_without_queries=True
    )

    assert summary.documents == 3
    assert summary.rejected_sources == 0
    corpus_documents = {
        item["metadata"]["document_id"]
        for item in _read(tmp_path / "beir" / "corpus.jsonl")
    }
    assert "doc-b" in corpus_documents


def test_allowing_missing_answers_keeps_a_cited_null_answer(tmp_path: Path) -> None:
    summary = build_dataset(
        _rows(), tmp_path, split="test", require_answer_text=False
    )

    assert summary.queries_retained == 3
    queries = {item["_id"]: item for item in _read(tmp_path / "beir" / "queries.jsonl")}
    assert queries["doc-b:q0"]["reference_answer"] is None


def test_query_rows_carry_reference_answer_and_citations(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")

    queries = {item["_id"]: item for item in _read(tmp_path / "beir" / "queries.jsonl")}
    assert queries["doc-a:q0"]["reference_answer"] == "Joao da Silva"
    assert queries["doc-a:q0"]["citations"] == ["p-0"]


def test_evidence_spans_quote_the_cleaned_document(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")

    texts = {
        item["document_id"]: item["text"]
        for item in _read(tmp_path / "documents" / "documents.jsonl")
    }
    for item in _read(tmp_path / "evidence" / "evidence.jsonl"):
        for span in item["spans"]:
            assert texts[item["document_id"]][span["start"] : span["end"]].strip()


def test_rejected_source_is_audited_and_skipped(tmp_path: Path) -> None:
    rows = [*_rows(), _row("doc-d", ["Uma?", "Duas?"], [{"citations": []}])]
    summary = build_dataset(rows, tmp_path, split="test")

    assert summary.rejected_sources == 2  # doc-b has no query, doc-d is malformed
    reasons = {
        item["document_id"]: item["reason"]
        for item in _read(tmp_path / "audits" / "rejected_sources.jsonl")
    }
    assert reasons["doc-d"].startswith("question_answer_count_mismatch")
    assert reasons["doc-b"] == "no_retained_queries"


def test_manifest_records_the_query_filter(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")
    manifest = json.loads((tmp_path / "meta" / "manifest.json").read_text())
    assert manifest["query_filter"] == "answer_text_and_resolved_citation"


def test_publishes_with_rollback(tmp_path: Path) -> None:
    output = tmp_path / "out"
    run(_rows(), output, split="test")
    assert (output / "beir" / "corpus.jsonl").is_file()
    assert json.loads((output / "meta" / "manifest.json").read_text())["pipeline"] == "redator-v4"


def test_beir_loader_still_reads_the_extended_query_rows(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")

    from beir.datasets.data_loader import GenericDataLoader

    corpus, queries, qrels = GenericDataLoader(
        corpus_file=str(tmp_path / "beir" / "corpus.jsonl"),
        query_file=str(tmp_path / "beir" / "queries.jsonl"),
        qrels_file=str(tmp_path / "beir" / "qrels" / "test.tsv"),
    ).load_custom()

    assert set(queries) == {"doc-a:q0", "doc-c:q0"}
    assert set(qrels) <= set(queries)
    assert all(chunk_id in corpus for judged in qrels.values() for chunk_id in judged)


def test_citation_cap_defaults_to_the_chunk_maximum(tmp_path: Path) -> None:
    from scripts.v4.chunking import V4_CHUNKING

    build_dataset(_rows(), tmp_path, split="test")
    manifest = json.loads((tmp_path / "meta" / "manifest.json").read_text())
    assert manifest["max_citation_words"] == V4_CHUNKING.max_words
    assert manifest["citation_filter"] == "trackable_and_within_max_citation_words"


def test_every_retained_citation_fits_the_cap_and_is_trackable(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")

    texts = {
        item["document_id"]: item["text"]
        for item in _read(tmp_path / "documents" / "documents.jsonl")
    }
    seen = 0
    for item in _read(tmp_path / "evidence" / "evidence.jsonl"):
        for span in item["spans"]:
            seen += 1
            assert span["word_count"] <= 300
            assert span["source_end"] > span["source_start"]
            quoted = texts[item["document_id"]][span["start"] : span["end"]]
            assert len(quoted.split()) == span["word_count"]
    assert seen


def test_emits_document_scoped_candidates(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")

    candidates = _read(tmp_path / "beir" / "candidates" / "test.jsonl")
    corpus = {item["_id"]: item for item in _read(tmp_path / "beir" / "corpus.jsonl")}
    queries = {item["_id"] for item in _read(tmp_path / "beir" / "queries.jsonl")}

    # One record per retained query, exactly what the scoped evaluator requires.
    assert {item["query_id"] for item in candidates} == queries
    for item in candidates:
        assert item["candidate_ids"]
        for chunk_id in item["candidate_ids"]:
            assert corpus[chunk_id]["metadata"]["document_id"] == item["document_id"]


def test_every_judged_chunk_is_among_its_query_candidates(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")

    candidates = {
        item["query_id"]: set(item["candidate_ids"])
        for item in _read(tmp_path / "beir" / "candidates" / "test.jsonl")
    }
    rows = (tmp_path / "beir" / "qrels" / "test.tsv").read_text().splitlines()[1:]
    for row in rows:
        query_id, chunk_id, _ = row.split("\t")
        assert chunk_id in candidates[query_id]


def test_corpus_text_is_plain_before_chunking(tmp_path: Path) -> None:
    from scripts.v4.cleaning import plain_text_violations

    dirty = (
        '<p id="p-0">O autor\x03 e Joao&nbsp;da Silva​ residente aqui</p>'
        '<p id="p-1">Segundo paragrafo &amp; com entidade</p>'
        '<p id="p-2">Terceiro paragrafo com texto suficiente no documento</p>'
    )
    rows = [
        _row("doc-x", ["Quem e o autor?"], [{"answer": "Joao", "citations": ["p-0"]}], dirty)
    ]
    summary = build_dataset(rows, tmp_path, split="test")

    assert summary.plain_text_violations == 0
    for item in _read(tmp_path / "documents" / "documents.jsonl"):
        assert plain_text_violations(item["text"]) == {}
    for item in _read(tmp_path / "beir" / "corpus.jsonl"):
        assert plain_text_violations(item["text"]) == {}


def test_cli_config_keeps_every_v4_default() -> None:
    """The CLI must not drop v4 defaults it does not name on the command line."""
    from dataclasses import fields, replace

    from scripts.v4.chunking import V4_CHUNKING

    built = replace(
        V4_CHUNKING,
        max_words=V4_CHUNKING.max_words,
        target_words=V4_CHUNKING.target_words,
        overlap_words=V4_CHUNKING.overlap_words,
        minimum_words=V4_CHUNKING.minimum_words,
    )
    for field in fields(V4_CHUNKING):
        assert getattr(built, field.name) == getattr(V4_CHUNKING, field.name)
    assert built.merge_small_chunks is True


def test_every_retained_query_has_at_least_one_qrel(tmp_path: Path) -> None:
    """A query with no qrel scores zero for every model, so it must not ship."""
    build_dataset(_rows(), tmp_path, split="test")

    queries = {item["_id"] for item in _read(tmp_path / "beir" / "queries.jsonl")}
    judged = {
        row.split("\t")[0]
        for row in (tmp_path / "beir" / "qrels" / "test.tsv").read_text().splitlines()[1:]
    }
    assert queries == judged


def test_qrels_are_exactly_the_citation_to_chunk_images(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path, split="test")

    images = {
        (item["query_id"], span["chunk_id"])
        for item in _read(tmp_path / "evidence" / "evidence.jsonl")
        for span in item["spans"]
        if span["chunk_id"]
    }
    judged = {
        (row.split("\t")[0], row.split("\t")[1])
        for row in (tmp_path / "beir" / "qrels" / "test.tsv").read_text().splitlines()[1:]
    }
    assert judged == images


def test_chunks_never_exceed_the_token_cap(tmp_path: Path) -> None:
    import tiktoken

    from scripts.v4.chunking import ENCODING_NAME, V4_CHUNKING

    build_dataset(_rows(), tmp_path, split="test")
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    for item in _read(tmp_path / "beir" / "corpus.jsonl"):
        assert len(encoding.encode(item["text"])) <= V4_CHUNKING.max_words


def _many_gold_row():
    html = "".join(
        f'<p id="p-{i}">{" ".join([f"trecho{i}"] * 200)}</p>' for i in range(14)
    )
    return _row(
        "doc-many",
        ["Pergunta ampla?"],
        [{"answer": "Sim", "citations": [f"p-{i}" for i in range(14)]}],
        html,
    )


def test_drops_a_query_needing_more_gold_chunks_than_the_cap(tmp_path: Path) -> None:
    build_dataset([*_rows(), _many_gold_row()], tmp_path, split="test", max_gold_chunks=10)

    dropped = {
        item["query_id"]: item
        for item in _read(tmp_path / "audits" / "dropped_queries.jsonl")
    }
    assert dropped["doc-many:q0"]["reason"] == "too_many_gold_chunks"
    assert dropped["doc-many:q0"]["gold_chunks"] > 10


def test_no_retained_query_exceeds_the_gold_cap(tmp_path: Path) -> None:
    import collections

    build_dataset([*_rows(), _many_gold_row()], tmp_path, split="test", max_gold_chunks=10)

    counts = collections.Counter(
        row.split("\t")[0]
        for row in (tmp_path / "beir" / "qrels" / "test.tsv").read_text().splitlines()[1:]
    )
    assert counts and max(counts.values()) <= 10


def test_the_cap_can_be_disabled(tmp_path: Path) -> None:
    build_dataset([*_rows(), _many_gold_row()], tmp_path, split="test", max_gold_chunks=None)

    queries = {item["_id"] for item in _read(tmp_path / "beir" / "queries.jsonl")}
    assert "doc-many:q0" in queries


# --- T006: dataset-integrity contract regressions ---------------------------


def test_query_ids_use_the_zero_based_answer_index(tmp_path: Path) -> None:
    """A query ID is {document_id}:q{index of its answer in the source}."""
    build_dataset(_rows(), tmp_path)
    queries = _read(tmp_path / "beir" / "queries.jsonl")
    assert {q["_id"] for q in queries} == {"doc-a:q0", "doc-c:q0"}


def test_dropping_a_question_does_not_renumber_the_later_ones(tmp_path: Path) -> None:
    """Position is identity: q1 stays q1 even when q0 is dropped."""
    rows = [
        _row(
            "doc-d",
            ["Pergunta sem citacao?", "Quem e o autor?"],
            [
                {"answer": "Nao", "citations": []},
                {"answer": "Joao da Silva", "citations": ["p-0"]},
            ],
        )
    ]
    build_dataset(rows, tmp_path)
    queries = _read(tmp_path / "beir" / "queries.jsonl")
    assert [q["_id"] for q in queries] == ["doc-d:q1"]


def test_passages_are_ordered_and_contiguous_within_a_document(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path)
    corpus = _read(tmp_path / "beir" / "corpus.jsonl")
    by_document: dict[str, list[int]] = {}
    for record in corpus:
        meta = record["metadata"]
        by_document.setdefault(meta["document_id"], []).append(meta["chunk_index"])
    for document_id, indexes in by_document.items():
        assert indexes == sorted(indexes), f"{document_id} emitted out of order"
        assert indexes == list(range(len(indexes))), f"{document_id} has a gap"


def test_passage_offsets_are_a_half_open_interval(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path)
    for record in _read(tmp_path / "beir" / "corpus.jsonl"):
        meta = record["metadata"]
        assert 0 <= meta["start_offset"] < meta["end_offset"]


def test_qrels_carry_a_positive_score_and_only_published_candidates(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path)
    corpus_ids = {r["_id"] for r in _read(tmp_path / "beir" / "corpus.jsonl")}
    candidates = {
        r["query_id"]: set(r["candidate_ids"])
        for r in _read(tmp_path / "beir" / "candidates" / "test.jsonl")
    }
    lines = (tmp_path / "beir" / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "query-id\tcorpus-id\tscore"
    assert len(lines) > 1
    for line in lines[1:]:
        query_id, chunk_id, score = line.split("\t")
        assert int(score) > 0
        assert chunk_id in corpus_ids, f"{chunk_id} is judged but not published"
        assert chunk_id in candidates[query_id], f"{chunk_id} is judged but not a candidate"


def test_candidate_sets_are_nonempty_and_from_the_source_document(tmp_path: Path) -> None:
    build_dataset(_rows(), tmp_path)
    corpus = {r["_id"]: r["metadata"]["document_id"] for r in _read(tmp_path / "beir" / "corpus.jsonl")}
    for record in _read(tmp_path / "beir" / "candidates" / "test.jsonl"):
        assert record["candidate_ids"], f"{record['query_id']} has no candidates"
        for chunk_id in record["candidate_ids"]:
            assert corpus[chunk_id] == record["document_id"]


# --- T010: the generation is validated before it is called successful -------


def test_the_synthetic_fixture_builds_its_expected_generation(tmp_path: Path) -> None:
    """contracts/fixture-v4.csv: 2 documents, 2 passages, 2 queries, 2 qrels."""
    from scripts.v4.source_contract import file_sha256, iter_source_rows, verify_source_file

    fixture = Path("specs/001-redator-v4-dataset-pipeline/contracts/fixture-v4.csv")
    fingerprint = verify_source_file(
        fixture, expected_sha256=file_sha256(fixture), expected_row_count=2
    )
    output = tmp_path / "out"
    summary = run(iter_source_rows(fixture), output, fingerprint=fingerprint)

    assert summary.documents == 2
    assert summary.chunks == 2
    assert summary.queries_retained == 2
    assert summary.qrels == 2

    manifest = json.loads((output / "meta" / "manifest.json").read_text())
    assert manifest["source_sha256"] == fingerprint.sha256
    assert manifest["source_row_count"] == 2
    assert manifest["summary"]["chunks"] == 2


def test_an_inconsistent_generation_is_not_published(tmp_path: Path, monkeypatch) -> None:
    """Validation failure must roll back rather than publish broken artifacts."""
    import pytest

    from scripts.v4 import run as run_module
    from scripts.v4.generation import GenerationError

    output = tmp_path / "out"
    run(_rows(), output)
    before = (output / "beir" / "corpus.jsonl").read_text(encoding="utf-8")

    def _reject(*_args, **_kwargs):
        raise GenerationError("synthetic contract violation")

    monkeypatch.setattr(run_module, "validate_generation", _reject)
    with pytest.raises(GenerationError):
        run(_rows(), output)

    # The previous generation is untouched.
    assert (output / "beir" / "corpus.jsonl").read_text(encoding="utf-8") == before
