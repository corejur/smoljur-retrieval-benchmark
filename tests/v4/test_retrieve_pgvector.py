"""Document-scoped retrieval from the stored Qwen pgvector build (T023)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("psycopg")

from scripts.v4.embeddings import (  # noqa: E402
    CHUNK_TEXT_PROFILE,
    QUERY_INSTRUCTION,
    QWEN_MODEL_ID,
    EmbeddingServiceError,
    RemoteEmbedder,
)
from scripts.v4.index import build_index  # noqa: E402
from scripts.v4.metrics import load_run  # noqa: E402
from scripts.v4.pgvector_store import IndexNotReadyError, connect, migrate  # noqa: E402
from scripts.v4.retrieve import (  # noqa: E402
    IndexCompatibilityError,
    RetrievalInputError,
    default_provenance_path,
    rank,
    rank_queries,
    write_run,
)

from .conftest import long_document  # noqa: E402


@pytest.fixture(autouse=True)
def _empty_tables(pg_env):
    with connect(pg_env) as conn:
        migrate(conn)
        conn.execute("TRUNCATE active_model_indexes, indexed_chunks, index_builds")
        conn.commit()


def _embedder(fake, **kwargs):
    return RemoteEmbedder(fake.base_url, sleep=lambda _s: None, **kwargs)


@pytest.fixture
def indexed(published, fake_vllm, pg_env):
    root, generation = published
    build = build_index(root, embedder=_embedder(fake_vllm), dsn_env=pg_env)
    fake_vllm.requests.clear()
    return root, generation, build


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rank(root, fake, pg_env, **kwargs):
    return rank(root / "current" / "beir", embedder=_embedder(fake), dsn_env=pg_env, **kwargs)


# --- what is sent -------------------------------------------------------------


def test_only_instructed_questions_are_embedded(indexed, fake_vllm, pg_env) -> None:
    root, generation, _ = indexed
    questions = [q["text"] for q in _jsonl(generation.path / "beir" / "queries.jsonl")]

    _rank(root, fake_vllm, pg_env)

    assert sorted(fake_vllm.texts) == sorted(QUERY_INSTRUCTION + q for q in questions)
    assert QUERY_INSTRUCTION == (
        "Instruct: Retrieve the passage from the same legal document that answers "
        "the question.\nQuery:"
    )


# --- what comes back ----------------------------------------------------------


def test_each_ranking_holds_only_its_documents_candidates(indexed, fake_vllm, pg_env) -> None:
    root, generation, _ = indexed
    candidates = {
        r["query_id"]: set(r["candidate_ids"])
        for r in _jsonl(generation.path / "beir" / "candidates" / "test.jsonl")
    }

    ranked = _rank(root, fake_vllm, pg_env, top_k=2)

    assert set(ranked.run) == set(candidates)
    for query_id, scores in ranked.run.items():
        assert 0 < len(scores) <= 2
        assert set(scores) <= candidates[query_id]


def test_the_passage_sharing_the_questions_words_ranks_first(indexed, fake_vllm, pg_env) -> None:
    root, generation, _ = indexed
    corpus = {r["_id"]: r["text"] for r in _jsonl(generation.path / "beir" / "corpus.jsonl")}

    ranked = _rank(root, fake_vllm, pg_env)

    for document in ("doc-a", "doc-b"):
        best = max(ranked.run[f"{document}:q1"].items(), key=lambda item: item[1])[0]
        assert "valor causa fixado" in corpus[best]
        best = max(ranked.run[f"{document}:q2"].items(), key=lambda item: item[1])[0]
        assert "recurso apelacao" in corpus[best]


def test_rankings_are_deterministic(indexed, fake_vllm, pg_env) -> None:
    root, _, _ = indexed

    assert _rank(root, fake_vllm, pg_env).run == _rank(root, fake_vllm, pg_env).run


def test_scores_are_finite_cosine_similarities(indexed, fake_vllm, pg_env) -> None:
    root, _, _ = indexed

    for scores in _rank(root, fake_vllm, pg_env).run.values():
        assert all(-1.0 - 1e-6 <= score <= 1.0 + 1e-6 for score in scores.values())


# --- inputs rejected before any question is sent -----------------------------


def test_a_non_positive_top_k_is_rejected(indexed, fake_vllm, pg_env) -> None:
    root, _, _ = indexed
    with pytest.raises(ValueError, match="top_k"):
        _rank(root, fake_vllm, pg_env, top_k=0)
    assert fake_vllm.requests == []


def test_a_split_other_than_test_is_rejected(indexed, fake_vllm, pg_env) -> None:
    root, _, _ = indexed
    with pytest.raises(RetrievalInputError, match="test"):
        _rank(root, fake_vllm, pg_env, split="validation")
    assert fake_vllm.requests == []


def _lower_level(pg_env, build, candidates, fake):
    with connect(pg_env) as conn:
        return rank_queries(
            conn,
            build.index_build_id,
            {query_id: "Quem?" for query_id in candidates},
            candidates,
            _embedder(fake),
            top_k=5,
        )


def test_a_candidate_missing_from_the_index_is_rejected(indexed, fake_vllm, pg_env) -> None:
    _, _, build = indexed

    with pytest.raises(RetrievalInputError, match="doc-a:ghost"):
        _lower_level(pg_env, build, {"doc-a:q0": ("doc-a", ["doc-a:ghost"])}, fake_vllm)
    assert fake_vllm.requests == []


def test_a_candidate_from_another_document_is_rejected(indexed, fake_vllm, pg_env) -> None:
    _, generation, build = indexed
    foreign = next(
        r["_id"]
        for r in _jsonl(generation.path / "beir" / "corpus.jsonl")
        if r["metadata"]["document_id"] == "doc-b"
    )

    with pytest.raises(RetrievalInputError, match=foreign):
        _lower_level(pg_env, build, {"doc-a:q0": ("doc-a", [foreign])}, fake_vllm)
    assert fake_vllm.requests == []


def test_a_query_without_candidates_is_rejected(indexed, fake_vllm, pg_env) -> None:
    _, _, build = indexed

    with pytest.raises(RetrievalInputError, match="doc-a:q0"):
        _lower_level(pg_env, build, {"doc-a:q0": ("doc-a", [])}, fake_vllm)
    assert fake_vllm.requests == []


# --- stale or incompatible index ---------------------------------------------


def test_a_generation_without_a_build_is_not_ready(indexed, fake_vllm, pg_env) -> None:
    from scripts.v4.run import run

    root, _, _ = indexed
    run([long_document("doc-new")], root)  # publishes a new, unindexed generation

    with pytest.raises(IndexNotReadyError, match="index_not_ready"):
        _rank(root, fake_vllm, pg_env)
    assert fake_vllm.requests == []


def test_a_build_for_a_changed_manifest_is_refused(indexed, fake_vllm, pg_env) -> None:
    root, generation, _ = indexed
    manifest_path = generation.path / "meta" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["relevance"] = "edited after the index was built"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(IndexCompatibilityError, match="manifest"):
        _rank(root, fake_vllm, pg_env)
    assert fake_vllm.requests == []


def test_a_build_for_another_model_revision_is_refused(indexed, fake_vllm, pg_env) -> None:
    root, _, _ = indexed

    with pytest.raises(IndexCompatibilityError, match="revision"):
        _rank(root, fake_vllm, pg_env, model_revision="some-other-revision")
    assert fake_vllm.requests == []


def test_a_build_missing_rows_is_refused(indexed, fake_vllm, pg_env) -> None:
    root, _, build = indexed
    with connect(pg_env) as conn:
        # Simulate damage outside the application: bypass the immutability
        # trigger the way only a superuser could.
        conn.execute("ALTER TABLE indexed_chunks DISABLE TRIGGER USER")
        conn.execute(
            "DELETE FROM indexed_chunks WHERE ctid IN (SELECT ctid FROM indexed_chunks "
            "WHERE index_build_id = %s LIMIT 1)",
            (build.index_build_id,),
        )
        conn.execute("ALTER TABLE indexed_chunks ENABLE TRIGGER USER")
        conn.commit()

    with pytest.raises(IndexCompatibilityError, match="rows"):
        _rank(root, fake_vllm, pg_env)
    assert fake_vllm.requests == []


# --- failures leave an existing run untouched --------------------------------


def _cli(monkeypatch, root, fake, run_file, *extra):
    from scripts.v4 import retrieve

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retrieve",
            str(root / "current" / "beir"),
            str(run_file),
            "--model",
            QWEN_MODEL_ID,
            "--base-url",
            fake.base_url,
            "--postgres-dsn-env",
            "V4_TEST_POSTGRES_DSN",
            *extra,
        ],
    )
    retrieve.main()


def test_the_cli_writes_a_run_that_metrics_can_score(indexed, fake_vllm, pg_env, tmp_path, monkeypatch) -> None:
    root, generation, _ = indexed
    run_file = tmp_path / "runs" / "qwen.json"

    _cli(monkeypatch, root, fake_vllm, run_file, "--top-k", "3")

    run = load_run(run_file)
    assert len(run) == generation.summary.queries_retained
    assert list((tmp_path / "runs").iterdir()) == [run_file]


def test_the_cli_writes_trec_too(indexed, fake_vllm, pg_env, tmp_path, monkeypatch) -> None:
    root, _, _ = indexed
    json_file, trec_file = tmp_path / "runs" / "a.json", tmp_path / "runs" / "b.trec"

    _cli(monkeypatch, root, fake_vllm, json_file)
    _cli(monkeypatch, root, fake_vllm, trec_file, "--format", "trec")

    json_run, trec_run = load_run(json_file), load_run(trec_file)
    assert json_run.keys() == trec_run.keys()
    for query_id, scores in json_run.items():
        assert trec_run[query_id] == pytest.approx(scores, abs=1e-5)


@pytest.mark.parametrize(
    "failure",
    [
        lambda inputs: (400, {"error": "bad"}),
        lambda inputs: (200, {"data": [{"index": i, "embedding": [0.0] * 1024} for i in range(len(inputs))]}),
    ],
    ids=["service-error", "invalid-query-vector"],
)
def test_a_failure_leaves_the_existing_run_untouched(
    indexed, fake_vllm, pg_env, tmp_path, monkeypatch, failure
) -> None:
    root, _, _ = indexed
    run_file = tmp_path / "runs" / "qwen.json"
    _cli(monkeypatch, root, fake_vllm, run_file)
    before = run_file.read_bytes()
    provenance = default_provenance_path(run_file)
    provenance_before = provenance.read_bytes()

    fake_vllm.script.append(failure)
    with pytest.raises(EmbeddingServiceError):
        _cli(monkeypatch, root, fake_vllm, run_file, "--top-k", "1")

    assert run_file.read_bytes() == before
    assert provenance.read_bytes() == provenance_before
    assert [p.name for p in (tmp_path / "runs").iterdir()] == ["qwen.json"]


# --- provenance (T030) --------------------------------------------------------


def test_provenance_is_recorded_outside_the_runs_directory(indexed, fake_vllm, pg_env, tmp_path, monkeypatch) -> None:
    root, generation, build = indexed
    run_file = tmp_path / "runs" / "qwen.json"

    _cli(monkeypatch, root, fake_vllm, run_file, "--top-k", "7")

    provenance_file = default_provenance_path(run_file)
    assert run_file.parent not in provenance_file.parents
    provenance = json.loads(provenance_file.read_text())
    assert set(provenance.pop("cost")) == {"wall_seconds", "embedding", "search"}
    assert provenance == {
        "run_file": str(run_file),
        "generation_id": generation.generation_id,
        "generation_manifest_sha256": build.generation_manifest_sha256,
        "index_build_id": build.index_build_id,
        "model_id": QWEN_MODEL_ID,
        "model_revision": "unpinned",
        "query_instruction": QUERY_INSTRUCTION,
        "chunk_text_profile": CHUNK_TEXT_PROFILE,
        "normalization": "l2",
        "score": "cosine_similarity",
        "split": "test",
        "top_k": 7,
        "queries": generation.summary.queries_retained,
    }


def test_provenance_inside_the_runs_directory_is_refused(indexed, fake_vllm, pg_env, tmp_path, monkeypatch) -> None:
    root, _, _ = indexed
    run_file = tmp_path / "runs" / "qwen.json"

    with pytest.raises(SystemExit):
        _cli(monkeypatch, root, fake_vllm, run_file, "--provenance", str(tmp_path / "runs" / "p.json"))
    assert not run_file.exists()


def test_the_cli_takes_no_credential_arguments(monkeypatch) -> None:
    from scripts.v4 import retrieve

    options = {o for action in retrieve.build_parser()._actions for o in action.option_strings}
    assert not {"--api-key", "--password", "--dsn"} & options


def test_write_run_is_atomic(tmp_path, monkeypatch) -> None:
    import os

    destination = tmp_path / "run.json"
    destination.write_text('{"old": {"c": 1.0}}')

    def refuse(*_args):
        raise OSError("disk full")

    monkeypatch.setattr("scripts.v4.retrieve.os.replace", refuse)
    with pytest.raises(OSError):
        write_run({"q": {"c": 0.5}}, destination)

    assert destination.read_text() == '{"old": {"c": 1.0}}'
    assert os.listdir(tmp_path) == ["run.json"]


# --- cost metadata ------------------------------------------------------------


def test_the_run_records_what_it_cost(indexed, fake_vllm, pg_env) -> None:
    from .fake_vllm import token_count

    root, generation, _ = indexed
    questions = [q["text"] for q in _jsonl(generation.path / "beir" / "queries.jsonl")]

    ranked = rank(
        root / "current" / "beir",
        embedder=_embedder(fake_vllm, batch_size=2),
        dsn_env=pg_env,
        top_k=3,
    )

    cost = ranked.provenance["cost"]
    embedding, search = cost["embedding"], cost["search"]
    assert embedding["texts"] == len(questions)
    assert embedding["requests"] == embedding["attempts"] == -(-len(questions) // 2)
    assert embedding["prompt_tokens"] == sum(
        token_count(QUERY_INSTRUCTION + q) for q in questions
    )
    assert embedding["seconds"] >= 0
    assert search["queries"] == len(questions)
    latency = search["latency_ms"]
    assert 0 <= latency["p50"] <= latency["p95"] <= latency["max"]
    assert latency["mean"] >= 0 and search["seconds"] >= 0
    assert cost["wall_seconds"] >= embedding["seconds"]


def test_cost_counts_only_this_runs_requests(indexed, fake_vllm, pg_env) -> None:
    root, _, _ = indexed
    embedder = _embedder(fake_vllm)
    embedder.embed_passages(["uma chamada anterior"])

    ranked = rank(root / "current" / "beir", embedder=embedder, dsn_env=pg_env)

    assert ranked.provenance["cost"]["embedding"]["texts"] == ranked.provenance["queries"]
