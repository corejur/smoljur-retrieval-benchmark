"""Building the Qwen pgvector index for one published generation (T022)."""

from __future__ import annotations

import collections
import json

import numpy as np
import pytest

pytest.importorskip("psycopg")

from scripts.v4.embeddings import (  # noqa: E402
    CHUNK_TEXT_PROFILE,
    QUERY_INSTRUCTION,
    QWEN_MODEL_ID,
    EmbeddingServiceError,
    RemoteEmbedder,
)
from scripts.v4.generation import manifest_sha256, resolve_current  # noqa: E402
from scripts.v4.index import build_index  # noqa: E402
from scripts.v4.pgvector_store import active_build, connect, migrate  # noqa: E402

from .conftest import long_document  # noqa: E402
from .fake_vllm import embed_text  # noqa: E402


@pytest.fixture(autouse=True)
def _empty_tables(pg_env):
    with connect(pg_env) as conn:
        migrate(conn)
        conn.execute("TRUNCATE active_model_indexes, indexed_chunks, index_builds")
        conn.commit()


def _embedder(fake, **kwargs):
    return RemoteEmbedder(fake.base_url, sleep=lambda _s: None, **kwargs)


def _corpus(generation):
    lines = (generation / "beir" / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _stored(pg_env, build_id):
    with connect(pg_env) as conn:
        rows = conn.execute(
            "SELECT chunk_id, document_id, chunk_index, embedding FROM indexed_chunks "
            "WHERE index_build_id = %s",
            (build_id,),
        ).fetchall()
    return {row[0]: row[1:] for row in rows}


def test_every_published_chunk_is_embedded_exactly_once(published, fake_vllm, pg_env) -> None:
    root, generation = published
    corpus = _corpus(generation.path)

    result = build_index(root, embedder=_embedder(fake_vllm, batch_size=3), dsn_env=pg_env)

    assert result.rows == generation.summary.chunks == len(corpus)
    assert collections.Counter(fake_vllm.texts) == collections.Counter(r["text"] for r in corpus)
    assert all(len(request["input"]) <= 3 for request in fake_vllm.requests)
    stored = _stored(pg_env, result.index_build_id)
    assert set(stored) == {r["_id"] for r in corpus}
    for record in corpus:
        document_id, chunk_index, embedding = stored[record["_id"]]
        assert (document_id, chunk_index) == (
            record["metadata"]["document_id"],
            record["metadata"]["chunk_index"],
        )
        expected = np.asarray(embed_text(record["text"]), dtype=np.float32)
        assert np.allclose(embedding.to_numpy(), expected / np.linalg.norm(expected), atol=1e-6)


def test_chunks_are_sent_in_document_then_position_order(published, fake_vllm, pg_env) -> None:
    root, generation = published
    corpus = _corpus(generation.path)
    ordered = sorted(
        corpus,
        key=lambda r: (r["metadata"]["document_id"], r["metadata"]["chunk_index"], r["_id"]),
    )

    build_index(root, embedder=_embedder(fake_vllm, batch_size=2), dsn_env=pg_env)

    assert fake_vllm.texts == [r["text"] for r in ordered]


def test_the_build_records_its_generation_and_text_provenance(published, fake_vllm, pg_env) -> None:
    root, generation = published
    manifest = json.loads((generation.path / "meta" / "manifest.json").read_text())

    result = build_index(
        root, embedder=_embedder(fake_vllm), dsn_env=pg_env, model_revision="rev-abc"
    )

    with connect(pg_env) as conn:
        build = active_build(conn, generation.generation_id, QWEN_MODEL_ID)
    assert build.index_build_id == result.index_build_id
    provenance = build.provenance
    assert provenance.generation_id == generation.generation_id == resolve_current(root).name
    assert provenance.generation_manifest_sha256 == manifest_sha256(generation.path)
    assert provenance.source_sha256 == manifest["source_sha256"]
    assert provenance.model_id == QWEN_MODEL_ID
    assert provenance.model_revision == "rev-abc"
    assert provenance.query_instruction == QUERY_INSTRUCTION
    assert provenance.chunk_text_profile == CHUNK_TEXT_PROFILE
    assert provenance.chunk_count == build.row_count == generation.summary.chunks


def test_only_the_qwen_model_can_be_indexed(published, pg_env) -> None:
    root, _ = published
    with pytest.raises(ValueError, match="Qwen"):
        build_index(
            root,
            embedder=RemoteEmbedder("http://127.0.0.1:1/v1", model="intfloat/e5-large"),
            dsn_env=pg_env,
        )


def test_a_rebuild_supersedes_the_previous_active_build(published, fake_vllm, pg_env) -> None:
    root, generation = published

    first = build_index(root, embedder=_embedder(fake_vllm), dsn_env=pg_env)
    second = build_index(root, embedder=_embedder(fake_vllm), dsn_env=pg_env)

    assert second.previous_index_build_id == first.index_build_id
    with connect(pg_env) as conn:
        assert active_build(conn, generation.generation_id, QWEN_MODEL_ID).index_build_id == second.index_build_id
        statuses = dict(conn.execute("SELECT index_build_id, status FROM index_builds").fetchall())
    assert statuses == {first.index_build_id: "superseded", second.index_build_id: "active"}


def test_a_failed_rebuild_keeps_the_previous_build_active(published, fake_vllm, pg_env) -> None:
    root, generation = published
    first = build_index(root, embedder=_embedder(fake_vllm), dsn_env=pg_env)
    # The second batch of the rebuild comes back with a zero vector.
    fake_vllm.script.extend(
        [
            lambda inputs: (200, {"data": [{"index": i, "embedding": embed_text(t)} for i, t in enumerate(inputs)]}),
            lambda inputs: (200, {"data": [{"index": i, "embedding": [0.0] * 1024} for i, _ in enumerate(inputs)]}),
        ]
    )

    with pytest.raises(EmbeddingServiceError, match="zero"):
        build_index(root, embedder=_embedder(fake_vllm, batch_size=2), dsn_env=pg_env)

    with connect(pg_env) as conn:
        build = active_build(conn, generation.generation_id, QWEN_MODEL_ID)
        statuses = collections.Counter(
            row[0] for row in conn.execute("SELECT status FROM index_builds")
        )
    assert build.index_build_id == first.index_build_id
    assert build.row_count == generation.summary.chunks
    assert statuses == {"active": 1, "failed": 1}


def test_the_active_build_survives_a_new_session(published, fake_vllm, pg_env) -> None:
    root, generation = published
    result = build_index(root, embedder=_embedder(fake_vllm), dsn_env=pg_env)

    # A fresh connection (as after a process or container restart) sees the
    # same active build and every row.
    with connect(pg_env) as conn:
        build = active_build(conn, generation.generation_id, QWEN_MODEL_ID)
    assert build.index_build_id == result.index_build_id
    assert build.row_count == generation.summary.chunks


def test_indexing_pins_current_once(published, fake_vllm, pg_env) -> None:
    from scripts.v4.run import run

    root, generation = published
    pinned = generation.generation_id

    def publish_midway(inputs):
        # A new generation is published while the index is being built.
        if not getattr(publish_midway, "done", False):
            publish_midway.done = True
            run([long_document("doc-z")], root)
        return 200, {"data": [{"index": i, "embedding": embed_text(t)} for i, t in enumerate(inputs)]}

    fake_vllm.script.extend([publish_midway] * 50)
    result = build_index(root, embedder=_embedder(fake_vllm, batch_size=2), dsn_env=pg_env)

    assert result.generation_id == pinned
    assert resolve_current(root).name != pinned
    assert result.rows == generation.summary.chunks


def test_the_cli_takes_no_credential_arguments() -> None:
    from scripts.v4 import index

    parser = index.build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert not {"--api-key", "--password", "--dsn"} & options
    assert "--postgres-dsn-env" in options
