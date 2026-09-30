"""PostgreSQL/pgvector schema and exact document-scoped search (T020).

Runs against a throwaway database (see conftest.py). Every test starts from
empty tables.
"""

from __future__ import annotations

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")

from scripts.v4.embeddings import DIMENSION, QUERY_INSTRUCTION, QWEN_MODEL_ID  # noqa: E402
from scripts.v4.pgvector_store import (  # noqa: E402
    BuildProvenance,
    IndexBuildError,
    IndexedChunk,
    IndexNotReadyError,
    activate_build,
    active_build,
    chunk_documents,
    connect,
    create_build,
    fail_build,
    insert_chunks,
    migrate,
    search,
    validate_build,
)


@pytest.fixture
def conn(pg_env):
    connection = connect(pg_env)
    migrate(connection)
    connection.execute("TRUNCATE active_model_indexes, indexed_chunks, index_builds")
    connection.commit()
    try:
        yield connection
    finally:
        connection.close()


def _unit(*weights: float) -> np.ndarray:
    vector = np.zeros(DIMENSION, dtype=np.float32)
    vector[: len(weights)] = weights
    return vector / np.linalg.norm(vector)


def _provenance(generation_id="gen-1", chunk_count=4, **overrides) -> BuildProvenance:
    values = dict(
        model_id=QWEN_MODEL_ID,
        model_revision="unpinned",
        generation_id=generation_id,
        generation_manifest_sha256="a" * 64,
        source_sha256="b" * 64,
        split="test",
        query_instruction=QUERY_INSTRUCTION,
        chunk_text_profile="corpus-text-v1",
        chunk_count=chunk_count,
    )
    values.update(overrides)
    return BuildProvenance(**values)


#: Two documents; doc-b:c0 is the closest chunk overall to the query below.
_CHUNKS = [
    IndexedChunk("doc-a:c0", "doc-a", 0, _unit(1.0, 0.0)),
    IndexedChunk("doc-a:c1", "doc-a", 1, _unit(0.6, 0.8)),
    IndexedChunk("doc-a:c2", "doc-a", 2, _unit(0.8, 0.6)),
    IndexedChunk("doc-b:c0", "doc-b", 0, _unit(0.0, 1.0)),
]
_CORPUS = {c.chunk_id: (c.document_id, c.chunk_index) for c in _CHUNKS}


def _active(conn, chunks=_CHUNKS, generation_id="gen-1") -> str:
    build_id = create_build(conn, _provenance(generation_id, len(chunks)))
    insert_chunks(conn, build_id, chunks)
    validate_build(conn, build_id, {c.chunk_id: (c.document_id, c.chunk_index) for c in chunks})
    activate_build(conn, build_id)
    return build_id


# --- schema -------------------------------------------------------------------


def _one(conn, sql, *params):
    return conn.execute(sql, params).fetchone()


def test_the_migration_is_idempotent(conn) -> None:
    migrate(conn)
    migrate(conn)
    assert _one(conn, "SELECT extversion FROM pg_extension WHERE extname = 'vector'")


def test_embedding_is_a_non_null_unsized_vector(conn) -> None:
    """Unsized so each model keeps its native dimension (checked per build)."""
    row = _one(
        conn,
        """SELECT format_type(a.atttypid, a.atttypmod), a.attnotnull
           FROM pg_attribute a WHERE a.attrelid = 'indexed_chunks'::regclass
           AND a.attname = 'embedding'""",
    )
    assert row == ("vector", True)


@pytest.mark.parametrize(
    "table, kind, columns",
    [
        ("indexed_chunks", "p", ["index_build_id", "chunk_id"]),
        ("indexed_chunks", "u", ["index_build_id", "document_id", "chunk_index"]),
        ("active_model_indexes", "p", ["generation_id", "model_id"]),
        ("index_builds", "p", ["index_build_id"]),
    ],
)
def test_keys_and_uniqueness(conn, table, kind, columns) -> None:
    rows = conn.execute(
        """SELECT array_agg(a.attname ORDER BY k.ordinality)
           FROM pg_constraint c
           CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ordinality)
           JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
           WHERE c.conrelid = %s::regclass AND c.contype = %s
           GROUP BY c.oid""",
        (table, kind),
    ).fetchall()
    assert columns in [list(row[0]) for row in rows]


def test_a_btree_narrows_by_build_and_document(conn) -> None:
    definitions = [
        row[0]
        for row in conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'indexed_chunks'"
        )
    ]
    assert any(
        "USING btree (index_build_id, document_id" in definition
        for definition in definitions
    )


def test_no_approximate_vector_index_exists(conn) -> None:
    methods = {
        row[0]
        for row in conn.execute(
            """SELECT am.amname FROM pg_index i
               JOIN pg_class c ON c.oid = i.indexrelid
               JOIN pg_am am ON am.oid = c.relam"""
        )
    }
    assert not methods & {"hnsw", "ivfflat"}


def test_the_active_pointer_references_a_build_of_the_same_generation_and_model(conn) -> None:
    build_id = create_build(conn, _provenance("gen-1", 0))
    conn.commit()
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute(
            "INSERT INTO active_model_indexes (generation_id, model_id, index_build_id) "
            "VALUES ('gen-OTHER', %s, %s)",
            (QWEN_MODEL_ID, build_id),
        )
    conn.rollback()


def test_a_wrong_dimension_is_refused_at_validation(conn) -> None:
    build_id = create_build(conn, _provenance(chunk_count=1))
    vector = np.ones(1023, dtype=np.float32) / np.sqrt(1023)
    insert_chunks(conn, build_id, [IndexedChunk("x", "d", 0, vector)])

    with pytest.raises(IndexBuildError, match="1024-dimensional"):
        validate_build(conn, build_id, {"x": ("d", 0)})


def test_only_benchmarked_model_test_split_builds_are_accepted(conn) -> None:
    with pytest.raises(IndexBuildError, match="intfloat/e5-large"):
        create_build(conn, _provenance(model_id="intfloat/e5-large"))
    with pytest.raises(IndexBuildError, match="test"):
        create_build(conn, _provenance(split="train"))


# --- immutability -------------------------------------------------------------


def test_rows_of_a_validated_build_cannot_change(conn) -> None:
    build_id = _active(conn)

    with pytest.raises(psycopg.errors.RaiseException, match="immutable"):
        conn.execute(
            "UPDATE indexed_chunks SET chunk_index = 9 WHERE index_build_id = %s", (build_id,)
        )
    conn.rollback()
    with pytest.raises(psycopg.errors.RaiseException, match="immutable"):
        conn.execute("DELETE FROM indexed_chunks WHERE index_build_id = %s", (build_id,))
    conn.rollback()
    with pytest.raises(psycopg.errors.RaiseException, match="immutable"):
        insert_chunks(conn, build_id, [IndexedChunk("doc-a:c9", "doc-a", 9, _unit(1.0))])
    conn.rollback()


# --- build lifecycle ----------------------------------------------------------


def test_validation_rejects_an_incomplete_build(conn) -> None:
    build_id = create_build(conn, _provenance(chunk_count=4))
    insert_chunks(conn, build_id, _CHUNKS[:3])

    with pytest.raises(IndexBuildError, match="3 rows.*4"):
        validate_build(conn, build_id, _CORPUS)


def test_validation_rejects_a_chunk_outside_the_corpus(conn) -> None:
    build_id = create_build(conn, _provenance(chunk_count=4))
    insert_chunks(conn, build_id, [*_CHUNKS[:3], IndexedChunk("doc-z:c0", "doc-z", 0, _unit(1.0))])

    with pytest.raises(IndexBuildError, match="doc-z:c0"):
        validate_build(conn, build_id, _CORPUS)


def test_validation_rejects_a_chunk_filed_under_the_wrong_document(conn) -> None:
    build_id = create_build(conn, _provenance(chunk_count=4))
    moved = IndexedChunk("doc-b:c0", "doc-a", 3, _unit(0.0, 1.0))
    insert_chunks(conn, build_id, [*_CHUNKS[:3], moved])

    with pytest.raises(IndexBuildError, match="doc-b:c0"):
        validate_build(conn, build_id, _CORPUS)


def test_validation_rejects_a_vector_that_is_not_unit_length(conn) -> None:
    build_id = create_build(conn, _provenance(chunk_count=4))
    scaled = IndexedChunk("doc-b:c0", "doc-b", 0, _unit(0.0, 1.0) * 3)
    insert_chunks(conn, build_id, [*_CHUNKS[:3], scaled])

    with pytest.raises(IndexBuildError, match="norm"):
        validate_build(conn, build_id, _CORPUS)


def test_only_a_validated_build_can_be_activated(conn) -> None:
    build_id = create_build(conn, _provenance(chunk_count=4))
    insert_chunks(conn, build_id, _CHUNKS)

    with pytest.raises(IndexBuildError, match="building"):
        activate_build(conn, build_id)


def test_activation_supersedes_the_previous_build(conn) -> None:
    first = _active(conn)
    second = _active(conn)

    assert active_build(conn, "gen-1", QWEN_MODEL_ID).index_build_id == second
    status = dict(conn.execute("SELECT index_build_id, status FROM index_builds").fetchall())
    assert status == {first: "superseded", second: "active"}


def test_a_failed_build_leaves_the_active_one_selected(conn) -> None:
    first = _active(conn)
    second = create_build(conn, _provenance(chunk_count=4))
    insert_chunks(conn, second, _CHUNKS[:2])
    fail_build(conn, second)

    assert active_build(conn, "gen-1", QWEN_MODEL_ID).index_build_id == first
    status = dict(conn.execute("SELECT index_build_id, status FROM index_builds").fetchall())
    assert status[second] == "failed"


def test_a_generation_without_an_active_build_is_not_ready(conn) -> None:
    _active(conn)

    with pytest.raises(IndexNotReadyError, match="index_not_ready.*gen-2"):
        active_build(conn, "gen-2", QWEN_MODEL_ID)


def test_the_active_build_carries_its_provenance(conn) -> None:
    build_id = _active(conn)

    build = active_build(conn, "gen-1", QWEN_MODEL_ID)
    assert build.index_build_id == build_id
    assert build.provenance == _provenance()
    assert build.row_count == 4
    assert chunk_documents(conn, build_id) == {c.chunk_id: c.document_id for c in _CHUNKS}


# --- exact document-scoped search --------------------------------------------


def test_search_ranks_only_the_documents_candidates_by_cosine(conn) -> None:
    build_id = _active(conn)
    query = _unit(0.0, 1.0)  # doc-b:c0 would win a global search

    results = search(conn, build_id, "doc-a", ["doc-a:c0", "doc-a:c1", "doc-a:c2"], query, top_k=10)

    assert [chunk_id for chunk_id, _ in results] == ["doc-a:c1", "doc-a:c2", "doc-a:c0"]
    assert results[0][1] == pytest.approx(0.8, abs=1e-5)
    assert results[2][1] == pytest.approx(0.0, abs=1e-5)


def test_search_never_returns_a_non_candidate_of_the_same_document(conn) -> None:
    build_id = _active(conn)

    results = search(conn, build_id, "doc-a", ["doc-a:c0", "doc-a:c2"], _unit(0.6, 0.8), top_k=10)

    assert [chunk_id for chunk_id, _ in results] == ["doc-a:c2", "doc-a:c0"]


def test_search_applies_top_k_after_filtering(conn) -> None:
    build_id = _active(conn)

    results = search(conn, build_id, "doc-a", ["doc-a:c0", "doc-a:c1", "doc-a:c2"], _unit(0.0, 1.0), top_k=1)

    assert [chunk_id for chunk_id, _ in results] == ["doc-a:c1"]


def test_ties_are_broken_by_chunk_id(conn) -> None:
    same = _unit(1.0)
    chunks = [IndexedChunk(f"doc-t:c{i}", "doc-t", i, same) for i in (2, 0, 1)]
    build_id = _active(conn, chunks, generation_id="gen-tie")

    results = search(conn, build_id, "doc-t", [c.chunk_id for c in chunks], same, top_k=3)

    assert [chunk_id for chunk_id, _ in results] == ["doc-t:c0", "doc-t:c1", "doc-t:c2"]


def test_search_is_scoped_to_its_build(conn) -> None:
    old = _active(conn)
    rotated = [
        IndexedChunk(c.chunk_id, c.document_id, c.chunk_index, _unit(0.0, 0.0, 1.0) if c.chunk_id == "doc-a:c0" else c.embedding)
        for c in _CHUNKS
    ]
    new = _active(conn, rotated)

    query = _unit(1.0)
    candidates = ["doc-a:c0", "doc-a:c1", "doc-a:c2"]
    assert search(conn, old, "doc-a", candidates, query, top_k=1)[0][0] == "doc-a:c0"
    assert search(conn, new, "doc-a", candidates, query, top_k=1)[0][0] == "doc-a:c2"


def test_search_rejects_a_non_positive_top_k(conn) -> None:
    build_id = _active(conn)
    with pytest.raises(ValueError, match="top_k"):
        search(conn, build_id, "doc-a", ["doc-a:c0"], _unit(1.0), top_k=0)


def test_connect_names_a_missing_environment_variable(monkeypatch) -> None:
    monkeypatch.delenv("V4_NOT_SET_DSN", raising=False)
    with pytest.raises(IndexBuildError, match="V4_NOT_SET_DSN"):
        connect("V4_NOT_SET_DSN")


# --- per-model dimensions -------------------------------------------------------


def test_a_build_must_declare_its_models_own_dimension(conn) -> None:
    from scripts.v4.embeddings import QWEN3_4B_MODEL_ID

    with pytest.raises(IndexBuildError, match="2560-dimensional; got 1024"):
        create_build(conn, _provenance(model_id=QWEN3_4B_MODEL_ID))
    conn.rollback()

    build_id = create_build(conn, _provenance(model_id=QWEN3_4B_MODEL_ID, dimension=2560))
    insert_chunks(conn, build_id, [IndexedChunk("c1", "d1", 0, np.eye(2560, dtype=np.float32)[0])])
    assert conn.execute(
        "SELECT vector_dims(embedding) FROM indexed_chunks WHERE index_build_id = %s", (build_id,)
    ).fetchone() == (2560,)


def test_validation_checks_rows_against_the_builds_dimension(conn) -> None:
    from scripts.v4.embeddings import QWEN3_4B_MODEL_ID

    build_id = create_build(
        conn, _provenance(model_id=QWEN3_4B_MODEL_ID, dimension=2560, chunk_count=1)
    )
    insert_chunks(conn, build_id, [IndexedChunk("c1", "d1", 0, np.eye(1024, dtype=np.float32)[0])])

    with pytest.raises(IndexBuildError, match="2560-dimensional"):
        validate_build(conn, build_id, {"c1": ("d1", 0)})


_FIXED_SIZE_SCHEMA = """
CREATE TABLE index_builds (
    index_build_id text PRIMARY KEY, model_id text NOT NULL, model_revision text NOT NULL,
    generation_id text NOT NULL, generation_manifest_sha256 text NOT NULL, source_sha256 text NOT NULL,
    split text NOT NULL, dimension integer NOT NULL CHECK (dimension = 1024), metric text NOT NULL,
    normalized boolean NOT NULL, query_instruction text NOT NULL, chunk_text_profile text NOT NULL,
    chunk_count integer NOT NULL, status text NOT NULL DEFAULT 'building',
    created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (index_build_id, generation_id, model_id)
);
CREATE TABLE indexed_chunks (
    index_build_id text NOT NULL REFERENCES index_builds (index_build_id), chunk_id text NOT NULL,
    document_id text NOT NULL, chunk_index integer NOT NULL, embedding vector(1024) NOT NULL,
    PRIMARY KEY (index_build_id, chunk_id)
);
INSERT INTO index_builds VALUES ('ib-old', 'Qwen/Qwen3-Embedding-0.6B', 'unpinned', 'gen-1',
    repeat('a', 64), repeat('b', 64), 'test', 1024, 'cosine', true, 'q', 'p', 1, 'active');
"""


def test_the_migration_unfixes_a_1024_only_database(pg_env) -> None:
    """A database created before per-model dimensions keeps its rows and accepts any size."""
    connection = connect(pg_env)
    try:
        connection.execute("CREATE SCHEMA legacy")
        connection.execute("SET search_path TO legacy, public")
        connection.execute(_FIXED_SIZE_SCHEMA)
        connection.execute(
            "INSERT INTO indexed_chunks VALUES ('ib-old', 'c1', 'd1', 0, %s)",
            ("[" + ",".join(["1"] + ["0"] * 1023) + "]",),
        )
        connection.commit()

        migrate(connection)
        migrate(connection)  # idempotent

        check = connection.execute(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conname = 'index_builds_dimension_check' "
            "AND conrelid = 'legacy.index_builds'::regclass"
        ).fetchone()[0]
        typmod = connection.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'legacy.indexed_chunks'::regclass AND attname = 'embedding'"
        ).fetchone()[0]
        kept = connection.execute("SELECT vector_dims(embedding) FROM legacy.indexed_chunks").fetchall()
        assert "dimension > 0" in check
        assert typmod == -1
        assert kept == [(1024,)]
    finally:
        connection.rollback()
        connection.execute("DROP SCHEMA IF EXISTS legacy CASCADE")
        connection.commit()
        connection.close()
