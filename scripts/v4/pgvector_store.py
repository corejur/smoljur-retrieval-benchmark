"""PostgreSQL/pgvector storage for the Qwen chunk index (contracts/model-index.md).

A build moves through `building -> validated -> active -> superseded`, or to
`failed`. Rows are inserted only while `building`; a database trigger keeps
them immutable afterwards. Activation happens in one transaction that
switches the `(generation_id, model_id)` pointer, so a reader sees the old
complete build or the new complete build, never a partial one.

Search is exact: rows are narrowed by build, document, and published
candidate IDs before ordering by cosine distance, then by chunk ID for
deterministic ties.
"""

from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from scripts.v4.embeddings import DIMENSION, QWEN_MODEL_ID

__all__ = [
    "BuildProvenance",
    "IndexBuild",
    "IndexBuildError",
    "IndexNotReadyError",
    "IndexedChunk",
    "activate_build",
    "active_build",
    "chunk_documents",
    "connect",
    "create_build",
    "fail_build",
    "insert_chunks",
    "migrate",
    "search",
    "validate_build",
]

MIGRATION = Path(__file__).with_name("sql") / "001_vector_schema.sql"

#: Stored vectors are unit length; allow float32 rounding, nothing more.
_NORM_TOLERANCE = 1e-3


class IndexBuildError(RuntimeError):
    """A build, its rows, or the database configuration is not usable."""


class IndexNotReadyError(IndexBuildError):
    """No complete active build exists for the requested generation and model."""


@dataclass(frozen=True)
class BuildProvenance:
    model_id: str
    model_revision: str
    generation_id: str
    generation_manifest_sha256: str
    source_sha256: str
    split: str
    query_instruction: str
    chunk_text_profile: str
    chunk_count: int
    dimension: int = DIMENSION
    metric: str = "cosine"
    normalized: bool = True


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    embedding: np.ndarray


@dataclass(frozen=True)
class IndexBuild:
    index_build_id: str
    status: str
    provenance: BuildProvenance
    #: Rows actually present in `indexed_chunks` for this build.
    row_count: int


_PROVENANCE_COLUMNS = (
    "model_id, model_revision, generation_id, generation_manifest_sha256, "
    "source_sha256, split, query_instruction, chunk_text_profile, chunk_count, "
    "dimension, metric, normalized"
)


def connect(dsn_env: str):
    """Open a connection from the DSN held in environment variable `dsn_env`.

    The DSN itself is never accepted as an argument or echoed in an error.
    """
    import psycopg

    dsn = os.environ.get(dsn_env)
    if not dsn:
        raise IndexBuildError(
            f"environment variable {dsn_env} is not set; it must hold the "
            "PostgreSQL DSN (see compose.yaml)"
        )
    try:
        conn = psycopg.connect(dsn)
    except psycopg.OperationalError as error:
        # libpq messages name host, port, and user — never the password.
        raise IndexBuildError(f"cannot connect to PostgreSQL via {dsn_env}: {error}") from None
    if conn.execute("SELECT 1 FROM pg_type WHERE typname = 'vector'").fetchone():
        _register(conn)
    conn.commit()
    return conn


def _register(conn) -> None:
    from pgvector.psycopg import register_vector

    register_vector(conn)


def migrate(conn) -> None:
    """Apply the idempotent schema migration, serialized across processes."""
    conn.execute("SELECT pg_advisory_xact_lock(hashtext('redator-v4-migration'))")
    conn.execute(MIGRATION.read_text(encoding="utf-8"))
    conn.commit()
    _register(conn)


def new_index_build_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"ib-{stamp}-{secrets.token_hex(4)}"


def create_build(conn, provenance: BuildProvenance) -> str:
    """Register a new `building` build that no reader will select."""
    if provenance.model_id != QWEN_MODEL_ID:
        raise IndexBuildError(
            f"only {QWEN_MODEL_ID} builds are allowed; got {provenance.model_id!r}"
        )
    if provenance.split != "test":
        raise IndexBuildError(
            f"only test-split generations may be indexed; got {provenance.split!r}"
        )
    build_id = new_index_build_id()
    conn.execute(
        f"INSERT INTO index_builds (index_build_id, {_PROVENANCE_COLUMNS}) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            build_id,
            provenance.model_id,
            provenance.model_revision,
            provenance.generation_id,
            provenance.generation_manifest_sha256,
            provenance.source_sha256,
            provenance.split,
            provenance.query_instruction,
            provenance.chunk_text_profile,
            provenance.chunk_count,
            provenance.dimension,
            provenance.metric,
            provenance.normalized,
        ),
    )
    conn.commit()
    return build_id


def insert_chunks(conn, build_id: str, chunks: Iterable[IndexedChunk]) -> int:
    """Insert one bounded batch of rows for a `building` build and commit."""
    rows = [
        (build_id, c.chunk_id, c.document_id, int(c.chunk_index), np.asarray(c.embedding, dtype=np.float32))
        for c in chunks
    ]
    if not rows:
        return 0
    try:
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO indexed_chunks "
                "(index_build_id, chunk_id, document_id, chunk_index, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                rows,
            )
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return len(rows)


def _build_row(conn, build_id: str, *, lock: bool = False):
    row = conn.execute(
        f"SELECT status, {_PROVENANCE_COLUMNS} FROM index_builds "
        f"WHERE index_build_id = %s{' FOR UPDATE' if lock else ''}",
        (build_id,),
    ).fetchone()
    if row is None:
        raise IndexBuildError(f"no index build {build_id}")
    return row[0], BuildProvenance(*row[1:])


def validate_build(
    conn, build_id: str, corpus: Mapping[str, tuple[str, int]]
) -> None:
    """Check a staged build against the published corpus; mark it validated.

    `corpus` maps every published chunk ID to its `(document_id,
    chunk_index)`. The build must hold exactly those chunks — a bijection —
    each filed under its own document and position, with a unit-length
    1,024-dimensional vector.
    """
    status, provenance = _build_row(conn, build_id)
    if status != "building":
        raise IndexBuildError(f"build {build_id} is {status}, not building")

    rows = conn.execute(
        "SELECT chunk_id, document_id, chunk_index FROM indexed_chunks "
        "WHERE index_build_id = %s",
        (build_id,),
    ).fetchall()
    expected = len(corpus)
    if len(rows) != expected or provenance.chunk_count != expected:
        raise IndexBuildError(
            f"build {build_id} has {len(rows)} rows but its generation publishes "
            f"{expected} chunks (build declares {provenance.chunk_count})"
        )
    stored = {chunk_id: (document_id, chunk_index) for chunk_id, document_id, chunk_index in rows}
    foreign = sorted(set(stored) - set(corpus))
    missing = sorted(set(corpus) - set(stored))
    if foreign or missing:
        raise IndexBuildError(
            f"build {build_id} does not match the published corpus: "
            f"unpublished {foreign[:5]}, missing {missing[:5]}"
        )
    misplaced = sorted(
        chunk_id for chunk_id, place in stored.items() if tuple(corpus[chunk_id]) != place
    )
    if misplaced:
        raise IndexBuildError(
            f"build {build_id} files chunks under the wrong document or position: "
            f"{misplaced[:5]}"
        )
    bad = conn.execute(
        "SELECT chunk_id FROM indexed_chunks WHERE index_build_id = %s AND "
        "(vector_dims(embedding) <> %s OR abs(vector_norm(embedding) - 1) > %s) "
        "ORDER BY chunk_id LIMIT 5",
        (build_id, DIMENSION, _NORM_TOLERANCE),
    ).fetchall()
    if bad:
        raise IndexBuildError(
            f"build {build_id} holds vectors that are not unit-norm "
            f"{DIMENSION}-dimensional: {[row[0] for row in bad]}"
        )
    conn.execute(
        "UPDATE index_builds SET status = 'validated', updated_at = now() "
        "WHERE index_build_id = %s",
        (build_id,),
    )
    conn.commit()


def activate_build(conn, build_id: str) -> str | None:
    """Make a validated build active in one transaction; return the one it replaced."""
    try:
        status, provenance = _build_row(conn, build_id, lock=True)
        if status != "validated":
            raise IndexBuildError(f"build {build_id} is {status}; only a validated build can be activated")
        key = (provenance.generation_id, provenance.model_id)
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s || '|' || %s))", key)
        previous = conn.execute(
            "SELECT index_build_id FROM active_model_indexes "
            "WHERE generation_id = %s AND model_id = %s FOR UPDATE",
            key,
        ).fetchone()
        previous_id = previous[0] if previous else None
        if previous_id and previous_id != build_id:
            conn.execute(
                "UPDATE index_builds SET status = 'superseded', updated_at = now() "
                "WHERE index_build_id = %s",
                (previous_id,),
            )
        conn.execute(
            "UPDATE index_builds SET status = 'active', updated_at = now() "
            "WHERE index_build_id = %s",
            (build_id,),
        )
        conn.execute(
            "INSERT INTO active_model_indexes (generation_id, model_id, index_build_id) "
            "VALUES (%s, %s, %s) ON CONFLICT (generation_id, model_id) DO UPDATE "
            "SET index_build_id = EXCLUDED.index_build_id, activated_at = now()",
            (*key, build_id),
        )
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return previous_id


def fail_build(conn, build_id: str) -> None:
    """Mark an unfinished build failed. Its rows are kept, never selected."""
    conn.rollback()
    conn.execute(
        "UPDATE index_builds SET status = 'failed', updated_at = now() "
        "WHERE index_build_id = %s AND status IN ('building', 'validated')",
        (build_id,),
    )
    conn.commit()


def active_build(conn, generation_id: str, model_id: str) -> IndexBuild:
    """The build selected for this generation and model, with its row count."""
    row = conn.execute(
        "SELECT b.index_build_id, b.status, "
        + ", ".join(f"b.{c.strip()}" for c in _PROVENANCE_COLUMNS.split(","))
        + ", (SELECT count(*) FROM indexed_chunks c WHERE c.index_build_id = b.index_build_id) "
        "FROM active_model_indexes a JOIN index_builds b USING (index_build_id) "
        "WHERE a.generation_id = %s AND a.model_id = %s",
        (generation_id, model_id),
    ).fetchone()
    if row is None or row[1] != "active":
        raise IndexNotReadyError(
            f"index_not_ready: no active {model_id} build for generation {generation_id}; "
            "run python -m scripts.v4.index for it first"
        )
    return IndexBuild(row[0], row[1], BuildProvenance(*row[2:-1]), int(row[-1]))


def chunk_documents(conn, build_id: str) -> dict[str, str]:
    """Every indexed chunk ID of a build, mapped to its document."""
    return dict(
        conn.execute(
            "SELECT chunk_id, document_id FROM indexed_chunks WHERE index_build_id = %s",
            (build_id,),
        ).fetchall()
    )


def search(
    conn,
    build_id: str,
    document_id: str,
    candidate_ids: Sequence[str],
    query_vector: np.ndarray,
    *,
    top_k: int,
) -> list[tuple[str, float]]:
    """Exact cosine ranking of one document's candidates; best first.

    Rows are filtered by build, document, and candidate IDs *before* the
    distance ordering, so the result is the document's true top-k.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    results = conn.execute(
        "SELECT chunk_id, 1 - (embedding <=> %s) AS score FROM indexed_chunks "
        "WHERE index_build_id = %s AND document_id = %s AND chunk_id = ANY(%s) "
        "ORDER BY embedding <=> %s, chunk_id LIMIT %s",
        (
            np.asarray(query_vector, dtype=np.float32),
            build_id,
            document_id,
            list(candidate_ids),
            np.asarray(query_vector, dtype=np.float32),
            top_k,
        ),
    ).fetchall()
    allowed = set(candidate_ids)
    seen: set[str] = set()
    ranked: list[tuple[str, float]] = []
    for chunk_id, score in results:
        if chunk_id not in allowed or chunk_id in seen:
            raise IndexBuildError(f"search returned {chunk_id}, outside the candidate set")
        if score is None or not math.isfinite(score):
            raise IndexBuildError(f"search returned a non-finite score for {chunk_id}")
        seen.add(chunk_id)
        ranked.append((chunk_id, float(score)))
    if len(ranked) > top_k:
        raise IndexBuildError(f"search returned {len(ranked)} results for top_k={top_k}")
    return ranked
