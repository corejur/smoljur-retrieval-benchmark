"""Release validation against the real compose.yaml container (T024).

This test restarts the `postgres` service, so it runs only when asked:
`V4_RUN_CONTAINER_TESTS=1`. On the release validation path
(`V4_REQUIRE_PGVECTOR=1`) it is mandatory — a missing Docker or container
fails instead of skipping.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

pytest.importorskip("psycopg")

from scripts.v4.embeddings import QWEN_MODEL_ID, RemoteEmbedder  # noqa: E402
from scripts.v4.index import build_index  # noqa: E402
from scripts.v4.pgvector_store import active_build, connect, migrate  # noqa: E402
from scripts.v4.retrieve import rank  # noqa: E402

from .conftest import REPOSITORY_ROOT  # noqa: E402

SERVICE = "postgres"
CONTAINER = "redator-v4-postgres"


def _required() -> bool:
    return os.environ.get("V4_REQUIRE_PGVECTOR") == "1"


@pytest.fixture(scope="module", autouse=True)
def _container_opt_in():
    if os.environ.get("V4_RUN_CONTAINER_TESTS") != "1" and not _required():
        pytest.skip("restarts the compose postgres service; set V4_RUN_CONTAINER_TESTS=1")
    if shutil.which("docker") is None:
        (pytest.fail if _required() else pytest.skip)("docker is not available")


def _compose(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _wait_healthy(timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Health.Status}}", CONTAINER],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if state == "healthy":
            return
        time.sleep(1)
    pytest.fail(f"{CONTAINER} did not become healthy within {timeout:.0f}s")


def test_the_compose_service_builds_persists_and_ranks_across_a_restart(
    published, fake_vllm, pg_env
) -> None:
    _compose("up", "-d", SERVICE)
    _wait_healthy()
    root, generation = published

    with connect(pg_env) as conn:
        migrate(conn)
        conn.execute("TRUNCATE active_model_indexes, indexed_chunks, index_builds")
        conn.commit()
        # The pinned image provides the extension; the migration enabled it.
        assert conn.execute(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        ).fetchone() == ("vector",)

    embedder = RemoteEmbedder(fake_vllm.base_url, sleep=lambda _s: None)
    build = build_index(root, embedder=embedder, dsn_env=pg_env)
    before = rank(root / "current" / "beir", embedder=embedder, dsn_env=pg_env, top_k=3).run

    _compose("restart", SERVICE)
    _wait_healthy()

    with connect(pg_env) as conn:
        after_restart = active_build(conn, generation.generation_id, QWEN_MODEL_ID)
    assert after_restart.index_build_id == build.index_build_id
    assert after_restart.row_count == generation.summary.chunks
    after = rank(root / "current" / "beir", embedder=embedder, dsn_env=pg_env, top_k=3).run
    assert after == before
