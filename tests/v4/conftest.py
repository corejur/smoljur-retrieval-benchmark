"""Shared fixtures for the pgvector index and retrieval tests (User Story 4).

PostgreSQL tests run against a throwaway database created from
`V4_POSTGRES_DSN` (read from the environment, or from the repository `.env`)
and dropped afterwards, so they never touch the operator's `redator_v4`
database. Without a reachable database they skip with the reason; with
`V4_REQUIRE_PGVECTOR=1` — the release validation path — a missing database
fails instead of skipping.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Iterator

import pytest

from .fake_vllm import FakeVllm, start_fake_vllm

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

_PARAGRAPHS = (
    '<p id="p-0">O autor da acao e Joao da Silva residente na capital</p>'
    '<p id="p-1">O valor da causa foi fixado pela parte interessada</p>'
    '<p id="p-2">Terceiro paragrafo com texto suficiente para o documento</p>'
)


def v4_environment(name: str) -> str | None:
    """An environment value, falling back to the gitignored `.env` file."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = REPOSITORY_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip()
    return None


def _unavailable(reason: str) -> None:
    if os.environ.get("V4_REQUIRE_PGVECTOR") == "1":
        pytest.fail(f"V4_REQUIRE_PGVECTOR=1 but {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="session")
def pg_admin_dsn() -> str:
    dsn = v4_environment("V4_POSTGRES_DSN")
    if not dsn:
        _unavailable("V4_POSTGRES_DSN is not set; start compose.yaml's postgres service")
    try:
        import psycopg
    except ImportError:
        _unavailable("psycopg is not installed; pip install -e '.[v4]'")
    try:
        psycopg.connect(dsn, connect_timeout=3).close()
    except psycopg.OperationalError as error:
        _unavailable(f"PostgreSQL at V4_POSTGRES_DSN is unreachable ({type(error).__name__})")
    return dsn


@pytest.fixture(scope="session")
def pg_dsn(pg_admin_dsn: str) -> Iterator[str]:
    """A fresh database for this test session, dropped at the end."""
    import psycopg
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    name = f"redator_v4_test_{secrets.token_hex(4)}"
    with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    params = conninfo_to_dict(pg_admin_dsn)
    params["dbname"] = name
    try:
        yield make_conninfo(**params)
    finally:
        with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
def pg_env(pg_dsn: str, monkeypatch) -> str:
    """Expose the session database as an environment-held DSN."""
    monkeypatch.setenv("V4_TEST_POSTGRES_DSN", pg_dsn)
    return "V4_TEST_POSTGRES_DSN"


@pytest.fixture
def fake_vllm() -> Iterator[FakeVllm]:
    fake = start_fake_vllm()
    try:
        yield fake
    finally:
        fake.close()


def source_row(document_id: str, questions: list[tuple[str, str]], html: str = _PARAGRAPHS):
    """One v4 source row; `questions` pairs question text with a cited paragraph."""
    return {
        "id": document_id,
        "question_texts": json.dumps([q for q, _ in questions], ensure_ascii=False),
        "output": json.dumps(
            [{"answer": "Sim", "citations": [c]} for _, c in questions],
            ensure_ascii=False,
        ),
        "data": html,
    }


def long_document(document_id: str) -> dict:
    """A document long enough to yield several passages to rank."""
    topics = [
        "autor Joao da Silva residente capital",
        "valor causa fixado parte interessada",
        "sentenca julgou procedente pedido indenizacao",
        "recurso apelacao interposto tribunal",
    ]
    html = "".join(
        f'<p id="p-{i}">{" ".join([topic] * 90)}</p>' for i, topic in enumerate(topics)
    )
    return source_row(
        document_id,
        [
            ("Quem e o autor Joao da Silva?", "p-0"),
            ("Qual o valor da causa fixado?", "p-1"),
            ("Houve recurso de apelacao ao tribunal?", "p-3"),
        ],
        html,
    )


@pytest.fixture
def published(tmp_path: Path):
    """A published synthetic generation with two multi-passage documents."""
    from scripts.v4.run import run

    root = tmp_path / "dataset"
    result = run([long_document("doc-a"), long_document("doc-b")], root)
    return root, result
