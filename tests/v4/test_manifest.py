from __future__ import annotations

from scripts.v4 import manifest


def test_manifest_agrees_with_the_file_tree() -> None:
    assert manifest.verify() == []


def test_v4_never_imports_an_excluded_module() -> None:
    offenders: list[str] = []
    for relative in manifest.VALID:
        source = (manifest.REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        for excluded in manifest.EXCLUDED:
            module = excluded.removesuffix(".py").replace("/", ".")
            if f"from {module} import" in source or f"import {module}" in source:
                offenders.append(f"{relative} -> {excluded}")
    assert offenders == []


def test_every_reused_module_is_actually_imported_by_v4() -> None:
    sources = "\n".join(
        (manifest.REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        for relative in manifest.VALID
        if relative.endswith(".py")
    )
    stale = [
        reused
        for reused in manifest.REUSED
        if reused.removesuffix(".py").replace("/", ".") not in sources
    ]
    assert stale == []


def test_the_sql_migration_is_declared() -> None:
    assert "scripts/v4/sql/001_vector_schema.sql" in manifest.VALID


def _problems_for(tmp_path, relative):
    return [p for p in manifest.verify(tmp_path) if p.path == relative]


def test_an_undeclared_non_python_artifact_below_v4_is_detected(tmp_path) -> None:
    stray = tmp_path / "scripts" / "v4" / "sql" / "002_extra.sql"
    stray.parent.mkdir(parents=True)
    stray.write_text("SELECT 1;\n", encoding="utf-8")

    assert [p.problem for p in _problems_for(tmp_path, "scripts/v4/sql/002_extra.sql")] == [
        "present but undeclared"
    ]


def test_bytecode_caches_are_not_build_artifacts(tmp_path) -> None:
    cache = tmp_path / "scripts" / "v4" / "__pycache__" / "run.cpython-313.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"\x00")

    assert _problems_for(tmp_path, "scripts/v4/__pycache__/run.cpython-313.pyc") == []
