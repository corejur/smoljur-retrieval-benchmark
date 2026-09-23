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
