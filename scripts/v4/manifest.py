"""The authoritative file set for the v4 pipeline.

The v1-v3 pipeline has been removed from the tree; it remains in the git
history (main's first commit, 66837db). The two v1-v3 modules v4 still
depends on now live in `scripts/v4/` and are listed in REUSED. EXCLUDED names
any older module that must never be imported by v4 code; it is empty now that
none is left in the tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["VALID", "REUSED", "EXCLUDED", "REPOSITORY_ROOT", "verify"]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Files written for v4 — Python modules and other build artifacts such as SQL
#: migrations. Every one is part of the build.
VALID: tuple[str, ...] = (
    "scripts/v4/__init__.py",
    "scripts/v4/manifest.py",
    "scripts/v4/offsets.py",
    "scripts/v4/cleaning.py",
    "scripts/v4/questions.py",
    "scripts/v4/chunking.py",
    "scripts/v4/ground_truth.py",
    "scripts/v4/streams.py",
    "scripts/v4/headings.py",
    "scripts/v4/source_contract.py",
    "scripts/v4/generation.py",
    "scripts/v4/publication.py",
    "scripts/v4/embeddings.py",
    "scripts/v4/pgvector_store.py",
    "scripts/v4/index.py",
    "scripts/v4/plots.py",
    "scripts/v4/sql/001_vector_schema.sql",
    "scripts/v4/metrics.py",
    "scripts/v4/retrieve.py",
    "scripts/v4/run.py",
)

#: Modules carried over from v1-v3 that the v4 pipeline depends on.
REUSED: dict[str, str] = {
    "scripts/v4/source_normalization.py": "one HTML parse yielding text and citation spans",
    "scripts/v4/legal_recursive.py": (
        "structure-aware chunk splitting; extended with an optional is_heading "
        "hook that defaults to the v1-v3 rules, so v1-v3 output is unchanged"
    ),
}

#: Older modules that v4 code must never import, with their v4 replacement.
#: Empty: the v1-v3 pipeline was removed from the tree.
EXCLUDED: dict[str, str] = {}


@dataclass(frozen=True)
class ManifestProblem:
    path: str
    problem: str


def verify(root: Path | None = None) -> list[ManifestProblem]:
    """Return every disagreement between this manifest and the file tree."""
    base = root or REPOSITORY_ROOT
    problems: list[ManifestProblem] = []
    for path in (*VALID, *REUSED, *EXCLUDED):
        if not (base / path).is_file():
            problems.append(ManifestProblem(path, "declared but missing"))
    package = base / "scripts" / "v4"
    if package.is_dir():
        declared = set(VALID) | set(REUSED)
        for path in sorted(package.rglob("*")):
            relative = path.relative_to(base).as_posix()
            # Bytecode caches are generated, never part of the build.
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if relative not in declared:
                problems.append(ManifestProblem(relative, "present but undeclared"))
    overlap = set(REUSED) & set(EXCLUDED)
    problems.extend(
        ManifestProblem(path, "both reused and excluded") for path in sorted(overlap)
    )
    return problems
