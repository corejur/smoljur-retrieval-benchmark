"""The authoritative file set for the v4 pipeline.

`scripts/` still contains the v1-v3 pipeline. Presence on disk no longer means
a module is part of the dataset build: this manifest decides. Anything listed
in EXCLUDED is kept only so the older generations stay reproducible, and must
not be imported by v4 code.
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

#: v1-v3 modules the v4 pipeline still depends on, unchanged.
REUSED: dict[str, str] = {
    "scripts/source_normalization.py": "one HTML parse yielding text and citation spans",
    "scripts/strategies/chunking/legal_recursive.py": (
        "structure-aware chunk splitting; extended with an optional is_heading "
        "hook that defaults to the v1-v3 rules, so v1-v3 output is unchanged"
    ),
}

#: v1-v3 modules the v4 pipeline replaces. Not imported by anything in v4.
EXCLUDED: dict[str, str] = {
    "scripts/artifact_publication.py": (
        "scripts/v4/publication.py; the shared publisher swaps managed "
        "directories one at a time, so readers can see a mixed generation"
    ),
    "scripts/prepare_dataset.py": "scripts/v4/run.py",
    "scripts/citation_ground_truth.py": "scripts/v4/ground_truth.py",
    "scripts/data_cleaner.py": "scripts/v4/cleaning.py",
    "scripts/query_preparation.py": "scripts/v4/questions.py",
    "scripts/query_spliter.py": "scripts/v4/questions.py",
    "scripts/corpus_generator.py": "scripts/v4/ground_truth.prepare_document",
    "scripts/chunk_corpus.py": "scripts/v4/run.py with scripts/v4/chunking.py",
    "scripts/export_beir_dataset.py": "scripts/v4/run.py",
}


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
        declared = set(VALID)
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
