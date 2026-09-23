"""The published generation: its manifest, its required files, and its identity.

A *generation* is one immutable, internally consistent publication of the v4
dataset. `meta/manifest.json` inside it is the **generation manifest** — the
dataset's identity, counts, and configuration. It is a different thing from
`scripts/v4/manifest.py`, which is the module registry saying which source
files belong to the v4 build; the two are never merged.

A consumer resolves `current` exactly once, validates the snapshot it points
at, and then reads only that immutable directory. A later publication cannot
change what an in-flight reader sees.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

__all__ = [
    "PIPELINE",
    "GenerationError",
    "GenerationManifest",
    "REQUIRED_MANIFEST_KEYS",
    "REQUIRED_SUMMARY_KEYS",
    "load_generation",
    "manifest_sha256",
    "new_generation_id",
    "pin_generation_path",
    "read_manifest",
    "required_artifacts",
    "resolve_current",
    "validate_generation",
]

#: The only pipeline label a v4 generation may carry.
PIPELINE = "redator-v4"

#: The only split this feature publishes.
PRODUCTION_SPLIT = "test"

CURRENT = "current"
GENERATIONS = "generations"
#: Prefix of a generation that is still being built. `resolve_current` never
#: accepts it, so an interrupted build cannot be activated by accident.
BUILDING_PREFIX = ".building-"
MANIFEST = "meta/manifest.json"

#: Top-level generation-manifest keys, per contracts/dataset.md.
REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "pipeline",
    "split",
    "generation_id",
    "generated_at",
    "source_sha256",
    "source_row_count",
    "chunking",
    "summary",
)

#: Counters inside `summary`. `chunks` is the completeness contract an index
#: build checks its PostgreSQL row count against.
REQUIRED_SUMMARY_KEYS: tuple[str, ...] = (
    "source_rows",
    "rejected_sources",
    "documents",
    "chunks",
    "queries",
    "queries_retained",
    "qrels",
    "unmapped_citations",
    "plain_text_violations",
)

_AUDITS = (
    "rejected_sources",
    "dropped_queries",
    "unmapped_citations",
    "cleaning_events",
    "plain_text_violations",
)


class GenerationError(ValueError):
    """A generation is missing, incomplete, or internally inconsistent."""


@dataclass(frozen=True)
class GenerationManifest:
    """The parsed generation manifest."""

    pipeline: str
    split: str
    generation_id: str
    generated_at: str
    source_sha256: str
    source_row_count: int
    chunking: Mapping[str, Any]
    summary: Mapping[str, int]
    raw: Mapping[str, Any]

    @property
    def chunks(self) -> int:
        """Published corpus chunk count — `summary.chunks`."""
        return int(self.summary["chunks"])


def required_artifacts(split: str) -> tuple[str, ...]:
    """Every file a complete generation must contain, generation-relative."""
    return (
        "beir/corpus.jsonl",
        "beir/queries.jsonl",
        f"beir/qrels/{split}.tsv",
        f"beir/candidates/{split}.jsonl",
        "documents/documents.jsonl",
        "evidence/evidence.jsonl",
        *(f"audits/{name}.jsonl" for name in _AUDITS),
        MANIFEST,
    )


def read_manifest(generation_root: str | Path) -> GenerationManifest:
    """Parse and schema-check `meta/manifest.json`."""
    path = Path(generation_root) / MANIFEST
    if not path.is_file():
        raise GenerationError(f"generation has no {MANIFEST}: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GenerationError(f"{MANIFEST} is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise GenerationError(f"{MANIFEST} must hold a JSON object")

    for key in REQUIRED_MANIFEST_KEYS:
        if key not in raw:
            raise GenerationError(f"{MANIFEST} is missing required key {key!r}")

    if raw["pipeline"] != PIPELINE:
        raise GenerationError(
            f"{MANIFEST} pipeline is {raw['pipeline']!r}, expected {PIPELINE!r}"
        )

    summary = raw["summary"]
    if not isinstance(summary, dict):
        raise GenerationError(f"{MANIFEST} summary must hold a JSON object")
    for key in REQUIRED_SUMMARY_KEYS:
        if key not in summary:
            raise GenerationError(
                f"{MANIFEST} summary is missing required counter {key!r}"
            )
        if not isinstance(summary[key], int) or isinstance(summary[key], bool):
            raise GenerationError(
                f"{MANIFEST} summary counter {key!r} must be an integer"
            )

    if not isinstance(raw["source_row_count"], int) or isinstance(
        raw["source_row_count"], bool
    ):
        raise GenerationError(f"{MANIFEST} source_row_count must be an integer")

    return GenerationManifest(
        pipeline=str(raw["pipeline"]),
        split=str(raw["split"]),
        generation_id=str(raw["generation_id"]),
        generated_at=str(raw["generated_at"]),
        source_sha256=str(raw["source_sha256"]),
        source_row_count=int(raw["source_row_count"]),
        chunking=raw["chunking"],
        summary=summary,
        raw=raw,
    )


def manifest_sha256(generation_root: str | Path) -> str:
    """SHA-256 over the serialized manifest bytes.

    An index build records this so a reader can detect a manifest that changed
    after the build was validated.
    """
    path = Path(generation_root) / MANIFEST
    if not path.is_file():
        raise GenerationError(f"generation has no {MANIFEST}: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise GenerationError(
                    f"{path.name} line {number} is not valid JSON: {error}"
                ) from error
            if not isinstance(record, dict):
                raise GenerationError(f"{path.name} line {number} is not an object")
            yield record


def _read_qrels(path: Path) -> list[tuple[str, str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != ["query-id", "corpus-id", "score"]:
            raise GenerationError(
                f"{path.name} header is {header!r}, expected "
                "['query-id', 'corpus-id', 'score']"
            )
        rows = []
        for number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 3:
                raise GenerationError(f"{path.name} line {number} has {len(row)} columns, expected 3")
            rows.append((row[0], row[1], row[2]))
        return rows


def validate_generation(
    generation_root: str | Path,
    *,
    check_directory_identity: bool = True,
) -> GenerationManifest:
    """Validate one snapshot end to end, or raise `GenerationError`.

    Checks identity, then file presence, then the dataset contract: unique
    identities, one candidate set per retained query, every judgment inside
    that set, passage provenance and offsets, and the manifest counters
    against the records actually emitted.

    `check_directory_identity` is False only while a build is still staged
    under `generations/.building-<id>/`. Once published into
    `generations/<generation-id>/` the directory *is* the identity.
    """
    root = Path(generation_root)
    parsed = read_manifest(root)

    if parsed.split != PRODUCTION_SPLIT:
        raise GenerationError(
            f"generation split is {parsed.split!r}; this feature publishes "
            f"only {PRODUCTION_SPLIT!r}"
        )

    if check_directory_identity and parsed.generation_id != root.name:
        raise GenerationError(
            f"manifest generation_id {parsed.generation_id!r} does not match "
            f"its directory {root.name!r}"
        )

    for relative in required_artifacts(parsed.split):
        if not (root / relative).is_file():
            raise GenerationError(f"generation is missing {relative}")

    corpus = list(_read_jsonl(root / "beir" / "corpus.jsonl"))
    corpus_ids = [str(r["_id"]) for r in corpus]
    query_ids = [str(r["_id"]) for r in _read_jsonl(root / "beir" / "queries.jsonl")]
    documents = list(_read_jsonl(root / "documents" / "documents.jsonl"))
    qrels = _read_qrels(root / "beir" / "qrels" / f"{parsed.split}.tsv")
    candidates = list(
        _read_jsonl(root / "beir" / "candidates" / f"{parsed.split}.jsonl")
    )

    known_chunks = set(corpus_ids)
    known_queries = set(query_ids)
    _reject_duplicates(corpus_ids, "corpus")
    _reject_duplicates(query_ids, "query")

    # Passage provenance and offsets.
    chunk_document: dict[str, str] = {}
    for record in corpus:
        chunk_id = str(record["_id"])
        metadata = record.get("metadata") or {}
        document_id = str(metadata.get("document_id", ""))
        if not document_id:
            raise GenerationError(f"corpus chunk {chunk_id} has no document_id")
        chunk_document[chunk_id] = document_id
        start = metadata.get("start_offset")
        end = metadata.get("end_offset")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
            raise GenerationError(
                f"corpus chunk {chunk_id} offsets [{start}, {end}) are not a "
                "half-open interval into the cleaned document"
            )

    # Exactly one candidate record per retained query.
    by_query: dict[str, list[str]] = {}
    for record in candidates:
        query_id = str(record["query_id"])
        if query_id in by_query:
            raise GenerationError(f"query {query_id} has more than one candidate record")
        if query_id not in known_queries:
            raise GenerationError(f"candidate record names unknown query {query_id}")
        candidate_ids = [str(c) for c in record.get("candidate_ids") or []]
        if not candidate_ids:
            raise GenerationError(f"query {query_id} has an empty candidate set")
        document_id = str(record.get("document_id", ""))
        for chunk_id in candidate_ids:
            if chunk_id not in known_chunks:
                raise GenerationError(
                    f"query {query_id} lists unpublished candidate {chunk_id}"
                )
            if chunk_document[chunk_id] != document_id:
                raise GenerationError(
                    f"candidate {chunk_id} belongs to document "
                    f"{chunk_document[chunk_id]}, not {document_id} as query "
                    f"{query_id} claims"
                )
        by_query[query_id] = candidate_ids

    missing_candidates = known_queries - set(by_query)
    if missing_candidates:
        raise GenerationError(
            f"retained queries without a candidate record: {sorted(missing_candidates)}"
        )

    # Every judgment resolves, and lands inside its query's candidate set.
    judged: set[str] = set()
    for query_id, chunk_id, _score in qrels:
        if query_id not in known_queries:
            raise GenerationError(f"qrel references unknown query {query_id}")
        if chunk_id not in known_chunks:
            raise GenerationError(f"qrel references unknown chunk {chunk_id}")
        if chunk_id not in by_query[query_id]:
            raise GenerationError(
                f"qrel judges {chunk_id} for query {query_id}, which is not in "
                "its candidate set"
            )
        judged.add(query_id)

    unjudged = known_queries - judged
    if unjudged:
        raise GenerationError(
            f"retained queries with no qrel: {sorted(unjudged)}; each would "
            "score every model zero"
        )

    audits = {
        name: sum(1 for _ in _read_jsonl(root / "audits" / f"{name}.jsonl"))
        for name in ("rejected_sources", "dropped_queries", "unmapped_citations", "plain_text_violations")
    }
    reconciled = [
        ("chunks", len(corpus_ids), "beir/corpus.jsonl"),
        ("queries_retained", len(query_ids), "beir/queries.jsonl"),
        ("documents", len(documents), "documents/documents.jsonl"),
        ("qrels", len(qrels), f"beir/qrels/{parsed.split}.tsv"),
        *(
            (name, count, f"audits/{name}.jsonl")
            for name, count in audits.items()
            if name in parsed.summary
        ),
    ]
    for counter, observed, where in reconciled:
        declared = int(parsed.summary[counter])
        if declared != observed:
            raise GenerationError(
                f"summary.{counter} is {declared} but {where} holds {observed} records"
            )

    # Every source row and every question is accounted for: published or
    # audited, never silently lost.
    for counter, observed, accounting in (
        (
            "source_rows",
            len(documents) + audits["rejected_sources"],
            "documents plus rejected sources",
        ),
        (
            "queries",
            len(query_ids) + audits["dropped_queries"],
            "retained plus dropped queries",
        ),
    ):
        declared = int(parsed.summary[counter])
        if declared != observed:
            raise GenerationError(
                f"summary.{counter} is {declared} but {accounting} total {observed}"
            )

    return parsed


def _reject_duplicates(identifiers: list[str], what: str) -> None:
    seen: set[str] = set()
    for identifier in identifiers:
        if identifier in seen:
            raise GenerationError(f"duplicate {what} id {identifier}")
        seen.add(identifier)


def new_generation_id() -> str:
    """A sortable, unique identity for one published generation."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def resolve_current(dataset_root: str | Path) -> Path:
    """Resolve `current` once, to the immutable generation it points at.

    A consumer calls this a single time and keeps the returned path. Reading
    through `current` on every access would let a later publication swap the
    dataset underneath an in-flight run.
    """
    root = Path(dataset_root)
    pointer = root / CURRENT
    if not pointer.is_symlink():
        raise GenerationError(
            f"dataset root has no {CURRENT} reference: {pointer}"
            + (" (it exists but is not a symlink)" if pointer.exists() else "")
        )
    # One readlink: the target is fixed from here on, whatever happens next.
    target = Path(os.readlink(pointer))
    resolved = (root / target).resolve() if not target.is_absolute() else target.resolve()
    generations = (root / GENERATIONS).resolve()
    if resolved.parent != generations or resolved.name.startswith("."):
        raise GenerationError(
            f"{CURRENT} points at {target}, which is not a published generation "
            f"directly under {generations}"
        )
    if not resolved.is_dir():
        raise GenerationError(f"{CURRENT} points at a missing generation: {resolved}")
    return resolved


def pin_generation_path(path: str | Path) -> Path:
    """Pin a path inside a dataset to one validated, immutable generation.

    `path` may run through `current` (`<root>/current/beir`) or name a
    published generation directly (`<root>/generations/<id>/beir/...`). The
    `current` reference is resolved exactly once; the returned path never
    passes through it, so a later publication cannot change what it names.
    The generation is validated before the path is returned.
    """
    absolute = Path(os.path.abspath(path))
    for ancestor in (absolute, *absolute.parents):
        if ancestor.name == CURRENT and ancestor.is_symlink():
            generation = resolve_current(ancestor.parent)
            validate_generation(generation)
            return generation / absolute.relative_to(ancestor)
    resolved = absolute.resolve()
    for ancestor in (resolved, *resolved.parents):
        if ancestor.parent.name == GENERATIONS and (ancestor / MANIFEST).is_file():
            if ancestor.name.startswith("."):
                raise GenerationError(f"{ancestor} is an unpublished build")
            validate_generation(ancestor)
            return resolved
    raise GenerationError(
        f"{path} is not inside a published generation (expected "
        f"<root>/{CURRENT}/... or <root>/{GENERATIONS}/<generation-id>/...)"
    )


def load_generation(dataset_root: str | Path) -> tuple[Path, GenerationManifest]:
    """Resolve `current` once and validate the snapshot it names."""
    resolved = resolve_current(dataset_root)
    return resolved, validate_generation(resolved)
