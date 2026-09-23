"""Contract tests for the generation manifest and one-time snapshot resolution.

`meta/manifest.json` is the *generation manifest* — the published dataset's
identity, counts, and configuration. It is not `scripts/v4/manifest.py`, which
is the module registry. See the plan's Structure Decision naming note.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.v4.generation import (
    PIPELINE,
    GenerationError,
    load_generation,
    read_manifest,
    required_artifacts,
    resolve_current,
    validate_generation,
)

GENERATION_ID = "20260922-120000-abcdef"
SOURCE_SHA = "d10d21f2074e48576cb715bc65ef01a36f369bf837dd2c404280775cef56fff3"


def _summary(**overrides):
    base = {
        "source_rows": 1,
        "rejected_sources": 0,
        "documents": 1,
        "chunks": 2,
        "queries": 1,
        "queries_retained": 1,
        "qrels": 1,
        "unmapped_citations": 0,
        "plain_text_violations": 0,
    }
    base.update(overrides)
    return base


def _manifest(**overrides):
    base = {
        "pipeline": PIPELINE,
        "split": "test",
        "generation_id": GENERATION_ID,
        "generated_at": "2026-09-22T12:00:00+00:00",
        "source_sha256": SOURCE_SHA,
        "source_row_count": 1,
        "chunking": {"max_words": 512, "target_words": 320, "overlap_words": 64},
        "length_unit": "o200k_base_tokens",
        "length_unit_encoding": "o200k_base",
        "relevance": "one_chunk_per_citation_by_maximum_overlap",
        "citation_filter": "trackable_and_within_max_citation_words",
        "corpus_text": "plain_text_verified_before_chunking",
        "max_citation_words": 512,
        "max_gold_chunks": 10,
        "document_filter": "requires_at_least_one_retained_query",
        "query_filter": "answer_text_and_resolved_citation",
        "summary": _summary(),
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _chunk(index, document_id="doc-a"):
    return {
        "_id": f"{document_id}:c{index}",
        "title": "",
        "text": f"texto do trecho {index}",
        "metadata": {
            "document_id": document_id,
            "chunk_index": index,
            "start_offset": index * 10,
            "end_offset": index * 10 + 9,
            "chunk_type": "paragraph",
        },
    }


def _build_generation(root: Path, *, split="test", manifest=None, qrels=None):
    """Write one structurally complete generation under `root`."""
    generation = root / "generations" / GENERATION_ID
    _write_jsonl(generation / "beir" / "corpus.jsonl", [_chunk(0), _chunk(1)])
    _write_jsonl(
        generation / "beir" / "queries.jsonl",
        [{"_id": "doc-a:q0", "text": "Quem e o autor?", "reference_answer": "Joao", "citations": ["p-0"]}],
    )
    rows = qrels if qrels is not None else [("doc-a:q0", "doc-a:c0", 1)]
    (generation / "beir" / "qrels").mkdir(parents=True, exist_ok=True)
    (generation / "beir" / "qrels" / f"{split}.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        + "".join(f"{q}\t{c}\t{s}\n" for q, c, s in rows),
        encoding="utf-8",
    )
    _write_jsonl(
        generation / "beir" / "candidates" / f"{split}.jsonl",
        [{"query_id": "doc-a:q0", "document_id": "doc-a", "candidate_ids": ["doc-a:c0", "doc-a:c1"]}],
    )
    _write_jsonl(generation / "documents" / "documents.jsonl", [{"document_id": "doc-a", "text": "texto"}])
    _write_jsonl(
        generation / "evidence" / "evidence.jsonl",
        [{"query_id": "doc-a:q0", "document_id": "doc-a", "spans": []}],
    )
    for audit in (
        "rejected_sources",
        "dropped_queries",
        "unmapped_citations",
        "cleaning_events",
        "plain_text_violations",
    ):
        _write_jsonl(generation / "audits" / f"{audit}.jsonl", [])
    (generation / "meta").mkdir(parents=True, exist_ok=True)
    (generation / "meta" / "manifest.json").write_text(
        json.dumps(manifest if manifest is not None else _manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "current").symlink_to(Path("generations") / GENERATION_ID)
    return generation


# --- manifest schema -------------------------------------------------------


def test_reads_a_complete_generation_manifest(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path)
    parsed = read_manifest(generation)
    assert parsed.pipeline == "redator-v4"
    assert parsed.split == "test"
    assert parsed.generation_id == GENERATION_ID
    assert parsed.source_sha256 == SOURCE_SHA
    assert parsed.source_row_count == 1
    # summary.chunks is the completeness contract for an index build.
    assert parsed.chunks == 2
    assert parsed.summary["queries_retained"] == 1


@pytest.mark.parametrize(
    "missing",
    ["pipeline", "split", "generation_id", "generated_at", "source_sha256", "source_row_count", "chunking", "summary"],
)
def test_rejects_a_manifest_missing_a_required_key(tmp_path: Path, missing: str) -> None:
    manifest = _manifest()
    del manifest[missing]
    generation = _build_generation(tmp_path, manifest=manifest)
    with pytest.raises(GenerationError, match=missing):
        read_manifest(generation)


def test_rejects_a_summary_missing_the_chunks_counter(tmp_path: Path) -> None:
    summary = _summary()
    del summary["chunks"]
    generation = _build_generation(tmp_path, manifest=_manifest(summary=summary))
    with pytest.raises(GenerationError, match="chunks"):
        read_manifest(generation)


def test_rejects_a_foreign_pipeline_label(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path, manifest=_manifest(pipeline="redator-v3"))
    with pytest.raises(GenerationError, match="redator-v4"):
        read_manifest(generation)


def test_rejects_a_non_test_split(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path, manifest=_manifest(split="train"))
    with pytest.raises(GenerationError, match="split"):
        validate_generation(generation)


# --- required artifacts ----------------------------------------------------


def test_required_artifacts_cover_the_published_contract() -> None:
    names = required_artifacts("test")
    for expected in (
        "beir/corpus.jsonl",
        "beir/queries.jsonl",
        "beir/qrels/test.tsv",
        "beir/candidates/test.jsonl",
        "documents/documents.jsonl",
        "evidence/evidence.jsonl",
        "audits/rejected_sources.jsonl",
        "audits/dropped_queries.jsonl",
        "audits/unmapped_citations.jsonl",
        "audits/cleaning_events.jsonl",
        "audits/plain_text_violations.jsonl",
        "meta/manifest.json",
    ):
        assert expected in names


def test_rejects_a_generation_missing_a_required_artifact(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path)
    (generation / "evidence" / "evidence.jsonl").unlink()
    with pytest.raises(GenerationError, match="evidence/evidence.jsonl"):
        validate_generation(generation)


# --- cross-file references and counts --------------------------------------


def test_accepts_an_internally_consistent_generation(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path)
    assert validate_generation(generation).generation_id == GENERATION_ID


def test_rejects_a_qrel_referencing_an_unknown_query(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path, qrels=[("doc-a:q9", "doc-a:c0", 1)])
    with pytest.raises(GenerationError, match="doc-a:q9"):
        validate_generation(generation)


def test_rejects_a_qrel_referencing_an_unknown_chunk(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path, qrels=[("doc-a:q0", "doc-a:c9", 1)])
    with pytest.raises(GenerationError, match="doc-a:c9"):
        validate_generation(generation)


def test_rejects_summary_counts_that_disagree_with_emitted_records(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path, manifest=_manifest(summary=_summary(chunks=99)))
    with pytest.raises(GenerationError, match="chunks"):
        validate_generation(generation)


def test_rejects_a_manifest_whose_generation_id_is_not_its_directory(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path, manifest=_manifest(generation_id="20260101-000000-other"))
    with pytest.raises(GenerationError, match="generation_id"):
        validate_generation(generation)


# --- single resolved snapshot ----------------------------------------------


def test_resolves_current_to_one_immutable_snapshot(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path)
    assert resolve_current(tmp_path).resolve() == generation.resolve()


def test_rejects_a_dataset_root_without_a_current_reference(tmp_path: Path) -> None:
    _build_generation(tmp_path)
    (tmp_path / "current").unlink()
    with pytest.raises(GenerationError, match="current"):
        resolve_current(tmp_path)


def test_load_generation_resolves_and_validates_once(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path)
    root, parsed = load_generation(tmp_path)
    assert root.resolve() == generation.resolve()
    assert parsed.chunks == 2


# --- T009: the dataset contract --------------------------------------------


def _rewrite(path: Path, records):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def test_rejects_duplicate_corpus_ids(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path)
    _rewrite(generation / "beir" / "corpus.jsonl", [_chunk(0), _chunk(0)])
    with pytest.raises(GenerationError, match="duplicate corpus id"):
        validate_generation(generation)


def test_rejects_duplicate_query_ids(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path)
    query = {"_id": "doc-a:q0", "text": "x", "reference_answer": "y", "citations": []}
    _rewrite(generation / "beir" / "queries.jsonl", [query, dict(query)])
    with pytest.raises(GenerationError, match="duplicate query id"):
        validate_generation(generation)


def test_rejects_a_chunk_whose_offsets_are_not_half_open(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path)
    broken = _chunk(1)
    broken["metadata"]["end_offset"] = broken["metadata"]["start_offset"]
    _rewrite(generation / "beir" / "corpus.jsonl", [_chunk(0), broken])
    with pytest.raises(GenerationError, match="half-open"):
        validate_generation(generation)


def test_rejects_a_retained_query_without_a_candidate_record(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path)
    _rewrite(generation / "beir" / "candidates" / "test.jsonl", [])
    with pytest.raises(GenerationError, match="without a candidate record"):
        validate_generation(generation)


def test_rejects_two_candidate_records_for_one_query(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path)
    record = {"query_id": "doc-a:q0", "document_id": "doc-a", "candidate_ids": ["doc-a:c0"]}
    _rewrite(generation / "beir" / "candidates" / "test.jsonl", [record, dict(record)])
    with pytest.raises(GenerationError, match="more than one candidate record"):
        validate_generation(generation)


def test_rejects_an_empty_candidate_set(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path)
    _rewrite(
        generation / "beir" / "candidates" / "test.jsonl",
        [{"query_id": "doc-a:q0", "document_id": "doc-a", "candidate_ids": []}],
    )
    with pytest.raises(GenerationError, match="empty candidate set"):
        validate_generation(generation)


def test_rejects_a_candidate_from_another_document(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path)
    foreign = _chunk(0, document_id="doc-b")
    _rewrite(generation / "beir" / "corpus.jsonl", [_chunk(0), _chunk(1), foreign])
    _rewrite(
        generation / "beir" / "candidates" / "test.jsonl",
        [{"query_id": "doc-a:q0", "document_id": "doc-a", "candidate_ids": ["doc-a:c0", "doc-b:c0"]}],
    )
    with pytest.raises(GenerationError, match="belongs to document doc-b"):
        validate_generation(generation)


def test_rejects_a_judgment_outside_its_candidate_set(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path, qrels=[("doc-a:q0", "doc-a:c1", 1)])
    _rewrite(
        generation / "beir" / "candidates" / "test.jsonl",
        [{"query_id": "doc-a:q0", "document_id": "doc-a", "candidate_ids": ["doc-a:c0"]}],
    )
    with pytest.raises(GenerationError, match="not in\n?.*its candidate set|candidate set"):
        validate_generation(generation)


def test_rejects_a_retained_query_with_no_judgment(tmp_path: Path) -> None:
    generation = _build_generation(tmp_path, qrels=[])
    manifest = _manifest(summary=_summary(qrels=0))
    (generation / "meta" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(GenerationError, match="no qrel"):
        validate_generation(generation)


def test_a_staged_build_can_skip_the_directory_identity_check(tmp_path: Path) -> None:
    """Before T017 publishes into generations/<id>/, the directory is staging."""
    generation = _build_generation(tmp_path)
    staged = tmp_path / "staging"
    generation.rename(staged)
    with pytest.raises(GenerationError, match="generation_id"):
        validate_generation(staged)
    assert validate_generation(staged, check_directory_identity=False).chunks == 2
