"""Readers see one complete immutable generation; a failed rebuild changes nothing.

User Story 3. Each build lands in its own `generations/<generation-id>/`
directory and becomes visible only through one atomic switch of the `current`
reference. The state machine is building -> validated -> active -> superseded,
with failed builds never becoming active.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from scripts.v4.generation import (
    GenerationError,
    pin_generation_path,
    required_artifacts,
    resolve_current,
    validate_generation,
)
from scripts.v4.publication import PublicationError, activate_generation, publish
from scripts.v4.run import build_dataset, run

_PARAGRAPHS = (
    '<p id="p-0">O autor da acao e Joao da Silva residente na capital</p>'
    '<p id="p-1">O valor da causa foi fixado pela parte interessada</p>'
    '<p id="p-2">Terceiro paragrafo com texto suficiente para o documento</p>'
)


def _row(document_id, citation="p-0"):
    return {
        "id": document_id,
        "question_texts": json.dumps(["Quem e o autor?"]),
        "output": json.dumps([{"answer": "Joao", "citations": [citation]}]),
        "data": _PARAGRAPHS,
    }


def _builder(*document_ids):
    def build(staging: Path, generation_id: str):
        return build_dataset(
            [_row(d) for d in document_ids], staging, generation_id=generation_id
        )

    return build


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _documents(generation: Path) -> list[str]:
    lines = (generation / "documents" / "documents.jsonl").read_text().splitlines()
    return [json.loads(line)["document_id"] for line in lines]


# --- building -> validated -> active -----------------------------------------


def test_a_publication_lands_in_its_own_generation_directory(tmp_path: Path) -> None:
    published = publish(tmp_path, _builder("doc-a"))

    generation = tmp_path / "generations" / published.generation_id
    assert published.path == generation
    assert resolve_current(tmp_path) == generation.resolve()
    # The directory name *is* the identity once published.
    validate_generation(generation)
    assert published.previous_generation_id is None


def test_current_is_a_relative_symlink_so_the_root_can_move(tmp_path: Path) -> None:
    published = publish(tmp_path, _builder("doc-a"))

    assert (tmp_path / "current").is_symlink()
    assert os.readlink(tmp_path / "current") == os.path.join(
        "generations", published.generation_id
    )


def test_no_managed_directory_is_written_at_the_root(tmp_path: Path) -> None:
    publish(tmp_path, _builder("doc-a"))

    assert sorted(p.name for p in tmp_path.iterdir()) == ["current", "generations"]


def test_run_publishes_an_immutable_generation(tmp_path: Path) -> None:
    published = run([_row("doc-a")], tmp_path)

    generation = resolve_current(tmp_path)
    assert generation.name == published.generation_id
    assert published.summary.documents == 1
    for relative in required_artifacts("test"):
        assert (generation / relative).is_file()


# --- active -> superseded -----------------------------------------------------


def test_a_successful_rebuild_supersedes_without_touching_the_old_one(tmp_path: Path) -> None:
    first = publish(tmp_path, _builder("doc-a"))
    before = _snapshot(first.path)

    second = publish(tmp_path, _builder("doc-b"))

    assert second.previous_generation_id == first.generation_id
    assert resolve_current(tmp_path) == second.path.resolve()
    assert _documents(second.path) == ["doc-b"]
    # Superseded, not removed: kept until a cleanup policy exists.
    assert _snapshot(first.path) == before


def test_rollback_reactivates_the_previous_generation(tmp_path: Path) -> None:
    first = publish(tmp_path, _builder("doc-a"))
    publish(tmp_path, _builder("doc-b"))

    activate_generation(tmp_path, first.generation_id)

    assert resolve_current(tmp_path) == first.path.resolve()


def test_rollback_refuses_an_unknown_or_invalid_generation(tmp_path: Path) -> None:
    first = publish(tmp_path, _builder("doc-a"))
    manifest = first.path / "meta" / "manifest.json"
    publish(tmp_path, _builder("doc-b"))
    current_before = os.readlink(tmp_path / "current")

    with pytest.raises(PublicationError, match="no-such-id"):
        activate_generation(tmp_path, "no-such-id")
    manifest.write_text("{}")
    with pytest.raises(GenerationError):
        activate_generation(tmp_path, first.generation_id)

    assert os.readlink(tmp_path / "current") == current_before


# --- failed (never active) ----------------------------------------------------


def _assert_unchanged(root: Path, active: Path, before: dict[str, bytes]) -> None:
    assert resolve_current(root) == active.resolve()
    assert _snapshot(active) == before
    # A failed build leaves neither a staged nor a published directory behind.
    assert [p.name for p in (root / "generations").iterdir()] == [active.name]


def test_a_failing_build_leaves_the_active_generation_in_place(tmp_path: Path) -> None:
    active = publish(tmp_path, _builder("doc-a")).path
    before = _snapshot(active)

    def explode(staging: Path, generation_id: str):
        build_dataset([_row("doc-b")], staging, generation_id=generation_id)
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        publish(tmp_path, explode)

    _assert_unchanged(tmp_path, active, before)


def test_an_invalid_build_is_never_activated(tmp_path: Path) -> None:
    active = publish(tmp_path, _builder("doc-a")).path
    before = _snapshot(active)

    def inconsistent(staging: Path, generation_id: str):
        summary = build_dataset([_row("doc-b")], staging, generation_id=generation_id)
        (staging / "beir" / "queries.jsonl").write_text("")
        return summary

    with pytest.raises(GenerationError):
        publish(tmp_path, inconsistent)

    _assert_unchanged(tmp_path, active, before)


def test_a_build_that_writes_another_generation_id_is_refused(tmp_path: Path) -> None:
    active = publish(tmp_path, _builder("doc-a")).path
    before = _snapshot(active)

    def wrong_identity(staging: Path, generation_id: str):
        return build_dataset([_row("doc-b")], staging, generation_id="someone-else")

    with pytest.raises(GenerationError, match="someone-else"):
        publish(tmp_path, wrong_identity)

    _assert_unchanged(tmp_path, active, before)


def test_a_failed_switch_leaves_the_active_generation_in_place(
    tmp_path: Path, monkeypatch
) -> None:
    active = publish(tmp_path, _builder("doc-a")).path
    before = _snapshot(active)

    real_replace = os.replace

    def refuse_current(source, destination):
        if Path(destination).name == "current":
            raise OSError("switch failed")
        return real_replace(source, destination)

    monkeypatch.setattr("scripts.v4.publication.os.replace", refuse_current)

    with pytest.raises(OSError, match="switch failed"):
        publish(tmp_path, _builder("doc-b"))

    _assert_unchanged(tmp_path, active, before)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["current", "generations"]


def test_run_rolls_back_a_failed_build(tmp_path: Path, monkeypatch) -> None:
    import scripts.v4.run as run_module

    first = run([_row("doc-a")], tmp_path)
    before = _snapshot(first.path)

    def explode(*_args, **_kwargs):
        raise RuntimeError("chunker failed")

    monkeypatch.setattr(run_module, "chunk_cleaned_document", explode)
    with pytest.raises(RuntimeError):
        run([_row("doc-b")], tmp_path)

    _assert_unchanged(tmp_path, first.path, before)


def test_a_non_symlink_current_is_refused_rather_than_overwritten(tmp_path: Path) -> None:
    (tmp_path / "current").mkdir()

    with pytest.raises(PublicationError, match="current"):
        publish(tmp_path, _builder("doc-a"))

    assert (tmp_path / "current").is_dir()


# --- readers never see a mixture ---------------------------------------------


def test_concurrent_readers_never_see_a_mixed_generation(tmp_path: Path) -> None:
    publish(tmp_path, _builder("doc-0"))
    errors: list[str] = []
    stop = threading.Event()

    def read() -> None:
        while not stop.is_set():
            generation = resolve_current(tmp_path)
            manifest = json.loads((generation / "meta" / "manifest.json").read_text())
            if manifest["generation_id"] != generation.name:
                errors.append(f"manifest {manifest['generation_id']} in {generation.name}")
            documents = _documents(generation)
            corpus = {
                json.loads(line)["metadata"]["document_id"]
                for line in (generation / "beir" / "corpus.jsonl").read_text().splitlines()
            }
            if set(documents) != corpus or len(documents) != 1:
                errors.append(f"{generation.name}: documents {documents} vs corpus {corpus}")

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for index in range(1, 6):
            publish(tmp_path, _builder(f"doc-{index}"))
    finally:
        stop.set()
        reader.join()

    assert errors == []
    assert _documents(resolve_current(tmp_path)) == ["doc-5"]


# --- consumers pin current once (T019) ---------------------------------------


def test_resolve_current_refuses_a_pointer_outside_generations(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "current").symlink_to("elsewhere")

    with pytest.raises(GenerationError, match="generations"):
        resolve_current(tmp_path)


def test_resolve_current_refuses_a_dangling_pointer(tmp_path: Path) -> None:
    (tmp_path / "generations").mkdir()
    (tmp_path / "current").symlink_to(os.path.join("generations", "gone"))

    with pytest.raises(GenerationError, match="current"):
        resolve_current(tmp_path)


def test_a_path_through_current_is_pinned_to_the_immutable_generation(tmp_path: Path) -> None:
    first = publish(tmp_path, _builder("doc-a"))

    pinned = pin_generation_path(tmp_path / "current" / "beir")
    publish(tmp_path, _builder("doc-b"))

    # The pin was taken before the switch: it still names the first snapshot.
    assert pinned == first.path.resolve() / "beir"
    assert "current" not in pinned.parts


def test_pinning_rejects_a_generation_with_missing_files(tmp_path: Path) -> None:
    published = publish(tmp_path, _builder("doc-a"))
    (published.path / "beir" / "candidates" / "test.jsonl").unlink()

    with pytest.raises(GenerationError, match="candidates"):
        pin_generation_path(tmp_path / "current" / "beir")


def test_pinning_a_published_generation_path_directly_validates_it(tmp_path: Path) -> None:
    published = publish(tmp_path, _builder("doc-a"))

    assert pin_generation_path(published.path / "beir" / "qrels" / "test.tsv") == (
        published.path.resolve() / "beir" / "qrels" / "test.tsv"
    )
