from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.chunk_corpus import write_chunk_csvs
from scripts.strategies.chunking.legal_recursive import (
    ChunkingConfig,
    chunk_document,
    chunk_documents,
)


class CharacterLengthCounter:
    """Deterministic counter test double for forcing recursive splits."""

    def count(self, text: str) -> int:
        return len(text)


def config(**overrides: int) -> ChunkingConfig:
    values = {
        "max_words": 64,
        "target_words": 48,
        "overlap_words": 8,
        "minimum_words": 8,
    }
    values.update(overrides)
    return ChunkingConfig(**values)


def test_long_single_line_uses_word_fallback_and_preserves_limit() -> None:
    text = " ".join(f"word{i}" for i in range(80))
    chunks = chunk_document(
        {"document_id": "doc-1", "text": text},
        CharacterLengthCounter(),
        config(),
    )

    assert len(chunks) > 1
    assert all(chunk.word_count <= 64 for chunk in chunks)
    assert all(text[chunk.start_offset : chunk.end_offset] == chunk.text for chunk in chunks)
    assert any(
        chunks[index].start_offset < chunks[index - 1].end_offset
        for index in range(1, len(chunks))
    )


def test_headings_create_sections_without_crossing_them() -> None:
    text = "I – DOS FATOS\nFato curto.\nII – DO DIREITO\nFundamento curto."
    chunks = chunk_document(
        {"document_id": "doc-2", "text": text},
        CharacterLengthCounter(),
        config(max_words=128, target_words=100),
    )

    assert len(chunks) == 1
    assert chunks[0].section == "I – DOS FATOS | II – DO DIREITO"
    assert "I – DOS FATOS" in chunks[0].text
    assert "II – DO DIREITO" in chunks[0].text


def test_table_chunks_repeat_header_and_preserve_rows() -> None:
    rows = "\n".join(f"| {index} | {'x' * 20} |" for index in range(10))
    text = f"| Item | Valor |\n| --- | --- |\n{rows}"
    chunks = chunk_document(
        {"document_id": "doc-table", "text": text},
        CharacterLengthCounter(),
        config(max_words=100, target_words=80),
    )

    assert len(chunks) > 1
    assert all(chunk.chunk_type == "table" for chunk in chunks)
    assert all(chunk.text.startswith("| Item | Valor |\n| --- | --- |") for chunk in chunks)
    assert all(chunk.word_count <= 100 for chunk in chunks)


def test_oversized_header_only_table_respects_hard_limit() -> None:
    text = "| " + " ".join(f"value{index}" for index in range(80)) + " |"

    chunks = chunk_document(
        {"document_id": "doc-header-only-table", "text": text},
        config=config(max_words=20, target_words=15),
    )

    assert len(chunks) > 1
    assert all(chunk.chunk_type == "table" for chunk in chunks)
    assert all(chunk.word_count <= 20 for chunk in chunks)


def test_short_prose_tail_is_merged_with_preceding_table_context() -> None:
    table = (
        "| Processo | Distribuição |\n"
        "| --- | --- |\n"
        "| 123 | 04/09/2025 |"
    )
    text = f"{table}\nEmitida em: 12/09/2025"
    chunks = chunk_document(
        {"document_id": "doc-mixed-tail", "text": text},
        CharacterLengthCounter(),
        config(max_words=128, target_words=100, minimum_words=30),
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "mixed"
    assert chunks[0].text == text


def test_page_separator_is_removed_and_is_a_hard_boundary() -> None:
    text = "Primeiro documento.\nPÁGINA DE SEPARAÇÃO\nSegundo documento."
    chunks = chunk_document(
        {"document_id": "doc-3", "text": text},
        CharacterLengthCounter(),
        config(max_words=128, target_words=100),
    )

    assert len(chunks) == 2
    assert all("PÁGINA DE SEPARAÇÃO" not in chunk.text for chunk in chunks)


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap_words"):
        chunk_documents(
            [{"document_id": "doc", "text": "text"}],
            CharacterLengthCounter(),
            ChunkingConfig(target_words=10, overlap_words=10),
        )


def test_writes_one_plain_text_csv_per_document(tmp_path: Path) -> None:
    chunks = chunk_documents(
        [
            {"document_id": "doc-a", "text": "First document."},
            {"document_id": "doc-b", "text": "Second document."},
        ],
        CharacterLengthCounter(),
        config(max_words=128, target_words=100),
    )

    assert write_chunk_csvs(chunks, tmp_path) == 2
    with (tmp_path / "doc-a.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["document_id"] == "doc-a"
    assert rows[0]["text"] == "First document."
