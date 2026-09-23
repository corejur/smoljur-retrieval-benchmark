"""Chunking profile for v4, measured in o200k_base tokens.

Portuguese legal text runs about 1.9 o200k tokens per whitespace word - case
numbers and long compounds tokenize badly - so word-sized chunks overflow real
encoders: at a 300-word cap, 4.7% of chunks exceeded 512 tokens and the largest
reached 5,518. Sizing in tokens removes that whole class of silent truncation.

Cited passages measured in o200k tokens: p50 90, p90 217, p95 240, p99 336. A
512-token cap therefore holds 99.84% of citations whole while matching the
context limit of the usual retrieval encoders; the 320-token target keeps the
typical chunk well inside it.

The structural work is reused from the v1-v3 chunker, whose heading detection,
hard document boundaries, table handling and offset preservation all still
apply to v4 case files.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Sequence

from scripts.v4.headings import is_heading as v4_is_heading
from scripts.strategies.chunking.legal_recursive import (
    Chunk,
    ChunkingConfig,
    LengthCounter,
    WordCounter,
    chunk_document,
)

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "V4_CHUNKING",
    "TokenCounter",
    "o200k_counter",
    "ENCODING_NAME",
    "chunk_cleaned_document",
]

#: Units are o200k_base tokens, not words: always pair with o200k_counter().
V4_CHUNKING = ChunkingConfig(
    max_words=512,
    target_words=320,
    overlap_words=64,
    minimum_words=64,
    strategy_version="v4-o200k-1",
    merge_small_chunks=True,
)
ENCODING_NAME = "o200k_base"


class TokenCounter:
    """Count text length with a real tokenizer.

    The chunker measures candidate slices repeatedly while binary-searching for
    a split point, so counts are memoized; encoding dominates runtime
    otherwise.
    """

    def __init__(self, tokenizer: Any, cache_size: int = 200_000) -> None:
        self.tokenizer = tokenizer
        self._count = lru_cache(maxsize=cache_size)(self._encode_length)

    def _encode_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def count(self, text: str) -> int:
        return self._count(text)


@lru_cache(maxsize=4)
def o200k_counter() -> TokenCounter:
    """The v4 length counter: OpenAI o200k_base, as used for chunk sizing."""
    import tiktoken

    return TokenCounter(tiktoken.get_encoding(ENCODING_NAME))


def chunk_cleaned_document(
    document_id: str,
    text: str,
    *,
    config: ChunkingConfig = V4_CHUNKING,
    counter: LengthCounter | None = None,
) -> list[Chunk]:
    """Chunk cleaned text, preserving offsets into that same cleaned text."""
    return chunk_document(
        {"document_id": document_id, "text": text},
        counter or o200k_counter(),
        config,
        v4_is_heading,
    )
