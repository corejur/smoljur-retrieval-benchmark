from __future__ import annotations

from scripts.v4.chunking import V4_CHUNKING, chunk_cleaned_document
from scripts.v4.ground_truth import map_citations_to_chunks, prepare_document
from scripts.v4.questions import QueryRecord


def _query(citations, index=0):
    return QueryRecord(
        document_id="doc",
        question_index=index,
        query_id=f"doc:q{index}",
        text="Pergunta?",
        citations=tuple(citations),
    )


def test_citation_maps_exactly_although_cleaning_changed_lengths() -> None:
    html = (
        '<p id="p-0">Veja www.exemplo.com para detalhes</p>'
        '<p id="p-1">O autor da acao e Joao da Silva</p>'
    )
    prepared = prepare_document(html, [_query(["p-1"])], document_id="doc")

    assert "https://www.exemplo.com" in prepared.text
    span = prepared.evidence[0].spans[0]
    assert prepared.text[span.start : span.end] == "O autor da acao e Joao da Silva"
    assert prepared.unmapped == ()


def test_unknown_citation_id_is_reported_not_guessed() -> None:
    html = '<p id="p-0">Texto unico do documento</p>'
    prepared = prepare_document(html, [_query(["p-99"])], document_id="doc")

    assert prepared.evidence[0].spans == ()
    assert [item.reason for item in prepared.unmapped] == ["element_not_found"]


def test_citation_covering_the_whole_document_is_rejected() -> None:
    # Short enough to pass the word cap, so this exercises the ratio guard.
    body = " ".join(["palavra"] * 100)
    html = f'<p id="p-0">ab</p><p id="p-1">{body}</p>'
    prepared = prepare_document(html, [_query(["p-1"])], document_id="doc")

    assert prepared.evidence[0].spans == ()
    assert [item.reason for item in prepared.unmapped] == ["covers_whole_document"]


def test_query_without_citations_yields_no_evidence_but_keeps_document() -> None:
    html = '<p id="p-0">Conteudo preservado do documento</p>'
    prepared = prepare_document(html, [_query([])], document_id="doc")

    assert prepared.text.strip()
    assert prepared.evidence[0].has_evidence is False
    assert prepared.evaluable == ()


def test_each_citation_maps_to_exactly_one_chunk() -> None:
    paragraphs = "".join(
        f'<p id="p-{index}">{" ".join([f"w{index}"] * 120)}</p>' for index in range(4)
    )
    prepared = prepare_document(paragraphs, [_query(["p-2"])], document_id="doc")
    chunks = chunk_cleaned_document("doc", prepared.text, config=V4_CHUNKING)
    records = [chunk.to_dict() for chunk in chunks]

    mapped = map_citations_to_chunks(prepared.evidence[0].spans, records)

    assert len(mapped) == 1
    assert mapped[0].citation_id == "p-2"
    chosen = next(c for c in chunks if c.chunk_id == mapped[0].chunk_id)
    span = prepared.evidence[0].spans[0]
    # The chosen chunk holds more of the span than any other chunk does.
    best = max(
        min(span.end, c.end_offset) - max(span.start, c.start_offset) for c in chunks
    )
    assert min(span.end, chosen.end_offset) - max(span.start, chosen.start_offset) == best


def test_overlapping_chunks_do_not_multiply_the_label() -> None:
    """Chunks overlap; a citation in the overlap must still label one chunk."""
    paragraphs = "".join(
        f'<p id="p-{index}">{" ".join([f"w{index}"] * 60)}</p>' for index in range(8)
    )
    prepared = prepare_document(
        paragraphs,
        [_query([f"p-{i}" for i in range(8)])],
        document_id="doc",
    )
    chunks = chunk_cleaned_document("doc", prepared.text, config=V4_CHUNKING)
    records = [chunk.to_dict() for chunk in chunks]

    mapped = map_citations_to_chunks(prepared.evidence[0].spans, records)

    assert len(mapped) == len(prepared.evidence[0].spans)
    assert all(entry.overlap > 0 for entry in mapped)


def test_mapping_is_deterministic_on_a_tie() -> None:
    paragraphs = "".join(
        f'<p id="p-{index}">{" ".join([f"w{index}"] * 90)}</p>' for index in range(5)
    )
    prepared = prepare_document(paragraphs, [_query(["p-3"])], document_id="doc")
    chunks = chunk_cleaned_document("doc", prepared.text, config=V4_CHUNKING)
    records = [chunk.to_dict() for chunk in chunks]

    first = map_citations_to_chunks(prepared.evidence[0].spans, records)
    again = map_citations_to_chunks(prepared.evidence[0].spans, list(reversed(records)))
    assert [e.chunk_id for e in first] == [e.chunk_id for e in again]


def test_citation_longer_than_the_cap_is_rejected() -> None:
    body = " ".join(["palavra"] * 350)
    html = f'<p id="p-0">{" ".join(["outro"] * 400)}</p><p id="p-1">{body}</p>'
    prepared = prepare_document(
        html, [_query(["p-1"])], document_id="doc", max_citation_words=300
    )

    assert prepared.evidence[0].spans == ()
    assert [item.reason for item in prepared.unmapped] == ["exceeds_max_words"]


def test_citation_at_the_cap_is_kept() -> None:
    body = " ".join(["palavra"] * 300)
    html = f'<p id="p-0">{" ".join(["outro"] * 400)}</p><p id="p-1">{body}</p>'
    prepared = prepare_document(
        html, [_query(["p-1"])], document_id="doc", max_citation_words=300
    )

    span = prepared.evidence[0].spans[0]
    assert span.word_count == 300
    assert prepared.unmapped == ()


def test_retained_citation_is_trackable_in_the_source_document() -> None:
    html = (
        '<p id="p-0">Veja www.exemplo.com para detalhes</p>'
        '<p id="p-1">O autor da acao e Joao da Silva</p>'
    )
    prepared = prepare_document(html, [_query(["p-1"])], document_id="doc")
    span = prepared.evidence[0].spans[0]

    from scripts.source_normalization import normalize_source_html

    source_text = normalize_source_html(html).text
    # The cleaned offsets moved because the URL grew; the source offsets did not.
    assert span.start != span.source_start
    assert source_text[span.source_start : span.source_end] == (
        "O autor da acao e Joao da Silva"
    )
    assert prepared.text[span.start : span.end] == "O autor da acao e Joao da Silva"


def test_undersized_drafts_are_merged_into_neighbours() -> None:
    from scripts.strategies.chunking.legal_recursive import (
        WordCounter,
        _DraftChunk,
        _merge_undersized,
    )

    text = "\n".join(f"linha {index} do formulario" for index in range(6))
    drafts, regions, cursor = [], [], 0
    for index in range(6):
        piece = f"linha {index} do formulario"
        drafts.append(
            _DraftChunk(piece, cursor, cursor + len(piece), "", "prose", 4)
        )
        regions.append(0)
        cursor += len(piece) + 1

    merged = _merge_undersized(
        text, drafts, regions, WordCounter(), V4_CHUNKING
    )

    assert len(merged) < len(drafts)
    assert all(d.word_count >= 4 for d in merged)
    # Offsets still address the real document text.
    for draft in merged:
        assert text[draft.start : draft.end].strip() == draft.text


def test_merging_never_crosses_a_region_boundary() -> None:
    from scripts.strategies.chunking.legal_recursive import (
        WordCounter,
        _DraftChunk,
        _merge_undersized,
    )

    text = "bloco um\nbloco dois"
    drafts = [
        _DraftChunk("bloco um", 0, 8, "", "prose", 2),
        _DraftChunk("bloco dois", 9, 19, "", "prose", 2),
    ]
    merged = _merge_undersized(
        text, drafts, [0, 1], WordCounter(), V4_CHUNKING
    )
    assert len(merged) == 2


def test_merging_preserves_offsets_and_covers_the_citations() -> None:
    from scripts.v4.chunking import V4_CHUNKING, chunk_cleaned_document

    paragraphs = "".join(
        f'<p id="p-{i}">Campo {i} com um pouco de texto do formulario</p>'
        for i in range(30)
    )
    prepared = prepare_document(paragraphs, [_query(["p-17"])], document_id="doc")
    chunks = chunk_cleaned_document("doc", prepared.text, config=V4_CHUNKING)

    for chunk in chunks:
        assert prepared.text[chunk.start_offset : chunk.end_offset].strip()
    span = prepared.evidence[0].spans[0]
    covering = [c for c in chunks if c.start_offset < span.end and span.start < c.end_offset]
    assert covering
    assert any("Campo 17" in c.text for c in covering)
