from __future__ import annotations

import json

from scripts.citation_ground_truth import (
    citation_spans,
    generate_ground_truth,
    remove_untraceable_documents,
)
from scripts.corpus_generator import extract_documents
from scripts.strategies.chunking.legal_recursive import Chunk


def chunk(
    chunk_id: str, document_id: str, index: int, start: int, end: int, text: str
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=index,
        text=text[start:end],
        word_count=len(text[start:end].split()),
        start_offset=start,
        end_offset=end,
        section="",
        chunk_type="prose",
    )



def test_corpus_text_and_citation_spans_share_normalization() -> None:
    html = (
        "<h1>  FATOS </h1>"
        "<p id='p-1'>Texto   citado.\nCom continuação.</p>"
    )
    document_text = extract_documents([{"id": "doc", "data": html}])[0]["text"]
    citation_text, spans = citation_spans(html)

    assert document_text == citation_text
    assert spans["p-1"].text == "Texto citado.\nCom continuação."

def test_traces_answer_citation_to_overlapping_chunk() -> None:
    html = (
        '<p id="p-1">Context only.</p>'
        '<p id="p-2">The cited evidence.</p>'
    )
    document_text, spans = citation_spans(html)
    cited = spans["p-2"]
    chunks = [
        chunk("c-1", "doc", 0, 0, cited.start, document_text),
        chunk("c-2", "doc", 1, cited.start, len(document_text), document_text),
    ]
    source_rows = [
        {
            "id": "doc",
            "data": html,
            "output": json.dumps(
                [{"answer": "yes", "citations": ["p-2"]}]
            ),
        }
    ]
    query_rows = [
        {
            "source_row": "0",
            "source_id": "doc",
            "source_query_number": "1",
            "source_answer_index": "0",
            "query": "Question?",
            "citations": '["p-2"]',
        }
    ]

    result = generate_ground_truth(
        source_rows,
        query_rows,
        {"doc": document_text},
        {"doc": chunks},
    )

    assert json.loads(result.rows[0]["relevant_chunks"]) == ["c-2"]
    assert result.unmapped_citations == []


def test_audits_citation_id_absent_from_html() -> None:
    html = "<pre>Document without paragraph IDs.</pre>"
    document_text, _ = citation_spans(html)
    source_rows = [
        {
            "id": "doc",
            "data": html,
            "output": json.dumps(
                [{"answer": "yes", "citations": ["p-99"]}]
            ),
        }
    ]
    query_rows = [
        {
            "source_row": "0",
            "source_id": "doc",
            "source_query_number": "1",
            "source_answer_index": "0",
            "query": "Question?",
        }
    ]
    chunks = [chunk("c-1", "doc", 0, 0, len(document_text), document_text)]

    result = generate_ground_truth(
        source_rows,
        query_rows,
        {"doc": document_text},
        {"doc": chunks},
    )

    assert json.loads(result.rows[0]["relevant_chunks"]) == []
    assert result.unmapped_citations[0]["reason"] == "citation_element_not_found"


def test_removes_entire_document_when_one_query_has_no_relevant_chunk() -> None:
    ground_truth = generate_ground_truth(
        [
            {
                "id": "doc",
                "data": '<p id="p-1">Evidence.</p>',
                "output": json.dumps(
                    [
                        {"answer": "yes", "citations": ["p-1"]},
                        {"answer": "no", "citations": []},
                    ]
                ),
            }
        ],
        [
            {
                "source_row": "0",
                "source_id": "doc",
                "source_query_number": str(number),
                "source_answer_index": str(number - 1),
                "query": f"Question {number}?",
            }
            for number in (1, 2)
        ],
        {"doc": "Evidence."},
        {"doc": [chunk("c-1", "doc", 0, 0, 9, "Evidence.")]},
    )

    result = remove_untraceable_documents(
        ground_truth, [{"document_id": "doc", "text": "Evidence."}]
    )

    assert result.removed_document_ids == ("doc",)
    assert result.query_rows == []
    assert result.documents == []
