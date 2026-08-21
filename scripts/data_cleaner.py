"""Clean document JSON and write one JSON audit file per operation."""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from scripts.persistence import write_csv, write_json

DEFAULT_DOCUMENTS = Path("data/documents/documents.json")
DEFAULT_QUERIES = Path("data/queries/representative-redator-2k-v1-v4.csv")
DEFAULT_OUTPUT = Path("data/documents/documents.cleaned.json")
DEFAULT_AUDIT_DIRECTORY = Path("data/data_preparation_output")

Document = dict[str, Any]
_DATA_IMAGE = re.compile(
    r"data:image/[\w.+-]+;base64,(?P<payload>[A-Za-z0-9+/\s]+={0,2})",
    re.IGNORECASE,
)
_BASE64 = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_IMAGE_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a",
    b"BM", b"II*\x00", b"MM\x00*",
)
_PROMPT = re.compile(
    r"\A\s*(?:\*{1,2}|_{1,2})?\s*analise\s+(?:o|os|a|as)\s+documentos?\s+"
    r"e\s+responda\s+(?:a|à|as|às)\s+perguntas?\s*[.:]?\s*"
    r"(?:\*{1,2}|_{1,2})?\s*(?:\r?\n)+",
    re.IGNORECASE,
)
_URL_ADDRESS = r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?::\d+)?(?:/[^\s<>\\\"']*)?(?:\?[^\s<>\\\"']*)?"
_SPACED_SCHEME = re.compile(
    rf"\b(?P<scheme>https?)(?:\s+:\s*/\s*/\s*|\s*:\s+/\s*/\s*|\s*:\s*/\s+/\s*)"
    rf"(?P<address>{_URL_ADDRESS})",
    re.IGNORECASE,
)
_SPACED_WWW = re.compile(
    rf"(?<![\w/@.-])www(?:\s+\.\s*|\s*\.\s+)(?P<address>{_URL_ADDRESS})",
    re.IGNORECASE,
)
_BARE_WWW = re.compile(
    rf"(?<![\w/@.-])www\.(?P<address>{_URL_ADDRESS})", re.IGNORECASE
)


def _copy_documents(documents: Sequence[Mapping[str, Any]]) -> list[Document]:
    copied = []
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise TypeError(f"document at index {index} is not an object")
        if "document_id" not in document or "text" not in document:
            raise ValueError(f"document at index {index} needs document_id and text")
        if not isinstance(document["text"], str):
            raise TypeError(f"text at index {index} is not a string")
        copied.append(dict(document))
    return copied


def _audit(name: str, entries: list[dict[str, Any]], directory: str | Path) -> None:
    write_json(Path(directory) / f"{name}.json", entries)

def _is_encoded_image(value: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    if len(compact) < 128 or len(compact) % 4 or not _BASE64.fullmatch(compact):
        return False
    try:
        prefix = base64.b64decode(compact[: min(128, len(compact))], validate=True)
    except (binascii.Error, ValueError):
        return False
    return any(prefix.startswith(item) for item in _IMAGE_SIGNATURES) or (
        prefix.startswith(b"RIFF") and b"WEBP" in prefix[:16]
    )


def remove_encode_images(
    documents: Sequence[Mapping[str, Any]],
    output_directory: str | Path = DEFAULT_AUDIT_DIRECTORY,
) -> list[Document]:
    """Remove embedded and stand-alone Base64-encoded images from text."""
    cleaned, entries = _copy_documents(documents), []
    for document in cleaned:
        original, count = document["text"], 0

        def remove(match: re.Match[str]) -> str:
            nonlocal count
            if not _is_encoded_image(match.group("payload")):
                return match.group(0)
            count += 1
            return ""

        text = _DATA_IMAGE.sub(remove, original)
        if _is_encoded_image(text.strip()):
            count += 1
            text = ""
        if text != original:
            document["text"] = text
            entries.append({
                "document_id": document["document_id"],
                "action": "removed_base64_encoded_image",
                "image_count": count,
                "original_length": len(original),
                "cleaned_length": len(text),
                "removed_characters": len(original) - len(text),
            })
    _audit("remove_encode_images", entries, output_directory)
    return cleaned


def remove_unicode(
    documents: Sequence[Mapping[str, Any]],
    output_directory: str | Path = DEFAULT_AUDIT_DIRECTORY,
) -> list[Document]:
    """Remove Unicode replacement characters (U+FFFD)."""
    cleaned, entries = _copy_documents(documents), []
    for document in cleaned:
        count = document["text"].count("\ufffd")
        if count:
            document["text"] = document["text"].replace("\ufffd", "")
            entries.append({
                "document_id": document["document_id"],
                "action": "removed_unicode_replacement_characters",
                "replacement_character_count": count,
            })
    _audit("remove_unicode", entries, output_directory)
    return cleaned


def correct_url(
    documents: Sequence[Mapping[str, Any]],
    output_directory: str | Path = DEFAULT_AUDIT_DIRECTORY,
) -> list[Document]:
    """Fix spaced URL components and add HTTPS to bare www addresses."""
    cleaned, entries = _copy_documents(documents), []
    for document in cleaned:
        original = document["text"]
        corrections: list[dict[str, str]] = []

        def replace_spaced_scheme(match: re.Match[str]) -> str:
            after = f"{match.group('scheme').lower()}://{match.group('address')}"
            corrections.append({"before": match.group(0), "after": after})
            return after

        def replace_spaced_www(match: re.Match[str]) -> str:
            after = f"https://www.{match.group('address')}"
            corrections.append({"before": match.group(0), "after": after})
            return after

        def replace_bare_www(match: re.Match[str]) -> str:
            after = f"https://www.{match.group('address')}"
            corrections.append({"before": match.group(0), "after": after})
            return after

        text, scheme_count = _SPACED_SCHEME.subn(replace_spaced_scheme, original)
        text, spaced_www_count = _SPACED_WWW.subn(replace_spaced_www, text)
        text, bare_count = _BARE_WWW.subn(replace_bare_www, text)
        if text != original:
            document["text"] = text
            entries.append({
                "document_id": document["document_id"],
                "action": "corrected_malformed_urls",
                "correction_count": len(corrections),
                "spaced_scheme_count": scheme_count,
                "spaced_www_count": spaced_www_count,
                "added_https_scheme_count": bare_count,
                "url_corrections": corrections,
            })
    _audit("correct_url", entries, output_directory)
    return cleaned


def check_empty_documents(
    documents: Sequence[Mapping[str, Any]],
    output_directory: str | Path = DEFAULT_AUDIT_DIRECTORY,
    *,
    queries: str | Path | Iterable[Mapping[str, Any]] | None = None,
    queries_output: str | Path | None = None,
    document_id_column: str = "source_id",
) -> list[Document]:
    """Remove empty documents and, when requested, their associated queries."""
    kept, entries = [], []
    removed_ids: set[str] = set()
    for index, document in enumerate(_copy_documents(documents)):
        if document["text"].strip():
            kept.append(document)
        else:
            document_id = str(document["document_id"])
            removed_ids.add(document_id)
            entries.append({
                "document_id": document_id,
                "action": "removed_empty_document",
                "original_index": index,
                "original_length": len(document["text"]),
                "removed_query_count": 0,
                "removed_queries": [],
            })
    if queries is not None:
        if queries_output is None:
            raise ValueError("queries_output is required when queries are supplied")
        query_rows = _query_rows(queries)
        if query_rows:
            fieldnames = list(query_rows[0])
            if document_id_column not in fieldnames:
                raise ValueError(
                    f"query rows are missing document ID column {document_id_column!r}"
                )
        elif isinstance(queries, (str, Path)):
            with Path(queries).open("r", encoding="utf-8", newline="") as source:
                fieldnames = list(csv.DictReader(source).fieldnames or [])
        else:
            fieldnames = []
        removed_by_id: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        kept_queries: list[dict[str, Any]] = []
        for row_index, row in enumerate(query_rows):
            document_id = str(row.get(document_id_column, ""))
            if document_id in removed_ids:
                removed_by_id[document_id].append({
                    "original_index": row_index,
                    "query": row.get("query", ""),
                    "source_query_number": row.get("source_query_number", ""),
                })
            else:
                kept_queries.append(row)
        for entry in entries:
            removed_queries = removed_by_id[entry["document_id"]]
            entry["removed_query_count"] = len(removed_queries)
            entry["removed_queries"] = removed_queries
        write_csv(Path(queries_output), kept_queries, fieldnames)
    _audit("check_empty_documents", entries, output_directory)
    return kept

def _query_rows(
    queries: str | Path | Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(queries, (str, Path)):
        with Path(queries).open("r", encoding="utf-8", newline="") as source:
            return [dict(row) for row in csv.DictReader(source)]
    return [dict(row) for row in queries]


def _queries_by_document(
    queries: str | Path | Iterable[Mapping[str, Any]], id_column: str, query_column: str
) -> dict[str, tuple[str, ...]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(_query_rows(queries)):
        if id_column not in row:
            raise ValueError(f"query row {index} is missing {id_column!r}")
        if query_column not in row:
            raise ValueError(f"query row {index} is missing {query_column!r}")
        grouped[str(row[id_column])].append(str(row[query_column]))
    return {key: tuple(value) for key, value in grouped.items()}


def deduplicate_documents(
    documents: Sequence[Mapping[str, Any]],
    queries: str | Path | Iterable[Mapping[str, Any]],
    output_directory: str | Path = DEFAULT_AUDIT_DIRECTORY,
    *,
    document_id_column: str = "source_id",
    query_column: str = "query",
) -> list[Document]:
    """Remove each later document having identical text and ordered queries."""
    query_sets = _queries_by_document(queries, document_id_column, query_column)
    seen: dict[tuple[str, tuple[str, ...]], str] = {}
    kept, entries = [], []
    for index, document in enumerate(_copy_documents(documents)):
        document_id = str(document["document_id"])
        associated_queries = query_sets.get(document_id, ())
        signature = (document["text"], associated_queries)
        if signature not in seen:
            seen[signature] = document_id
            kept.append(document)
        else:
            entries.append({
                "document_id": document["document_id"],
                "action": "removed_duplicate_document",
                "duplicate_of": seen[signature],
                "original_index": index,
                "query_count": len(associated_queries),
            })
    _audit("deduplicate_documents", entries, output_directory)
    return kept


def check_documents_with_prompt(
    documents: Sequence[Mapping[str, Any]],
    output_directory: str | Path = DEFAULT_AUDIT_DIRECTORY,
) -> list[Document]:
    """Remove a leading 'Analise ... e responda ...' query prompt."""
    cleaned, entries = _copy_documents(documents), []
    for document in cleaned:
        match = _PROMPT.match(document["text"])
        if match:
            document["text"] = document["text"][match.end():]
            entries.append({
                "document_id": document["document_id"],
                "action": "removed_leading_query_prompt",
                "removed_prompt": match.group(0).strip(),
                "removed_characters": match.end(),
            })
    _audit("check_documents_with_prompt", entries, output_directory)
    return cleaned


def clean_documents(
    documents: Sequence[Mapping[str, Any]],
    queries: str | Path | Iterable[Mapping[str, Any]],
    output_directory: str | Path = DEFAULT_AUDIT_DIRECTORY,
    *,
    document_id_column: str = "source_id",
    queries_output: str | Path | None = None,
) -> list[Document]:
    """Run every cleaning function in dependency-safe order."""
    cleaned = check_documents_with_prompt(documents, output_directory)
    cleaned = remove_encode_images(cleaned, output_directory)
    cleaned = remove_unicode(cleaned, output_directory)
    cleaned = correct_url(cleaned, output_directory)
    cleaned = check_empty_documents(
        cleaned,
        output_directory,
        queries=queries,
        queries_output=queries_output,
        document_id_column=document_id_column,
    )
    deduplication_queries = queries_output if queries_output is not None else queries
    return deduplicate_documents(
        cleaned,
        deduplication_queries,
        output_directory,
        document_id_column=document_id_column,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", type=Path, nargs="?", default=DEFAULT_DOCUMENTS)
    parser.add_argument("queries", type=Path, nargs="?", default=DEFAULT_QUERIES)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-directory", type=Path, default=DEFAULT_AUDIT_DIRECTORY)
    parser.add_argument("--document-id-column", default="source_id")
    parser.add_argument(
        "--queries-output",
        type=Path,
        help="Filtered query CSV; defaults to atomically updating the input query CSV",
    )
    args = parser.parse_args()

    with args.documents.open("r", encoding="utf-8") as source:
        documents = json.load(source)
    if not isinstance(documents, list):
        raise ValueError("documents JSON must contain an array")
    cleaned = clean_documents(
        documents, args.queries, args.audit_directory,
        document_id_column=args.document_id_column,
        queries_output=args.queries_output or args.queries,
    )
    write_json(args.output, cleaned)
    print(
        f"Wrote {len(cleaned)} cleaned documents to {args.output}; "
        f"audit files are in {args.audit_directory}"
    )


if __name__ == "__main__":
    main()
