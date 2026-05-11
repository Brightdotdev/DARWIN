# src/features/Query/QueryPreprocessor.py
"""
Preprocess a RawQuery into a QueryContext.
Fixed: PastedText now passes the required 'meta' field to the legacy utility type.
"""
from __future__ import annotations

from typing import List, Tuple
from src.features.Query.QueryTypes import (
    RawQuery,
    QueryContext,
    PastedText,
    URLRef,
)
from src.features.Query.QueryUtils import (
    extract_explicit_blobs,
    resolve_blob_references,
    extract_and_replace_urls,
    extract_file_tags,
    PastedText as UtilsPastedText,
)
from src.features.Storage.DarwinStore import DarwinStore


def is_searchable(filename: str) -> bool:
    store = DarwinStore()
    status = store.status()
    files = status.get("files", {})
    info = files.get(filename)
    return info.get("is_hybrid_searchable", False) if info else False


def get_all_searchable_filenames() -> List[str]:
    store = DarwinStore()
    status = store.status()
    files = status.get("files", {})
    return [fname for fname, info in files.items() if info.get("is_hybrid_searchable", False)]


def validate_searchable_files(
    files: List[str],
) -> Tuple[List[str], List[str]]:
    searchable, non_searchable = [], []
    for f in files:
        (searchable if is_searchable(f) else non_searchable).append(f)
    return searchable, non_searchable


def handle_file_integrity(
    query: str,
    referenced_files: List[str],
    pasted_texts: List[UtilsPastedText],
):
    """Legacy wrapper – keep it for existing callers."""
    blobs: List[dict] = []

    query, explicit_blobs = extract_explicit_blobs(query)
    blobs.extend(explicit_blobs)

    for p in pasted_texts:
        blobs.append({"id": p.id, "content": p.content, "type": p.type})
        query = query.replace(p.content, f"[BLOB:{p.id}]")

    query = resolve_blob_references(query, blobs)

    all_files = get_all_searchable_filenames()
    tagged_files, query = extract_file_tags(query, all_files)
    combined_files = list(dict.fromkeys(referenced_files + tagged_files))

    if not combined_files:
        query, urls = extract_and_replace_urls(query)
        return {
            "status": "no_files_specified",
            "query": query,
            "urls": urls,
            "files_scope": [],
            "rejected_files": [],
            "blobs": blobs,
            "pasted_texts": pasted_texts,
        }

    searchable, non_searchable = validate_searchable_files(combined_files)
    if not searchable:
        query, urls = extract_and_replace_urls(query)
        return {
            "status": "requires_indexing",
            "query": query,
            "urls": urls,
            "files_scope": [],
            "rejected_files": non_searchable,
            "blobs": blobs,
            "pasted_texts": pasted_texts,
            "indexing_suggestion": {
                "message": "These files are not indexed yet.",
                "files": non_searchable,
                "action": "index_now",
                "next_step": "Would you like to index these files?",
            },
        }

    query, urls = extract_and_replace_urls(query)
    status = "indexed" if not non_searchable else "partial"

    return {
        "status": status,
        "query": query,
        "urls": urls,
        "files_scope": searchable,
        "rejected_files": non_searchable,
        "blobs": blobs,
        "pasted_texts": pasted_texts,
    }


def preprocess_raw_query(raw: RawQuery, allow_search: bool = True) -> QueryContext:
    """
    Main entry point for the new pipeline.
    """
    # ── FIX: supply the required 'meta' field for the legacy PastedText ──
    utils_pasted = []
    for p in raw.pasted_texts:
        utils_pasted.append(
            UtilsPastedText(
                id=p.id or "",
                content=p.content,
                type=p.type,
                meta={},           # <-- THIS WAS MISSING
            )
        )

    result = handle_file_integrity(
        query=raw.query,
        referenced_files=raw.pinned_files,
        pasted_texts=utils_pasted,
    )

    # Map back to our domain types
    blobs = []
    for b in result.get("blobs", []):
        blobs.append(PastedText(
            id=b.get("id"),
            content=b.get("content", ""),
            type=b.get("type", "document"),
        ))

    extracted_urls: List[URLRef] = []
    for u in result.get("urls", []):
        if isinstance(u, dict):
            extracted_urls.append(URLRef(**u))
        elif isinstance(u, str):
            extracted_urls.append(URLRef(url=u, domain="", placeholder=u))

    return QueryContext(
        query=result["query"],
        blobs=blobs,
        searchable_files=result.get("files_scope", []),
        non_searchable_files=result.get("rejected_files", []),
        extracted_urls=extracted_urls,
        fetched_url_contents=[],   # filled later via URL fetcher
    )