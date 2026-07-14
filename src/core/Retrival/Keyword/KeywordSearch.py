# KeywordSearch.py
from typing import List, Optional
from src.core.Retrival.Keyword.KeywordUtilities import calculate_bm25
from src.core.Retrival.lib import KeywordSearchResult, OverlapDirection, RETRIVAL_COLUMNS, DEFAULT_TOP_K, DEFAULT_K1, DEFAULT_B
from src.core.Logger.DarwinLogger import darwin
from src.core.Storage.DarwinStore import DarwinStore


def search_single_document_keyword(
    query: str,
    filename: str,
    store: DarwinStore,
    top_k: Optional[int] = DEFAULT_TOP_K,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    min_score: float = 0.0,
    per_doc_top_k: Optional[int] = None,
) -> List[KeywordSearchResult]:
    """Search a single document using BM25 with query expansion."""
    if not query:
        return []

    all_chunks = store.query_records(
        filename,
        RETRIVAL_COLUMNS

    )
    data = store.load_bm25(filename)
    if not all_chunks or not data:
        darwin.warning(f"Skipping {filename} -- File not indexed for keyword search")
        return []

    chunk_by_index = {c.get("index"): c for c in all_chunks}
    from src.core.Pipeline.Indexing.BM25.BM25Utils import tokenize_text

    query_terms = tokenize_text(query)
    if not query_terms:
        darwin.warning("Query produced no usable terms")
        return []

    tokens = data.get("tokens", [])
    index_map = data.get("chunk_indices")
    if index_map is None or all(idx is None for idx in index_map):
        darwin.warning(
            f"No valid chunk_indices in BM25 data for {filename}; assuming positional alignment"
        )
        index_map = list(range(len(tokens)))

    results = []
    for bm25_pos in range(len(tokens)):
        score, matched = calculate_bm25(query_terms, bm25_pos, data, k1, b)
        if score >= min_score:
            if bm25_pos >= len(index_map):
                continue
            chunk_idx = index_map[bm25_pos]
            chunk = chunk_by_index.get(chunk_idx)
            if not chunk:
                continue

            od_raw = chunk.get("overlap_direction")
            if isinstance(od_raw, str):
                od = OverlapDirection(od_raw)
            elif isinstance(od_raw, OverlapDirection):
                od = od_raw
            else:
                od = OverlapDirection.NONE

            results.append(
                KeywordSearchResult(
                    text=chunk["text"],
                    clean=chunk["clean"],
                    score=score,
                    source=filename,
                    page=chunk["page"],
                    page_end=chunk["page_end"],
                    heading=chunk["heading"],
                    chunk_index=chunk_idx,
                    matched_terms=set(matched),
                    links=chunk.get("links", []),
                    overlap_direction=od,
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    if per_doc_top_k is not None and per_doc_top_k > 0:
        results = results[:per_doc_top_k]
    if top_k is not None:
        results = results[:top_k]
    return results


def search_across_documents_keyword(
    query: str,
    filenames: List[str],
    store: "DarwinStore",
    top_k: int = DEFAULT_TOP_K,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
    min_score: float = 0.0,
    per_doc_top_k: Optional[int] = None,  # NEW: per-document limit; None = unlimited
) -> List[KeywordSearchResult]:
    """Search across multiple documents using BM25."""
    all_results: List[KeywordSearchResult] = []

    for filename in filenames:
        if not store.has_bm25(filename) or not store.has_records(filename):
            darwin.warning(
                f"Skipping {filename}: missing index or records"
            )
            continue
        try:
            results = search_single_document_keyword(
                query=query,
                filename=filename,
                store=store,
                top_k=per_doc_top_k,  # pass the per-doc limit
                k1=k1,
                b=b,
                min_score=min_score,
            )
            all_results.extend(results)
        except Exception as e:
            darwin.error(
                f"Search failed for {filename}: {e}", exc_info=True
            )

    # Normalize BM25 scores across documents (BM25 is document-dependent)
    if all_results:
        scores = [r.score for r in all_results]
        min_s, max_s = min(scores), max(scores)
        span = max_s - min_s or 1.0
        normalized = []
        for r in all_results:
            normalized.append(KeywordSearchResult(
                text=r.text,
                clean=r.clean,
                score=(r.score - min_s) / span,
                source=r.source,
                page=r.page,
                page_end=r.page_end,
                heading=r.heading,
                chunk_index=r.chunk_index,
                matched_terms=r.matched_terms,
                links=r.links,
                overlap_direction=r.overlap_direction,
            ))
        all_results = normalized

    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:top_k]
