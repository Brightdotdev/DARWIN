"""
src/core/Search/Keyword/KeywordSearch.py
----------------------------------
Pure BM25 (sparse) retrieval over chunked document text.
With automatic query expansion via term bank (falls back to raw query).
"""

from typing import List
from src.core.Search.Keyword.KeywordUtilities import calculate_bm25
from src.core.Search.lib import KeywordSearchResult
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Storage.DarwinStore import DarwinStore

def search_single_document_keyword(
    query: str,
    filename: str,
    store: 'DarwinStore',
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
    min_score: float = 0.0,
    expand_query: bool = True,
) -> List[KeywordSearchResult]:
    """
    Search a single document and return results sorted by score.
    
    Query expansion is attempted automatically using the term bank.
    If expansion fails (no term bank, embedder unavailable, etc.),
    falls back to the original query silently.
    """
    if not query.strip():
        return []

    chunks = store.load_chunks(filename)
    data = store.load_bm25(filename)
    if not chunks or not data:
        darwin.log(f"Skipping {filename} — no data for keyword search", LogType.INFO)
        darwin.log("Try running `darwin index` first", LogType.INFO)
        return []

    from src.core.BM25.BM25Utils import tokenize_text, STANDARD_STOPS

    stopwords = set(data.domain_stopwords).union(STANDARD_STOPS)

    # ── Query expansion (with automatic fallback) ──────────────────────────
    search_query = query  # Default: use original query
    
    if expand_query:
        try:
            expanded = store.expand_query(filename, query, top_n=5, min_similarity=0.5)
            if expanded != query:
                search_query = expanded
        except Exception:
            pass

    query_terms = tokenize_text(search_query, domain_stops=stopwords)

    if not query_terms:
        darwin.log("Query produced no usable terms", LogType.WARNING)
        return []

    results = []
    for chunk_idx in range(len(data.tokens)):
        score, matched = calculate_bm25(query_terms, chunk_idx, data, k1, b)
        
        if score >= min_score:
            chunk = chunks[chunk_idx]
            results.append(KeywordSearchResult(
                text=chunk.text,
                clean=chunk.clean,
                score=score,
                source=filename,
                page=chunk.page,
                page_end=chunk.page_end,
                heading=chunk.heading,
                chunk_index=chunk_idx,
                matched_terms=matched,
                links=chunk.links if chunk.links else [],
                overlap_direction=chunk.overlap_direction,
            ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]

def search_across_documents_keyword(
    filenames: List[str],
    query: str,
    store: 'DarwinStore',
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
    expand_query: bool = True,
) -> List[KeywordSearchResult]:
    """Search across multiple documents using BM25 with query expansion."""
    all_results: List[KeywordSearchResult] = []

    for filename in filenames:
        if not store.has_bm25(filename):
            darwin.log(f"Skipping {filename} — no BM25 index", LogType.INFO)
            continue
        if not store.has_chunks(filename):
            darwin.log(f"Skipping {filename} — no chunks", LogType.INFO)
            continue
        try:
            results = search_single_document_keyword(
                query=query,
                filename=filename,
                store=store,
                top_k=top_k,
                k1=k1,
                b=b,
                expand_query=expand_query,
            )
            all_results.extend(results)
        except Exception as e:
            darwin.log(f"Search failed for {filename}: {e}", LogType.ERROR)

    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:top_k]