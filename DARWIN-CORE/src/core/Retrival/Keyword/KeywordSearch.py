# KeywordSearch.py
from typing import List, Optional
from src.core.Retrival.Keyword.KeywordUtilities import calculate_bm25
from src.core.Retrival.lib import KeywordSearchResult, OverlapDirection
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Storage.DarwinStore import DarwinStore


def search_single_document_keyword(
    query: str,
    filename: str,
    store: DarwinStore,
    top_k: Optional[int] = 10,
    k1: float = 1.5,
    b: float = 0.75,
    min_score: float = 0.0,
) -> List[KeywordSearchResult]:
    """Search a single document using BM25 with query expansion."""
    if not query:
        return []

    all_chunks = store.query_records(
        filename,
        [
            "index",
            "text",
            "clean",
            "page",
            "page_end",
            "heading",
            "structural_type",
            "links",
            "overlap_direction",
        ],
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
    if top_k is not None:
        return results[:top_k]
    return results


def search_across_documents_keyword(
    query: str,
    filenames: List[str],
    store: "DarwinStore",
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
    min_score: float = 0.0,
    per_doc_top_k: Optional[int] = None,  # NEW: per-document limit; None = unlimited
) -> List[KeywordSearchResult]:
    """Search across multiple documents using BM25."""
    all_results: List[KeywordSearchResult] = []

    for filename in filenames:
        if not store.has_bm25(filename) or not store.has_records(filename):
            darwin.log(
                f"Skipping {filename} -- missing index or records", LogType.WARNING
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
            darwin.log(
                f"Search failed for {filename}: {e}", LogType.ERROR, exc_info=True
            )

    # Normalize BM25 scores across documents (BM25 is document-dependent)
    if all_results:
        scores = [r.score for r in all_results]
        min_s, max_s = min(scores), max(scores)
        span = max_s - min_s or 1.0
        for r in all_results:
            r.score = (r.score - min_s) / span

    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:top_k]
