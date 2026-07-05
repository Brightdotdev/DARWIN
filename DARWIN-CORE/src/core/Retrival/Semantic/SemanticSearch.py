"""
src/features/Retrival/Semantic/SemanticSearch.py
-----------------------------------------------
Semantic search orchestration: embed query once, reuse across documents.
"""

import numpy as np
from typing import List, Optional

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Pipeline.Indexing.Embedding.Embedder import EmbeddingSession
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Retrival.lib import OverlapDirection, SemanticSearchResult, ChunkLink

_RESULT_COLUMNS = [
    "text",
    "clean",
    "source",
    "page",
    "page_end",
    "heading",
    "index",
    "links",
    "overlap_direction",
    "structural_type",
]


def _normalize_links(links_raw: list) -> list:
    out = []
    for l in links_raw or []:
        if isinstance(l, dict):
            out.append(ChunkLink(url=l.get("url", ""), text=l.get("text", "")))
        elif isinstance(l, ChunkLink):
            out.append(l)
        else:
            out.append(ChunkLink(url=str(l), text=""))
    return out


def _rows_to_results(
    rows: List[dict], default_source: str = ""
) -> List[SemanticSearchResult]:
    results = []
    for row in rows:
        od = row.get("overlap_direction")
        if od is None:
            od = OverlapDirection.NONE
        elif isinstance(od, str):
            od = OverlapDirection(od)
        results.append(
            SemanticSearchResult(
                text=row.get("text", ""),
                clean=row.get("clean", ""),
                score=row.get("_score", 0.0),
                source=row.get("source", default_source),
                page=row.get("page", 0),
                page_end=row.get("page_end", 0),
                heading=row.get("heading", ""),
                chunk_index=row.get("index", 0),
                links=_normalize_links(row.get("links", [])),
                overlap_direction=od,
            )
        )
    return results


def search_single_document(
    query: str,
    model_name: str,
    filename: str,
    store: DarwinStore,
    top_k: int = 5,
    min_score: float = 0.0,
    use_usearch: bool = True,
    session: Optional[EmbeddingSession] = None,
    query_vec: Optional[np.ndarray] = None,
) -> List[SemanticSearchResult]:
    """
    Search within a single document.
    If session and query_vec are provided, reuse them; otherwise create a new session.
    """
    if not filename:
        return []
    if not model_name and (session is None or query_vec is None):
        return []

    # If no session/vector provided, create a temporary one
    if session is None or query_vec is None:
        with EmbeddingSession(model_key=model_name) as sess:
            if sess.embedder is None:
                return []
            qvec = sess.embed_query(query)
            if qvec is None:
                return []
            rows = store.search_single(
                query_vec=qvec,
                filename=filename,
                k=top_k,
                min_score=min_score,
                columns=_RESULT_COLUMNS,
                force_numpy=not use_usearch,
            )
        if not rows:
            return []
        return _rows_to_results(rows, default_source=filename)
    else:
        # Reuse provided session and vector
        rows = store.search_single(
            query_vec=query_vec,
            filename=filename,
            k=top_k,
            min_score=min_score,
            columns=_RESULT_COLUMNS,
            force_numpy=not use_usearch,
        )
        if not rows:
            return []
        return _rows_to_results(rows, default_source=filename)


def _search_across_documents_with_vector(
    query_vec: np.ndarray,
    session: EmbeddingSession,
    store: DarwinStore,
    filenames: List[str],
    top_k: int,
    min_score: float,
    use_usearch: bool,
    per_doc_top_k: Optional[int],
) -> List[SemanticSearchResult]:
    """Internal helper that uses a pre‑embedded query vector and a loaded session."""
    if per_doc_top_k is None:
        # Fast multi‑search (global top‑k across all files)
        rows = store.search_multi(
            query_vec=query_vec,
            filenames=filenames,
            top_k=top_k,
            min_score=min_score,
            columns=_RESULT_COLUMNS,
            force_numpy=not use_usearch,
        )
        if not rows:
            return []
        darwin.log(
            f"Multi-doc search (fast): {len(rows)} results across {len(filenames)} files"
        )
        return _rows_to_results(rows)

    # Per‑document limiting: loop over files, reuse session/vector
    all_results: List[SemanticSearchResult] = []
    for fname in filenames:
        try:
            single_results = search_single_document(
                query="",  # not used when session & vector are provided
                model_name="",  # not used
                filename=fname,
                store=store,
                top_k=per_doc_top_k,
                min_score=min_score,
                use_usearch=use_usearch,
                session=session,
                query_vec=query_vec,
            )
            all_results.extend(single_results)
        except Exception as e:
            darwin.log(f"Search failed for {fname}: {e}", LogType.ERROR, exc_info=True)

    # Global sort and slice
    all_results.sort(key=lambda r: r.score, reverse=True)
    final_results = all_results[:top_k]
    darwin.log(
        f"Per-doc limited search: {len(final_results)} results (pooled from {len(filenames)} files)"
    )
    return final_results


def search_across_documents(
    query: str,
    model_name: str,
    store: DarwinStore,
    filenames: Optional[List[str]] = None,
    top_k: int = 10,
    min_score: float = 0.0,
    use_usearch: bool = True,
    per_doc_top_k: Optional[int] = None,
) -> List[SemanticSearchResult]:
    """
    Search across multiple documents with a single model load.
    """
    if not filenames:
        darwin.log("No files specified for search", LogType.WARNING)
        return []

    if not model_name:
        darwin.log("No embedding model configured", LogType.ERROR)
        return []

    # Create one session and embed the query once
    with EmbeddingSession(model_key=model_name) as session:
        if session.embedder is None:
            darwin.log("Failed to load embedding model", LogType.ERROR)
            return []
        query_vec = session.embed_query(query)
        if query_vec is None:
            darwin.log("Failed to embed query", LogType.ERROR)
            return []

        return _search_across_documents_with_vector(
            query_vec=query_vec,
            session=session,
            store=store,
            filenames=filenames,
            top_k=top_k,
            min_score=min_score,
            use_usearch=use_usearch,
            per_doc_top_k=per_doc_top_k,
        )
