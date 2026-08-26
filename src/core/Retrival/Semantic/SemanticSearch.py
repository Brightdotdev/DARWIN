import numpy as np
import re
from typing import List, Optional, Set

from core.Storage.DarwinStore import DarwinStore
from core.Pipeline.Indexing.Embedding.Embedder import EmbeddingSession
from core.Logger.DarwinLogger import darwin
from core.Retrival.lib import OverlapDirection, SemanticSearchResult
from core.Retrival.retrieval_utils import normalize_links

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

# Terms too common to be meaningful as matched terms
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "that", "this", "these", "those", "it",
    "he", "she", "they", "we", "you", "i", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "what", "which",
    "who", "whom", "where", "when", "why", "how", "if", "then",
})


def _extract_matched_terms(query: str, text: str, max_terms: int = 10) -> Set[str]:
    """Extract query terms that appear in the chunk text (case-insensitive)."""
    if not query or not text:
        return set()
    query_tokens = {
        t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in _STOP_WORDS and len(t) > 1
    }
    if not query_tokens:
        return set()
    text_lower = text.lower()
    matched = {t for t in query_tokens if t in text_lower}
    return set(list(matched)[:max_terms])


def _rows_to_results(
    rows: List[dict], default_source: str = "", query: str = ""
) -> List[SemanticSearchResult]:
    results = []
    for row in rows:
        od = row.get("overlap_direction")
        if od is None:
            od = OverlapDirection.NONE
        elif isinstance(od, str):
            try:
                od = OverlapDirection(od)
            except ValueError:
                od = OverlapDirection.NONE
        elif isinstance(od, OverlapDirection):
            pass
        else:
            od = OverlapDirection.NONE
        text = row.get("text", "")
        matched = _extract_matched_terms(query, text) if query else set()
        results.append(
            SemanticSearchResult(
                text=text,
                clean=row.get("clean", ""),
                score=row.get("_score", 0.0),
                source=row.get("source", default_source),
                page=row.get("page", 0),
                page_end=row.get("page_end", 0),
                heading=row.get("heading", ""),
                chunk_index=row.get("index", 0),
                links=normalize_links(row.get("links", [])),
                overlap_direction=od,
                structural_type=row.get("structural_type", "body"),
                matched_terms=matched,
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
    per_doc_top_k: Optional[int] = None,
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
        results = _rows_to_results(rows, default_source=filename, query=query)
        if per_doc_top_k is not None and per_doc_top_k > 0:
            results = results[:per_doc_top_k]
        return results
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
        results = _rows_to_results(rows, default_source=filename, query=query)
        if per_doc_top_k is not None and per_doc_top_k > 0:
            results = results[:per_doc_top_k]
        return results


def _search_across_documents_with_vector(
    query_vec: np.ndarray,
    session: EmbeddingSession,
    store: DarwinStore,
    filenames: List[str],
    top_k: int,
    min_score: float,
    use_usearch: bool,
    per_doc_top_k: Optional[int],
    query: str = "",
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
        darwin.info(
            f"Multi-doc search: {len(rows)} results across {len(filenames)} files"
        )
        return _rows_to_results(rows, query=query)

    # Per‑document limiting: loop over files, reuse session/vector
    all_results: List[SemanticSearchResult] = []
    for fname in filenames:
        try:
            single_results = search_single_document(
                query=query,
                model_name="",  # not used when session & vector are provided
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
            darwin.error(f"Search failed for {fname}: {e}", exc_info=True)

    # Global sort and slice
    all_results.sort(key=lambda r: r.score, reverse=True)
    final_results = all_results[:top_k]
    darwin.info(
        f"Per-doc limited search: {len(final_results)} results pooled from {len(filenames)} files"
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
        darwin.warning("No files specified for search")
        return []

    if not model_name:
        darwin.error("No embedding model configured")
        return []

    # Create one session and embed the query once
    with EmbeddingSession(model_key=model_name) as session:
        if session.embedder is None:
            darwin.error("Failed to load embedding model")
            return []
        query_vec = session.embed_query(query)
        if query_vec is None:
            darwin.error("Failed to embed query")
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
            query=query,
        )
