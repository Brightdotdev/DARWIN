"""
src/core/Search/Semantic/SemanticSearch.py
-----------------------------------------
Semantic search using the new DarwinStore lazy retrieval pipeline.

Thin orchestration layer: embeds queries, delegates storage/retrieval to
DarwinStore (SearchMixin), and maps raw dict rows to SemanticSearchResult.
"""

import numpy as np
from typing import List, Optional

from src.core.Storage.DarwinStore import DarwinStore
from src.pipleline.Indexing.Embedding.Embedder import EmbeddingSession
from src.features.Retrival.Semantic.SematicSearchUtils import SemanticSearchResult
from src.core.Logger.DarwinLogger import darwin, LogType
from src.pipleline.Ingestion.Chunking.StructuralClassifier import is_structural_for_embedding

# Columns needed to populate a SemanticSearchResult.
# These must match the field names produced by normalize_record(Chunk).
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
    "structural_type",  # NEW
]


def _embed_query_safe(query: str, model_name: str) -> Optional[np.ndarray]:
    """Safely embed query with error handling."""
    if not model_name:
        darwin.log("No embedding model configured", LogType.ERROR)
        return None
    with EmbeddingSession(model_key=model_name) as session:
        query_vec = session.embed_query(query)

    if query_vec is None or len(query_vec) == 0:
        darwin.log("Failed to embed query — check model availability", LogType.ERROR)
        return None

    return query_vec


def _rows_to_results(rows: List[dict]) -> List[SemanticSearchResult]:
    """Map raw storage dicts to SemanticSearchResult objects, filtering structural chunks."""
    results = []
    for row in rows:
        # NEW: Hard gate — skip structural chunks at result build time
        if is_structural_for_embedding(row.get("structural_type")):
            continue
            
        results.append(SemanticSearchResult(
            text=row.get("text", ""),
            clean=row.get("clean", ""),
            score=row.get("_score", 0.0),
            source=row.get("source", ""),
            page=row.get("page", 0),
            page_end=row.get("page_end", 0),
            heading=row.get("heading", ""),
            chunk_index=row.get("index", 0),
            links=row.get("links", []),
            overlap_direction=row.get("overlap_direction", None),
        ))
    return results


def search_single_document(
    query: str,
    model_name: str,
    filename: str,
    store: DarwinStore,
    top_k: int = 5,
    min_score: float = 0.0,
    use_usearch: bool = True,
) -> List[SemanticSearchResult]:
    """Search within a single document using the new lazy retrieval pipeline."""
    try:
        if not model_name:
            darwin.log("No embedding model configured", LogType.ERROR)
            return []

        query_vec = _embed_query_safe(query, model_name)
        if query_vec is None:
            return []

        # Delegate entirely to SearchMixin; get raw dicts back with '_score'
        rows = store.search_single(
            query_vec=query_vec,
            filename=filename,
            k=top_k,
            min_score=min_score,
            columns=_RESULT_COLUMNS,
            force_numpy=not use_usearch,
        )

        if not rows:
            darwin.log(f"No results for {filename}", LogType.INFO)
            return []

        return _rows_to_results(rows)

    except FileNotFoundError:
        darwin.log(f"No embeddings found for {filename}", LogType.ERROR)
        return []
    except Exception as e:
        darwin.log(f"Search failed for {filename}: {e}", LogType.ERROR)
        return []


def search_across_documents(
    query: str,
    model_name: str,
    store: DarwinStore,
    filenames: Optional[List[str]] = None,
    top_k: int = 10,
    min_score: float = 0.0,
    use_usearch: bool = True,
) -> List[SemanticSearchResult]:
    """Search across multiple documents with automatic method selection."""
    if not filenames:
        darwin.log("No files specified for search", LogType.WARNING)
        return []
    
    if not model_name:
        darwin.log("No embedding model configured", LogType.ERROR)
        return []

    query_vec = _embed_query_safe(query, model_name)
    if query_vec is None:
        return []

    # Delegate multi-doc search to SearchMixin
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
        f"Multi-doc search: {len(rows)} results across {len(filenames)} files",
        LogType.INFO,
    )
    return _rows_to_results(rows)