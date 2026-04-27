"""
src/core/Search/Sematic/SematicSearch.py
-----------------------------------------
Semantic search using USearch vector index.

Supports both:
- USearch index (fast, production)
- Fallback to numpy (when index doesn't exist)
"""

import numpy as np
from typing import List, Optional

from src.core.Chunking.Chunker import Chunk
from src.core.Embedding.Embedder import embed_query
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI
from src.core.Search.Sematic.SematicSearchUtils import reconstruct_chunks, SematicSearchResult


# ============================================================================
# Core search function (USearch + Fallback)
# ============================================================================

def semantic_search(
    query: str,
    vectors: np.ndarray,
    chunks: List[Chunk],
    top_k: int = 5,
    min_score: float = 0.0,
    use_usearch: bool = True,
    store: Optional[DarwinStore] = None,
    filename: Optional[str] = None,
) -> List[SematicSearchResult]:
    """
    Pure semantic (dense) search over pre-computed embeddings.
    
    Args:
        query: Raw user query string
        vectors: np.ndarray of shape (N, D) — L2-normalized chunk embeddings
        chunks: List of Chunk objects corresponding to vectors
        top_k: Number of results to return
        min_score: Minimum similarity threshold (0-1)
        use_usearch: Try USearch index first (falls back to numpy)
        store: DarwinStore instance (required for USearch)
        filename: Document filename (required for USearch)
    
    Returns:
        List[SematicSearchResult] sorted by descending similarity
    """
    if len(vectors) == 0 or len(chunks) == 0:
        return []
    
    config = store.load_config()
    model_name = config.get("embedding_model")

    if model_name is None:
        TerminalUI.warn("Embedding model not found in config")
        return []
    
    # Embed query (already L2-normalized)
    query_vec = embed_query(query, model_name)
    
    if query_vec is None or len(query_vec) == 0:
        TerminalUI.error("Failed to embed query — check model availability")
        return []
    
    # Try USearch first if available
    if use_usearch and store and filename:
        try:
            if store.has_vector(filename):
                return _usearch_search(
                    query_vec=query_vec,
                    store=store,
                    filename=filename,
                    chunks=chunks,
                    top_k=top_k,
                    min_score=min_score,
                )
        except Exception as e:
            TerminalUI.dim(f"USearch search failed, falling back to numpy: {e}")
    
    # Fallback to numpy exhaustive search
    return _numpy_search(
        query_vec=query_vec,
        vectors=vectors,
        chunks=chunks,
        top_k=top_k,
        min_score=min_score,
    )


def _usearch_search(
    query_vec: np.ndarray,
    store: DarwinStore,
    filename: str,
    chunks: List[Chunk],
    top_k: int,
    min_score: float,
) -> List[SematicSearchResult]:
    """
    Search using USearch index via the VectorMixin.
    
    The USearch index stores integer keys (0, 1, 2, ...) that correspond
    to array positions. vector.json stores these as strings in chunk_ids[].
    We convert the string back to int and use it as a direct list index
    into the chunks[] array loaded from disk.
    """
    chunk_scores = store.search_vector(filename, query_vec, k=top_k)
    
    if not chunk_scores:
        return []
    
    results = []
    for chunk_id_str, score in chunk_scores:
        if score < min_score:
            continue
        
        # chunk_id_str looks like "0", "1", "2", ... — convert to int
        # and use as direct index into the chunks list (same ordering)
        try:
            idx = int(chunk_id_str)
        except (ValueError, TypeError):
            continue
        
        if idx < 0 or idx >= len(chunks):
            continue
        
        chunk = chunks[idx]
        
        results.append(SematicSearchResult(
            text=chunk.text,
            clean=chunk.clean,
            score=score,
            source=chunk.source,
            page=chunk.page,
            heading=chunk.heading,
            chunk_index=chunk.index,
            links=chunk.links,
            is_overlap=chunk.is_overlap,
        ))
    
    return results


def _numpy_search(
    query_vec: np.ndarray,
    vectors: np.ndarray,
    chunks: List[Chunk],
    top_k: int,
    min_score: float,
) -> List[SematicSearchResult]:
    """Fallback: exhaustive numpy dot product search."""
    if vectors.size == 0:
        return []
    
    # Dot product = cosine similarity (vectors are normalized)
    similarities = np.dot(vectors, query_vec)
    
    # Get top-k indices
    if top_k >= len(similarities):
        top_indices = np.argsort(similarities)[::-1]
    else:
        # Faster for large N: partial sort
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    
    # Build results
    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        
        if score < min_score:
            continue
            
        chunk = chunks[idx]
        results.append(SematicSearchResult(
            text=chunk.text,
            clean=chunk.clean,
            score=score,
            source=chunk.source,
            page=chunk.page,
            heading=chunk.heading,
            chunk_index=chunk.index,
            links=chunk.links,
            is_overlap=chunk.is_overlap
        ))
    
    return results


# ============================================================================
# Shared validation logic
# ============================================================================

def _get_validated_search_method(
    filename: str,
    store: DarwinStore,
    expected_model: str,
    use_usearch: bool = True,
) -> str:
    """
    Determine which search method to use with full validation.
    
    Also handles model mismatch warnings.
    """
    # Check if USearch index exists
    has_vector = store.has_vector(filename)
    
    if use_usearch and has_vector:
        try:
            vector_meta = store.load_vector_metadata(filename)
            index_model = vector_meta.get("embedding_model") if vector_meta else None
            
            if index_model and index_model != expected_model:
                TerminalUI.warn(
                    f"   Model mismatch for {filename}\n"
                    f"   Index: {index_model}\n"
                    f"   Current: {expected_model}\n"
                    f"   Falling back to numpy. Run 'darwin embeddings --rerun' to update."
                )
                return "numpy"
            
            return "usearch"
            
        except Exception as e:
            TerminalUI.warn(f"Failed to load vector index for {filename}: {e}")
            return "numpy"
    
    # Check if numpy embeddings exist
    if store.has_embeddings(filename):
        return "numpy"
    
    # Nothing exists
    return "error"


def _execute_usearch_search(
    filename: str,
    query_vec: np.ndarray,
    store: DarwinStore,
    top_k: int,
    min_score: float,
) -> List[SematicSearchResult]:
    """Execute search using USearch index."""
    chunks = store.load_chunks(filename)
    if not chunks:
        TerminalUI.dim(f"No chunks found for {filename}")
        return []
    
    return _usearch_search(
        query_vec=query_vec,
        store=store,
        filename=filename,
        chunks=chunks,
        top_k=top_k,
        min_score=min_score,
    )


def _execute_numpy_search(
    filename: str,
    query_vec: np.ndarray,
    store: DarwinStore,
    top_k: int,
    min_score: float,
) -> List[SematicSearchResult]:
    """Execute search using numpy fallback."""
    data = store.load_embeddings(filename)
    if data is None:
        TerminalUI.dim(f"No embeddings found for {filename}")
        return []
    
    vectors = data["vectors"]
    chunks = reconstruct_chunks(data, filename)
    
    return _numpy_search(
        query_vec=query_vec,
        vectors=vectors,
        chunks=chunks,
        top_k=top_k,
        min_score=min_score,
    )


# ============================================================================
# Multi-document search
# ============================================================================

def search_across_documents(
    query: str,
    store: DarwinStore,
    filenames: Optional[List[str]] = None,
    top_k: int = 10,
    min_score: float = 0.0,
    use_usearch: bool = True,
) -> List[SematicSearchResult]:
    """
    Search across multiple documents with automatic method selection.
    """
    if not filenames:
        TerminalUI.warn("No files specified for search")
        return []
    
    all_results = []
    per_doc_k = max(5, top_k // len(filenames) + 1)
    
    # Track stats for summary
    stats = {"usearch": 0, "numpy": 0, "failed": 0}
    
    # Get the current model name from config
    config = store.load_config()
    model_name = config.get("embedding_model")
    if model_name is None:
        TerminalUI.warn("No embedding model found in config")
        return []
    
    # Embed query ONCE for all documents
    query_vec = embed_query(query, model_name)
    
    if query_vec is None or len(query_vec) == 0:
        TerminalUI.error("Failed to embed query — check model availability")
        return []
    
    for i, filename in enumerate(filenames):
        try:
            method = _get_validated_search_method(
                filename=filename,
                store=store,
                expected_model=model_name,
                use_usearch=use_usearch,
            )
            
            if method == "error":
                TerminalUI.dim(f"  No embeddings for {filename}, skipping")
                stats["failed"] += 1
                continue
            
            # Execute search with chosen method
            if method == "usearch":
                doc_results = _execute_usearch_search(
                    filename=filename,
                    query_vec=query_vec,
                    store=store,
                    top_k=per_doc_k,
                    min_score=min_score,
                )
                stats["usearch"] += 1
            else:  # numpy
                doc_results = _execute_numpy_search(
                    filename=filename,
                    query_vec=query_vec,
                    store=store,
                    top_k=per_doc_k,
                    min_score=min_score,
                )
                stats["numpy"] += 1
            
            if doc_results:
                all_results.extend(doc_results)
            else:
                stats["failed"] += 1
                
        except Exception as e:
            TerminalUI.error(f"Failed to search {filename}: {e}")
            stats["failed"] += 1
            continue
    
    # Summary stats
    TerminalUI.dim(
        f"Search completed: "
        f"[green]{stats['usearch']} USearch[/green] · "
        f"[yellow]{stats['numpy']} numpy[/yellow] · "
        f"[red]{stats['failed']} failed[/red]"
    )
    
    # Re-rank combined results
    all_results.sort(key=lambda r: r.score, reverse=True)
    
    return all_results[:top_k]


# ============================================================================
# Single document search
# ============================================================================

def search_single_document(
    query: str,
    filename: str,
    store: DarwinStore,
    top_k: int = 5,
    min_score: float = 0.0,
    use_usearch: bool = True,
) -> List[SematicSearchResult]:
    """
    Search within a single document with automatic method selection.
    """
    try:
        config = store.load_config()
        model_name = config.get("embedding_model")
        
        if model_name is None:
            TerminalUI.error("No embedding model configured")
            return []
        
        # Embed query once
        query_vec = embed_query(query, model_name)
        
        if query_vec is None or len(query_vec) == 0:
            TerminalUI.error("Failed to embed query — check model availability")
            return []
        
        # Validate and choose method
        method = _get_validated_search_method(
            filename=filename,
            store=store,
            expected_model=model_name,
            use_usearch=use_usearch,
        )
        
        if method == "error":
            TerminalUI.error(f"No embeddings found for {filename}")
            TerminalUI.dim("Try running 'darwin embeddings' on this document first")
            return []
        
        # Execute with chosen method
        if method == "usearch":
            TerminalUI.dim(f"⚡ Using USearch index for {filename}")
            return _execute_usearch_search(
                filename=filename,
                query_vec=query_vec,
                store=store,
                top_k=top_k,
                min_score=min_score,
            )
        else:
            TerminalUI.dim(f"📦 Using numpy fallback for {filename}")
            return _execute_numpy_search(
                filename=filename,
                query_vec=query_vec,
                store=store,
                top_k=top_k,
                min_score=min_score,
            )
        
    except FileNotFoundError:
        TerminalUI.error(f"No embeddings found for {filename}")
        return []
    except Exception as e:
        TerminalUI.error(f"Search failed: {e}")
        return []