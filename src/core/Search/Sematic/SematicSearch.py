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
from src.core.Search.Sematic.SematicSearchUtils import reconstruct_chunks, SematicSearchResult
from src.core.Logger.DarwinLogger import darwin, LogType


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
    """
    if len(vectors) == 0 or len(chunks) == 0:
        return []
    
    config = store.load_config()
    model_name = config.get("embedding_model")

    if model_name is None:
        darwin.log("Embedding model not found in config", LogType.WARNING)
        return []
    
    query_vec = embed_query(query, model_name)
    
    if query_vec is None or len(query_vec) == 0:
        darwin.log("Failed to embed query — check model availability", LogType.ERROR)
        return []
    
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
            darwin.log(f"USearch search failed, falling back to numpy: {e}", LogType.INFO)
    
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
    """Search using USearch index."""
    chunk_scores = store.search_vector(filename, query_vec, k=top_k)
    
    if not chunk_scores:
        return []
    
    results = []
    for chunk_id_str, score in chunk_scores:
        if score < min_score:
            continue
        
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
            page_end=chunk.page_end,
            heading=chunk.heading,
            chunk_index=chunk.index,
            links=chunk.links,
            overlap_direction=chunk.overlap_direction,
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
    
    similarities = np.dot(vectors, query_vec)
    
    if top_k >= len(similarities):
        top_indices = np.argsort(similarities)[::-1]
    else:
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    
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
            page_end=chunk.page_end,
            heading=chunk.heading,
            chunk_index=chunk.index,
            links=chunk.links,
            overlap_direction=chunk.overlap_direction,
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
    """Determine which search method to use with full validation."""
    has_vector = store.has_vector(filename)
    
    if use_usearch and has_vector:
        try:
            vector_meta = store.load_vector_metadata(filename)
            index_model = vector_meta.get("embedding_model") if vector_meta else None
            
            if index_model and index_model != expected_model:
                darwin.log(
                    f"Model mismatch for {filename}: index={index_model}, current={expected_model}. "
                    f"Falling back to numpy. Run 'darwin embeddings --rerun' to update.",
                    LogType.WARNING
                )
                return "numpy"
            
            return "usearch"
            
        except Exception as e:
            darwin.log(f"Failed to load vector index for {filename}: {e}", LogType.WARNING)
            return "numpy"
    
    if store.has_embeddings(filename):
        return "numpy"
    
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
        darwin.log(f"No chunks found for {filename}", LogType.INFO)
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

    vectors, chunks = store.load_embeddings(filename)
    if vectors and chunks is None:
        darwin.log(f"No embeddings found for {filename}", LogType.INFO)
        return []
    
    
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
    """Search across multiple documents with automatic method selection."""
    if not filenames:
        darwin.log("No files specified for search", LogType.WARNING)
        return []
    
    all_results = []
    per_doc_k = max(5, top_k // len(filenames) + 1)
    
    stats = {"usearch": 0, "numpy": 0, "failed": 0}
    
    config = store.load_config()
    model_name = config.get("embedding_model")
    if model_name is None:
        darwin.log("No embedding model found in config", LogType.WARNING)
        return []
    
    query_vec = embed_query(query, model_name)
    
    if query_vec is None or len(query_vec) == 0:
        darwin.log("Failed to embed query — check model availability", LogType.ERROR)
        return []
    
    for filename in filenames:
        try:
            method = _get_validated_search_method(
                filename=filename,
                store=store,
                expected_model=model_name,
                use_usearch=use_usearch,
            )
            
            if method == "error":
                darwin.log(f"No embeddings for {filename}, skipping", LogType.INFO)
                stats["failed"] += 1
                continue
            
            if method == "usearch":
                doc_results = _execute_usearch_search(
                    filename=filename, query_vec=query_vec,
                    store=store, top_k=per_doc_k, min_score=min_score,
                )
                stats["usearch"] += 1
            else:
                doc_results = _execute_numpy_search(
                    filename=filename, query_vec=query_vec,
                    store=store, top_k=per_doc_k, min_score=min_score,
                )
                stats["numpy"] += 1
            
            if doc_results:
                all_results.extend(doc_results)
            else:
                stats["failed"] += 1
                
        except Exception as e:
            darwin.log(f"Failed to search {filename}: {e}", LogType.ERROR)
            stats["failed"] += 1
    
    darwin.log(
        f"Search completed: {stats['usearch']} USearch · {stats['numpy']} numpy · {stats['failed']} failed",
        LogType.INFO
    )
    
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
    """Search within a single document with automatic method selection."""
    try:
        config = store.load_config()
        model_name = config.get("embedding_model")
        
        if model_name is None:
            darwin.log("No embedding model configured", LogType.ERROR)
            return []
        
        query_vec = embed_query(query, model_name)
        
        if query_vec is None or len(query_vec) == 0:
            darwin.log("Failed to embed query — check model availability", LogType.ERROR)
            return []
        
        method = _get_validated_search_method(
            filename=filename, store=store,
            expected_model=model_name, use_usearch=use_usearch,
        )
        
        if method == "error":
            darwin.log(f"No embeddings found for {filename}", LogType.ERROR)
            darwin.log("Try running 'darwin embeddings' on this document first", LogType.INFO)
            return []
        
        if method == "usearch":
            darwin.log(f"Using USearch index for {filename}", LogType.INFO)
            return _execute_usearch_search(
                filename=filename, query_vec=query_vec,
                store=store, top_k=top_k, min_score=min_score,
            )
        else:
            darwin.log(f"Using numpy fallback for {filename}", LogType.INFO)
            return _execute_numpy_search(
                filename=filename, query_vec=query_vec,
                store=store, top_k=top_k, min_score=min_score,
            )
        
    except FileNotFoundError:
        darwin.log(f"No embeddings found for {filename}", LogType.ERROR)
        return []
    except Exception as e:
        darwin.log(f"Search failed: {e}", LogType.ERROR)
        return []