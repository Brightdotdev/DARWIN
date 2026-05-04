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

from src.features.Ingestion.Chunking.lib import Chunk
from src.features.Indexing.Embedding.Embedder import embed_query
from src.features.Storage.DarwinStore import DarwinStore
from src.features.Retrival.Semantic.SematicSearchUtils import SematicSearchResult
from src.core.Logger.DarwinLogger import darwin, LogType


# ============================================================================
# Core search functions (USearch + Fallback)
# ============================================================================

def _usearch_search(
    query_vec: np.ndarray,
    store: DarwinStore,
    filename: str,
    chunks: List[Chunk],
    top_k: int,
    min_score: float,
) -> List[SematicSearchResult]:
    """Search using USearch index."""
    darwin.log(f"Searching your {filename}")
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
    if vectors is None or vectors.size == 0:
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
# Validation
# ============================================================================

def _get_search_method(
    filename: str,
    store: DarwinStore,
    expected_model: str,
    use_usearch: bool = True,
) -> str:
    """Determine which search method to use (usearch, numpy, or error)."""
    has_vector = store.has_vector(filename)
    
    if use_usearch and has_vector:
        try:
            vector_meta = store.load_vector_metadata(filename)
            index_model = vector_meta.get("embedding_model") if vector_meta else None
            
            if index_model and index_model != expected_model:
                darwin.log(
                    f"Model mismatch for {filename}: index={index_model}, "
                    f"current={expected_model}. Falling back to numpy.",
                    LogType.WARNING
                )
                return "numpy"
            
            return "usearch"
            
        except Exception as e:
            darwin.log(f"Failed to load vector index for {filename}: {e}", LogType.WARNING)
            return "numpy"
    
    if store.has_embeddings(filename):
        return "numpy"
    
    return "none"


def _embed_query_safe(query: str, model_name: str) -> Optional[np.ndarray]:
    """Safely embed query with error handling."""
    if not model_name:
        darwin.log("No embedding model configured", LogType.ERROR)
        return None
    
    query_vec = embed_query(query, model_name)
    
    if query_vec is None or len(query_vec) == 0:
        darwin.log("Failed to embed query — check model availability", LogType.ERROR)
        return None
    
    return query_vec


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
        # Load config and validate
        config = store.load_config()
        model_name = config.get("embedding_model")
        
        # Check if file is searchable first
        method = _get_search_method(
            filename=filename,
            store=store,
            expected_model=model_name or "",
            use_usearch=use_usearch,
        )
        
        if method == "none":
            darwin.log(f"No embeddings found for {filename}", LogType.ERROR)
            darwin.log("Try running 'darwin embeddings' on this document first", LogType.INFO)
            return []
        
        # Embed query (only once we know file is searchable)
        query_vec = _embed_query_safe(query, model_name)
        if query_vec is None:
            return []
        
        # Load embeddings
        darwin.log("Loading Embeddings...")
        vectors, chunks = store.load_embeddings(filename)
        darwin.log("Loaded Embeddings...")
        
        if chunks is None or len(chunks) == 0:
            darwin.log(f"No chunks found for {filename}", LogType.WARNING)
            return []
        
        # Execute search
        if method == "usearch":
            darwin.log(f"Using USearch index for {filename}", LogType.INFO)
            return _usearch_search(
                query_vec=query_vec,
                store=store,
                filename=filename,
                chunks=chunks,
                top_k=top_k,
                min_score=min_score,
            )
        else:
            darwin.log(f"Using numpy fallback for {filename}", LogType.INFO)
            return _numpy_search(
                query_vec=query_vec,
                vectors=vectors,
                chunks=chunks,
                top_k=top_k,
                min_score=min_score,
            )
        
    except FileNotFoundError:
        darwin.log(f"No embeddings found for {filename}", LogType.ERROR)
        return []
    except Exception as e:
        darwin.log(f"Search failed for {filename}: {e}", LogType.ERROR)
        return []


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
    
    # Distribute top_k across documents (minimum 5 per doc)
    per_doc_k = max(5, top_k // len(filenames) + 1)
    
    stats = {"searched": 0, "failed": 0, "total_results": 0}
    
    for filename in filenames:
        try:
            doc_results = search_single_document(
                query=query,
                filename=filename,
                store=store,
                top_k=per_doc_k,
                min_score=min_score,
                use_usearch=use_usearch,
            )
            
            if doc_results:
                all_results.extend(doc_results)
                stats["searched"] += 1
                stats["total_results"] += len(doc_results)
            else:
                stats["failed"] += 1
                
        except Exception as e:
            darwin.log(f"Failed to search {filename}: {e}", LogType.ERROR)
            stats["failed"] += 1
    
    darwin.log(
        f"Search completed: {stats['searched']} files searched, "
        f"{stats['failed']} failed, {stats['total_results']} results found",
        LogType.INFO
    )
    
    # Sort by score descending and return top_k
    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:top_k]