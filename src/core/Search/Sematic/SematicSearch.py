"""
src/core/Search/Searcher.py
---------------------------
Semantic search over embedded document chunks using cosine similarity.

This module provides pure semantic (dense) retrieval over pre-computed
embeddings. It loads .npz bundles from DarwinStore and performs
exhaustive k-nearest neighbor search using dot product (cosine similarity
since vectors are L2-normalized).

Search Pipeline
---------------
query string
    → prefix_query() [e5 format]
    → embed_query() [single vector]
    → dot product against all chunk vectors
    → top-k indices
    → return SearchResult objects with metadata

Why dot product?
----------------
Vectors are L2-normalized during embedding, so:
    cosine_similarity(a, b) = dot(a, b)

This makes search extremely fast — just a matrix-vector multiplication.

Output
------
List[SearchResult] sorted by relevance (highest similarity first).
Each result contains:
    - chunk text (original + clean)
    - similarity score (0-1, higher is better)
    - metadata (source, page, heading)
    - chunk index

Integration
-----------
Used by:
    - CLI: `darwin search "query"`
    - Agent: retrieve context for LLM
    - Hybrid search: combined with BM25 results
"""

import numpy as np
from typing import List, Optional

from src.core.Chunking.Chunker import Chunk
from src.core.Embedding.Embedder import embed_query
from src.core.Embedding.EmbedderUtils import DEFAULT_MODEL
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI
from src.core.Search.Sematic.SematicSearchUtils   import (reconstruct_chunks,SearchResult)


# ============================================================================
# Core search function
# ============================================================================

def semantic_search(
    query: str,
    vectors: np.ndarray,
    chunks: List[Chunk],
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
    min_score: float = 0.0,
) -> List[SearchResult]:
    """
    Pure semantic (dense) search over pre-computed embeddings.
    
    Args:
        query: Raw user query string
        vectors: np.ndarray of shape (N, D) — L2-normalized chunk embeddings
        chunks: List of Chunk objects corresponding to vectors
        top_k: Number of results to return
        model_name: Embedding model (must match what was used for chunks)
        min_score: Minimum similarity threshold (0-1)
    
    Returns:
        List[SearchResult] sorted by descending similarity
    """
    if len(vectors) == 0 or len(chunks) == 0:
        return []
    
    # Embed query (already L2-normalized)
    query_vec = embed_query(query, model_name)
    
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
        results.append(SearchResult(
            text=chunk.text,
            clean=chunk.clean,
            score=score,
            source=chunk.source,
            page=chunk.page,
            heading=chunk.heading,
            chunk_index=chunk.index,
        ))
    
    return results


# ============================================================================
# Multi-document search
# ============================================================================

def search_across_documents(
    query: str,
    store: DarwinStore,
    filenames: Optional[List[str]] = None,
    top_k: int = 10,
    model_name: str = DEFAULT_MODEL,
    min_score: float = 0.0,
) -> List[SearchResult]:
    """
    Search across multiple documents in the store.
    
    Args:
        query: User query
        store: DarwinStore instance
        filenames: List of documents to search (None = all embedded docs)
        top_k: Total results to return (distributed across docs)
        model_name: Embedding model name
        min_score: Minimum similarity threshold
    
    Returns:
        Combined and re-ranked results from all documents
    """
    # Determine which files to search
    if filenames is None:
        status = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info["components"]["embeddings"]
        ]
    
    if not filenames:
        TerminalUI.warn("No documents with embeddings found")
        TerminalUI.dim("Try running 'darwin embeddings' on this directorys first")
        TerminalUI.dim("Read Embedding/Readme for more info")
        return []
    
    all_results = []
    
    # For small number of docs, proportional top_k per doc
    per_doc_k = max(5, top_k // len(filenames) + 1)
    
    for filename in filenames:
        try:
            data = store.load_embeddings(filename)
            vectors = data["vectors"]
            
            # Reconstruct minimal chunks from stored data
            chunks = reconstruct_chunks(data, filename)
            
            results = semantic_search(
                query=query,
                vectors=vectors,
                chunks=chunks,
                top_k=per_doc_k,
                model_name=model_name,
                min_score=min_score,
            )
            
            all_results.extend(results)
            
        except Exception as e:
            TerminalUI.error(f"Failed to search {filename}: {e}")
            continue
    
    # Re-rank combined results
    all_results.sort(key=lambda r: r.score, reverse=True)
    
    return all_results[:top_k]



# ============================================================================
# Convenience — single document search
# ============================================================================

def search_single_document(
    query: str,
    filename: str,
    store: DarwinStore,
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
    min_score: float = 0.0,
) -> List[SearchResult]:
    """
    Search within a single document.
    
    Args:
        query: User query
        filename: Document to search
        store: DarwinStore instance
        top_k: Number of results
        model_name: Embedding model name
        min_score: Minimum similarity threshold
    
    Returns:
        List[SearchResult] from this document only
    """
    try:
        data = store.load_embeddings(filename)
        if data is None:
            TerminalUI.error(f"No embeddings found for {filename}")
            TerminalUI.dim("Try running 'darwin embeddings' on this document first")
            TerminalUI.dim("Read Embedding/Readme for more info")
            return []
        vectors = data["vectors"]
        chunks = reconstruct_chunks(data, filename)
        
        return semantic_search(
            query=query,
            vectors=vectors,
            chunks=chunks,
            top_k=top_k,
            model_name=model_name,
            min_score=min_score,
        )
        
    except FileNotFoundError:
        TerminalUI.error(f"No embeddings found for {filename}")
        return []
    except Exception as e:
        TerminalUI.error(f"Search failed: {e}")
        return []