"""
src/core/Search/KeywordSearcher.py
----------------------------------
Pure BM25 (sparse) retrieval over chunked document text.

This module provides exact keyword matching using BM25Okapi with
pre-computed indexes from the BM25 module. It loads .pkl bundles
and performs term-based retrieval.

Search Pipeline
---------------
query string
    → tokenize (same as indexing)
    → BM25 scoring against all chunks
    → top-k indices
    → return SearchResult objects with metadata

BM25 Formula
------------
score(D, Q) = Σ IDF(qi) × TF(qi, D)

Where:
    IDF(qi) = log((N - df + 0.5) / (df + 0.5) + 1)
    TF(qi, D) = (freq × (k1 + 1)) / (freq + k1 × (1 - b + b × len/avgdl))

Parameters:
    k1 = 1.5 (default) — term frequency saturation
    b  = 0.75 (default) — length normalization

Integration
-----------
Used by:
    - CLI: `darwin keyword "query"`
    - Hybrid search: combined with semantic results via RRF
"""

import numpy as np
from typing import List
from src.core.Chunking.Chunker import Chunk
from src.core.Search.Keyword.KeywordUtilities import KeywordSearchResult
from src.lib.Terminal import TerminalUI



"""
src/core/BM25/BM25Search.py
----------------------------
Search over a loaded BM25Data index.

Two search modes
----------------
    keyword   — standard BM25 scoring over individual terms
    phrase    — keyword score + phrase-match bonus for "quoted terms"

Typical call from a retrieval pipeline
---------------------------------------
    store  = DarwinStore(project_dir)
    chunks = store.load_chunks(filename)
    data   = store.load_bm25(filename)

    results = bm25_search(query, data, chunks, top_k=10)
    for r in results:
        print(r.score, r.chunk.heading, r.chunk.text[:200])

Why not use rank_bm25.BM25Okapi directly?
------------------------------------------
BM25Okapi re-tokenizes everything on every query and holds the full
token corpus in memory.  We store raw frequencies so we can score
directly against doc_freqs + precomputed_idfs without rebuilding the
corpus object.  This also lets us tune k1/b at query time without
reindexing.
"""


from src.core.BM25.BM25Utils import (STANDARD_STOPS)
from src.core.Search.Keyword.KeywordUtilities import (calculate_bm25)
from src.lib.Terminal import TerminalUI
from src.core.Storage.DarwinStore import DarwinStore

# ── Public search API ──────────────────────────────────────────────────────────

def search_single_document_keyword(
    query:          str,
    filename:       str,
    store : DarwinStore,
    top_k:          int   = 10,
    k1:             float = 1.5,
    b:              float = 0.75,
    min_score:      float = 0.0
) -> List[KeywordSearchResult]:
    """
    Search a single document and return results sorted by score.
    """ 
    

    if not query.strip():
        return []


    chunks = store.load_chunks(filename)
    data   = store.load_bm25(filename)
    if not chunks or not data:
        TerminalUI.dim(f"Skipping {filename} — no Data provided for keyword search for this file")
        TerminalUI.dim("Try running `darwin index` first")
        return []
    # Simple tokenization - treat everything as individual terms
    from src.core.BM25.BM25Utils import tokenize_text
    stopwords = set(data.domain_stopwords).union(STANDARD_STOPS)
    query_terms = tokenize_text(query, domain_stops=stopwords)

    if not query_terms:
        TerminalUI.warn("Query produced no usable terms")
        return []

    results = []
    for chunk_idx in range(len(data.tokens)):
        score, matched = calculate_bm25(query_terms, chunk_idx, data, k1, b)
        
        if score >= min_score:
            results.append(KeywordSearchResult(
                text=chunks[chunk_idx].text,
                clean=chunks[chunk_idx].clean,
                score=score,
                source=filename,
                page=chunks[chunk_idx].page,
                heading=chunks[chunk_idx].heading,
                chunk_index=chunk_idx,
                matched_terms=matched,
                links=chunks[chunk_idx].links if chunks[chunk_idx].links else [],
                is_overlap=chunks[chunk_idx].is_overlap
            ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


# ── Multi-file search ──────────────────────────────────────────────────────────

def search_across_documents_keyword(
    filenames: List[str],
    query:     str,
    store:     DarwinStore,
    top_k:     int   = 10,
    k1:        float = 1.5,
    b:         float = 0.75,
) -> List[KeywordSearchResult]:
    all_results: List[KeywordSearchResult] = []

    for filename in filenames:
        if not store.has_bm25(filename):
            TerminalUI.dim(f"Skipping {filename} — no BM25 index")
            continue
        if not store.has_chunks(filename):
            TerminalUI.dim(f"Skipping {filename} — no chunks")
            continue
        try:
            results = search_single_document_keyword(
                query=query,
                filename=filename,
                 store=store, top_k=top_k, k1=k1, b=b
            )
            all_results.extend(results)
        except Exception as e:
            TerminalUI.error(f"Search failed for {filename}: {e}")

    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:top_k]