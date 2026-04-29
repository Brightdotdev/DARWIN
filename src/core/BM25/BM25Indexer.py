"""
src/core/BM25/BM25Indexer.py
-----------------------------
BM25 sparse indexing for keyword retrieval.

Indexing pipeline
-----------------
chunks
    → detect domain stopwords
    → tokenize every chunk
    → accumulate doc_freqs, doc_lengths
    → precompute IDFs
    → save BM25 index via store
    → build and save term bank for query expansion
    → embed terms and save term embeddings
"""

from collections import Counter
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from src.core.Chunking.ChunkerUtils import Chunk
from src.core.BM25.BM25Utils import (
    BM25Data,
    DOMAIN_STOP_THRESHOLD,
    DEFAULT_IDF_SMOOTHING,
    detect_domain_stopwords,
    tokenize_text,
    stream_tokenize_chunks,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType


def index_chunks_for_bm25(
    chunks: List[Chunk],
    filename: str,
    threshold: float = DOMAIN_STOP_THRESHOLD,
    idf_smoothing: float = DEFAULT_IDF_SMOOTHING,
) -> Optional[BM25Data]:
    """Build BM25Data from a list of Chunk objects."""
    if not chunks:
        darwin.log(f"No chunks to index for {filename}", LogType.WARNING)
        return None

    darwin.log(f"Building BM25 index for {filename} ({len(chunks)} chunks)", LogType.INFO)

    # 1. Domain stopwords
    darwin.progress_start(f"bm25_{filename}", 3, f"BM25: {filename}")
    darwin.progress_update(f"bm25_{filename}", 1, "Detecting domain stopwords")
    
    domain_stops = detect_domain_stopwords(chunks, threshold)

    # 2. Tokenize
    darwin.progress_update(f"bm25_{filename}", 2, "Tokenizing chunks")
    
    valid_tokens = []
    doc_lengths = []
    doc_freqs: Counter = Counter()

    should_stream = len(chunks) > 5_000
    if should_stream:
        darwin.log(f"Streaming mode ({len(chunks)} chunks)", LogType.INFO)

    def _accumulate(tokens: List[str]) -> None:
        if not tokens:
            return
        valid_tokens.append(tokens)
        doc_lengths.append(len(tokens))
        doc_freqs.update(set(tokens))

    if should_stream:
        for batch_results in stream_tokenize_chunks(chunks, domain_stops):
            for tokens in batch_results:
                _accumulate(tokens)
    else:
        for chunk in chunks:
            tokens = tokenize_text(chunk.clean, domain_stops=domain_stops)
            _accumulate(tokens)

    # 3. Package
    avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

    data = BM25Data(
        tokens=valid_tokens,
        doc_freqs=dict(doc_freqs),
        avgdl=avgdl,
        doc_lengths=doc_lengths,
        domain_stopwords=sorted(domain_stops),
        threshold=threshold,
        chunk_count=len(valid_tokens),
        vocab_size=len(doc_freqs),
        idf_smoothing=idf_smoothing,
    )

    darwin.progress_end(f"bm25_{filename}")

    suffix = f" [IDF s={idf_smoothing}]" if idf_smoothing != DEFAULT_IDF_SMOOTHING else ""
    suffix += " [streaming]" if should_stream else ""
    darwin.log(
        f"Index built: {data.chunk_count} chunks, {data.vocab_size} terms, "
        f"{len(domain_stops)} domain stops{suffix}",
        LogType.SUCCESS
    )
    return data


def run_bm25_indexing_pipeline(
    filename: str,
    chunks: List[Chunk],
    store: DarwinStore,
    threshold: float = DOMAIN_STOP_THRESHOLD,
    idf_smoothing: float = DEFAULT_IDF_SMOOTHING,
) -> Optional[BM25Data]:
    """
    Build and persist a BM25 index + term bank for one document.
    """
    if not chunks:
        darwin.log(f"No chunks to index for {filename}", LogType.WARNING)
        return None

    data = index_chunks_for_bm25(
        chunks, filename, threshold=threshold, idf_smoothing=idf_smoothing
    )

    if data:
        # Save BM25 index
        store.save_bm25(filename, data)
        store.save_stopwords(filename, set(data.domain_stopwords))

        # ── Build and save term bank ──────────────────────────────────────
        try:
            from src.core.BM25.TermBank import build_term_bank, embed_terms

            terms, term_freqs, chunk_count = build_term_bank(
                chunks=chunks,
                domain_stopwords=set(data.domain_stopwords),
                min_freq=5,
                max_terms=2000,
            )

            if terms:
                # Save raw terms (model-agnostic)
                store.save_term_bank(filename, terms, term_freqs, chunk_count)

                # Embed and save term embeddings (model-specific)
                term_embeddings = embed_terms(terms)
                if term_embeddings is not None:
                    store.save_term_embeddings(filename, term_embeddings, terms)
                else:
                    darwin.log(
                        f"Term embeddings not created — query expansion will be unavailable "
                        f"until embedding model is available",
                        LogType.INFO
                    )
            else:
                darwin.log(f"No terms met threshold for term bank", LogType.INFO)

        except Exception as e:
            darwin.log(f"Term bank creation failed (non-fatal): {e}", LogType.WARNING)

    return data


def inspect_bm25_index(data: BM25Data, show_samples: int = 3) -> None:
    """Print a human-readable summary of a BM25Data object."""
    darwin.log("")
    darwin.log("── BM25 Index Inspection ──", LogType.INFO)
    darwin.log("")

    darwin.log("  metric              value", LogType.INFO)
    darwin.log("  ──────────────────  ─────", LogType.INFO)
    darwin.log(f"  chunks indexed      {data.chunk_count}", LogType.INFO)
    darwin.log(f"  vocabulary size     {data.vocab_size}", LogType.INFO)
    darwin.log(f"  avg chunk length    {data.avgdl:.1f} tokens", LogType.INFO)
    darwin.log(f"  domain stopwords    {len(data.domain_stopwords)}", LogType.INFO)
    darwin.log(f"  threshold           {data.threshold:.2f}", LogType.INFO)
    darwin.log(f"  IDF smoothing       {data.idf_smoothing}", LogType.INFO)
    darwin.log("")

    if data.domain_stopwords:
        preview = ", ".join(data.domain_stopwords[:15])
        darwin.log(f"  Domain Stopwords ({len(data.domain_stopwords)} total): {preview}", LogType.INFO)
        darwin.log("")

    darwin.log(f"── Token samples (first {show_samples} chunks) ──", LogType.INFO)
    for i, tokens in enumerate(data.tokens[:show_samples]):
        preview = " ".join(tokens[:20])
        if len(tokens) > 20:
            preview += f" … (+{len(tokens) - 20})"
        darwin.log(f"  Chunk {i}: {len(tokens)} tokens", LogType.INFO)
        darwin.log(f"    {preview}", LogType.INFO)
        darwin.log("")

    idfs = data.precomputed_idfs or {}
    if idfs:
        darwin.log("── Top discriminative terms (highest IDF) ──", LogType.INFO)
        for term, idf_val in sorted(idfs.items(), key=lambda x: x[1], reverse=True)[:10]:
            df = data.doc_freqs.get(term, 0)
            darwin.log(f"  {term}: IDF={idf_val:.3f}  (in {df} chunks)", LogType.INFO)

    darwin.log("─" * 60, LogType.INFO)
    darwin.log("")


def create_bm25_scorer(data: BM25Data, k1: float = 1.5, b: float = 0.75):
    """Build a BM25Okapi scorer from stored token data."""
    from rank_bm25 import BM25Okapi
    return BM25Okapi(corpus=data.tokens, k1=k1, b=b)