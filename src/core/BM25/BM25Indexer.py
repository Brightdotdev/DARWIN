"""
src/core/BM25/BM25Indexer.py
-----------------------------
BM25 sparse indexing for keyword retrieval.

Indexing pipeline
-----------------
chunks
    → detect domain stopwords          (BM25Utils.detect_domain_stopwords)
    → tokenize every chunk             (streaming or standard)
    → accumulate doc_freqs, doc_lengths
    → precompute IDFs                  (in BM25Data.__post_init__)
    → save via store.save_bm25()

Why store raw frequencies, not scores?
---------------------------------------
BM25 scoring needs k1 and b at query time:
    TF  = (freq × (k1 + 1)) / (freq + k1 × (1 − b + b × len/avgdl))
    IDF = log((N − df + s) / (df + s) + 1)

Saving raw counts lets us tune k1/b/smoothing at search time without
rebuilding the index. precomputed_idfs uses the index-time smoothing
value as a fast default.

Save format
-----------
Two files per document:

    bm25.npz        — compressed numpy: token arrays + doc_lengths
    bm25_meta.json  — human-readable: doc_freqs, IDFs, stopwords, scalars

Kept separate because numpy is efficient for variable-length arrays and
JSON is efficient for key-value lookups. Pickle was rejected: it breaks
on class renames, field additions, and Python version changes.
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
from src.lib.Terminal import TerminalUI




# ── Core indexing ──────────────────────────────────────────────────────────────
def index_chunks_for_bm25(
    chunks: List[Chunk],
    filename: str,
    threshold: float = DOMAIN_STOP_THRESHOLD,
    idf_smoothing: float = DEFAULT_IDF_SMOOTHING,
) -> Optional[BM25Data]:
    """
    Build BM25Data from a list of Chunk objects.

    Args
    ----
    chunks        Chunk objects from chunk_document()
    filename      Source document name — used only for log messages
    threshold     Domain-stopword threshold (0–1). 0.3 = terms in more
                  than 30% of chunks are treated as noise.
    idf_smoothing Smoothing parameter s in IDF formula. Stored in the
                  index so search time can use it as the default.

    Returns
    -------
    BM25Data ready to pass to save_bm25(), or None if chunks is empty.
    """
    if not chunks:
        TerminalUI.warn(f"No chunks to index for {filename}")
        return None

    TerminalUI.bold(f"Building BM25 index for {filename} ({len(chunks)} chunks)")

    # 1. Domain stopwords ─────────────────────────────────────────────────────
    t            = TerminalUI.timer("domain_stopwords")
    domain_stops = detect_domain_stopwords(chunks, threshold)
    t.done()

    # 2. Tokenize ─────────────────────────────────────────────────────────────
    t              = TerminalUI.timer("tokenization")
    valid_tokens   = []
    doc_lengths    = []
    doc_freqs: Counter = Counter()

    should_stream = len(chunks) > 5_000
    if should_stream:
        TerminalUI.dim(f"Streaming mode ({len(chunks)} chunks)")

    def _accumulate(tokens: List[str]) -> None:
        """Accumulate token statistics for a single chunk."""
        if not tokens:
            return
        valid_tokens.append(tokens)
        doc_lengths.append(len(tokens))
        doc_freqs.update(set(tokens))

    if should_stream:
        with TerminalUI.progress_bar(
            description="Tokenizing chunks (streaming)",
            total=len(chunks),
            transient=False,
        ) as (progress, task_id):
            
            for batch_results in stream_tokenize_chunks(chunks, domain_stops):
                for tokens in batch_results:
                    _accumulate(tokens)
                progress.advance(task_id, len(batch_results)) 
                
    else:
        with TerminalUI.progress_bar(
            description="Tokenizing chunks",
            total=len(chunks),
            transient=False,
        ) as (progress, task_id):
            
            for chunk in chunks:
                tokens = tokenize_text(chunk.clean, domain_stops=domain_stops)
                _accumulate(tokens)
                progress.advance(task_id, 1)

    t.done()

    # 3. Package ──────────────────────────────────────────────────────────────
    avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

    data = BM25Data(
        tokens           = valid_tokens,
        doc_freqs        = dict(doc_freqs),
        avgdl            = avgdl,
        doc_lengths      = doc_lengths,
        domain_stopwords = sorted(domain_stops),
        threshold        = threshold,
        chunk_count      = len(valid_tokens),
        vocab_size       = len(doc_freqs),
        idf_smoothing    = idf_smoothing,
    )

    suffix = f" [IDF s={idf_smoothing}]" if idf_smoothing != DEFAULT_IDF_SMOOTHING else ""
    suffix += " [streaming]" if should_stream else ""
    TerminalUI.success(
        f"Index built: {data.chunk_count} chunks, "
        f"{data.vocab_size} terms, "
        f"{len(domain_stops)} domain stops{suffix}"
    )
    return data

# ── Pipeline entry point ───────────────────────────────────────────────────────

def run_bm25_indexing_pipeline(
    filename:      str,
    chunks:        List[Chunk],
    store:         DarwinStore,
    threshold:     float = DOMAIN_STOP_THRESHOLD,
    idf_smoothing: float = DEFAULT_IDF_SMOOTHING,
) -> Optional[BM25Data]:
    """
    Build and persist a BM25 index for one document.

    Freshness check is delegated to save_bm25() — no duplicate check here.
    Returns BM25Data on success, None if chunks is empty.
    """
    if not chunks:
        TerminalUI.warn(f"No chunks to index for {filename}")
        return None

    data = index_chunks_for_bm25(chunks, filename, threshold=threshold, idf_smoothing=idf_smoothing)

    if data:
        store.save_bm25(filename, data)
        store.save_stopwords(filename, set(data.domain_stopwords))

    return data


# ── Inspection ─────────────────────────────────────────────────────────────────

def inspect_bm25_index(data: BM25Data, show_samples: int = 3) -> None:
    """Print a human-readable summary of a BM25Data object."""
    TerminalUI.rule("BM25 Index Inspection")
    TerminalUI.blank()

    TerminalUI.table(
        headers=["Metric", "Value"],
        rows=[
            ["Chunks indexed",   str(data.chunk_count)],
            ["Vocabulary size",  str(data.vocab_size)],
            ["Avg chunk length", f"{data.avgdl:.1f} tokens"],
            ["Domain stopwords", str(len(data.domain_stopwords))],
            ["Threshold",        f"{data.threshold:.2f}"],
            ["IDF smoothing",    str(data.idf_smoothing)],
        ],
    )
    TerminalUI.blank()

    if data.domain_stopwords:
        TerminalUI.panel(
            ", ".join(data.domain_stopwords[:15]),
            title=f"Domain Stopwords ({len(data.domain_stopwords)} total)",
            border_style="dim",
        )
    TerminalUI.blank()

    TerminalUI.dim(f"── Token samples (first {show_samples} chunks) ──")
    for i, tokens in enumerate(data.tokens[:show_samples]):
        preview = " ".join(tokens[:20])
        if len(tokens) > 20:
            preview += f" … (+{len(tokens) - 20})"
        TerminalUI.info(f"Chunk {i}", f"{len(tokens)} tokens")
        TerminalUI.dim(f"  {preview}")
        TerminalUI.blank()

    idfs = data.precomputed_idfs or {}
    if idfs:
        TerminalUI.text("── Top discriminative terms (highest IDF) ──")
        for term, idf in sorted(idfs.items(), key=lambda x: x[1], reverse=True)[:10]:
            df = data.doc_freqs.get(term, 0)
            TerminalUI.dim(f"  {term}: IDF={idf:.3f}  (in {df} chunks)")

    TerminalUI.rule()


def create_bm25_scorer(data: BM25Data, k1: float = 1.5, b: float = 0.75):
    """
    Build a BM25Okapi scorer from stored token data.
    For offline inspection and unit tests only — not production search.
    """
    from rank_bm25 import BM25Okapi
    return BM25Okapi(corpus=data.tokens, k1=k1, b=b)