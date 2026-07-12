# BM25Indexer.py
from collections import Counter
from typing import List, Optional, Any, Dict
from src.core.Pipeline.Indexing.BM25.lib import (
    BM25Data,
    DEFAULT_IDF_SMOOTHING,
    DOMAIN_STOP_THRESHOLD,
)
from src.core.Pipeline.Indexing.BM25.BM25Utils import (
    detect_domain_stopwords,
    tokenize_text,
    stream_tokenize_chunks,
)
from src.core.Storage import DarwinStore
from src.core.Logger.DarwinLogger import darwin


def index_chunks_for_bm25(
    chunks: List[Dict[str, Any]],
    filename: str,
    threshold: float = DOMAIN_STOP_THRESHOLD,
    idf_smoothing: float = DEFAULT_IDF_SMOOTHING,
) -> Optional[BM25Data]:
    if not chunks:
        darwin.warning(f"No chunks to index for {filename}")
        return None

    darwin.info(
        f"Building BM25 index for {filename} ({len(chunks)} chunks)"
    )
    darwin.progress_start(f"bm25_{filename}", 3, f"BM25: {filename}")
    darwin.progress_update(f"bm25_{filename}", 1, "Detecting domain stopwords")
    domain_stops = detect_domain_stopwords(chunks, threshold)

    darwin.progress_update(f"bm25_{filename}", 2, "Tokenizing chunks")
    valid_tokens = []
    doc_lengths = []
    doc_freqs: Counter = Counter()

    should_stream = len(chunks) > 5_000
    if should_stream:
        darwin.info(f"Using streaming tokenization for {len(chunks)} chunks")

    # FIX: Always append tokens and length, even if tokens is empty
    def _accumulate(tokens: List[str]) -> None:
        valid_tokens.append(tokens)  # always append
        doc_lengths.append(len(tokens))  # always append length
        if tokens:
            doc_freqs.update(set(tokens))  # update only if non-empty

    if should_stream:
        for batch_results in stream_tokenize_chunks(chunks, domain_stops):
            for tokens in batch_results:
                _accumulate(tokens)
    else:
        for chunk in chunks:
            tokens = tokenize_text(chunk.get("clean"), domain_stops=domain_stops)
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
        chunk_indices=[
            c.get("index", i) for i, c in enumerate(chunks)
        ],  # fallback to position
    )
    darwin.progress_update(f"bm25_{filename}", 3, "Finishing Up..")
    darwin.progress_end(f"bm25_{filename}")

    suffix = (
        f" [IDF s={idf_smoothing}]" if idf_smoothing != DEFAULT_IDF_SMOOTHING else ""
    )
    suffix += " [streaming]" if should_stream else ""
    darwin.success(
        f"Index built: {data.chunk_count} chunks, {data.vocab_size} terms, "
        f"{len(domain_stops)} domain stops{suffix}"
    )
    return data


def run_bm25_indexing_pipeline(
    filename: str,
    chunks: List[Dict[str, Any]],
    store: DarwinStore,
    threshold: float = DOMAIN_STOP_THRESHOLD,
    idf_smoothing: float = DEFAULT_IDF_SMOOTHING,
) -> Optional[BM25Data]:
    if not chunks:
        darwin.warning(f"No chunks to index for {filename}")
        return None

    data = index_chunks_for_bm25(
        chunks, filename, threshold=threshold, idf_smoothing=idf_smoothing
    )

    if data:
        store.save_bm25(filename, data)
        store.save_stopwords(filename, set(data.domain_stopwords))
    return data
