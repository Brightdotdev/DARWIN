"""
src/core/Search/KeywordUtils.py
-------------------------------
Utilities for BM25 keyword search: query tokenization,
result formatting, and argument handling.

Query Tokenization
------------------
Uses BlingFire text_to_words() — same tokenizer as indexing.
Query gets same treatment as chunks:
    - Lowercase
    - Remove standard English stopwords
    - Minimum token length 2
    - Strip punctuation and numbers

This ensures query tokens match the BM25 index vocabulary.
"""

from dataclasses import dataclass
from typing import List, Set, Optional
import numpy as np
from typing import List, Optional, Dict
from math import log

from src.core.BM25.BM25Utils import STANDARD_STOPS, MIN_TOKEN_LENGTH



# ============================================================================
# Data model
# ============================================================================

@dataclass
class KeywordSearchResult:
    """Single BM25 search result with all metadata."""
    text: str           # Original text (for LLM context)
    clean: str          # Cleaned text (what was indexed)
    score: float        # BM25 score (not normalized to 0-1)
    source: str         # Document filename
    page: int           # Page number
    heading: str        # Section heading
    chunk_index: int    # Original chunk position
    
    def __repr__(self) -> str:
        return f"KeywordSearchResult(score={self.score:.3f}, source='{self.source}', page={self.page})"


# ============================================================================
# Constants
# ============================================================================

DEFAULT_TOP_K = 5
DEFAULT_K1 = 1.5      # BM25 term frequency saturation
DEFAULT_B = 0.75      # BM25 length normalization


# ============================================================================
# Args
# ============================================================================

@dataclass
class KeywordSearchArgs:
    query: str
    file: Optional[str] = None
    project: Optional[str] = None
    top_k: int = DEFAULT_TOP_K
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    inspect: bool = False
    json_output: bool = False


# ============================================================================
# Query tokenization
# ============================================================================

def tokenize_query(
    query: str,
    standard_stops: Set[str] = STANDARD_STOPS,
) -> List[str]:
    """
    Tokenize search query for BM25 matching.
    
    Uses same tokenization as indexing to ensure vocabulary match.
    
    Args:
        query: Raw user query string
        standard_stops: Standard English stopwords to remove
    
    Returns:
        List of lowercase tokens
    """
    try:
        from blingfire import text_to_words
        
        words = text_to_words(query).split()
        
        tokens = []
        for w in words:
            w_lower = w.lower().strip()
            
            if (len(w_lower) >= MIN_TOKEN_LENGTH and
                w_lower not in standard_stops and
                not w_lower.isdigit()):
                tokens.append(w_lower)
        
        return tokens
        
    except ImportError:
        # Fallback to simple regex
        import re
        words = re.findall(r"[a-zA-Z]{2,}", query.lower())
        return [w for w in words if w not in standard_stops]


# ============================================================================
# Result formatting
# ============================================================================

def format_keyword_results(results: list, json_output: bool = False) -> str:
    """Format BM25 search results for display or JSON export."""
    import json
    
    if json_output:
        output = []
        for r in results:
            output.append({
                "score": round(r.score, 4),
                "source": r.source,
                "page": r.page,
                "heading": r.heading,
                "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
            })
        return json.dumps(output, indent=2)
    
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.score:.3f} | {r.source} | page {r.page}")
        if r.heading:
            lines.append(f"    └─ {r.heading}")
        preview = r.text[:200].replace('\n', ' ').strip()
        lines.append(f"       {preview}...")
        lines.append("")
    
    return "\n".join(lines)


def to_llm_context(results: list, max_chunks: int = 5) -> str:
    """
    Convert BM25 results to LLM-ready context string.
    
    Format:
        [Document: filename, Page: X]
        Heading: section title
        Content: chunk text
        ---
    """
    context_parts = []
    
    for r in results[:max_chunks]:
        part = f"[Document: {r.source}, Page: {r.page}]"
        if r.heading:
            part += f"\nHeading: {r.heading}"
        part += f"\nContent: {r.text}\n---"
        context_parts.append(part)
    
    return "\n".join(context_parts)






# ============================================================================
# BM25 scoring (from saved data)
# ============================================================================
def compute_bm25_scores(
    query_tokens: List[str],
    data,  # This is a BM25Data object, not a dict!
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> np.ndarray:
    """
    Compute BM25 scores for all chunks from saved index data.
    
    Args:
        query_tokens: Tokenized query
        data: BM25Data object from store.load_bm25()
        k1: Term frequency saturation parameter
        b: Length normalization parameter
    
    Returns:
        np.ndarray of scores, shape (N,)
    """
    # Access attributes, not dict keys
    tokens_corpus = data.tokens          # ← data.tokens (not data["tokens"])
    doc_freqs = data.doc_freqs           # ← data.doc_freqs
    avgdl = data.avgdl                   # ← data.avgdl
    doc_lengths = data.doc_lengths       # ← data.doc_lengths
    N = data.chunk_count                 # ← data.chunk_count
    
    scores = np.zeros(N)
    
    for term in query_tokens:
        # IDF: inverse document frequency
        df = doc_freqs.get(term, 0)
        if df == 0:
            continue
        
        idf = log((N - df + 0.5) / (df + 0.5) + 1)
        
        # TF: term frequency per document
        for i, doc_tokens in enumerate(tokens_corpus):
            freq = doc_tokens.count(term)
            if freq == 0:
                continue
            
            # BM25 TF component
            len_norm = 1 - b + b * (doc_lengths[i] / avgdl)
            tf = (freq * (k1 + 1)) / (freq + k1 * len_norm)
            
            scores[i] += idf * tf
    
    return scores