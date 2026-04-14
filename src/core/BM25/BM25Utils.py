"""
src/core/BM25/BM25Utils.py
--------------------------
Utilities for BM25 indexing: tokenization, stopword detection,
and argument handling.

Stopword Detection
------------------
Standard English stopwords are hardcoded (small list).
Domain stopwords are detected per-document:
    - Terms appearing in >30% of chunks are noise
    - "photosynthesis" in a biology textbook = domain stopword
    - Removed before BM25 indexing to improve precision

Tokenization
------------
Uses BlingFire text_to_words() for fast, consistent tokenization.
No lemmatization — BM25 + dense hybrid covers semantic variation.
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Optional

from src.core.Chunking.Chunker import Chunk
from src.lib.Terminal import TerminalUI


# ============================================================================
# Constants
# ============================================================================

# Standard English stopwords — small, hardcoded, no nltk dependency
STANDARD_STOPS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "that", "these", "those",
    "or", "nor", "but", "not", "so", "yet", "if", "then", "than", "when",
    "i", "you", "we", "they", "my", "your", "our", "their", "me", "him",
    "her", "us", "them", "his", "hers", "ours", "yours", "theirs", "mine",
}

# Domain stopword threshold: terms in >30% of chunks become stopwords
DOMAIN_STOP_THRESHOLD = 0.3

# Minimum token length (skip single chars, numbers)
MIN_TOKEN_LENGTH = 2


# ============================================================================
# Args
# ============================================================================

@dataclass
class BM25IndexingArgs:
    target: str
    file: Optional[str] = None
    project: Optional[str] = None
    threshold: float = DOMAIN_STOP_THRESHOLD
    inspect: bool = False
    rerun: bool = False


# ============================================================================
# Tokenization
# ============================================================================

def tokenize_text(
    text: str,
    standard_stops: Set[str] = STANDARD_STOPS,
    domain_stops: Set[str] = set(),
) -> List[str]:
    """
    Tokenize text for BM25 indexing using BlingFire.
    
    Args:
        text: Raw text to tokenize (chunk.clean)
        standard_stops: Set of standard English stopwords
        domain_stops: Set of document-specific stopwords
    
    Returns:
        List of lowercase tokens
    """
    try:
        from blingfire import text_to_words
        
        # BlingFire handles punctuation, hyphenation, contractions
        words = text_to_words(text).split()
        
        tokens = []
        for w in words:
            w_lower = w.lower().strip()
            
            # Filter: length, digits, stopwords
            if (len(w_lower) >= MIN_TOKEN_LENGTH and
                w_lower not in standard_stops and
                w_lower not in domain_stops and
                not w_lower.isdigit()):
                tokens.append(w_lower)
        
        return tokens
        
    except ImportError:
        # Fallback to simple regex (should never happen, but graceful)
        TerminalUI.warn("BlingFire not available, using regex fallback")
        words = re.findall(r"[a-zA-Z]{2,}", text.lower())
        return [w for w in words if w not in standard_stops and w not in domain_stops]


def tokenize_chunks(
    chunks: List[Chunk],
    domain_stops: Set[str] = set(),
) -> List[List[str]]:
    """
    Tokenize all chunks for BM25 indexing.
    
    Args:
        chunks: List of Chunk objects
        domain_stops: Document-specific stopwords to filter
    
    Returns:
        List of tokenized chunks (each chunk = list of tokens)
    """
    return [tokenize_text(c.clean, domain_stops=domain_stops) for c in chunks]


# ============================================================================
# Domain stopword detection
# ============================================================================

def detect_domain_stopwords(
    chunks: List[Chunk],
    threshold: float = DOMAIN_STOP_THRESHOLD,
) -> Set[str]:
    """
    Detect document-specific stopwords based on chunk frequency.
    
    Terms appearing in >threshold% of chunks are considered noise
    because they don't discriminate between chunks.
    
    Args:
        chunks: List of Chunk objects
        threshold: Fraction of chunks a term must appear in to be a stopword
    
    Returns:
        Set of domain-specific stopwords
    """
    if not chunks:
        return set()
    
    n_chunks = len(chunks)
    doc_freq: Counter = Counter()
    
    TerminalUI.dim(f"Detecting domain stopwords across {n_chunks} chunks...")
    
    for chunk in chunks:
        # Simple tokenization for counting (no stopword filtering yet)
        tokens = set(re.findall(r"[a-zA-Z]{3,}", chunk.clean.lower()))
        doc_freq.update(tokens)
    
    domain_stops = {
        term for term, count in doc_freq.items()
        if count / n_chunks > threshold
    }
    
    # Log top domain stopwords for inspection
    if domain_stops:
        top_stops = sorted(domain_stops, key=lambda t: doc_freq[t], reverse=True)[:10]
        TerminalUI.dim(f"Found {len(domain_stops)} domain stopwords: {', '.join(top_stops)}...")
    else:
        TerminalUI.dim("No domain stopwords detected")
    
    return domain_stops


# ============================================================================
# BM25 data structures for saving
# ============================================================================

@dataclass
class BM25Data:
    """Container for all BM25 indexing data to be saved."""
    tokens: List[List[str]]           # Tokenized chunks
    doc_freqs: dict                   # Term → document frequency
    avgdl: float                      # Average document length
    doc_lengths: List[int]            # Length of each chunk in tokens
    domain_stopwords: List[str]       # Detected domain stopwords
    threshold: float                  # Threshold used for detection
    chunk_count: int                  # Total chunks indexed
    vocab_size: int                   # Unique terms in vocabulary