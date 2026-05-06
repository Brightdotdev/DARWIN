"""
src/features/Retrival/WikiSearchUtils.py
-----------------------------------------
Tunable config, scoring helpers, gate logic, and data structures 
for wiki search. Keep all utility logic here so the main wiki search 
file focuses purely on orchestration.
"""

from __future__ import annotations

import math
import re
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np





# ═══════════════════════════════════════════════════════════════════════════
# Gate config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WikiGateConfig:
    """
    Configuration for wiki search scoring and satisfaction gate.
    
    Gate thresholds (all must pass for satisfied=True):
        1. min_top_score   — best wiki result's composite score >= this
        2. min_results     — at least this many wiki pages returned
        3. min_keyword_hit — at least one result has >= this many keyword hits
    
    Scoring weights:
        - centroid_weight: Importance of embedding similarity
        - keyword_weight:  Importance of keyword overlap
        - freq_weight:     Importance of concept frequency
        - freq_cap:        Max frequency for normalization
    """
    # Gate thresholds
    min_top_score:   float = 0.35
    min_results:     int   = 2
    min_keyword_hit: int   = 1
    
    # Scoring weights
    centroid_weight: float = 0.55
    keyword_weight:  float = 0.30
    freq_weight:     float = 0.15
    freq_cap:        int   = 50


# Sensible presets
GATE_STRICT  = WikiGateConfig(min_top_score=0.55, min_results=3, min_keyword_hit=2)
GATE_DEFAULT = WikiGateConfig()
GATE_RELAXED = WikiGateConfig(min_top_score=0.20, min_results=1, min_keyword_hit=0)


@dataclass
class WikiSearchArgs:
    
    query: str
    ws_port: int
    no_ws: bool
    files: Optional[List[str]] = None
    top_k : int = 5
    config: WikiGateConfig = field(default_factory=lambda: copy.deepcopy(GATE_DEFAULT))
    inspect: bool = False
    json_output : bool = False

# ═══════════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WikiSearchResult:
    """A single wiki-stage hit."""
    concept:        str
    filename:       str
    score:          float               # composite score [0, 1]
    centroid_sim:   float               # cosine sim to query embedding
    keyword_hits:   int                 # how many query tokens matched
    frequency:      int                 # how often concept appears
    related:        List[str]           = field(default_factory=list)
    keywords:       List[str]           = field(default_factory=list)
    summary:        Optional[str]       = None
    chunk_indices:  List[int]           = field(default_factory=list)

    # Satisfy _display_results / format helpers that read .source / .page / .text
    @property
    def source(self) -> str:
        return self.filename

    @property
    def page(self) -> int:
        return 0          # wiki results are concept-level, not page-level

    @property
    def text(self) -> str:
        if self.summary:
            return self.summary
        kw = ", ".join(self.keywords) if self.keywords else "—"
        return f"[concept] {self.concept} | keywords: {kw}"


@dataclass
class FileWikiResults:
    """Wiki results for a single file with its own satisfaction status."""
    filename:   str
    results:    List[WikiSearchResult]
    satisfied:  bool
    gate_reason: str = ""


@dataclass
class WikiSearchResponse:
    """Complete wiki search response grouped by file."""
    file_results: Dict[str, FileWikiResults]
    satisfied:    bool      # True only if ALL files satisfied
    gate_reason:  str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Tokenization & similarity utilities
# ═══════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> List[str]:
    """Lowercase word tokens; strips punctuation."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def cosine_similarity(a: List[float], b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if not a:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, b) / (norm_a * norm_b))


# ═══════════════════════════════════════════════════════════════════════════
# Scoring logic
# ═══════════════════════════════════════════════════════════════════════════

def score_wiki_page(
    page: Any,  # WikiPage
    query_tokens: List[str],
    query_vec: Optional[np.ndarray],
    config: WikiGateConfig = GATE_DEFAULT,
) -> Tuple[float, float, int]:
    """
    Score a single wiki page against query.
    
    Returns:
        (composite_score, centroid_sim, keyword_hits)
    """
    # ── centroid similarity ──
    centroid_sim = 0.0
    if query_vec is not None and page.centroid:
        centroid_sim = max(0.0, cosine_similarity(page.centroid, query_vec))

    # ── keyword overlap ──
    concept_tokens = set(tokenize(page.concept))
    kw_tokens: set = set()
    for kw in page.keywords:
        kw_tokens.update(tokenize(kw))
    all_tokens = concept_tokens | kw_tokens

    hits = sum(1 for t in query_tokens if t in all_tokens)
    keyword_score = hits / max(len(query_tokens), 1)

    # ── frequency score (log-normalised) ──
    freq_score = math.log1p(min(page.frequency, config.freq_cap)) / math.log1p(config.freq_cap)

    # ── composite ──
    composite = (
        config.centroid_weight * centroid_sim
        + config.keyword_weight * keyword_score
        + config.freq_weight * freq_score
    )
    return composite, centroid_sim, hits


# ═══════════════════════════════════════════════════════════════════════════
# Gate logic
# ═══════════════════════════════════════════════════════════════════════════

def is_gate_satisfied(
    results: List[WikiSearchResult],
    config: WikiGateConfig = GATE_DEFAULT,
) -> bool:
    """
    Return True if wiki results are good enough.
    All three conditions in config must pass simultaneously.
    """
    if not results:
        return False
    if len(results) < config.min_results:
        return False
    if results[0].score < config.min_top_score:
        return False
    if config.min_keyword_hit > 0:
        if not any(r.keyword_hits >= config.min_keyword_hit for r in results):
            return False
    return True


def get_gate_reason(
    results: List[WikiSearchResult],
    config: WikiGateConfig = GATE_DEFAULT,
) -> str:
    """
    Human-readable explanation of why the gate passed or failed.
    """
    if not results:
        return "FAIL — no wiki results"
    if len(results) < config.min_results:
        return f"FAIL — only {len(results)} result(s), need {config.min_results}"
    if results[0].score < config.min_top_score:
        return (
            f"FAIL — top score {results[0].score:.3f} "
            f"< threshold {config.min_top_score}"
        )
    if config.min_keyword_hit > 0:
        max_kw = max(r.keyword_hits for r in results)
        if max_kw < config.min_keyword_hit:
            return (
                f"FAIL — max keyword hits {max_kw} "
                f"< threshold {config.min_keyword_hit}"
            )
    top = results[0]
    return (
        f"PASS — top score {top.score:.3f}, "
        f"n={len(results)}, "
        f"kw_hits={top.keyword_hits}"
    )