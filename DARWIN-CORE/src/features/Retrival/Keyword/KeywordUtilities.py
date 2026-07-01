
from dataclasses import dataclass,field
from typing import List,Optional,Dict,Any
from src.features.Indexing.BM25.lib import BM25Data,compute_idf
        


# ============================================================================
# Constants
# ============================================================================

DEFAULT_TOP_K = 40
DEFAULT_K1 = 1.5   # higher == favour repeated stuff ,lower == favour scarcer stuff :: range from 1.2 to 2.0 is common in literature, 1.5 is a good default

DEFAULT_B = 0.75      # BM25 length normalization


# ============================================================================
# Args
# ============================================================================

@dataclass
class KeywordSearchArgs:
    query: str
    files: list[str] = field(default_factory=list)
    top_k: int = DEFAULT_TOP_K
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    inspect: bool = False
    json_output: bool = False


def calculate_bm25(
    query_terms: List[str],
    chunk_idx:   int,
    data:        Dict[str, Any],
    k1:          float,
    b:           float,
) -> tuple[float, List[str]]:
    """Compute BM25 score with safe defaults."""
    avgdl = data.get("avgdl", 0.0)
    if avgdl == 0:
        return 0.0, []

    doc_lengths = data.get("doc_lengths", [])
    if chunk_idx >= len(doc_lengths):
        return 0.0, []

    dl = doc_lengths[chunk_idx]
    length_norm = 1 - b + b * (dl / avgdl)
    chunk_tokens = data.get("tokens", [])[chunk_idx] if chunk_idx < len(data.get("tokens", [])) else []

    tf_map = {}
    for token in chunk_tokens:
        tf_map[token] = tf_map.get(token, 0) + 1

    score = 0.0
    matched_terms = []
    idfs = data.get("precomputed_idfs") or {}
    doc_freqs = data.get("doc_freqs", {})
    chunk_count = data.get("chunk_count", 0)
    idf_smoothing = data.get("idf_smoothing", 0.0)

    for term in set(query_terms):
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue

        idf = idfs.get(term)
        if idf is None:
            df = doc_freqs.get(term, 0)
            idf = compute_idf(df, chunk_count, idf_smoothing)

        tf_norm = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        score += idf * tf_norm
        matched_terms.append(term)

    return score, matched_terms


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


