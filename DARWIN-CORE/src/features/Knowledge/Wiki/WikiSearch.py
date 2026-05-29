# =============================================================================
# TUNABLE PARAMETERS — adjust these to control search behavior
# =============================================================================

# --- WikiSearchEngine ---
TEMPERATURE = 0.05                 # softmax temperature for vector score normalization
VECTOR_WEIGHT_INITIAL = 0.20      # weight of vector score in initial ranking
KEYWORD_WEIGHT_INITIAL = 0.35     # weight of keyword score in initial ranking
CONCEPT_WEIGHT_INITIAL = 0.20     # weight of concept score in initial ranking
CONTENT_QUALITY_WEIGHT_INITIAL = 0.25  # weight of content quality in initial ranking

VECTOR_WEIGHT_FINAL = 0.20        # weight of vector score in final reranking
KEYWORD_WEIGHT_FINAL = 0.45       # weight of keyword score in final reranking
CONTENT_QUALITY_WEIGHT_FINAL = 0.35  # weight of content quality in final reranking

SCORE_THRESHOLD_RATIO = 0.3       # keep candidates above this fraction of top score
SCORE_THRESHOLD_MIN = 0.15        # absolute minimum score threshold

SUBSTRING_MATCH_PREFIX_RATIO = 0.3  # min length ratio for prefix/suffix token matching
SUBSTRING_MATCH_PREFIX_SCORE = 0.8  # score multiplier for query_token prefix of keyword
SUBSTRING_MATCH_SUFFIX_SCORE = 0.6  # score multiplier for keyword prefix of query_token
MULTI_WORD_OVERLAP_SCORE = 0.7      # score multiplier for multi-word token overlap
MULTI_WORD_OVERLAP_RATIO = 0.3      # min overlap ratio for multi-word matching

CONCEPT_JACCARD_WEIGHT = 0.5        # weight of Jaccard similarity in concept scoring
CONCEPT_SUBSTRING_WEIGHT = 0.5      # weight of substring match in concept scoring

CONTENT_MATCH_COVERAGE_WEIGHT = 0.7  # weight of keyword match coverage in content quality
CONTENT_CONCEPT_MATCH_WEIGHT = 0.2  # weight of concept token matches in content quality
CONTENT_CHUNK_BONUS = 0.1           # bonus for having 3+ chunks
CONTENT_MIN_MATCH_SCORE = 0.5       # threshold for counting a keyword as "matched"

EXPAND_GRAPH_DEPTH = 1              # DFS depth for graph expansion
EXPAND_GRAPH_MAX_NODES = 10         # max nodes to collect during expansion

TOP_K_MULTIPLIER = 3                # fetch top_k * N candidates before threshold cutoff

# --- WikiOrchestrator ---
FILTER_STRUCTURAL_DEFAULT = True     # default for filter_structural parameter
MERGE_SOURCES_DEFAULT = True        # default for merge_sources parameter
FILL_GAPS_DEFAULT = True            # default for fill_gaps parameter
MAX_GAP_SIZE = 2                    # max chunk gap to fill
FALLBACK_IF_ALL_FILTERED = True     # if all results filtered, return raw top-k anyway

# --- Structural Detection ---
STRUCTURAL_MIN_CHUNK_LENGTH = 200       # chunks shorter than this are structural
STRUCTURAL_SIGNAL_RATIO = 0.30          # fraction of structural words to flag chunk
STRUCTURAL_SHORT_LINE_RATIO = 0.5       # fraction of short numbered lines to flag
STRUCTURAL_EARLY_CHUNK_RATIO = 0.05     # first N% of document
STRUCTURAL_EARLY_MIN_WORDS = 20         # min unique words to avoid early-chunk flag
STRUCTURAL_MAX_CONCEPT_LENGTH = 120     # concepts longer than this are structural
STRUCTURAL_MAX_DASH_SECTIONS = 4        # max " - ch" patterns in concept
STRUCTURAL_PAGE_PENALTY_LENGTH = 80     # concepts longer than this get a penalty
STRUCTURAL_PAGE_DASH_PENALTY = 3        # max dashes before penalty applies
STRUCTURAL_PENALTY_LONG_CONCEPT = -0.1  # penalty for long concept
STRUCTURAL_PENALTY_MANY_DASHES = -0.15  # penalty for many dashes
STRUCTURAL_PENALTY_SINGLE_CHUNK = -0.2  # penalty for single-chunk pages
STRUCTURAL_PENALTY_ABSOLUTE = -1.0      # penalty for definite structural pages
STRUCTURAL_CHUNK_COUNT_STD_MULTIPLIER = 2  # std dev multiplier for chunk count rule

# =============================================================================
# IMPORTS
# =============================================================================

from typing import List, Dict, Tuple, Set, Optional
import math
import re
import numpy as np
from collections import defaultdict
from src.features.Knowledge.Wiki.WikiUtils import DocumentWiki, WikiPage, WikiSearchResult


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cosine_similarity(a, b) -> float:
    """
    Safe cosine similarity optimized for retrieval systems.
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom < 1e-12:
        return 0.0

    return float(np.dot(a, b) / denom)


def preprocess_text(text: str, min_len: int = 1) -> List[str]:
    """
    Tokenize and clean text. No length filtering by default;
    callers may apply their own length thresholds.
    """
    text = re.sub(r"[^\w\s-]", " ", text.lower())
    tokens = [token.strip() for token in text.split() if token.strip()]
    return [t for t in tokens if len(t) >= min_len]


# =============================================================================
# WikiSearchEngine
# =============================================================================

class WikiSearchEngine:

    def __init__(self, documents):
        if isinstance(documents, DocumentWiki):
            documents = [documents]

        self.documents = documents

        self.common_words = {
            "the", "who", "are", "what", "is", "of", "in", "and",
            "a", "an", "to", "it", "for", "on", "with", "as", "at",
            "by", "from", "that", "this", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "how",
            "why", "where", "when", "which", "whom", "whose"
        }

        self.min_token_length = 3

        self.pages = self._flatten_pages(documents)
        self.concept_index = self._build_concept_index(self.pages)

        self._preprocess_keywords()
        self._compute_page_stats()

    def _flatten_pages(self, docs: List[DocumentWiki]) -> List[WikiPage]:
        return [p for doc in docs for p in doc.pages]

    def _build_concept_index(self, pages: List[WikiPage]) -> Dict[str, List[WikiPage]]:
        """
        Build a case‑insensitive index. Multiple pages can share the same concept name.
        """
        index = defaultdict(list)
        for p in pages:
            index[p.concept.lower()].append(p)
        return index

    def _preprocess_keywords(self):
        for page in self.pages:
            page._original_keywords = page.keywords.copy() if page.keywords else []
            page._processed_keywords = set()

            concept_clean = re.sub(r"[^\w\s-]", "", page.concept.lower()).strip()

            if concept_clean:
                page._processed_keywords.add(concept_clean)

                for word in concept_clean.split():
                    if len(word) >= self.min_token_length and word not in self.common_words:
                        page._processed_keywords.add(word)

            for keyword in page.keywords:
                keyword_lower = keyword.lower().strip()

                page._processed_keywords.add(keyword_lower)

                cleaned = re.sub(r"[^\w\s-]", "", keyword_lower)

                if cleaned and len(cleaned) >= self.min_token_length:
                    page._processed_keywords.add(cleaned)

                for word in cleaned.split():
                    if len(word) >= self.min_token_length and word not in self.common_words:
                        page._processed_keywords.add(word)

    def _compute_page_stats(self):
        chunk_counts = [len(p.chunk_indices) for p in self.pages]

        self.avg_chunk_count = np.mean(chunk_counts) if chunk_counts else 0
        self.std_chunk_count = np.std(chunk_counts) if chunk_counts else 1

        self.keyword_frequency = {}

        for page in self.pages:
            for keyword in page._processed_keywords:
                self.keyword_frequency[keyword] = self.keyword_frequency.get(keyword, 0) + 1

        total_pages = max(len(self.pages), 1)

        self.keyword_idf = {
            kw: math.log(total_pages / (freq + 1))
            for kw, freq in self.keyword_frequency.items()
        }

        self.default_idf = math.log(total_pages)

    def _normalize_vector_scores(
        self,
        raw_scores: np.ndarray,
        temperature: float = TEMPERATURE
    ) -> np.ndarray:

        shifted = raw_scores - np.max(raw_scores)

        exp = np.exp(shifted / max(temperature, 1e-6))

        denom = np.sum(exp)

        if denom < 1e-12:
            return np.zeros_like(raw_scores)

        return exp / denom

    def _is_structural_page(self, page: WikiPage) -> bool:
        concept = page.concept.lower()

        structural_patterns = [
            "cover title page",
            "copyright",
            "table of contents",
            "title page",
            "introduction - about",
            "acknowledgments",
            "preface",
            "foreword",
            "appendix",
            "index of",
            "glossary",
            "bibliography",
        ]

        if any(pattern in concept for pattern in structural_patterns):
            return True

        if len(page.concept) > STRUCTURAL_MAX_CONCEPT_LENGTH:
            return True

        if concept.count(" - ch") > STRUCTURAL_MAX_DASH_SECTIONS:
            return True

        if len(page.chunk_indices) > self.avg_chunk_count + STRUCTURAL_CHUNK_COUNT_STD_MULTIPLIER * self.std_chunk_count:
            return True

        return False

    def _concept_score(self, query: str, page: WikiPage) -> float:
        query_tokens = {
            t for t in preprocess_text(query)
            if t not in self.common_words and len(t) >= self.min_token_length
        }

        concept_tokens = {
            t for t in preprocess_text(page.concept)
            if t not in self.common_words and len(t) >= self.min_token_length
        }

        score = 0.0

        if query_tokens and concept_tokens:
            intersection = len(query_tokens & concept_tokens)
            union = len(query_tokens | concept_tokens)

            score += CONCEPT_JACCARD_WEIGHT * (intersection / max(union, 1))

        query_lower = query.lower()
        concept_lower = page.concept.lower()

        if query_lower in concept_lower or concept_lower in query_lower:
            score += CONCEPT_SUBSTRING_WEIGHT

        return min(score, 1.0)

    def _find_best_match(self, query_token: str, keywords: Set[str]) -> Tuple[float, Optional[str]]:
        """
        Returns (match_score, matched_keyword_or_None).
        Improved matching with word‑boundary awareness and no false substring matches.
        """
        if not query_token or len(query_token) < self.min_token_length:
            return 0.0, None

        # Exact match first
        if query_token in keywords:
            return 1.0, query_token

        best_score = 0.0
        best_kw = None

        # Pre‑compile a regex for word‑boundary matching
        pattern = re.compile(r'\b' + re.escape(query_token) + r'\b', re.IGNORECASE)

        for keyword in keywords:
            if len(keyword) < self.min_token_length:
                continue

            # Word‑boundary match anywhere in the keyword
            if pattern.search(keyword):
                score = 1.0
                if score > best_score:
                    best_score = score
                    best_kw = keyword
                continue

            # Check for prefix/suffix at word boundaries (e.g., "watcher" inside "watchers")
            kw_tokens = keyword.split()
            token_tokens = query_token.split()

            # If multi‑word, compare sets of tokens
            if len(kw_tokens) > 1 or len(token_tokens) > 1:
                kw_set = set(kw_tokens)
                token_set = set(token_tokens)
                overlap = kw_set & token_set
                if overlap:
                    ratio = len(overlap) / max(len(kw_set), len(token_set))
                    if ratio > MULTI_WORD_OVERLAP_RATIO:
                        score = MULTI_WORD_OVERLAP_SCORE * ratio
                        if score > best_score:
                            best_score = score
                            best_kw = keyword
                continue

            # Single‑token comparisons with word boundaries
            # query_token is a prefix of keyword token
            if keyword.startswith(query_token):
                if len(keyword) == len(query_token) or not keyword[len(query_token)].isalnum():
                    ratio = len(query_token) / len(keyword)
                    if ratio > SUBSTRING_MATCH_PREFIX_RATIO:
                        score = SUBSTRING_MATCH_PREFIX_SCORE * ratio
                        if score > best_score:
                            best_score = score
                            best_kw = keyword
            # keyword token is a prefix of query_token
            elif query_token.startswith(keyword):
                if len(query_token) == len(keyword) or not query_token[len(keyword)].isalnum():
                    ratio = len(keyword) / len(query_token)
                    if ratio > SUBSTRING_MATCH_PREFIX_RATIO:
                        score = SUBSTRING_MATCH_SUFFIX_SCORE * ratio
                        if score > best_score:
                            best_score = score
                            best_kw = keyword

        return best_score, best_kw

    def _keyword_score(self, query_tokens: List[str], page: WikiPage) -> float:

        specific_tokens = [
            t for t in query_tokens
            if t not in self.common_words and len(t) >= self.min_token_length
        ]

        if not specific_tokens:
            return 0.0

        total_score = 0.0
        max_possible_score = 0.0

        for token in specific_tokens:
            match_score, matched_kw = self._find_best_match(
                token,
                page._processed_keywords
            )

            if matched_kw and matched_kw in self.keyword_idf:
                idf_weight = self.keyword_idf[matched_kw]
            else:
                idf_weight = self.keyword_idf.get(token, self.default_idf)

            max_possible_score += idf_weight
            total_score += match_score * idf_weight

        if max_possible_score < 1e-12:
            return 0.0

        return min(total_score / max_possible_score, 1.0)

    def _content_quality_score(
        self,
        page: WikiPage,
        query_tokens: List[str]
    ) -> float:

        score = 0.0

        specific_tokens = [
            t for t in query_tokens
            if t not in self.common_words and len(t) >= self.min_token_length
        ]

        if specific_tokens:
            matches = 0

            for token in specific_tokens:
                match_score, _ = self._find_best_match(token, page._processed_keywords)
                if match_score >= CONTENT_MIN_MATCH_SCORE:
                    matches += 1

            score += CONTENT_MATCH_COVERAGE_WEIGHT * (matches / len(specific_tokens))

        concept_tokens = {
            t for t in preprocess_text(page.concept)
            if t not in self.common_words and len(t) >= self.min_token_length
        }

        if specific_tokens:
            concept_matches = sum(
                1 for t in specific_tokens if t in concept_tokens
            )

            score += CONTENT_CONCEPT_MATCH_WEIGHT * (concept_matches / len(specific_tokens))

        if len(page.chunk_indices) >= 3:
            score += CONTENT_CHUNK_BONUS

        return min(score, 1.0)
    def _structural_penalty(self, page: WikiPage) -> float:
        """
        DISABLED absolute kill-shot.
        The orchestrator now applies per-chunk multiplicative penalties.
        We keep mild additive nudges for long concepts / dash noise.
        """
        if self._is_structural_page(page):
            return 0.0

        penalty = 0.0

        if len(page.concept) > STRUCTURAL_PAGE_PENALTY_LENGTH:
            penalty += STRUCTURAL_PENALTY_LONG_CONCEPT

        if page.concept.lower().count(" - ") > STRUCTURAL_PAGE_DASH_PENALTY:
            penalty += STRUCTURAL_PENALTY_MANY_DASHES

        if len(page.chunk_indices) <= 1:
            penalty += STRUCTURAL_PENALTY_SINGLE_CHUNK

        return penalty

    def _expand_graph(
        self,
        page: WikiPage,
        depth: int = EXPAND_GRAPH_DEPTH,
        max_nodes: int = EXPAND_GRAPH_MAX_NODES
    ) -> List[WikiPage]:
        visited = set()
        results = []

        def dfs(p: WikiPage, d: int):
            if d > depth:
                return
            if p.concept in visited:
                return
            if len(results) >= max_nodes:
                return

            visited.add(p.concept)

            if not self._is_structural_page(p):
                if len(p.chunk_indices) >= 2:
                    results.append(p)

            for related_name in p.related:
                neighbors = self.concept_index.get(related_name.lower(), [])
                for neighbor in neighbors:
                    if neighbor.concept not in visited and not self._is_structural_page(neighbor):
                        dfs(neighbor, d + 1)

        dfs(page, 0)
        return results

    def search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[WikiSearchResult]:

        query_tokens = preprocess_text(query)

        raw_vector_scores = np.array([
            cosine_similarity(query_embedding, p.centroid)
            for p in self.pages
        ], dtype=np.float32)

        vector_scores_norm = self._normalize_vector_scores(
            raw_vector_scores,
            temperature=TEMPERATURE
        )

        scored_pages = []

        for idx, page in enumerate(self.pages):

            vector_score = vector_scores_norm[idx]
            keyword_score = self._keyword_score(query_tokens, page)
            concept_score = self._concept_score(query, page)
            content_quality = self._content_quality_score(page, query_tokens)
            structural_penalty = self._structural_penalty(page)

            score = (
                vector_score * VECTOR_WEIGHT_INITIAL +
                keyword_score * KEYWORD_WEIGHT_INITIAL +
                concept_score * CONCEPT_WEIGHT_INITIAL +
                content_quality * CONTENT_QUALITY_WEIGHT_INITIAL +
                structural_penalty
            )

            scored_pages.append((score, page))

        scored_pages.sort(key=lambda x: x[0], reverse=True)

        if scored_pages:
            score_threshold = max(scored_pages[0][0] * SCORE_THRESHOLD_RATIO, SCORE_THRESHOLD_MIN)

            top_pages = [
                p for s, p in scored_pages[:top_k * TOP_K_MULTIPLIER]
                if s >= score_threshold
            ][:top_k]
        else:
            top_pages = []

        expanded = []
        seen = {p.concept for p in top_pages}

        for p in top_pages:
            neighbors = self._expand_graph(p, depth=EXPAND_GRAPH_DEPTH)

            for n in neighbors:
                if n.concept not in seen:
                    expanded.append(n)
                    seen.add(n.concept)

        final_candidates = top_pages + expanded

        unique_candidates = []
        seen_concepts = set()

        for page in final_candidates:
            if page.concept not in seen_concepts:
                unique_candidates.append(page)
                seen_concepts.add(page.concept)

        expanded_concepts = {n.concept for n in expanded}

        final_scored = []

        for page in unique_candidates:

            vector_score = cosine_similarity(
                query_embedding,
                page.centroid
            )

            keyword_score = self._keyword_score(query_tokens, page)

            content_quality = self._content_quality_score(
                page,
                query_tokens
            )

            structural_penalty = self._structural_penalty(page)

            concept_score = self._concept_score(query, page)

            final_score = (
                vector_score * VECTOR_WEIGHT_FINAL +
                keyword_score * KEYWORD_WEIGHT_FINAL +
                content_quality * CONTENT_QUALITY_WEIGHT_FINAL +
                structural_penalty
            )

            final_scored.append(WikiSearchResult(
                page=page,
                score=final_score,
                vector_score=vector_score,
                keyword_score=keyword_score,
                concept_score=concept_score,
                content_quality_score=content_quality,
                structural_penalty=structural_penalty,
                is_expanded=page.concept in expanded_concepts,
            ))

        final_scored.sort(key=lambda x: x.score, reverse=True)

        return final_scored[:top_k]