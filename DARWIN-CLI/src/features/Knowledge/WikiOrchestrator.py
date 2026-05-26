"""
src/core/Knowledge/WikiOrchestrator.py
--------------------------------------
Broad / conceptual search over document wiki.

Returns concept pointers (where concepts live, what they're related to).
NO passage text assembly — that's RetrievalOrchestrator's domain.
"""

from typing import List, Dict, Optional,Any
from dataclasses import dataclass, field
from collections import Counter
import re

from src.core.Storage.DarwinStore import DarwinStore
from src.features.Indexing.Embedding.Embedder import embed_query
from src.features.Knowledge.Wiki.WikiUtils import DocumentWiki, WikiPage, WikiSearchResult
from src.features.Knowledge.Wiki.WikiSearch import WikiSearchEngine
from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Query.StructuralClassifier import classify_structural_type, get_penalty_factor
from src.features.Retrival.lib import SearchAudit


# =============================================================================
# Tunables
# =============================================================================

MERGE_SOURCES_DEFAULT = True
FILTER_STRUCTURAL_DEFAULT = True
FALLBACK_IF_ALL_FILTERED = True
MAX_GAP_SIZE = 2
MAX_PASSAGE_CHUNKS = 12  # chunk index budget, NOT text chars


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class WikiConcept:
    """
    A wiki concept with resolved pointers. NO passage text.
    """
    concept: str           # Display name (may include merged suffix)
    raw_concept: str       # Primary concept name without suffix
    keywords: List[str]
    page_start: int
    page_end: int
    chunk_indices: List[int]
    score: float
    source_count: int
    related_topics: List[str] = field(default_factory=list)
    source_file: str = ""  # which file this concept came from
    audit: Optional[SearchAudit] = None

    def get_page_range(self) -> str:
        if self.page_start == self.page_end:
            return f"Page {self.page_start}"
        return f"Pages {self.page_start}-{self.page_end}"


@dataclass
class WikiContext:
    """Complete wiki search context."""
    query: str
    concepts: List[WikiConcept]
    related_topics: List[str]
    total_concepts: int
    penalized_count: int


# =============================================================================
# Gap Filling (for chunk indices only)
# =============================================================================

def _fill_gaps(indices: List[int], max_gap: int = MAX_GAP_SIZE) -> List[int]:
    if len(indices) < 2:
        return indices
    sorted_idx = sorted(indices)
    filled = [sorted_idx[0]]
    for i in range(1, len(sorted_idx)):
        prev, curr = filled[-1], sorted_idx[i]
        gap = curr - prev - 1
        if 0 < gap <= max_gap:
            for fill_idx in range(prev + 1, curr):
                filled.append(fill_idx)
        filled.append(curr)
    return filled


# =============================================================================
# Merge Gating
# =============================================================================

def _should_merge_concepts(
    c1_indices: List[int],
    c2_indices: List[int],
    chunk_map: Dict[int, Dict[str, Any]],
) -> bool:
    """Pure proximity check."""
    for i1 in c1_indices:
        for i2 in c2_indices:
            if abs(i1 - i2) <= MAX_GAP_SIZE:
                return True
            chunk1 = chunk_map.get(i1)
            chunk2 = chunk_map.get(i2)
            if chunk1 and chunk2:
                if chunk1.get("page") == chunk2.get("page") and chunk1.get("page") > 0:
                    if abs(i1 - i2) <= 8:
                        return True
    return False


def _has_semantic_overlap(r1: WikiSearchResult, r2: WikiSearchResult) -> bool:
    """Require shared keywords OR shared substantive concept words."""
    shared_keywords = set(r1.page.keywords) & set(r2.page.keywords)
    if len(shared_keywords) >= 2:
        return True

    words1 = set(r1.page.concept.lower().split())
    words2 = set(r2.page.concept.lower().split())
    stops = {"the", "and", "of", "in", "to", "a", "is", "for", "on", "with", "as", "book", "ch"}
    words1 -= stops
    words2 -= stops
    if len(words1 & words2) > 0:
        return True

    return False


# =============================================================================
# WikiOrchestrator
# =============================================================================

class WikiOrchestrator:
    """
    Conceptual / broad search over document wiki.

    Returns concept pointers (where concepts live, what they're related to).
    Does NOT assemble passage text — that's RetrievalOrchestrator's domain.
    """

    def __init__(self, store: DarwinStore):
        self.store = store
        self._engines: Dict[str, WikiSearchEngine] = {}

    def _get_wiki_search(self, wiki: DocumentWiki, filename: str) -> WikiSearchEngine:
        if filename not in self._engines:
            self._engines[filename] = WikiSearchEngine(wiki)
        return self._engines[filename]

    def search(
        self,
        query: str,
        filename: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        merge_sources: bool = MERGE_SOURCES_DEFAULT,
        filter_structural: bool = FILTER_STRUCTURAL_DEFAULT,
    ) -> WikiContext:
        darwin.log(f"[WikiOrchestrator] Searching: '{query}' in {filename}", LogType.INFO)

        if not query.strip():
            return WikiContext(
                query=query, concepts=[], related_topics=[],
                total_concepts=0, penalized_count=0,
            )

        pages_dicts = self.store.load_wiki(filename)
        if not pages_dicts:
            darwin.log(f"No wiki found for {filename}", LogType.WARNING)
            return WikiContext(
                query=query, concepts=[], related_topics=[],
                total_concepts=0, penalized_count=0,
            )
        
        wiki = DocumentWiki(
            filename="doc.pdf",
            pages=[WikiPage.from_dict(p) for p in pages_dicts],
        )

        if not wiki:
            return WikiContext(
                query=query, concepts=[], related_topics=[],
                total_concepts=0, penalized_count=0,
            )

        engine = self._get_wiki_search(wiki, filename)
        if query_embedding is None:
            query_embedding = embed_query(query)

        raw_results: List[WikiSearchResult] = engine.search(
            query=query, query_embedding=query_embedding, top_k=top_k * 2  # larger pool for penalties
        )

        if not raw_results:
            return WikiContext(
                query=query, concepts=[], related_topics=[],
                total_concepts=0, penalized_count=0,
            )

        all_chunks = self.store.query_records(filename, columns=["index", "clean","text", "page", "page_end"])

        chunk_map = {c.get("index", ""): c for c in all_chunks}

        # Precompute duplicate chunk hashes for structural detection
        chunk_hashes = Counter()
        for c in all_chunks:
            chunk_hashes[hash(c.get("clean", "").lower()[:200])] += 1

        # Apply structural penalties instead of hard filtering
        penalized_results: List[WikiSearchResult] = []
        penalized_count = 0

        for r in raw_results:
            chunk_types = []
            for idx in r.page.chunk_indices:
                chunk = chunk_map.get(idx)
                if chunk:
                    stype = classify_structural_type(chunk, all_chunks, chunk_hashes=chunk_hashes, aggressive=True)
                    if stype:
                        chunk_types.append(stype)

            if chunk_types and filter_structural:
                factors = [get_penalty_factor(t) for t in chunk_types]
                factor = min(factors)
                stype = chunk_types[factors.index(factor)]

                raw_score = r.score
                r.score = raw_score * factor
                penalized_count += 1

                audit = SearchAudit(
                    raw_score=raw_score,
                    final_score=r.score,
                    structural_type=stype,
                    penalty_factor=factor,
                    was_penalized=True,
                    penalty_reason=f"concept has {len(chunk_types)} structural chunk(s): {set(chunk_types)}",
                    concept_source=r.page.concept,
                )
                if hasattr(r, 'audit'):
                    r.audit = audit
                else:
                    object.__setattr__(r, 'audit', audit)

                darwin.log(
                    f"[WikiPenalty] Concept '{r.page.concept}' {stype} factor={factor} "
                    f"{raw_score:.3f}→{r.score:.3f}",
                    LogType.INFO,
                )

            penalized_results.append(r)

        # Re-sort after penalties
        penalized_results.sort(key=lambda x: x.score, reverse=True)
        penalized_results = penalized_results[:top_k]

        # Note if all top concepts are heavily penalized
        has_unpenalized = any(
            getattr(r, 'audit', None) and r.audit.penalty_factor == 1.0
            for r in penalized_results
        )
        if not has_unpenalized and penalized_results:
            darwin.log(
                "[WikiOrchestrator] All top concepts are structural; "
                "returning penalized results.",
                LogType.WARNING,
            )

        # Group concepts by proximity + semantic overlap
        if merge_sources and len(penalized_results) > 1:
            groups = self._group_concepts(penalized_results, chunk_map)
        else:
            groups = [[r] for r in penalized_results]

        # Resolve each group into a WikiConcept (pointers only)
        concepts = []
        all_related = []

        for group in groups:
            all_indices = []
            all_keywords = []
            max_score = 0.0

            for r in group:
                all_indices.extend(r.page.chunk_indices)
                all_keywords.extend(r.page.keywords)
                max_score = max(max_score, r.score)
                if r.page.related:
                    all_related.extend(r.page.related)

            # Deduplicate and budget chunk indices
            all_indices = sorted(set(all_indices))
            all_indices = _fill_gaps(all_indices)[:MAX_PASSAGE_CHUNKS]
            all_keywords = list(dict.fromkeys(all_keywords))

            # Compute page range from chunks
            page_start, page_end = 0, 0
            if all_indices:
                contributors = [chunk_map[idx] for idx in all_indices if idx in chunk_map]
                if contributors:
                    all_pages = []
                    for c in contributors:
                        if c.get("page") is not None:
                            all_pages.append(c.get("page"))
                        if c.get("page_end") is not None:
                            all_pages.append(c.get("page_end"))
                    page_start = min(all_pages) if all_pages else contributors[0].get("page") or 0
                    page_end = max(all_pages) if all_pages else contributors[-1].get("page_end") or 0

            # Pick highest-scored concept as primary name
            primary = max(group, key=lambda r: r.score)
            concept_name = primary.page.concept
            raw_name = primary.page.concept

            if len(group) > 1:
                other_names = [r.page.concept for r in group if r.page.concept != concept_name]
                if other_names:
                    display_names = other_names[:2]
                    if len(other_names) > 2:
                        display_names.append(f"+{len(other_names) - 2} more")
                    concept_name += f" (with: {', '.join(display_names)})"

            # FIX: Flatten related topics properly (r.page.related is a list)
            _related_flat = []
            for r in group:
                if r.page.related:
                    _related_flat.extend(r.page.related)
            group_related = list(dict.fromkeys(_related_flat))[:5]

            primary_audit = getattr(primary, 'audit', None)

            concept = WikiConcept(
                concept=concept_name,
                raw_concept=raw_name,
                keywords=all_keywords[:10],
                page_start=page_start,
                page_end=page_end,
                chunk_indices=all_indices,
                score=max_score,
                source_count=len(group),
                related_topics=group_related,
                source_file=filename,
                audit=primary_audit,
            )
            concepts.append(concept)

        related_topics = list(dict.fromkeys(all_related))[:10]

        return WikiContext(
            query=query,
            concepts=concepts,
            related_topics=related_topics,
            total_concepts=len(concepts),
            penalized_count=penalized_count,
        )

    def _group_concepts(
        self,
        results: List[WikiSearchResult],
        chunk_map: Dict[int, Dict[str, Any]],
    ):
        if not results:
            return []
        sorted_results = sorted(
            results,
            key=lambda r: min(r.page.chunk_indices) if r.page.chunk_indices else 0,
        )
        groups = []
        current_group = [sorted_results[0]]

        for r in sorted_results[1:]:
            should_merge = False
            for existing in current_group:
                proximity = _should_merge_concepts(
                    r.page.chunk_indices, existing.page.chunk_indices, chunk_map
                )
                semantic = _has_semantic_overlap(r, existing)
                if proximity and semantic:
                    should_merge = True
                    break

            if should_merge:
                current_group.append(r)
            else:
                groups.append(current_group)
                current_group = [r]

        if current_group:
            groups.append(current_group)

        return groups

    def get_concept_passage(
        self,
        concept: WikiConcept,
        filename: str,
        fill_gaps: bool = True,
        max_chars: int = 4000,
    ) -> str:
        """
        Lazy passage resolution — hydrate a WikiConcept with text on demand.
        Use sparingly; prefer letting RetrievalOrchestrator handle evidence.
        """
        all_chunks = self.store.query_records(filename, columns=["index", "clean", "text", "page"])
        chunk_map = {c.get("index", ""): c for c in all_chunks}

        indices = concept.chunk_indices
        if fill_gaps and len(indices) > 1:
            indices = _fill_gaps(sorted(indices))

        contributors = [chunk_map[idx] for idx in indices if idx in chunk_map]
        if not contributors:
            return ""

        # Simple stitch (no sentence dedup for lazy load)
        segments = []
        for i, chunk in enumerate(contributors):
            if i > 0 and chunk.get("page") != contributors[i - 1].get("page") and chunk.get("page") > 0:
                segments.append(f"\n[Page {chunk.get('page')}]\n")
            segments.append(chunk.get("text", chunk.get('clean', '')))

        text = " ".join(segments)
        if len(text) > max_chars:
            text = text[:max_chars]
            # Truncate at sentence boundary
            for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                idx = text.rfind(punct)
                if idx != -1 and idx > len(text) * 0.5:
                    text = text[:idx + 1]
                    break
        return text