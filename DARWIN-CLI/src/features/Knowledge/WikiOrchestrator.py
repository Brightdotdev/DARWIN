# =============================================================================
# TUNABLE PARAMETERS — adjust these to control orchestrator behavior
# =============================================================================

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

# --- Passage Assembly ---
MAX_PASSAGE_CHARS = 4000              # hard character budget per merged passage
MAX_PASSAGE_CHUNKS = 12                # hard chunk budget per merged passage

# --- Merge Gating ---
MERGE_KEYWORD_OVERLAP_REQUIRED = True   # require shared keywords before merging
MERGE_CONCEPT_WORD_OVERLAP_REQUIRED = True  # require shared concept words before merging
MAX_MERGED_CONCEPT_NAMES = 2           # max additional concept names in merged title

# =============================================================================
# IMPORTS
# =============================================================================

from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import Counter
import re

from src.features.Storage.DarwinStore import DarwinStore
from src.features.Ingestion.Chunking.lib import Chunk
from src.features.Ingestion.Chunking.ContextResolver import get_full_context
from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Knowledge.Wiki.WikiUtils import DocumentWiki, WikiSearchResult
from src.features.Indexing.Embedding.Embedder import embed_query
from src.features.Knowledge.Wiki.WikiSearch import WikiSearchEngine



# =============================================================================
# STRUCTURAL SIGNALS (shared, not tunable — add/remove patterns here)
# =============================================================================

STRUCTURAL_SIGNALS = {
    "cover", "title", "copyright", "table of contents", "contents",
    "acknowledgments", "preface", "foreword", "introduction", "about",
    "appendix", "index", "glossary", "bibliography", "references",
    "page", "chapter", "book", "part", "section", "volume",
    "ebook", "edition", "published", "publisher", "isbn",
    "all rights reserved", "printed in", "first edition",
    "dedication", "prologue", "epilogue", "notes", "footnote",
    "abstract", "summary", "overview", "list of", "figure", "table",
    "extras", "chart", "works cited",
}

SECTION_PREFIX_RE = re.compile(
    r'^(Book|Chapter|Ch\.|Part|Section|Vol\.|Volume|Appendix|Excerpt|'
    r'Reference|Note|Figure|Table|Chart|Map|Illustration)s?\s*[\dIVX]+',
    re.IGNORECASE
)


# =============================================================================
# STRUCTURAL DETECTION
# =============================================================================

def _is_structural_chunk(chunk: Chunk, all_chunks: List[Chunk],
                         chunk_hashes: Optional[Counter] = None) -> bool:
    """Detect structural chunks by content, using precomputed hashes for performance."""
    text = chunk.clean.lower()
    words = set(re.findall(r'\b[a-z]+\b', text))
    lines = [l.strip() for l in chunk.clean.split('\n') if l.strip()]

    if len(chunk.clean) < STRUCTURAL_MIN_CHUNK_LENGTH:
        return True

    structural_found = words & STRUCTURAL_SIGNALS
    if len(words) > 0 and len(structural_found) / len(words) > STRUCTURAL_SIGNAL_RATIO:
        return True

    if len(lines) > 3:
        short_numbered = sum(1 for l in lines if len(l) < 60 and re.search(r'\d', l))
        if short_numbered / len(lines) > STRUCTURAL_SHORT_LINE_RATIO:
            return True

    # Deduplication check using precomputed hashes
    if chunk_hashes is not None:
        text_hash = hash(text[:200])
        if chunk_hashes.get(text_hash, 0) > 1:
            return True

    if chunk.index <= len(all_chunks) * STRUCTURAL_EARLY_CHUNK_RATIO and len(words) < STRUCTURAL_EARLY_MIN_WORDS:
        return True

    section_headers = sum(1 for l in lines if SECTION_PREFIX_RE.match(l))
    if section_headers >= 3:
        return True

    return False


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class WikiConcept:
    """A wiki concept with resolved passage text."""
    concept: str
    summary: str
    keywords: List[str]
    page_start: int
    page_end: int
    chunk_indices: List[int]
    passage_text: str
    score: float
    source_count: int


@dataclass
class WikiContext:
    """Complete wiki search context."""
    query: str
    concepts: List[WikiConcept]
    related_topics: List[str]
    total_concepts: int
    filtered_count: int


# =============================================================================
# SOURCE MERGER WITH GAP FILLING
# =============================================================================

def _should_merge_concepts(c1_indices: List[int], c2_indices: List[int],
                           chunk_map: Dict[int, Chunk]) -> bool:
    """Pure proximity check — no semantic gating here."""
    for i1 in c1_indices:
        for i2 in c2_indices:
            if abs(i1 - i2) <= MAX_GAP_SIZE:
                return True
            chunk1 = chunk_map.get(i1)
            chunk2 = chunk_map.get(i2)
            if chunk1 and chunk2:
                if chunk1.page == chunk2.page and chunk1.page > 0:
                    if abs(i1 - i2) <= 8:
                        return True
    return False


def _has_semantic_overlap(r1: WikiSearchResult, r2: WikiSearchResult) -> bool:
    """Check if two concepts are semantically related enough to merge."""
    # Keyword overlap
    if MERGE_KEYWORD_OVERLAP_REQUIRED:
        shared_keywords = set(r1.page.keywords) & set(r2.page.keywords)
        if len(shared_keywords) > 0:
            return True
    
    # Concept word overlap
    if MERGE_CONCEPT_WORD_OVERLAP_REQUIRED:
        words1 = set(r1.page.concept.lower().split())
        words2 = set(r2.page.concept.lower().split())
        # Filter out common words
        stops = {"the", "and", "of", "in", "to", "a", "is", "for", "on", "with", "as", "book", "ch"}
        words1 -= stops
        words2 -= stops
        shared_words = words1 & words2
        if len(shared_words) > 0:
            return True
    
    return False


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


def _stitch_chunks(chunks: List[Chunk]) -> str:
    """
    Stitch chunk texts with sentence-level overlap deduplication.
    Injects page markers when crossing page boundaries.
    """
    if not chunks:
        return ""

    from src.features.Ingestion.Chunking.ChunkerUtils import _split_sentences

    # Build segments with page markers
    segments: List[str] = []
    for i, chunk in enumerate(chunks):
        if i > 0 and chunk.page != chunks[i - 1].page and chunk.page > 0:
            segments.append(f"\n[Page {chunk.page}]\n")
        segments.append(chunk.text)

    combined = " ".join(segments)
    sentences = _split_sentences(combined)

    # Trim duplicate sentences from overlap seams
    deduped = []
    for s in sentences:
        norm = " ".join(s.split())
        if deduped and norm == " ".join(deduped[-1].split()):
            continue
        deduped.append(s)

    return " ".join(deduped)


def _smart_truncate(text: str, max_len: int) -> str:
    """Truncate at the last sentence boundary before max_len."""
    if len(text) <= max_len:
        return text
    text = text[:max_len]
    for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n', '." ', '!" ', '?" ']:
        idx = text.rfind(punct)
        if idx != -1 and idx > len(text) * 0.5:
            return text[:idx + 1]
    return text


def _resolve_concept_passage(
    chunk_indices: List[int],
    chunk_map: Dict[int, Chunk],
    store: DarwinStore,
    filename: str,
    fill_gaps: bool = FILL_GAPS_DEFAULT,
) -> Tuple[str, int, int]:
    """
    Resolve a group of chunks into one continuous passage.
    """
    filled_indices = _fill_gaps(sorted(chunk_indices)) if fill_gaps else sorted(chunk_indices)

    # Apply chunk budget
    if len(filled_indices) > MAX_PASSAGE_CHUNKS:
        filled_indices = filled_indices[:MAX_PASSAGE_CHUNKS]

    if len(filled_indices) == 1:
        anchor_chunk = chunk_map.get(filled_indices[0])
        if not anchor_chunk:
            return "", 0, 0
        context = get_full_context(anchor_chunk, store, filename, deduplicate=True)
        text = context.text
        page_start = context.page
        page_end = context.page_end
    else:
        contributors = [chunk_map[idx] for idx in filled_indices if idx in chunk_map]
        if not contributors:
            return "", 0, 0
        text = _stitch_chunks(contributors)
        all_pages = []
        for c in contributors:
            if c.page is not None:
                all_pages.append(c.page)
            if c.page_end is not None:
                all_pages.append(c.page_end)
        page_start = min(all_pages) if all_pages else contributors[0].page or 0
        page_end = max(all_pages) if all_pages else contributors[-1].page_end or 0

    # Apply character budget
    text = _smart_truncate(text, MAX_PASSAGE_CHARS)

    return text, page_start, page_end


# =============================================================================
# WikiOrchestrator
# =============================================================================

class WikiOrchestrator:
    """
    Wiki search with structural filtering, source merging, and gap filling.
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
        top_k: int = 5,
        merge_sources: bool = MERGE_SOURCES_DEFAULT,
        filter_structural: bool = FILTER_STRUCTURAL_DEFAULT,
        fill_gaps: bool = FILL_GAPS_DEFAULT,
    ) -> WikiContext:
        darwin.log(f"[WikiOrchestrator] Searching: '{query}' with wiki", LogType.INFO)

        if not query.strip():
            return WikiContext(query=query, concepts=[], related_topics=[], total_concepts=0, filtered_count=0)

        wiki = self.store.load_wiki(filename)
        if not wiki:
            return WikiContext(query=query, concepts=[], related_topics=[], total_concepts=0, filtered_count=0)

        engine = self._get_wiki_search(wiki, filename)
        query_embed = embed_query(query)
        raw_results: List[WikiSearchResult] = engine.search(query=query, query_embedding=query_embed, top_k=top_k)

        if not raw_results:
            return WikiContext(query=query, concepts=[], related_topics=[], total_concepts=0, filtered_count=0)

        all_chunks = self.store.load_chunks(filename)
        chunk_map = {c.index: c for c in all_chunks}

        # Precompute duplicate chunk hashes for structural detection
        chunk_hashes = Counter()
        for c in all_chunks:
            chunk_hashes[hash(c.clean.lower()[:200])] += 1

        # Filter structural concepts
        filtered_results: List[WikiSearchResult] = []
        filtered_count = 0

        for r in raw_results:
            is_struct = False
            for idx in r.page.chunk_indices:
                chunk = chunk_map.get(idx)
                if chunk and _is_structural_chunk(chunk, all_chunks, chunk_hashes):
                    is_struct = True
                    break

            if is_struct and filter_structural:
                filtered_count += 1
                darwin.log(f"[WikiFilter] Concept '{r.page.concept}' has structural chunks", LogType.INFO)
            else:
                filtered_results.append(r)

        # FALLBACK: if all results were filtered, return raw unfiltered results
        if not filtered_results and FALLBACK_IF_ALL_FILTERED:
            darwin.log("[WikiOrchestrator] All results filtered — falling back to raw results", LogType.WARNING)
            filtered_results = raw_results
            filtered_count = 0

        # Group concepts by proximity
        if merge_sources and len(filtered_results) > 1:
            groups = self._group_concepts(filtered_results, chunk_map)
        else:
            groups = [[r] for r in filtered_results]

        # Resolve each group into a WikiConcept
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

            all_indices = sorted(set(all_indices))[:MAX_PASSAGE_CHUNKS]
            all_keywords = list(dict.fromkeys(all_keywords))

            passage_text, page_start, page_end = _resolve_concept_passage(
                all_indices, chunk_map, self.store, filename, fill_gaps=fill_gaps
            )

            concept_name = group[0].page.concept
            if len(group) > 1:
                other_names = [r.page.concept for r in group[1:] if r.page.concept != concept_name]
                if other_names:
                    # Cap at MAX_MERGED_CONCEPT_NAMES additional names
                    display_names = other_names[:MAX_MERGED_CONCEPT_NAMES]
                    if len(other_names) > MAX_MERGED_CONCEPT_NAMES:
                        display_names.append(f"+{len(other_names) - MAX_MERGED_CONCEPT_NAMES} more")
                    concept_name += f" (with: {', '.join(display_names)})"

            concept = WikiConcept(
                concept=concept_name,
                summary=group[0].page.summary or "",
                keywords=all_keywords[:10],
                page_start=page_start,
                page_end=page_end,
                chunk_indices=all_indices,
                passage_text=passage_text,
                score=max_score,
                source_count=len(group),
            )
            concepts.append(concept)

        related_topics = list(dict.fromkeys(all_related))[:10]

        return WikiContext(
            query=query,
            concepts=concepts,
            related_topics=related_topics,
            total_concepts=len(concepts),
            filtered_count=filtered_count,
        )

    def _group_concepts(self, results: List[WikiSearchResult], chunk_map: Dict[int, Chunk]):
        if not results:
            return []
        sorted_results = sorted(results, key=lambda r: min(r.page.chunk_indices) if r.page.chunk_indices else 0)
        groups = []
        current_group = [sorted_results[0]]
        
        for r in sorted_results[1:]:
            should_merge = False
            
            for existing in current_group:
                # Proximity check
                proximity = _should_merge_concepts(r.page.chunk_indices, existing.page.chunk_indices, chunk_map)
                
                # Semantic gate: REQUIRED for ALL merges now
                semantic = _has_semantic_overlap(r, existing)
                
                # Both must pass
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

    def build_llm_prompt(self, context: WikiContext, query: str) -> str:
        """Build LLM prompt from wiki concepts with passages."""
        if not context.concepts:
            return f"Question: {query}\n\nNo relevant concepts found."

        blocks = []
        for i, c in enumerate(context.concepts):
            gap_info = ""
            if c.source_count > 1:
                gap_info = f" [merged from {c.source_count} concepts]"

            blocks.append(
                f"[Concept {i+1}: {c.concept} | Page {c.page_start}-{c.page_end} | "
                f"Score: {c.score:.3f}{gap_info}]\n"
                f"Summary: {c.summary}\n"
                f"Passage: {c.passage_text}\n"
                f"Keywords: {', '.join(c.keywords[:5])}\n"
            )

        concept_text = "\n---\n".join(blocks)

        related = ""
        if context.related_topics:
            related = f"\nRelated topics: {', '.join(context.related_topics[:5])}\n"

        return f"""Answer based on the following document concepts and passages.

Question: {query}

Concepts:
{concept_text}
{related}
Answer:"""


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def wiki_search_for_llm(query: str, filename: str, store: DarwinStore,
                        top_k: int = 5, merge_sources: bool = MERGE_SOURCES_DEFAULT,
                        fill_gaps: bool = FILL_GAPS_DEFAULT) -> Tuple[str, WikiContext]:
    """One‑liner: search wiki, merge adjacent, resolve passages, build prompt."""
    orch = WikiOrchestrator(store)
    context = orch.search(query, filename, top_k=top_k,
                          merge_sources=merge_sources, fill_gaps=fill_gaps)
    prompt = orch.build_llm_prompt(context, query)
    return prompt, context