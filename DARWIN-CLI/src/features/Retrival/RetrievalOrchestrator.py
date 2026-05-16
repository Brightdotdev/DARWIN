"""
src/core/Knowledge/RetrievalOrchestrator.py
-------------------------------------------
Retrieval pipeline with:
1. Structural chunk filtering
2. Source merging (adjacent/same-page chunks → one Source)
3. Gap filling (1-2 chunk gaps get included for full narrative)
4. Context resolution via full-span stitching with page markers
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter
import re

from src.features.Retrival.main import handle_retrieval
from src.features.Retrival.RetrivalUtils import RetrievalArgs, RetrievalMethod
from src.features.Retrival.RetrievalResult import RetrievalResult

from src.features.Ingestion.Chunking.lib import Chunk
from src.features.Ingestion.Chunking.ChunkerUtils import _split_sentences
from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType


# ═══════════════════════════════════════════════════════════════════════════
# Tunables
# ═══════════════════════════════════════════════════════════════════════════

MAX_GAP_SIZE = 2                    # Fill gaps up to 2 chunks
MAX_PASSAGE_CHUNKS = 12             # Hard chunk budget per merged passage
MAX_PASSAGE_CHARS = 6000            # Hard character budget per passage
FALLBACK_MIN_SCORE = 0.3            # Minimum score for fallback results to be accepted


# ═══════════════════════════════════════════════════════════════════════════
# Structural detection
# ═══════════════════════════════════════════════════════════════════════════

STRUCTURAL_SIGNALS = {
    "cover", "title", "copyright", "table of contents", "contents",
    "acknowledgments", "preface", "foreword", "introduction", "about",
    "appendix", "index", "glossary", "bibliography", "references",
    "page", "chapter", "book", "part", "section", "volume",
    "ebook", "edition", "published", "publisher", "isbn",
    "all rights reserved", "printed in", "first edition",
    "dedication", "prologue", "epilogue", "notes", "footnote",
    "abstract", "summary", "overview", "list of", "figure", "table",
    "extras", "chart", "works cited", "front matter", "back matter",
}

SECTION_PREFIX_RE = re.compile(
    r'^(Book|Chapter|Ch\.|Part|Section|Vol\.|Volume|Appendix|Excerpt|'
    r'Reference|Note|Figure|Table|Chart|Map|Illustration)s?\s*[\dIVX]+',
    re.IGNORECASE
)

TOC_ENTRY_RE = re.compile(
    r'([\.•\-–—]\s*\d+\s*$)|(^\d+[\.\)]\s+\w+)',
    re.IGNORECASE
)


def _is_structural_chunk(chunk: Chunk, all_chunks: List[Chunk]) -> bool:
    """Detect structural chunks by content, not index."""
    text = chunk.clean
    text_lower = text.lower()
    words = set(re.findall(r'\b[a-z]+\b', text_lower))
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if len(text) < 200:
        return True

    structural_found = words & STRUCTURAL_SIGNALS
    if len(words) > 0 and len(structural_found) / len(words) > 0.30:
        return True

    if len(lines) > 3:
        short_numbered = sum(1 for l in lines if len(l) < 60 and re.search(r'\d', l))
        if short_numbered / len(lines) > 0.5:
            return True

    text_hash = hash(text_lower[:200])
    repeat_count = sum(1 for c in all_chunks if hash(c.clean.lower()[:200]) == text_hash)
    if repeat_count > 2:
        return True

    if chunk.index <= len(all_chunks) * 0.05 and len(words) < 20:
        return True

    section_headers = sum(1 for l in lines if SECTION_PREFIX_RE.match(l))
    if section_headers >= 3:
        return True

    if len(lines) > 3:
        toc_entries = sum(1 for l in lines if TOC_ENTRY_RE.search(l))
        if toc_entries / len(lines) > 0.40:
            return True

    if chunk.index <= len(all_chunks) * 0.10 and len(lines) > 0:
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        if (len(words) > 0 and 
            len(structural_found) / len(words) > 0.15 and 
            avg_line_len < 80):
            return True

    if len(lines) >= 3:
        prefixes = []
        for l in lines:
            m = re.match(r'^([A-Za-z]+)[\s\-–—]', l)
            if m:
                prefixes.append(m.group(1).lower())
        if prefixes:
            most_common = Counter(prefixes).most_common(1)[0]
            if most_common[1] >= 3:
                return True

    return False


def _is_obvious_structural(chunk: Chunk, all_chunks: Optional[List[Chunk]] = None) -> bool:
    """Light-mode filter: only obviously structural chunks (>50% signals).
    
    Accepts optional all_chunks for API compatibility with _is_structural_chunk.
    """
    text = chunk.clean.lower()
    words = set(re.findall(r'\b[a-z]+\b', text))
    structural_found = words & STRUCTURAL_SIGNALS
    return len(words) > 0 and len(structural_found) / len(words) > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResolvedPassage:
    """A retrieval result resolved into a full passage."""
    original_results: List[RetrievalResult]
    passage_text: str
    page_start: int
    page_end: int
    chunk_indices: List[int]
    source_chunks: int
    was_structural: bool
    # Score breakdown from best original result
    avg_score: float = 0.0
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    matched_terms: List[str] = field(default_factory=list)


@dataclass
class RetrievalContext:
    """Complete context for LLM generation."""
    query: str
    method: str
    passages: List[ResolvedPassage]
    total_chars: int
    filtered_count: int
    fallback_used: bool
    raw_results: List[RetrievalResult]


# ═══════════════════════════════════════════════════════════════════════════
# Source Merger with Gap Filling
# ═══════════════════════════════════════════════════════════════════════════

def _should_merge_results(r1: RetrievalResult, r2: RetrievalResult, 
                          chunk_map: Dict[int, Chunk]) -> bool:
    """
    Decide if two results should be merged into one Source.
    """
    c1 = chunk_map.get(r1.chunk_index)
    c2 = chunk_map.get(r2.chunk_index)

    if not c1 or not c2:
        return False

    # Adjacent chunks (within MAX_GAP_SIZE) — primary merge rule
    if abs(c1.index - c2.index) <= MAX_GAP_SIZE:
        return True

    # Same page with index proximity guard
    if c1.page == c2.page and c1.page > 0:
        if abs(c1.index - c2.index) <= 8:
            return True

    # Overlapping or adjacent page ranges
    if c1.page_end is not None and c2.page is not None:
        if abs(c1.page_end - c2.page) <= 1:
            return True
    if c2.page_end is not None and c1.page is not None:
        if abs(c2.page_end - c1.page) <= 1:
            return True

    return False


def _fill_gaps(
    chunk_indices: List[int],
    all_chunks: List[Chunk],
    max_gap: int = MAX_GAP_SIZE,
) -> List[int]:
    """
    Fill small gaps between chunks to create a continuous passage.
    """
    if len(chunk_indices) < 2:
        return chunk_indices

    sorted_indices = sorted(chunk_indices)
    filled = [sorted_indices[0]]

    for i in range(1, len(sorted_indices)):
        prev = filled[-1]
        curr = sorted_indices[i]
        gap = curr - prev - 1

        if gap > 0 and gap <= max_gap:
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


def _merge_result_group(
    results: List[RetrievalResult],
    chunk_map: Dict[int, Chunk],
    all_chunks: List[Chunk],
    fill_gaps: bool = True,
) -> ResolvedPassage:
    """
    Merge a group of results into one ResolvedPassage.
    """
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    anchor_result = sorted_results[0]

    group_indices = sorted(set(r.chunk_index for r in results))

    if fill_gaps and len(group_indices) > 1:
        filled_indices = _fill_gaps(group_indices, all_chunks, max_gap=MAX_GAP_SIZE)
    else:
        filled_indices = group_indices

    # Apply chunk budget
    if len(filled_indices) > MAX_PASSAGE_CHUNKS:
        filled_indices = filled_indices[:MAX_PASSAGE_CHUNKS]

    # Build passage text from full span
    contributors = [chunk_map[idx] for idx in filled_indices if idx in chunk_map]
    passage_text = _stitch_chunks(contributors) if contributors else ""

    # Get page range from all involved chunks
    all_pages = []
    for idx in filled_indices:
        c = chunk_map.get(idx)
        if c:
            all_pages.extend([p for p in [c.page, c.page_end] if p is not None])

    page_start = min(all_pages) if all_pages else contributors[0].page if contributors else 0
    page_end = max(all_pages) if all_pages else contributors[-1].page_end if contributors else 0

    # Score breakdown from best result
    best = anchor_result
    matched = list(best.matched_terms) if best.matched_terms else []

    return ResolvedPassage(
        original_results=sorted_results,
        passage_text=passage_text,
        page_start=page_start,
        page_end=page_end,
        chunk_indices=filled_indices,
        source_chunks=len(filled_indices),
        was_structural=False,
        avg_score=best.score,
        semantic_score=getattr(best, 'semantic_score', None),
        keyword_score=getattr(best, 'keyword_score', None),
        rrf_score=getattr(best, 'rrf_score', None),
        rerank_score=getattr(best, 'rerank_score', None),
        matched_terms=matched,
    )


def _group_results_for_merging(
    results: List[RetrievalResult],
    chunk_map: Dict[int, Chunk],
) -> List[List[RetrievalResult]]:
    """Group retrieval results into mergeable clusters."""
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: r.chunk_index)
    groups = []
    current_group = [sorted_results[0]]

    for r in sorted_results[1:]:
        should_merge = any(
            _should_merge_results(r, existing_r, chunk_map)
            for existing_r in current_group
        )

        if should_merge:
            current_group.append(r)
        else:
            groups.append(current_group)
            current_group = [r]

    if current_group:
        groups.append(current_group)

    return groups


# ═══════════════════════════════════════════════════════════════════════════
# Core orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class RetrievalOrchestrator:
    """Wraps retrieval with structural filtering + source merging + gap filling."""

    def __init__(self, store: DarwinStore):
        self.store = store

    def retrieve(self, query, filename=None, method="hybrid", top_k=5, **kwargs):
        args = RetrievalArgs(
            query=query,
            method=RetrievalMethod(method),
            files=[filename] if filename else [],
            top_k=top_k,
            **kwargs
        )
        return handle_retrieval(args)

    def filter_structural(self, results, all_chunks, aggressive=True):
        is_bad = _is_structural_chunk if aggressive else _is_obvious_structural
        filtered, removed = [], 0

        for r in results:
            chunk = next((c for c in all_chunks if c.index == r.chunk_index), None)
            if not chunk:
                filtered.append(r)
                continue
            if is_bad(chunk, all_chunks):
                removed += 1
                darwin.log(f"[Filter] Chunk {r.chunk_index} structural", LogType.INFO)
            else:
                filtered.append(r)

        return filtered, removed

    def resolve_passages(self, results, filename, max_chars=MAX_PASSAGE_CHARS, 
                         merge_sources=True, fill_gaps=True):
        passages, total_chars = [], 0

        all_chunks = self.store.load_chunks(filename)
        chunk_map = {c.index: c for c in all_chunks}

        if merge_sources and len(results) > 1:
            groups = _group_results_for_merging(results, chunk_map)
        else:
            groups = [[r] for r in results]

        for group in groups:
            if merge_sources and len(group) > 1:
                passage = _merge_result_group(group, chunk_map, all_chunks, fill_gaps=fill_gaps)
            else:
                chunk = chunk_map.get(group[0].chunk_index)
                if not chunk:
                    continue
                from src.features.Ingestion.Chunking.ContextResolver import get_full_context
                context = get_full_context(chunk, self.store, filename, deduplicate=True)
                best = group[0]
                passage = ResolvedPassage(
                    original_results=group,
                    passage_text=context.text,
                    page_start=context.page,
                    page_end=context.page_end,
                    chunk_indices=context.contributing_indices,
                    source_chunks=len(context.contributing_indices),
                    was_structural=False,
                    avg_score=best.score,
                    semantic_score=getattr(best, 'semantic_score', None),
                    keyword_score=getattr(best, 'keyword_score', None),
                    rrf_score=getattr(best, 'rrf_score', None),
                    rerank_score=getattr(best, 'rerank_score', None),
                    matched_terms=list(best.matched_terms) if best.matched_terms else [],
                )

            # Apply character budget
            passage_chars = len(passage.passage_text)
            if total_chars + passage_chars > max_chars:
                remaining = max_chars - total_chars
                if remaining > 200:
                    passage.passage_text = _smart_truncate(passage.passage_text, remaining)
                    passage_chars = len(passage.passage_text)
                else:
                    break

            total_chars += passage_chars
            passages.append(passage)

        return passages

    def retrieve_and_resolve(
        self, query, filename, method="hybrid", top_k=5, 
        max_chars=MAX_PASSAGE_CHARS,
        filter_structural=True, aggressive_filter=True, 
        merge_sources=True, fill_gaps=True, **kwargs
    ):
        darwin.log(f"[Orchestrator] '{query}' [{method}]", LogType.INFO)
        results = self.retrieve(query, filename, method, top_k, **kwargs)
        raw_results = list(results)

        if not results:
            return RetrievalContext(
                query=query, method=method, passages=[], 
                total_chars=0, filtered_count=0, 
                fallback_used=False, raw_results=raw_results
            )

        chunks = self.store.load_chunks(filename)
        filtered_count = 0
        if filter_structural:
            results, filtered_count = self.filter_structural(results, chunks, aggressive=aggressive_filter)

        fallback_used = False
        if not results and filter_structural:
            darwin.log("[Orchestrator] Fallback to light filter", LogType.WARNING)
            results = self.retrieve(query, filename, method, top_k * 3, **kwargs)
            raw_results = list(results)
            results, _ = self.filter_structural(results, chunks, aggressive=False)
            fallback_used = True
            max_chars = max_chars * 2
            
            # NEW: Quality gate — reject weak fallback results
            if results:
                best_score = max(r.score for r in results)
                if best_score < FALLBACK_MIN_SCORE:
                    darwin.log(
                        f"[Orchestrator] Fallback results too weak (best={best_score:.3f} < {FALLBACK_MIN_SCORE}), returning empty",
                        LogType.WARNING
                    )
                    results = []

        passages = self.resolve_passages(
            results, filename, max_chars, 
            merge_sources=merge_sources, fill_gaps=fill_gaps
        )
        total_chars = sum(len(p.passage_text) for p in passages)

        return RetrievalContext(
            query=query, method=method, passages=passages, 
            total_chars=total_chars, filtered_count=filtered_count, 
            fallback_used=fallback_used, raw_results=raw_results
        )

    def build_llm_prompt(self, context):
        if not context.passages:
            return f"Question: {context.query}\n\nNo relevant passages found."

        blocks = []
        for i, p in enumerate(context.passages):
            # Score breakdown line
            score_parts = [f"avg={p.avg_score:.3f}"]
            if p.semantic_score:
                score_parts.append(f"sem={p.semantic_score:.3f}")
            if p.keyword_score:
                score_parts.append(f"kw={p.keyword_score:.3f}")
            if p.rrf_score:
                score_parts.append(f"rrf={p.rrf_score:.3f}")
            if p.rerank_score:
                score_parts.append(f"rerank={p.rerank_score:.3f}")

            gap_info = ""
            if len(p.chunk_indices) > len(p.original_results):
                gap_count = len(p.chunk_indices) - len(p.original_results)
                gap_info = f" (includes {gap_count} gap-filled chunks)"

            blocks.append(
                f"[Source {i+1}: Page {p.page_start}-{p.page_end} | "
                f"{' | '.join(score_parts)}{gap_info}]\n"
                f"{p.passage_text}\n"
            )

        passage_text = "\n---\n".join(blocks)
        warning = ""
        if context.fallback_used:
            warning = "\n[Note: All initial results were structural metadata. Using fallback.]\n"

        return f"""Answer based on the following passages.{warning}

Question: {context.query}

Passages:
{passage_text}

Answer:"""

    def diagnose_structural_chunks(self, filename, sample_size=20):
        chunks = self.store.load_chunks(filename)
        return [
            {
                "index": c.index,
                "is_structural": _is_structural_chunk(c, chunks),
                "length": len(c.clean),
                "preview": c.clean[:100].replace("\n", " ")
            }
            for c in chunks[:sample_size]
        ]


def retrieve_for_llm(query, filename, store, method="hybrid", top_k=5, 
                     max_chars=MAX_PASSAGE_CHARS, filter_structural=True, 
                     merge_sources=True, fill_gaps=True):
    """One-liner: retrieve, merge, resolve passages, build prompt."""
    orch = RetrievalOrchestrator(store)
    context = orch.retrieve_and_resolve(
        query=query, filename=filename, method=method, top_k=top_k,
        max_chars=max_chars, filter_structural=filter_structural,
        merge_sources=merge_sources, fill_gaps=fill_gaps
    )
    prompt = orch.build_llm_prompt(context)
    return prompt, context