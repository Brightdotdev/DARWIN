"""
src/core/Knowledge/RetrievalOrchestrator.py
---------------------------------------------
Dense retrieval orchestrator with wiki-aware merging.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import re

from src.features.Retrival.main import handle_retrieval
from src.features.Retrival.RetrivalUtils import RetrievalArgs, RetrievalMethod
from src.features.Retrival.RetrievalResult import RetrievalResult

from src.features.Ingestion.Chunking.lib import Chunk
from src.features.Ingestion.Chunking.ChunkerUtils import _split_sentences
from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Query.StructuralClassifier import is_structural_chunk
from src.features.Knowledge.Wiki.WikiUtils import DocumentWiki


# ═══════════════════════════════════════════════════════════════════════════
# Tunables
# ═══════════════════════════════════════════════════════════════════════════

MAX_GAP_SIZE = 2
MAX_PASSAGE_CHUNKS = 12
MAX_PASSAGE_CHARS = 6000
FALLBACK_MIN_SCORE = 0.3


# ═══════════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResolvedPassage:
    original_results: List[RetrievalResult]
    passage_text: str
    page_start: int
    page_end: int
    chunk_indices: List[int]
    source_chunks: int
    was_structural: bool
    source: str = ""
    avg_score: float = 0.0
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    matched_terms: List[str] = field(default_factory=list)
    primary_concept: Optional[str] = None        # NEW: wiki concept tag
    related_concepts: List[str] = field(default_factory=list)  # NEW


@dataclass
class RetrievalContext:
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

def _should_merge_results(
    r1: RetrievalResult,
    r2: RetrievalResult,
    chunk_map: Dict[int, Chunk],
    wiki: Optional[DocumentWiki] = None,
) -> bool:
    c1 = chunk_map.get(r1.chunk_index)
    c2 = chunk_map.get(r2.chunk_index)
    if not c1 or not c2:
        return False

    # Standard proximity
    if abs(c1.index - c2.index) <= MAX_GAP_SIZE:
        return True
    if c1.page == c2.page and c1.page > 0:
        if abs(c1.index - c2.index) <= 8:
            return True
    if c1.page_end is not None and c2.page is not None:
        if abs(c1.page_end - c2.page) <= 1:
            return True
    if c2.page_end is not None and c1.page is not None:
        if abs(c2.page_end - c1.page) <= 1:
            return True

    # WIKI HINT: same concept gets larger merge tolerance (up to 8 chunks)
    if wiki is not None:
        for page in wiki.pages:
            indices = set(page.chunk_indices)
            if r1.chunk_index in indices and r2.chunk_index in indices:
                if abs(c1.index - c2.index) <= 8:
                    return True

    return False


def _fill_gaps(
    chunk_indices: List[int],
    all_chunks: List[Chunk],
    max_gap: int = MAX_GAP_SIZE,
) -> List[int]:
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
    if not chunks:
        return ""
    segments = []
    for i, chunk in enumerate(chunks):
        if i > 0 and chunk.page != chunks[i - 1].page and chunk.page > 0:
            segments.append(f"\n[Page {chunk.page}]\n")
        segments.append(chunk.text)
    combined = " ".join(segments)
    sentences = _split_sentences(combined)
    deduped = []
    for s in sentences:
        norm = " ".join(s.split())
        if deduped and norm == " ".join(deduped[-1].split()):
            continue
        deduped.append(s)
    return " ".join(deduped)


def _smart_truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    text = text[:max_len]
    for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n', '." ', '!" ', '?" ']:
        idx = text.rfind(punct)
        if idx != -1 and idx > len(text) * 0.5:
            return text[:idx + 1]
    return text


def _resolve_concepts_for_passage(
    chunk_indices: List[int],
    wiki: Optional[DocumentWiki],
) -> Tuple[Optional[str], List[str]]:
    """Find primary concept and related concepts for a passage."""
    if not wiki:
        return None, []

    concept_hits: Dict[str, int] = {}
    for idx in chunk_indices:
        for page in wiki.pages:
            if idx in page.chunk_indices:
                concept_hits[page.concept] = concept_hits.get(page.concept, 0) + 1

    if not concept_hits:
        return None, []

    primary = max(concept_hits.items(), key=lambda x: x[1])[0]
    related = [c for c in concept_hits.keys() if c != primary]
    return primary, related


def _merge_result_group(
    results: List[RetrievalResult],
    chunk_map: Dict[int, Chunk],
    all_chunks: List[Chunk],
    fill_gaps: bool = True,
    wiki: Optional[DocumentWiki] = None,
) -> ResolvedPassage:
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    anchor_result = sorted_results[0]
    source = sorted_results[0].source if sorted_results else ""

    group_indices = sorted(set(r.chunk_index for r in results))

    if fill_gaps and len(group_indices) > 1:
        filled_indices = _fill_gaps(group_indices, all_chunks, max_gap=MAX_GAP_SIZE)
    else:
        filled_indices = group_indices

    if len(filled_indices) > MAX_PASSAGE_CHUNKS:
        filled_indices = filled_indices[:MAX_PASSAGE_CHUNKS]

    contributors = [chunk_map[idx] for idx in filled_indices if idx in chunk_map]
    passage_text = _stitch_chunks(contributors) if contributors else ""

    all_pages = []
    for idx in filled_indices:
        c = chunk_map.get(idx)
        if c:
            all_pages.extend([p for p in [c.page, c.page_end] if p is not None])

    page_start = min(all_pages) if all_pages else contributors[0].page if contributors else 0
    page_end = max(all_pages) if all_pages else contributors[-1].page_end if contributors else 0

    best = anchor_result
    matched = list(best.matched_terms) if best.matched_terms else []

    # NEW: resolve wiki concepts
    primary_concept, related_concepts = _resolve_concepts_for_passage(filled_indices, wiki)

    return ResolvedPassage(
        original_results=sorted_results,
        passage_text=passage_text,
        page_start=page_start,
        page_end=page_end,
        chunk_indices=filled_indices,
        source_chunks=len(filled_indices),
        was_structural=False,
        source=source,
        avg_score=best.score,
        semantic_score=getattr(best, 'semantic_score', None),
        keyword_score=getattr(best, 'keyword_score', None),
        rrf_score=getattr(best, 'rrf_score', None),
        rerank_score=getattr(best, 'rerank_score', None),
        matched_terms=matched,
        primary_concept=primary_concept,
        related_concepts=related_concepts,
    )


def _group_results_for_merging(
    results: List[RetrievalResult],
    chunk_map: Dict[int, Chunk],
    wiki: Optional[DocumentWiki] = None,
) -> List[List[RetrievalResult]]:
    if not results:
        return []
    sorted_results = sorted(results, key=lambda r: r.chunk_index)
    groups = []
    current_group = [sorted_results[0]]
    for r in sorted_results[1:]:
        should_merge = any(
            _should_merge_results(r, existing_r, chunk_map, wiki=wiki)
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
    def __init__(self, store: DarwinStore):
        self.store = store

    def retrieve(self, query, files=None, method="hybrid", top_k=5, **kwargs):
        if isinstance(files, str):
            files = [files]
        args = RetrievalArgs(
            query=query,
            method=RetrievalMethod(method),
            files=files or [],
            top_k=top_k,
            **kwargs
        )
        return handle_retrieval(args, self.store)

    def filter_structural(self, results, all_chunks, aggressive=True):
        filtered, removed = [], 0
        for r in results:
            chunk = next((c for c in all_chunks if c.index == r.chunk_index), None)
            if not chunk:
                filtered.append(r)
                continue
            if is_structural_chunk(chunk, all_chunks, aggressive=aggressive):
                removed += 1
                darwin.log(f"[Filter] Chunk {r.chunk_index} structural", LogType.INFO)
            else:
                filtered.append(r)
        return filtered, removed

    def resolve_passages(
        self,
        results,
        max_chars=MAX_PASSAGE_CHARS,
        merge_sources=True,
        fill_gaps=True,
        wiki: Optional[DocumentWiki] = None,
    ):
        passages, total_chars = [], 0
        if not results:
            return passages

        by_source = defaultdict(list)
        for r in results:
            by_source[r.source].append(r)

        for source, source_results in by_source.items():
            all_chunks = self.store.load_chunks(source)
            chunk_map = {c.index: c for c in all_chunks}

            if merge_sources and len(source_results) > 1:
                groups = _group_results_for_merging(source_results, chunk_map, wiki=wiki)
            else:
                groups = [[r] for r in source_results]

            for group in groups:
                if merge_sources and len(group) > 1:
                    passage = _merge_result_group(
                        group, chunk_map, all_chunks,
                        fill_gaps=fill_gaps, wiki=wiki,
                    )
                else:
                    chunk = chunk_map.get(group[0].chunk_index)
                    if not chunk:
                        continue
                    from src.features.Ingestion.Chunking.ContextResolver import get_full_context
                    context = get_full_context(chunk, self.store, source, deduplicate=True)
                    best = group[0]

                    # NEW: resolve concepts for single-chunk passage too
                    primary_concept, related_concepts = _resolve_concepts_for_passage(
                        context.contributing_indices, wiki
                    )

                    passage = ResolvedPassage(
                        original_results=group,
                        passage_text=context.text,
                        page_start=context.page,
                        page_end=context.page_end,
                        chunk_indices=context.contributing_indices,
                        source_chunks=len(context.contributing_indices),
                        was_structural=False,
                        source=source,
                        avg_score=best.score,
                        semantic_score=getattr(best, 'semantic_score', None),
                        keyword_score=getattr(best, 'keyword_score', None),
                        rrf_score=getattr(best, 'rrf_score', None),
                        rerank_score=getattr(best, 'rerank_score', None),
                        matched_terms=list(best.matched_terms) if best.matched_terms else [],
                        primary_concept=primary_concept,
                        related_concepts=related_concepts,
                    )

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
        self,
        query,
        files=None,
        method="hybrid",
        top_k=5,
        max_chars=MAX_PASSAGE_CHARS,
        filter_structural=True,
        aggressive_filter=True,
        merge_sources=True,
        fill_gaps=True,
        wiki: Optional[DocumentWiki] = None,
        **kwargs
    ):
        darwin.log(f"[RetrievalOrchestrator] '{query}' [{method}]", LogType.INFO)

        if isinstance(files, str):
            files = [files]

        results = self.retrieve(query, files, method, top_k, **kwargs)
        raw_results = list(results)

        if not results:
            return RetrievalContext(
                query=query, method=method, passages=[],
                total_chars=0, filtered_count=0,
                fallback_used=False, raw_results=raw_results,
            )

        filtered_count = 0
        if filter_structural:
            by_source = defaultdict(list)
            for r in results:
                by_source[r.source].append(r)

            filtered_results = []
            for source, source_results in by_source.items():
                chunks = self.store.load_chunks(source)
                res, rem = self.filter_structural(source_results, chunks, aggressive=aggressive_filter)
                filtered_results.extend(res)
                filtered_count += rem
            results = filtered_results

        fallback_used = False
        if not results and filter_structural:
            darwin.log("[RetrievalOrchestrator] Fallback to light filter", LogType.WARNING)
            results = self.retrieve(query, files, method, top_k * 3, **kwargs)
            raw_results = list(results)

            by_source = defaultdict(list)
            for r in results:
                by_source[r.source].append(r)

            filtered_results = []
            for source, source_results in by_source.items():
                chunks = self.store.load_chunks(source)
                res, _ = self.filter_structural(source_results, chunks, aggressive=False)
                filtered_results.extend(res)
            results = filtered_results

            fallback_used = True
            max_chars = max_chars * 2

            if results:
                best_score = max(r.score for r in results)
                if best_score < FALLBACK_MIN_SCORE:
                    darwin.log(
                        f"[RetrievalOrchestrator] Fallback results too weak "
                        f"(best={best_score:.3f} < {FALLBACK_MIN_SCORE}), returning empty",
                        LogType.WARNING,
                    )
                    results = []

        passages = self.resolve_passages(
            results, max_chars=max_chars,
            merge_sources=merge_sources, fill_gaps=fill_gaps,
            wiki=wiki,
        )
        total_chars = sum(len(p.passage_text) for p in passages)

        return RetrievalContext(
            query=query, method=method, passages=passages,
            total_chars=total_chars, filtered_count=filtered_count,
            fallback_used=fallback_used, raw_results=raw_results,
        )

    def build_llm_prompt(self, context):
        """Backward-compatible prompt builder. SearchOrchestrator replaces this."""
        if not context.passages:
            return f"Question: {context.query}\n\nNo relevant passages found."

        blocks = []
        for i, p in enumerate(context.passages):
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

            concept_info = ""
            if p.primary_concept:
                concept_info = f" | Concept: {p.primary_concept}"

            blocks.append(
                f"[Source {i+1}: {p.source} | Page {p.page_start}-{p.page_end}{concept_info} | "
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
                "is_structural": is_structural_chunk(c, chunks, aggressive=True),
                "length": len(getattr(c, 'clean', getattr(c, 'text', ''))),
                "preview": getattr(c, 'clean', getattr(c, 'text', ''))[:100].replace("\n", " ")
            }
            for c in chunks[:sample_size]
        ]


def retrieve_for_llm(
    query, files, store, method="hybrid", top_k=5,
    max_chars=MAX_PASSAGE_CHARS, filter_structural=True,
    merge_sources=True, fill_gaps=True,
    wiki: Optional[DocumentWiki] = None,
):
    orch = RetrievalOrchestrator(store)
    context = orch.retrieve_and_resolve(
        query=query, files=files, method=method, top_k=top_k,
        max_chars=max_chars, filter_structural=filter_structural,
        merge_sources=merge_sources, fill_gaps=fill_gaps,
        wiki=wiki,
    )
    prompt = orch.build_llm_prompt(context)
    return prompt, context