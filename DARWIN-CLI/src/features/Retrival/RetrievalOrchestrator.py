from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any

from src.features.Retrival.main import handle_retrieval
from src.features.Retrival.RetrivalUtils import RetrievalArgs, RetrievalMethod
from src.features.Retrival.RetrievalResult import RetrievalResult, SearchAudit

from src.features.Ingestion.Chunking.lib import Chunk
from src.features.Ingestion.Chunking.ChunkerUtils import _split_sentences
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Query.StructuralClassifier import (
    classify_structural_type,
    get_penalty_factor,
)
from src.features.Knowledge.Wiki.WikiUtils import DocumentWiki
from src.features.Ingestion.Chunking.ContextResolver import get_full_context


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
    primary_concept: Optional[str] = None
    related_concepts: List[str] = field(default_factory=list)
    audits: List[SearchAudit] = field(default_factory=list)
    dominant_structural_type: Optional[str] = None
    overall_penalty_factor: float = 1.0


@dataclass
class RetrievalContext:
    query: str
    method: str
    passages: List[ResolvedPassage]
    total_chars: int
    penalized_count: int
    fallback_used: bool
    raw_results: List[RetrievalResult]
    audit_summary: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers — work with dicts (fast, no reconstruction overhead)
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a storage row dict for internal use. No-op if already normalized."""
    return row


def _get_index(row: dict[str, Any]) -> int:
    return row.get("index", -1)


def _get_page(row: dict[str, Any]) -> int:
    return row.get("page", 0) or 0


def _get_page_end(row: dict[str, Any]) -> int:
    return row.get("page_end", 0) or 0


def _get_text(row: dict[str, Any]) -> str:
    return row.get("text", "") or row.get("clean", "") or ""


def _should_merge_rows(
    r1: RetrievalResult,
    r2: RetrievalResult,
    chunk_map: Dict[int, dict[str, Any]],
    wiki: Optional[DocumentWiki] = None,
) -> bool:
    """Dict-based merge decision — no Chunk reconstruction needed."""
    c1 = chunk_map.get(r1.chunk_index)
    c2 = chunk_map.get(r2.chunk_index)
    if not c1 or not c2:
        return False

    idx1, idx2 = _get_index(c1), _get_index(c2)
    p1, p2 = _get_page(c1), _get_page(c2)
    pe1, pe2 = _get_page_end(c1), _get_page_end(c2)

    # Standard proximity
    if abs(idx1 - idx2) <= MAX_GAP_SIZE:
        return True
    if p1 == p2 and p1 > 0:
        if abs(idx1 - idx2) <= 8:
            return True
    if pe1 and p2:
        if abs(pe1 - p2) <= 1:
            return True
    if pe2 and p1:
        if abs(pe2 - p1) <= 1:
            return True

    # WIKI HINT: same concept gets larger merge tolerance
    if wiki is not None:
        for page in wiki.pages:
            indices = set(page.chunk_indices)
            if r1.chunk_index in indices and r2.chunk_index in indices:
                if abs(idx1 - idx2) <= 8:
                    return True

    return False


def _fill_gaps(
    chunk_indices: List[int],
    index_set: set[int],
    max_gap: int = MAX_GAP_SIZE,
) -> List[int]:
    """Fill gaps using a set of available indices (no Chunk objects needed)."""
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
                if fill_idx in index_set:
                    filled.append(fill_idx)
        filled.append(curr)
    return filled


def _stitch_texts(texts: List[str], pages: List[int]) -> str:
    """Stitch text segments with page breaks. Pure strings, no Chunk objects."""
    if not texts:
        return ""
    segments = []
    for i, (text, page) in enumerate(zip(texts, pages)):
        if i > 0 and page != pages[i - 1] and page > 0:
            segments.append(f"\n[Page {page}]\n")
        segments.append(text)
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


def _group_results_for_merging(
    results: List[RetrievalResult],
    chunk_map: Dict[int, dict[str, Any]],
    wiki: Optional[DocumentWiki] = None,
) -> List[List[RetrievalResult]]:
    if not results:
        return []
    sorted_results = sorted(results, key=lambda r: r.chunk_index)
    groups = []
    current_group = [sorted_results[0]]
    for r in sorted_results[1:]:
        should_merge = any(
            _should_merge_rows(r, existing_r, chunk_map, wiki=wiki)
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
# Passage resolution — dict-based for speed, Chunk-based only when needed
# ═══════════════════════════════════════════════════════════════════════════

def _build_chunk_map(
    store: DarwinStore,
    source: str,
    indices: List[int],
) -> Dict[int, dict[str, Any]]:
    """Fetch minimal chunk data as dicts. No Chunk reconstruction."""
    if not indices:
        return {}

    rows = store.query_records(
        filename=source,
        columns=["index", "text", "clean", "page", "page_end", "overlap_direction"],
        index__in=list(set(indices)),  # dedupe
    )
    return {_get_index(r): r for r in rows if _get_index(r) >= 0}


def _resolve_single_chunk_passage(
    result: RetrievalResult,
    store: DarwinStore,
    source: str,
    wiki: Optional[DocumentWiki],
) -> Optional[ResolvedPassage]:
    """Single-chunk passage using ContextResolver (handles Chunk or dict)."""
    # Fetch the chunk row
    rows = store.query_records(
        filename=source,
        columns=["index", "text", "clean", "page", "page_end", "overlap_direction", "links"],
        index__in=[result.chunk_index],
    )
    if not rows:
        return None

    chunk_row = rows[0]
    context = get_full_context(chunk_row, store, source, deduplicate=True)

    primary, related = _resolve_concepts_for_passage(context.contributing_indices, wiki)

    audit = getattr(result, 'audit', None)
    audits = [audit] if audit else []
    overall_factor = audit.penalty_factor if audit else 1.0
    dominant_type = audit.structural_type if audit else None

    return ResolvedPassage(
        original_results=[result],
        passage_text=context.text,
        page_start=context.page,
        page_end=context.page_end,
        chunk_indices=context.contributing_indices,
        source_chunks=len(context.contributing_indices),
        was_structural=(overall_factor < 1.0),
        source=source,
        avg_score=result.score,
        semantic_score=getattr(result, 'semantic_score', None),
        keyword_score=getattr(result, 'keyword_score', None),
        rrf_score=getattr(result, 'rrf_score', None),
        rerank_score=getattr(result, 'rerank_score', None),
        matched_terms=list(result.matched_terms) if result.matched_terms else [],
        primary_concept=primary,
        related_concepts=related,
        audits=audits,
        dominant_structural_type=dominant_type,
        overall_penalty_factor=overall_factor,
    )


def _resolve_merged_passage(
    group: List[RetrievalResult],
    store: DarwinStore,
    source: str,
    fill_gaps: bool,
    wiki: Optional[DocumentWiki],
) -> ResolvedPassage:
    """Multi-chunk merged passage — dict-based stitch, no Chunk reconstruction."""
    sorted_results = sorted(group, key=lambda r: r.score, reverse=True)
    anchor = sorted_results[0]

    group_indices = sorted(set(r.chunk_index for r in group))

    # Build chunk map for merge decision + gap filling
    chunk_map = _build_chunk_map(store, source, group_indices)

    if fill_gaps and len(group_indices) > 1:
        index_set = set(chunk_map.keys())
        filled_indices = _fill_gaps(group_indices, index_set, max_gap=MAX_GAP_SIZE)
    else:
        filled_indices = group_indices

    if len(filled_indices) > MAX_PASSAGE_CHUNKS:
        filled_indices = filled_indices[:MAX_PASSAGE_CHUNKS]

    # Fetch full texts for stitching
    text_rows = _build_chunk_map(store, source, filled_indices)
    contributors = [text_rows[idx] for idx in filled_indices if idx in text_rows]

    # Stitch texts directly
    texts = [_get_text(r) for r in contributors]
    pages = [_get_page(r) for r in contributors]
    passage_text = _stitch_texts(texts, pages) if contributors else ""

    # Page span
    all_pages = []
    for idx in filled_indices:
        r = text_rows.get(idx)
        if r:
            all_pages.extend([p for p in [_get_page(r), _get_page_end(r)] if p > 0])

    page_start = min(all_pages) if all_pages else _get_page(contributors[0]) if contributors else 0
    page_end = max(all_pages) if all_pages else _get_page_end(contributors[-1]) if contributors else 0

    matched = list(anchor.matched_terms) if anchor.matched_terms else []
    primary, related = _resolve_concepts_for_passage(filled_indices, wiki)

    # Aggregate audits from all results in the group
    audits = []
    types = []
    factors = []
    for res in sorted_results:
        a = getattr(res, 'audit', None)
        if a:
            audits.append(a)
            if a.structural_type:
                types.append(a.structural_type)
            factors.append(a.penalty_factor)

    dominant_type = None
    if types:
        dominant_type = max(set(types), key=types.count)
    overall_factor = min(factors) if factors else 1.0

    return ResolvedPassage(
        original_results=sorted_results,
        passage_text=passage_text,
        page_start=page_start,
        page_end=page_end,
        chunk_indices=filled_indices,
        source_chunks=len(filled_indices),
        was_structural=(overall_factor < 1.0),
        source=source,
        avg_score=anchor.score,
        semantic_score=getattr(anchor, 'semantic_score', None),
        keyword_score=getattr(anchor, 'keyword_score', None),
        rrf_score=getattr(anchor, 'rrf_score', None),
        rerank_score=getattr(anchor, 'rerank_score', None),
        matched_terms=matched,
        primary_concept=primary,
        related_concepts=related,
        audits=audits,
        dominant_structural_type=dominant_type,
        overall_penalty_factor=overall_factor,
    )


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

    def penalize_structural(self, results, chunk_rows, aggressive=True):
        """
        Apply structural score penalties instead of filtering.
        Attaches SearchAudit to each result explaining the penalty.
        """
        penalized = []
        penalty_count = 0

        for r in results:
            chunk_row = next((c for c in chunk_rows if _get_index(c) == r.chunk_index), None)
            if not chunk_row:
                penalized.append(r)
                continue

            stype = classify_structural_type(chunk_row, chunk_rows, aggressive=aggressive)
            factor = get_penalty_factor(stype)

            raw_score = r.score
            new_score = raw_score * factor

            audit = SearchAudit(
                raw_score=raw_score,
                final_score=new_score,
                structural_type=stype,
                penalty_factor=factor,
                was_penalized=(factor < 1.0),
                penalty_reason=f"structural_type={stype}" if stype else None,
            )
            r.audit = audit
            r.score = new_score

            if factor < 1.0:
                penalty_count += 1
                darwin.log(
                    f"[Penalty] Chunk {r.chunk_index} {stype} factor={factor} "
                    f"{raw_score:.3f}→{new_score:.3f}",
                    LogType.INFO,
                )

            penalized.append(r)

        penalized.sort(key=lambda x: x.score, reverse=True)
        return penalized, penalty_count

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
            if merge_sources and len(source_results) > 1:
                # Build minimal chunk map for merge decisions
                all_indices = list(set(r.chunk_index for r in source_results))
                chunk_map = _build_chunk_map(self.store, source, all_indices)
                groups = _group_results_for_merging(source_results, chunk_map, wiki=wiki)
            else:
                groups = [[r] for r in source_results]

            for group in groups:
                if merge_sources and len(group) > 1:
                    passage = _resolve_merged_passage(
                        group, self.store, source,
                        fill_gaps=fill_gaps, wiki=wiki,
                    )
                else:
                    passage = _resolve_single_chunk_passage(
                        group[0], self.store, source, wiki=wiki,
                    )

                if not passage:
                    continue

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

        # Retrieve larger pool so body chunks can surface after penalty
        pool_multiplier = 3 if filter_structural else 1
        results = self.retrieve(query, files, method, top_k * pool_multiplier, **kwargs)
        raw_results = list(results)

        if not results:
            return RetrievalContext(
                query=query, method=method, passages=[],
                total_chars=0, penalized_count=0,
                fallback_used=False, raw_results=raw_results,
                audit_summary={"error": "no retrieval results"},
            )

        penalized_count = 0
        if filter_structural:
            by_source = defaultdict(list)
            for r in results:
                by_source[r.source].append(r)

            penalized_results = []
            for source, source_results in by_source.items():
                # Load minimal rows for structural penalty
                indices = list(set(r.chunk_index for r in source_results))
                chunk_rows = self.store.query_records(
                    source,
                    columns=["index", "text", "clean", "heading", "page", "token_count"],
                    index__in=indices,
                )
                res, rem = self.penalize_structural(source_results, chunk_rows, aggressive=aggressive_filter)
                penalized_results.extend(res)
                penalized_count += rem
            results = penalized_results

        # Truncate to top_k after penalty application
        results = results[:top_k]

        has_body = any(
            getattr(r, 'audit', None) and r.audit.penalty_factor == 1.0
            for r in results
        )
        fallback_used = not has_body and len(results) > 0

        if fallback_used:
            darwin.log(
                "[RetrievalOrchestrator] No unpenalized body chunks in top results; "
                "proceeding with penalized structural chunks.",
                LogType.WARNING,
            )
            max_chars = max_chars * 2

        # Build audit summary
        audit_summary = {
            "total_results": len(results),
            "penalized_count": penalized_count,
            "structural_type_breakdown": {},
            "fallback_used": fallback_used,
        }
        for r in results:
            a = getattr(r, 'audit', None)
            if a and a.structural_type:
                t = a.structural_type
                audit_summary["structural_type_breakdown"][t] = audit_summary["structural_type_breakdown"].get(t, 0) + 1

        passages = self.resolve_passages(
            results, max_chars=max_chars,
            merge_sources=merge_sources, fill_gaps=fill_gaps,
            wiki=wiki,
        )
        total_chars = sum(len(p.passage_text) for p in passages)

        return RetrievalContext(
            query=query, method=method, passages=passages,
            total_chars=total_chars, penalized_count=penalized_count,
            fallback_used=fallback_used, raw_results=raw_results,
            audit_summary=audit_summary,
        )

    def build_llm_prompt(self, context):
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

            penalty_info = ""
            if p.overall_penalty_factor < 1.0:
                penalty_info = f" | PENALTY: {p.dominant_structural_type} factor={p.overall_penalty_factor}"

            blocks.append(
                f"[Source {i+1}: {p.source} | Page {p.page_start}-{p.page_end}{concept_info} | "
                f"{' | '.join(score_parts)}{gap_info}{penalty_info}]\n"
                f"{p.passage_text}\n"
            )

        passage_text = "\n---\n".join(blocks)
        warning = ""
        if context.fallback_used:
            warning = "\n[Note: Top results are structural metadata (penalized).]\n"

        return f"""Answer based on the following passages.{warning}

Question: {context.query}

Passages:
{passage_text}

Answer:"""

    def diagnose_structural_chunks(self, filename, sample_size=20):
        """Fetch sample chunks as dicts, reconstruct only those being diagnosed."""
        rows = self.store.query_records(
            filename,
            columns=["index", "text", "clean", "heading", "page", "token_count"],
        )[:sample_size]

        # For is_structural_chunk, we need Chunk objects and the full list
        # Reconstruct all sample chunks at once (small batch)
        chunks = self.store.reconstruct_records(
            filename,
            cls=Chunk,
            columns=["index", "text", "clean", "heading", "page", "token_count"],
            index__in=[_get_index(r) for r in rows],
        )

        return [
            {
                "index": c.index,
                "is_structural": classify_structural_type(c, chunks, aggressive=True) is not None,
                "structural_type": classify_structural_type(c, chunks, aggressive=True),
                "length": len(c.clean or c.text),
                "preview": (c.clean or c.text)[:100].replace("\n", " ")
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