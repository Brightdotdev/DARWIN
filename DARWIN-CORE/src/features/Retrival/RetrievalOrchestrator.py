from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Optional, Any

from src.features.Ingestion.Chunking.StructuralClassifier import (
    get_penalty_factor,
)
from src.features.Retrival.main import handle_retrieval
from src.features.Retrival.RetrivalUtils import RetrievalArgs, RetrievalMethod
from src.features.Retrival.RetrievalResult import (
    RetrievalResult,
    SearchAudit,
)

from src.features.Ingestion.Chunking.ChunkerUtils import split_sentences
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType

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
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

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
    max_gap: int = MAX_GAP_SIZE,
    max_same_page_gap: int = 8,
) -> bool:
    """Dict-based merge decision."""
    c1 = chunk_map.get(r1.chunk_index)
    c2 = chunk_map.get(r2.chunk_index)
    if not c1 or not c2:
        return False

    idx1, idx2 = _get_index(c1), _get_index(c2)
    p1, p2 = _get_page(c1), _get_page(c2)
    pe1, pe2 = _get_page_end(c1), _get_page_end(c2)

    # Standard proximity
    if abs(idx1 - idx2) <= max_gap:
        return True
    if p1 == p2 and p1 > 0:
        if abs(idx1 - idx2) <= max_same_page_gap:
            return True
    if pe1 and p2:
        if abs(pe1 - p2) <= 1:
            return True
    if pe2 and p1:
        if abs(pe2 - p1) <= 1:
            return True

    return False

def _fill_gaps(
    chunk_indices: List[int],
    index_set: set[int],
    max_gap: int = MAX_GAP_SIZE,
) -> List[int]:
    """Fill gaps using a set of available indices (must include candidate gaps)."""
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
    sentences = split_sentences(combined)
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

def _group_results_for_merging(
    results: List[RetrievalResult],
    chunk_map: Dict[int, dict[str, Any]],
    max_gap: int = MAX_GAP_SIZE,
    max_same_page_gap: int = 8,
) -> List[List[RetrievalResult]]:
    """
    Group results using connected components (not a chain).
    Only results that are directly or transitively mergeable end up together.
    """
    if not results:
        return []

    # Build adjacency list
    n = len(results)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if _should_merge_rows(results[i], results[j], chunk_map,
                                  max_gap, max_same_page_gap):
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    groups = []
    for i in range(n):
        if i not in visited:
            stack = [i]
            visited.add(i)
            comp = []
            while stack:
                cur = stack.pop()
                comp.append(results[cur])
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            groups.append(comp)
    return groups

# ═══════════════════════════════════════════════════════════════════════════
# Passage resolution
# ═══════════════════════════════════════════════════════════════════════════

def _build_chunk_map(
    store: DarwinStore,
    source: str,
    indices: List[int],
) -> Dict[int, dict[str, Any]]:
    """Fetch minimal chunk data as dicts."""
    if not indices:
        return {}
    rows = store.query_records(
        filename=source,
        columns=["index", "text", "clean", "page", "page_end", "overlap_direction"],
        index__in=list(set(indices)),
    )
    return {_get_index(r): r for r in rows if _get_index(r) >= 0}

def _build_chunk_map_range(
    store: DarwinStore,
    source: str,
    min_idx: int,
    max_idx: int,
) -> Dict[int, dict[str, Any]]:
    """Fetch chunk data for a continuous index range."""
    indices = list(range(min_idx, max_idx + 1))
    return _build_chunk_map(store, source, indices)

def _resolve_single_chunk_passage(
    result: RetrievalResult,
    store: DarwinStore,
    source: str,
) -> Optional[ResolvedPassage]:
    """Single-chunk passage using ContextResolver."""
    rows = store.query_records(
        filename=source,
        columns=["index", "text", "clean", "page", "page_end", "overlap_direction", "links"],
        index__in=[result.chunk_index],
    )
    if not rows:
        return None

    chunk_row = rows[0]
    context = get_full_context(chunk_row, store, source, deduplicate=True)

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
        primary_concept=None,
        related_concepts=[],
        audits=audits,
        dominant_structural_type=dominant_type,
        overall_penalty_factor=overall_factor,
    )

def _resolve_merged_passage(
    group: List[RetrievalResult],
    store: DarwinStore,
    source: str,
    fill_gaps: bool,
    chunk_map: Optional[Dict[int, dict[str, Any]]] = None,
    max_gap: int = MAX_GAP_SIZE,
) -> ResolvedPassage:
    """Multi-chunk merged passage — dict-based stitch, no Chunk reconstruction."""
    sorted_results = sorted(group, key=lambda r: r.score, reverse=True)
    anchor = sorted_results[0]

    group_indices = sorted(set(r.chunk_index for r in group))

    # Use provided chunk_map or build it
    if chunk_map is None:
        chunk_map = _build_chunk_map(store, source, group_indices)

    if fill_gaps and len(group_indices) > 1:
        # Extend map to include all indices that could fill gaps
        min_idx = min(group_indices) - max_gap
        max_idx = max(group_indices) + max_gap
        extended_map = _build_chunk_map_range(store, source, min_idx, max_idx)
        # Merge with existing chunk_map (priority to existing)
        merged_map = {**extended_map, **chunk_map}
        index_set = set(merged_map.keys())
        filled_indices = _fill_gaps(group_indices, index_set, max_gap=max_gap)
    else:
        filled_indices = group_indices
        merged_map = chunk_map

    if len(filled_indices) > MAX_PASSAGE_CHUNKS:
        filled_indices = filled_indices[:MAX_PASSAGE_CHUNKS]

    # Fetch full texts for stitching using merged_map (which includes gaps)
    contributors = [merged_map[idx] for idx in filled_indices if idx in merged_map]

    texts = [_get_text(r) for r in contributors]
    pages = [_get_page(r) for r in contributors]
    passage_text = _stitch_texts(texts, pages) if contributors else ""

    # Page span
    all_pages = []
    for idx in filled_indices:
        r = merged_map.get(idx)
        if r:
            all_pages.extend([p for p in [_get_page(r), _get_page_end(r)] if p > 0])

    page_start = min(all_pages) if all_pages else _get_page(contributors[0]) if contributors else 0
    page_end = max(all_pages) if all_pages else _get_page_end(contributors[-1]) if contributors else 0

    matched = list(anchor.matched_terms) if anchor.matched_terms else []

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
        primary_concept=None,
        related_concepts=[],
        audits=audits,
        dominant_structural_type=dominant_type,
        overall_penalty_factor=overall_factor,
    )

# ═══════════════════════════════════════════════════════════════════════════
# Core orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class RetrievalOrchestrator:
    def __init__(
        self,
        store: DarwinStore,
        max_gap_size: int = MAX_GAP_SIZE,
        max_same_page_gap: int = 8,
    ):
        self.store = store
        self.max_gap_size = max_gap_size
        self.max_same_page_gap = max_same_page_gap

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

    def penalize_structural(self, results, chunk_rows):
        """
        Apply structural score penalties using pre-computed chunk metadata.
        FIX: uses r.chunk_index (not r.index) and always attaches an audit.
        """
        penalized = []
        penalty_count = 0
        chunk_map = {_get_index(c): c for c in chunk_rows}

        for r in results:
            chunk = chunk_map.get(r.chunk_index)
            if not chunk:
                # Create a default audit treating the chunk as body
                audit = SearchAudit(
                    raw_score=r.score,
                    final_score=r.score,
                    structural_type="body",
                    penalty_factor=1.0,
                    was_penalized=False,
                )
                r.audit = audit
                penalized.append(r)
                continue

            stype = chunk.get("structural_type", "body")
            factor = get_penalty_factor(stype)
            raw_score = r.score
            new_score = raw_score * factor

            audit = SearchAudit(
                raw_score=raw_score,
                final_score=new_score,
                structural_type=stype,
                penalty_factor=factor,
                was_penalized=(factor < 1.0),
                penalty_reason=f"structural_type={stype}" if factor < 1.0 else None,
            )
            r.audit = audit
            r.score = new_score

            if factor < 1.0:
                penalty_count += 1

            penalized.append(r)

        penalized.sort(key=lambda x: x.score, reverse=True)
        return penalized, penalty_count

    def resolve_passages(
        self,
        results,
        max_chars=MAX_PASSAGE_CHARS,
        merge_sources=True,
        fill_gaps=True,
    ):
        passages, total_chars = [], 0
        if not results:
            return passages

        by_source = defaultdict(list)
        for r in results:
            by_source[r.source].append(r)

        for source, source_results in by_source.items():
            if merge_sources and len(source_results) > 1:
                all_indices = list(set(r.chunk_index for r in source_results))
                chunk_map = _build_chunk_map(self.store, source, all_indices)
                groups = _group_results_for_merging(
                    source_results,
                    chunk_map,
                    max_gap=self.max_gap_size,
                    max_same_page_gap=self.max_same_page_gap,
                )
            else:
                groups = [[r] for r in source_results]
                chunk_map = None   # single result, no need to pass

            for group in groups:
                if merge_sources and len(group) > 1:
                    # Pass cached chunk_map to avoid duplicate queries
                    passage = _resolve_merged_passage(
                        group, self.store, source,
                        fill_gaps=fill_gaps,
                        chunk_map=chunk_map,
                        max_gap=self.max_gap_size,
                    )
                else:
                    passage = _resolve_single_chunk_passage(
                        group[0], self.store, source,
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
        merge_sources=True,
        fill_gaps=True,
        **kwargs
    ):
        darwin.log(f"[RetrievalOrchestrator] '{query}' [{method}]", LogType.INFO)

        if isinstance(files, str):
            files = [files]

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
                indices = list(set(r.chunk_index for r in source_results))
                # FIX: add "structural_type" column
                chunk_rows = self.store.query_records(
                    source,
                    columns=["index", "text", "clean", "heading", "page", "token_count", "structural_type"],
                    index__in=indices,
                )
                res, rem = self.penalize_structural(source_results, chunk_rows)
                penalized_results.extend(res)
                penalized_count += rem
            results = penalized_results

        results = results[:top_k]

        # Fallback detection: true if NO result has an audit with penalty_factor == 1.0
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
        """Fetch sample chunks and report structural metadata."""
        from src.features.Ingestion.Chunking.StructuralClassifier import is_structural_for_embedding
        
        rows = self.store.query_records(
            filename,
            columns=["index", "text", "clean", "heading", "page", "structural_type"],
        )[:sample_size]

        return [
            {
                "index": r.get("index"),
                "is_structural": is_structural_for_embedding(r.get("structural_type")),
                "structural_type": r.get("structural_type"),
                "length": len(r.get("clean", "") or r.get("text", "")),
                "preview": (r.get("clean", "") or r.get("text", ""))[:100].replace("\n", " ")
            }
            for r in rows
        ]

def retrieve_for_llm(
    query, files, store, method="hybrid", top_k=5,
    max_chars=MAX_PASSAGE_CHARS, filter_structural=True,
    merge_sources=True, fill_gaps=True,
):
    orch = RetrievalOrchestrator(store)
    context = orch.retrieve_and_resolve(
        query=query, files=files, method=method, top_k=top_k,
        max_chars=max_chars, filter_structural=filter_structural,
        merge_sources=merge_sources, fill_gaps=fill_gaps,
    )
    prompt = orch.build_llm_prompt(context)
    return prompt, context