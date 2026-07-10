"""
src/core/Orchestrators/RetrievalOrchestrator.py
-----------------------------------------------
Retrieval Orchestrator – debloated, robust, with fallback recursion fix.
"""

from __future__ import annotations

import time
import traceback
from collections import defaultdict
from typing import List, Optional, Dict, Any, Set, Tuple

from src.core.Logger.DarwinLogger import darwin
from src.core.Storage.DarwinStore import DarwinStore

from src.core.Retrival.main import handle_retrieval
from src.core.Retrival.RetrievalArgs import RetrievalArgs, RetrievalMethod
from src.core.Retrival.RetrievalResult import RetrievalResult
from src.core.Retrival.lib import SearchAudit

from src.core.Pipeline.Ingestion.Chunking.StructuralClassifier import get_penalty_factor
from src.core.Pipeline.Ingestion.Chunking.ChunkerUtils import split_sentences

from src.core.Retrival.OrchestratorTypes import (
    OrchestratorConfig,
    MergeStrategy,
    ResolvedPassage,
    PassageMetadata,
    RetrievalContext,
)

from src.core.Orchestrators.lib import RetrievalParameters as RetrievalInput  # public API


class RetrievalOrchestrator:
    """
    Orchestrates retrieval: retrieve, filter, penalize, merge, and resolve passages.
    """

    def __init__(self, store: DarwinStore, config: Optional[RetrievalInput] = None):
        self.store = store
        self._default_config = config or RetrievalInput()
        self._chunk_meta_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}

    # ================================================================
    # Public API
    # ================================================================

    def retrieve_and_resolve(
        self,
        config: Optional[RetrievalInput] = None,
    ) -> Optional[RetrievalContext]:
        """Public API — accepts RetrievalParameters from Orchestrators.lib."""
        params = config or self._default_config
        cfg = params.build()
        if isinstance(cfg.files, str):
            cfg.files = [cfg.files]
        return self._retrieve_and_resolve_internal(cfg, _is_fallback=False)

    # ════════════════════════════════════════════════════════════════
    # Internal — operates on raw OrchestratorConfig
    # ════════════════════════════════════════════════════════════════

    def _retrieve_and_resolve_internal(
        self,
        cfg: OrchestratorConfig,
        _is_fallback: bool = False,
    ) -> Optional[RetrievalContext]:
        start_time = time.perf_counter()

        darwin.info(
            f"Query '{cfg.query}' using method '{cfg.method}'"
        )

        try:
            # 1. Retrieve
            raw_results = self._retrieve(cfg)
            if not raw_results:
                return self._empty_context(cfg, elapsed_ms=0)

            # 2. Apply source filters
            raw_results = self._apply_source_filters(raw_results, cfg)
            if not raw_results:
                return self._empty_context(cfg, elapsed_ms=0)

            # 3. Structural penalties
            raw_results, penalized_count = self._apply_structural_penalties(
                raw_results, cfg
            )

            # 4. Sort globally
            raw_results.sort(key=lambda r: r.score, reverse=True)

            # 5. Check fallback
            if self._needs_fallback(raw_results, cfg):
                darwin.warning("Retrieval fallback triggered due to insufficient qualifying results")
                if _is_fallback:
                    darwin.warning(
                        "Already in fallback mode, returning empty context"
                    )
                    return self._empty_context(
                        cfg, elapsed_ms=int((time.perf_counter() - start_time) * 1000)
                    )
                return self._execute_fallback(cfg, start_time)

            # 6. Resolve passages
            passages = self._resolve_passages(raw_results, cfg)
            if not passages and not cfg.return_empty_on_fallback:
                if _is_fallback:
                    return self._empty_context(
                        cfg, elapsed_ms=int((time.perf_counter() - start_time) * 1000)
                    )
                return self._execute_fallback(cfg, start_time)

            # 7. Build context
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return self._build_context(
                cfg,
                passages,
                raw_results,
                penalized_count,
                elapsed_ms,
                fallback_used=False,
            )

        except Exception as e:
            darwin.error(f"Retrieval failed: {e}")
            darwin.error(traceback.format_exc())
            return self._empty_context(
                cfg, elapsed_ms=int((time.perf_counter() - start_time) * 1000)
            )

    # ================================================================
    # Step 1: Retrieval (uses unified args with per_doc_top_k)
    # ================================================================

    def _retrieve(self, cfg: OrchestratorConfig) -> List[RetrievalResult]:
        pool_size = cfg.top_k * 4  # over‑fetch for processing
        args = RetrievalArgs(
            query=cfg.query,
            method=RetrievalMethod(cfg.method),
            files=cfg.files or [],
            top_k=pool_size,
            pool_size=pool_size,
            per_doc_top_k=cfg.max_per_source * 3
            if cfg.max_per_source
            else None,  # over-fetch for penalty headroom
            min_score=0.0,
            rerank=cfg.rerank,
            rerank_weight=cfg.rerank_weight,
            rerank_top_k=cfg.rerank_top_k,
        )
        return handle_retrieval(args, self.store)

    # ================================================================
    # Step 2: Source Filters
    # ================================================================

    def _apply_source_filters(
        self, results: List[RetrievalResult], cfg: OrchestratorConfig
    ) -> List[RetrievalResult]:
        if not results:
            return results
        filtered = []
        for r in results:
            if cfg.exclude_sources and r.source in cfg.exclude_sources:
                continue
            if cfg.prefer_sources and r.source not in cfg.prefer_sources:
                continue
            filtered.append(r)
        return filtered

    # ================================================================
    # Step 3: Structural Penalties
    # ================================================================

    def _apply_structural_penalties(
        self,
        results: List[RetrievalResult],
        cfg: OrchestratorConfig,
    ) -> Tuple[List[RetrievalResult], int]:
        if not results or not cfg.filter_structural:
            return results, 0

        penalized = []
        penalty_count = 0
        by_source = defaultdict(list)
        for r in results:
            by_source[r.source].append(r)

        for source, items in by_source.items():
            indices = [r.chunk_index for r in items if r.chunk_index is not None]
            if not indices:
                # If no valid indices, we cannot apply penalties; keep scores as is.
                penalized.extend(items)
                continue
            meta_map = self._fetch_chunk_metadata(source, indices)

            for r in items:
                # For items with None chunk_index, we cannot get structural type; default to body.
                if r.chunk_index is None:
                    stype = "body"
                else:
                    meta = meta_map.get(r.chunk_index, {})
                    stype = meta.get("structural_type", "body")
                factor = get_penalty_factor(stype)
                if stype == "body":
                    factor = 1.0
                else:
                    factor = cfg.structural_penalty_factor

                new_score = r.score * factor
                was_penalized = factor < 1.0
                r.score = new_score
                r.audit = SearchAudit(
                    raw_score=r.score / factor if factor else r.score,
                    final_score=new_score,
                    structural_type=stype,
                    penalty_factor=factor,
                    was_penalized=was_penalized,
                    penalty_reason=f"structural_type={stype}"
                    if was_penalized
                    else None,
                )
                if was_penalized:
                    penalty_count += 1
                penalized.append(r)

        penalized.sort(key=lambda x: x.score, reverse=True)
        return penalized, penalty_count

    # ================================================================
    # Step 4: Fallback Detection & Execution
    # ================================================================

    def _needs_fallback(
        self, results: List[RetrievalResult], cfg: OrchestratorConfig
    ) -> bool:
        if not results:
            return True
        top = results[: cfg.top_k]
        if not top:
            return True
        qualifying = [r for r in top if r.score >= cfg.min_passage_score]
        if len(qualifying) >= cfg.fallback_min_passages:
            return False
        return True

    def _execute_fallback(
        self, cfg: OrchestratorConfig, start_time: float
    ) -> RetrievalContext:
        darwin.warning("Executing fallback retrieval with relaxed thresholds")
        # Fallback config: relaxed thresholds, no structural penalties, no rerank
        # Inherit exclude/prefer sources from original config.
        fallback_cfg = OrchestratorConfig(
            query=cfg.query,
            method=cfg.method,
            files=cfg.files,
            top_k=cfg.fallback_top_k,
            max_chars=cfg.fallback_max_chars,
            min_passage_score=cfg.fallback_min_score,
            max_per_source=cfg.max_per_source,
            merge_strategy=MergeStrategy.ADJACENT,
            filter_structural=False,  # no penalties in fallback
            rerank=False,  # faster
            return_empty_on_fallback=True,
            # Inherit source filters if they were set
            exclude_sources=cfg.exclude_sources,
            prefer_sources=cfg.prefer_sources,
        )
        # Call with _is_fallback=True to prevent recursion
        context = self._retrieve_and_resolve_internal(fallback_cfg, _is_fallback=True)
        if context is None:
            context = self._empty_context(
                cfg, elapsed_ms=int((time.perf_counter() - start_time) * 1000)
            )
        context.fallback_used = True
        context.fallback_reason = "Insufficient primary results"
        return context

    # ================================================================
    # Step 5: Passage Resolution (with merging & gap filling)
    # ================================================================

    def _resolve_passages(
        self,
        results: List[RetrievalResult],
        cfg: OrchestratorConfig,
    ) -> List[ResolvedPassage]:
        if not results:
            return []

        passages = []
        total_chars = 0
        source_counts = defaultdict(int)
        processed: Set[Tuple[str, Optional[int]]] = set()
        lookup = {(r.source, r.chunk_index): r for r in results}  # includes None

        for r in results:
            if len(passages) >= cfg.max_passages or total_chars >= cfg.max_chars:
                break
            if r.score < cfg.min_passage_score:
                continue
            if cfg.max_per_source > 0 and source_counts[r.source] >= cfg.max_per_source:
                continue
            if (r.source, r.chunk_index) in processed:
                continue

            # If chunk_index is None, we cannot merge; treat as single passage.
            if r.chunk_index is None:
                passage = self._build_single_passage(r, cfg)
                if passage:
                    processed.add((r.source, None))
                    passages.append(passage)
                    total_chars += len(passage.text)
                    source_counts[r.source] += 1
                continue

            if cfg.merge_strategy != MergeStrategy.NONE:
                adjacent = self._get_adjacent(r, cfg, processed, lookup)
                if adjacent:
                    passage = self._build_merged_passage([r] + adjacent, cfg)
                    if passage:
                        for idx in passage.metadata.chunk_indices:
                            processed.add((r.source, idx))
                        passages.append(passage)
                        total_chars += len(passage.text)
                        source_counts[r.source] += 1
                        continue

            passage = self._build_single_passage(r, cfg)
            if passage:
                processed.add((r.source, r.chunk_index))
                passages.append(passage)
                total_chars += len(passage.text)
                source_counts[r.source] += 1

        return passages

    def _get_adjacent(
        self,
        result: RetrievalResult,
        cfg: OrchestratorConfig,
        processed: Set[Tuple[str, Optional[int]]],
        lookup: Dict[Tuple[str, Optional[int]], RetrievalResult],
    ) -> List[RetrievalResult]:
        adj = []
        idx = result.chunk_index
        if idx is None:
            return adj
        for delta in (-1, 1):
            neighbor = lookup.get((result.source, idx + delta))
            if neighbor and (result.source, idx + delta) not in processed:
                adj.append(neighbor)
        return adj

    # ================================================================
    # Passage Builders
    # ================================================================

    def _build_single_passage(
        self, result: RetrievalResult, cfg: OrchestratorConfig
    ) -> Optional[ResolvedPassage]:
        text = result.text or result.clean or ""
        if not text:
            return None
        stype = self._get_structural_type(result.source, result.chunk_index)

        metadata = PassageMetadata(
            source=result.source,
            page_start=result.page or 0,
            page_end=result.page_end or result.page or 0,
            chunk_indices=[result.chunk_index]
            if result.chunk_index is not None
            else [],
            chunk_count=1,
            is_merged=False,
            gap_filled=False,
            structural_type=stype,
            penalty_factor=1.0,
            was_penalized=False,
            avg_score=result.score,
            semantic_score=result.semantic_score,
            keyword_score=result.keyword_score,
            rrf_score=result.rrf_score,
            rerank_score=result.rerank_score,
            matched_terms=list(result.matched_terms) if result.matched_terms else [],
        )
        audit = result.audit.to_dict() if (cfg.include_audit and result.audit) else None
        return ResolvedPassage(text=text, metadata=metadata, audit=audit)

    def _build_merged_passage(
        self,
        results: List[RetrievalResult],
        cfg: OrchestratorConfig,
    ) -> Optional[ResolvedPassage]:
        if not results:
            return None
        sorted_results = sorted(results, key=lambda r: r.chunk_index or 0)
        source = sorted_results[0].source
        indices = [r.chunk_index for r in sorted_results if r.chunk_index is not None]

        all_indices = []
        gap_filled = False
        if cfg.fill_gaps and len(indices) > 1:
            for i in range(len(indices) - 1):
                all_indices.append(indices[i])
                for gap in range(indices[i] + 1, indices[i + 1]):
                    # Only add gap if we have metadata for it; otherwise skip
                    meta = self._fetch_chunk_metadata(source, [gap]).get(gap, {})
                    if meta.get("text"):
                        all_indices.append(gap)
                        gap_filled = True
                    else:
                        darwin.warning(
                            f"Gap chunk {gap} missing for source '{source}', skipping"
                        )
            all_indices.append(indices[-1])
        else:
            all_indices = indices

        texts = []
        pages = []
        for idx in all_indices:
            result = next((r for r in sorted_results if r.chunk_index == idx), None)
            if result and result.text:
                texts.append(result.text)
                pages.append(result.page or 0)
            else:
                chunk = self._fetch_chunk_metadata(source, [idx]).get(idx, {})
                if chunk.get("text"):
                    texts.append(chunk["text"])
                    pages.append(chunk.get("page", 0))
                else:
                    # If still missing, skip this chunk (should not happen after gap check)
                    darwin.warning(
                        f"No text for chunk {idx} in source '{source}', skipping"
                    )
                    continue

        if not texts:
            return None

        stitched = self._stitch_texts(
            texts, pages, deduplicate=True
        )  # keep dedup for now

        anchor = sorted_results[0]
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0
        stypes = [self._get_structural_type(source, idx) for idx in all_indices]
        dominant = max(set(stypes), key=stypes.count) if stypes else "body"

        metadata = PassageMetadata(
            source=source,
            page_start=min(pages) if pages else 0,
            page_end=max(pages) if pages else 0,
            chunk_indices=all_indices,
            chunk_count=len(all_indices),
            is_merged=True,
            gap_filled=gap_filled,
            structural_type=dominant,
            penalty_factor=1.0,
            was_penalized=False,
            avg_score=avg_score,
            semantic_score=anchor.semantic_score,
            keyword_score=anchor.keyword_score,
            rrf_score=anchor.rrf_score,
            rerank_score=anchor.rerank_score,
            matched_terms=list(anchor.matched_terms) if anchor.matched_terms else [],
        )
        return ResolvedPassage(text=stitched, metadata=metadata, audit=None)

    # ================================================================
    # Helpers: Structural type & stitching
    # ================================================================

    def _get_structural_type(self, source: str, chunk_index: Optional[int]) -> str:
        if chunk_index is None:
            return "body"
        meta = self._fetch_chunk_metadata(source, [chunk_index])
        return meta.get(chunk_index, {}).get("structural_type", "body")

    def _fetch_chunk_metadata(
        self, source: str, indices: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        if not indices:
            return {}
        if source not in self._chunk_meta_cache:
            try:
                rows = self.store.query_records(
                    source, columns=["index", "structural_type", "text", "page"]
                )
                self._chunk_meta_cache[source] = {row["index"]: row for row in rows}
            except Exception:
                self._chunk_meta_cache[source] = {}
        cache = self._chunk_meta_cache[source]
        return {idx: cache.get(idx, {}) for idx in indices}

    def _stitch_texts(
        self, texts: List[str], pages: List[int], deduplicate: bool = True
    ) -> str:
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0]
        result = []
        seen = set()
        for text, page in zip(texts, pages):
            if page != pages[0]:
                result.append(f"\n[Page {page}]\n")
            for sent in split_sentences(text):
                norm = " ".join(sent.split())
                if deduplicate and norm in seen:
                    continue
                if deduplicate:
                    seen.add(norm)
                result.append(sent)
        return " ".join(result)

    # ================================================================
    # Context Builder
    # ================================================================

    def _build_context(
        self,
        cfg: OrchestratorConfig,
        passages: List[ResolvedPassage],
        raw_results: List[RetrievalResult],
        penalized_count: int,
        elapsed_ms: int,
        fallback_used: bool = False,
    ) -> RetrievalContext:
        total_chars = sum(len(p.text) for p in passages)
        scores = [p.metadata.avg_score for p in passages]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        sources = list(set(p.metadata.source for p in passages))
        source_diversity = len(sources) / max(1, min(len(passages), 5))

        audit_summary = {
            "total_results": len(raw_results),
            "penalized_count": penalized_count,
            "candidates_found": len(raw_results),
            "passages_resolved": len(passages),
            "structural_types": {},
        }
        for p in passages:
            t = p.metadata.structural_type or "unknown"
            audit_summary["structural_types"][t] = (
                audit_summary["structural_types"].get(t, 0) + 1
            )

        return RetrievalContext(
            query=cfg.query,
            method=cfg.method,
            passages=passages,
            total_chars=total_chars,
            total_passages=len(passages),
            elapsed_ms=elapsed_ms,
            candidates_found=len(raw_results),
            penalized_count=penalized_count,
            max_score=max_score,
            avg_score=avg_score,
            sources_used=sources,
            source_diversity=source_diversity,
            fallback_used=fallback_used,
            raw_results=raw_results if cfg.include_raw_results else [],
            audit_summary=audit_summary,
        )

    def _empty_context(
        self, cfg: OrchestratorConfig, elapsed_ms: int
    ) -> RetrievalContext:
        return RetrievalContext(
            query=cfg.query,
            method=cfg.method,
            passages=[],
            total_chars=0,
            total_passages=0,
            elapsed_ms=elapsed_ms,
            candidates_found=0,
            penalized_count=0,
            max_score=0.0,
            avg_score=0.0,
            sources_used=[],
            source_diversity=0.0,
            fallback_used=True,
            fallback_reason="No results",
            raw_results=[],
            audit_summary={"error": "empty_context"},
        )
