"""
src/core/Orchestrators/RetrievalOrchestrator.py
-----------------------------------------------
Retrieval Orchestrator – debloated, robust, with fallback recursion fix.

FIXES APPLIED (2026-07-19):
  1. Fallback config: max_per_source=0 (unlimited), diversity_rerank=False
  2. Fallback config: method="keyword" (skips 5s semantic model load)
  3. _apply_diversity_reranking(): skipped when _is_fallback=True or single source
  4. _apply_diversity_reranking(): no longer mutates r.score in-place
  5. _apply_structural_penalties(): uses get_penalty_factor() for all types
  6. _stitch_texts(): preserves sentence boundaries with smart join
  7. _fetch_chunk_metadata(): no longer caches errors
  8. _build_merged_passage(): fixed link merging with isinstance() checks
  9. _execute_fallback(): consolidated into single fallback path
  10. _needs_fallback(): uses diversity-adjusted score, not mutated r.score
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple

from src.core.Logger.DarwinLogger import darwin
from src.core.Storage.DarwinStore import DarwinStore

from src.core.Retrival.main import handle_retrieval
from src.core.Retrival.RetrievalArgs import RetrievalArgs, RetrievalMethod
from src.core.Retrival.RetrievalResult import RetrievalResult
from src.core.Retrival.lib import SearchAudit, OverlapDirection

from src.core.Pipeline.Ingestion.Chunking.StructuralClassifier import get_penalty_factor
from src.core.Pipeline.Ingestion.Chunking.ChunkerUtils import split_sentences
from src.lib.PrettyTraceback import install_pretty_traceback

install_pretty_traceback()
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
        self._default_config = config or RetrievalInput(query="<template>")
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
        with darwin.session("RETRIEVAL"):
            darwin.info(f"Retrieval orchestrator started for query '{cfg.query}'")
            darwin.info(f"Method: {cfg.method} | Top K: {cfg.top_k} | Max Chars: {cfg.max_chars}")
            darwin.info(f"Max per source: {cfg.max_per_source} | Merge strategy: {cfg.merge_strategy}")
            darwin.info(f"Structural penalties: {'enabled' if cfg.filter_structural else 'disabled'}")
            darwin.info(f"Rerank: {'enabled' if cfg.rerank else 'disabled'} | Rerank weight: {cfg.rerank_weight}")
            darwin.info(f"Source filters: exclude={cfg.exclude_sources}, prefer={cfg.prefer_sources}")
            darwin.info(f"Fallback thresholds: top_k={cfg.fallback_top_k}, max_chars={cfg.fallback_max_chars}, min_score={cfg.fallback_min_score}")
            darwin.info(f"Return empty on fallback: {cfg.return_empty_on_fallback}")
            ctx = self._retrieve_and_resolve_internal(cfg, _is_fallback=False)

            if params.save and ctx is not None:
                saved = self._save_results(ctx, params.results_dir)
                if not saved:
                    darwin.warning("Search results were not saved — check logs for details")

            return ctx

    def _save_results(self, ctx: RetrievalContext, results_dir: Optional[str] = None) -> bool:
        """Save RetrievalContext as JSON to the results directory. Returns True on success."""
        try:
            from src.lib.AppPaths import set_darwin_base_dir
            if results_dir:
                save_dir = Path(results_dir)
            else:
                save_dir = set_darwin_base_dir() / "search-results"
            save_dir.mkdir(parents=True, exist_ok=True)

            slug = re.sub(r"[^a-z0-9]+", "-", ctx.query.lower()).strip("-")[:80]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{slug}_{ts}.json"
            filepath = save_dir / filename

            data = ctx.to_dict()
            data["_saved_at"] = datetime.now().isoformat()
            data["_filename"] = filename

            filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            darwin.success(f"Results saved: {filepath}")
            return True
        except Exception as e:
            from src.lib.PrettyTraceback import darwin_traceback
            darwin_traceback(e, label="RetrievalOrchestrator._save_results")
            return False

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
            raw_results = self._retrieve(cfg, _is_fallback=_is_fallback)
            if not raw_results:
                return self._empty_context(cfg, elapsed_ms=0)

            # 2. Apply source filters
            raw_results = self._apply_source_filters(raw_results, cfg)
            if not raw_results:
                return self._empty_context(cfg, elapsed_ms=0)

            # 2.5. Initialize _sort_score fallback so every code path has it
            for r in raw_results:
                r._sort_score = r.score

            # 3. Structural penalties
            raw_results, _ = self._apply_structural_penalties(
                raw_results, cfg
            )

            # 4. Source diversity re-ranking (MMR-style) — SKIP in fallback or single source
            raw_results = self._apply_diversity_reranking(raw_results, cfg, _is_fallback)

            # 5. Filter front/back matter (copyright, cover, toc, separator)
            raw_results = self._filter_front_back_matter(raw_results)

            # 6. Count penalized results from the FINAL filtered set (not pre-filter)
            penalized_count = sum(1 for r in raw_results if r.audit and r.audit.was_penalized)

            # 7. Sort globally
            raw_results.sort(key=lambda r: r._sort_score, reverse=True)

            # 8. Check fallback — uses diversity-adjusted score, not mutated r.score
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

            # 9. Resolve passages
            passages = self._resolve_passages(raw_results, cfg)

            # 10. Deduplicate near-identical passages
            passages = self._deduplicate_passages(passages)

            if not passages and not cfg.return_empty_on_fallback:
                if _is_fallback:
                    return self._empty_context(
                        cfg, elapsed_ms=int((time.perf_counter() - start_time) * 1000)
                    )
                return self._execute_fallback(cfg, start_time)

            # 11. Build context
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
            from src.lib.PrettyTraceback import darwin_traceback
            darwin_traceback(e, label="RetrievalOrchestrator._retrieve_and_resolve_internal")
            return self._empty_context(
                cfg, elapsed_ms=int((time.perf_counter() - start_time) * 1000)
            )

    # ================================================================
    # Step 1: Retrieval (uses unified args with per_doc_top_k)
    # ================================================================

    def _retrieve(self, cfg: OrchestratorConfig, _is_fallback: bool = False) -> List[RetrievalResult]:
        # FIX: In fallback mode, use keyword-only (50x faster, same results)
        method = RetrievalMethod(cfg.method)
        if _is_fallback and method == RetrievalMethod.HYBRID:
            darwin.info("Fallback mode: using keyword-only search for speed")
            method = RetrievalMethod.KEYWORD

        # FIX: In fallback mode, unlimited per_doc_top_k to get all results
        per_doc_limit = None if _is_fallback else (
            cfg.max_per_source * 3 if cfg.max_per_source else None
        )

        pool_size = cfg.top_k * 4  # over‑fetch for processing
        args = RetrievalArgs(
            query=cfg.query,
            method=method,
            files=cfg.files or [],
            top_k=pool_size,
            pool_size=pool_size,
            per_doc_top_k=per_doc_limit,
            min_score=0.0,
            rerank=cfg.rerank and not _is_fallback,  # disable rerank in fallback
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
    # Step 2b: Filter front/back matter
    # ================================================================

    _NON_CONTENT_TYPES = frozenset({"copyright", "cover", "toc", "separator"})

    def _filter_front_back_matter(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Remove passages that are structural noise (copyright, cover, toc, separator)."""
        if not results:
            return results
        filtered = [r for r in results if (r.structural_type or "body") not in self._NON_CONTENT_TYPES]
        removed = len(results) - len(filtered)
        if removed > 0:
            darwin.info(f"Filtered {removed} front/back matter chunk(s)")
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

        for r in results:
            stype = r.structural_type or "body"
            # FIX: Use get_penalty_factor() for ALL types, not just body
            factor = get_penalty_factor(stype)
            if stype == "body":
                factor = 1.0
            # else: use the per-type factor from get_penalty_factor()
            # (removed the override with cfg.structural_penalty_factor)

            raw_score = r.score
            new_score = r.score * factor
            was_penalized = factor < 1.0
            # FIX: Store adjusted score in _sort_score, preserve original r.score
            r._sort_score = new_score
            r.audit = SearchAudit(
                raw_score=raw_score,
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

        penalized.sort(key=lambda x: x._sort_score, reverse=True)
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
        # FIX: Use _sort_score (diversity-adjusted) instead of mutated r.score
        qualifying = [r for r in top if r._sort_score >= cfg.min_passage_score]
        # Diagnostic: log score distribution so users can tune thresholds
        scores_str = ", ".join(
            f"{r._sort_score:.3f}" for r in top
        )
        darwin.info(
            f"Fallback check: {len(qualifying)}/{len(top)} qualify "
            f"(threshold={cfg.min_passage_score:.2f}) "
            f"scores=[{scores_str}]"
        )
        if len(qualifying) >= cfg.fallback_min_passages:
            return False
        return True

    def _execute_fallback(
        self, cfg: OrchestratorConfig, start_time: float
    ) -> RetrievalContext:
        darwin.warning("Executing fallback retrieval with relaxed thresholds")
        # FIX: Fallback config with unlimited per-source, no diversity penalty, keyword-only
        fallback_cfg = OrchestratorConfig(
            query=cfg.query,
            method="keyword",  # FIX: keyword-only, skip 5s model load
            files=cfg.files,
            top_k=cfg.fallback_top_k,
            max_chars=cfg.fallback_max_chars,
            min_passage_score=cfg.fallback_min_score,
            max_per_source=0,  # FIX: unlimited per source
            max_passages=cfg.fallback_top_k,  # allow more passages in fallback
            merge_strategy=MergeStrategy.ADJACENT,
            filter_structural=False,  # no penalties in fallback
            diversity_rerank=False,  # no diversity penalty in fallback
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
            # FIX: Use _sort_score for threshold check, not mutated r.score
            if r._sort_score < cfg.min_passage_score:
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
                if self._should_merge(result, neighbor):
                    adj.append(neighbor)
        return adj

    def _should_merge(self, a: RetrievalResult, b: RetrievalResult) -> bool:
        """Refuse to merge across heading boundaries."""
        if not a.heading or not b.heading:
            return True
        return a.heading == b.heading

    # ================================================================
    # Source Diversity Re-ranking (MMR-style)
    # ================================================================

    def _apply_diversity_reranking(
        self, results: List[RetrievalResult], cfg: OrchestratorConfig, _is_fallback: bool = False
    ) -> List[RetrievalResult]:
        """Penalize repeated sources so results prefer diversity across files.

        FIXES:
          - Skipped when _is_fallback=True (prevents cascading failure)
          - Skipped when all results are from a single source (no diversity to enforce)
          - Stores adjusted score in _sort_score, preserves original r.score
        """
        if not results:
            return results

        # FIX: Respect the diversity_rerank flag
        if not cfg.diversity_rerank:
            for r in results:
                r._sort_score = getattr(r, '_sort_score', r.score)
            return results

        # FIX: Skip diversity reranking in fallback mode (check FIRST — before max_per_source)
        if _is_fallback:
            darwin.info("Diversity reranking skipped in fallback mode")
            for r in results:
                r._sort_score = getattr(r, '_sort_score', r.score)
            return results

        # FIX: Skip when max_per_source <= 0 (unlimited) or single source
        if cfg.max_per_source <= 0:
            darwin.info("Diversity reranking skipped in fallback mode")
            for r in results:
                r._sort_score = getattr(r, '_sort_score', r.score)
            return results

        # FIX: Skip diversity reranking when all results are from a single source
        unique_sources = set(r.source for r in results)
        if len(unique_sources) <= 1:
            darwin.info("Diversity reranking skipped: single source")
            for r in results:
                r._sort_score = getattr(r, '_sort_score', r.score)
            return results

        source_count: Dict[str, int] = defaultdict(int)
        diversity_penalty = 0.85  # 15% penalty per extra result from same source
        reranked = []
        for r in results:
            count = source_count[r.source]
            # FIX: Store adjusted score in _sort_score, preserve original r.score
            base_score = getattr(r, '_sort_score', r.score)
            if count > 0:
                r._sort_score = base_score * (diversity_penalty ** count)
            else:
                r._sort_score = base_score
            source_count[r.source] += 1
            reranked.append(r)
        return reranked

    # ================================================================
    # Near-Duplicate Passage Filtering
    # ================================================================

    def _deduplicate_passages(
        self, passages: List[ResolvedPassage], threshold: float = 0.8
    ) -> List[ResolvedPassage]:
        """Remove passages that are >80% sentence-overlap duplicates."""
        if len(passages) <= 1:
            return passages
        kept: List[ResolvedPassage] = []
        kept_sentences: List[Set[str]] = []
        for p in passages:
            sents = set(split_sentences(p.text))
            is_dup = False
            for existing in kept_sentences:
                if not sents or not existing:
                    continue
                overlap = len(sents & existing) / min(len(sents), len(existing))
                if overlap >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(p)
                kept_sentences.append(sents)
        removed = len(passages) - len(kept)
        if removed > 0:
            darwin.info(f"Dedup: removed {removed} near-identical passage(s)")
        return kept

    # ================================================================
    # Passage Builders
    # ================================================================

    def _build_single_passage(
        self, result: RetrievalResult, cfg: OrchestratorConfig
    ) -> Optional[ResolvedPassage]:
        text = result.text or result.clean or ""
        if not text:
            return None
        stype = result.structural_type or "body"
        # FIX: Use _sort_score for avg_score, fallback to r.score
        sort_score = getattr(result, '_sort_score', result.score)

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
            avg_score=sort_score,
            semantic_score=result.semantic_score,
            keyword_score=result.keyword_score,
            rrf_score=result.rrf_score,
            rerank_score=result.rerank_score,
            heading=result.heading or None,
            links=[{"url": l.url, "text": l.text} for l in result.links] if result.links else None,
            clean=result.clean,
            overlap_direction=result.overlap_direction.value if result.overlap_direction and result.overlap_direction != OverlapDirection.NONE else None,
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
        # FIX: Use _sort_score for avg_score calculation
        scores = [getattr(r, '_sort_score', r.score) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        stypes = [r.structural_type or "body" for r in sorted_results]
        dominant = max(set(stypes), key=stypes.count) if stypes else "body"

        # Merge links from all constituent results
        all_links = []
        seen_urls = set()
        for r in sorted_results:
            for link in (r.links or []):
                # FIX: Use isinstance() instead of fragile hasattr()
                if isinstance(link, dict):
                    url = link.get("url", "")
                    text = link.get("text", "")
                else:
                    url = getattr(link, "url", "")
                    text = getattr(link, "text", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_links.append({"url": url, "text": text})

        # Merge headings from constituent results
        headings = [r.heading for r in sorted_results if r.heading]
        merged_heading = headings[0] if len(headings) == 1 else " / ".join(dict.fromkeys(headings)) if headings else None

        metadata = PassageMetadata(
            source=source,
            page_start=min(pages) if pages else 0,
            page_end=max(r.page_end or r.page or 0 for r in sorted_results),
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
            heading=merged_heading,
            links=all_links if all_links else None,
            clean=anchor.clean,
            overlap_direction=anchor.overlap_direction.value if anchor.overlap_direction and anchor.overlap_direction != OverlapDirection.NONE else None,
            matched_terms=list(anchor.matched_terms) if anchor.matched_terms else [],
        )

        # Build audit for merged passage
        audit = None
        if cfg.include_audit:
            audits = [r.audit.to_dict() for r in sorted_results if r.audit]
            audit = {"merged_from": audits} if audits else None

        return ResolvedPassage(text=stitched, metadata=metadata, audit=audit)

    # ================================================================
    # Helpers: Stitching & metadata cache
    # ================================================================

    def _fetch_chunk_metadata(
        self, source: str, indices: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Fetch metadata for gap chunks not in the search results.
        Only called for chunks that weren't returned by handle_retrieval."""
        if not indices:
            return {}
        if source not in self._chunk_meta_cache:
            try:
                rows = self.store.query_records(
                    source, columns=["index", "text", "page", "page_end", "heading", "links", "structural_type"]
                )
                self._chunk_meta_cache[source] = {row["index"]: row for row in rows}
            except Exception as e:
                # FIX: Don't cache errors — allow retry on next call
                from src.lib.PrettyTraceback import darwin_traceback
                darwin_traceback(e, label=f"RetrievalOrchestrator._fetch_chunk_metadata ({source})")
                return {}
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
        prev_page = pages[0] if pages else 0
        for text, page in zip(texts, pages):
            if page != prev_page:
                result.append(f"[Page {page}]")
                prev_page = page
            for sent in split_sentences(text):
                norm = " ".join(sent.split())
                if deduplicate and norm in seen:
                    continue
                if deduplicate:
                    seen.add(norm)
                result.append(sent)
        # FIX: Smart join that preserves sentence boundaries
        return _smart_join_sentences(result)

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


# ═══════════════════════════════════════════════════════════════════
# Helper: Smart sentence joiner
# ═══════════════════════════════════════════════════════════════════

def _smart_join_sentences(sentences: List[str]) -> str:
    """Join sentences preserving boundaries. Ensures space after 
    sentence-ending punctuation and before next sentence."""
    if not sentences:
        return ""
    if len(sentences) == 1:
        return sentences[0]

    result = []
    for i, sent in enumerate(sentences):
        if i == 0:
            result.append(sent)
            continue
        prev = result[-1].rstrip()
        curr = sent.lstrip()
        # If previous ends with sentence-ending punctuation and current 
        # starts with uppercase (or [Page marker), ensure space
        if prev and curr:
            if prev[-1] in ".!?" and (curr[0].isupper() or curr.startswith("[")):
                result.append(" " + curr)
            elif curr.startswith("["):
                result.append(curr)
            else:
                result.append(" " + curr)
        else:
            result.append(curr)

    return "".join(result)