
from __future__ import annotations

import time
import traceback
from collections import defaultdict
from typing import List, Optional, Dict, Any, Set, Tuple

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Storage.DarwinStore import DarwinStore

from src.features.Retrival.main import handle_retrieval
from src.features.Retrival.RetrivalUtils import RetrievalArgs, RetrievalMethod
from src.features.Retrival.RetrievalResult import (
    RetrievalResult,
    SearchAudit,
)

from src.pipleline.Ingestion.Chunking.StructuralClassifier import get_penalty_factor
from src.pipleline.Ingestion.Chunking.ChunkerUtils import split_sentences

from ...features.Retrival.OrchestratorTypes import (
    OrchestratorConfig,
    MergeStrategy,
    FallbackStrategy,
    ResolvedPassage,
    PassageMetadata,
    RetrievalContext,
)


class RetrievalOrchestrator:
    """
    The main orchestrator for retrieval operations.
    
    Features:
    - Global scoring (not per-source)
    - Fair token budgeting
    - Source diversity & filtering
    - Smart fallback strategies
    - Rich metadata for agents
    """
    
    def __init__(
        self,
        store: DarwinStore,
        config: Optional[OrchestratorConfig] = None,
    ):
        self.store = store
        self.config = config or OrchestratorConfig()
        # Cache: source -> {chunk_index: row_dict}
        self._chunk_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
        # Cache: source -> all rows (to avoid multiple full fetches)
        self._source_fetched: Set[str] = set()
    
    # ═══════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════
    
    def retrieve_and_resolve(
        self,
        config: Optional[OrchestratorConfig] = None,
    ) -> Optional[RetrievalContext]:
        """
        Main entry point: retrieve, resolve, and return context.
        Returns None on fatal error (with error logged).
        """
        start_time = time.perf_counter()
        cfg = config or self.config

        try:
            darwin.log(f"[Orchestrator] Starting: '{cfg.query}' [{cfg.method}]", LogType.INFO)
            darwin.log(f"[Orchestrator] Config: top_k={cfg.top_k}, max_chars={cfg.max_chars}, "
                       f"max_per_source={cfg.max_per_source}", LogType.INFO)
            
            # ── Step 1: Retrieve ──────────────────────────────────────────
            raw_results = self._retrieve(cfg)
            candidates_found = len(raw_results)
            
            if not raw_results:
                darwin.log("[Orchestrator] No results from retrieval", LogType.WARNING)
                return self._empty_context(cfg, elapsed_ms=0)
            
            darwin.log(f"[Orchestrator] Retrieved {candidates_found} candidates", LogType.INFO)
            
            # ── Step 1b: Apply source filters ────────────────────────────
            raw_results = self._apply_source_filters(raw_results, cfg)
            if not raw_results:
                darwin.log("[Orchestrator] All results filtered out by source preferences", LogType.WARNING)
                return self._empty_context(cfg, elapsed_ms=0)
            
            # ── Step 2: Apply structural penalties ──────────────────────
            if cfg.filter_structural:
                raw_results, penalized_count = self._apply_structural_penalties(
                    raw_results, cfg
                )
            else:
                penalized_count = 0
            
            # ── Step 3: Sort globally by score ──────────────────────────
            raw_results.sort(key=lambda r: r.score, reverse=True)
            
            # ── Step 4: Check for fallback ──────────────────────────────
            if self._needs_fallback(raw_results, cfg):
                darwin.log("[Orchestrator] Triggering fallback strategy", LogType.WARNING)
                return self._execute_fallback(cfg, start_time)
            
            # ── Step 5: Resolve passages ────────────────────────────────
            passages = self._resolve_passages(raw_results, cfg)
            
            if not passages:
                darwin.log("[Orchestrator] No passages resolved", LogType.WARNING)
                if cfg.return_empty_on_fallback:
                    return self._empty_context(cfg, elapsed_ms=0)
                return self._execute_fallback(cfg, start_time)
            
            # ── Step 6: Build context ────────────────────────────────────
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            
            context = self._build_context(
                query=cfg.query,
                method=cfg.method,
                passages=passages,
                raw_results=raw_results if cfg.include_raw_results else [],
                candidates_found=candidates_found,
                penalized_count=penalized_count,
                elapsed_ms=elapsed_ms,
                fallback_used=False,
            )
            
            darwin.log(f"[Orchestrator] Done: {len(passages)} passages, {context.total_chars} chars, {elapsed_ms}ms", LogType.SUCCESS)
            return context

        except Exception as e:
            # Catch any exception, log full traceback and return None
            darwin.log(f"[Orchestrator] Fatal error: {e}", LogType.ERROR)
            darwin.log(traceback.format_exc(), LogType.ERROR)
            # Return an empty context with error info if possible, else None
            try:
                ctx = self._empty_context(cfg, elapsed_ms=int((time.perf_counter() - start_time)*1000))
                ctx.fallback_reason = f"Fatal error: {e}"
                return ctx
            except:
                return None
    
    # ═══════════════════════════════════════════════════════════════
    # Step 1: Retrieval
    # ═══════════════════════════════════════════════════════════════
    
    def _retrieve(self, cfg: OrchestratorConfig) -> List[RetrievalResult]:
        """Execute the retrieval step."""
        pool_size = cfg.top_k * 4
        
        args = RetrievalArgs(
            query=cfg.query,
            method=RetrievalMethod(cfg.method),
            files=cfg.files or [],
            top_k=pool_size,
            pool_size=pool_size,
            min_score=0.0,
            inspect=False,
            json_output=False,
        )
        
        return handle_retrieval(args, self.store)
    
    # ═══════════════════════════════════════════════════════════════
    # Source Filtering
    # ═══════════════════════════════════════════════════════════════
    
    def _apply_source_filters(
        self,
        results: List[RetrievalResult],
        cfg: OrchestratorConfig,
    ) -> List[RetrievalResult]:
        """Apply source preference and exclusion filters."""
        if not results:
            return results
        
        filtered = []
        for r in results:
            # Exclusion filter
            if cfg.exclude_sources and r.source in cfg.exclude_sources:
                continue
            # Inclusion filter (if specified)
            if cfg.prefer_sources and r.source not in cfg.prefer_sources:
                continue
            filtered.append(r)
        
        return filtered
    
    # ═══════════════════════════════════════════════════════════════
    # Step 2: Structural Penalties
    # ═══════════════════════════════════════════════════════════════
    
    def _apply_structural_penalties(
        self,
        results: List[RetrievalResult],
        cfg: OrchestratorConfig,
    ) -> Tuple[List[RetrievalResult], int]:
        """Apply structural penalties to results."""
        if not results:
            return results, 0
        
        penalized = []
        penalty_count = 0
        
        by_source = defaultdict(list)
        for r in results:
            by_source[r.source].append(r)
        
        for source, source_results in by_source.items():
            indices = [r.chunk_index for r in source_results if r.chunk_index is not None]
            if not indices:
                continue
            
            chunk_map = self._fetch_chunks(source, indices, columns=["structural_type"])
            
            for r in source_results:
                chunk = chunk_map.get(r.chunk_index, {})
                stype = chunk.get("structural_type", "body")
                
                # Base penalty from classifier
                factor = get_penalty_factor(stype)
                
                # Apply additional configurable penalty ONLY if it's not "body"
                if stype != "body" and cfg.structural_penalty_factor < 1.0:
                    factor = min(factor, cfg.structural_penalty_factor)
                
                # If it's body, ensure factor is 1.0 (no penalty)
                if stype == "body":
                    factor = 1.0
                
                new_score = r.score * factor
                was_penalized = factor < 1.0
                
                audit = SearchAudit(
                    raw_score=r.score,
                    final_score=new_score,
                    structural_type=stype,
                    penalty_factor=factor,
                    was_penalized=was_penalized,
                    penalty_reason=f"structural_type={stype}" if was_penalized else None,
                )
                r.audit = audit
                r.score = new_score
                
                if was_penalized:
                    penalty_count += 1
                
                penalized.append(r)
        
        penalized.sort(key=lambda x: x.score, reverse=True)
        return penalized, penalty_count
    
    # ═══════════════════════════════════════════════════════════════
    # Step 3: Fallback Detection
    # ═══════════════════════════════════════════════════════════════
    
    def _needs_fallback(self, results: List[RetrievalResult], cfg: OrchestratorConfig) -> bool:
        """Check if fallback is needed based on config."""
        if not results:
            return True
        
        # Check minimum score
        has_high_score = any(r.score >= cfg.min_passage_score for r in results[:cfg.top_k])
        if not has_high_score:
            return True
        
        # Check source diversity (if required)
        if cfg.require_diverse_sources or cfg.prefer_diverse_sources:
            sources = set(r.source for r in results[:cfg.top_k])
            if len(sources) < 2:
                if cfg.require_diverse_sources:
                    return True
                # If only prefer, log a warning but don't fallback
                darwin.log("[Orchestrator] Low source diversity (prefer_diverse_sources=True, but not required)", LogType.INFO)
        
        # Check if we have enough passages
        top_candidates = results[:cfg.top_k]
        if len(top_candidates) < cfg.fallback_min_passages:
            return True
        
        return False
    
    # ═══════════════════════════════════════════════════════════════
    # Step 4: Fallback Execution
    # ═══════════════════════════════════════════════════════════════
    
    def _execute_fallback(
        self,
        cfg: OrchestratorConfig,
        start_time: float,
    ) -> RetrievalContext:
        """Execute fallback strategy."""
        darwin.log("[Orchestrator] Executing fallback", LogType.WARNING)
        
        fallback_cfg = OrchestratorConfig(
            query=cfg.query,
            method=cfg.method,
            files=cfg.files,
            top_k=cfg.fallback_top_k,
            max_chars=cfg.fallback_max_chars,
            max_passages=cfg.max_passages * 2,
            min_passage_score=cfg.fallback_min_score,
            merge_strategy=MergeStrategy.ADJACENT,
            max_merge_gap=cfg.max_merge_gap,
            fill_gaps=cfg.fill_gaps,
            max_per_source=cfg.max_per_source,
            prefer_diverse_sources=cfg.prefer_diverse_sources,
            require_diverse_sources=False,  # Don't require diversity in fallback
            prefer_sources=cfg.prefer_sources,
            exclude_sources=cfg.exclude_sources,
            filter_structural=False,        # Disable penalties in fallback
            include_metadata=cfg.include_metadata,
            include_audit=cfg.include_audit,
            include_raw_results=cfg.include_raw_results,
            return_empty_on_fallback=True,
        )
        
        context = self.retrieve_and_resolve(fallback_cfg)
        context.fallback_used = True
        context.fallback_reason = (
            f"Initial search with top_k={cfg.top_k}, min_score={cfg.min_passage_score} "
            f"returned insufficient results"
        )
        return context
    
    # ═══════════════════════════════════════════════════════════════
    # Step 5: Passage Resolution (Global, Fair Budgeting)
    # ═══════════════════════════════════════════════════════════════
    
    def _resolve_passages(
        self,
        results: List[RetrievalResult],
        cfg: OrchestratorConfig,
    ) -> List[ResolvedPassage]:
        """
        Resolve passages with global budgeting and source diversity.
        """
        if not results:
            return []
        
        # Build lookup map for adjacent merging
        result_lookup = {}
        for r in results:
            if r.chunk_index is not None:
                result_lookup[(r.source, r.chunk_index)] = r
        
        passages: List[ResolvedPassage] = []
        total_chars = 0
        source_counts: Dict[str, int] = defaultdict(int)
        processed_indices: Set[int] = set()
        
        # Process results globally (already sorted by score)
        for r in results:
            # Stop conditions
            if len(passages) >= cfg.max_passages:
                break
            if total_chars >= cfg.max_chars:
                break
            if r.score < cfg.min_passage_score:
                continue
            
            # Enforce source diversity
            if cfg.max_per_source > 0:
                source_counts[r.source] = source_counts.get(r.source, 0) + 1
                if source_counts[r.source] > cfg.max_per_source:
                    continue
            
            if r.chunk_index in processed_indices:
                continue
            
            passage = self._resolve_single_passage(r, cfg, processed_indices, result_lookup)
            if not passage:
                continue
            
            passage_chars = len(passage.text)
            if total_chars + passage_chars > cfg.max_chars:
                remaining = cfg.max_chars - total_chars
                if remaining > 200 and passage.metadata.avg_score > 0.8:
                    passage.text = self._smart_truncate(passage.text, remaining)
                    passage_chars = len(passage.text)
                else:
                    continue
            
            total_chars += passage_chars
            passages.append(passage)
            
            if passage.metadata.is_merged:
                processed_indices.update(passage.metadata.chunk_indices)
            else:
                processed_indices.add(r.chunk_index)
        
        return passages
    
    # ═══════════════════════════════════════════════════════════════
    # Passage Resolution Helpers
    # ═══════════════════════════════════════════════════════════════
    
    def _resolve_single_passage(
        self,
        result: RetrievalResult,
        cfg: OrchestratorConfig,
        processed_indices: Set[int],
        result_lookup: Dict[Tuple[str, int], RetrievalResult],
    ) -> Optional[ResolvedPassage]:
        if cfg.merge_strategy != MergeStrategy.NONE:
            adjacent = self._find_adjacent_chunks(result, cfg, processed_indices, result_lookup)
            if adjacent:
                return self._resolve_merged_passage([result] + adjacent, cfg)
        return self._resolve_chunk_passage(result, cfg)
    
    def _find_adjacent_chunks(
        self,
        result: RetrievalResult,
        cfg: OrchestratorConfig,
        processed_indices: Set[int],
        result_lookup: Dict[Tuple[str, int], RetrievalResult],
    ) -> List[RetrievalResult]:
        if not result.chunk_index:
            return []
        
        adjacent = []
        index = result.chunk_index
        
        # Next chunk
        nxt = result_lookup.get((result.source, index + 1))
        if nxt and nxt.chunk_index not in processed_indices:
            adjacent.append(nxt)
        
        # Previous chunk
        prv = result_lookup.get((result.source, index - 1))
        if prv and prv.chunk_index not in processed_indices:
            adjacent.insert(0, prv)
        
        return adjacent
    
    def _resolve_chunk_passage(
        self,
        result: RetrievalResult,
        cfg: OrchestratorConfig,
    ) -> Optional[ResolvedPassage]:
        chunk = self._fetch_chunk(result.source, result.chunk_index)
        if not chunk:
            return None
        
        text, indices, pages = self._get_full_context(chunk, result.source, cfg)
        
        metadata = PassageMetadata(
            source=result.source,
            page_start=min(pages) if pages else 0,
            page_end=max(pages) if pages else 0,
            chunk_indices=indices,
            chunk_count=len(indices),
            is_merged=False,
            gap_filled=False,
            structural_type=chunk.get("structural_type", "body"),
            penalty_factor=1.0,
            was_penalized=False,
            avg_score=result.score,
            semantic_score=getattr(result, 'semantic_score', None),
            keyword_score=getattr(result, 'keyword_score', None),
            rrf_score=getattr(result, 'rrf_score', None),
            rerank_score=getattr(result, 'rerank_score', None),
            matched_terms=list(result.matched_terms) if result.matched_terms else [],
        )
        
        audit_dict = getattr(result, 'audit', None)
        if audit_dict and cfg.include_audit:
            audit_dict = audit_dict.to_dict() if hasattr(audit_dict, 'to_dict') else audit_dict
        
        return ResolvedPassage(
            text=text,
            metadata=metadata,
            audit=audit_dict if cfg.include_audit else None,
        )
    
    def _resolve_merged_passage(
        self,
        results: List[RetrievalResult],
        cfg: OrchestratorConfig,
    ) -> Optional[ResolvedPassage]:
        if not results:
            return None
        
        sorted_results = sorted(results, key=lambda r: r.chunk_index or 0)
        chunks = []
        indices = []
        pages = []
        
        for r in sorted_results:
            chunk = self._fetch_chunk(r.source, r.chunk_index)
            if chunk:
                chunks.append(chunk)
                indices.append(r.chunk_index)
                pages.extend([chunk.get("page", 0), chunk.get("page_end", 0)])
        
        if not chunks:
            return None
        
        texts = [c.get("text", "") for c in chunks]
        stitched_text = self._stitch_texts(texts, [c.get("page", 0) for c in chunks])
        
        anchor = sorted_results[0]
        avg_score = sum(r.score for r in sorted_results) / len(sorted_results)
        structural_types = [c.get("structural_type", "body") for c in chunks]
        dominant_type = max(set(structural_types), key=structural_types.count) if structural_types else None
        
        metadata = PassageMetadata(
            source=anchor.source,
            page_start=min(pages) if pages else 0,
            page_end=max(pages) if pages else 0,
            chunk_indices=indices,
            chunk_count=len(indices),
            is_merged=True,
            gap_filled=cfg.fill_gaps and len(indices) > len(results),
            structural_type=dominant_type,
            penalty_factor=1.0,
            was_penalized=False,
            avg_score=avg_score,
            semantic_score=anchor.semantic_score,
            keyword_score=anchor.keyword_score,
            rrf_score=anchor.rrf_score,
            rerank_score=anchor.rerank_score,
            matched_terms=list(anchor.matched_terms) if anchor.matched_terms else [],
        )
        
        return ResolvedPassage(
            text=stitched_text,
            metadata=metadata,
            audit=None,
        )
    
    # ═══════════════════════════════════════════════════════════════
    # Storage Helpers (Cached, all columns)
    # ═══════════════════════════════════════════════════════════════
    
    def _fetch_chunk(
        self,
        source: str,
        index: int,
    ) -> Optional[Dict[str, Any]]:
        chunks = self._fetch_chunks(source, [index])
        return chunks.get(index)
    
    def _fetch_chunks(
        self,
        source: str,
        indices: List[int],
        columns: Optional[List[str]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        if not indices:
            return {}
        
        if source not in self._source_fetched:
            try:
                rows = self.store.query_records(filename=source)
            except Exception as e:
                darwin.log(f"[Orchestrator] Failed to fetch chunks for {source}: {e}", LogType.ERROR)
                return {}
            
            self._chunk_cache[source] = {}
            for row in rows:
                idx = row.get("index")
                if idx is None:
                    idx = row.get("chunk_index")
                if idx is not None:
                    self._chunk_cache[source][idx] = row
            self._source_fetched.add(source)
            darwin.log(f"[Orchestrator] Cached {len(self._chunk_cache[source])} chunks for {source}", LogType.INFO)
        
        result = {}
        for idx in indices:
            row = self._chunk_cache[source].get(idx)
            if row:
                if columns:
                    filtered = {col: row.get(col) for col in columns if col in row}
                    result[idx] = filtered
                else:
                    result[idx] = row
        return result
    
    # ═══════════════════════════════════════════════════════════════
    # Text Processing Helpers
    # ═══════════════════════════════════════════════════════════════
    
    def _get_full_context(
        self,
        chunk: Dict[str, Any],
        source: str,
        cfg: OrchestratorConfig,
    ) -> Tuple[str, List[int], List[int]]:
        texts = [chunk.get("text", "")]
        indices = [chunk.get("index", 0)]
        pages = [chunk.get("page", 0), chunk.get("page_end", 0)]
        
        direction = chunk.get("overlap_direction", "none")
        if direction in ("left", "both"):
            prev_idx = chunk.get("index", 0) - 1
            if prev_idx >= 0:
                prev = self._fetch_chunk(source, prev_idx)
                if prev:
                    texts.insert(0, prev.get("text", ""))
                    indices.insert(0, prev_idx)
                    pages.extend([prev.get("page", 0), prev.get("page_end", 0)])
        
        if direction in ("right", "both"):
            next_idx = chunk.get("index", 0) + 1
            nxt = self._fetch_chunk(source, next_idx)
            if nxt:
                texts.append(nxt.get("text", ""))
                indices.append(next_idx)
                pages.extend([nxt.get("page", 0), nxt.get("page_end", 0)])
        
        stitched = self._stitch_texts(texts, pages)
        return stitched, indices, pages
    
    def _stitch_texts(self, texts: List[str], pages: List[int]) -> str:
        if not texts:
            return ""
        
        all_sentences = []
        for text in texts:
            all_sentences.extend(split_sentences(text))
        
        deduped = []
        for s in all_sentences:
            normalized = " ".join(s.split())
            if not deduped or normalized != " ".join(deduped[-1].split()):
                deduped.append(s)
        
        if len(texts) > 1:
            result = []
            for i, text in enumerate(texts):
                if i > 0 and pages and pages[i] != pages[i-1]:
                    result.append(f"\n[Page {pages[i]}]\n")
                result.append(text)
            return " ".join(result)
        
        return " ".join(deduped)
    
    def _smart_truncate(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        truncated = text[:max_len]
        for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            idx = truncated.rfind(punct)
            if idx != -1 and idx > max_len * 0.5:
                return truncated[:idx + 1]
        return truncated
    
    # ═══════════════════════════════════════════════════════════════
    # Context Builder
    # ═══════════════════════════════════════════════════════════════
    
    def _build_context(
        self,
        query: str,
        method: str,
        passages: List[ResolvedPassage],
        raw_results: List[RetrievalResult],
        candidates_found: int,
        penalized_count: int,
        elapsed_ms: int,
        fallback_used: bool = False,
    ) -> RetrievalContext:
        total_chars = sum(len(p.text) for p in passages)
        total_passages = len(passages)
        scores = [p.metadata.avg_score for p in passages]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        sources = list(set(p.metadata.source for p in passages))
        source_diversity = len(sources) / min(len(passages), 5) if passages else 0.0
        source_diversity = min(1.0, source_diversity)
        
        audit_summary = {
            "total_results": len(raw_results),
            "penalized_count": penalized_count,
            "candidates_found": candidates_found,
            "passages_resolved": total_passages,
            "structural_types": {},
        }
        for p in passages:
            if p.metadata.structural_type:
                t = p.metadata.structural_type
                audit_summary["structural_types"][t] = audit_summary["structural_types"].get(t, 0) + 1
        
        return RetrievalContext(
            query=query,
            method=method,
            passages=passages,
            total_chars=total_chars,
            total_passages=total_passages,
            elapsed_ms=elapsed_ms,
            candidates_found=candidates_found,
            penalized_count=penalized_count,
            max_score=max_score,
            avg_score=avg_score,
            sources_used=sources,
            source_diversity=source_diversity,
            fallback_used=fallback_used,
            raw_results=raw_results,
            audit_summary=audit_summary,
        )
    
    def _empty_context(
        self,
        cfg: OrchestratorConfig,
        elapsed_ms: int,
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
            fallback_reason="No results found",
            raw_results=[],
            audit_summary={"error": "empty_context"},
        )