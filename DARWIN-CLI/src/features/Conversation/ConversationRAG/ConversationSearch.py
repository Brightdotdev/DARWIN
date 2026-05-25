"""
Conversation search: semantic, temporal, contradiction, synthesis.
"""

import json
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np

from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Indexing.Embedding.Embedder import embed_query
from src.features.Conversation.ConversationRAG.types import (
    ConvoChunk, MemoryDocument, Contradiction, SearchResult, SearchFilters
)


class ConvoSearch:
    """
    Search across indexed conversation threads.
    """
    
    def __init__(self, store):
        self.store = store  # DarwinStore with ConvoEmbeddingsMixin
    
    # ── Standard search ──────────────────────────────────────────────────────
    
    def search(self, query: str, top_k: int = 5, filters: Optional[SearchFilters] = None) -> List[SearchResult]:
        """
        Semantic search across all indexed threads.
        """
        if filters is None:
            filters = SearchFilters()
        
        query_embed = embed_query(query)
        results = []
        
        # Search across all thread NPZs
        for thread_id in self.store.list_indexed_convos():
            thread_results = self._search_thread(thread_id, query_embed, filters, top_k)
            results.extend(thread_results)
        
        # Global sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    def _search_thread(
        self,
        thread_id: str,
        query_embed: np.ndarray,
        filters: SearchFilters,
        top_k: int,
    ) -> List[SearchResult]:
        """Search within a single thread's NPZ."""
        vectors, chunks = self.store.load_convo_embeddings(thread_id)
        if vectors is None or chunks is None:
            return []
        
        # Apply filters
        filtered_chunks = self._apply_filters(chunks, filters)
        if not filtered_chunks:
            return []
        
        # Get indices of filtered chunks
        filtered_indices = [c.index for c in filtered_chunks]
        filtered_vectors = vectors[filtered_indices]
        
        # Cosine similarity
        scores = self._cosine_similarity(query_embed, filtered_vectors)
        
        # Build results
        results = []
        for idx, score_idx in enumerate(filtered_indices):
            if scores[idx] < filters.min_relevance:
                continue
            
            chunk = chunks[score_idx]
            results.append(SearchResult(
                doc=MemoryDocument(
                    doc_id=f"{thread_id}_{score_idx}",
                    text=chunk.text,
                    clean=chunk.clean,
                    embedding=vectors[score_idx],
                    metadata=chunk,
                ),
                score=float(scores[idx]),
                context_snippet=self._get_context_snippet(thread_id, score_idx, chunks),
            ))
        
        return results
    
    # ── Temporal search ──────────────────────────────────────────────────────
    
    def search_at_time(
        self,
        query: str,
        target_time: datetime,
        window: timedelta = timedelta(days=7),
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search within time window."""
        filters = SearchFilters(
            date_range=(target_time - window, target_time + window)
        )
        return self.search(query, top_k=top_k, filters=filters)
    
    # ── Contradiction search (hybrid) ────────────────────────────────────────
    
    def find_contradictions(
        self,
        claim: str,
        thread_id: Optional[str] = None,
        lookback: timedelta = timedelta(days=30),
    ) -> List[Contradiction]:
        """
        Hybrid contradiction detection.
        Phase 1: Rule-based filter (cheap)
        Phase 2: LLM judge for survivors (expensive but accurate)
        """
        # Phase 1: Retrieve similar statements
        now = datetime.now()
        filters = SearchFilters(
            date_range=(now - lookback, now),
            min_relevance=0.6,
        )
        if thread_id:
            filters.thread_ids = [thread_id]
        
        similar = self.search(claim, top_k=20, filters=filters)
        
        # Rule-based filtering
        candidates = []
        for result in similar:
            doc = result.doc
            if doc.metadata.is_preference_statement:
                if self._rule_based_contradiction(claim, doc.clean):
                    candidates.append(doc)
            elif doc.metadata.is_factual_claim:
                if self._factual_mismatch(claim, doc.clean):
                    candidates.append(doc)
        
        # Phase 2: LLM judge (placeholder — actual implementation uses fast-tier)
        contradictions = []
        for doc in candidates:
            severity = self._estimate_contradiction_severity(claim, doc.clean)
            if severity > 0.5:
                contradictions.append(Contradiction(
                    claim_a=doc,
                    claim_b=MemoryDocument(
                        doc_id="current_claim",
                        text=claim,
                        clean=claim,
                    ),
                    contradiction_type=self._classify_contradiction_type(claim, doc.clean),
                    severity=severity,
                    time_delta=timedelta(days=1),  # Placeholder
                    detected_by="rule",  # Would be "llm_judge" after LLM call
                ))
        
        return contradictions
    
    def _rule_based_contradiction(self, claim: str, past: str) -> bool:
        """Rule-based contradiction detection."""
        claim_lower = claim.lower()
        past_lower = past.lower()
        
        # Negation patterns
        negation_pairs = [
            ("love", "hate"), ("like", "dislike"), ("want", "don't want"),
            ("need", "don't need"), ("should", "shouldn't"), ("is", "isn't"),
            ("are", "aren't"), ("was", "wasn't"), ("has", "doesn't have"),
        ]
        
        for pos, neg in negation_pairs:
            if (pos in claim_lower and neg in past_lower) or (neg in claim_lower and pos in past_lower):
                return True
        
        # Numeric mismatch (simple)
        import re
        claim_nums = set(re.findall(r'\d+', claim_lower))
        past_nums = set(re.findall(r'\d+', past_lower))
        if claim_nums and past_nums and not claim_nums.intersection(past_nums):
            return True
        
        return False
    
    def _factual_mismatch(self, claim: str, past: str) -> bool:
        """Detect factual/numeric mismatches."""
        import re
        
        # Date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}\b',
        ]
        
        claim_dates = []
        past_dates = []
        for pattern in date_patterns:
            claim_dates.extend(re.findall(pattern, claim.lower()))
            past_dates.extend(re.findall(pattern, past.lower()))
        
        if claim_dates and past_dates:
            # Simple: different dates mentioned
            if set(claim_dates) != set(past_dates):
                return True
        
        return False
    
    def _estimate_contradiction_severity(self, claim: str, past: str) -> float:
        """Estimate how strongly two statements contradict."""
        # Placeholder — would use LLM judge
        if self._rule_based_contradiction(claim, past):
            return 0.8
        return 0.3
    
    def _classify_contradiction_type(self, claim: str, past: str) -> str:
        """Classify the type of contradiction."""
        claim_lower = claim.lower()
        
        preference_markers = ["like", "love", "hate", "prefer", "want"]
        if any(m in claim_lower for m in preference_markers):
            return "preference_reversal"
        
        goal_markers = ["build", "create", "make", "need", "should"]
        if any(m in claim_lower for m in goal_markers):
            return "goal_change"
        
        return "direct_negation"
    
    # ── Pushback injection ───────────────────────────────────────────────────
    
    def inject_pushback(self, claim: str, contradiction: Contradiction) -> Optional[str]:
        """Generate pushback message if contradiction is severe and old enough."""
        if contradiction.severity > 0.7 and contradiction.time_delta > timedelta(days=1):
            return (
                f"[Memory Pushback] On {contradiction.claim_a.metadata.timestamp.strftime('%b %d')}, "
                f"you said: '{contradiction.claim_a.text[:100]}...' "
                f"Today you said: '{contradiction.claim_b.text[:100]}...' "
                f"Has your position changed?"
            )
        return None
    
    # ── Cross-thread synthesis ───────────────────────────────────────────────
    
    def synthesize_across_threads(self, query: str, max_threads: int = 3) -> dict:
        """
        Find patterns across multiple threads.
        Returns synthesis result (placeholder — actual uses smart-tier LLM).
        """
        thread_hits = self.search(query, top_k=max_threads * 5)
        
        # Group by thread
        by_thread = {}
        for hit in thread_hits:
            tid = hit.doc.metadata.thread_id
            if tid not in by_thread:
                by_thread[tid] = []
            by_thread[tid].append(hit)
        
        # Take top thread per group
        top_threads = list(by_thread.keys())[:max_threads]
        
        return {
            "query": query,
            "threads": top_threads,
            "total_hits": len(thread_hits),
            "synthesis": f"Found discussions in {len(top_threads)} threads related to: {query}",
        }
    
    # ── Internal helpers ─────────────────────────────────────────────────────
    
    def _apply_filters(self, chunks: List[ConvoChunk], filters: SearchFilters) -> List[ConvoChunk]:
        """Apply SearchFilters to chunk list."""
        result = chunks
        
        if filters.date_range:
            start, end = filters.date_range
            result = [c for c in result if start <= c.timestamp <= end]
        
        if filters.roles:
            result = [c for c in result if c.role in filters.roles]
        
        if filters.thread_ids:
            result = [c for c in result if c.thread_id in filters.thread_ids]
        
        if filters.topics:
            result = [c for c in result if any(t in c.topics for t in filters.topics)]
        
        if filters.sentiment_range:
            min_s, max_s = filters.sentiment_range
            result = [c for c in result if min_s <= c.sentiment <= max_s]
        
        return result
    
    def _cosine_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and vectors."""
        # Normalize
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        
        # Dot product
        return np.dot(vectors_norm, query_norm)
    
    def _get_context_snippet(self, thread_id: str, chunk_idx: int, chunks: List[ConvoChunk], window: int = 2) -> str:
        """Get surrounding chunks for context."""
        start = max(0, chunk_idx - window)
        end = min(len(chunks), chunk_idx + window + 1)
        
        snippet_chunks = chunks[start:end]
        return " | ".join([c.text[:100] for c in snippet_chunks])