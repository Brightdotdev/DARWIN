"""
Darwin Query Module — Retriever
Thin wrapper around Darwin's existing retrieval infrastructure.
"""

from typing import Any, Dict, List, Optional

from src.features.QueryModule.types import (
    BlobHandle,
    RetrievedConcept,
    RetrievedPassage,
    RetrievalContext,
    RetrievalPolicy,
)


class Retriever:
    """
    Executes search based on RetrievalPolicy.
    Wraps existing Darwin orchestrators:
      - RetrievalOrchestrator (deep hybrid)
      - WikiOrchestrator (shallow concept)
      - BlobProcessor (blob chunks)
    """

    def __init__(
        self,
        retrieval_orchestrator=None,
        wiki_orchestrator=None,
        darwin_store=None,
    ):
        self.retrieval_orchestrator = retrieval_orchestrator
        self.wiki_orchestrator = wiki_orchestrator
        self.darwin_store = darwin_store

    def retrieve(
        self,
        query: str,
        policy: RetrievalPolicy,
        blob: Optional[BlobHandle] = None,
        pinned_files: Optional[List[str]] = None,
    ) -> RetrievalContext:
        """Execute retrieval based on policy."""
        
        # If RAG not required, return empty context
        if not policy.requires_rag:
            return RetrievalContext(
                passages=[],
                concepts=[],
                sources=[],
                total_found=0,
                query_used=query,
                policy=policy,
            )
        
        passages: List[RetrievedPassage] = []
        concepts: List[RetrievedConcept] = []
        sources: List[str] = []
        
        # Deep hybrid search via RetrievalOrchestrator
        if policy.retrieval.value in ("hybrid", "semantic") and self.retrieval_orchestrator:
            try:
                deep_results = self.retrieval_orchestrator.search(
                    query=query,
                    top_k=policy.top_k,
                    mode=policy.retrieval.value,
                    files=pinned_files,
                )
                for r in deep_results:
                    passages.append(RetrievedPassage(
                        content=r.get("content", ""),
                        source=r.get("source", "unknown"),
                        score=r.get("score", 0.0),
                        page=r.get("page"),
                        metadata=r.get("metadata", {}),
                    ))
                    if r.get("source") and r["source"] not in sources:
                        sources.append(r["source"])
            except Exception as e:
                # Log but don't fail — fallback to wiki/blob
                pass
        
        # Keyword search fallback or supplement
        if policy.retrieval.value in ("hybrid", "keyword"):
            if self.darwin_store:
                try:
                    keyword_results = self.darwin_store.keyword_search(
                        query=query,
                        top_k=policy.top_k,
                        files=pinned_files,
                    )
                    for r in keyword_results:
                        # Avoid duplicates by content similarity
                        if not self._is_duplicate(passages, r.get("content", "")):
                            passages.append(RetrievedPassage(
                                content=r.get("content", ""),
                                source=r.get("source", "unknown"),
                                score=r.get("score", 0.0) * 0.9,  # Slightly lower weight
                                page=r.get("page"),
                                metadata=r.get("metadata", {}),
                            ))
                            if r.get("source") and r["source"] not in sources:
                                sources.append(r["source"])
                except Exception:
                    pass
        
        # Shallow concept search via WikiOrchestrator
        if self.wiki_orchestrator:
            try:
                wiki_results = self.wiki_orchestrator.search_concepts(
                    query=query,
                    top_k=max(3, policy.top_k // 2),
                )
                for r in wiki_results:
                    concepts.append(RetrievedConcept(
                        name=r.get("name", ""),
                        description=r.get("description", ""),
                        related=r.get("related", []),
                        score=r.get("score", 0.0),
                    ))
            except Exception:
                pass
        
        # Blob chunk retrieval
        if blob:
            blob_chunks = blob.get_relevant_chunks(query, top_k=min(5, policy.top_k))
            for chunk in blob_chunks:
                passages.append(RetrievedPassage(
                    content=chunk.content,
                    source=f"blob:{blob.blob_id}",
                    score=0.85,  # High relevance for attached blobs
                    metadata={"blob_type": blob.blob_type.value, "chunk_index": chunk.index},
                ))
            if blob.blob_id not in sources:
                sources.append(f"blob:{blob.blob_id}")
        
        # Sort by score and deduplicate
        passages.sort(key=lambda p: p.score, reverse=True)
        passages = passages[:policy.top_k]
        
        return RetrievalContext(
            passages=passages,
            concepts=concepts,
            sources=sources,
            total_found=len(passages) + len(concepts),
            query_used=query,
            policy=policy,
        )

    def _is_duplicate(self, passages: List[RetrievedPassage], content: str, threshold: float = 0.85) -> bool:
        """Check if content is substantially duplicate of existing passage."""
        for p in passages:
            similarity = self._jaccard_similarity(p.content, content)
            if similarity > threshold:
                return True
        return False

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Simple Jaccard similarity for deduplication."""
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0