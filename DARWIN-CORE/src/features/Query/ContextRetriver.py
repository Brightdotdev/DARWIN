from typing import Dict, List, Optional
import numpy as np
from src.features.Query.lib import PreprocessedQuery
from src.features.Indexing.Embedding.Embedder import embed_text

class DummyVectorStore:
    """Mock vector store for testing."""
    def search(self, query: str, top_k: int, retrieval_type: str = "semantic") -> List[Dict]:
        return [{"text": f"Mock result for: {query}", "metadata": {"source": "mock"}}]

class ContextRetriever:
    def __init__(self, vector_store=None, embedding_model=None):
        self.vector_store = vector_store or DummyVectorStore()

    def retrieve(
        self,
        query: PreprocessedQuery,
        conversation_history: List,  # list of ConversationTurn
        user_profile: Optional[Dict] = None,
        retrieval_params: Optional[Dict] = None,
    ) -> Dict:
        results = {}
        # Document RAG
        if query.tier in ("complex", "moderate") and self.vector_store:
            params = retrieval_params or {"top_k": 5, "retrieval_type": "semantic"}
            results["doc_chunks"] = self.vector_store.search(
                query.rag_query, params.get("top_k", 5), params.get("retrieval_type", "semantic")
            )
        else:
            results["doc_chunks"] = []

        # Conversation memory (always, but limited for simple)
        if conversation_history:
            query_embedding = None
            if query.cleaned_text:
                query_embedding = embed_text(query.cleaned_text)
            results["relevant_history"] = self._retrieve_history(
                query_embedding, conversation_history, top_k=3
            )
        else:
            results["relevant_history"] = []

        # Link resolution (mock)
        results["resolved_links"] = {}
        for link in query.extracted_links:
            results["resolved_links"][link.placeholder] = f"Content of {link.raw_url} (mock)"

        # Blob resolution
        results["resolved_blobs"] = {}
        for blob in query.extracted_blobs:
            results["resolved_blobs"][blob.placeholder] = blob.content_preview  # full would be fetched

        # User profile
        results["user_profile"] = user_profile or {}

        return results

    def _retrieve_history(self, query_embedding, history, top_k=3):
        if query_embedding is None:
            # Fallback: last N
            return history[-top_k:]
        # Compute similarity
        turns_with_sim = []
        for turn in history:
            if turn.embedding:
                sim = np.dot(query_embedding, turn.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(turn.embedding)
                )
                turns_with_sim.append((sim, turn))
        turns_with_sim.sort(key=lambda x: x[0], reverse=True)
        # Return top_k and sort chronologically
        top = turns_with_sim[:top_k]
        top.sort(key=lambda x: x[1].turn_id)
        return [t for _, t in top]