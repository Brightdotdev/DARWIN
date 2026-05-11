# src/features/Query/RetrievalPolicy.py
"""
Static mapping from QuestionType to RetrievalPolicy.
"""
from src.features.Query.QueryTypes import QuestionType, RetrievalPolicy

QUESTION_RETRIEVAL_MAP: dict[QuestionType, RetrievalPolicy] = {
    "fact":          RetrievalPolicy(retrieval="hybrid", top_k=3, requires_rag=True),
    "instruction":   RetrievalPolicy(retrieval="hybrid", top_k=5, requires_rag=True),
    "reasoning":     RetrievalPolicy(retrieval="hybrid", top_k=8, requires_rag=True),
    "comparison":    RetrievalPolicy(retrieval="hybrid", top_k=10, searchable_topics=["both_entities"], requires_rag=True),
    "opinion":       RetrievalPolicy(retrieval="hybrid", top_k=5, requires_rag=False),
    "hypothetical":  RetrievalPolicy(retrieval="semantic", top_k=6, requires_rag=False),
    "clarification": RetrievalPolicy(retrieval="keyword", top_k=3, requires_rag=False),
    "multi_part":    RetrievalPolicy(retrieval="hybrid", top_k=10, requires_rag=True),
    "yes_no":        RetrievalPolicy(retrieval="keyword", top_k=3, requires_rag=False),
    "statement":     RetrievalPolicy(retrieval="keyword", top_k=2, requires_rag=False),
}


def get_retrieval_policy(question_type: QuestionType) -> RetrievalPolicy:
    return QUESTION_RETRIEVAL_MAP.get(question_type, QUESTION_RETRIEVAL_MAP["statement"])