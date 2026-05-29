from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class RetrievalParams:
    top_k: int = 5
    retrieval_type: str = "semantic"  # "semantic", "hybrid", "keyword"

@dataclass
class QueryOrchestratorConfig:
    # Tier thresholds
    max_history_turns: int = 5
    max_history_tokens: int = 1000
    max_context_tokens: int = 2000
    max_total_tokens: int = 4000
    blob_threshold: int = 500
    desperation_threshold: float = 0.6
    quality_threshold: float = 0.4
    link_summary_max_tokens: int = 200
    blob_summary_max_tokens: int = 150

    # Retrieval parameters per query intent
    retrieval_params: Dict[str, RetrievalParams] = field(default_factory=lambda: {
        "deep_dive": RetrievalParams(top_k=8, retrieval_type="hybrid"),
        "simple": RetrievalParams(top_k=3, retrieval_type="semantic"),
        "lookup": RetrievalParams(top_k=1, retrieval_type="keyword"),
        "entity": RetrievalParams(top_k=5, retrieval_type="keyword"),
    })

    # LLM model names (placeholders)
    fast_model: str = "gpt-4o-mini"
    normal_model: str = "gpt-4o"
    think_model: str = "gpt-4o"

    # Embedding dimension
    embedding_dim: int = 1536

    # Profanity filter
    enable_profanity_filter: bool = True
    profanity_list: List[str] = field(default_factory=list)  # to be loaded

    # SpaCy quality checks
    min_meaningful_tokens: int = 2
    max_junk_token_ratio: float = 0.2