"""
ConversationConfig.py
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ConversationConfig:
    max_context_tokens: int = 8192
    system_prompt_tokens: int = 800
    response_budget_tokens: int = 512
    memory_top_k: int = 3
    summary_trigger_tokens: int = 6000
    max_turns_before_summary: int = 10

    budget_mode: bool = False
    min_confidence_for_llm: float = 0.7

    max_npz_mb: float = 20.0
    rechunk_keep_recent: int = 3

    memory_format: str = "claude_xml"
    enable_cross_conversation_memory: bool = True

    embedding_dim: int = 384
    embedding_dtype: str = "float16"

    def effective_budget(self) -> int:
        return self.max_context_tokens - self.system_prompt_tokens - self.response_budget_tokens