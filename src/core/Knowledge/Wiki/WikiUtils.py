"""
src/core/Knowledge/Wiki/WikiUtils.py
-------------------------------------
Data models and CLI args for wiki generation.
"""

from dataclasses import dataclass
from typing import Optional


DEFAULT_MAX_CONCEPTS = 30
DEFAULT_CHUNKS_PER_CONCEPT = 8


@dataclass
class WikiGenerationArgs:
    target: str
    file: Optional[str] = None
    project: Optional[str] = None
    mode: str = "index"          # "index" | "summary"
    llm_model: Optional[str] = None
    max_concepts: int = DEFAULT_MAX_CONCEPTS
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT
    inspect: bool = False
    rerun: bool = False