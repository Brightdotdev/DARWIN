"""
src/features/Knowledge/lib.py
------------------------------
Shared args for combined knowledge pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KnowledgePipelineArgs:
    """Combined graph + wiki pipeline arguments."""

    # file selection
    files: Optional[List[str]] = None
    llm_provider: Optional[str] = None
    elevation : str = "none"

    # pass control
    graph_only: bool = False
    wiki_only: bool = False

    # graph config (new spaCy-based pipeline)
    preset: Optional[str] = None  # academic|research|legal|narrative|technical
    processing_mode: str = "auto"
    chunk_size: int = 512
    chunk_overlap: float = 0.2
    relation_threshold: float = 0.0
    dedup_strategy: str = "svo_canonical"
    max_cooccurrence_edges: int = 500

    # legacy overrides
    threshold: Optional[float] = None
    min_freq: int = 2
    graph_max_concepts: int = 75
    visualize: bool = False

    # wiki config
    wiki_mode: str = "index"
    chunks_per_concept: int = 5

    # shared
    inspect: bool = False
    rerun: bool = False
    ws_port: int = 8765
    no_ws: bool = False
    json_output: bool = False