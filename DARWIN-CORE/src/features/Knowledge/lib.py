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

    # pass control
    graph_only: bool = False
    wiki_only: bool = False

    # graph config (new spaCy-based pipeline)
    preset: Optional[str] = None  # academic|research|legal|narrative|technical
    processing_mode: str = "auto"
    chunk_size: int = 512
    chunk_overlap: float = 0.2
    relation_threshold: float = 0.3
    dedup_strategy: str = "svo_canonical"
    max_cooccurrence_edges: int = 500

    # legacy overrides
    threshold: Optional[float] = None
    min_freq: int = 2
    graph_max_concepts: int = 75
    visualize: bool = True

    # wiki config

    # shared
    inspect: bool = False
    rerun: bool = False
    json_output: bool = False