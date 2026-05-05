from dataclasses import dataclass
from typing import List, Optional



@dataclass
class KnowledgePipelineArgs:
    # file selectio
    llm : Optional[str] = None
    files: Optional[List[str]] = None

    # pass control
    graph_only: bool = False
    wiki_only: bool = False

    elevation : str = "none"
    # graph args
    threshold: Optional[float] = None
    min_freq: int = 2
    graph_max_concepts: int = 75
    visualize: bool = False

    # wiki args
    wiki_mode: str = "index"
    chunks_per_concept: int = 5

    # shared
    inspect: bool = False
    rerun: bool = False
    ws_port: int = 8765
    no_ws: bool = False
    json_output: bool = False