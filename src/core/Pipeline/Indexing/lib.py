
from dataclasses import dataclass, field
from typing import Optional, List



@dataclass
class IndexingArgs:
    threshold:     float 
    idf_smoothing: float 
    only_kw: bool = False
    path:          str = "."
    only_semantic: bool = False
    files: list[str] = field(default_factory=list)
    rerun:         bool = False