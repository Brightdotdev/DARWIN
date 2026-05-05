
from dataclasses import dataclass, field
from typing import Optional, List



@dataclass
class IndexingArgs:
    path:          str
    threshold:     float 
    idf_smoothing: float 
    only_kw: bool = False
    only_semantic: bool = False
    files: list[str] = field(default_factory=list)
    inspect:       bool = False
    rerun:         bool = False