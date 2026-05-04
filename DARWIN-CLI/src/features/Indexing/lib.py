
from dataclasses import dataclass, field
from typing import Optional, List



@dataclass
class IndexingArgs:
    path:          str
    threshold:     float 
    idf_smoothing: float 
    file:          Optional[str] = None
    cluster:       Optional[List[str]] = None
    inspect:       bool = False
    rerun:         bool = False