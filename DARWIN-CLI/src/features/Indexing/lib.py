
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IndexingArgs:
    path : str
    idf_smoothing : float         
    threshold     : float      
    file:          Optional[str] = None
    inspect:       bool          = False
    rerun:         bool          = False

