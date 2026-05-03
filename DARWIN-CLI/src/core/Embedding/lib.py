"""
Utility functions and configuration for embedding models.

"""

from typing import Optional
from dataclasses import dataclass



@dataclass
class EmbeddingArgs:
    target:     str
    file:       Optional[str] = None
    project:    Optional[str] = None
    inspect:    bool = False
    rerun:      bool = False

