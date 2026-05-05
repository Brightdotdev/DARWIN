"""
Utility functions and configuration for embedding models.

"""

from dataclasses import dataclass,field



@dataclass
class EmbeddingArgs:
    files: list[str] = field(default_factory=list)
    inspect:    bool = False
    rerun:      bool = False

