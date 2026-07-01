from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbeddingArgs:
    files: list[str] = field(default_factory=list)
    inspect:    bool = False
    rerun:      bool = False

    # FIXED: CLI --batch-size was parsed but never stored
    batch_size: Optional[int] = None

    # FIXED: CLI --min-batch-cpu / --min-batch-gpu were parsed but never stored
    min_batch_cpu: Optional[int] = None
    min_batch_gpu: Optional[int] = None

    # NEW: length-aware batching controls
    use_length_batching: bool = True
    token_budget: int = 2048