
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class IndexingArgs:
    threshold: float
    idf_smoothing: float
    only_kw: bool = False
    path: str = "."
    only_semantic: bool = False
    files: Optional[list[str]] = None
    rerun: bool = False
    batch_size: Optional[int] = None
    token_budget: int = 2048
    use_length_batching: bool = True

    def __post_init__(self):
        self.threshold = self._clamp(self.threshold, 0.0, 1.0)
        self.idf_smoothing = self._clamp(self.idf_smoothing, 0.0, 2.0)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))