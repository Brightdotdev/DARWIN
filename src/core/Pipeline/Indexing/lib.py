
from dataclasses import dataclass, field

@dataclass
class IndexingArgs:
    threshold: float
    idf_smoothing: float
    only_kw: bool = False
    path: str = "."
    only_semantic: bool = False
    files: list[str] = field(default_factory=list)
    rerun: bool = False

    def __post_init__(self):
        self.threshold = self._clamp(self.threshold, 0.0, 1.0)
        self.idf_smoothing = self._clamp(self.idf_smoothing, 0.0, 2.0)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))