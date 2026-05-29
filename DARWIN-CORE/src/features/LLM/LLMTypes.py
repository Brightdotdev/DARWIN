from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LLMResponse:
    """Unified return type for all LLMClient calls."""
    text: str = ""
    json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """True if the call succeeded and no error is present."""
        return self.error is None and (bool(self.text) or self.json is not None)

    def raise_for_error(self) -> None:
        """Convenience helper to raise a RuntimeError if the response carries an error."""
        if self.error:
            raise RuntimeError(f"LLM call failed: {self.error}")