from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.core.Logger.DarwinLogger import darwin


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

    def raise_for_error(self) -> str:
        """Convenience helper to log and return the error message if the response carries an error."""
        if self.error:
            darwin.error(f"LLM call failed: {self.error}")
            return f"LLM call failed: {self.error}"
        return ""