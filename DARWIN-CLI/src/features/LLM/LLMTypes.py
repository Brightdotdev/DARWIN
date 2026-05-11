from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LLMResponse:
    """Unified return type for all LLMClient calls."""
    text: str = ""
    json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None