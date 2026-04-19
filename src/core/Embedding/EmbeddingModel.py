


# ============================================================================
# Constants
# ============================================================================
from src.lib.Terminal import TerminalUI
import psutil
import torch
from typing import Optional,Dict
from dataclasses import dataclass




DEFAULT_MODEL      = "intfloat/e5-small-v2"
DEFAULT_BATCH_SIZE = 64




# ============================================================================
# Model Presets - Token limits & Prompting strategies
# ============================================================================

@dataclass
class ModelPreset:
    """Configuration for a specific embedding model."""
    max_tokens: int
    min_tokens: int
    overlap: int
    category: str
    # Prompting strategy
    needs_prefix: bool = True
    passage_prefix: str = "passage: "
    query_prefix: str = "query: "
    # Special handling flags
    is_instruction_tuned: bool = False
    instruction_template: Optional[str] = None

PRESETS: Dict[str, ModelPreset] = {
    "intfloat/e5-small-v2": ModelPreset(
        max_tokens=480,  # 512 - safety margin
        min_tokens=120,
        overlap=3,
        category="plain_text",
        needs_prefix=True,
        passage_prefix="passage: ",
        query_prefix="query: "
    ),
    
    "GritLM/GritLM-7B": ModelPreset(
        max_tokens=3800,  # 4096 - safety margin
        min_tokens=300,
        overlap=4,
        category="semantic_markdown",
        needs_prefix=False,  # GritLM uses instruction format instead
        is_instruction_tuned=True,
        instruction_template="<|user|>\nRepresent this passage for retrieval: {text}\n<|assistant|>\n"
    ),
    
    "BAAI/bge-m3": ModelPreset(
        max_tokens=8000,  # 8192 - safety margin
        min_tokens=400,
        overlap=5,
        category="semantic_markdown",
        needs_prefix=False,  # BGE doesn't need prefixes
        is_instruction_tuned=False
    ),
}

# Short name aliases
MODEL_ALIASES = {
    "e5-small": "intfloat/e5-small-v2",
    "gritlm": "GritLM/GritLM-7B",
    "bge-m3": "BAAI/bge-m3",
}

# ============================================================================
# Token-aware text preparation
# ============================================================================

def get_preset(model_name: str) -> ModelPreset:
    """Resolve model name to its preset configuration."""
    # Check aliases first
    resolved = MODEL_ALIASES.get(model_name, model_name)
    
    if resolved not in PRESETS:
        TerminalUI.warn(f"No preset for {resolved}, using e5-small defaults")
        return PRESETS["intfloat/e5-small-v2"]
    
    return PRESETS[resolved]




def get_optimal_batch_size(requested_size: int = 32) -> int:
    """
    Detects LIVE hardware availability to set a safe batch size.
    """
    # 1. GPU Check (High Priority)
    if torch.cuda.is_available():
        vram_free = torch.cuda.mem_get_info()[0] / (1024**3) # Free VRAM in GB
        if vram_free > 4: return 128
        return 64

    # 2. CPU/RAM Check (The "Chrome" Protector)
    # We look at 'available' RAM, not 'total'
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    cpu_usage = psutil.cpu_percent(interval=0.1)

    # Logic: If system is already struggling, go small
    if available_gb < 2.0 or cpu_usage > 80:
        TerminalUI.warn("System resources low! Throttling batch size for stability.")
        return 16

    # For your 12GB Dell (usually has 4-6GB available with apps open)
    if available_gb > 4.0:
        return 64
    
    return requested_size # Fallback to 32


def get_model_specific_batch_size(model_name: str, requested_size: int) -> int:
    """Adjust batch size based on model memory footprint."""
    base_size = get_optimal_batch_size(requested_size)
    
    # Model-specific scaling factors
    if "GritLM-7B" in model_name:
        # 7B parameter model needs smaller batches
        return max(8, base_size // 4)
    elif "bge-m3" in model_name:
        # BGE-M3 is memory efficient
        return base_size
    elif "e5-small" in model_name:
        # Small model can handle larger batches
        return base_size * 2
    
    return base_size





# ============================================================================
# Model loader — lazy, cached per session
# ============================================================================

_model_cache: dict = {}


def load_model(model_id: str):
    if model_id in _model_cache:
        return _model_cache[model_id]

    import torch # Add this import
    from sentence_transformers import SentenceTransformer

    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = TerminalUI.timer(f"load_model:{model_id.split('/')[-1]}")
    TerminalUI.dim(f"Loading {model_id} on [bold]{device.upper()}[/bold]")

    # Pass the device to the model
    model = SentenceTransformer(model_id, device=device)
    _model_cache[model_id] = model

    t.done()
    return model

