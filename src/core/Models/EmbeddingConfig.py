

from dataclasses import dataclass
from typing import Literal, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Model preset — complete config for one embedding model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelPreset:
    """Complete configuration for a specific embedding model."""
    # ── Chunking ──
    max_tokens: int
    min_tokens: int
    overlap: int
    category: Literal["plain_text", "semantic_markdown"]
    
    # ── Prompting ──
    needs_prefix: bool = True
    passage_prefix: str = "passage: "
    query_prefix: str = "query: "
    
    # ── Instruction-tuned models ──
    is_instruction_tuned: bool = False
    instruction_template: Optional[str] = None
    
    # ── Metadata ──
    model_name: str = ""  # HF ID


# ═══════════════════════════════════════════════════════════════════════════
# Registry — keyed by short name (what loader.preferences returns)
# ═══════════════════════════════════════════════════════════════════════════

PRESETS: dict[str, ModelPreset] = {
    # ── Plain-text models ──
    "e5-small-v2": ModelPreset(
        max_tokens=480,
        min_tokens=120,
        overlap=3,
        category="plain_text",
        needs_prefix=True,
        passage_prefix="passage: ",
        query_prefix="query: ",
        model_name="intfloat/e5-small-v2",
    ),

    # ── Semantic-markdown models ──
    "bge-m3": ModelPreset(
        max_tokens=8000,
        min_tokens=400,
        overlap=5,
        category="semantic_markdown",
        needs_prefix=False,
        model_name="BAAI/bge-m3",
    ),
    "gritlm-7b": ModelPreset(
        max_tokens=3800,
        min_tokens=300,
        overlap=4,
        category="semantic_markdown",
        needs_prefix=False,
        is_instruction_tuned=True,
        instruction_template="<|user|>\nRepresent this passage for retrieval: {text}\n<|assistant|>\n",
        model_name="GritLM/GritLM-7B",
    ),
}

# Fallback for unknown models
DEFAULT_PRESET = ModelPreset(
    max_tokens=480, min_tokens=120, overlap=3,
    category="plain_text", model_name="unknown",
)




def get_preset() -> ModelPreset:
    """
    Returns the full ModelPreset for the CURRENTLY ACTIVE embedding model.
    Auto-detects from the global ModelLoader. No parameters needed.

    Usage:
        preset = get_preset()
        # preset.max_tokens, preset.category, preset.query_prefix, ...
    """
    try:
        from src.core.Models import get_loader, ModelCategory
        loader = get_loader()
        model_key = loader.get_model_key(ModelCategory.EMBEDDING)
        return PRESETS.get(model_key, DEFAULT_PRESET)
    except RuntimeError:
        return DEFAULT_PRESET



# ═══════════════════════════════════════════════════════════════════════════
# Hardware-aware batch sizing
# ═══════════════════════════════════════════════════════════════════════════

def get_optimal_batch_size(requested_size: int = 32) -> int:
    """
    Detects LIVE hardware availability to set a safe batch size.
    """
    import psutil
    import torch

    if torch.cuda.is_available():
        vram_free = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        if vram_free > 4:
            return 128
        return 64

    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    cpu_usage = psutil.cpu_percent(interval=0.1)

    if available_gb < 2.0 or cpu_usage > 80:
        return 16

    if available_gb > 4.0:
        return 64

    return requested_size


def get_model_batch_size(preset : ModelPreset, requested_size: int = 32) -> int:
    """Batch size adjusted for the currently active model's footprint."""
    base_size = get_optimal_batch_size(requested_size)

    if "GritLM-7B" in preset.model_name:
        return max(8, base_size // 4)
    elif "bge-m3" in preset.model_name:
        return base_size
    elif "e5-small" in preset.model_name:
        return base_size * 2

    return base_size