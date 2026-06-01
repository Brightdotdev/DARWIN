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
    model_name: str = ""  # HF ID (e.g., 'intfloat/e5-small-v2')


# ═══════════════════════════════════════════════════════════════════════════
# Registry — keyed by canonical model_id (from ModelRegistry REGISTRY keys)
# NEW: Added ONNX variants
# ═══════════════════════════════════════════════════════════════════════════

PRESETS: dict[str, ModelPreset] = {
    # ── PyTorch models ──
    "intfloat/e5-small-v2": ModelPreset(
        max_tokens=480,
        min_tokens=120,
        overlap=3,
        category="plain_text",
        needs_prefix=True,
        passage_prefix="passage: ",
        query_prefix="query: ",
        model_name="intfloat/e5-small-v2",
    ),
    "BAAI/bge-m3": ModelPreset(
        max_tokens=8000,
        min_tokens=400,
        overlap=5,
        category="semantic_markdown",
        needs_prefix=False,
        model_name="BAAI/bge-m3",
    ),
    "GritLM/GritLM-7B": ModelPreset(
        max_tokens=3800,
        min_tokens=300,
        overlap=4,
        category="semantic_markdown",
        needs_prefix=False,
        is_instruction_tuned=True,
        instruction_template="<|user|>\nRepresent this passage for retrieval: {text}\n<|assistant|>\n",
        model_name="GritLM/GritLM-7B",
    ),
    
    # ── ONNX variants (same behavior as PyTorch counterparts) ──
    "intfloat/e5-small-v2-onnx": ModelPreset(
        max_tokens=480,
        min_tokens=120,
        overlap=3,
        category="plain_text",
        needs_prefix=True,
        passage_prefix="passage: ",
        query_prefix="query: ",
        model_name="intfloat/e5-small-v2",
    ),
    "BAAI/bge-m3-onnx": ModelPreset(
        max_tokens=8000,
        min_tokens=400,
        overlap=5,
        category="semantic_markdown",
        needs_prefix=False,
        model_name="BAAI/bge-m3",
    ),
    
    # ── API models ──
    "openai/text-embedding-3-small": ModelPreset(
        max_tokens=8000,
        min_tokens=400,
        overlap=5,
        category="semantic_markdown",
        needs_prefix=False,
        model_name="openai/text-embedding-3-small",
    ),
    "openai/text-embedding-3-large": ModelPreset(
        max_tokens=8000,
        min_tokens=400,
        overlap=5,
        category="semantic_markdown",
        needs_prefix=False,
        model_name="openai/text-embedding-3-large",
    ),
    "cohere/embed-english-v3.0": ModelPreset(
        max_tokens=512,
        min_tokens=100,
        overlap=3,
        category="plain_text",
        needs_prefix=False,
        model_name="cohere/embed-english-v3.0",
    ),
    "voyage/voyage-3-lite": ModelPreset(
        max_tokens=4000,
        min_tokens=200,
        overlap=4,
        category="semantic_markdown",
        needs_prefix=False,
        model_name="voyage/voyage-3-lite",
    ),
}

# Fallback for unknown models
DEFAULT_PRESET = ModelPreset(
    max_tokens=480, min_tokens=120, overlap=3,
    category="plain_text", model_name="unknown",
)


def get_preset(model_key: Optional[str] = None) -> ModelPreset:
    """
    Returns the full ModelPreset for a given model_key, or the
    CURRENTLY ACTIVE embedding model if none is provided.
    
    Handles ONNX variants by looking up the base model preset.
    
    Args:
        model_key: Canonical model_id (e.g., 'BAAI/bge-m3') or None
                   to auto-detect from the active loader.
    
    Returns:
        ModelPreset matching the model, or DEFAULT_PRESET if unknown.
    """
    if model_key is not None:
        # Direct lookup
        if model_key in PRESETS:
            return PRESETS[model_key]
        # Try stripping -onnx suffix for fallback
        if model_key.endswith("-onnx"):
            base_key = model_key[:-5]
            if base_key in PRESETS:
                return PRESETS[base_key]
        return DEFAULT_PRESET
    
    try:
        from src.core.Models import ModelManager, ModelCategory
        loader = ModelManager.get_loader()
        model_key = loader.get_model_key(ModelCategory.EMBEDDING)
        return get_preset(model_key)  # Recursive with the key
    except (RuntimeError, ValueError, ImportError):
        return DEFAULT_PRESET


# ═══════════════════════════════════════════════════════════════════════════
# Hardware-aware batch sizing
# ═══════════════════════════════════════════════════════════════════════════

def get_optimal_batch_size(requested_size: int = 32) -> int:
    """
    Batch size tuned using pre-detected device config.
    No torch.cuda checks here anymore.
    """
    from src.features.Hardware.DeviceManager import DeviceManager
    import psutil

    try:
        cfg = DeviceManager.get()
    except RuntimeError:
        # DeviceManager not initialized — use safe defaults
        return min(requested_size, 16)

    # GPU path
    if cfg.use_gpu:
        try:
            import torch
            vram_free = torch.cuda.mem_get_info()[0] / (1024 ** 3)

            if vram_free > 8:
                return 128
            elif vram_free > 4:
                return 64
            else:
                return 32
        except Exception:
            return 32

    # CPU path
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    cpu_usage = psutil.cpu_percent(interval=0.1)

    if available_gb < 2.0 or cpu_usage > 80:
        return 16

    if available_gb > 4.0:
        return 64

    return requested_size


def get_model_batch_size(preset: Optional[ModelPreset] = None, requested_size: int = 32) -> int:
    """
    Batch size adjusted for the currently active model's memory footprint.
    
    Args:
        preset: ModelPreset for the active model (auto-detected if None).
        requested_size: Base batch size requested by the caller.
    
    Returns:
        Safe batch size considering both hardware and model requirements.
    """
    if preset is None:
        preset = get_preset()
    
    base_size = get_optimal_batch_size(requested_size)

    # Adjust for specific model families based on model_id
    model_name = preset.model_name.lower()
    
    if "gritlm-7b" in model_name or "gritlm" in model_name:
        return max(8, base_size // 4)
    elif "bge-m3" in model_name or "bge" in model_name:
        return base_size
    elif "e5-small" in model_name or "e5" in model_name:
        return base_size * 2

    return base_size