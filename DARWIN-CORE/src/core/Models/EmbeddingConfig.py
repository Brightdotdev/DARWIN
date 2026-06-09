"""
src/core/Models/EmbeddingConfig.py
-----------------------------------
Model presets for chunking, prompting, and batch sizing.
Covers embedding models, reranker models, and API-based variants.

PURE DATA MODULE — no runtime object creation, no ModelManager imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Model preset — complete config for one embedding/reranker model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelPreset:
    """Complete configuration for a specific embedding or reranker model."""
    # ── Chunking ──
    max_tokens: int
    min_tokens: int
    dimensions: int
    overlap: int
    category: Literal["plain_text", "semantic_markdown", "reranker"]

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
# ═══════════════════════════════════════════════════════════════════════════

PRESETS: dict[str, ModelPreset] = {
    # ──────────────────────────────────────────────────────────────────
    # Embedding — PyTorch
    # ──────────────────────────────────────────────────────────────────
    "intfloat/e5-small-v2": ModelPreset(
        max_tokens=480,
        min_tokens=120,
        dimensions=384,
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
        dimensions=1024,
        overlap=5,
        category="semantic_markdown",
        needs_prefix=False,
        model_name="BAAI/bge-m3",
    ),
    "GritLM/GritLM-7B": ModelPreset(
        max_tokens=3800,
        min_tokens=300,
        dimensions=4096,
        overlap=4,
        category="semantic_markdown",
        needs_prefix=False,
        is_instruction_tuned=True,
        instruction_template="|<|user|>\nRepresent this passage for retrieval: {text}\n<|<|assistant|>\n",
        model_name="GritLM/GritLM-7B",
    ),

    # ──────────────────────────────────────────────────────────────────
    # Embedding — ONNX
    # ──────────────────────────────────────────────────────────────────
    "intfloat/e5-small-v2-onnx": ModelPreset(
        max_tokens=480,
        min_tokens=120,
        dimensions=384,
        overlap=3,
        category="plain_text",
        needs_prefix=True,
        passage_prefix="passage: ",
        query_prefix="query: ",
        model_name="intfloat/e5-small-v2-onnx",
    ),
    "BAAI/bge-m3-onnx": ModelPreset(
        max_tokens=8000,
        min_tokens=400,
        dimensions=1024,
        overlap=5,
        category="semantic_markdown",
        needs_prefix=False,
        model_name="BAAI/bge-m3-onnx",
    ),

    # ──────────────────────────────────────────────────────────────────
    # Embedding — API
    # ──────────────────────────────────────────────────────────────────
    "openai/text-embedding-3-small": ModelPreset(
        max_tokens=8000,
        min_tokens=400,
        dimensions=1536,
        overlap=5,
        category="semantic_markdown",
        needs_prefix=False,
        model_name="openai/text-embedding-3-small",
    ),
    "openai/text-embedding-3-large": ModelPreset(
        max_tokens=8000,
        min_tokens=400,
        dimensions=3072,
        overlap=5,
        category="semantic_markdown",
        needs_prefix=False,
        model_name="openai/text-embedding-3-large",
    ),
    "cohere/embed-english-v3.0": ModelPreset(
        max_tokens=512,
        min_tokens=100,
        dimensions=1024,
        overlap=3,
        category="plain_text",
        needs_prefix=False,
        model_name="cohere/embed-english-v3.0",
    ),
    "voyage/voyage-3-lite": ModelPreset(
        max_tokens=4000,
        min_tokens=200,
        dimensions=512,
        overlap=4,
        category="semantic_markdown",
        needs_prefix=False,
        model_name="voyage/voyage-3-lite",
    ),

    # ──────────────────────────────────────────────────────────────────
    # Reranker — Local
    # ──────────────────────────────────────────────────────────────────
    "cross-encoder/ms-marco-MiniLM-L-12-v2": ModelPreset(
        max_tokens=512,
        min_tokens=10,
        dimensions=384,
        overlap=0,
        category="reranker",
        needs_prefix=False,
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
    ),

    # ──────────────────────────────────────────────────────────────────
    # Reranker — ONNX
    # ──────────────────────────────────────────────────────────────────
    "cross-encoder/ms-marco-MiniLM-L-6-v2-onnx": ModelPreset(
        max_tokens=512,
        min_tokens=10,
        dimensions=384,
        overlap=0,
        category="reranker",
        needs_prefix=False,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2-onnx",
    ),

    # ──────────────────────────────────────────────────────────────────
    # Reranker — API
    # ──────────────────────────────────────────────────────────────────
    "cohere/rerank-english-v3.0": ModelPreset(
        max_tokens=512,
        min_tokens=10,
        dimensions=1,  # Reranker returns scalar score, not a vector
        overlap=0,
        category="reranker",
        needs_prefix=False,
        model_name="cohere/rerank-english-v3.0",
    ),
}

# Fallback for unknown models
DEFAULT_PRESET = ModelPreset(
    max_tokens=480,
    min_tokens=120,
    dimensions=384,
    overlap=3,
    category="plain_text",
    model_name="unknown",
)


def get_preset(model_key: Optional[str] = None) -> ModelPreset:
    """
    Returns the full ModelPreset for a given model_key.

    Args:
        model_key: Canonical model_id (e.g., 'BAAI/bge-m3') or None.
                   If None, returns DEFAULT_PRESET.

    Returns:
        ModelPreset matching the model, or DEFAULT_PRESET if unknown.
    """
    if model_key is None:
        return DEFAULT_PRESET

    if model_key in PRESETS:
        return PRESETS[model_key]

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
        return min(requested_size, 16)

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

    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    cpu_usage = psutil.cpu_percent(interval=0.1)

    if available_gb < 2.0 or cpu_usage > 80:
        return 16
    if available_gb > 4.0:
        return 64
    return requested_size


def get_model_batch_size(preset: Optional[ModelPreset] = None,model_key = "", requested_size: int = 32) -> int:
    """
    Batch size adjusted for the currently active model's memory footprint.
    """
    if preset is None and model_key is not None and model_key in PRESETS:
        preset = get_preset(model_key)

    base_size = get_optimal_batch_size(requested_size)
    model_name = preset.model_name.lower()

    if "gritlm-7b" in model_name or "gritlm" in model_name:
        return max(8, base_size // 4)
    elif "bge-m3" in model_name or "bge" in model_name:
        return base_size
    elif "e5-small" in model_name or "e5" in model_name:
        return base_size * 2
    elif "minilm" in model_name:
        return base_size * 4

    return base_size