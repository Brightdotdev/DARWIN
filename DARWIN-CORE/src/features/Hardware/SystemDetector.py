# src/core/Hardware/SystemDetector.py

import psutil
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# Registry integration – single source of truth for model metadata
# ═══════════════════════════════════════════════════════════════════════════
from src.core.Models.ModelRegistry import get_spec


@dataclass
class SystemInfo:
    ram_gb: int
    cpu_cores: int
    has_gpu: bool
    gpu_name: str = ""
    disk_free_gb: float = 0.0
    device: str = "cpu"
    dtype: str = "float32"

    @property
    def tier(self) -> str:
        """Auto-detect best tier for this hardware."""
        if self.ram_gb >= 32:
            return "whale"
        elif self.ram_gb >= 16:
            return "orca"
        elif self.ram_gb >= 8:
            return "dolphin"
        else:
            return "minnow"

    def can_run(self, spec) -> bool:
        """Can this system comfortably run the given ModelSpec?"""
        return self.ram_gb >= spec.ram_gb

    def has_disk_space(self, spec) -> bool:
        """Enough free disk for this model?"""
        return self.disk_free_gb >= (spec.disk_mb / 1024) + 1

    def has_disk_space_for_all(self, specs: list) -> bool:
        """Enough free disk for a list of models?"""
        total_gb = sum(s.disk_mb for s in specs) / 1024
        return self.disk_free_gb >= total_gb + 1


def detect_system() -> SystemInfo:
    """Detect system hardware including GPU availability. No torch dependency."""
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.home()))

    # GPU detection without torch
    has_gpu = False
    gpu_name = ""
    device = "cpu"
    dtype = "float32"

    # 1. Try GPUtil
    try:
        import gputil
        gpus = gputil.getGPUs()
        if gpus:
            has_gpu = True
            gpu_name = gpus[0].name
            device = "cuda"
            dtype = "float16"
    except Exception:
        pass

    # 2. Windows WMI fallback
    if not has_gpu and platform.system() == "Windows":
        try:
            import wmi
            c = wmi.WMI()
            for gpu in c.Win32_VideoController():
                if gpu.AdapterRAM and int(gpu.AdapterRAM) > 512 * 1024 * 1024:
                    has_gpu = True
                    gpu_name = gpu.Name
                    device = "cuda"
                    dtype = "float16"
                    break
        except Exception:
            pass

    # 3. macOS Apple Silicon
    if not has_gpu and platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=5
            )
            if "Apple" in result.stdout:
                has_gpu = True
                gpu_name = "Apple Silicon GPU"
                device = "mps"
                dtype = "float16"
        except Exception:
            pass

    return SystemInfo(
        ram_gb=int(ram.total / (1024**3)),
        cpu_cores=psutil.cpu_count(logical=True),
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        disk_free_gb=disk.free / (1024**3),
        device=device,
        dtype=dtype,
    )


def _detect_api_keys() -> dict[str, bool]:
    """Check which API keys are available in environment."""
    import os

    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "cohere": bool(os.getenv("COHERE_API_KEY")),
        "voyage": bool(os.getenv("VOYAGE_API_KEY")),
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "together": bool(os.getenv("TOGETHER_API_KEY")),
    }


def recommend_defaults(system: SystemInfo) -> dict[str, str]:
    """
    Recommend models based on system hardware + available API keys.
    All model references are validated against the central ModelRegistry.
    """
    recommendations = {}
    api_keys = _detect_api_keys()

    # ═══════════════════════════════════════════════════════════════
    # Embedding – preference order: API → high‑end local → ONNX → small ONNX
    # ═══════════════════════════════════════════════════════════════
    embedding_candidates = []

    if api_keys["openai"]:
        embedding_candidates.append("openai/text-embedding-3-small")
    if api_keys["openrouter"]:
        embedding_candidates.append("openrouter/openai/text-embedding-3-small")
    if api_keys["cohere"]:
        embedding_candidates.append("cohere/embed-english-v3.0")
    if api_keys["voyage"]:
        embedding_candidates.append("voyage/voyage-3-lite")

    # GPU‑accelerated heavy lifter (only if RAM >= 32 GB and a real GPU)
    if system.ram_gb >= 32 and system.has_gpu:
        embedding_candidates.append("GritLM/GritLM-7B")

    # ONNX BGE‑M3 – sweet spot for most users
    if system.ram_gb >= 8:
        embedding_candidates.append("BAAI/bge-m3-onnx")

    # Ultimate fallback – runs on anything with 2 GB RAM
    embedding_candidates.append("intfloat/e5-small-v2-onnx")

    for model_id in embedding_candidates:
        try:
            spec = get_spec(model_id)
        except KeyError:
            continue  # model removed from registry – skip gracefully

        # Extra GPU requirement for GritLM (not encoded in spec)
        if model_id == "GritLM/GritLM-7B" and not system.has_gpu:
            continue

        if not spec.loader.startswith("api_"):
            if not system.can_run(spec) or not system.has_disk_space(spec):
                continue

        recommendations["embedding"] = model_id
        break

    # ═══════════════════════════════════════════════════════════════
    # Reranker – Cohere API (or OpenRouter) → local MiniLM cross‑encoder
    # ═══════════════════════════════════════════════════════════════
    reranker_candidates = []
    if api_keys["cohere"]:
        reranker_candidates.append("cohere/rerank-english-v3.0")
    if api_keys["openrouter"]:
        reranker_candidates.append("openrouter/cohere/rerank-english-v3.0")
    reranker_candidates.append("cross-encoder/ms-marco-MiniLM-L-12-v2")

    for model_id in reranker_candidates:
        try:
            spec = get_spec(model_id)
        except KeyError:
            continue
        if not spec.loader.startswith("api_"):
            if not system.can_run(spec) or not system.has_disk_space(spec):
                continue
        recommendations["reranker"] = model_id
        break

    # ═══════════════════════════════════════════════════════════════
    # Vision – API first, then local with GPU, then tiny local
    # ═══════════════════════════════════════════════════════════════
    vision_candidates = []
    if api_keys["openai"]:
        vision_candidates.append("openai/gpt-4o")
    if api_keys["openrouter"]:
        vision_candidates.append("openrouter/meta-llama/llama-4-maverick")
    if api_keys["gemini"]:
        vision_candidates.append("gemini/gemini-2.0-flash")

    if system.ram_gb >= 16 and system.has_gpu:
        vision_candidates.append("vikhyatk/moondream2")
    if system.ram_gb >= 4:
        vision_candidates.append("HuggingFaceTB/SmolVLM-256M-Instruct")

    for model_id in vision_candidates:
        try:
            spec = get_spec(model_id)
        except KeyError:
            continue
        if not spec.loader.startswith("api_"):
            if not system.can_run(spec) or not system.has_disk_space(spec):
                continue
        recommendations["vision"] = model_id
        break

    # No vision model found at all
    if "vision" not in recommendations:
        recommendations["vision"] = None

    # ═══════════════════════════════════════════════════════════════
    # Tokenizer – large spaCy model if plenty of RAM+disk, else medium
    # ═══════════════════════════════════════════════════════════════
    # Heuristic: prefer en_core_web_lg only on comfortably large machines
    if system.ram_gb >= 16 and system.disk_free_gb > 2:
        tokenizer_id = "en_core_web_lg"
    else:
        tokenizer_id = "en_core_web_md"

    # Final validation – if the chosen tokenizer can’t actually run, fall back to medium
    try:
        tok_spec = get_spec(tokenizer_id)
        if not system.can_run(tok_spec) or not system.has_disk_space(tok_spec):
            tokenizer_id = "en_core_web_md"
    except KeyError:
        tokenizer_id = "en_core_web_md"

    recommendations["tokenizer"] = tokenizer_id
    return recommendations