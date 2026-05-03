# src/core/Hardware/SystemDetector.py

import psutil
from dataclasses import dataclass
from pathlib import Path


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
            return "overkill"
        elif self.ram_gb >= 16:
            return "large"
        elif self.ram_gb >= 8:
            return "medium"
        else:
            return "light"

    def can_run(self, spec) -> bool:
        """Can this system comfortably run the given ModelSpec?"""
        return self.ram_gb >= spec.ram_gb

    def has_disk_space(self, spec) -> bool:
        """Enough free disk for this model?"""
        return self.disk_free_gb >= (spec.disk_mb / 1024) + 1  # +1GB buffer

    def has_disk_space_for_all(self, specs: list) -> bool:
        """Enough free disk for a list of models?"""
        total_gb = sum(s.disk_mb for s in specs) / 1024
        return self.disk_free_gb >= total_gb + 1


def detect_system() -> SystemInfo:
    """Detect system hardware including GPU availability."""
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.home()))

    # GPU detection
    has_gpu = False
    gpu_name = ""
    device = "cpu"
    dtype = "float32"

    try:
        import torch
        
        if torch.cuda.is_available():
            has_gpu = True
            gpu_name = torch.cuda.get_device_name(0)
            device = "cuda"
            dtype = "float16"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            has_gpu = True
            gpu_name = "Apple GPU (MPS)"
            device = "mps"
            dtype = "float16"
    except ImportError:
        # PyTorch not installed, CPU only
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



def _detect_api_junkies() -> dict[str, bool]:
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
    Recommend models based on system + available API keys.
    
    Priority order for each category:
        1. API keys detected → use best API model for that provider
        2. No API keys → use best local model for hardware
    
    Users can always override with: manager.switch_embedding("BAAI/bge-m3")
    """
    recommendations = {}
    api_junkies = _detect_api_junkies()
    
    has_any_api = any(api_junkies.values())
    
    # ═══════════════════════════════════════════════════════════════
    # Reranker
    # ═══════════════════════════════════════════════════════════════
    if api_junkies.get("cohere"):
        recommendations["reranker"] = "cohere/rerank-english-v3.0"
    elif api_junkies.get("openrouter"):
        recommendations["reranker"] = "openrouter/cohere/rerank-english-v3.0"
    else:
        recommendations["reranker"] = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # ═══════════════════════════════════════════════════════════════
    # Embedding
    # ═══════════════════════════════════════════════════════════════
    if api_junkies.get("openai"):
        recommendations["embedding"] = "openai/text-embedding-3-small"
    elif api_junkies.get("openrouter"):
        # OpenRouter can route to any embedding provider
        recommendations["embedding"] = "openrouter/openai/text-embedding-3-small"
    elif api_junkies.get("cohere"):
        recommendations["embedding"] = "cohere/embed-english-v3.0"
    elif api_junkies.get("voyage"):
        recommendations["embedding"] = "voyage/voyage-3-lite"
    elif system.ram_gb >= 32 and system.has_gpu:
        recommendations["embedding"] = "GritLM/GritLM-7B"
    elif system.ram_gb >= 16:
        recommendations["embedding"] = "BAAI/bge-m3"
    else:
        recommendations["embedding"] = "intfloat/e5-small-v2"
    
    # ═══════════════════════════════════════════════════════════════
    # Vision
    # ═══════════════════════════════════════════════════════════════
    if api_junkies.get("openai"):
        recommendations["vision"] = "openai/gpt-4o"
    elif api_junkies.get("openrouter"):
        recommendations["vision"] = "openrouter/meta-llama/llama-4-maverick"
    elif api_junkies.get("gemini"):
        recommendations["vision"] = "gemini/gemini-2.0-flash"
    elif system.ram_gb >= 16 and system.has_gpu:
        recommendations["vision"] = "vikhyatk/moondream2"
    elif system.ram_gb >= 4:
        recommendations["vision"] = "HuggingFaceTB/SmolVLM-256M-Instruct"
    else:
        recommendations["vision"] = None
    
    # ═══════════════════════════════════════════════════════════════
    # Tokenizer (always local spaCy — no API equivalent)
    # ═══════════════════════════════════════════════════════════════
    if system.ram_gb >= 16 and system.disk_free_gb > 2:
        recommendations["tokenizer"] = "en_core_web_lg"
    else:
        recommendations["tokenizer"] = "en_core_web_md"
    
    return recommendations