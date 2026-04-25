# src/core/Models/ModelStore.py

from pathlib import Path
import shutil
from src.core.Models.ModelRegistry import get_spec, REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
# Required files per loader type (minimum for inference)
# ═══════════════════════════════════════════════════════════════════════════

_REQUIRED_FILES: dict[str, list[str]] = {
    "sentence_transformers": [
        "config.json",
        "tokenizer_config.json",
        "sentence_bert_config.json",
    ],
    "cross_encoder": [
        "config.json",
        "tokenizer_config.json",
    ],
    "transformers": [
        "config.json",
        "preprocessor_config.json",
         "tokenizer_config.json",
        "generation_config.json", 
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# Download filters — ONLY get what's needed for PyTorch inference
# ═══════════════════════════════════════════════════════════════════════════

# Patterns we WANT — everything essential for inference

# src/core/Models/ModelStore.py - Fix the _REQUIRED_FILES and _ALLOW_PATTERNS

_REQUIRED_FILES: dict[str, list[str]] = {
    "sentence_transformers": [
        "config.json",
        "tokenizer_config.json",
        "sentence_bert_config.json",
    ],
    "cross_encoder": [
        "config.json",
        "tokenizer_config.json",
    ],
    "transformers": [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",  # ADD THIS - transformers models need tokenizer too
        "generation_config.json",  # ADD THIS - needed for text generation
    ],
}

# Fix the _ALLOW_PATTERNS to include everything transformers needs
_ALLOW_PATTERNS = [
    # ── Weights (PyTorch only, skip duplicates) ──
    "*.safetensors",
    "pytorch_model*.bin",
    
    # ── Config files ──
    "config.json",
    "preprocessor_config.json",  # CRITICAL: This was already here
    "generation_config.json",     # ADD THIS
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    
    # ── Tokenizer / vocab ──
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "added_tokens.json",
    
    # ── Image processor files ──
    "image_processor.json",      # ADD THIS - critical for vision models
    "processor_config.json",     # ADD THIS - some models use this
    
    # ── Sentence Transformers specific ──
    "modules.json",
    "sentence_bert_config.json",
    
    # ── Model index (for sharded models) ──
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    
    # ── Chat template ──
    "chat_template.json",        # ADD THIS - needed for vision-language models
]


# Patterns to AGGRESSIVELY IGNORE — duplicates, other frameworks, docs
_IGNORE_PATTERNS = [
    # ── ONNX — entire folder is a duplicate of the model ──
    "onnx/**",
    "onnx/*",
    "*.onnx",
    
    # ── TensorFlow duplicates ──
    "*.h5",
    "*.keras",
    "tf_model.h5",
    "tf_model/**",
    
    # ── Flax / JAX duplicates ──
    "*.msgpack",
    "flax_model.msgpack",
    "flax_model/**",
    
    # ── Rust / Candle duplicates ──
    "*.ot",
    "rust_model.ot",
    "rust_model/**",
    
    # ── GGUF / llama.cpp — not needed for transformers ──
    "*.gguf",
    "ggml-model-*",
    
    # ── CoreML (Apple) ──
    "coreml/**",
    "coreml/*",
    "*.mlmodel",
    "*.mlpackage",
    
    # ── Documentation / metadata ──
    "README.md",
    "*.md",
    ".gitattributes",
    ".gitignore",
    ".git/**",
    
    # ── Training artifacts ──
    "checkpoint-*/**",
    "*.ckpt",
    "optimizer.pt",
    "scheduler.pt",
    
    # ── Evaluation / testing ──
    "eval/**",
    "test/**",
    "tests/**",
    "*.ipynb",
    
    # ── Logs / cache ──
    "*.log",
    "*.txt",                   # Caution: also blocks vocab.txt — handled by allow_patterns
    "runs/**",
]


class ModelStore:
    def __init__(self, models_dir: Path):
        self.root = models_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def model_path(self, key: str) -> Path:
        spec = get_spec(key)
        return self.root / spec.category / spec.model_id.split("/")[-1]

    # ═══════════════════════════════════════════════════════════════════════
    # Integrity checks
    # ═══════════════════════════════════════════════════════════════════════

    def is_downloaded(self, key: str) -> bool:
        """
        Check if model is fully downloaded and valid.

        Checks:
        1. Folder exists
        2. No .incomplete marker
        3. All required config files present
        4. At least one weight file exists
        """
        path = self.model_path(key)

        if not path.exists():
            return False

        if (path / ".incomplete").exists():
            return False

        # Clean up bloat if present (silent — user doesn't need to know)
        self._strip_bloat(path)

        # Required config files
        spec     = get_spec(key)
        required = _REQUIRED_FILES.get(spec.loader, ["config.json"])
        if any(not (path / f).exists() for f in required):
            return False

        # At least one weight file
        has_weights = (
            any(path.glob("*.safetensors"))
            or any(path.glob("pytorch_model*.bin"))
        )
        if not has_weights:
            return False

        return True

    def _strip_bloat(self, path: Path) -> None:
        """
        Silently remove known bloat folders if they exist.
        Called during integrity checks so previously-downloaded models
        get cleaned up automatically on next app launch.
        """
        bloat_dirs = [
            "onnx",
            "coreml",
            "rust_model",
            "flax_model",
            "tf_model",
            "checkpoint-1",
            "eval",
            "runs",
        ]
        for name in bloat_dirs:
            bloat_path = path / name
            if bloat_path.exists() and bloat_path.is_dir():
                shutil.rmtree(bloat_path)

        # Also strip individual bloat files
        bloat_globs = [
            "*.msgpack",
            "*.h5",
            "*.gguf",
            "*.onnx",
            "*.ot",
            "*.ckpt",
        ]
        for pattern in bloat_globs:
            for f in path.glob(pattern):
                if f.is_file():
                    f.unlink()

    # ═══════════════════════════════════════════════════════════════════════
    # Download (minimal by default)
    # ═══════════════════════════════════════════════════════════════════════

        
    def download(
        self,
        key: str,
        force: bool = False,
        minimal: bool = True,
    ) -> Path:
        """Download a model with resume support and integrity checks."""
        from huggingface_hub import snapshot_download

        spec = get_spec(key)
        path = self.model_path(key)

        # Already downloaded and valid
        if self.is_downloaded(key) and not force:
            return path

        # Clean up for forced re-download
        if path.exists() and force:
            shutil.rmtree(path)

        path.mkdir(parents=True, exist_ok=True)
        incomplete_marker = path / ".incomplete"
        incomplete_marker.touch()

        try:
            # CRITICAL FIX: For vision models, ALWAYS download everything
            # The minimal patterns filter out files the processor needs
            if minimal and spec.category == "vision":
                print(f"   ⬇️  Downloading {spec.model_id} (full download — vision models need all files)...")
                snapshot_download(
                    repo_id=spec.model_id,
                    local_dir=str(path),
                    resume_download=True,
                    # Don't use allow_patterns/ignore_patterns for vision models
                )
            elif minimal:
                print(f"   ⬇️  Downloading {spec.model_id} (inference-only — skipping ONNX/TF/Flax)...")
                snapshot_download(
                    repo_id=spec.model_id,
                    local_dir=str(path),
                    resume_download=True,
                    allow_patterns=_ALLOW_PATTERNS,
                    ignore_patterns=_IGNORE_PATTERNS,
                )
            else:
                print(f"   ⬇️  Downloading {spec.model_id} (full — including all formats)...")
                snapshot_download(
                    repo_id=spec.model_id,
                    local_dir=str(path),
                    resume_download=True,
                )

            # Success — remove marker
            incomplete_marker.unlink()

            # Strip bloat but KEEP all config files
            self._strip_bloat(path)

            # Final integrity check
            if not self.is_downloaded(key):
                spec     = get_spec(key)
                required = _REQUIRED_FILES.get(spec.loader, ["config.json"])
                missing  = [f for f in required if not (path / f).exists()]
                has_weights = (
                    any(path.glob("*.safetensors"))
                    or any(path.glob("pytorch_model*.bin"))
                )
                problems = []
                if missing:
                    problems.append(f"missing config: {', '.join(missing)}")
                if not has_weights:
                    problems.append("no weight files")
                raise RuntimeError(
                    f"Download completed but integrity check failed "
                    f"({'; '.join(problems)}). Delete {path} and retry."
                )

            # Report actual size
            size = self.installed_size(key)
            print(f"   ✅ {key} ready ({size})")
            return path

        except KeyboardInterrupt:
            print(f"\n   ⚠️  Download interrupted. Run again to resume.")
            raise

        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            print(f"      Partial files kept at {path}")
            print(f"      Run again to resume, or delete this folder to start fresh.")
            raise

    def ensure(self, key: str) -> Path:
        """Download if not present, verify integrity, return path."""
        if not self.is_downloaded(key):
            self.download(key)

        path = self.model_path(key)

        # Double-check after download
        if not self.is_downloaded(key):
            spec     = get_spec(key)
            required = _REQUIRED_FILES.get(spec.loader, ["config.json"])
            missing  = [f for f in required if not (path / f).exists()]
            raise RuntimeError(
                f"Model '{key}' failed integrity check. "
                f"Missing: {', '.join(missing) if missing else 'weight files'}. "
                f"Delete {path} and retry."
            )

        return path

    def delete(self, key: str) -> None:
        """Free disk space by removing a model."""
        path = self.model_path(key)
        if path.exists():
            shutil.rmtree(path)

    def list_installed(self) -> list[str]:
        return [k for k in REGISTRY if self.is_downloaded(k)]

    def installed_size(self, key: str) -> str:
        """Human-readable size of installed model."""
        path = self.model_path(key)
        if not path.exists():
            return "0 MB"

        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        for unit in ["B", "KB", "MB", "GB"]:
            if total < 1024:
                return f"{total:.1f} {unit}"
            total /= 1024
        return f"{total:.1f} TB"