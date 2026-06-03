import shutil
from typing import Optional
from pathlib import Path
from src.core.Models.lib import REQUIRED_FILES

class ModelPathResolver:
    """Unified path resolution for all model categories and formats."""

    def __init__(self, root: Path):
        self.root = root

    def model_path(self, spec, format: Optional[str] = None) -> Path:
        """Return the canonical nested path for a model."""
        if spec.loader.startswith("api_"):
            return Path("__api__")

        if spec.loader == "spacy":
            base = self.root / spec.category / spec.model_id
            return base / format if format else base

        model_id = spec.model_id
        if "/" in model_id:
            org, name = model_id.split("/", 1)
            base = self.root / spec.category / org / name
        else:
            base = self.root / spec.category / model_id

        if format:
            return base / format
        return base

    def legacy_model_path(self, spec) -> Path:
        """Old flat structure for backward compatibility."""
        if spec.loader == "spacy":
            return self.root / spec.category / spec.model_id
        return self.root / spec.category / spec.model_id.split("/")[-1]

    @staticmethod
    def infer_format(spec) -> str:
        """Map spec loader to a storage format."""
        if spec.loader in ("onnx_embedding", "onnx_reranker"):
            return "onnx"
        if spec.loader == "spacy":
            return "spacy"
        return "pytorch"




class ModelVerifier:
    """Verify model completeness and correctness."""

    def __init__(self, resolver: ModelPathResolver):
        self.resolver = resolver

    def is_downloaded(self, spec, key: str, format: Optional[str] = None) -> bool:
        """Check whether model is present and valid."""
        if spec.disk_mb == 0 or spec.loader.startswith("api_"):
            return True

        if format is None:
            format = self.resolver.infer_format(spec)

        path = self.resolver.model_path(spec, format)
        if self._verify_at_path(path, spec):
            return True

        legacy = self.resolver.legacy_model_path(spec)
        return self._verify_at_path(legacy, spec)

    def _verify_at_path(self, path: Path, spec) -> bool:
        if not path.exists():
            return False
        if (path / ".incomplete").exists():
            return False

        if spec.loader == "spacy":
            return self._verify_spacy_model(path)

        if spec.loader in ("onnx_embedding", "onnx_reranker"):
            return any(path.glob("*.onnx"))

        required = REQUIRED_FILES.get(spec.loader)
        if required is None:
            return True

        for req_file in required:
            if not (path / req_file).exists():
                return False

        if spec.loader not in ["spacy", "none", "onnx_embedding", "onnx_reranker"]:
            has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
            if not has_weights:
                return False

        self._strip_bloat(path)
        return True

    def _verify_spacy_model(self, path: Path) -> bool:
        essential = ["tokenizer", "meta.json", "config.cfg", "vocab"]
        for component in essential:
            comp_path = path / component
            if not comp_path.exists():
                return False
            if comp_path.is_dir():
                if not any(comp_path.iterdir()):
                    return False
            elif comp_path.is_file():
                if comp_path.stat().st_size == 0:
                    return False
        try:
            config_path = path / "config.cfg"
            if config_path.exists():
                content = config_path.read_text()
                if "tokenizer" not in content.lower():
                    return False
        except Exception:
            return False
        return True

    def _strip_bloat(self, path: Path) -> None:
        bloat_dirs = [
            "onnx", "coreml", "rust_model", "flax_model", "tf_model",
            "checkpoint-1", "eval", "runs",
        ]
        for name in bloat_dirs:
            bloat_path = path / name
            if bloat_path.exists() and bloat_path.is_dir():
                shutil.rmtree(bloat_path, ignore_errors=True)

