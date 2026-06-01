"""
src/core/Plugins/ModelDownloader.py
-----------------------------------
Downloads models to ~/.darwin/models/{category}/{org}/{model}/{format}/
Uses hf_hub_download for ONNX (precise files), snapshot_download for PyTorch (full repo).
"""

from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from src.lib.AppPaths import default_models_dir
from src.core.Models.ModelRegistry import get_spec


class ModelDownloader:
    def __init__(self, cache_root: Optional[Path] = None):
        self.cache_root = cache_root or default_models_dir()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _get_model_dir(self, model_id: str, category: str, format: str) -> Path:
        """
        ~/.darwin/models/{category}/{org}/{model}/{format}/
        e.g., ~/.darwin/models/embedding/BAAI/bge-m3/onnx/
              ~/.darwin/models/embedding/BAAI/bge-m3/pytorch/
              ~/.darwin/models/vision/HuggingFaceTB/SmolVLM-256M-Instruct/pytorch/
              ~/.darwin/models/tokenizer/en_core_web_md/spacy/
        """
        org, name = model_id.split("/", 1)
        return self.cache_root / category / org / name / format

    def is_cached(self, key: str, format: str = "onnx") -> bool:
        """Check if model is cached at the nested path."""
        spec = get_spec(key)
        path = self._get_model_dir(spec.model_id, spec.category, format)

        if format == "onnx":
            return any(path.glob("*.onnx"))
        elif format == "pytorch":
            return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
        elif format == "spacy":
            return (path / "config.cfg").exists() and (path / "meta.json").exists()
        return False

    def download_onnx(self, key: str, repo_id: Optional[str] = None) -> Path:
        """
        Downloads ONLY ONNX files needed (~133MB instead of 500MB).
        Uses hf_hub_download for precise file selection.
        """
        spec = get_spec(key)
        target = self._get_model_dir(spec.model_id, spec.category, "onnx")
        target.mkdir(parents=True, exist_ok=True)

        required_files = ["model.onnx", "tokenizer.json", "config.json"]
        optional_files = [
            "vocab.txt", "tokenizer_config.json", "special_tokens_map.json",
            "preprocessor_config.json",
        ]

        repo_candidates = []
        if repo_id:
            repo_candidates.append(repo_id)
        repo_candidates.append(f"Xenova/{spec.model_id.split('/')[-1]}")
        repo_candidates.append(spec.model_id)

        last_error = None
        for repo in repo_candidates:
            try:
                for filename in required_files:
                    try:
                        hf_hub_download(
                            repo_id=repo,
                            filename=filename,
                            local_dir=target,
                            local_dir_use_symlinks=False,
                        )
                    except Exception:
                        if filename == "model.onnx":
                            for subpath in ["onnx/model.onnx", "model.onnx"]:
                                try:
                                    hf_hub_download(
                                        repo_id=repo,
                                        filename=subpath,
                                        local_dir=target,
                                        local_dir_use_symlinks=False,
                                    )
                                    downloaded = target / subpath
                                    if downloaded.exists():
                                        downloaded.rename(target / "model.onnx")
                                    break
                                except Exception:
                                    continue
                            else:
                                raise RuntimeError(f"model.onnx not found in {repo}")
                        else:
                            raise

                for filename in optional_files:
                    try:
                        hf_hub_download(
                            repo_id=repo,
                            filename=filename,
                            local_dir=target,
                            local_dir_use_symlinks=False,
                        )
                    except Exception:
                        pass

                if (target / "model.onnx").exists():
                    return target

            except RepositoryNotFoundError:
                continue
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(
            f"Could not download ONNX model {key} (tried: {repo_candidates}). "
            f"Last error: {last_error}"
        )

    def download_pytorch(self, key: str) -> Path:
        """Downloads full PyTorch model repo."""
        spec = get_spec(key)
        target = self._get_model_dir(spec.model_id, spec.category, "pytorch")
        target.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=spec.model_id,
            local_dir=target,
        )
        return target

    def download_spacy(self, key: str) -> Path:
        """Placeholder — spaCy uses ModelStore's custom download."""
        spec = get_spec(key)
        target = self._get_model_dir(spec.model_id, spec.category, "spacy")
        return target

    def delete_model(self, key: str, format: str = "onnx") -> bool:
        import shutil
        spec = get_spec(key)
        path = self._get_model_dir(spec.model_id, spec.category, format)
        if path.exists():
            shutil.rmtree(path)
            return True
        return False