# src/core/Models/ModelStore.py

from pathlib import Path
import shutil
import json
import urllib.request
import tarfile
import tempfile
import zipfile
import time
from typing import Optional, Callable
from dataclasses import dataclass

from src.core.Models.ModelRegistry import get_spec, REGISTRY
from src.core.Models.lib import (IGNORE_PATTERNS, ALLOW_PATTERNS, REQUIRED_FILES, SPACY_URLS, tarball_urls)
from src.lib.AppPaths import default_models_dir


# ── Singleton management ─────────────────────────────────────────────────
_STORE_INSTANCE = None

def get_model_store(progress_callback=None) -> 'ModelStore':
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        _STORE_INSTANCE = ModelStore(progress_callback=progress_callback)
    return _STORE_INSTANCE
# ────────────────────────────────────────────────────────────────────────


@dataclass
class DownloadProgress:
    bytes_downloaded: int = 0
    total_bytes: int = 0
    
    @property
    def percentage(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return min(100.0, (self.bytes_downloaded / self.total_bytes) * 100)


class DownloadError(Exception):
    def __init__(self, message: str, hint: str = "", recoverable: bool = True):
        self.hint = hint
        self.recoverable = recoverable
        super().__init__(message)


class IntegrityError(Exception):
    pass


class ModelStore:
    def __init__(self, progress_callback: Optional[Callable[[DownloadProgress], None]] = None):
        self.root = default_models_dir()
        self.progress_callback = progress_callback
        self._active_downloads: set[str] = set()
        self.SPACY_URLS = SPACY_URLS

    def model_path(self, key: str) -> Path:
        """Get the filesystem path for a model."""
        spec = get_spec(key)
        if spec.loader == "spacy":
            return self.root / spec.category / spec.model_id
        return self.root / spec.category / spec.model_id.split("/")[-1]

    # ═══════════════════════════════════════════════════════════════════════
    # Integrity Verification
    # ═══════════════════════════════════════════════════════════════════════

    def is_downloaded(self, key: str) -> bool:
        """Check if model is fully downloaded and valid."""
        
        spec = get_spec(key)
        
        # API models are always "downloaded" — nothing to download
        if spec.disk_mb == 0:
            return True
        
        # Check for active download
        if key in self._active_downloads:
            return False
        
        spec = get_spec(key)
        path = self.model_path(key)
        
        if not path.exists():
            return False
        
        # Check for incomplete marker
        if (path / ".incomplete").exists():
            return False
        
        # spaCy-specific check
        if spec.loader == "spacy":
            return self._verify_spacy_model(path)
        
        # Generic model check
        required = REQUIRED_FILES.get(spec.loader)
        if required is None:
            return True
        
        for req_file in required:
            if not (path / req_file).exists():
                return False
        
        # Check for weight files
        if spec.loader not in ["spacy", "none"]:
            has_weights = (
                any(path.glob("*.safetensors")) or 
                any(path.glob("pytorch_model*.bin"))
            )
            if not has_weights:
                return False
        
        # Strip bloat files
        self._strip_bloat(path)
        return True
    
    def _verify_spacy_model(self, path: Path) -> bool:
        """Verify spaCy model has minimum required components."""
        essential = [
            "tokenizer",
            "meta.json",
            "config.cfg",
            "vocab",
        ]
        
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
        
        # Check config.cfg is valid
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
        """Remove unnecessary files to save disk space."""
        bloat_dirs = [
            "onnx", "coreml", "rust_model", "flax_model", "tf_model",
            "checkpoint-1", "eval", "runs",
        ]
        for name in bloat_dirs:
            bloat_path = path / name
            if bloat_path.exists() and bloat_path.is_dir():
                shutil.rmtree(bloat_path, ignore_errors=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Download System
    # ═══════════════════════════════════════════════════════════════════════

    def download(self, key: str, force: bool = False, minimal: bool = True) -> Path:
        """Download a model with fallback strategies."""
        from src.core.Logger.DarwinLogger import darwin, LogType
        
        spec = get_spec(key)
        
        
        if spec.disk_mb == 0:
            darwin.log(f"{key} is an API model — no download needed. Ensure your API key is set.", LogType.INFO)
            return
        
        spec = get_spec(key)
        path = self.model_path(key)
        
        if self.is_downloaded(key) and not force:
            return path
        
        if key in self._active_downloads:
            raise DownloadError(
                f"Download already in progress for {key}",
                "Wait for current download to complete",
                recoverable=True
            )
        
        self._active_downloads.add(key)
        
        try:
            if path.exists() and force:
                shutil.rmtree(path, ignore_errors=True)
            
            path.mkdir(parents=True, exist_ok=True)
            (path / ".incomplete").touch()
            
            try:
                if spec.loader == "spacy":
                    return self._download_spacy_model(spec, path, key)
                return self._download_huggingface(spec, path, key, minimal)
            except Exception as e:
                if isinstance(e, (DownloadError, IntegrityError)):
                    raise
                raise DownloadError(
                    f"Failed to download {key}: {str(e)}",
                    "Check internet connection and disk space"
                )
        finally:
            self._active_downloads.discard(key)
            if (path / ".incomplete").exists():
                if self.is_downloaded(key):
                    (path / ".incomplete").unlink(missing_ok=True)

    def _download_huggingface(self, spec, path: Path, key: str, minimal: bool) -> Path:
        """Download from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download
            
            print(f"   ⬇️  Downloading {spec.model_id}...")
            
            kwargs = {
                "repo_id": spec.model_id,
                "local_dir": str(path),
                "resume_download": True,
                "local_files_only": False,
            }
            
            if minimal and spec.category != "vision":
                kwargs.update({
                    "allow_patterns": ALLOW_PATTERNS,
                    "ignore_patterns": IGNORE_PATTERNS,
                })
            
            snapshot_download(**kwargs)
            
            if not self.is_downloaded(key):
                raise IntegrityError("Download completed but verification failed")
            
            size = self.installed_size(key)
            print(f"   ✅ {key} ready ({size})")
            return path
            
        except ImportError:
            raise DownloadError(
                "huggingface_hub package required",
                "Install with: pip install huggingface-hub",
                recoverable=False
            )
        except Exception as e:
            raise DownloadError(f"HuggingFace download failed: {e}")

    def _download_spacy_model(self, spec, target_path: Path, key: str) -> Path:
        """Download spaCy model using multiple strategies."""
        model_name = spec.model_id
        
        print(f"   ⬇️  Downloading {model_name} (minimal NLP components)...")
        
        strategies = [
            ("GitHub wheel download", lambda: self._download_spacy_wheel(model_name, target_path)),
            ("GitHub tarball fallback", lambda: self._download_spacy_tarball(model_name, target_path)),
        ]
        
        try:
            strategies.append(
                ("spaCy package download", lambda: self._download_spacy_via_package(model_name, target_path))
            )
        except ImportError:
            pass
        
        errors = []
        for strategy_name, strategy_fn in strategies:
            try:
                print(f"   📥 Attempting: {strategy_name}")
                result = strategy_fn()
                if result and self._verify_spacy_model(target_path):
                    self._optimize_spacy_model(target_path)
                    size = self.installed_size(key)
                    print(f"   ✅ {key} ready ({size})")
                    return target_path
                else:
                    errors.append(f"{strategy_name}: verification failed")
            except Exception as e:
                errors.append(f"{strategy_name}: {str(e)}")
                continue
        
        raise DownloadError(
            f"Failed to download spaCy model '{model_name}'.\n" +
            "Errors:\n  " + "\n  ".join(errors),
            "Check internet connection. You can also install manually:\n"
            f"  pip install {model_name}\n"
            f"  Then copy from site-packages to: {target_path}"
        )

    def _download_spacy_wheel(self, model_name: str, target_path: Path) -> bool:
        """Download spaCy model wheel from GitHub and extract model files."""
        urls = self.SPACY_URLS.get(model_name, [])
        if not urls:
            raise DownloadError(f"No download URLs for {model_name}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            wheel_path = tmp_path / f"{model_name}.whl"
            
            last_error = None
            for url in urls:
                try:
                    self._download_file(url, wheel_path)
                    
                    if not wheel_path.exists() or wheel_path.stat().st_size == 0:
                        continue
                    
                    print(f"   📦 Extracting wheel...")
                    extract_dir = tmp_path / "extracted"
                    extract_dir.mkdir(exist_ok=True)
                    
                    with zipfile.ZipFile(wheel_path, 'r') as zf:
                        zf.extractall(extract_dir)
                    
                    model_dir = self._find_model_in_wheel(extract_dir, model_name)
                    
                    if model_dir:
                        if target_path.exists():
                            shutil.rmtree(target_path, ignore_errors=True)
                        
                        print(f"   📋 Copying model files...")
                        self._copy_directory(model_dir, target_path)
                        
                        essential_files = ["config.cfg", "meta.json", "tokenizer"]
                        for f in essential_files:
                            if not (target_path / f).exists():
                                raise IntegrityError(f"Missing critical file: {f}")
                        
                        return True
                    else:
                        last_error = f"Model directory not found in wheel: {url}"
                        
                except Exception as e:
                    last_error = str(e)
                    continue
            
            raise DownloadError(f"Wheel download failed: {last_error}")

    def _download_spacy_tarball(self, model_name: str, target_path: Path) -> bool:
        """Fallback: Download tarball from GitHub releases."""
        urls = tarball_urls.get(model_name, [])
        if not urls:
            raise DownloadError(f"No tarball URLs for {model_name}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            for url in urls:
                try:
                    tarball_path = tmp_path / f"{model_name}.tar.gz"
                    self._download_file(url, tarball_path)
                    
                    print(f"   📦 Extracting tarball...")
                    with tarfile.open(tarball_path, 'r:gz') as tar:
                        tar.extractall(tmp_path)
                    
                    model_dir = self._find_model_in_extracted(tmp_path, model_name)
                    
                    if model_dir:
                        if target_path.exists():
                            shutil.rmtree(target_path, ignore_errors=True)
                        self._copy_directory(model_dir, target_path)
                        return True
                    
                except Exception as e:
                    print(f"   ⚠️  Tarball attempt failed: {e}")
                    continue
            
            raise DownloadError("Tarball download failed for all URLs")

    def _download_spacy_via_package(self, model_name: str, target_path: Path) -> bool:
        """Use spaCy package to download (requires spaCy installed)."""
        try:
            import spacy
            from spacy.cli.download import download as spacy_download
            
            spacy_download(model_name, direct=False)
            installed_path = Path(spacy.util.get_package_path(model_name))
            
            if target_path.exists():
                shutil.rmtree(target_path, ignore_errors=True)
            self._copy_directory(installed_path, target_path)
            
            return True
            
        except Exception as e:
            raise DownloadError(f"spaCy package download failed: {e}")

    def _find_model_in_wheel(self, extract_dir: Path, model_name: str) -> Optional[Path]:
        """Find the actual model directory inside an extracted wheel."""
        pkg_name = model_name.replace("-", "_")
        
        for candidate_name in [model_name, pkg_name]:
            for path in extract_dir.rglob(candidate_name):
                if path.is_dir():
                    if (path / "config.cfg").exists() and (path / "meta.json").exists():
                        return path
                    for subdir in path.iterdir():
                        if subdir.is_dir() and (subdir / "config.cfg").exists():
                            return subdir
        
        for config_path in extract_dir.rglob("config.cfg"):
            parent = config_path.parent
            if (parent / "meta.json").exists() and (parent / "tokenizer").exists():
                return parent
        
        for meta_path in extract_dir.rglob("meta.json"):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                if meta.get('name') == model_name or meta.get('lang') == 'en':
                    if (meta_path.parent / "tokenizer").exists():
                        return meta_path.parent
            except Exception:
                continue
        
        return None

    def _find_model_in_extracted(self, base_path: Path, model_name: str) -> Optional[Path]:
        """Find model directory in extracted tarball."""
        for config_path in base_path.rglob("config.cfg"):
            parent = config_path.parent
            if (parent / "meta.json").exists() and (parent / "tokenizer").exists():
                return parent
        
        for meta_path in base_path.rglob("meta.json"):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                if meta.get('name') == model_name:
                    if (meta_path.parent / "tokenizer").exists():
                        return meta_path.parent
            except Exception:
                continue
        
        return None

    def _download_file(self, url: str, dest_path: Path, max_retries: int = 3) -> None:
        """Download a file with retry logic."""
        for attempt in range(max_retries):
            try:
                print(f"   📥 Downloading... ({url[:60]}...)")
                
                req = urllib.request.Request(
                    url,
                    headers={
                        'User-Agent': 'Darwin-ModelManager/1.0',
                        'Accept': '*/*',
                    }
                )
                
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.geturl() != url:
                        print(f"   ↪️  Redirected to: {response.geturl()}")
                    
                    total_size = int(response.headers.get('Content-Length', 0))
                    downloaded = 0
                    
                    with open(dest_path, 'wb') as f:
                        while True:
                            chunk = response.read(65536)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if self.progress_callback and total_size > 0:
                                self.progress_callback(DownloadProgress(
                                    bytes_downloaded=downloaded,
                                    total_bytes=total_size
                                ))
                
                if dest_path.exists() and dest_path.stat().st_size > 0:
                    suffixes = [s.lower() for s in dest_path.suffixes]
                    is_zip = ".whl" in suffixes or ".zip" in suffixes
                    is_tar = ".tar" in suffixes or ".tgz" in suffixes

                    if is_zip:
                        with zipfile.ZipFile(dest_path, 'r') as zf:
                            if len(zf.namelist()) == 0:
                                raise Exception("Empty archive")
                    elif is_tar:
                        with tarfile.open(dest_path) as tar:
                            if len(tar.getmembers()) == 0:
                                raise Exception("Empty archive")
                    
                    print(f"   ✅ Downloaded {downloaded / 1024 / 1024:.1f} MB")
                    return
                    
                raise Exception("Download resulted in empty file")
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if attempt == max_retries - 1:
                    raise DownloadError(f"Download failed after {max_retries} attempts: {e}")
                print(f"   ⚠️  Attempt {attempt + 1} failed, retrying in {2 ** attempt}s...")
                time.sleep(2 ** attempt)

    def _copy_directory(self, src: Path, dst: Path) -> None:
        """Copy directory contents recursively."""
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)

    def _optimize_spacy_model(self, model_path: Path) -> None:
        """Remove unnecessary spaCy components to save space."""
        removable_components = [
            "textcat", "textcat_multilabel", "entity_linker",
            "entity_ruler", "sentiment",
        ]
        removable_files = ["training", "train", "accuracy.json"]

        for component in removable_components + removable_files:
            comp_path = model_path / component
            if comp_path.exists():
                try:
                    if comp_path.is_dir():
                        shutil.rmtree(comp_path, ignore_errors=True)
                    else:
                        comp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        for pattern in ["*.onnx"]:
            for f in model_path.glob(pattern):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass

        safe_pt_names = {"optimizer.pt", "scheduler.pt"}
        for f in model_path.glob("*.pt"):
            if f.name in safe_pt_names:
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass

   
    def ensure(self, key: str, force: bool = False) -> Path:
                 
        spec = get_spec(key)

        # API models have nothing to download or verify
        if spec.loader.startswith("api_"):
            return Path("__api__")

        if not self.is_downloaded(key) or force:
            self.download(key, force=force)

        path = self.model_path(key)

        if not self.is_downloaded(key):
            shutil.rmtree(path, ignore_errors=True)
            self.download(key, force=True)
            if not self.is_downloaded(key):
                raise IntegrityError(
                    f"Model '{key}' failed verification after forced re-download. "
                    f"Check disk space and try again, or inspect: {path}"
                )

        return path

    def delete(self, key: str) -> None:
        """Remove a model."""
        path = self.model_path(key)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def list_installed(self) -> list[str]:
        """List all installed models."""
        return [k for k in REGISTRY if self.is_downloaded(k)]

    def installed_size(self, key: str) -> str:
        """Get human-readable size of installed model."""
        
        spec = get_spec(key)
        
        if spec.disk_mb == 0:
            return "API (0MB)"
        
        
        path = self.model_path(key)
        if not path.exists():
            return "0 MB"
        
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        
        for unit in ["B", "KB", "MB", "GB"]:
            if total < 1024:
                return f"{total:.1f} {unit}"
            total /= 1024
        return f"{total:.1f} TB"

    def get_size_info(self) -> dict:
        """Get detailed size information."""
        info = {}
        for key in self.list_installed():
            path = self.model_path(key)
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            file_count = sum(1 for f in path.rglob("*") if f.is_file())
            info[key] = {
                "size_bytes": total,
                "size_human": self.installed_size(key),
                "files": file_count,
                "path": str(path),
            }
        return info

    def cleanup_failed_downloads(self) -> int:
        """Remove incomplete/corrupted downloads. Returns bytes freed."""
        freed = 0
        for key in REGISTRY:
            path = self.model_path(key)
            if path.exists() and not self.is_downloaded(key):
                freed += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                shutil.rmtree(path, ignore_errors=True)
        return freed