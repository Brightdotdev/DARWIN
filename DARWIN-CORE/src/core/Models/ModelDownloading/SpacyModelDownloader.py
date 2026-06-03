
from pathlib import Path
import shutil
import tarfile
import tempfile
from time import time
from typing import Optional
import zipfile
import json
import urllib

from src.core.Logger.DarwinLogger import  darwin ,LogType
from src.core.Models.ModelUtils import DownloadError, IntegrityError
from src.core.Models.lib import (
    SPACY_URLS,
    tarball_urls,
)

class SpacyModelDownloader:
    """Downloads spaCy models using multiple strategies, emitting Darwin progress."""

    def __init__(self):
        self.SPACY_URLS = SPACY_URLS

    def download(self, model_name: str, target_path: Path) -> Path:
        progress_id = f"dl_{model_name}"
        darwin.spinner_start(progress_id,f"Downloading {model_name}")

        strategies = [
            ("GitHub wheel", lambda: self._download_wheel(model_name, target_path, progress_id)),
            ("GitHub tarball", lambda: self._download_tarball(model_name, target_path, progress_id)),
        ]
        try:
            strategies.append(
                ("spaCy package", lambda: self._download_via_package(model_name, target_path, progress_id))
            )
        except ImportError:
            pass

        errors = []
        for name, strategy in strategies:
            try:
                darwin.log(f"Attempting {name} download for {model_name}")
                result = strategy()
                if result and self._verify_spacy_model(target_path):
                    self._optimize_model(target_path)
                    darwin.spinner_end(progress_id, success=True)
                    return target_path
                errors.append(f"{name}: verification failed")
            except Exception as e:
                errors.append(f"{name}: {e}")

        darwin.spinner_end(progress_id, success=False)
        raise DownloadError(
            f"Failed to download spaCy model '{model_name}'.\n" +
            "Errors:\n  " + "\n  ".join(errors),
            f"Check internet connection. You can also install manually:\n"
            f"  pip install {model_name}\n"
            f"  Then copy from site-packages to: {target_path}",
            f"If thats too much problem, try installing it again...or really check your internet connection."
        )

    # ── Wheel strategy ───────────────────────────────────────────────
    def _download_wheel(self, model_name: str, target_path: Path, progress_id: str) -> bool:
        urls = self.SPACY_URLS.get(model_name, [])
        if not urls:
            raise DownloadError(f"No download URLs for {model_name}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            wheel_path = tmp / f"{model_name}.whl"

            last_error = None
            for url in urls:
                try:
                    self._download_file(url, wheel_path, progress_id)
                    if not wheel_path.exists() or wheel_path.stat().st_size == 0:
                        continue

                    darwin.log("Extracting wheel...")
                    extract_dir = tmp / "extracted"
                    extract_dir.mkdir(exist_ok=True)
                    with zipfile.ZipFile(wheel_path, 'r') as zf:
                        zf.extractall(extract_dir)

                    model_dir = self._find_model_in_wheel(extract_dir, model_name)
                    if model_dir:
                        if target_path.exists():
                            shutil.rmtree(target_path, ignore_errors=True)
                        darwin.log("Copying model files...")
                        shutil.copytree(model_dir, target_path, symlinks=True, dirs_exist_ok=True)
                        for f in ["config.cfg", "meta.json", "tokenizer"]:
                            if not (target_path / f).exists():
                                raise IntegrityError(f"Missing critical file: {f}")
                        return True
                    last_error = f"Model directory not found in wheel from {url}"
                except Exception as e:
                    last_error = str(e)

            raise DownloadError(f"Wheel download failed: {last_error}")

    # ── Tarball strategy ─────────────────────────────────────────────
    def _download_tarball(self, model_name: str, target_path: Path, progress_id: str) -> bool:
        urls = tarball_urls.get(model_name, [])
        if not urls:
            raise DownloadError(f"No tarball URLs for {model_name}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for url in urls:
                try:
                    tarball_path = tmp / f"{model_name}.tar.gz"
                    self._download_file(url, tarball_path, progress_id)

                    darwin.log("Extracting tarball...")
                    with tarfile.open(tarball_path, 'r:gz') as tar:
                        tar.extractall(tmp)

                    model_dir = self._find_model_in_extracted(tmp, model_name)
                    if model_dir:
                        if target_path.exists():
                            shutil.rmtree(target_path, ignore_errors=True)
                        shutil.copytree(model_dir, target_path, symlinks=True, dirs_exist_ok=True)
                        return True
                except Exception as e:
                    darwin.log(f"Tarball attempt failed: {e}", LogType.WARNING)
            raise DownloadError("Tarball download failed for all URLs")

    # ── Package strategy ─────────────────────────────────────────────
    def _download_via_package(self, model_name: str, target_path: Path, progress_id: str) -> bool:
        import spacy
        from spacy.cli.download import download as spacy_download

        spacy_download(model_name, direct=False)
        installed_path = Path(spacy.util.get_package_path(model_name))
        if target_path.exists():
            shutil.rmtree(target_path, ignore_errors=True)
        shutil.copytree(installed_path, target_path, symlinks=True, dirs_exist_ok=True)
        return True

    # ── Shared file download with Darwin progress ────────────────────
    def _download_file(self, url: str, dest: Path, progress_id: str, max_retries: int = 3) -> None:
        for attempt in range(max_retries):
            try:
                darwin.log(f"Downloading {url[:80]}...")
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Darwin-ModelManager/1.0", "Accept": "*/*"},
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    total = int(response.headers.get("Content-Length", 0))
                    downloaded = 0

                    # Start a real progress bar if size known
                    if total > 0:
                        darwin.progress_start(progress_id, total=total,
                                                       label=f"Downloading {dest.name}")

                    with open(dest, "wb") as f:
                        while True:
                            chunk = response.read(65536)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                darwin.progress_update(progress_id, downloaded,status=f"{downloaded / 1024 / 1024:.1f} MB")


                    if total > 0:
                        darwin.progress_end(progress_id)

                if dest.exists() and dest.stat().st_size > 0:
                    # quick archive check
                    suffix = "".join(dest.suffixes).lower()
                    if ".whl" in suffix or ".zip" in suffix:
                        with zipfile.ZipFile(dest, 'r') as zf:
                            if not zf.namelist():
                                raise Exception("Empty archive")
                    elif ".tar" in suffix or ".tgz" in suffix:
                        with tarfile.open(dest) as tar:
                            if not tar.getmembers():
                                raise Exception("Empty archive")
                    darwin.log(f"Downloaded {downloaded / 1024 / 1024:.1f} MB", LogType.SUCCESS)
                    return
                raise Exception("Empty file after download")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if attempt == max_retries - 1:
                    raise DownloadError(f"Download failed after {max_retries} attempts: {e}")
                darwin.log(f"Attempt {attempt + 1} failed, retrying...", LogType.WARNING)
                time.sleep(2 ** attempt)

    # ── Model finding helpers (unchanged) ────────────────────────────
    def _find_model_in_wheel(self, extract_dir: Path, model_name: str) -> Optional[Path]:
        pkg_name = model_name.replace("-", "_")
        for candidate in (model_name, pkg_name):
            for path in extract_dir.rglob(candidate):
                if path.is_dir() and (path / "config.cfg").exists() and (path / "meta.json").exists():
                    return path
                if path.is_dir():
                    for sub in path.iterdir():
                        if sub.is_dir() and (sub / "config.cfg").exists():
                            return sub

        for config in extract_dir.rglob("config.cfg"):
            parent = config.parent
            if (parent / "meta.json").exists() and (parent / "tokenizer").exists():
                return parent

        for meta in extract_dir.rglob("meta.json"):
            try:
                data = json.loads(meta.read_text())
                if data.get("name") == model_name or data.get("lang") == "en":
                    if (meta.parent / "tokenizer").exists():
                        return meta.parent
            except Exception:
                continue
        return None

    def _find_model_in_extracted(self, base_path: Path, model_name: str) -> Optional[Path]:
        for config in base_path.rglob("config.cfg"):
            parent = config.parent
            if (parent / "meta.json").exists() and (parent / "tokenizer").exists():
                return parent

        for meta in base_path.rglob("meta.json"):
            try:
                data = json.loads(meta.read_text())
                if data.get("name") == model_name:
                    if (meta.parent / "tokenizer").exists():
                        return meta.parent
            except Exception:
                continue
        return None

    def _verify_spacy_model(self, path: Path) -> bool:
        essential = ["tokenizer", "meta.json", "config.cfg", "vocab"]
        return all((path / c).exists() for c in essential)

    def _optimize_model(self, model_path: Path) -> None:
        removable_dirs = ["textcat", "textcat_multilabel", "entity_linker", "entity_ruler", "sentiment"]
        removable_files = ["training", "train", "accuracy.json"]
        for name in removable_dirs + removable_files:
            p = model_path / name
            if p.exists():
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        for f in model_path.glob("*.onnx"):
            f.unlink(missing_ok=True)
        for f in list(model_path.glob("optimizer.pt")) + list(model_path.glob("scheduler.pt")):
            f.unlink(missing_ok=True)
