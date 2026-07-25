# src/core/Models/ModelDownloading/SpacyNLPModelDownloader.py
"""
SpaCy NLP Model Downloader
Downloads spaCy models from multiple sources with robust fallback strategies,
Darwin progress tracking, and post-download optimization.
"""

import shutil
import tarfile
import tempfile
import zipfile
import json
import urllib.request
import ssl
from pathlib import Path
from time import sleep
from typing import Optional

from src.core.Logger.DarwinLogger import darwin
from src.lib.InternetChecker import wait_for_internet, check_internet_robust
from src.core.Models.lib import SPACY_URLS, tarball_urls


class SpacyNLPModelDownloader:
    """Downloads and manages spaCy models with robust fallback strategies,
    Darwin progress tracking, and post-download optimization."""

    def __init__(self, model_name: str, target_dir: Path):
        self.model_name = model_name
        self.target_dir = Path(target_dir)
        self.spacy_urls = SPACY_URLS.get(model_name, [])
        self.tarball_urls = tarball_urls.get(model_name, [])

    # ═══════════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════════

    def ensure(
        self,
        force: bool = False,
        recache: bool = False,
        wait_internet: bool = True,
        max_wait: int = 60,
    ) -> bool:
        """
        One-call lifecycle: check → wait for net → download → verify → optimize.
        
        Args:
            force: If True, redownload even if present
            recache: Alias for force — clearer naming for "delete and re-download"
            wait_internet: Block until internet is available
            max_wait: Max seconds to wait for internet
        """
        should_force = force or recache

        if not should_force and self.verify_integrity():
            darwin.info(f"{self.model_name} already present and valid.")
            return True

        if should_force:
            darwin.info("Force/recache mode: clearing existing files...")
            self._clear_cache()

        # ── Internet ──────────────────────────────────────────────────────
        if wait_internet:
            if not wait_for_internet(pause=3, max_wait=max_wait, verbose=True):
                darwin.error("No internet connection available.")
                return False
        else:
            if not check_internet_robust():
                darwin.error("No internet connection available.")
                return False

        # ── Download ──────────────────────────────────────────────────────
        if not self.download(force=should_force):
            darwin.error("Download failed.")
            return False

        # ── Verify ──────────────────────────────────────────────────────
        if not self.verify_integrity():
            darwin.error("Integrity check failed after download.")
            return False

        # ── Optimize ──────────────────────────────────────────────────────
        self._optimize_model(self.target_dir)
        darwin.success(f"{self.model_name} ready at {self.target_dir}")
        return True

    def download(self, force: bool = False) -> bool:
        """Attempt all download strategies. Returns True on success."""
        if self.target_dir.exists() and not force:
            pass  # may be corrupt; strategies will overwrite if they succeed
        self.target_dir.mkdir(parents=True, exist_ok=True)

        progress_id = f"dl_{self.model_name}"
        darwin.spinner_start(progress_id, f"Downloading {self.model_name}")

        strategies = [
            ("GitHub wheel", self._download_wheel),
            ("GitHub tarball", self._download_tarball),
        ]

        # Dynamic spaCy CLI strategy (only if spaCy is importable)
        try:
            import spacy  # noqa: F401
            strategies.append(
                ("spaCy package", self._download_via_package)
            )
        except ImportError:
            darwin.warning("spaCy not available, skipping pip package strategy")

        errors = []
        for name, strategy in strategies:
            try:
                darwin.info(f"Attempting {name} download for {self.model_name}")
                if strategy():
                    if self.verify_integrity():
                        darwin.spinner_stop(progress_id, success=True)
                        return True
                    errors.append(f"{name}: verification failed after extraction")
                else:
                    errors.append(f"{name}: strategy returned False")
            except Exception as e:
                errors.append(f"{name}: {e}")

        darwin.spinner_stop(progress_id, success=False)
        darwin.info(
            f"Failed to download spaCy model '{self.model_name}'.\n"
            +             "Errors:\n  " + "\n  ".join(errors),
        )
        return False

    def verify_integrity(self) -> bool:
        """Check that all essential spaCy model files are present."""
        required = [
            self.target_dir / "config.cfg",
            self.target_dir / "meta.json",
            self.target_dir / "tokenizer",
            self.target_dir / "vocab",
        ]
        return all(p.exists() for p in required)

    def _clear_cache(self) -> None:
        """Remove all cached files to force fresh download."""
        if self.target_dir.exists():
            try:
                shutil.rmtree(self.target_dir, ignore_errors=True)
                darwin.info(f"Cleared: {self.target_dir}")
            except Exception as e:
                darwin.warning(f"Failed to clear {self.target_dir}: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # Download strategies
    # ═══════════════════════════════════════════════════════════════════════

    def _download_wheel(self) -> bool:
        if not self.spacy_urls:
            darwin.warning("No wheel URLs configured.")
            return False

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            wheel_path = tmp / f"{self.model_name}.whl"

            last_error = None
            for url in self.spacy_urls:
                try:
                    self._download_file(url, wheel_path)
                    if not wheel_path.exists() or wheel_path.stat().st_size == 0:
                        continue

                    darwin.info("Extracting wheel...")
                    extract_dir = tmp / "extracted"
                    extract_dir.mkdir(exist_ok=True)
                    with zipfile.ZipFile(wheel_path, "r") as zf:
                        self._safe_extract_zip(zf, extract_dir)

                    model_dir = self._find_model_in_wheel(extract_dir)
                    if model_dir:
                        self._atomic_copy(model_dir, self.target_dir)
                        # Critical file sanity check before declaring success
                        for f in ["config.cfg", "meta.json", "tokenizer"]:
                            if not (self.target_dir / f).exists():
                                darwin.error(f"Missing critical file: {f}")
                                return False
                        return True
                    last_error = f"Model directory not found in wheel from {url}"
                except Exception as e:
                    last_error = str(e)
                    darwin.warning(f"Wheel attempt failed: {e}")

            darwin.error(f"Wheel download failed: {last_error}")
            return False

    def _download_tarball(self) -> bool:
        if not self.tarball_urls:
            darwin.warning("No tarball URLs configured.")
            return False

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for url in self.tarball_urls:
                try:
                    tarball_path = tmp / f"{self.model_name}.tar.gz"
                    self._download_file(url, tarball_path)

                    darwin.info("Extracting tarball...")
                    with tarfile.open(tarball_path, "r:gz") as tar:
                        self._safe_extract_tar(tar, tmp)

                    model_dir = self._find_model_in_extracted(tmp)
                    if model_dir:
                        self._atomic_copy(model_dir, self.target_dir)
                        return True
                except Exception as e:
                    darwin.warning(f"Tarball attempt failed: {e}")

            darwin.error("Tarball download failed for all URLs.")
            return False

    def _download_via_package(self) -> bool:
        """Fallback: use spaCy's own CLI downloader."""
        import spacy
        from spacy.cli.download import download as spacy_download

        darwin.info("Using spaCy CLI downloader...")
        spacy_download(self.model_name, direct=False)
        installed_path = Path(spacy.util.get_package_path(self.model_name))
        self._atomic_copy(installed_path, self.target_dir)
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # Shared file download with Darwin progress
    # ═══════════════════════════════════════════════════════════════════════

    def _download_file(
        self,
        url: str,
        dest: Path,
        max_retries: int = 3,
    ) -> None:
        progress_id = f"dl_file_{dest.name}"

        for attempt in range(max_retries):
            try:
                darwin.info(f"Downloading {url[:80]}...")
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Darwin-ModelManager/1.0",
                        "Accept": "*/*",
                    },
                )
                ctx = ssl.create_default_context()
                with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
                    total = int(response.headers.get("Content-Length", 0))
                    downloaded = 0

                    if total > 0:
                        darwin.progress_start(
                            progress_id,
                            total=total,
                            label=f"Downloading {dest.name}",
                        )

                    with open(dest, "wb") as f:
                        while True:
                            chunk = response.read(65536)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                darwin.progress_update(
                                    progress_id,
                                    downloaded,
                                    status=f"{downloaded / 1024 / 1024:.1f} MB",
                                )

                    if total > 0:
                        darwin.progress_end(progress_id)

                # ── Post-download validation ──────────────────────────────
                if not dest.exists() or dest.stat().st_size == 0:
                    darwin.error("Empty file after download")
                    return

                suffix = "".join(dest.suffixes).lower()
                if ".whl" in suffix or ".zip" in suffix:
                    with zipfile.ZipFile(dest, "r") as zf:
                        if not zf.namelist():
                            darwin.error("Empty ZIP archive")
                            return
                elif ".tar" in suffix or ".tgz" in suffix:
                    with tarfile.open(dest) as tar:
                        if not tar.getmembers():
                            darwin.error("Empty TAR archive")
                            return

                darwin.success(
                    f"Downloaded {downloaded / 1024 / 1024:.1f} MB",
                )
                return

            except KeyboardInterrupt:
                raise
            except Exception as e:
                if attempt == max_retries - 1:
                    darwin.error(
                        f"Download failed after {max_retries} attempts: {e}"
                    )
                    return
                darwin.warning(
                    f"Attempt {attempt + 1} failed, retrying...",
                )
                sleep(2 ** attempt)

    # ═══════════════════════════════════════════════════════════════════════
    # Model finding helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _find_model_in_wheel(self, extract_dir: Path) -> Optional[Path]:
        """Multi-tier fallback for locating model inside a wheel archive."""
        pkg_name = self.model_name.replace("-", "_")

        # Tier 1: exact name or package-name match
        for candidate in (self.model_name, pkg_name):
            for path in extract_dir.rglob(candidate):
                if path.is_dir() and (path / "config.cfg").exists() and (path / "meta.json").exists():
                    return path
                if path.is_dir():
                    for sub in path.iterdir():
                        if sub.is_dir() and (sub / "config.cfg").exists():
                            return sub

        # Tier 2: any config.cfg with meta.json + tokenizer
        for config in extract_dir.rglob("config.cfg"):
            parent = config.parent
            if (parent / "meta.json").exists() and (parent / "tokenizer").exists():
                return parent

        # Tier 3: meta.json name match
        for meta in extract_dir.rglob("meta.json"):
            try:
                data = json.loads(meta.read_text())
                if data.get("name") == self.model_name or data.get("lang") == "en":
                    if (meta.parent / "tokenizer").exists():
                        return meta.parent
            except Exception:
                continue
        return None

    def _find_model_in_extracted(self, base_path: Path) -> Optional[Path]:
        """Locate model in tarball or generic extracted tree."""
        for config in base_path.rglob("config.cfg"):
            parent = config.parent
            if (parent / "meta.json").exists() and (parent / "tokenizer").exists():
                return parent

        for meta in base_path.rglob("meta.json"):
            try:
                data = json.loads(meta.read_text())
                if data.get("name") == self.model_name:
                    if (meta.parent / "tokenizer").exists():
                        return meta.parent
            except Exception:
                continue
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _atomic_copy(self, source: Path, target: Path) -> None:
        """Replace target atomically with source contents."""
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        darwin.info("Copying model files...")
        shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)

    @staticmethod
    def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
        """Extract a ZIP archive with path-traversal protection."""
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"Path traversal attempt blocked in ZIP: {member}")
        zf.extractall(dest)

    @staticmethod
    def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
        """Extract a TAR archive with path-traversal protection."""
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"Path traversal attempt blocked in TAR: {member.name}")
        tar.extractall(dest, filter="data" if hasattr(tarfile, "data") else None)

    def _optimize_model(self, model_path: Path) -> None:
        """Strip unused spaCy components to reduce disk footprint."""
        darwin.info("Optimizing model (stripping unused components)...")
        removable_dirs = [
            "textcat",
            "textcat_multilabel",
            "entity_linker",
            "entity_ruler",
            "sentiment",
        ]
        removable_files = ["training", "train", "accuracy.json"]

        for name in removable_dirs + removable_files:
            p = model_path / name
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)

        for f in model_path.glob("*.onnx"):
            f.unlink(missing_ok=True)
        for f in list(model_path.glob("optimizer.pt")) + list(model_path.glob("scheduler.pt")):
            f.unlink(missing_ok=True)

        darwin.success("Optimization complete.")