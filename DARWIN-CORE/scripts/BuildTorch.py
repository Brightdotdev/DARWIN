# scripts/build_torch_plugin.py
"""
Builds darwin-torch-{cpu|cuda}-{os}-{arch}-v{version}.zip
Usage:
    python build_torch_plugin.py [--cuda] [--python 3.11] [--arch arm64]
"""

import argparse
import subprocess
import sys
import shutil
import zipfile
from pathlib import Path


def build_plugin(cuda: bool = False, python_version: str = "3.11", arch: str = None):
    if arch is None:
        arch = platform.machine().lower()
        # Normalize: arm64 on macOS reports 'arm64', Linux 'aarch64'
        if arch == "aarch64":
            arch = "arm64"

    suffix = "cuda" if cuda else "cpu"
    os_name = sys.platform  # linux, win32, darwin
    version = "2.1.0"  # update when needed

    build_dir = Path(f"build/torch-plugin-{suffix}-{os_name}-{arch}-v{version}")
    plugin_dir = build_dir / "python"

    # Clean
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    # Create a virtualenv
    venv_dir = build_dir / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    pip = str(venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "pip")

    # Install packages
    deps = [
        "torch>=2.1.0",
        "transformers>=4.49.0,<5.0.0",
        "sentence-transformers>=5.3.0",
        "accelerate>=0.25.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
    ]

    # For CUDA, pip will pick the appropriate index automatically if we don't force cpu
    if not cuda:
        # Force CPU-only torch to avoid pulling CUDA wheels on machines with GPU
        deps[0] = "torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu"

    subprocess.run([pip, "install", "--no-cache-dir"] + deps, check=True)

    # Locate site-packages
    if sys.platform == "win32":
        site_packages = venv_dir / "Lib" / "site-packages"
    else:
        site_packages = list((venv_dir / "lib").glob("python*/site-packages"))[0]

    # Copy site-packages to plugin structure
    plugin_dir.mkdir(parents=True)
    shutil.copytree(site_packages, plugin_dir / "site-packages", dirs_exist_ok=True)

    # Create .installed marker
    (build_dir / ".installed").write_text(f"torch-{suffix}\n")

    # Zip it (compressing large files)
    zip_name = f"darwin-torch-{suffix}-{os_name}-{arch}-v{version}.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in build_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(build_dir))

    print(f"Built: {zip_name}")
    print(f"Size: {Path(zip_name).stat().st_size / 1024 / 1024:.1f} MB")
    return zip_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="Build CUDA version")
    parser.add_argument("--python", default="3.11", help="Python version (ignored)")
    parser.add_argument("--arch", help="Target architecture (arm64, x86_64)")
    args = parser.parse_args()
    build_plugin(cuda=args.cuda, arch=args.arch)