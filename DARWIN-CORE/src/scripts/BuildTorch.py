# scripts/build_torch_plugin.py
"""
Builds darwin-torch-cpu.zip and darwin-torch-cuda.zip
Run this on a clean machine/VM with Python 3.11 installed.
"""

import subprocess
import sys
import shutil
import zipfile
from pathlib import Path


def build_plugin(cuda: bool = False):
    suffix = "cuda" if cuda else "cpu"
    platform = sys.platform  # linux, win32, darwin
    machine = "x86_64"  # or detect via platform.machine()
    
    build_dir = Path(f"build/torch-plugin-{suffix}")
    plugin_dir = build_dir / "python"
    
    # Clean
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    
    # Create venv
    venv_python = build_dir / "venv" / "bin" / "python"
    if sys.platform == "win32":
        venv_python = build_dir / "venv" / "Scripts" / "python.exe"
    
    subprocess.run([sys.executable, "-m", "venv", str(build_dir / "venv")], check=True)
    
    # Install deps
    deps = [
        "torch>=2.1.0",
        "transformers>=4.49.0,<5.0.0",
        "sentence-transformers>=5.3.0",
        "accelerate>=0.25.0",
        "pillow>=10.0.0",  # for vision models
        "numpy>=1.24.0",
    ]
    
    if cuda:
        deps[0] = "torch>=2.1.0"  # pip will auto-resolve CUDA variant
    
    subprocess.run([str(venv_python), "-m", "pip", "install", "--no-cache-dir", *deps], check=True)
    
    # Copy site-packages to plugin structure
    if sys.platform == "win32":
        site_packages = build_dir / "venv" / "Lib" / "site-packages"
    else:
        site_packages = list((build_dir / "venv" / "lib").glob("python*/site-packages"))[0]
    
    plugin_dir.mkdir(parents=True)
    shutil.copytree(site_packages, plugin_dir / "site-packages", dirs_exist_ok=True)
    
    # Create .installed marker
    (plugin_dir.parent / ".installed").write_text(f"torch-{suffix}\n")
    
    # Zip it
    zip_name = f"darwin-torch-{suffix}-{sys.platform}-{machine}-v2.1.0.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in plugin_dir.rglob("*"):
            zf.write(f, f.relative_to(plugin_dir.parent))
    
    print(f"Built: {zip_name}")
    print(f"Size: {Path(zip_name).stat().st_size / 1024 / 1024:.1f} MB")
    return zip_name


if __name__ == "__main__":
    build_plugin(cuda=False)  # CPU version
    # build_plugin(cuda=True)  # CUDA version (run on machine with CUDA)