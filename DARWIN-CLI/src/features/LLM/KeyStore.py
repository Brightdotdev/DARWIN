import json
import os
import stat
from pathlib import Path
from typing import List, Dict
from src.lib.AppPaths import default_models_config_dir

class KeyStore:
    def __init__(self, path: Path = None):
        self.path =  default_models_config_dir() / "keys.json"
        self._data: Dict[str, List[str]] = {}
        self._loaded = False

    def load(self):
        if not self._loaded:
            self._ensure_file()
            with open(self.path, 'r') as f:
                content = f.read().strip()
                if content:
                    self._data = json.loads(content)
                else:
                    self._data = {}
            self._loaded = True

    def save(self):
        self._ensure_file()
        with open(self.path, 'w') as f:
            json.dump(self._data, f, indent=2)
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)  # 600

    def _ensure_file(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        # Also fix if file exists but is empty/invalid
        elif self.path.stat().st_size == 0:
            with open(self.path, 'w') as f:
                json.dump({}, f)

    def get_keys(self, provider: str) -> List[str]:
        self.load()
        return self._data.get(provider, [])

    def add_key(self, provider: str, key: str):
        self.load()
        if provider not in self._data:
            self._data[provider] = []
        if key not in self._data[provider]:
            self._data[provider].append(key)
        self.save()

    def remove_key(self, provider: str, key: str):
        self.load()
        if provider in self._data and key in self._data[provider]:
            self._data[provider].remove(key)
            if not self._data[provider]:
                del self._data[provider]
            self.save()

    def rotate_active(self, provider: str, current_index: int) -> int:
        keys = self.get_keys(provider)
        if not keys:
            raise RuntimeError(f"No API keys for provider {provider}")
        return (current_index + 1) % len(keys)

    def set_env_for_provider(self, provider: str, key_index: int = 0):
        """Temporarily set environment variables needed by LiteLLM."""
        keys = self.get_keys(provider)
        if not keys:
            raise RuntimeError(f"No API keys for provider {provider}")
        key = keys[key_index % len(keys)]
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GEMINI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "groq": "GROQ_API_KEY",
            "cohere": "COHERE_API_KEY",
            "together": "TOGETHER_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_var = env_map.get(provider)
        if env_var:
            os.environ[env_var] = key