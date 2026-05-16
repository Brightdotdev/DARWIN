import json
import os
from pathlib import Path
from typing import Optional, Dict

from src.features.LLM.LLMConfig import LLMConfig, ProviderConfig, TierConfig
from src.features.LLM.LLMClient import LLMClient
from src.lib.AppPaths import default_models_config_dir
from src.core.Logger.DarwinLogger import darwin, LogType


class LLMManager:
    _instance = None
    _client: Optional[LLMClient] = None
    _config: Optional[LLMConfig] = None
    _config_path = Path(default_models_config_dir()) / "llm_config.json"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, config_path: Optional[Path] = None) -> bool:
        """Load configuration from JSON file. Returns True if config exists, False if needs setup."""
        if config_path:
            cls._config_path = config_path
        if cls._config_path.exists():
            try:
                with open(cls._config_path, 'r') as f:
                    raw = json.load(f)
                cls._config = cls._deserialize_config(raw)
                cls._client = None
                return True
            except (json.JSONDecodeError, Exception) as e:
                darwin.log(
                    f"Failed to load config from {cls._config_path}: {e}. Starting with clean config.",
                    LogType.WARNING
                )
                cls._config = None
                return False
        else:
            cls._config = None
            return False

    @classmethod
    def get_client(cls) -> LLMClient:
        if cls._config is None:
            raise RuntimeError("LLMManager not initialized. Call initialize() first.")
        if cls._client is None:
            cls._client = LLMClient(
                providers=cls._config.providers,
                default_provider=cls._config.default_provider,
            )
        return cls._client

    @classmethod
    def switch_default_provider(cls, provider: str):
        if cls._config is None:
            raise RuntimeError("No config loaded.")
        if provider not in cls._config.providers:
            raise ValueError(f"Provider '{provider}' not configured.")
        cls._config.default_provider = provider
        cls._client = None
        cls._save_config()

    @classmethod
    def add_provider(cls, name: str, fast_model: str, smart_model: str,
                     fast_temp: float = 0.1, fast_max_tokens: int = 1024,
                     smart_temp: float = 0.7, smart_max_tokens: int = 4096,
                     base_url: Optional[str] = None):
        # Validate inputs
        if not name or not fast_model or not smart_model:
            raise ValueError("Provider name and model IDs cannot be empty.")
        if not base_url and (" " in fast_model or " " in smart_model):
            raise ValueError("Cloud provider model IDs should not contain spaces. Use the correct model slug.")

        if cls._config is None:
            cls._config = LLMConfig(default_provider=name, providers={})
        cls._config.providers[name] = ProviderConfig(
            fast=TierConfig(model=fast_model, temperature=fast_temp, max_tokens=fast_max_tokens),
            smart=TierConfig(model=smart_model, temperature=smart_temp, max_tokens=smart_max_tokens),
            base_url=base_url
        )
        cls._client = None
        cls._save_config()

    @classmethod
    def remove_provider(cls, name: str):
        """Remove a provider from the configuration."""
        if cls._config is None:
            raise RuntimeError("No config loaded.")
        if name not in cls._config.providers:
            raise ValueError(f"Provider '{name}' not found.")
        del cls._config.providers[name]
        # If default was removed, pick the first available or leave empty
        if cls._config.default_provider == name:
            if cls._config.providers:
                cls._config.default_provider = next(iter(cls._config.providers))
            else:
                cls._config.default_provider = ""
        cls._client = None
        cls._save_config()

    @classmethod
    def _save_config(cls):
        cls._config_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write with backup
        tmp_path = cls._config_path.with_suffix(".tmp")
        try:
            with open(tmp_path, 'w') as f:
                json.dump(cls._serialize_config(), f, indent=2)
            # Create backup only if original exists
            if cls._config_path.exists():
                backup_path = cls._config_path.with_suffix(".json.bak")
                os.replace(cls._config_path, backup_path)
            os.replace(tmp_path, cls._config_path)
        except Exception:
            # Clean up tmp if something goes wrong
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @classmethod
    def _serialize_config(cls) -> dict:
        config = cls._config
        result = {"default_provider": config.default_provider, "providers": {}}
        for name, pc in config.providers.items():
            result["providers"][name] = {
                "fast": {
                    "model": pc.fast.model,
                    "temperature": pc.fast.temperature,
                    "max_tokens": pc.fast.max_tokens,
                    "extra_body": pc.fast.extra_body
                },
                "smart": {
                    "model": pc.smart.model,
                    "temperature": pc.smart.temperature,
                    "max_tokens": pc.smart.max_tokens,
                    "extra_body": pc.smart.extra_body
                },
            }
            if pc.base_url:
                result["providers"][name]["base_url"] = pc.base_url
        return result

    @classmethod
    def _deserialize_config(cls, raw: dict) -> LLMConfig:
        providers = {}
        for name, pd in raw.get("providers", {}).items():
            fast_raw = pd["fast"]
            smart_raw = pd["smart"]
            providers[name] = ProviderConfig(
                fast=TierConfig(**fast_raw),
                smart=TierConfig(**smart_raw),
                base_url=pd.get("base_url")
            )
        return LLMConfig(default_provider=raw["default_provider"], providers=providers)

    @classmethod
    def list_providers(cls) -> list:
        if cls._config is None:
            return []
        return list(cls._config.providers.keys())