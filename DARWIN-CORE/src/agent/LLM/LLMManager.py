import json
import os
import threading
from pathlib import Path
from typing import Optional

from src.agent.LLM.LLMConfig import LLMConfig, ProviderConfig, TierConfig, PROVIDER_ENV_MAP
from src.agent.LLM.LLMClient import LLMClient
from src.lib.AppPaths import default_models_config_dir
from src.core.Logger.DarwinLogger import darwin, LogType


class LLMManager:
    _instance = None
    _client: Optional[LLMClient] = None
    _config: Optional[LLMConfig] = None
    _config_path = Path(default_models_config_dir()) / "llm_config.json"
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, config_path: Optional[Path] = None, auto_load_keys: bool = True) -> bool:
        """Load configuration from JSON file. Returns True if config exists, False if needs setup."""
        if config_path:
            cls._config_path = config_path

        exists = cls._config_path.exists()
        if exists:
            try:
                with open(cls._config_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                cls._config = cls._deserialize_config(raw)
                cls._client = None
                darwin.log(f"Configuration loaded from {cls._config_path}", LogType.SUCCESS)
            except (json.JSONDecodeError, Exception) as e:
                darwin.log(
                    f"Failed to load config from {cls._config_path}: {e}. Starting with clean config.",
                    LogType.WARNING
                )
                cls._config = None
                exists = False
        else:
            cls._config = None

        if auto_load_keys:
            cls._hydrate_keys()

        return exists

    @classmethod
    def _hydrate_keys(cls) -> None:
        """
        Populate ProviderConfig.api_key from the OS keychain and environment variables.
        Called automatically during initialize() unless disabled.
        """
        try:
            from src.lib.KeyManager import load_keys_from_keychain
            load_keys_from_keychain(log=True)
        except Exception as e:
            darwin.log(f"Keychain loading failed (non-fatal): {e}", LogType.WARNING)

        if cls._config is None:
            return

        with cls._lock:
            for name, pc in cls._config.providers.items():
                env_var = PROVIDER_ENV_MAP.get(name)
                if env_var:
                    env_key = os.environ.get(env_var, "").strip()
                    if env_key:
                        pc.api_key = env_key
                        darwin.log(
                            f"Hydrated api_key for provider '{name}' from {env_var}."                        )
                    elif pc.api_key:
                        os.environ[env_var] = pc.api_key

    @classmethod
    def get_client(cls) -> LLMClient:
        with cls._lock:
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
        with cls._lock:
            if cls._config is None:
                raise RuntimeError("No config loaded.")
            if provider not in cls._config.providers:
                raise ValueError(f"Provider '{provider}' not configured.")
            cls._config.default_provider = provider
            cls._client = None
            cls._save_config()

    @classmethod
    def add_provider(
        cls,
        name: str,
        fast_model: str,
        smart_model: str,
        normal_model: Optional[str] = None,
        fast_temp: float = 0.1,
        fast_max_tokens: int = 1024,
        smart_temp: float = 0.7,
        smart_max_tokens: int = 4096,
        normal_temp: float = 0.3,
        normal_max_tokens: int = 2048,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Add or update a provider with fast, smart, and optional normal tiers."""
        name = name.lower().strip()
        fast_model = fast_model.strip()
        smart_model = smart_model.strip()
        normal_model = normal_model.strip() if normal_model else smart_model

        # ── Validation: reject menu numbers and empty strings ──
        for field_name, val in [
            ("fast_model", fast_model),
            ("smart_model", smart_model),
            ("normal_model", normal_model),
        ]:
            if not val:
                raise ValueError(f"{field_name} cannot be empty.")
            if val.isdigit():
                raise ValueError(
                    f"{field_name} looks like a menu selection number ('{val}') instead of a model ID. "
                    "Please enter the actual model name (e.g. 'llama3.2', 'qwen2.5')."
                )
        # ─────────────────────────────────────────────────────

        if not base_url and (" " in fast_model or " " in smart_model):
            raise ValueError("Cloud provider model IDs should not contain spaces. Use the correct model slug.")

        with cls._lock:
            if cls._config is None:
                cls._config = LLMConfig(default_provider=name, providers={})

            cls._config.providers[name] = ProviderConfig(
                fast=TierConfig(model=fast_model, temperature=fast_temp, max_tokens=fast_max_tokens),
                smart=TierConfig(model=smart_model, temperature=smart_temp, max_tokens=smart_max_tokens),
                normal=TierConfig(model=normal_model, temperature=normal_temp, max_tokens=normal_max_tokens),
                base_url=base_url,
                api_key=api_key,
            )
            cls._client = None
            cls._save_config()

    @classmethod
    def remove_provider(cls, name: str):
        with cls._lock:
            if cls._config is None:
                raise RuntimeError("No config loaded.")
            if name not in cls._config.providers:
                raise ValueError(f"Provider '{name}' not found.")
            del cls._config.providers[name]
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
        tmp_path = cls._config_path.with_suffix(".tmp")
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(cls._serialize_config(), f, indent=2)
            if cls._config_path.exists():
                backup_path = cls._config_path.with_suffix(".json.bak")
                os.replace(cls._config_path, backup_path)
            os.replace(tmp_path, cls._config_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @classmethod
    def _serialize_config(cls) -> dict:
        """Serialize config to JSON. SECURITY: api_key is NEVER written to disk."""
        config = cls._config
        result = {"default_provider": config.default_provider, "providers": {}}
        for name, pc in config.providers.items():
            result["providers"][name] = {
                "fast": {
                    "model": pc.fast.model,
                    "temperature": pc.fast.temperature,
                    "max_tokens": pc.fast.max_tokens,
                    "extra_body": pc.fast.extra_body,
                },
                "smart": {
                    "model": pc.smart.model,
                    "temperature": pc.smart.temperature,
                    "max_tokens": pc.smart.max_tokens,
                    "extra_body": pc.smart.extra_body,
                },
                "normal": {
                    "model": pc.normal.model,
                    "temperature": pc.normal.temperature,
                    "max_tokens": pc.normal.max_tokens,
                    "extra_body": pc.normal.extra_body,
                },
            }
            if pc.base_url:
                result["providers"][name]["base_url"] = pc.base_url
            # SECURITY: api_key is intentionally omitted. Never persist secrets to JSON.
        return result

    @classmethod
    def _deserialize_config(cls, raw: dict) -> LLMConfig:
        providers = {}
        for name, pd in raw.get("providers", {}).items():
            fast_raw = pd["fast"]
            smart_raw = pd["smart"]
            normal_raw = pd.get("normal", smart_raw)
            providers[name] = ProviderConfig(
                fast=TierConfig(**fast_raw),
                smart=TierConfig(**smart_raw),
                normal=TierConfig(**normal_raw),
                base_url=pd.get("base_url"),
                api_key=pd.get("api_key"),
            )
        return LLMConfig(default_provider=raw["default_provider"], providers=providers)

    @classmethod
    def list_providers(cls) -> list:
        if cls._config is None:
            return []
        return list(cls._config.providers.keys())