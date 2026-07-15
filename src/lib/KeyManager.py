"""
src/lib/KeyManager.py
Secure API key management using OS keychain.
"""

import os

# ── Silence litellm BEFORE importing it ──────────────────────────────
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_SUPPRESS_LOGGING"] = "1"

import logging
logging.getLogger("litellm").setLevel(logging.ERROR)

import keyring
import litellm

litellm.suppress_debug_info = True
litellm.telemetry = False
litellm.turn_off_message_logging = True

from src.core.Logger.DarwinLogger import darwin, LogType
from src.agent.LLM.LLMConfig import PROVIDER_ENV_MAP



APP_NAME = "darwin"


# --------------------------------------------------------------------------------
# Validation functions – each makes a cheap, authenticated API call to verify
# --------------------------------------------------------------------------------

def _validate_openai(key: str) -> bool:
    from openai import OpenAI
    try:
        OpenAI(api_key=key).models.list(limit=1, timeout=10)
        return True
    except Exception as e:
        if "authentication" not in str(e).lower() and "401" not in str(e):
            raise RuntimeError(f"Network/validation error: {type(e).__name__}: {e}") from e
        return False


def _validate_anthropic(key: str) -> bool:
    try:
        litellm.completion(
            model="anthropic/claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1, api_key=key, timeout=10, num_retries=1,
        )
        return True
    except Exception as e:
        if "authentication" not in str(e).lower() and "401" not in str(e):
            raise RuntimeError(f"Network/validation error: {type(e).__name__}: {e}") from e
        return False


def _validate_google(key: str) -> bool:
    try:
        litellm.completion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1, api_key=key, timeout=10, num_retries=1,
        )
        return True
    except Exception as e:
        if "authentication" not in str(e).lower() and "401" not in str(e):
            raise RuntimeError(f"Network/validation error: {type(e).__name__}: {e}") from e
        return False


def _validate_mistral(key: str) -> bool:
    try:
        litellm.completion(
            model="mistral/mistral-tiny",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1, api_key=key, timeout=10, num_retries=1,
        )
        return True
    except Exception as e:
        if "authentication" not in str(e).lower() and "401" not in str(e):
            raise RuntimeError(f"Network/validation error: {type(e).__name__}: {e}") from e
        return False


def _validate_groq(key: str) -> bool:
    try:
        litellm.completion(
            model="groq/llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1, api_key=key, timeout=10, num_retries=1,
        )
        return True
    except Exception as e:
        if "authentication" not in str(e).lower() and "401" not in str(e):
            raise RuntimeError(f"Network/validation error: {type(e).__name__}: {e}") from e
        return False



def _validate_cohere(key: str) -> bool:
    try:
        litellm.completion(
            model="cohere/command-r",
            messages=[{"role": "user", "content": "hello"}],
            api_key=key,
            max_tokens=1,
        )
        return True
    except Exception:
        return False

def _validate_together(key: str) -> bool:
    try:
        litellm.completion(
            model="together/mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1, api_key=key, timeout=10, num_retries=1,
        )
        return True
    except Exception as e:
        if "authentication" not in str(e).lower() and "401" not in str(e):
            raise RuntimeError(f"Network/validation error: {type(e).__name__}: {e}") from e
        return False


def _validate_openrouter(key: str) -> bool:
    try:
        litellm.completion(
            model="openrouter/openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1, api_key=key, timeout=10, num_retries=1,
        )
        return True
    except Exception as e:
        if "authentication" not in str(e).lower() and "401" not in str(e):
            raise RuntimeError(f"Network/validation error: {type(e).__name__}: {e}") from e
        return False


def _validate_deepseek(key: str) -> bool:
    try:
        litellm.completion(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1, api_key=key, timeout=10, num_retries=1,
        )
        return True
    except Exception as e:
        if "authentication" not in str(e).lower() and "401" not in str(e):
            raise RuntimeError(f"Network/validation error: {type(e).__name__}: {e}") from e
        return False


# --------------------------------------------------------------------------------
# Registry derived from single source of truth (PROVIDER_ENV_MAP) to avoid drift
# --------------------------------------------------------------------------------

_VALIDATORS = {
    "openai": _validate_openai,
    "anthropic": _validate_anthropic,
    "google": _validate_google,
    "mistral": _validate_mistral,
    "groq": _validate_groq,
    "cohere": _validate_cohere,
    "together": _validate_together,
    "openrouter": _validate_openrouter,
    "deepseek": _validate_deepseek,
}

PROVIDER_REGISTRY = {
    name: (env_var, _VALIDATORS[name])
    for name, env_var in PROVIDER_ENV_MAP.items()
    if name in _VALIDATORS
}


def is_key_set(provider: str) -> bool:
    """Check if the API key for `provider` is set in environment."""
    env_var = PROVIDER_REGISTRY.get(provider, (None,))[0]
    return bool(env_var and os.getenv(env_var))

# --------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------

def set_api_key(provider: str, key: str, validate: bool = True) -> None:
    """
    Save an API key for `provider` into the OS keychain and set it
    for the current process. Optionally validate before saving.
    """
    provider = provider.lower().strip()
    if provider not in PROVIDER_REGISTRY:
        darwin.error(f"Unsupported provider: '{provider}'. Supported: {list(PROVIDER_REGISTRY)}")
        return None
    key = key.strip()
    if not key:
        darwin.error("API key must not be empty")
        return None

    env_var, validator = PROVIDER_REGISTRY[provider]

    if validate:
        darwin.info(f"Validating API key for {provider}...")
        try:
            validator(key)
            darwin.success("API key is valid")
        except Exception as e:
            darwin.error(f"Invalid API key for {provider}: {e}")
            return None

    try:
        keyring.set_password(APP_NAME, provider, key)
    except Exception as e:
        darwin.error(f"Failed to store key in keychain: {e}")
        return None

    os.environ[env_var] = key
    darwin.success(f"API key for {provider} saved and activated")



def load_keys_from_keychain(log: bool = True) -> None:
    """
    Load all stored API keys from keychain into environment variables.
    Skips any key already set in the environment (respects manual overrides).
    Call once at application startup.
    """
    loaded = 0
    skipped = 0

    for provider, (env_var, _) in PROVIDER_REGISTRY.items():
        # Already set in env — respect override, no log spam
        if os.environ.get(env_var, "").strip():
            skipped += 1
            continue

        try:
            key = keyring.get_password(APP_NAME, provider)
        except Exception as e:
            # Only warn if there was a real read error, not "not found"
            if log:
                darwin.warning(f"Keychain error for {provider}: {e}")
            continue

        if key and key.strip():
            os.environ[env_var] = key.strip()
            loaded += 1
            if log:
                darwin.info(f"Loaded {env_var} from keychain")

    if log:
        if loaded:
            darwin.info(f"Keychain: {loaded} loaded, {skipped} already present")
        else:
            darwin.warning(f"Keychain: {loaded} loaded, {skipped} already present")

def clear_api_key(provider: str) -> None:
    """Remove a saved API key from the keychain and the current environment."""
    provider = provider.lower().strip()
    env_var = PROVIDER_ENV_MAP.get(provider)
    if not env_var:
        darwin.error(f"Unknown provider: '{provider}'")
        return None
    try:
        keyring.delete_password(APP_NAME, provider)
    except keyring.errors.PasswordDeleteError:
        pass
    os.environ.pop(env_var, None)
    darwin.info(f"Removed API key for {provider}")


def remove_key(provider: str) -> bool:
    """Remove an API key. Returns True on success, False on failure."""
    provider = provider.lower().strip()
    env_var = PROVIDER_ENV_MAP.get(provider)
    if not env_var:
        darwin.error(f"Unknown provider: '{provider}'")
        return False
    try:
        keyring.delete_password(APP_NAME, provider)
    except keyring.errors.PasswordDeleteError:
        pass
    except Exception as e:
        darwin.error(f"Failed to remove key for {provider}: {e}")
        return False
    os.environ.pop(env_var, None)
    darwin.info(f"Removed API key for {provider}")
    return True