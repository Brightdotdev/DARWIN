"""
src/lib/KeyManager.py
Secure API key management using OS keychain.
"""

import os
import logging

import keyring
import litellm

from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.LLM.LLMConfig import PROVIDER_ENV_MAP

APP_NAME = "darwin"


# --------------------------------------------------------------------------------
# Validation functions – each makes a cheap, authenticated API call to verify
# --------------------------------------------------------------------------------

def _validate_openai(key: str) -> bool:
    from openai import OpenAI
    OpenAI(api_key=key).models.list(limit=1, timeout=10)
    return True


def _validate_anthropic(key: str) -> bool:
    litellm.completion(
        model="anthropic/claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1, api_key=key, timeout=10, num_retries=1,
    )
    return True


def _validate_google(key: str) -> bool:
    litellm.completion(
        model="gemini/gemini-2.0-flash",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1, api_key=key, timeout=10, num_retries=1,
    )
    return True


def _validate_mistral(key: str) -> bool:
    litellm.completion(
        model="mistral/mistral-tiny",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1, api_key=key, timeout=10, num_retries=1,
    )
    return True


def _validate_groq(key: str) -> bool:
    litellm.completion(
        model="groq/llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1, api_key=key, timeout=10, num_retries=1,
    )
    return True


def _validate_cohere(key: str) -> bool:
    import cohere
    cohere.Client(api_key=key).list_models()
    return True


def _validate_together(key: str) -> bool:
    litellm.completion(
        model="together/mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1, api_key=key, timeout=10, num_retries=1,
    )
    return True


def _validate_openrouter(key: str) -> bool:
    litellm.completion(
        model="openrouter/openai/gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1, api_key=key, timeout=10, num_retries=1,
    )
    return True


def _validate_deepseek(key: str) -> bool:
    litellm.completion(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1, api_key=key, timeout=10, num_retries=1,
    )
    return True


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
        raise ValueError(
            f"Unsupported provider: '{provider}'. Supported: {list(PROVIDER_REGISTRY)}"
        )
    key = key.strip()
    if not key:
        raise ValueError("API key must not be empty")

    env_var, validator = PROVIDER_REGISTRY[provider]

    if validate:
        darwin.log(f"Validating key for {provider}…")
        try:
            validator(key)
            darwin.log("Key is valid.")
        except Exception as e:
            raise ValueError(f"Invalid API key for {provider}: {e}") from e

    try:
        keyring.set_password(APP_NAME, provider, key)
    except Exception as e:
        raise RuntimeError(f"Failed to store key in keychain: {e}") from e

    os.environ[env_var] = key
    darwin.log(f"API key for {provider} saved and activated.")



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
                darwin.log(f"Keychain error for {provider}: {e}", LogType.WARNING)
            continue

        if key and key.strip():
            os.environ[env_var] = key.strip()
            loaded += 1
            if log:
                darwin.log(f"Loaded {env_var} from keychain.")

    if log:
        darwin.log(
            f"Keychain: {loaded} loaded, {skipped} already present.",
            LogType.INFO if loaded else LogType.WARNING,
        )

def clear_api_key(provider: str) -> None:
    """Remove a saved API key from the keychain and the current environment."""
    provider = provider.lower().strip()
    env_var = PROVIDER_ENV_MAP.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: '{provider}'")
    try:
        keyring.delete_password(APP_NAME, provider)
    except keyring.errors.PasswordDeleteError:
        pass
    os.environ.pop(env_var, None)
    darwin.log(f"Removed API key for {provider}.")