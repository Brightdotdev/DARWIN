"""
src/Utils/api_key_manager.py
Secure API key management using OS keychain.
"""

import os
import logging

import keyring
import litellm

from src.core.Logger.DarwinLogger import darwin , LogEvent
APP_NAME = "darwin"

# --------------------------------------------------------------------------------
# Validation functions – each makes a cheap, authenticated API call to verify the key
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


# --------------------------------------------------------------------------------
# Registry of supported providers – maps provider name → (env var, validator)
# --------------------------------------------------------------------------------
PROVIDER_REGISTRY = {
    "openai":      ("OPENAI_API_KEY",      _validate_openai),
    "anthropic":   ("ANTHROPIC_API_KEY",   _validate_anthropic),
    "google":      ("GEMINI_API_KEY",      _validate_google),
    "mistral":     ("MISTRAL_API_KEY",     _validate_mistral),
    "groq":        ("GROQ_API_KEY",        _validate_groq),
    "cohere":      ("COHERE_API_KEY",      _validate_cohere),
    "together":    ("TOGETHER_API_KEY",    _validate_together),
    "openrouter":  ("OPENROUTER_API_KEY",  _validate_openrouter),
}

# Also used by LLMClient for early‑warning logging (optional, kept for convenience)
PROVIDER_ENV_MAP = {name: env for name, (env, _) in PROVIDER_REGISTRY.items()}
# Note: Ollama is handled separately (no API key required).


# --------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------

def set_api_key(provider: str, key: str, validate: bool = True) -> None:
    """
    Save an API key for `provider` into the OS keychain and set it
    for the current process. Optionally validate before saving.
    """
    provider = provider.lower()
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Supported: {list(PROVIDER_REGISTRY)}"
        )
    if not key.strip():
        raise ValueError("API key must not be empty")

    env_var, validator = PROVIDER_REGISTRY[provider]

    if validate:
        darwin.log(f"Validating key for {provider}…")
        try:
            validator(key)
            darwin.log("Key is valid.")
        except Exception as e:
            raise ValueError(f"Invalid API key for {provider}: {e}") from e

    # Persist to keychain
    try:
        keyring.set_password(APP_NAME, provider, key)
    except Exception as e:
        raise RuntimeError(f"Failed to store key in keychain: {e}") from e

    # Set environment variable for the running process
    os.environ[env_var] = key
    darwin.log(f"API key for {provider} saved and activated.")


def load_keys_from_keychain(log: bool = False) -> None:
    """
    Load all stored API keys from keychain into environment variables.
    Skips any key already set in the environment (respects manual overrides).
    Call once at application startup.
    """
    for provider, (env_var, _) in PROVIDER_REGISTRY.items():
        if env_var in os.environ:
            if log:
                darwin.log(f"{env_var} already set, skipping keychain.")
            continue
        try:
            key = keyring.get_password(APP_NAME, provider)
        except Exception as e:
            if log:
                darwin.log(f"Could not read {provider} from keychain: {e}")
            continue
        if key:
            os.environ[env_var] = key
            if log:
                darwin.log(f"Loaded {env_var} from keychain.")
        else:
            if log:
                darwin.log(f"No saved key for {provider} in keychain.")


def clear_api_key(provider: str) -> None:
    """Remove a saved API key from the keychain and the current environment."""
    provider = provider.lower()
    env_var, _ = PROVIDER_REGISTRY.get(provider, (None, None))
    if not env_var:
        raise ValueError(f"Unknown provider: '{provider}'")
    try:
        keyring.delete_password(APP_NAME, provider)
    except keyring.errors.PasswordDeleteError:
        pass  # already gone
    os.environ.pop(env_var, None)
    darwin.log(f"Removed API key for {provider}.")