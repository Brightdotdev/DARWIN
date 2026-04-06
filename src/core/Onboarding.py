"""
src/core/Onboarding.py
----------------------
First-run onboarding wizard for Darwin.

Guides the user through:
    1. LLM provider selection (arrow keys)
    2. Model selection for chosen provider
    3. API key entry (stored in OS keychain via keyring)
    4. Embedding model selection
    5. Reranker selection
    6. Saves everything to .darwin/config.json

Also handles the `darwin config` command for changing
settings after initial onboarding.

OS keychain storage:
    Windows → Credential Manager
    macOS   → Keychain
    Linux   → libsecret / KWallet

Keys are never written to disk as plaintext.
Config stores provider + model name only, never the key.

Usage:
    from src.core.Onboarding import run_onboarding, run_config
    run_onboarding(store)   # first run
    run_config(store)       # change settings later
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
from rich.panel   import Panel
from rich.text    import Text

from src.lib.Terminal import TerminalUI

console = Console()

# ── keyring service name ───────────────────────────────────────────────────────
# internal identifier — user never sees this
KEYRING_SERVICE = "darwin-ai"


# ── Model catalogue ────────────────────────────────────────────────────────────

@dataclass
class ModelOption:
    id:          str    # what gets saved to config + passed to LiteLLM
    label:       str    # what the user sees
    note:        str    # extra context shown next to the option
    is_default:  bool = False


@dataclass
class ProviderOption:
    id:          str
    label:       str
    note:        str
    needs_key:   bool
    key_name:    str        # keyring key name e.g. "openrouter_api_key"
    key_label:   str        # human label e.g. "OpenRouter API key"
    key_url:     str        # where to get the key
    models:      list[ModelOption]
    is_default:  bool = False


PROVIDERS: list[ProviderOption] = [
    ProviderOption(
        id        = "openrouter",
        label     = "OpenRouter",
        note      = "recommended — free models available",
        needs_key = True,
        key_name  = "openrouter_api_key",
        key_label = "OpenRouter API key",
        key_url   = "https://openrouter.ai/keys",
        is_default = True,
        models    = [
            ModelOption(
                id       = "openrouter/meta-llama/llama-3.2-3b-instruct:free",
                label    = "Llama 3.2 3B",
                note     = "free tier, fast, good for extraction",
                is_default = True,
            ),
            ModelOption(
                id       = "openrouter/meta-llama/llama-3.1-8b-instruct:free",
                label    = "Llama 3.1 8B",
                note     = "free tier, smarter, slightly slower",
            ),
            ModelOption(
                id       = "openrouter/google/gemma-3-4b-it:free",
                label    = "Gemma 3 4B",
                note     = "free tier, Google model",
            ),
            ModelOption(
                id       = "openrouter/mistralai/mistral-7b-instruct:free",
                label    = "Mistral 7B",
                note     = "free tier, strong instruction following",
            ),
            ModelOption(
                id       = "openrouter/anthropic/claude-haiku-4-5",
                label    = "Claude Haiku",
                note     = "paid, fast and cheap, excellent quality",
            ),
            ModelOption(
                id       = "openrouter/openai/gpt-4o-mini",
                label    = "GPT-4o Mini",
                note     = "paid, fast, good quality",
            ),
        ],
    ),
    ProviderOption(
        id        = "ollama",
        label     = "Ollama",
        note      = "fully local, no key needed, no internet",
        needs_key = False,
        key_name  = "",
        key_label = "",
        key_url   = "",
        models    = [
            ModelOption(
                id       = "ollama/llama3.2",
                label    = "Llama 3.2 (3B)",
                note     = "run: ollama pull llama3.2",
                is_default = True,
            ),
            ModelOption(
                id       = "ollama/llama3.1",
                label    = "Llama 3.1 (8B)",
                note     = "run: ollama pull llama3.1",
            ),
            ModelOption(
                id       = "ollama/mistral",
                label    = "Mistral 7B",
                note     = "run: ollama pull mistral",
            ),
            ModelOption(
                id       = "ollama/gemma3",
                label    = "Gemma 3",
                note     = "run: ollama pull gemma3",
            ),
        ],
    ),
    ProviderOption(
        id        = "anthropic",
        label     = "Anthropic",
        note      = "Claude models, needs API key",
        needs_key = True,
        key_name  = "anthropic_api_key",
        key_label = "Anthropic API key",
        key_url   = "https://console.anthropic.com/keys",
        models    = [
            ModelOption(
                id       = "claude-haiku-4-5-20251001",
                label    = "Claude Haiku",
                note     = "fast, cheap, great for extraction",
                is_default = True,
            ),
            ModelOption(
                id       = "claude-sonnet-4-6",
                label    = "Claude Sonnet",
                note     = "smarter, costs more",
            ),
        ],
    ),
    ProviderOption(
        id        = "openai",
        label     = "OpenAI",
        note      = "GPT models, needs API key",
        needs_key = True,
        key_name  = "openai_api_key",
        key_label = "OpenAI API key",
        key_url   = "https://platform.openai.com/api-keys",
        models    = [
            ModelOption(
                id       = "gpt-4o-mini",
                label    = "GPT-4o Mini",
                note     = "fast, cheap, good quality",
                is_default = True,
            ),
            ModelOption(
                id       = "gpt-4o",
                label    = "GPT-4o",
                note     = "smarter, costs more",
            ),
        ],
    ),
    ProviderOption(
        id        = "deepseek",
        label     = "DeepSeek",
        note      = "DeepSeek models, needs API key",
        needs_key = True,
        key_name  = "deepseek_api_key",
        key_label = "DeepSeek API key",
        key_url   = "https://platform.deepseek.com/api_keys",
        models    = [
            ModelOption(
                id       = "deepseek/deepseek-chat",
                label    = "DeepSeek V3",
                note     = "fast, very affordable",
                is_default = True,
            ),
            ModelOption(
                id       = "deepseek/deepseek-reasoner",
                label    = "DeepSeek R1",
                note     = "reasoning model, slower",
            ),
        ],
    ),
    ProviderOption(
        id        = "none",
        label     = "None",
        note      = "skip LLM — raw graph only, no concept extraction",
        needs_key = False,
        key_name  = "",
        key_label = "",
        key_url   = "",
        models    = [],
    ),
]

EMBEDDING_MODELS: list[ModelOption] = [
    ModelOption(
        id       = "intfloat/e5-small-v2",
        label    = "e5-small",
        note     = "light — 8GB RAM, default, fast",
        is_default = True,
    ),
    ModelOption(
        id       = "BAAI/bge-m3",
        label    = "BGE-M3",
        note     = "medium — 16GB RAM, better quality",
    ),
]

RERANKER_MODELS: list[ModelOption] = [
    ModelOption(
        id       = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        label    = "TinyBERT",
        note     = "light — fast, low memory, default",
        is_default = True,
    ),
    ModelOption(
        id       = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        label    = "MiniLM L6",
        note     = "medium — better quality, slightly slower",
    ),
]


# ── Keyring helpers ────────────────────────────────────────────────────────────

def save_api_key(key_name: str, api_key: str) -> bool:
    """Save API key to OS keychain. Returns True on success."""
    try:
        import keyring
        keyring.set_password(KEYRING_SERVICE, key_name, api_key)
        return True
    except Exception as e:
        console.print(f"{TerminalUI.c(TerminalUI.YELLOW, 'warning:')} could not save to keychain: {e}")
        console.print(TerminalUI.c(TerminalUI.DIM, "key will work this session but won't persist"))
        return False


def get_api_key(key_name: str) -> str | None:
    """Retrieve API key from OS keychain."""
    try:
        import keyring
        return keyring.get_password(KEYRING_SERVICE, key_name)
    except Exception:
        return None


def delete_api_key(key_name: str) -> None:
    """Remove API key from OS keychain."""
    try:
        import keyring
        keyring.delete_password(KEYRING_SERVICE, key_name)
    except Exception:
        pass


def has_api_key(key_name: str) -> bool:
    """Check if API key exists in keychain."""
    return bool(get_api_key(key_name))


# ── Arrow key selection ────────────────────────────────────────────────────────

def _select(
    prompt:   str,
    options:  list,
    label_fn,
    note_fn,
    default_idx: int = 0,
) -> int:
    """
    Arrow key selection menu.
    Returns index of selected option.

    Uses questionary if available, falls back to numbered list.
    questionary gives the clean arrow-key UX.
    """
    try:
        import questionary
        from questionary import Choice

        choices = [
            Choice(
                title = f"{label_fn(o)}  {TerminalUI.c(TerminalUI.DIM, note_fn(o))}"
                        if note_fn(o) else label_fn(o),
                value = i,
            )
            for i, o in enumerate(options)
        ]

        result = questionary.select(
            prompt,
            choices  = choices,
            default  = choices[default_idx],
            style    = questionary.Style([
                ("selected",        "fg:#a78bfa bold"),
                ("pointer",         "fg:#7c3aed bold"),
                ("highlighted",     "fg:#a78bfa"),
                ("answer",          "fg:#a78bfa bold"),
                ("question",        "fg:#ffffff bold"),
                ("instruction",     "fg:#4a4a6a"),
            ]),
        ).ask()

        return result if result is not None else default_idx

    except ImportError:
        # fallback — numbered list
        console.print(f"\n[bold]{prompt}[/bold]")
        for i, o in enumerate(options):
            marker = "❯" if i == default_idx else " "
            note   = f"  {TerminalUI.c(TerminalUI.DIM, note_fn(o))}" if note_fn(o) else ""
            console.print(f"  {marker} [bold cyan]{i + 1}[/bold cyan]  {label_fn(o)}{note}")

        console.print()
        while True:
            raw = console.input(
                f"  {TerminalUI.c(TerminalUI.DIM, f'enter number (1-{len(options)}, default {default_idx + 1}):')} "
            ).strip()

            if not raw:
                return default_idx

            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return idx
                console.print(f"  {TerminalUI.c(TerminalUI.RED, f'please enter a number between 1 and {len(options)}')}")
            except ValueError:
                console.print(f"  {TerminalUI.c(TerminalUI.RED, 'please enter a number')}")


# ── Splash for onboarding ──────────────────────────────────────────────────────

def _print_onboarding_splash() -> None:
    console.print()
    for line in TerminalUI.ASCII:
        console.print(f"[cyan]{line}[/cyan]")
    console.print("\n")
    console.print(Panel(
        "[bold]welcome to darwin[/bold]\n\n"
        "let's get you set up. this takes about 2 minutes.\n"
        "you can change any of these settings later with:\n"
        "[cyan]python main.py config[/cyan]",
        border_style = "dim",
        padding      = (1, 2),
    ))
    console.print()

# ── Individual config steps ────────────────────────────────────────────────────

def _step_llm() -> tuple[str, str, str]:
    """
    Guide user through LLM provider + model selection.
    Returns (provider_id, model_id, key_name)
    """
    console.print(TerminalUI.c(TerminalUI.BOLD, "step 1 of 3 — language model"))
    console.print(TerminalUI.c(TerminalUI.DIM, "used for concept extraction and graph refinement\n"))

    # provider selection
    default_provider = next(
        (i for i, p in enumerate(PROVIDERS) if p.is_default), 0
    )
    provider_idx = _select(
        prompt      = "which LLM provider?",
        options     = PROVIDERS,
        label_fn    = lambda p: p.label,
        note_fn     = lambda p: p.note,
        default_idx = default_provider,
    )
    provider = PROVIDERS[provider_idx]

    # none — skip everything
    if provider.id == "none":
        console.print(
            f"\n  {TerminalUI.c(TerminalUI.DIM, 'no LLM selected — graph will use raw chunk text as labels')}"
        )
        console.print(
            f"  {TerminalUI.c(TerminalUI.DIM, 'you can enable an LLM later with: python main.py config')}\n"
        )
        return "none", "", ""

    # ollama — no key needed
    if provider.id == "ollama":
        console.print(
            f"\n  {TerminalUI.c(TerminalUI.DIM, 'make sure Ollama is running: ollama serve')}"
        )

    # api providers — ask for key
    if provider.needs_key:
        console.print()
        existing = get_api_key(provider.key_name)

        if existing:
            masked = existing[:8] + "..." + existing[-4:]
            console.print(
                f"  {TerminalUI.c(TerminalUI.GREEN, '✓')} found existing {provider.key_label}: {TerminalUI.c(TerminalUI.DIM, masked)}"
            )
            change = _select(
                prompt      = "use existing key?",
                options     = ["yes, use existing key", "no, enter a new key"],
                label_fn    = lambda x: x,
                note_fn     = lambda x: "",
                default_idx = 0,
            )
            if change == 1:
                existing = None

        if not existing:
            console.print(
                f"\n  get your key at: [cyan]{provider.key_url}[/cyan]"
            )
            console.print()
            while True:
                key = console.input(
                    f"  {TerminalUI.c(TerminalUI.BOLD, f'{provider.key_label}:')} "
                ).strip()

                if not key:
                    console.print(f"  {TerminalUI.c(TerminalUI.RED, 'key cannot be empty')}")
                    continue

                if len(key) < 10:
                    console.print(f"  {TerminalUI.c(TerminalUI.RED, 'that does not look like a valid key')}")
                    continue

                save_api_key(provider.key_name, key)
                console.print(
                    f"  {TerminalUI.c(TerminalUI.GREEN, '✓')} key saved to OS keychain securely"
                )
                break

    console.print()

    # model selection
    if not provider.models:
        return provider.id, "", provider.key_name

    default_model = next(
        (i for i, m in enumerate(provider.models) if m.is_default), 0
    )
    model_idx = _select(
        prompt      = f"which {provider.label} model?",
        options     = provider.models,
        label_fn    = lambda m: m.label,
        note_fn     = lambda m: m.note,
        default_idx = default_model,
    )
    model = provider.models[model_idx]

    console.print(
        f"\n  {TerminalUI.c(TerminalUI.GREEN, '✓')} {provider.label} / {model.label}\n"
    )
    return provider.id, model.id, provider.key_name


def _step_embedding() -> str:
    """Guide user through embedding model selection. Returns model id."""
    console.print(TerminalUI.c(TerminalUI.BOLD, "step 2 of 3 — embedding model"))
    console.print(
        f"{TerminalUI.c(TerminalUI.DIM, 'converts text to vectors for semantic search — runs locally')}\n"
    )

    default = next(
        (i for i, m in enumerate(EMBEDDING_MODELS) if m.is_default), 0
    )
    idx = _select(
        prompt      = "which embedding model?",
        options     = EMBEDDING_MODELS,
        label_fn    = lambda m: m.label,
        note_fn     = lambda m: m.note,
        default_idx = default,
    )
    model = EMBEDDING_MODELS[idx]
    console.print(f"\n  {TerminalUI.c(TerminalUI.GREEN, '✓')} {model.label}\n")
    return model.id


def _step_reranker() -> str:
    """Guide user through reranker selection. Returns model id."""
    console.print(TerminalUI.c(TerminalUI.BOLD, "step 3 of 3 — reranker"))
    console.print(
        f"{TerminalUI.c(TerminalUI.DIM, 'improves search result ordering — runs locally')}\n"
    )

    default = next(
        (i for i, m in enumerate(RERANKER_MODELS) if m.is_default), 0
    )
    idx = _select(
        prompt      = "which reranker?",
        options     = RERANKER_MODELS,
        label_fn    = lambda m: m.label,
        note_fn     = lambda m: m.note,
        default_idx = default,
    )
    model = RERANKER_MODELS[idx]
    console.print(f"\n  {TerminalUI.c(TerminalUI.GREEN, '✓')} {model.label}\n")
    return model.id


def _print_summary(config: dict) -> None:
    """Show final config summary."""
    llm_mode  = config.get("llm_mode", "none")
    llm_model = config.get("llm_model", "")
    embedding = config.get("embedding_model", "")
    reranker  = config.get("reranker_model", "")

    console.print(Panel(
        f"[bold green]darwin is ready[/bold green]\n\n"
        f"  [dim]llm[/dim]        {llm_model or 'none'}\n"
        f"  [dim]embedding[/dim]  {embedding}\n"
        f"  [dim]reranker[/dim]   {reranker}\n\n"
        f"[dim]change any setting later:[/dim]\n"
        f"[cyan]  python main.py config[/cyan]",
        border_style = "green",
        padding      = (1, 2),
    ))
    console.print()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_onboarding(store) -> dict:
    """
    Run the full first-time onboarding wizard.
    Saves results to .darwin/config.json and OS keychain.

    Args:
        store: DarwinStore instance (already initialized)

    Returns:
        completed config dict
    """
    _print_onboarding_splash()

    # run steps
    provider_id, model_id, key_name = _step_llm()
    embedding_model                  = _step_embedding()
    reranker_model                   = _step_reranker()

    # determine llm_mode from provider
    if provider_id == "none":
        llm_mode = "none"
    elif provider_id == "ollama":
        llm_mode = "local"
    else:
        llm_mode = "api"

    # build config
    config = store.load_config()
    config.update({
        "llm_mode":        llm_mode,
        "llm_provider":    provider_id,
        "llm_model":       model_id,
        "llm_key_name":    key_name,
        "embedding_model": embedding_model,
        "reranker_model":  reranker_model,
        "onboarded":       True,
    })
    store.save_config(config)

    _print_summary(config)
    return config


def run_config(store) -> dict:
    """
    Change settings after initial onboarding.
    Shows current values and lets user update any setting.

    Args:
        store: DarwinStore instance

    Returns:
        updated config dict
    """
    config = store.load_config()

    console.print()
    console.print(TerminalUI.c(TerminalUI.BOLD, "darwin config\n"))

    # show current settings
    console.print("  current settings:\n")
    console.print(f"  [dim]llm[/dim]        {config.get('llm_model') or 'none provided'}")
    console.print(f"  [dim]embedding[/dim]  {config.get('embedding_model', 'none...try running onboarding if this is empty')}")
    console.print(f"  [dim]reranker[/dim]   {config.get('reranker_model', 'none...try running onboarding if this is empty ')}")
    console.print()

    # what to change
    options = [
        "LLM provider and model",
        "embedding model",
        "reranker",
        "nothing — exit",
    ]

    while True:
        idx = _select(
            prompt      = "what would you like to change?",
            options     = options,
            label_fn    = lambda x: x,
            note_fn     = lambda x: "",
            default_idx = 3,
        )

        if idx == 0:
            provider_id, model_id, key_name = _step_llm()
            if provider_id == "none":
                llm_mode = "none"
            elif provider_id == "ollama":
                llm_mode = "local"
            else:
                llm_mode = "api"
            config.update({
                "llm_mode":     llm_mode,
                "llm_provider": provider_id,
                "llm_model":    model_id,
                "llm_key_name": key_name,
            })

        elif idx == 1:
            config["embedding_model"] = _step_embedding()

        elif idx == 2:
            config["reranker_model"] = _step_reranker()

        else:
            break

        store.save_config(config)
        console.print(f"  {TerminalUI.c(TerminalUI.GREEN, '✓')} saved\n")

    return config


def needs_onboarding(store) -> bool:
    """Returns True if onboarding hasn't been completed yet."""
    if not store.is_initialized():
        return True
    config = store.load_config()
    return not config.get("onboarded", False)

def get_llm_config(store) -> dict:
    """
    Get LLM config with API key resolved from keychain.
    This is what LLM.py calls to get its full config.

    Returns dict with:
        llm_mode, llm_model, api_key (from keychain or None), llm_provider
    """
    config = store.load_config()
    key_name = config.get("llm_key_name", "")
    
    # Special case for Ollama - no key needed
    if config.get("llm_provider") == "ollama":
        api_key = None
    else:
        api_key = get_api_key(key_name) if key_name else None
    
    # Debug output
    console.print(f"[dim]DEBUG: llm_mode={config.get('llm_mode')}, provider={config.get('llm_provider')}, model={config.get('llm_model')}[/dim]")
    if key_name:
        console.print(f"[dim]DEBUG: key_name={key_name}, has_key={api_key is not None}[/dim]")
    
    return {
        "llm_mode":  config.get("llm_mode", "none"),
        "llm_model": config.get("llm_model", ""),
        "llm_provider": config.get("llm_provider", ""),
        "api_key":   api_key,
    }