"""
src/core/Onboarding.py
----------------------
First-run onboarding wizard and config management for Darwin.

Keys stored in OS keychain via keyring — never written to disk.
Config stores provider + model name only, never the key itself.

Usage:
    from src.core.Onboarding import run_onboarding, run_config, get_llm_config
    run_onboarding(store)       # first run wizard
    run_config(store)           # change settings later
    cfg = get_llm_config(store) # get config + key for LLM.from_config()
"""

from __future__ import annotations
from dataclasses import dataclass
from rich.console import Console
from rich.panel   import Panel

console = Console()

KEYRING_SERVICE = "darwin-ai"


# ── Model catalogue ────────────────────────────────────────────────────────────

@dataclass
class ModelOption:
    id:         str
    label:      str
    note:       str
    is_default: bool = False


@dataclass
class ProviderOption:
    id:         str
    label:      str
    note:       str
    needs_key:  bool
    key_name:   str
    key_label:  str
    key_url:    str
    models:     list[ModelOption]
    is_default: bool = False


PROVIDERS: list[ProviderOption] = [
    ProviderOption(
        id="openrouter", label="OpenRouter",
        note="recommended — free models available",
        needs_key=True, key_name="openrouter_api_key",
        key_label="OpenRouter API key", key_url="https://openrouter.ai/keys",
        is_default=True,
        models=[
            ModelOption("openrouter/meta-llama/llama-3.2-3b-instruct:free",
                        "Llama 3.2 3B", "free tier, fast, good for extraction", True),
            ModelOption("openrouter/meta-llama/llama-3.1-8b-instruct:free",
                        "Llama 3.1 8B", "free tier, smarter, slightly slower"),
            ModelOption("openrouter/google/gemma-3-4b-it:free",
                        "Gemma 3 4B", "free tier, Google model"),
            ModelOption("openrouter/mistralai/mistral-7b-instruct:free",
                        "Mistral 7B", "free tier, strong instruction following"),
            ModelOption("openrouter/anthropic/claude-haiku-4-5",
                        "Claude Haiku", "paid, fast and cheap, excellent quality"),
            ModelOption("openrouter/openai/gpt-4o-mini",
                        "GPT-4o Mini", "paid, fast, good quality"),
        ],
    ),
    ProviderOption(
        id="ollama", label="Ollama",
        note="fully local, no key needed, no internet",
        needs_key=False, key_name="", key_label="", key_url="",
        models=[
            ModelOption("ollama/llama3.2", "Llama 3.2 (3B)",
                        "run: ollama pull llama3.2", True),
            ModelOption("ollama/llama3.1", "Llama 3.1 (8B)",
                        "run: ollama pull llama3.1"),
            ModelOption("ollama/mistral",  "Mistral 7B",
                        "run: ollama pull mistral"),
            ModelOption("ollama/gemma3",   "Gemma 3",
                        "run: ollama pull gemma3"),
        ],
    ),
    ProviderOption(
        id="anthropic", label="Anthropic",
        note="Claude models, needs API key",
        needs_key=True, key_name="anthropic_api_key",
        key_label="Anthropic API key",
        key_url="https://console.anthropic.com/keys",
        models=[
            ModelOption("claude-haiku-4-5-20251001", "Claude Haiku",
                        "fast, cheap, great for extraction", True),
            ModelOption("claude-sonnet-4-6", "Claude Sonnet",
                        "smarter, costs more"),
        ],
    ),
    ProviderOption(
        id="openai", label="OpenAI",
        note="GPT models, needs API key",
        needs_key=True, key_name="openai_api_key",
        key_label="OpenAI API key",
        key_url="https://platform.openai.com/api-keys",
        models=[
            ModelOption("gpt-4o-mini", "GPT-4o Mini",
                        "fast, cheap, good quality", True),
            ModelOption("gpt-4o", "GPT-4o", "smarter, costs more"),
        ],
    ),
    ProviderOption(
        id="deepseek", label="DeepSeek",
        note="DeepSeek models, needs API key",
        needs_key=True, key_name="deepseek_api_key",
        key_label="DeepSeek API key",
        key_url="https://platform.deepseek.com/api_keys",
        models=[
            ModelOption("deepseek/deepseek-chat", "DeepSeek V3",
                        "fast, very affordable", True),
            ModelOption("deepseek/deepseek-reasoner", "DeepSeek R1",
                        "reasoning model, slower"),
        ],
    ),
    ProviderOption(
        id="none", label="None",
        note="skip LLM — raw graph only, no concept extraction",
        needs_key=False, key_name="", key_label="", key_url="",
        models=[],
    ),
]

EMBEDDING_MODELS: list[ModelOption] = [
    ModelOption("intfloat/e5-small-v2", "e5-small",
                "light — 8GB RAM, default, fast", True),
    ModelOption("BAAI/bge-m3", "BGE-M3",
                "medium — 16GB RAM, better quality"),
]

RERANKER_MODELS: list[ModelOption] = [
    ModelOption("cross-encoder/ms-marco-TinyBERT-L-2-v2", "TinyBERT",
                "light — fast, low memory, default", True),
    ModelOption("cross-encoder/ms-marco-MiniLM-L-6-v2", "MiniLM L6",
                "medium — better quality, slightly slower"),
]


# ── Keyring helpers ────────────────────────────────────────────────────────────

def save_api_key(key_name: str, api_key: str) -> bool:
    try:
        import keyring
        keyring.set_password(KEYRING_SERVICE, key_name, api_key)
        return True
    except Exception as e:
        console.print(f"[yellow]warning:[/yellow] could not save to keychain: {e}")
        return False


def get_api_key(key_name: str) -> str | None:
    try:
        import keyring
        return keyring.get_password(KEYRING_SERVICE, key_name)
    except Exception:
        return None


def has_api_key(key_name: str) -> bool:
    return bool(get_api_key(key_name))


# ── Arrow key selection ────────────────────────────────────────────────────────

def _select(
    prompt:      str,
    options:     list,
    label_fn,
    note_fn,
    default_idx: int = 0,
) -> int:
    """Arrow key menu via questionary, falls back to numbered list."""
    try:
        import questionary
        from questionary import Choice

        choices = [
            Choice(
                title = f"{label_fn(o)}  \033[2m{note_fn(o)}\033[0m"
                        if note_fn(o) else label_fn(o),
                value = i,
            )
            for i, o in enumerate(options)
        ]

        result = questionary.select(
            prompt,
            choices = choices,
            default = choices[default_idx],
            style   = questionary.Style([
                ("selected",    "fg:#a78bfa bold"),
                ("pointer",     "fg:#7c3aed bold"),
                ("highlighted", "fg:#a78bfa"),
                ("answer",      "fg:#a78bfa bold"),
                ("question",    "fg:#ffffff bold"),
                ("instruction", "fg:#4a4a6a"),
            ]),
        ).ask()

        return result if result is not None else default_idx

    except ImportError:
        console.print(f"\n[bold]{prompt}[/bold]")
        for i, o in enumerate(options):
            marker = "❯" if i == default_idx else " "
            note   = f"  [dim]{note_fn(o)}[/dim]" if note_fn(o) else ""
            console.print(f"  {marker} [bold cyan]{i + 1}[/bold cyan]  {label_fn(o)}{note}")
        console.print()

        while True:
            raw = console.input(
                f"  [dim]enter number (1-{len(options)}, "
                f"default {default_idx + 1}):[/dim] "
            ).strip()

            if not raw:
                return default_idx
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return idx
                console.print(f"  [red]enter a number between 1 and {len(options)}[/red]")
            except ValueError:
                console.print("  [red]please enter a number[/red]")


# ── Splash ─────────────────────────────────────────────────────────────────────

def _splash() -> None:
    console.print()
    console.print("[cyan]    ------    [/cyan]")
    console.print("[cyan]   |      |   [/cyan]  [bold white]darwin[/bold white]")
    console.print("[cyan]   |      |   [/cyan]  [dim]your knowledge. mapped.[/dim]")
    console.print("[cyan]   |      │   [/cyan]")
    console.print("[cyan]    ------    [/cyan]")
    console.print()
    console.print(Panel(
        "[bold]welcome to darwin[/bold]\n\n"
        "let's get you set up. this takes about 2 minutes.\n"
        "you can change any of these settings later with:\n"
        "[cyan]python main.py config[/cyan]",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()


# ── Steps ──────────────────────────────────────────────────────────────────────

def _step_llm() -> tuple[str, str, str]:
    """Returns (provider_id, model_id, key_name)."""
    console.print("[bold]step 1 of 3 — language model[/bold]")
    console.print("[dim]used for concept extraction and graph refinement[/dim]\n")

    default_idx = next((i for i, p in enumerate(PROVIDERS) if p.is_default), 0)
    idx         = _select("which LLM provider?", PROVIDERS,
                          lambda p: p.label, lambda p: p.note, default_idx)
    provider    = PROVIDERS[idx]

    if provider.id == "none":
        console.print("\n  [dim]no LLM — graph will use raw chunk text as labels[/dim]")
        console.print("  [dim]enable later with: python main.py config[/dim]\n")
        return "none", "", ""

    if provider.id == "ollama":
        console.print("\n  [dim]make sure Ollama is running: ollama serve[/dim]")

    if provider.needs_key:
        console.print()
        existing = get_api_key(provider.key_name)

        if existing:
            masked = existing[:8] + "..." + existing[-4:]
            console.print(f"  [green]✓[/green] found existing {provider.key_label}: [dim]{masked}[/dim]")
            change = _select("use existing key?",
                             ["yes, use existing key", "no, enter a new key"],
                             lambda x: x, lambda x: "", 0)
            if change == 1:
                existing = None

        if not existing:
            console.print(f"\n  get your key at: [cyan]{provider.key_url}[/cyan]\n")
            while True:
                key = console.input(f"  [bold]{provider.key_label}:[/bold] ").strip()
                if not key:
                    console.print("  [red]key cannot be empty[/red]")
                    continue
                if len(key) < 10:
                    console.print("  [red]that doesn't look like a valid key[/red]")
                    continue
                save_api_key(provider.key_name, key)
                console.print("  [green]✓[/green] key saved to OS keychain securely")
                break

    console.print()

    if not provider.models:
        return provider.id, "", provider.key_name

    default_model = next((i for i, m in enumerate(provider.models) if m.is_default), 0)
    model_idx     = _select(f"which {provider.label} model?", provider.models,
                            lambda m: m.label, lambda m: m.note, default_model)
    model         = provider.models[model_idx]

    console.print(f"\n  [green]✓[/green] {provider.label} / {model.label}\n")
    return provider.id, model.id, provider.key_name


def _step_embedding() -> str:
    console.print("[bold]step 2 of 3 — embedding model[/bold]")
    console.print("[dim]converts text to vectors — runs locally[/dim]\n")

    default = next((i for i, m in enumerate(EMBEDDING_MODELS) if m.is_default), 0)
    idx     = _select("which embedding model?", EMBEDDING_MODELS,
                      lambda m: m.label, lambda m: m.note, default)
    model   = EMBEDDING_MODELS[idx]
    console.print(f"\n  [green]✓[/green] {model.label}\n")
    return model.id


def _step_reranker() -> str:
    console.print("[bold]step 3 of 3 — reranker[/bold]")
    console.print("[dim]improves search result ordering — runs locally[/dim]\n")

    default = next((i for i, m in enumerate(RERANKER_MODELS) if m.is_default), 0)
    idx     = _select("which reranker?", RERANKER_MODELS,
                      lambda m: m.label, lambda m: m.note, default)
    model   = RERANKER_MODELS[idx]
    console.print(f"\n  [green]✓[/green] {model.label}\n")
    return model.id


def _step_images() -> str:
    """
    Guide user through image handling choice.
    Returns one of: "skip" | "extract" | "caption"
    """
    console.print("[bold]step 4 of 4 — image handling[/bold]")
    console.print("[dim]PDFs often contain diagrams, charts, and figures[/dim]\n")

    options = [
        ("skip",    "skip images entirely",        "fastest, images not searchable"),
        ("extract", "extract images, no captions", "images saved but not described"),
        ("caption", "extract + caption with AI",   "downloads ~500MB SmolVLM, images become searchable"),
    ]

    idx = _select(
        prompt      = "how should Darwin handle images in PDFs?",
        options     = options,
        label_fn    = lambda x: x[1],
        note_fn     = lambda x: x[2],
        default_idx = 0,
    )
    mode = options[idx][0]

    if mode == "caption":
        console.print()
        console.print("  [dim]SmolVLM-256M downloaded on first use (~500MB)[/dim]")
        console.print("  [dim]runs on CPU — no GPU needed[/dim]")

    console.print(f"\n  [green]✓[/green] {options[idx][1]}\n")
    return mode


def _summary(config: dict) -> None:
    image_mode = config.get("image_mode", "skip")
    console.print(Panel(
        f"[bold green]darwin is ready[/bold green]\n\n"
        f"  [dim]llm[/dim]        {config.get('llm_model') or 'none'}\n"
        f"  [dim]embedding[/dim]  {config.get('embedding_model', '')}\n"
        f"  [dim]reranker[/dim]   {config.get('reranker_model', '')}\n"
        f"  [dim]images[/dim]     {image_mode}\n\n"
        f"[dim]change later:[/dim]  [cyan]python main.py config[/cyan]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_onboarding(store) -> dict:
    """Run first-time setup wizard. Saves to config.json and OS keychain."""
    _splash()

    provider_id, model_id, key_name = _step_llm()
    embedding_model                  = _step_embedding()
    reranker_model                   = _step_reranker()
    image_mode                       = _step_images()

    llm_mode = (
        "none"  if provider_id == "none"   else
        "local" if provider_id == "ollama" else
        "api"
    )

    config = store.load_config()
    config.update({
        "llm_mode":        llm_mode,
        "llm_provider":    provider_id,
        "llm_model":       model_id,
        "llm_key_name":    key_name,
        "embedding_model": embedding_model,
        "reranker_model":  reranker_model,
        "image_mode":      image_mode,
        "onboarded":       True,
    })
    store.save_config(config)
    _summary(config)
    return config


def run_config(store) -> dict:
    """Change settings after initial onboarding."""
    config = store.load_config()

    console.print()
    console.print("[bold]darwin config[/bold]\n")
    console.print("  current settings:\n")
    console.print(f"  [dim]llm[/dim]        {config.get('llm_model') or 'none'}")
    console.print(f"  [dim]embedding[/dim]  {config.get('embedding_model', 'not set')}")
    console.print(f"  [dim]reranker[/dim]   {config.get('reranker_model', 'not set')}")
    console.print()

    console.print(f"  [dim]images[/dim]     {config.get('image_mode', 'skip')}")
    console.print()

    options = [
        "LLM provider and model",
        "embedding model",
        "reranker",
        "image handling",
        "nothing — exit",
    ]

    while True:
        idx = _select("what would you like to change?", options,
                      lambda x: x, lambda x: "", 4)

        if idx == 0:
            provider_id, model_id, key_name = _step_llm()
            llm_mode = (
                "none"  if provider_id == "none"   else
                "local" if provider_id == "ollama" else
                "api"
            )
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
        elif idx == 3:
            config["image_mode"] = _step_images()
        else:
            break

        store.save_config(config)
        console.print("  [green]✓[/green] saved\n")

    return config


def needs_onboarding(store) -> bool:
    """True if onboarding hasn't been completed yet."""
    if not store.is_initialized():
        return True
    return not store.load_config().get("onboarded", False)


def get_llm_config(store) -> dict:
    """
    Get full LLM config including API key from keychain.
    Pass the result directly to LLM.from_config().

    Keys are never in config.json — this is the only place
    they get resolved from the OS keychain.
    """
    config   = store.load_config()
    key_name = config.get("llm_key_name", "")
    provider = config.get("llm_provider", "")

    # Ollama is local — no key needed
    api_key = None if provider == "ollama" else get_api_key(key_name) if key_name else None

    return {
        "llm_mode":     config.get("llm_mode",  "none"),
        "llm_model":    config.get("llm_model", ""),
        "llm_provider": provider,
        "api_key":      api_key,
    }