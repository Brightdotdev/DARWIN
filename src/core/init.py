"""
darwin init — First-time bootstrap.

Creates ~/.darwin/, initialises store, downloads required models,
prints next steps.
"""


from rich.console import Console
from rich.panel import Panel

console = Console()


def run_init(profile: str = "required", no_download: bool = False):
    console.print(Panel.fit("[bold blue]Darwin Init[/bold blue]"))

    # 1. Store
    console.print("\n[bold]1/3[/bold] Initialising store...")
    from src.core.Storage.DarwinStore import DarwinStore

    store = DarwinStore()
    store.bootstrap(quiet=True)
    console.print(f"  Store: [green]{store.store_name}[/green] @ {store.darwin_dir}")
    console.print(
        f"  Sources: {len(store.get_pipeline_report() or [])} files registered"
    )

    # 2. Models
    console.print(f"\n[bold]2/3[/bold] Setting up {profile} models...")
    from src.core.Orchestrators.ModelOrchestrator import modelsOrchestrator

    if no_download:
        console.print("  [dim]--no-download set, skipping model downloads[/dim]")
    else:
        results = modelsOrchestrator.setup_profile(profile=profile, auto_download=True)
        for cat, res in results.items():
            if res.success:
                console.print(f"  [green]✓[/green] {cat.value}: {res.model_id}")
            else:
                console.print(f"  [red]✗[/red] {cat.value}: {res.error or 'failed'}")

    # 3. API key check
    console.print(f"\n[bold]3/3[/bold] Checking API keys...")
    statuses = modelsOrchestrator.list_key_status()
    api_models = [
        s for s in modelsOrchestrator.list_registered() if s.get("requires_provider")
    ]
    providers_needed = set()
    for s in api_models:
        if s.get("requires_provider") and not statuses.get(s["requires_provider"]):
            providers_needed.add(s["requires_provider"])

    if providers_needed:
        console.print(
            f"  [yellow]⚠[/yellow] API keys needed: {', '.join(providers_needed)}"
        )
        console.print(
            f"  Set them:  [cyan]darwin admin models keys set <provider> <key>[/cyan]"
        )
    else:
        console.print(
            "  [green]✓[/green] All API keys configured"
            if any(statuses.values())
            else "  [dim]No API keys needed (local models sufficient)[/dim]"
        )

    # Next steps
    console.print(
        Panel.fit(
            "[bold]Next steps[/bold]\n\n"
            "  [cyan]darwin mcp[/cyan]           Start MCP server (for AI agents)\n"
            "  [cyan]darwin admin[/cyan]         Full admin control (models, store, pipeline)\n"
            "  [cyan]darwin ingest ./docs[/cyan] Ingest documents\n"
            "  [cyan]darwin search <query>[/cyan] Quick search",
            border_style="green",
        )
    )
