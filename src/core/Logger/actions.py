# =============================================================================
# actions.py — All business-logic callables
# Imported and registered by app.py at startup.
# Never imports TerminalUI or FrontendBridge.
# =============================================================================
from __future__ import annotations

from typing import Any, Dict

from src.core.Logger.DarwinLogger import darwin, LogType


# -----------------------------------------------------------------------------
# Action signatures:  func(choice: str, **params) -> Any
# -----------------------------------------------------------------------------

def action_select_project(choice: str, **params: Any) -> str:
    """Handle project selection prompt."""
    darwin.log(f"Project selected: {choice}", LogType.SUCCESS)
    return choice


def action_confirm_delete(choice: str, **params: Any) -> bool:
    """Handle delete confirmation prompt."""
    if choice == "yes":
        darwin.log(f"Deleted {params.get('target', 'unknown')}", LogType.WARNING)
        return True
    darwin.log("Deletion cancelled", LogType.INFO)
    return False


def action_select_model(choice: str, **params: Any) -> Dict[str, str]:
    """Handle model selection."""
    context = params.get("context", "default")
    darwin.log(f"Model '{choice}' selected for {context}", LogType.INFO)
    return {"model": choice, "context": context}


def action_resolve_path(choice: str, **params: Any) -> str:
    """Handle file path resolution."""
    darwin.log(f"Path resolved: {choice}", LogType.SUCCESS)
    return choice


# -----------------------------------------------------------------------------
# Registry map — used by app.py to register all actions at once
# -----------------------------------------------------------------------------
ACTION_REGISTRY = {
    "select_project": action_select_project,
    "confirm_delete": action_confirm_delete,
    "select_model": action_select_model,
    "resolve_path": action_resolve_path,
}