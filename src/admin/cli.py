"""
darwin admin — Full orchestrator control for power users.
All knobs, all dials. Wraps the four orchestrators.

Entry point:  src.admin:main()   or   darwin-admin
"""

import argparse
import sys
import traceback

from src.admin.renderer import _r


# ── Lazy orchestrator singletons ──

_store = None
_pipeline = None
_models = None


def _get_store(store_ref: str | None = None):
    global _store
    if _store is None or store_ref is not None:
        from src.core.Orchestrators.lib import resolve_store
        resolved = resolve_store(store_ref)
        if store_ref is None:
            _store = resolved
        return resolved
    return _store


def _get_pipeline(store_ref: str | None = None):
    global _pipeline
    if _pipeline is None or store_ref is not None:
        from src.core.Orchestrators.PipelineOrchestrator import PipelineOrchestrator
        _pipeline = PipelineOrchestrator(store=_get_store(store_ref))
    return _pipeline


def _get_models(init_models: bool = False):
    global _models
    if _models is None:
        from src.core.Orchestrators.ModelOrchestrator import modelsOrchestrator
        _models = modelsOrchestrator
        if init_models:
            _models.status()
    return _models


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        prog="darwin-admin",
        description="Darwin Admin — Full orchestrator control for power users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  darwin-admin setup interactive
  darwin-admin status
  darwin-admin models status
  darwin-admin models install intfloat/multilingual-e5-small-onnx
  darwin-admin models keys set openai sk-...
  darwin-admin store files
  darwin-admin store audit
  darwin-admin pipeline run ./docs --ocr
  darwin-admin pipeline purge myfile.pdf --stages ingestion
  darwin-admin system info
  darwin-admin uninstall --dry-run
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose error output")

    # Register all command groups — each module wires its own argparse + handlers
    from src.admin import setup as _setup_mod
    from src.admin import commands_models as _models_mod
    from src.admin import commands_store as _store_mod
    from src.admin import commands_pipeline as _pipeline_mod
    from src.admin import commands_logs as _logs_mod
    from src.admin import commands_system as _system_mod
    from src.admin import uninstall as _uninstall_mod

    _setup_mod.add_args(sub, _get_models)
    _models_mod.add_args(sub, _get_models)
    _store_mod.add_args(sub, _get_store)
    _pipeline_mod.add_args(sub, _get_pipeline)
    _logs_mod.add_args(sub)
    _system_mod.add_args(sub, _get_models)
    _uninstall_mod.add_args(sub)

    # Top-level status command
    from src.admin.commands_system import cmd_status
    sp = sub.add_parser("status", help="Overview: store files, installed models, preferences")
    sp.add_argument("--store", help="Store ID, name, or path")
    sp.add_argument("--initialize", action="store_true", help="Also initialize models (slow)")
    sp.set_defaults(func=cmd_status)

    args = parser.parse_args()

    from src.core.Logger.DarwinLogger import darwin
    darwin.configure(show_banner=False)

    try:
        args.func(args)
    except KeyboardInterrupt:
        _r._out()
        _r.warn("Interrupted.")
        sys.exit(130)
    except Exception as e:
        _r.err(f"{type(e).__name__}: {e}")
        if getattr(args, "verbose", False):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
