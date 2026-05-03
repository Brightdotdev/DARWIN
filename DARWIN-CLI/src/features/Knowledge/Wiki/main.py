# src/core/Knowledge/Wiki/main.py - Fixed with WebSocket support

"""
src/core/Knowledge/Wiki/main.py
--------------------------------
Wiki generation CLI.

Usage:
  uv run src/core/Knowledge/Wiki/main.py --file biology.pdf
  uv run src/core/Knowledge/Wiki/main.py --file biology.pdf --inspect
  uv run src/core/Knowledge/Wiki/main.py --file biology.pdf --mode summary --llm ollama/llama3.2
  uv run src/core/Knowledge/Wiki/main.py --file biology.pdf --max-concepts 20
"""

import sys
import argparse
import asyncio
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.features.Knowledge.Wiki.WikiGenerator import run_wiki_pipeline
from src.features.Knowledge.Wiki.WikiUtils import (
    WikiGenerationArgs, DEFAULT_MAX_CONCEPTS, DEFAULT_CHUNKS_PER_CONCEPT,
)
from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket



def _print_inspection(wiki, filename: str) -> None:
    darwin.log(f"\n{'='*60}", LogType.INFO)
    darwin.log(f"Wiki — {filename}", LogType.INFO)
    darwin.log(f"{'='*60}", LogType.INFO)

    darwin.log(f"Pages: {len(wiki.pages)}", LogType.INFO)
    darwin.log(f"Manifest: {'generated' if wiki.manifest else 'pending'}", LogType.INFO)
    summaries_done = sum(1 for p in wiki.pages if p.summary)
    darwin.log(f"Summaries: {summaries_done}/{len(wiki.pages)}", LogType.INFO)

    if wiki.manifest:
        darwin.log(f"\n--- Manifest ---", LogType.INFO)
        darwin.log(wiki.manifest[:400], LogType.INFO)

    darwin.log(f"\n--- Concept Pages ---", LogType.INFO)
    for page in wiki.pages[:5]:
        related_str = ", ".join(page.related[:3]) if page.related else "—"
        darwin.log(
            f"  {page.concept} — {page.frequency} chunks | "
            f"keywords: {', '.join(page.keywords[:3]) or '—'} | "
            f"related: {related_str}",
            LogType.INFO,
        )
        if page.summary:
            darwin.log(f"  {page.summary[:150]}...", LogType.INFO)
        darwin.log("-" * 40)

    darwin.log(f"{'='*60}", LogType.INFO)

def handle_wiki(args: WikiGenerationArgs) -> None:
    """Main handler for wiki generation."""
    store = DarwinStore()

    if args.file:
        filenames = [args.file] if store.has_graph(args.file) else []
    else:
        filenames = [
            record.original_name
            for record in store.registry.list_all()
            if store.has_graph(record.original_name)
        ]

    if not filenames:
        darwin.log("No files with graphs found. Run graph extraction first.", LogType.ERROR)
        return

    darwin.log(f"Wiki generation — {len(filenames)} file(s)", LogType.INFO)
    darwin.log(f"Mode: {args.mode} | Max concepts: {args.max_concepts}")
    if args.mode == "summary":
        darwin.log(f"LLM: {args.llm_model}")

    succeeded = failed = 0

    for filename in filenames:
        try:
            wiki = run_wiki_pipeline(
                filename=filename, store=store,
                mode=args.mode, llm_model=args.llm_model,
                max_concepts=args.max_concepts,
                chunks_per_concept=args.chunks_per_concept,
                rerun=args.rerun,
            )
            if wiki:
                succeeded += 1
                darwin.log(f"{filename} → {len(wiki.pages)} pages", LogType.SUCCESS)
                if args.inspect:
                    _print_inspection(wiki, filename)
            else:
                failed += 1
        except Exception as e:
            import traceback
            darwin.log(f"{filename} — {e}", LogType.ERROR)
            darwin.log(traceback.format_exc())
            failed += 1

    darwin.log(f"Done — {succeeded} generated · {failed} failed", LogType.SUCCESS)


async def main() -> int:
    """Async entry point with WebSocket support."""
    parser = argparse.ArgumentParser(description="Generate Wikipedia-style docs from concept graphs")
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--file", "-f", default=None)
    parser.add_argument("--mode", "-m", default="index", choices=["index", "summary"])
    parser.add_argument("--llm", default="ollama/llama3.2", help="LLM model for summary mode")
    parser.add_argument("--max-concepts", "-n", type=int, default=DEFAULT_MAX_CONCEPTS)
    parser.add_argument("--chunks-per-concept", type=int, default=DEFAULT_CHUNKS_PER_CONCEPT)
    parser.add_argument("--inspect", "-i", action="store_true")
    parser.add_argument("--rerun", "-R", action="store_true")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")

    args = parser.parse_args()

    # Setup WebSocket
    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        wiki_args = WikiGenerationArgs(
            target=args.path,
            file=args.file,
            mode=args.mode,
            llm_model=args.llm,
            max_concepts=args.max_concepts,
            chunks_per_concept=args.chunks_per_concept,
            inspect=args.inspect,
            rerun=args.rerun,
        )
        
        handle_wiki(wiki_args)
        return 0

    except KeyboardInterrupt:
        darwin.log("", LogType.INFO)
        darwin.log("Goodbye!", LogType.INFO)
        return 130
    except Exception as e:
        import traceback
        darwin.log(f"Fatal error: {e}", LogType.ERROR)
        darwin.log(''.join(traceback.format_exception(e)), LogType.ERROR)
        return 1
    finally:
        shutdown_websocket(ws_server)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))