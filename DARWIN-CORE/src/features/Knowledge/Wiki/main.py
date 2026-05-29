import sys
import os
import warnings
import argparse
import asyncio
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))



from src.features.Knowledge.Wiki.WikiGenerator import run_wiki_pipeline
from src.features.Knowledge.Wiki.WikiUtils import (
    WikiGenerationArgs
)
from typing import List
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity



def _print_inspection(wiki, filename: str) -> None:
    darwin.log(f"\n{'='*60}", LogType.INFO)
    darwin.log(f"Wiki — {filename}", LogType.INFO)
    darwin.log(f"{'='*60}", LogType.INFO)

    darwin.log(f"Pages: {len(wiki.pages)}", LogType.INFO)


    darwin.log(f"\n--- Concept Pages ---", LogType.INFO)
    for page in wiki.pages[:5]:
        related_str = ", ".join(page.related[:3]) if page.related else "—"
        darwin.log(
            f"  {page.concept} — {page.frequency} chunks | "
            f"keywords: {', '.join(page.keywords[:3]) or '—'} | "
            f"related: {related_str}",
            LogType.INFO,
        )
    darwin.log(f"{'='*60}", LogType.INFO)


def handle_wiki(args: WikiGenerationArgs,store) -> List[str] | None:

    if args.files:
        filenames = [f for f in args.files if store.has_graph(f)]
        missing = [f for f in args.files if not store.has_graph(f)]
        for f in missing:
            darwin.log(f"'{f}' has no graph — skipping", LogType.WARNING)
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
    

    succeeded = failed = 0
    processed_wikis = []

    for filename in filenames:
        try:
            wiki = run_wiki_pipeline(
                filename=filename, store=store,
                rerun=args.rerun,
                completed_files=processed_wikis
            )
            if wiki:
                succeeded += 1
                darwin.log(f"{filename} → {len(wiki.pages)} pages", LogType.SUCCESS)
                processed_wikis.append(filename)
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
    return processed_wikis if processed_wikis is not None else None


async def main() -> int:
    """Async entry point with WebSocket support."""
    from src.lib.Utils import ensure_writable_directory

    parser = argparse.ArgumentParser(description="Generate Wikipedia-style docs from concept graphs")
    parser.add_argument("--file", "-f", nargs="+", default=None)
    parser.add_argument("--inspect", "-i", action="store_true")
    parser.add_argument("--rerun", "-R", action="store_true")
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")
    parser.add_argument("--verbosity", choices=["normal", "quiet"], default="normal", help="Log verbosity")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")

    args = parser.parse_args()



    # Setup WebSocket
    darwin.configure(
        enable_websocket=not args.no_ws,
        ws_port=args.ws_port,
        enable_history=True,
        verbosity_default=Verbosity.QUIET if args.verbosity == "quiet" else Verbosity.NORMAL,
        show_banner=True,
    )

    try:
        wiki_args = WikiGenerationArgs(
            files=args.file or [],
            inspect=args.inspect,
            rerun=args.rerun,
        )
        
        with darwin.session("wiki_generation"):
            
            if args.store_dir and ensure_writable_directory(args.store_dir):
                store = DarwinStore(args.store_dir)
            else:
                store = DarwinStore()
            handle_wiki(wiki_args, store)
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
        darwin.shutdown()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))