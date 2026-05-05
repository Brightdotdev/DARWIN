"""
src/core/Knowledge/main.py
---------------------------
Combined graph + wiki pipeline CLI.

Usage:
  uv run src/core/Knowledge/main.py -f biology.pdf chemistry.pdf
  uv run src/core/Knowledge/main.py -f biology.pdf --mode summary --llm ollama/llama3.2
  uv run src/core/Knowledge/main.py                             # process all stored files
  uv run src/core/Knowledge/main.py -f biology.pdf --graph-only
  uv run src/core/Knowledge/main.py -f biology.pdf --wiki-only
"""

import sys
import argparse
import asyncio
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.Knowledge.lib import KnowledgePipelineArgs
from src.features.Knowledge.Graph.main import handle_graph_extraction
from src.features.Knowledge.Wiki.main import handle_wiki
from src.features.Knowledge.Graph.GraphUtils import (
    GraphExtractionArgs,
    DEFAULT_MIN_CONCEPT_FREQ,
    DEFAULT_MAX_CONCEPTS as GRAPH_MAX_CONCEPTS,
)
from src.features.Knowledge.Wiki.WikiUtils import (
    WikiGenerationArgs,
    DEFAULT_CHUNKS_PER_CONCEPT,
)
from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket


# ── file resolution ───────────────────────────────────────────────────────────

def _resolve_files(raw: list[str], store: DarwinStore) -> list[str]:
    """
    Accept paths or bare filenames, return canonical store filenames
    that actually have chunks ready.
    """
    resolved: list[str] = []
    for r in raw:
        candidate = Path(r).name
        if store.has_chunks(candidate):
            resolved.append(candidate)
        else:
            darwin.log(f"'{r}' → '{candidate}' has no chunks — skipping", LogType.WARNING)
    # deduplicate, preserve order
    seen: set[str] = set()
    return [f for f in resolved if not seen.add(f) and f not in seen - {f}]


def _all_chunked(store: DarwinStore) -> list[str]:
    status = store.status()
    return [
        fname for fname, info in status.get("files", {}).items()
        if info.get("is_chunked", False)
    ]


# ── pass 1: graphs ────────────────────────────────────────────────────────────

def _run_graphs(args : KnowledgePipelineArgs) -> list[str]  | None:
    """
    Run graph extraction for every file.
    Returns the list of filenames that ended up with a valid graph
    (either freshly extracted or already existing).
    """
   
        
    graph_args = GraphExtractionArgs(
            files=args.files or [], 
            min_freq=args.min_freq,
            max_concepts=args.graph_max_concepts,
            similarity_threshold=args.threshold,
            elevation_mode=args.elevation,
            visualize=args.visualize,
            inspect=args.inspect,
            json_output=args.json_output,
            rerun=args.rerun,
        )

    return handle_graph_extraction(graph_args)


# ── pass 2: wikis ─────────────────────────────────────────────────────────────

def _run_wikis( args : KnowledgePipelineArgs) -> list[str] | None:
        
        wiki_args = WikiGenerationArgs(
            files=args.files or [],
            mode=args.wiki_mode,
            llm_model=args.llm,
            chunks_per_concept=args.chunks_per_concept,
            inspect=args.inspect,
            rerun=args.rerun,
        )
        
        return handle_wiki(wiki_args)

# ── main handler ──────────────────────────────────────────────────────────────

def handle_knowledge(args : KnowledgePipelineArgs) -> None:
    store = DarwinStore()

    # resolve filenames
    if args.files:
        filenames = _resolve_files(args.files, store)
    else:
        filenames = _all_chunked(store)

    if not filenames:
        darwin.log("No chunked files found. Run chunking first.", LogType.ERROR)
        return

    darwin.log(f"Knowledge pipeline — {len(filenames)} file(s)", LogType.INFO)
    darwin.log(
        f"graph={not args.wiki_only} | "
        f"wiki={not args.graph_only} | "
        f"elevation={args.elevation} | "
        f"wiki_mode={args.wiki_mode}"
    )

    graph_ready = filenames  # assume all ready if skipping graph pass

    if not args.wiki_only:
        graph_ready = _run_graphs(args)

    if not args.graph_only and graph_ready:
        _run_wikis(args)
    elif not graph_ready:
        darwin.log("No files with valid graphs — skipping wiki pass.", LogType.WARNING)


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run graph extraction and wiki generation in one pipeline"
    )

    # file selection
    parser.add_argument("--file", "-f", nargs="+", default=None,
                        help="Files to process (paths or bare filenames). Omit for all.")

    # pass control
    parser.add_argument("--graph-only", action="store_true",
                        help="Run graph extraction only, skip wiki")
    parser.add_argument("--wiki-only", action="store_true",
                        help="Run wiki generation only (graphs must already exist)")


    parser.add_argument("--elevation", "-e", default="none",
                        choices=["spacy", "llm", "none"])
    parser.add_argument("--threshold", "-t", type=float, default=None)
    parser.add_argument("--min-freq", type=int, default=DEFAULT_MIN_CONCEPT_FREQ)
    parser.add_argument("--graph-max-concepts", type=int, default=GRAPH_MAX_CONCEPTS)
    parser.add_argument("--visualize", "-v", action="store_true")

    # wiki args
    parser.add_argument("--wiki-mode", "--wm", default="index",
                        choices=["index", "summary"])
    parser.add_argument("--llm", default="ollama/llama3.2")
    parser.add_argument("--chunks-per-concept", type=int, default=DEFAULT_CHUNKS_PER_CONCEPT)

    # shared
    parser.add_argument("--inspect", "-i", action="store_true")
    parser.add_argument("--rerun", "-R", action="store_true")
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--no-ws", action="store_true")

    args = parser.parse_args()

    if args.graph_only and args.wiki_only:
        darwin.log("--graph-only and --wiki-only are mutually exclusive.", LogType.ERROR)
        return 1

    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        args = KnowledgePipelineArgs(
            files=args.file,
            graph_only=args.graph_only,
            wiki_only=args.wiki_only,
            elevation=args.elevation,
            threshold=args.threshold,
            min_freq=args.min_freq,
            graph_max_concepts=args.graph_max_concepts,
            visualize=args.visualize,
            wiki_mode=args.wiki_mode,
            llm=args.llm,
            chunks_per_concept=args.chunks_per_concept,
            inspect=args.inspect,
            rerun=args.rerun,
            ws_port=args.ws_port,
            no_ws=args.no_ws,
        )
        handle_knowledge(args)
        return 0
    except KeyboardInterrupt:
        darwin.log("", LogType.INFO)
        darwin.log("Goodbye!", LogType.INFO)
        return 130
    except Exception as e:
        import traceback
        darwin.log(f"Fatal error: {e}", LogType.ERROR)
        darwin.log("".join(traceback.format_exception(e)), LogType.ERROR)
        return 1
    finally:
        shutdown_websocket(ws_server)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))