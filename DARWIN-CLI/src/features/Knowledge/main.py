"""
src/core/Knowledge/main.py
---------------------------
Combined graph + wiki pipeline CLI.

Usage:
  uv run src/core/Knowledge/main.py -f biology.pdf chemistry.pdf
  
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
from src.features.Knowledge.Graph.GraphConfig import GraphExtractionConfig, PRESETS
from src.lib.Utils import ensure_writable_directory
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

def _run_graphs(args : KnowledgePipelineArgs, store: DarwinStore) -> list[str]  | None:
    """
    Run graph extraction for every file.
    Returns the list of filenames that ended up with a valid graph
    (either freshly extracted or already existing).
    """
    graph_config = GraphExtractionConfig(
            processing_mode=args.processing_mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_concepts=args.graph_max_concepts,
            min_freq=args.min_freq,
            relation_confidence_threshold=args.relation_threshold,
            dedup_strategy=args.dedup_strategy,
            similarity_threshold=args.threshold,
            max_cooccurrence_edges=args.max_cooccurrence_edges,
            visualize=args.visualize,
            inspect=args.inspect,
            json_output=args.json_output,
            rerun=args.rerun,
        )



    return handle_graph_extraction(graph_config, args.files or [],store=store)


# ── pass 2: wikis ─────────────────────────────────────────────────────────────

def _run_wikis( args : KnowledgePipelineArgs, store: DarwinStore) -> list[str] | None:
        
        wiki_args = WikiGenerationArgs(
            files=args.files or [],
            mode=args.wiki_mode,
            provider=args.llm_provider,
            chunks_per_concept=args.chunks_per_concept,
            inspect=args.inspect,
            rerun=args.rerun,
        )
        
        return handle_wiki(wiki_args,store=store)

# ── main handler ──────────────────────────────────────────────────────────────

def handle_knowledge(args : KnowledgePipelineArgs, store_dir: str = None) -> None:

    if store_dir and ensure_writable_directory(store_dir):
        store = DarwinStore(store_dir)
    else:
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
        f"preset={args.preset or 'none'} | "
        f"wiki_mode={args.wiki_mode}"
    )

    graph_ready = filenames  # assume all ready if skipping graph pass

    if not args.wiki_only:
        graph_ready = _run_graphs(args,store=store)

    if not args.graph_only and graph_ready:
        _run_wikis(args,store=store)
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

    # graph config
    parser.add_argument("--preset", "-p", default=None,
                        choices=list(PRESETS.keys()),
                        help="Document-type preset: academic|research|legal|narrative|technical")
    parser.add_argument("--processing-mode", "-m", default="auto",
                        choices=["auto", "full", "chunked", "sliding_window"])
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=float, default=0.2)
    parser.add_argument("--relation-threshold", type=float, default=0.0)
    parser.add_argument("--dedup-strategy", default="svo_canonical",
                        choices=["svo_canonical", "strict", "none"])
    parser.add_argument("--max-cooccurrence-edges", type=int, default=500)
    parser.add_argument("--threshold", "-t", type=float, default=None)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--graph-max-concepts", type=int, default=75)
    parser.add_argument("--visualize", "-v", action="store_true")
    parser.add_argument("--store-dir", "-s", default=None, help="Store directory")


    # wiki args
    parser.add_argument("--wiki-mode", "--wm", default="index",
                        choices=["index", "summary"])
    parser.add_argument("--llm-provider")
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
        knowledge_args = KnowledgePipelineArgs(
            files=args.file,
            graph_only=args.graph_only,
            wiki_only=args.wiki_only,
            preset=args.preset,
            processing_mode=args.processing_mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            relation_threshold=args.relation_threshold,
            dedup_strategy=args.dedup_strategy,
            max_cooccurrence_edges=args.max_cooccurrence_edges,
            threshold=args.threshold,
            min_freq=args.min_freq,
            graph_max_concepts=args.graph_max_concepts,
            visualize=args.visualize,
            wiki_mode=args.wiki_mode,
            llm_provider=args.llm_provider,
            chunks_per_concept=args.chunks_per_concept,
            inspect=args.inspect,
            rerun=args.rerun,
            ws_port=args.ws_port,
            no_ws=args.no_ws,
            json_output=getattr(args, 'json_output', False),
        )
        handle_knowledge(knowledge_args,store_dir=args.store_dir)
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