# src/core/Knowledge/Graph/main.py - Fixed with WebSocket support

"""
src/core/Knowledge/Graph/main.py
---------------------------------
Graph extraction CLI entry point.

Usage:
  uv run src/core/Knowledge/Graph/main.py --file biology.pdf
  uv run src/core/Knowledge/Graph/main.py --file biology.pdf --visualize --inspect
  uv run src/core/Knowledge/Graph/main.py --file biology.pdf --mode spacy --elevation llm
  uv run src/core/Knowledge/Graph/main.py --file biology.pdf --max-concepts 30
"""

import sys
import argparse
import asyncio
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Knowledge.Graph.GraphExtractor import run_graph_extraction_pipeline
from src.core.Knowledge.Graph.GraphUtils import (
    GraphExtractionArgs,
    DEFAULT_MIN_CONCEPT_FREQ,
    DEFAULT_MAX_CONCEPTS,
    export_graph_json,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket
from src.core.Models import setup


def _print_inspection(graph, filename: str) -> None:
    """Print detailed graph inspection."""
    darwin.log(f"\n{'='*60}", LogType.INFO)
    darwin.log(f"Graph — {filename}", LogType.INFO)
    darwin.log(f"{'='*60}", LogType.INFO)

    darwin.log(f"Concepts (nodes): {len(graph.nodes)}", LogType.INFO)
    darwin.log(f"Edges: {len(graph.edges)}", LogType.INFO)
    darwin.log(f"Avg edges/node: {len(graph.edges)/max(len(graph.nodes),1):.1f}", LogType.INFO)
    darwin.log(f"Chunks processed: {graph.chunk_count}", LogType.INFO)
    darwin.log(f"Raw keywords extracted: {graph.total_concepts_extracted}", LogType.INFO)

    darwin.log(f"\n--- Top Concepts ---", LogType.INFO)
    for i, node in enumerate(graph.nodes[:10], 1):
        darwin.log(f"  {i}. {node.label} — in {node.frequency} chunks", LogType.INFO)

    if graph.edges:
        darwin.log(f"\n--- Strongest Relationships ---", LogType.INFO)
        node_labels = {n.id: n.label for n in graph.nodes}
        for edge in sorted(graph.edges, key=lambda e: e.weight, reverse=True)[:10]:
            src = node_labels.get(edge.source, edge.source)
            tgt = node_labels.get(edge.target, edge.target)
            darwin.log(f"  {src} ↔ {tgt} — {edge.weight:.3f}", LogType.INFO)

    darwin.log(f"{'='*60}", LogType.INFO)


def handle_graph_extraction(args: GraphExtractionArgs) -> None:
    """Main handler for graph extraction."""
    store = DarwinStore()

    # Pre-load tokenizer once before processing files
    if args.extraction_mode == "spacy" or args.elevation_mode == "spacy":
        try:
            from src.core.Models import get_loader, ModelCategory
            loader = get_loader()
            darwin.log("Pre-loading tokenizer model...", LogType.INFO)
            loader.get(ModelCategory.TOKENIZER)
            darwin.log("Tokenizer model loaded", LogType.SUCCESS)
        except Exception as e:
            darwin.log(f"Failed to load tokenizer: {e}. Falling back to regex.", LogType.WARNING)
            args.extraction_mode = "regex"
            if args.elevation_mode == "spacy":
                args.elevation_mode = "none"

    if args.file:
        filenames = [args.file]
    else:
        status = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info.get("is_chunked", False)
        ]

    if not filenames:
        darwin.log("No chunked files found. Run chunking first.", LogType.ERROR)
        return

    darwin.log(f"Extracting graphs from {len(filenames)} file(s)", LogType.INFO)
    darwin.log(
        f"Extraction: {args.extraction_mode} | "
        f"Elevation: {args.elevation_mode} | "
        f"Max concepts: {args.max_concepts} | "
        f"Threshold: {args.similarity_threshold if args.similarity_threshold else 'auto'}"
    )

    succeeded = failed = skipped = 0

    for filename in filenames:
        if not args.rerun and store.has_graph(filename):
            darwin.log(f"Skipping {filename} (fresh graph exists)", LogType.INFO)
            skipped += 1
            continue

        try:
            graph = run_graph_extraction_pipeline(
                filename=filename,
                store=store,
                min_freq=args.min_freq,
                max_concepts=args.max_concepts,
                similarity_threshold=args.similarity_threshold,
                extraction_mode=args.extraction_mode,
                elevation_mode=args.elevation_mode,
                visualize=args.visualize,
                rerun=args.rerun,
            )

            if graph:
                succeeded += 1
                if args.inspect:
                    _print_inspection(graph, filename)
                if args.json_output:
                    print(export_graph_json(graph))

        except Exception as e:
            import traceback
            darwin.log(f"{filename} — {type(e).__name__}: {e}", LogType.ERROR)
            darwin.log(traceback.format_exc())
            failed += 1

    darwin.log(f"Done — {succeeded} extracted · {skipped} skipped · {failed} failed", LogType.SUCCESS)

    
async def main() -> int:
    """Async entry point with WebSocket support."""
    parser = argparse.ArgumentParser(description="Extract concept knowledge graphs from documents")
    parser.add_argument("path", nargs="?", default=".", help="Project directory")
    parser.add_argument("--file", "-f", default=None, help="Single file to process")
    parser.add_argument("--project", "-p", default=None, help="Project directory")
    parser.add_argument("--mode", "-m", default="spacy", choices=[ "regex", "spacy"],
                        help="Keyword extraction mode (default: spacy)")
    parser.add_argument("--elevation", "-e", default="none", choices=["spacy", "llm", "none"],
                        help="Concept elevation mode (default: spacy, offline)")
    parser.add_argument("--threshold", "-t", type=float, default=None,
                        help="Similarity threshold (default: auto)")
    parser.add_argument("--min-freq", type=int, default=DEFAULT_MIN_CONCEPT_FREQ)
    parser.add_argument("--max-concepts", "-n", type=int, default=DEFAULT_MAX_CONCEPTS,
                        help="Max concept nodes (default: 75)")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate HTML")
    parser.add_argument("--inspect", "-i", action="store_true", help="Show graph details")
    parser.add_argument("--json", "-j", action="store_true", help="Export JSON")
    parser.add_argument("--rerun", "-R", action="store_true", help="Force re-extraction")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")

    args = parser.parse_args()

    # Setup WebSocket
    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        graph_args = GraphExtractionArgs(
            target=args.path,
            file=args.file,
            project=args.project or args.path,
            min_freq=args.min_freq,
            max_concepts=args.max_concepts,
            similarity_threshold=args.threshold,
            extraction_mode=args.mode,
            elevation_mode=args.elevation,
            visualize=args.visualize,
            inspect=args.inspect,
            json_output=args.json,
            rerun=args.rerun,
        )
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, handle_graph_extraction, graph_args)
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