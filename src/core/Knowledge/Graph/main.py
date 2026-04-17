"""
src/core/Graph/main.py - Graph extraction CLI

Extract semantic concept knowledge graphs from chunked documents.

Usage examples
--------------
# Extract graph (yake + semantic similarity, recommended)
uv run src/core/Graph/main.py --file biology.pdf

# Use regex fallback (no yake needed)
uv run src/core/Graph/main.py --file biology.pdf --mode regex

# Use spaCy (slow, requires: uv run python -m spacy download en_core_web_sm)
uv run src/core/Graph/main.py --file biology.pdf --mode spacy

# Adjust similarity threshold (lower = more edges, higher = fewer, tighter clusters)
uv run src/core/Graph/main.py --file biology.pdf --threshold 0.55

# Generate interactive HTML visualization
uv run src/core/Graph/main.py --file biology.pdf --visualize

# Inspect graph structure
uv run src/core/Graph/main.py --file biology.pdf --inspect

# Export as JSON
uv run src/core/Graph/main.py --file biology.pdf --json

# Force re-extraction
uv run src/core/Graph/main.py --file biology.pdf --rerun
"""

import sys
import argparse
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Knowledge.Graph.GraphExtractor import run_graph_extraction_pipeline, GraphData
from src.core.Knowledge.Graph.GraphUtils import (
    GraphExtractionArgs,
    build_concept_index,
    export_graph_json,
    DEFAULT_MIN_CONCEPT_FREQ,
    DEFAULT_MAX_CONCEPTS,
    DEFAULT_MIN_RELATION_FREQ,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ============================================================================
# Inspect
# ============================================================================

def _print_inspection(graph: GraphData, filename: str) -> None:
    TerminalUI.blank()
    TerminalUI.rule(f"graph inspection — {filename}")
    TerminalUI.blank()

    TerminalUI.table(
        headers=["metric", "value"],
        rows=[
            ["filename",                  graph.filename],
            ["chunks processed",          str(graph.chunk_count)],
            ["raw concepts extracted",    str(graph.total_concepts_extracted)],
            ["final concepts (nodes)",    str(len(graph.nodes))],
            ["relationships (edges)",     str(len(graph.edges))],
            ["avg edges per node",        f"{len(graph.edges) / max(len(graph.nodes), 1):.1f}"],
        ],
    )

    TerminalUI.blank()
    TerminalUI.dim("── top concepts ──")
    TerminalUI.blank()

    for i, node in enumerate(graph.nodes[:10], 1):
        TerminalUI.info(f"{i}. {node.label}", f"frequency: {node.frequency}")

    TerminalUI.blank()
    TerminalUI.dim("── strongest relationships ──")
    TerminalUI.blank()

    node_labels = {n.id: n.label for n in graph.nodes}
    top_edges   = sorted(graph.edges, key=lambda e: e.weight, reverse=True)[:10]

    for i, edge in enumerate(top_edges, 1):
        src = node_labels.get(edge.source, edge.source)
        tgt = node_labels.get(edge.target, edge.target)
        TerminalUI.info(f"{src} ↔ {tgt}", f"weight: {edge.weight}")

    TerminalUI.rule()


# ============================================================================
# Core handler
# ============================================================================

def handle_graph_extraction(args: GraphExtractionArgs) -> None:
    store = DarwinStore(Path(args.project) if args.project else Path.cwd())

    # Resolve files
    if args.file:
        filenames = [args.file]
    else:
        status    = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info["components"]["chunks"]
        ]

    if not filenames:
        TerminalUI.error("No chunked files found. Run chunking first.")
        return

    TerminalUI.bold(f"Extracting graphs from {len(filenames)} file(s)")
    TerminalUI.dim(
        f"Mode: {args.extraction_mode} | "
        f"Min freq: {args.min_freq} | "
        f"Max concepts: {args.max_concepts} | "
        f"Threshold: {args.similarity_threshold if args.similarity_threshold else 'auto'} | "
        f"AI elevation: {'✨ on' if args.use_ai else 'off'}"
    )
    TerminalUI.blank()

    succeeded = failed = skipped = 0

    for filename in filenames:
        if not args.rerun and store.is_fresh(filename) and store.has_graph(filename):
            TerminalUI.dim(f"skipping {filename} (fresh graph exists)")
            skipped += 1
            continue

        chunks = store.load_chunks(filename)
        if not chunks:
            TerminalUI.error(f"{filename} — no chunks found")
            failed += 1
            continue

        try:
            graph = run_graph_extraction_pipeline(
                filename=filename,
                chunks=chunks,
                store=store,
                min_freq=args.min_freq,
                max_concepts=args.max_concepts,
                min_rel_freq=args.min_rel_freq,
                similarity_threshold=args.similarity_threshold,
                extraction_mode=args.extraction_mode,
                use_ai=args.use_ai,
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
            traceback.print_exc()
            TerminalUI.error(f"{filename} — {type(e).__name__}: {e}")
            failed += 1

    TerminalUI.blank()
    TerminalUI.success(f"done — {succeeded} extracted · {skipped} skipped · {failed} failed")


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract semantic concept knowledge graphs from documents"
    )
    parser.add_argument("path",            nargs="?", default=".", help="Project directory")
    parser.add_argument("--file",   "-f",  default=None,           help="Single file to process")
    parser.add_argument("--project", "-p", default=None,           help="Project directory")
    parser.add_argument("--mode",   "-m",
                        default="yake",
                        choices=["yake", "regex", "spacy"],
                        help="Concept extraction mode (default: yake)")
    parser.add_argument("--threshold", "-t",
                        type=float,
                        default=DEFAULT_SIMILARITY_THRESHOLD,
                        help="Cosine similarity threshold for edges (default: 0.45)")
    parser.add_argument("--min-freq",
                        type=int,
                        default=DEFAULT_MIN_CONCEPT_FREQ,
                        help="Minimum concept frequency")
    parser.add_argument("--max-concepts",
                        type=int,
                        default=DEFAULT_MAX_CONCEPTS,
                        help="Maximum concepts to extract")
    parser.add_argument("--min-rel-freq",
                        type=int,
                        default=DEFAULT_MIN_RELATION_FREQ,
                        help="Minimum co-occurrence (fallback mode only)")
    parser.add_argument("--ai", "-a", action="store_true", dest="use_ai", help="Elevate keywords to real concepts via LLM ✨")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate HTML visualization")
    parser.add_argument("--inspect",   "-i", action="store_true", help="Show detailed graph structure")
    parser.add_argument("--json",      "-j", action="store_true", help="Export as JSON")
    parser.add_argument("--rerun",     "-R", action="store_true", help="Force re-extraction")

    args = parser.parse_args()

    handle_graph_extraction(GraphExtractionArgs(
        target=args.path,
        file=args.file,
        project=args.project or args.path,
        min_freq=args.min_freq,
        max_concepts=args.max_concepts,
        min_rel_freq=args.min_rel_freq,
        similarity_threshold=args.threshold,
        extraction_mode=args.mode,
        use_ai=args.use_ai,
        visualize=args.visualize,
        inspect=args.inspect,
        json_output=args.json,
        rerun=args.rerun,
    ))

    return 0


if __name__ == "__main__":
    sys.exit(main())