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
from src.lib.Terminal import TerminalUI
from src.core.Models import setup

def _print_inspection(graph, filename: str) -> None:
    TerminalUI.blank()
    TerminalUI.rule(f"graph — {filename}")
    TerminalUI.blank()

    TerminalUI.table(
        headers=["metric", "value"],
        rows=[
            ["concepts (nodes)", str(len(graph.nodes))],
            ["edges", str(len(graph.edges))],
            ["avg edges/node", f"{len(graph.edges)/max(len(graph.nodes),1):.1f}"],
            ["chunks processed", str(graph.chunk_count)],
            ["raw keywords extracted", str(graph.total_concepts_extracted)],
        ],
    )

    TerminalUI.blank()
    TerminalUI.dim("── top concepts ──")
    for i, node in enumerate(graph.nodes[:10], 1):
        TerminalUI.info(f"  {i}. {node.label}", f"in {node.frequency} chunks")

    if graph.edges:
        TerminalUI.blank()
        TerminalUI.dim("── strongest relationships ──")
        node_labels = {n.id: n.label for n in graph.nodes}
        for edge in sorted(graph.edges, key=lambda e: e.weight, reverse=True)[:10]:
            src = node_labels.get(edge.source, edge.source)
            tgt = node_labels.get(edge.target, edge.target)
            TerminalUI.info(f"  {src} ↔ {tgt}", f"{edge.weight:.3f}")

    TerminalUI.rule()


def handle_graph_extraction(args: GraphExtractionArgs) -> None:
    store = DarwinStore()

    if args.file:
        filenames = [args.file]
    else:
        status = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info["components"]["chunks"]
        ]

    if not filenames:
        TerminalUI.error("No chunked files found. Run chunking first.")
        return

    TerminalUI.bold(f"Extracting graphs from {len(filenames)} file(s)")
    TerminalUI.dim(
        f"Extraction: {args.extraction_mode} | "
        f"Elevation: {args.elevation_mode} | "
        f"Max concepts: {args.max_concepts} | "
        f"Threshold: {args.similarity_threshold if args.similarity_threshold else 'auto'}"
    )
    setup()

    succeeded = failed = skipped = 0

    for filename in filenames:
        if not args.rerun and store.is_fresh(filename) and store.has_graph(filename):
            TerminalUI.dim(f"  ⏭️  {filename} (fresh graph exists)")
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
            traceback.print_exc()
            TerminalUI.error(f"  ❌ {filename} — {type(e).__name__}: {e}")
            failed += 1

    TerminalUI.success(f"Done — {succeeded} extracted · {skipped} skipped · {failed} failed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract concept knowledge graphs from documents")
    parser.add_argument("path", nargs="?", default=".", help="Project directory")
    parser.add_argument("--file", "-f", default=None, help="Single file to process")
    parser.add_argument("--project", "-p", default=None, help="Project directory")
    parser.add_argument("--mode", "-m", default="yake", choices=["yake", "regex", "spacy"],
                        help="Keyword extraction mode (default: yake)")
    parser.add_argument("--elevation", "-e", default="spacy", choices=["spacy", "llm", "none"],
                        help="Concept elevation mode (default: spacy, offline)")
    parser.add_argument("--threshold", "-t", type=float, default=None,
                        help="Similarity threshold (default: auto)")
    parser.add_argument("--min-freq", type=int, default=DEFAULT_MIN_CONCEPT_FREQ)
    parser.add_argument("--max-concepts", "-n", type=int, default=DEFAULT_MAX_CONCEPTS,
                        help="Max concept nodes (default: 50)")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate HTML")
    parser.add_argument("--inspect", "-i", action="store_true", help="Show graph details")
    parser.add_argument("--json", "-j", action="store_true", help="Export JSON")
    parser.add_argument("--rerun", "-R", action="store_true", help="Force re-extraction")

    args = parser.parse_args()

    handle_graph_extraction(GraphExtractionArgs(
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
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())