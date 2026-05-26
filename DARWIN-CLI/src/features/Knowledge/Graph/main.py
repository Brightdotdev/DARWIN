import sys
import os
import warnings
import argparse
import asyncio
from pathlib import Path
from typing import List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# CHANGED: suppress LiteLLM Bedrock warning spam
warnings.filterwarnings("ignore", module="litellm")
os.environ["LITELLM_SUPPRESS_DEBUG_INFO"] = "True"

from src.features.Knowledge.Graph.GraphExtractor import run_graph_extraction_pipeline
from src.features.Knowledge.Graph.GraphUtils import (
    DEFAULT_MIN_CONCEPT_FREQ,
    DEFAULT_MAX_CONCEPTS,
    export_graph_json,
)
from src.features.Knowledge.Graph.GraphConfig import GraphExtractionConfig, PRESETS
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity


def _print_inspection(graph, filename: str) -> None:
    """Print detailed graph inspection."""
    darwin.log(f"\n{'='*60}", LogType.INFO)
    darwin.log(f"Graph — {filename}", LogType.INFO)
    darwin.log(f"{'='*60}", LogType.INFO)

    darwin.log(f"Entities (nodes): {len(graph.nodes)}", LogType.INFO)
    darwin.log(f"Edges: {len(graph.edges)}", LogType.INFO)

    semantic = len([e for e in graph.edges if e.relation_type != "co_occurrence"])
    cooc = len([e for e in graph.edges if e.relation_type == "co_occurrence"])
    darwin.log(f"  Semantic relations: {semantic}", LogType.INFO)
    darwin.log(f"  Co-occurrence edges: {cooc}", LogType.INFO)

    darwin.log(f"Avg edges/node: {len(graph.edges)/max(len(graph.nodes),1):.1f}", LogType.INFO)
    darwin.log(f"Chunks processed: {graph.chunk_count}", LogType.INFO)
    darwin.log(f"Raw entities extracted: {graph.total_entities_extracted}", LogType.INFO)
    darwin.log(f"Raw relations extracted: {graph.total_relations_extracted}", LogType.INFO)
    darwin.log(f"Processing mode: {graph.processing_mode}", LogType.INFO)

    darwin.log(f"\n--- Top Entities ---", LogType.INFO)
    for i, node in enumerate(graph.nodes[:10], 1):
        darwin.log(
            f"  {i}. {node.display_name} ({node.entity_type}) — "
            f"in {node.frequency} chunks, first @ chunk {node.first_mention_chunk}",
            LogType.INFO,
        )

    if graph.edges:
        darwin.log(f"\n--- Strongest Semantic Relations ---", LogType.INFO)
        semantic_edges = [e for e in graph.edges if e.relation_type != "co_occurrence"]
        for edge in sorted(semantic_edges, key=lambda e: e.weight, reverse=True)[:10]:
            src = next((n.display_name for n in graph.nodes if n.id == edge.source), edge.source)
            tgt = next((n.display_name for n in graph.nodes if n.id == edge.target), edge.target)
            darwin.log(
                f"  {src} —[{edge.relation_type}]→ {tgt} (conf: {edge.weight:.2f})",
                LogType.INFO,
            )

    darwin.log(f"{'='*60}", LogType.INFO)


def handle_graph_extraction(config: GraphExtractionConfig,store,files: List[str] = None,) -> List[str] | None:

    if files:
        filenames = [f for f in files if store.has_records(f)]
        missing = [f for f in files if not store.has_records(f)]
        for f in missing:
            darwin.log(f"'{f}' has no chunks — skipping", LogType.WARNING)
    else:
        status = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info.get("is_chunked", False)
        ]

    if not filenames:
        darwin.log("No chunked files found. Run chunking first.", LogType.ERROR)
        return None

    darwin.log(f"Extracting graphs from {len(filenames)} file(s)", LogType.INFO)
    darwin.log(f"Mode: {config.processing_mode} | Max concepts: {config.max_concepts}", LogType.INFO)
    darwin.log(f"Relation threshold: {config.relation_confidence_threshold} | Dedup: {config.dedup_strategy}", LogType.INFO)

    succeeded = failed = skipped = 0
    completed_files = []

    for filename in filenames:
        if not config.rerun and store.has_graph(filename):
            darwin.log(f"Skipping {filename} (fresh graph exists)", LogType.INFO)
            skipped += 1
            completed_files.append(filename)
            continue

        try:
            graph = run_graph_extraction_pipeline(
                filename=filename,
                store=store,
                config=config,
            )

            if graph:
                succeeded += 1
                completed_files.append(filename)
                if config.inspect:
                    _print_inspection(graph, filename)
                if config.json_output:
                    print(export_graph_json(graph))

        except Exception as e:
            import traceback
            darwin.log(f"{filename} — {type(e).__name__}: {e}", LogType.ERROR)
            darwin.log(traceback.format_exc(), LogType.ERROR)
            failed += 1

    darwin.log(f"Done — {succeeded} extracted · {skipped} skipped · {failed} failed", LogType.SUCCESS)
    return completed_files if completed_files else None


async def main() -> int:
    """Async entry point with WebSocket support."""
    from src.lib.Utils import ensure_writable_directory

    parser = argparse.ArgumentParser(description="Extract knowledge graphs from documents")

    # File selection
    parser.add_argument("--file", "-f", nargs="+", default=None, help="One or more files to process")

    # Document type preset (overrides individual settings)
    parser.add_argument(
        "--preset", "-p",
        default=None,
        choices=list(PRESETS.keys()),
        help="Document-type preset: academic|research|legal|narrative|technical. Overrides other settings.",
    )

    # Processing mode
    parser.add_argument(
        "--processing-mode", "-m",
        default="auto",
        choices=["auto", "full", "chunked", "sliding_window"],
        help="Document processing mode (default: auto)",
    )
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=float, default=0.2, help="Chunk overlap fraction (0.0-1.0)")

    # Graph size control
    parser.add_argument("--max-concepts", "-n", type=int, default=DEFAULT_MAX_CONCEPTS, help="Max concept nodes")
    parser.add_argument("--min-freq", type=int, default=DEFAULT_MIN_CONCEPT_FREQ, help="Min chunk frequency")

    # Relationship extraction
    parser.add_argument("--relation-threshold", "-t", type=float, default=0.6, help="Relation confidence threshold")
    parser.add_argument("--dedup-strategy", default="svo_canonical", choices=["svo_canonical", "strict", "none"])
    parser.add_argument("--similarity-threshold", type=float, default=None, help="Co-occurrence edge threshold")

    # spaCy requirements
    parser.add_argument("--require-ner", action="store_true", default=True, help="Require NER (default: True)")
    parser.add_argument("--no-require-ner", action="store_true", dest="require_ner", default=False, help="Allow fallback without NER")
    parser.add_argument("--require-parser", action="store_true", default=True, help="Require dependency parser")
    parser.add_argument("--no-require-parser", action="store_true", dest="require_parser", default=False, help="Allow fallback without parser")

    # Performance
    parser.add_argument("--spacy-batch", type=int, default=128, help="spaCy batch size")
    parser.add_argument("--embed-batch", type=int, default=64, help="Embedding batch size")

    # Output
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate HTML")
    parser.add_argument("--inspect", "-i", action="store_true", help="Show graph details")
    parser.add_argument("--json", "-j", action="store_true", help="Export JSON")
    parser.add_argument("--rerun", "-R", action="store_true", help="Force re-extraction")

    # WebSocket
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")
    parser.add_argument("--verbosity", choices=["normal", "quiet"], default="normal", help="Log verbosity")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")

    args = parser.parse_args()



    # Handle --no-require-ner / --no-require-parser
    require_ner = not args.no_require_ner if hasattr(args, "no_require_ner") else args.require_ner
    require_parser = not args.no_require_parser if hasattr(args, "no_require_parser") else args.require_parser

    # Build config from preset or individual args
    if args.preset:
        darwin.log(f"Using '{args.preset}' preset", LogType.INFO)
        config = GraphExtractionConfig.from_preset(args.preset)
        # Allow individual overrides even with preset
        if args.processing_mode != "auto":
            config.processing_mode = args.processing_mode
        if args.max_concepts != DEFAULT_MAX_CONCEPTS:
            config.max_concepts = args.max_concepts
        if args.relation_threshold != 0.6:
            config.relation_confidence_threshold = args.relation_threshold
        if args.visualize:
            config.visualize = True
        if args.inspect:
            config.inspect = True
        if args.json:
            config.json_output = True
        if args.rerun:
            config.rerun = True
    else:
        config = GraphExtractionConfig(
            processing_mode=args.processing_mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_concepts=args.max_concepts,
            min_freq=args.min_freq,
            relation_confidence_threshold=args.relation_threshold,
            dedup_strategy=args.dedup_strategy,
            similarity_threshold=args.similarity_threshold,
            require_typed_entities=require_ner,
            require_parser=require_parser,
            batch_size_spacy=args.spacy_batch,
            batch_size_embed=args.embed_batch,
            visualize=args.visualize,
            inspect=args.inspect,
            json_output=args.json,
            rerun=args.rerun,
        )

    # Setup WebSocket
    darwin.configure(
        enable_websocket=not args.no_ws,
        ws_port=args.ws_port,
        enable_history=True,
        verbosity_default=Verbosity.QUIET if args.verbosity == "quiet" else Verbosity.NORMAL,
        show_banner=True,
    )

    try:
        with darwin.session("graph_extraction"):
            if args.store_dir and ensure_writable_directory(args.store_dir):
                store = DarwinStore(args.store_dir)
            else:
                store = DarwinStore()
            handle_graph_extraction(config, store, files=args.file)
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