"""
src/features/Retrival/Semantic/main.py - Semantic search CLI
"""

import sys
import argparse
from pathlib import Path
from typing import List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Retrival.Semantic.SemanticSearch import search_across_documents
from src.core.Retrival.lib import (
    SemanticSearchArgs,
    DEFAULT_TOP_K,
    DEFAULT_MIN_SCORE,
    SemanticSearchResult,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType


def _validate_files(files: List[str], store: DarwinStore, mode: str = "semantic") -> List[str]:
    valid = []
    for file_path in files:
        filename = Path(file_path).name
        if store.is_search_ready(filename, mode=mode):
            valid.append(filename)
        else:
            darwin.log(f"{file_path} is not searchable in {mode} mode", LogType.WARNING)
    return valid


def handle_semantic_search(args: SemanticSearchArgs, store: DarwinStore) -> List[SemanticSearchResult] | None:
    from src.core.Models.ModelConfig import get_active_model
    model_name = get_active_model("embedding")

    if model_name is None:
        darwin.log("No active embedding model found. Run model setup first.", LogType.ERROR)
        return None

    darwin.log("")
    darwin.log("Semantic Search")
    darwin.log(f"Query: '{args.query}'")
    darwin.log(f"Model: {model_name}")
    darwin.log(f"Per‑doc limit: {args.per_doc_top_k if args.per_doc_top_k is not None else 'unlimited'}")

    status = store.status()
    tobe_searched = []

    if args.files:
        tobe_searched = _validate_files(args.files, store)
        if not tobe_searched:
            darwin.log("No valid files to search. Aborted.", LogType.ERROR)
            return None
        darwin.log(f"Searching {len(tobe_searched)} valid files for '{args.query}'")
    else:
        tobe_searched = [
            fname for fname, info in status.get("files", {}).items()
            if info.get("is_embedding_searchable", False)
        ]
        darwin.log(f"Searching {len(tobe_searched)} embedding-searchable files for '{args.query}'")

    darwin.log("")

    if not tobe_searched:
        darwin.log("No searchable files found. Run the embedding pipeline first on your files.", LogType.WARNING)
        return None

    results = search_across_documents(
        query=args.query,
        model_name=model_name,
        store=store,
        filenames=tobe_searched,
        top_k=args.top_k,
        min_score=args.min_score,
        use_usearch=True,
        per_doc_top_k=args.per_doc_top_k,   # <-- pass new parameter
    )

    if not results:
        darwin.log(f"No results found for '{args.query}'", LogType.WARNING)
        darwin.log("")
        darwin.log("Try:")
        darwin.log("  - Using different or more specific keywords")
        darwin.log("  - Lowering the minimum score threshold (--min-score)")
        darwin.log("  - Searching a specific file with --file")
        return None

    darwin.log(f"Found {len(results)} results", LogType.SUCCESS)
    darwin.log("")
    darwin.log("  #    Score       Source                Page  Preview")
    darwin.log("  ---  ----------  --------------------  ----  -------")

    for i, r in enumerate(results[:10], 1):
        preview = r.text[:60].replace("\n", " ")
        darwin.log(f"  {i:<4} {r.score:.3f}      {r.source[:20]:<20}  {r.page:<4}  {preview}...")

    darwin.log("")
    darwin.log(f"  Min score threshold: {args.min_score}")
    darwin.log(f"  Top score: {results[0].score:.3f}")
    darwin.log(f"  Sources: {len(set(r.source for r in results))} files")
    darwin.log("")

    return results


def main() -> int:
    from src.core.Storage.DarwinStore import DarwinStore
    from src.lib.Utils import ensure_writable_directory

    parser = argparse.ArgumentParser(
        description="Semantic search over embedded documents",
        usage="darwin search [QUERY] [options]"
    )
    parser.add_argument("query", nargs="?", default=None, help="Search query string")
    parser.add_argument("--query", "-q", dest="query_flag", help="Search query (alternative)")
    parser.add_argument("--file", "-f", nargs="+", default=None, help="One or more files to search")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K, help="Number of results")
    parser.add_argument("--min-score", "-s", type=float, default=DEFAULT_MIN_SCORE, help="Minimum score (0-1)")
    parser.add_argument("--per-doc-top-k", type=int, default=None,
                        help="Limit results per document (None=unlimited). Use to prevent large documents from dominating.")
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")
    parser.add_argument("--sse-port", type=int, default=8765, help="WebSocket server port")

    args = parser.parse_args()

    query = args.query or args.query_flag
    if not query:
        parser.error("Query required (either positional or --query)")

    darwin.configure(sse_port=args.sse_port, show_banner=True)

    try:
        search_args = SemanticSearchArgs(
            query=query,
            files=args.file,
            top_k=args.top_k,
            min_score=args.min_score,
            per_doc_top_k=args.per_doc_top_k,   # <-- new
        )

        with darwin.session("semantic_search"):
            if args.store_dir and ensure_writable_directory(args.store_dir):
                store = DarwinStore(args.store_dir)
            else:
                store = DarwinStore()
            handle_semantic_search(search_args, store)
        return 0

    except KeyboardInterrupt:
        darwin.log("")
        darwin.log("Goodbye!")
        return 130
    except Exception as e:
        import traceback
        darwin.log(f"Fatal error: {e}", LogType.ERROR)
        darwin.log("".join(traceback.format_exception(e)), LogType.ERROR)
        return 1
    finally:
        darwin.shutdown()


if __name__ == "__main__":
    main()