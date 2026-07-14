import sys
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
from src.core.Logger.DarwinLogger import darwin


def _validate_files(files: List[str], store: DarwinStore, mode: str = "semantic") -> List[str]:
    valid = []
    for file_path in files:
        filename = Path(file_path).name
        if store.is_search_ready(filename, mode=mode):
            valid.append(filename)
        else:
            darwin.warning(f"{file_path} is not searchable in {mode} mode")
    return valid


def handle_semantic_search(args: SemanticSearchArgs, store: DarwinStore) -> List[SemanticSearchResult]:
    from src.core.Models.ModelConfig import get_active_model
    model_name = get_active_model("embedding")

    if model_name is None:
        darwin.error("No active embedding model found. Run model setup first.")
        return []

    darwin.info("")
    darwin.info("Semantic Search")
    darwin.info(f"Query: '{args.query}'")
    darwin.info(f"Model: {model_name}")
    darwin.info(f"Per-doc limit: {args.per_doc_top_k if args.per_doc_top_k is not None else 'unlimited'}")

    status = store.status()
    tobe_searched = []

    if args.files:
        tobe_searched = _validate_files(args.files, store)
        if not tobe_searched:
            darwin.error("No valid files to search. Search aborted.")
            return []
        darwin.info(f"Searching {len(tobe_searched)} valid files for '{args.query}'")
    else:
        tobe_searched = [
            fname for fname, info in status.get("files", {}).items()
            if info.get("is_embedding_searchable", False)
        ]
        darwin.info(f"Searching {len(tobe_searched)} embedding-searchable files for '{args.query}'")

    darwin.info("")

    if not tobe_searched:
        darwin.warning("No searchable files found. Run the embedding pipeline first.")
        return []

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
        darwin.warning(f"No results found for '{args.query}'")
        darwin.info("")
        darwin.info("Try:")
        darwin.info("  - Using different or more specific keywords")
        darwin.info("  - Lowering the minimum score threshold (--min-score)")
        darwin.info("  - Searching a specific file with --file")
        return []

    darwin.success(f"Found {len(results)} results")
    darwin.info("")
    darwin.info("  #    Score       Source                Page  Preview")
    darwin.info("  ---  ----------  --------------------  ----  -------")

    for i, r in enumerate(results[:10], 1):
        preview = r.text[:60].replace("\n", " ")
        darwin.info(f"  {i:<4} {r.score:.3f}      {r.source[:20]:<20}  {r.page:<4}  {preview}...")

    darwin.info("")
    darwin.info(f"  Min score threshold: {args.min_score}")
    darwin.info(f"  Top score: {results[0].score:.3f}")
    darwin.info(f"  Sources: {len(set(r.source for r in results))} files")
    darwin.info("")

    return results


def main() -> int:
    import argparse

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
    parser.add_argument("--sse", action="store_true", help="Enable SSE logging")

    args = parser.parse_args()

    query = args.query or args.query_flag
    if not query:
        parser.error("Query required (either positional or --query)")

    from src.core.Logger.DarwinLogger import OutputMode

    darwin.configure(
        show_banner=True,
        mode=OutputMode.BOTH if args.sse else OutputMode.TERMINAL,
    )

    try:
        search_args = SemanticSearchArgs(
            query=query,
            files=args.file or [],
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
        darwin.info("")
        darwin.info("Goodbye!")
        return 130
    except Exception as e:
        import traceback
        darwin.error(f"Fatal error: {e}")
        darwin.error("".join(traceback.format_exception(e)))
        return 1
    finally:
        darwin.shutdown()


if __name__ == "__main__":
    main()