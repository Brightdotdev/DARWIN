# main.py (search)
import sys
import argparse
import time
from pathlib import Path
from typing import List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Retrival.Keyword.KeywordSearch import search_across_documents_keyword
from src.core.Retrival.lib import (
    KeywordSearchResult,
    KeywordSearchArgs,
    DEFAULT_TOP_K,
    DEFAULT_K1,
    DEFAULT_B,
)


from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin


def _validate_files(
    files: List[str], store: DarwinStore, mode: str = "kw"
) -> List[str]:
    valid = []
    for file_path in files:
        filename = Path(file_path).name
        if store.is_search_ready(filename, mode=mode):
            valid.append(filename)
        else:
            darwin.warning(f"{file_path} is not searchable in {mode} mode")
    return valid


def _search_wrapper(
    store: DarwinStore, args: KeywordSearchArgs
) -> List[KeywordSearchResult]:
    status = store.status()
    tobe_searched = []

    if args.files:
        darwin.info(f"Searching across {', '.join(args.files)}")
        tobe_searched = _validate_files(args.files, store)
        if not tobe_searched:
            darwin.error("No valid files provided. Search aborted.")
            return []
        darwin.info(f"Searching {len(tobe_searched)} valid files for '{args.query}'")
    else:
        tobe_searched = [
            fname
            for fname, info in status.get("files", {}).items()
            if info.get("is_kw_searchable", False)
        ]
        darwin.info(f"Searching {len(tobe_searched)} files for '{args.query}'")

    if not tobe_searched:
        darwin.warning("No searchable files found. Run indexing first.")
        return []

    start_time = time.perf_counter()
    results = search_across_documents_keyword(
        query=args.query,
        filenames=tobe_searched,
        store=store,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b,
        per_doc_top_k=args.per_doc_top_k,
    )
    elapsed = time.perf_counter() - start_time

    if results:
        darwin.success(f"Found {len(results)} results in {elapsed:.2f}s")
    else:
        darwin.warning(f"No results found for '{args.query}'")

    return results


def _display_search_results(
    results: List[KeywordSearchResult], args: KeywordSearchArgs
) -> None:
    if not results:
        darwin.info("")
        darwin.warning(f"No results found for '{args.query}'")
        darwin.info("")
        darwin.info("Try:")
        darwin.info("  - Using fewer or different keywords")
        darwin.info("  - Checking spelling")
        darwin.info("  - Searching a specific file with --file")
        return


def handle_keyword_search(
    args: KeywordSearchArgs, store: DarwinStore
) -> List[KeywordSearchResult] | None:
    darwin.info("")
    darwin.info("Keyword Search")
    darwin.info(f"Query: {args.query}")
    darwin.info(f"BM25: k1={args.k1}, b={args.b}")
    darwin.info(f"Top-K: {args.top_k}")
    darwin.info(f"Scope: {', '.join(args.files) if args.files else 'all documents'}")
    darwin.info("")

    results = _search_wrapper(store, args)
    _display_search_results(results, args)

    if not results:
        return None
    return results


def main() -> int:
    from src.core.Storage.DarwinStore import DarwinStore

    parser = argparse.ArgumentParser(
        description="BM25 keyword search over indexed documents"
    )
    parser.add_argument("query", nargs="?", default=None, help="Search query string")
    parser.add_argument(
        "--query", "-q", dest="query_flag", help="Search query (alternative)"
    )
    parser.add_argument(
        "--file", "-f", nargs="+", default=None, help="One or more files to search"
    )
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--json", "-j", action="store_true")
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")
    parser.add_argument("--sse-port", type=int, default=8765, help="server port")
    parser.add_argument(
        "--top-k-per-doc", type=int, default=None, help="Max retrival per document"
    )

    args = parser.parse_args()

    store = DarwinStore(args.store_dir or None)
    query = args.query or args.query_flag

    if not query:
        parser.error("Query required (either as positional or --query)")

    darwin.configure(sse_port=args.sse_port, show_banner=True)

    try:
        search_args = KeywordSearchArgs(
            query=query,
            files=args.file,
            top_k=args.top_k,
            k1=args.k1,
            b=args.b,
            per_doc_top_k=args.top_k_per_doc
            if hasattr(args, "top_k_per_doc")
            else None,
        )

        with darwin.session("keyword_search"):
            if args.store_dir:
                store = DarwinStore(args.store_dir)
            handle_keyword_search(search_args, store)
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
