"""
main.py
-------
Darwin CLI entry point.

Commands:
    start   — scan folder, parse and index all supported files
    search  — search indexed documents with a query
    parse   — parse and chunk a single file (dev/debug)
    status  — show what has been processed
    help    — show available commands
"""

import sys
import time
import argparse
from pathlib import Path

from rich.progress import track

from src.core.LLM import LLM
from src.lib.Terminal import TerminalUI


def cmd_start(args=None):
    """Scan folder and index all supported files."""
    from src.core.Parser  import parse, SUPPORTED_TYPES
    from src.core.Chunker import chunk_document
    from src.core.Embedder import Embedder
    from src.core.Query   import BM25Index
    from src.core.Store   import DarwinStore
    import numpy as np

    directory = Path(args.directory).resolve() if args and args.directory else Path.cwd()
    store     = DarwinStore(directory)
    store.init()

    TerminalUI.blank()
    TerminalUI.bold(f"darwin — indexing {directory}")
    TerminalUI.blank()

    # find supported files
    # if --file passed, only index that one file
    if args and args.file:
        target = directory / args.file
        if not target.exists():
            TerminalUI.error(f"file not found: {args.file}")
            return
        if not target.suffix.lower() in SUPPORTED_TYPES:
            TerminalUI.error(f"unsupported type: {target.suffix}")
            return
        all_files = [target]
    else:
        all_files = [
            p for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_TYPES
            and not p.name.startswith(".")
        ]

    if not all_files:
        TerminalUI.warn("no supported files found")
        TerminalUI.dim(f"supported: {', '.join(SUPPORTED_TYPES.keys())}")
        return

    TerminalUI.success(f"found {len(all_files)} file(s)")
    TerminalUI.blank()

    new_files = skip_files = error_files = 0

    for file_path in track(all_files, description="indexing..."):
        filename = file_path.name

        # freshness check — skip if unchanged
        if store.is_fresh(filename):
            skip_files += 1
            continue

        try:
            # parse
            doc = parse(file_path)
            if doc.error:
                TerminalUI.warn(f"{filename} — {doc.error}")
                error_files += 1
                continue

            # chunk
            chunks = chunk_document(doc.text, source=filename)

            # embed
            embedder   = Embedder()
            result     = embedder.embed(chunks)
            embeddings = np.array([ec.embedding for ec in result.embedded_chunks])

            # BM25
            bm25 = BM25Index()
            bm25.build(chunks)

            # save
            store.save_chunks(filename, chunks,
                              page_count=doc.page_count, parser=doc.strategy)
            store.save_embeddings(filename, embeddings)
            store.save_bm25(filename, bm25)
            store.save_stopwords(filename, bm25._domain_stopwords)

            new_files += 1

        except Exception as e:
            TerminalUI.error(f"{filename} — {e}")
            error_files += 1

    TerminalUI.blank()
    TerminalUI.success("done")
    TerminalUI.info("indexed", str(new_files))
    TerminalUI.info("skipped", str(skip_files))
    if error_files:
        TerminalUI.info("errors", str(error_files))
    TerminalUI.blank()


def cmd_search(args=None):
    """Search indexed documents."""
    from src.core.Search import main as search_main
    import sys

    # inject args into sys.argv for Search.main() argparse
    directory = args.directory if args and args.directory else str(Path.cwd())
    query     = args.query     if args and args.query     else ""

    if not query:
        TerminalUI.error("query is required")
        TerminalUI.dim("usage: python main.py search . \"your question\"")
        return

    # patch sys.argv so Search.main()'s argparse sees the right args
    sys.argv = ["search", directory, query]
    if args and hasattr(args, "top_k") and args.top_k:
        sys.argv += ["--top-k", str(args.top_k)]
    if args and hasattr(args, "strategy") and args.strategy:
        sys.argv += ["--strategy", args.strategy]
    if args and hasattr(args, "file") and args.file:
        sys.argv += ["--file", args.file]

    search_main()


def cmd_parse(args=None):
    """Parse and chunk a single file — useful for debugging."""
    from src.core.Parser  import parse
    from src.core.Chunker import chunk_document

    if not args or not args.file:
        TerminalUI.error("file path required")
        TerminalUI.dim("usage: python main.py parse path/to/file.pdf")
        return

    file_path = Path(args.file)
    if not file_path.exists():
        TerminalUI.error(f"file not found: {file_path}")
        return

    TerminalUI.blank()
    TerminalUI.bold(f"parsing: {file_path.name}")
    TerminalUI.blank()

    t   = time.time()
    doc = parse(file_path, strategy=args.strategy if args else "light")

    if doc.error:
        TerminalUI.error(f"failed: {doc.error}")
        return

    TerminalUI.info("pages", str(doc.page_count))
    TerminalUI.info("characters", f"{len(doc.text):,}")
    TerminalUI.info("file type", doc.file_type)
    TerminalUI.info("parse time", f"{time.time()-t:.2f}s")

    if args and args.chunk:
        TerminalUI.blank()
        t      = time.time()
        chunks = chunk_document(doc.text, source=file_path.name)
        TerminalUI.info("chunks", str(len(chunks)))
        TerminalUI.info("chunk time", f"{time.time()-t:.2f}s")
        TerminalUI.blank()

        for i, chunk in enumerate(chunks[:5]):
            TerminalUI.dim(f"[{i}] {chunk.token_count} tokens — {chunk.text[:80].strip()}...")
        if len(chunks) > 5:
            TerminalUI.dim(f"... and {len(chunks) - 5} more")
    else:
        TerminalUI.blank()
        TerminalUI.dim("preview:")
        TerminalUI.divider()
        TerminalUI.dim(doc.text[:500])
        TerminalUI.divider()

    TerminalUI.blank()


def cmd_status(args=None):
    """Show what has been indexed."""
    from src.core.Store import DarwinStore

    directory = Path(args.directory).resolve() if args and args.directory else Path.cwd()
    store     = DarwinStore(directory)

    if not store.is_initialized():
        TerminalUI.blank()
        TerminalUI.warn(f"darwin not initialized in {directory}")
        TerminalUI.dim("run: python main.py start")
        TerminalUI.blank()
        return

    status = store.status()
    files  = status["files"]

    TerminalUI.blank()
    TerminalUI.bold(f"darwin status — {directory}")
    TerminalUI.blank()

    if not files:
        TerminalUI.dim("no files indexed yet")
        TerminalUI.dim("run: python main.py start")
        TerminalUI.blank()
        return

    for f in files:
        fresh   = "fresh" if f["is_fresh"] else "stale"
        fresh_color = TerminalUI.GREEN if f["is_fresh"] else TerminalUI.YELLOW
        
        has_all = all([f["has_chunks"], f["has_embeddings"], f["has_bm25"]])
        indexed = "indexed" if has_all else "partial"
        indexed_color = TerminalUI.GREEN if has_all else TerminalUI.YELLOW
        
        graph   = "graph ✓" if f["has_graph"] else "no graph"
        graph_color = TerminalUI.GREEN if f["has_graph"] else TerminalUI.DIM

        TerminalUI.bold(f.filename)
        print(f"    {TerminalUI.c(fresh_color, fresh)} · {TerminalUI.c(indexed_color, indexed)} · {TerminalUI.c(graph_color, graph)}")
        TerminalUI.dim(f"    {f['chunk_count']} chunks · {f['page_count']} pages")
        if f["concept_count"]:
            TerminalUI.dim(f"    {f['concept_count']} concepts")
        TerminalUI.blank()


def cmd_graph(args=None):
    """Build or render the knowledge graph for a file."""
    from src.core.Store import DarwinStore
    from src.core.LLM   import LLM
    from src.core.Graph import GraphBuilder
    from src.core.Onboarding import get_llm_config

    directory = Path(args.directory).resolve() if args and args.directory else Path.cwd()
    store     = DarwinStore(directory)

    if not store.is_initialized():
        TerminalUI.blank()
        TerminalUI.error("darwin not initialized — run: python main.py start")
        return

    filename = args.file if args and args.file else None
    if not filename:
        TerminalUI.error("--file required for graph command")
        TerminalUI.dim("usage: python main.py graph --file biology.pdf")
        return

    llm_config = get_llm_config(store)
    llm = LLM.from_config(llm_config)
    builder = GraphBuilder(store, llm)

    # --refine flag: rebuild with LLM even if graph exists
    refine = args.refine if args and hasattr(args, "refine") else False

    if refine:
        if not llm.is_available:
            TerminalUI.error("--refine requires an LLM")
            TerminalUI.dim("set one up: python main.py config")
            return
        chunks = store.load_chunks(filename)
        result = builder.refine(filename, chunks)
    elif not store.has_graph(filename):
        # build graph for first time
        if not store.is_fresh(filename):
            TerminalUI.warn(f"{filename} not indexed yet")
            TerminalUI.dim(f"run: python main.py start --file {filename}")
            return
        chunks = store.load_chunks(filename)
        result = builder.build(filename, chunks)
        TerminalUI.blank()
        TerminalUI.success(f"graph built — {result.concept_count} concepts")
        if result.is_raw:
            TerminalUI.dim("tip: run with --refine after setting up an LLM for cleaner labels")
    
    # render at threshold
    threshold = args.threshold if args and hasattr(args, "threshold") and args.threshold else 0.40
    html_path = builder.render(filename, threshold=threshold, open_browser=True)
    TerminalUI.dim(f"opened in browser")
    TerminalUI.blank()


def cmd_config(args=None):
    """Change Darwin settings."""
    from src.core.Store       import DarwinStore
    from src.core.Onboarding  import run_config

    directory = Path(args.directory).resolve() if args and args.directory else Path.cwd()
    store     = DarwinStore(directory)

    if not store.is_initialized():
        store.init()

    run_config(store)


def cmd_help(args=None):
    TerminalUI.blank()
    TerminalUI.bold("darwin — your knowledge. mapped.")
    TerminalUI.blank()
    TerminalUI.green("start")
    TerminalUI.dim_inline("  <dir>              ")
    TerminalUI.dim("scan and index all documents")
    TerminalUI.green("search")
    TerminalUI.dim_inline(" <dir> \"query\"      ")
    TerminalUI.dim("search indexed documents")
    TerminalUI.green("parse")
    TerminalUI.dim_inline("  <file>              ")
    TerminalUI.dim("parse and preview a single file")
    TerminalUI.green("status")
    TerminalUI.dim_inline(" <dir>              ")
    TerminalUI.dim("show index status")
    TerminalUI.green("help")
    TerminalUI.dim_inline("                       ")
    TerminalUI.dim("show this message")
    TerminalUI.blank()
    TerminalUI.dim("examples:")
    TerminalUI.dim("  uv run main.py start")
    TerminalUI.dim("  uv run main.py search . \"what is photosynthesis\"")
    TerminalUI.dim("  uv run main.py parse biology.pdf --chunk")
    TerminalUI.dim("  uv run main.py status")
    TerminalUI.blank()


# ── Argument parsing ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="darwin",
        description="darwin — your knowledge. mapped.",
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest="command")

    # start
    p_start = subparsers.add_parser("start")
    p_start.add_argument("directory", nargs="?", default=str(Path.cwd()))
    p_start.add_argument("--strategy", choices=["light", "high"], default="light")
    p_start.add_argument("--file", default=None,
                         help="index only this file instead of all files")

    # search
    p_search = subparsers.add_parser("search")
    p_search.add_argument("directory", nargs="?", default=str(Path.cwd()))
    p_search.add_argument("query")
    p_search.add_argument("--top-k",   type=int, default=5, dest="top_k")
    p_search.add_argument("--strategy", choices=["light", "high"], default="light")
    p_search.add_argument("--file", default=None,
                          help="search only this file instead of all indexed files")

    # parse
    p_parse = subparsers.add_parser("parse")
    p_parse.add_argument("file")
    p_parse.add_argument("--chunk",    action="store_true")
    p_parse.add_argument("--strategy", choices=["light", "high"], default="light")

    # graph
    p_graph = subparsers.add_parser("graph")
    p_graph.add_argument("directory", nargs="?", default=str(Path.cwd()))
    p_graph.add_argument("--file",      required=True,  help="file to build graph for")
    p_graph.add_argument("--threshold", type=float, default=0.40,
                         help="edge threshold 0.25 (dense) to 0.65 (sparse). default: 0.40")
    p_graph.add_argument("--refine",    action="store_true",
                         help="rebuild graph using LLM for cleaner concept labels")

    # status
    p_status = subparsers.add_parser("status")
    p_status.add_argument("directory", nargs="?", default=str(Path.cwd()))

    # config
    p_config = subparsers.add_parser("config")
    p_config.add_argument("directory", nargs="?", default=str(Path.cwd()))

    # help
    subparsers.add_parser("help")

    args = parser.parse_args()

    commands = {
        "start":  cmd_start,
        "search": cmd_search,
        "parse":  cmd_parse,
        "graph":  cmd_graph,
        "config": cmd_config,
        "status": cmd_status,
        "help":   cmd_help,
        None:     cmd_help,
    }

    # auto-trigger onboarding on first run
    # skip for help, config, and parse — no store needed
    if args.command not in ("help", "config", "parse", None):
        from src.core.Store      import DarwinStore
        from src.core.Onboarding import needs_onboarding, run_onboarding

        directory = Path(
            getattr(args, "directory", str(Path.cwd()))
        ).resolve()

        store = DarwinStore(directory)
        store.init()

        if needs_onboarding(store):
            run_onboarding(store)

    commands.get(args.command, cmd_help)(args)


if __name__ == "__main__":
    main()