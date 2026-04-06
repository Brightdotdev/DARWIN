"""
darwin/search.py
----------------
End to end search pipeline.

Wires together:
    parser → chunker → embedder → BM25 → query pipeline → results

Run it:
    python search.py /path/to/your/notes "your question here"
    python search.py /path/to/your/notes "your question here" --strategy high
    python search.py /path/to/your/notes "your question here" --top-k 3

Example:
    python search.py ~/biology-notes "how does chlorophyll absorb light"
"""

import sys
import time
import os
import warnings
import argparse
from pathlib import Path

from src.core.Parser  import parse, scan_directory, ParsedDocument
from src.core.Chunker import chunk_document
from src.core.Embedder import Embedder
from src.core.Query   import QueryPipeline, BM25Index, SemanticIndex
from src.core.Store   import DarwinStore
from src.lib.Terminal import TerminalUI


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


# ── Core pipeline ──────────────────────────────────────────────────────────────

def index_directory(
    store:     DarwinStore,
    directory: Path,
    strategy:  str = "light",
    only_file: str | None = None,
) -> dict[str, list]:
    """
    Parse, chunk, and index all supported files in directory.
    Uses cached data from .darwin if file is fresh — skips reprocessing.
    If only_file is set, indexes only that one file.

    Returns dict of filename → chunks for all successfully indexed files.
    """
    from src.core.Parser import SUPPORTED_TYPES

    # --file flag: index only one specific file
    if only_file:
        target = directory / only_file
        if not target.exists():
            print(f"  {TerminalUI.c(TerminalUI.RED, 'error:')} file not found: {only_file}")
            sys.exit(1)
        if target.suffix.lower() not in SUPPORTED_TYPES:
            print(f"  {TerminalUI.c(TerminalUI.RED, 'error:')} unsupported type: {target.suffix}")
            sys.exit(1)
        all_files = [target]
    else:
        # find all supported files (skip hidden + .darwin)
        all_files = [
            p for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_TYPES
            and not p.name.startswith(".")
        ]

    if not all_files:
        print(f"  {TerminalUI.c(TerminalUI.RED, '✗')} no supported files found in {directory}")
        print(f"  {TerminalUI.c(TerminalUI.DIM, 'supported: pdf, txt, md, docx')}")
        sys.exit(1)

    all_chunks = {}

    for file_path in all_files:
        filename = file_path.name
        print(f"\n  {TerminalUI.c(TerminalUI.BOLD, filename)}")

        # ── check freshness ───────────────────────────────────────────────────
        if store.is_fresh(filename):
            TerminalUI.step("loading from cache")
            t = time.time()
            chunks = store.load_chunks(filename)
            TerminalUI.done(f"{len(chunks)} chunks in {time.time()-t:.2f}s")
            all_chunks[filename] = chunks
            continue

        # ── parse ─────────────────────────────────────────────────────────────
        TerminalUI.step(f"parsing ({strategy})")
        t   = time.time()
        doc = parse(file_path, strategy=strategy)

        if doc.error:
            print(f" {TerminalUI.c(TerminalUI.RED, 'failed')}")
            print(f"  {TerminalUI.c(TerminalUI.DIM, doc.error)}")
            continue

        TerminalUI.done(f"{doc.page_count} pages in {time.time()-t:.2f}s")

        # ── chunk ─────────────────────────────────────────────────────────────
        TerminalUI.step("chunking")
        t      = time.time()
        chunks = chunk_document(doc.text, source=filename)
        TerminalUI.done(f"{len(chunks)} chunks in {time.time()-t:.2f}s")

        # ── embed ─────────────────────────────────────────────────────────────
        TerminalUI.step("embedding")
        t        = time.time()
        embedder = Embedder()
        result   = embedder.embed(chunks)
        embeddings = [ec.embedding for ec in result.embedded_chunks]
        TerminalUI.done(f"dim={result.dimension} in {time.time()-t:.2f}s")

        # ── BM25 ──────────────────────────────────────────────────────────────
        TerminalUI.step("building BM25 index")
        t     = time.time()
        bm25  = BM25Index()
        bm25.build(chunks)
        TerminalUI.done(f"{len(bm25._domain_stopwords)} domain stopwords in {time.time()-t:.2f}s")

        # ── save to store ─────────────────────────────────────────────────────
        TerminalUI.step("saving to .darwin")
        t = time.time()
        import numpy as np
        store.save_chunks(
            filename,
            chunks,
            page_count = doc.page_count,
            parser     = strategy,
        )
        store.save_embeddings(filename, np.array(embeddings))
        store.save_bm25(filename, bm25)
        store.save_stopwords(filename, bm25._domain_stopwords)
        TerminalUI.done(f"in {time.time()-t:.2f}s")

        all_chunks[filename] = chunks

    return all_chunks


def build_pipeline(
    store:      DarwinStore,
    all_chunks: dict[str, list],
) -> QueryPipeline:
    """
    Build QueryPipeline from all indexed chunks.
    Loads embeddings and BM25 from store for each file.
    """
    import numpy as np
    from src.core.Chunker import Chunk

    combined_chunks:     list[Chunk]     = []
    combined_embeddings: list            = []

    for filename, chunks in all_chunks.items():
        combined_chunks.extend(chunks)

        try:
            emb = store.load_embeddings(filename)
            combined_embeddings.append(emb)
        except FileNotFoundError:
            pass

    # build pipeline
    pipeline = QueryPipeline()

    # inject pre-computed embeddings directly into semantic index
    # avoids re-embedding everything on startup
    if combined_embeddings:
        pipeline._semantic._chunks     = combined_chunks
        pipeline._semantic._embeddings = np.vstack(combined_embeddings)

    # build fresh BM25 across all chunks
    # BM25 is fast to build so we don't cache the combined index
    pipeline._bm25.build(combined_chunks)
    pipeline._built = True

    return pipeline


def print_results(result, store: DarwinStore) -> None:
    """Pretty print search results with citations."""
    print()
    print(f"  {TerminalUI.c(TerminalUI.BOLD+TerminalUI.WHITE, 'Results')}  {TerminalUI.c(TerminalUI.DIM, f'({result.total_found} chunks considered)')}")
    print()

    # show query variations
    if len(result.variations) > 1:
        print(f"  {TerminalUI.c(TerminalUI.DIM, 'query variations:')}")
        for v in result.variations:
            print(f"    {TerminalUI.c(TerminalUI.DIM, '·')} {v}")
        print()

    # show results
    for r in result.results:
        rank_label  = TerminalUI.c(TerminalUI.CYAN,   f"#{r.rank + 1}")
        score_label = TerminalUI.c(TerminalUI.GREEN,  f"{r.rerank_score:.3f}")
        source      = TerminalUI.c(TerminalUI.YELLOW, r.chunk.source)
        chunk_ref   = TerminalUI.c(TerminalUI.DIM,    f"chunk {r.chunk.index}")

        print(f"  {rank_label}  {score_label}  {source}  {chunk_ref}")
        print()

        # chunk text — wrap at 70 chars for readability
        text  = r.chunk.text
        words = text.split()
        line  = "     "
        for word in words:
            if len(line) + len(word) + 1 > 74:
                print(TerminalUI.c(TerminalUI.DIM, line))
                line = "     " + word
            else:
                line += " " + word
        if line.strip():
            print(TerminalUI.c(TerminalUI.DIM, line))

        # score breakdown
        print()
        print(
            f"     {TerminalUI.c(TerminalUI.DIM, 'semantic:')} {r.semantic_score:.3f}  "
            f"{TerminalUI.c(TerminalUI.DIM, 'bm25:')} {r.bm25_score:.1f}  "
            f"{TerminalUI.c(TerminalUI.DIM, 'rrf:')} {r.rrf_score:.4f}  "
            f"{TerminalUI.c(TerminalUI.DIM, 'rerank:')} {r.rerank_score:.3f}"
        )
        print()
        print(f"  {'─' * 60}")
        print()

    # LLM-ready context block
    print(f"  {TerminalUI.c(TerminalUI.BOLD+TerminalUI.WHITE, 'Context block')} {TerminalUI.c(TerminalUI.DIM, '(ready for LLM)')}")
    print()
    context = result.pipeline_context if hasattr(result, "pipeline_context") else ""
    print(TerminalUI.c(TerminalUI.DIM, "  " + "\n  ".join(
        f'<chunk index="{r.chunk.index}" source="{r.chunk.source}" '
        f'score="{r.rerank_score:.3f}">'
        for r in result.results
    )))
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Darwin search — hybrid semantic + keyword search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python search.py ~/notes "how does chlorophyll absorb light"
  python search.py ~/notes "what is ATP" --top-k 3
  python search.py ~/notes "Calvin cycle" --strategy high
        """,
    )
    parser.add_argument(
        "directory",
        help="path to folder containing your documents",
    )
    parser.add_argument(
        "query",
        help="your question or search query",
    )
    parser.add_argument(
        "--strategy",
        choices=["light", "high"],
        default="light",
        help="parser strategy: light (fast) or high (detailed). default: light",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="number of results to return. default: 5",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="search only this specific file e.g. --file biology.pdf",
    )

    args      = parser.parse_args()
    directory = Path(args.directory).resolve()

    if not directory.exists():
        print(f"{TerminalUI.c(TerminalUI.RED, 'error:')} directory not found: {directory}")
        sys.exit(1)

    if not directory.is_dir():
        print(f"{TerminalUI.c(TerminalUI.RED, 'error:')} not a directory: {directory}")
        sys.exit(1)

    # ── init store ────────────────────────────────────────────────────────────
    store = DarwinStore(directory)
    store.init()

    # ── splash ────────────────────────────────────────────────────────────────
   
    TerminalUI.print_project_status(store, args.query)

    # ── index ─────────────────────────────────────────────────────────────────
    print(f"  {TerminalUI.c(TerminalUI.BOLD, 'Indexing')}")
    t          = time.time()
    all_chunks = index_directory(store, directory, strategy=args.strategy,
                                  only_file=getattr(args, "file", None))
    total_time = time.time() - t

    if not all_chunks:
        print(f"\n  {TerminalUI.c(TerminalUI.RED, 'no files could be indexed')}")
        sys.exit(1)

    total_chunks = sum(len(c) for c in all_chunks.values())
    print(f"\n  {TerminalUI.c(TerminalUI.GREEN, '✓')} indexed {len(all_chunks)} files · "
          f"{total_chunks} chunks · {total_time:.2f}s")
    print()

    # ── build query pipeline ──────────────────────────────────────────────────
    print(f"  {TerminalUI.c(TerminalUI.BOLD, 'Building pipeline')}")
    TerminalUI.step("wiring semantic + BM25")
    t        = time.time()
    pipeline = build_pipeline(store, all_chunks)
    TerminalUI.done(f"in {time.time()-t:.2f}s")
    print()

    # ── search ────────────────────────────────────────────────────────────────
    print(f"  {TerminalUI.c(TerminalUI.BOLD, 'Searching')}")
    TerminalUI.step(f'"{args.query}"')
    t      = time.time()
    result = pipeline.query(args.query, top_k=args.top_k)
    TerminalUI.done(f"in {time.time()-t:.2f}s")

    # ── print results ─────────────────────────────────────────────────────────
    print_results(result, store)

    # ── context for LLM ───────────────────────────────────────────────────────
    context = pipeline.format_context(result)
    print(f"  {TerminalUI.c(TerminalUI.DIM, 'full context block saved to .darwin/last_context.xml')}")

    context_path = store.darwin_dir / "last_context.xml"
    context_path.write_text(context, encoding="utf-8")
    print()


if __name__ == "__main__":
    main()