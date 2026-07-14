


from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# CHANGED: suppress LiteLLM Bedrock warning spam globally
import warnings
warnings.filterwarnings("ignore", module="litellm")
import os
os.environ["LITELLM_SUPPRESS_DEBUG_INFO"] = "True"

from src.core.Pipeline.Ingestion.Parsing.main import handle_parsing
from src.core.Pipeline.Ingestion.Chunking.main import handle_chunking
from src.core.Pipeline.Ingestion.Parsing.lib import ParsingArgs
from src.core.Pipeline.Ingestion.Chunking.lib import ChunkingArgs
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin
from src.lib.Utils import ensure_writable_directory


# ── helpers ───────────────────────────────────────────────────────────────

def resolve_target(path: str, files: Optional[List[str]]) -> str:
    if not files:
        return path

    parents = {str(Path(f).parent) for f in files}
    if len(parents) == 1:
        return parents.pop()

    darwin.warning("Files span multiple directories. Using provided path.")
    return path


def normalize_filenames(files: Optional[List[str]]) -> Optional[List[str]]:
    if not files:
        return None
    return [Path(f).name for f in files]


def split_markdown_state(files: List[str], store: DarwinStore):
    has_md = [f for f in files if store.has_markdown(f)]
    needs_md = [f for f in files if not store.has_markdown(f)]
    return has_md, needs_md


# ── pipeline ──────────────────────────────────────────────────────────────

def run_parsing(parsing_args: ParsingArgs, store: DarwinStore) -> List[str]:
    files = normalize_filenames(parsing_args.files)

    if parsing_args.skip_if_markdown and files:
        has_md, needs_md = split_markdown_state(files, store)

        if has_md:
            darwin.info(f" 1/2 Parsing skipped for {len(has_md)} file(s)")
            for f in has_md:
                darwin.info(f"  ✓ {f}")
            darwin.info("")

        if not needs_md:
            return has_md

        darwin.info(f" 1/2 Parsing {len(needs_md)} file(s)")
        darwin.info("")

        args = ParsingArgs(**{**vars(parsing_args), "files": needs_md})
        docs = handle_parsing(args, store=store)

        parsed = [d.filename for d in docs if not d.error]
        return has_md + parsed

    # normal path
    darwin.info(" 1/2 Parsing")
    darwin.info("")

    docs =  handle_parsing(parsing_args, store=store)
    if not docs:
        return []

    parsed = [d.filename for d in docs if not d.error]
    failed = [d.filename for d in docs if d.error]

    if failed:
        darwin.warning(f"Skipping {len(failed)} failed parse(s)")
        for f in failed:
            darwin.warning(f"  ✗ {f}")
        darwin.info("")

    return parsed


def run_chunking(filenames: List[str], rerun: bool, store: DarwinStore) -> None:
    darwin.info("")
    darwin.info("━" * 50)
    darwin.info("")
    darwin.info(" 2/2 Chunking")
    darwin.info("")

    args = ChunkingArgs(rerun=rerun, files=filenames)
    handle_chunking(args, store=store)


def  handle_ingestion(parsing_args: ParsingArgs, store: DarwinStore) -> None:
    darwin.info("")
    darwin.info("━" * 50)
    darwin.info("  INGESTION PIPELINE")
    darwin.info("━" * 50)
    darwin.info("")

    parsed_files = run_parsing(parsing_args, store)

    if not parsed_files:
        darwin.error("No documents available to chunk. Aborting.")
        return

    run_chunking(parsed_files, parsing_args.rerun, store)

    darwin.info("")
    darwin.info("━" * 50)
    darwin.success("INGESTION COMPLETE")
    darwin.info("━" * 50)
    darwin.info("")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Unified ingestion pipeline - parse and chunk documents"
    )

    parser.add_argument("path", nargs="?", default=".", help="Directory to scan for documents")
    parser.add_argument("--file", "-f", nargs="+", default=None, help="Specific file(s) to process")

    # parsing flags
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan directories recursively")
    parser.add_argument("--images", "-img", action="store_true", help="Extract images from documents")
    parser.add_argument("--caption", "-c", action="store_true", help="Generate captions for images (requires --images)")
    parser.add_argument("--precise", action="store_true", help="Use pymupdf4llm for layout-accurate extraction")
    parser.add_argument("--rerun", "-R", action="store_true", help="Reprocess cached files")
    parser.add_argument("--no-save", action="store_true", help="Don't save markdown files (preview only)")
    parser.add_argument("--skip-if-markdown", action="store_true", help="Skip files already parsed")
    parser.add_argument("--ocr", action="store_true", help="Run OCR on extracted images (requires --images)")

    # chunking flags

    parser.add_argument("--store-dir", default=None, help="Darwin store directory")

    args = parser.parse_args()

    # CHANGED: use darwin.configure() — no manual WebSocket imports
    darwin.configure(
        show_banner=True,
    )

    try:
        # CHANGED: wrap entire pipeline in session context manager
        with darwin.session("ingestion"):
            target = resolve_target(args.path, args.file)

            parsing_args = ParsingArgs(
                target=target,
                files=args.file,
                images=args.images,
                caption=args.caption,
                recursive=args.recursive,
                rerun=args.rerun,
                ocr=args.ocr,
                precise=args.precise,
                save=not args.no_save,
                skip_if_markdown=args.skip_if_markdown,
            )

         
         
                    
            if args.store_dir and ensure_writable_directory(args.store_dir):
                store = DarwinStore(args.store_dir)
            else:
                store = DarwinStore()


            handle_ingestion(parsing_args, store=store)

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