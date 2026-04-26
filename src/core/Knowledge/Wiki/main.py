"""
src/core/Knowledge/Wiki/main.py
--------------------------------
Wiki generation CLI.

Usage:
  uv run src/core/Knowledge/Wiki/main.py --file biology.pdf
  uv run src/core/Knowledge/Wiki/main.py --file biology.pdf --inspect
  uv run src/core/Knowledge/Wiki/main.py --file biology.pdf --mode summary --llm ollama/llama3.2
  uv run src/core/Knowledge/Wiki/main.py --file biology.pdf --max-concepts 20
"""

import sys
import argparse
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Knowledge.Wiki.WikiGenerator import run_wiki_pipeline
from src.core.Knowledge.Wiki.WikiUtils import (
    WikiGenerationArgs, DEFAULT_MAX_CONCEPTS, DEFAULT_CHUNKS_PER_CONCEPT,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


def _print_inspection(wiki, filename: str) -> None:
    TerminalUI.blank()
    TerminalUI.rule(f"wiki — {filename}")
    TerminalUI.blank()
    TerminalUI.table(
        headers=["metric", "value"],
        rows=[
            ["mode", wiki.mode],
            ["pages", str(len(wiki.pages))],
            ["concepts", str(wiki.concept_count)],
        ],
    )
    if wiki.pages:
        TerminalUI.blank()
        TerminalUI.dim("── index preview ──")
        TerminalUI.panel(wiki.pages[0].content[:600], title=wiki.pages[0].title)
        TerminalUI.blank()
        TerminalUI.dim("── sample concept pages ──")
        for page in wiki.pages[1:4]:
            preview = page.content[:250].replace("\n", " ")
            TerminalUI.info(page.title, f"{page.chunk_count} chunks")
            TerminalUI.dim(f"  {preview}...")
            TerminalUI.divider()
    TerminalUI.rule()


def handle_wiki(args: WikiGenerationArgs) -> None:
    store = DarwinStore()

    if args.file:
        filenames = [args.file]
    else:
        status = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info["components"]["graph"]
        ]

    if not filenames:
        TerminalUI.error("No files with graphs found. Run graph extraction first.")
        return

    TerminalUI.bold(f"Wiki generation — {len(filenames)} file(s)")
    TerminalUI.dim(f"Mode: {args.mode} | Max concepts: {args.max_concepts}")
    if args.mode == "summary":
        TerminalUI.dim(f"LLM: {args.llm_model}")

    succeeded = failed = 0

    for filename in filenames:
        try:
            wiki = run_wiki_pipeline(
                filename=filename, store=store,
                mode=args.mode, llm_model=args.llm_model,
                max_concepts=args.max_concepts,
                chunks_per_concept=args.chunks_per_concept,
                rerun=args.rerun,
            )
            if wiki:
                succeeded += 1
                TerminalUI.success(f"  {filename} → {len(wiki.pages)} pages")
                if args.inspect:
                    _print_inspection(wiki, filename)
            else:
                failed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            TerminalUI.error(f"  {filename} — {e}")
            failed += 1

    TerminalUI.success(f"Done — {succeeded} generated · {failed} failed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Wikipedia-style docs from concept graphs")
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--file", "-f", default=None)
    parser.add_argument("--mode", "-m", default="index", choices=["index", "summary"])
    parser.add_argument("--llm", default="ollama/llama3.2")
    parser.add_argument("--max-concepts", "-n", type=int, default=DEFAULT_MAX_CONCEPTS)
    parser.add_argument("--chunks-per-concept", type=int, default=DEFAULT_CHUNKS_PER_CONCEPT)
    parser.add_argument("--inspect", "-i", action="store_true")
    parser.add_argument("--rerun", "-R", action="store_true")
    args = parser.parse_args()

    handle_wiki(WikiGenerationArgs(
        target=args.path,
        file=args.file,
        mode=args.mode,
        llm_model=args.llm,
        max_concepts=args.max_concepts,
        chunks_per_concept=args.chunks_per_concept,
        inspect=args.inspect,
        rerun=args.rerun,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())