"""
src/core/Wiki/main.py - Wiki generation CLI

Generate Wikipedia-style documentation from document knowledge graphs.

Usage examples
--------------
# Generate concept index (phonebook, no LLM)
uv run src/core/Wiki/main.py --file bible.pdf

# Generate all documents in index mode
uv run src/core/Wiki/main.py

# Generate Wikipedia-style summaries (requires LLM)
uv run src/core/Wiki/main.py --file biology.pdf --mode summary --llm ollama/llama3.2

# Save to specific output directory
uv run src/core/Wiki/main.py --file bible.pdf --output ./wiki_output

# Limit concepts
uv run src/core/Wiki/main.py --file bible.pdf --max-concepts 30

# Inspect generated content
uv run src/core/Wiki/main.py --file bible.pdf --inspect

# Force regeneration
uv run src/core/Wiki/main.py --file bible.pdf --rerun
"""

import sys
import argparse
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Knowledge.Wiki.WikiGenerator import (
    run_wiki_pipeline,
    WikiIndex,
    WikiPage,
)
from src.core.Knowledge.Wiki.WikiUtils import (
    WikiGenerationArgs,
    WIKI_MODES,
    DEFAULT_MODE,
    DEFAULT_MAX_CONCEPTS,
    DEFAULT_CHUNKS_PER_CONCEPT,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ============================================================================
# Inspect
# ============================================================================

def _print_inspection(wiki: WikiIndex) -> None:
    """Print detailed inspection of generated wiki."""
    TerminalUI.blank()
    TerminalUI.rule(f"wiki inspection — {wiki.filename}")
    TerminalUI.blank()
    
    TerminalUI.table(
        headers=["metric", "value"],
        rows=[
            ["mode", wiki.mode],
            ["total pages", str(len(wiki.pages))],
            ["concepts covered", str(wiki.concept_count)],
        ],
    )
    
    TerminalUI.blank()
    TerminalUI.dim("── index page preview ──")
    TerminalUI.blank()
    
    if wiki.pages:
        index_page = wiki.pages[0]
        preview = index_page.content[:500]
        TerminalUI.panel(preview, title=index_page.title, border_style="dim")
    
    TerminalUI.blank()
    TerminalUI.dim("── sample concept pages ──")
    TerminalUI.blank()
    
    for page in wiki.pages[1:4]:
        TerminalUI.info("concept", page.title)
        TerminalUI.info("chunks referenced", str(page.chunk_count))
        if page.related:
            TerminalUI.info("related", ", ".join(page.related[:5]))
        
        preview = page.content[:300].replace('\n', ' ')
        TerminalUI.dim(f"  {preview}...")
        TerminalUI.blank()
        TerminalUI.divider()
        TerminalUI.blank()
    
    TerminalUI.rule()


# ============================================================================
# Core handler
# ============================================================================

def handle_wiki_generation(args: WikiGenerationArgs) -> None:
    store = DarwinStore(Path(args.project) if args.project else Path.cwd())
    
    # Resolve files
    if args.file:
        filenames = [args.file]
    else:
        status = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info["components"]["graph"]
        ]
    
    if not filenames:
        TerminalUI.error("No files with extracted graphs found. Run graph extraction first.")
        return
    
    TerminalUI.print_darwin()
    TerminalUI.blank()
    TerminalUI.bold(f"Wiki generation — {len(filenames)} file(s)")
    TerminalUI.dim(f"Mode: {args.mode} | Max concepts: {args.max_concepts}")
    if args.mode == "summary":
        TerminalUI.dim(f"LLM: {args.llm_model}")
    TerminalUI.blank()
    
    succeeded = failed = skipped = 0
    
    for filename in filenames:
        TerminalUI.info("Processing", filename)
        
        try:
            output_dir = Path(args.output_dir) if args.output_dir else None
            
            wiki = run_wiki_pipeline(
                filename=filename,
                store=store,
                mode=args.mode,
                llm_model=args.llm_model,
                max_concepts=args.max_concepts,
                chunks_per_concept=args.chunks_per_concept,
                output_dir=output_dir,
                rerun=args.rerun,
            )
            
            if wiki:
                succeeded += 1
                TerminalUI.success(f"  → {len(wiki.pages)} pages generated")
                
                if args.inspect:
                    _print_inspection(wiki)
            else:
                failed += 1
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            TerminalUI.error(f"{filename} — {type(e).__name__}: {e}")
            failed += 1

    
    TerminalUI.blank()
    TerminalUI.success(f"done — {succeeded} generated · {failed} failed")


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Wikipedia-style documentation from knowledge graphs"
    )
    parser.add_argument("path", nargs="?", default=".", help="Project directory")
    parser.add_argument("--file", "-f", default=None, help="Generate for specific file")
    parser.add_argument("--project", "-p", default=None, help="Project directory")
    parser.add_argument("--mode", "-m", default=DEFAULT_MODE, choices=WIKI_MODES,
                       help="Generation mode: index (no LLM) or summary (LLM)")
    parser.add_argument("--llm", default="ollama/llama3.2",
                       help="LLM model for summary mode")
    parser.add_argument("--max-concepts", type=int, default=DEFAULT_MAX_CONCEPTS,
                       help="Maximum concepts to include")
    parser.add_argument("--chunks-per-concept", type=int, default=DEFAULT_CHUNKS_PER_CONCEPT,
                       help="Max chunks per concept for context")
    parser.add_argument("--output", "-o", default=None,
                       help="Output directory for .md files")
    parser.add_argument("--inspect", "-i", action="store_true",
                       help="Show detailed preview of generated content")
    parser.add_argument("--rerun", "-R", action="store_true",
                       help="Force regeneration")
    
    args = parser.parse_args()
    
    handle_wiki_generation(WikiGenerationArgs(
        target=args.path,
        file=args.file,
        project=args.project or args.path,
        mode=args.mode,
        llm_model=args.llm,
        max_concepts=args.max_concepts,
        chunks_per_concept=args.chunks_per_concept,
        output_dir=args.output,
        inspect=args.inspect,
        rerun=args.rerun,
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())