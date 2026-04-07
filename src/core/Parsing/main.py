"""
src/core/parser.py - Document parser integrated with DarwinStore
"""

import sys
from pathlib import Path
from typing import Optional, List, Callable
from datetime import datetime

# Add parent to path if needed
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.Parsing.Parser import parse as _parse, parse_many as _parse_many
from src.core.Parsing.ParserUtils import ParsedDocument
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


def parse_and_store(
    file_path: str | Path,
    store: DarwinStore,
    extract_images: bool = False,
    caption_images: bool = False,
    strategy: str = "light",
    progress_callback: Optional[Callable] = None,
) -> ParsedDocument:
    """
    Parse a document and save results to DarwinStore.
    
    Args:
        file_path: Path to the document
        store: DarwinStore instance for the project
        force_reprocess: If True, ignore cached data and reprocess
        extract_images: Extract and save images
        caption_images: Generate AI captions for images
        strategy: "light" or "high"
        progress_callback: Optional progress callback
    
    Returns:
        ParsedDocument object
    """
    file_path = Path(file_path).resolve()
    filename = file_path.name
    
    # Check if we can use cached data
    
    if store.has_markdown(filename) :
        TerminalUI.bold("Skipping parsing (cached markdown found)")
        md_txt =  store.load_markdown(filename)
        metadata = store.load_meta_file(filename)

        return ParsedDocument(
            filename=filename,
            text=md_txt,
            page_count=metadata.get("page_count"),
            file_type=metadata.get("file_type"),
            strategy=metadata.get("strategy"),
            error=None,
        )
    
       # Parse the document
    if progress_callback:
        progress_callback(f"📄 Parsing {filename}...")
    
    # Setup image directory if needed
    images_dir = None
    if extract_images:
        images_dir = store._file_dir(filename) / "extracted_images"
    
    doc = _parse(
        file_path,
        strategy=strategy,
        images_dir=images_dir,
        caption=caption_images,
    )
    
    if doc.error:
        if progress_callback:
            progress_callback(f"❌ Failed: {doc.error}")
        return doc
    
    if progress_callback:
        progress_callback(f"✅ Parsed {filename}: {len(doc.text):,} chars, {doc.page_count} pages")
 
    return doc


def parse_many_and_store(
    file_paths: List[str | Path],
    store: DarwinStore,
    force_reprocess: bool = False,
    extract_images: bool = False,
    caption_images: bool = False,
    strategy: str = "light",
    progress_callback: Optional[Callable] = None,
) -> List[ParsedDocument]:
    """
    Parse multiple documents and save to DarwinStore.
    """
    results = []
    
    for i, file_path in enumerate(file_paths):
        if progress_callback:
            progress_callback(f"\n[{i+1}/{len(file_paths)}] Processing...")
        
        doc = parse_and_store(
            file_path,
            store,
            force_reprocess=force_reprocess,
            extract_images=extract_images,
            caption_images=caption_images,
            strategy=strategy,
            progress_callback=progress_callback,
        )
        results.append(doc)
    
    return results


def scan_and_store(
    directory: str | Path,
    store: DarwinStore,
    recursive: bool = False,
    force_reprocess: bool = False,
    extract_images: bool = False,
    caption_images: bool = False,
    strategy: str = "light",
    progress_callback: Optional[Callable] = None,
) -> List[ParsedDocument]:
    """
    Scan a directory and parse all supported files, saving to DarwinStore.
    """
    from src.core.Parsing.Parser import scan_directory
    
    directory = Path(directory).resolve()
    
    if progress_callback:
        progress_callback(f"🔍 Scanning {directory}...")
    
    # Get all files (this doesn't parse them yet)
    all_files = []
    pattern = "**/*" if recursive else "*"
    
    for p in directory.glob(pattern):
        if p.is_file() and not any(part.startswith(".") for part in p.parts):
            from src.core.Parsing.Parser import is_supported
            if is_supported(p):
                all_files.append(p)
    
    if progress_callback:
        progress_callback(f"📁 Found {len(all_files)} supported files")
    
    # Parse and store
    results = parse_many_and_store(
        all_files,
        store,
        force_reprocess=force_reprocess,
        extract_images=extract_images,
        caption_images=caption_images,
        strategy=strategy,
        progress_callback=progress_callback,
    )
    
    return results


# ── CLI Integration ─────────────────────────────────────────────────────────

def main():
    """CLI entry point using DarwinStore"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse documents and store in .darwin")
    parser.add_argument("path", nargs="?", default=".", help="File or directory to parse")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocess (ignore cache)")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan directories recursively")
    parser.add_argument("--images", "-i", action="store_true", help="Extract images")
    parser.add_argument("--caption", "-c", action="store_true", help="Generate image captions")
    parser.add_argument("--strategy", "-s", choices=["light", "high"], default="light")
    parser.add_argument("--project", "-p", help="Project directory (default: current dir)")
    
    args = parser.parse_args()
    
    # Initialize store
    project_dir = Path(args.project) if args.project else Path.cwd()
    store = DarwinStore(project_dir)
    store.init()
    
    print(f"📁 Project: {project_dir}")
    print(f"💾 Store: {store.darwin_dir}")
    print(f"🔄 Mode: {'Force reprocess' if args.force else 'Use cache if fresh'}")
    
    # Process
    path = Path(args.path)
    
    if path.is_file():
        doc = parse_and_store(
            path,
            store,
            force_reprocess=args.force,
            extract_images=args.images,
            caption_images=args.caption,
            strategy=args.strategy,
            progress_callback=print,
        )
        
        if doc.error:
            print(f"\n❌ Error: {doc.error}")
            return 1
        else:
            print(f"\n✅ Success! Saved to {store.darwin_dir}")
            
    elif path.is_dir():
        docs = scan_and_store(
            path,
            store,
            recursive=args.recursive,
            force_reprocess=args.force,
            extract_images=args.images,
            caption_images=args.caption,
            strategy=args.strategy,
            progress_callback=print,
        )
        
        successful = [d for d in docs if not d.error]
        failed = [d for d in docs if d.error]
        
        print(f"\n{'='*60}")
        print(f"📊 SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        if failed:
            print(f"\nFailed files:")
            for doc in failed[:5]:
                print(f"  - {doc.filename}: {doc.error[:80]}")
        
        print(f"\n💾 All data saved to: {store.darwin_dir}")
        
    else:
        print(f"❌ Path not found: {path}")
        return 1
    
    # Show status
    status = store.status()
    print(f"\n📊 Project Status:")
    print(f"   Total files processed: {len(status['files'])}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())