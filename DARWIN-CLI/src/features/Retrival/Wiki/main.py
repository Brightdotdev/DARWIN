"""
Wiki concept search — scores wiki pages against query with per-file 
satisfaction gating. Follows the same pattern as KeywordSearch.

Usage:
check the readme
"""

from __future__ import annotations

import sys
import argparse
import time
import asyncio
from pathlib import Path
from typing import List, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.features.Retrival.Wiki.WikiSearch import wiki_search
from src.features.Retrival.Wiki.WikiSearchUtils import (
    WikiSearchArgs,
    WikiSearchResponse,
    WikiGateConfig,
    GATE_DEFAULT,
    GATE_STRICT,
    GATE_RELAXED,
)

from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket
from src.core.Logger.DarwinLogger import darwin, LogType

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Inspect — detailed view of results
# ═══════════════════════════════════════════════════════════════════════════

def _print_inspection(response: WikiSearchResponse, query: str) -> None:
    """Print detailed inspection of wiki search results per file."""
    darwin.log("")
    darwin.log(f"── wiki search inspection — '{query}' ──", LogType.INFO)
    darwin.log("")
    
    darwin.log("  metric              value", LogType.INFO)
    darwin.log("  ──────────────────  ─────", LogType.INFO)
    darwin.log(f"  total files         {len(response.file_results)}", LogType.INFO)
    darwin.log(f"  files satisfied     {sum(1 for fr in response.file_results.values() if fr.satisfied)}/{len(response.file_results)}", LogType.INFO)
    darwin.log(f"  overall satisfied   {response.satisfied}", LogType.INFO)
    darwin.log("")
    
    for filename, file_result in response.file_results.items():
        darwin.log(f"── {filename} ──", LogType.INFO)
        darwin.log(f"  satisfied: {file_result.satisfied} — {file_result.gate_reason}", LogType.INFO)
        darwin.log(f"  results:   {len(file_result.results)}", LogType.INFO)
        
        if file_result.results:
            top_score = f"{file_result.results[0].score:.4f}"
            darwin.log(f"  top score: {top_score}", LogType.INFO)
        
        darwin.log("")
        
        for i, r in enumerate(file_result.results[:5], 1):
            darwin.log(f"  rank:          {i}", LogType.INFO)
            darwin.log(f"  concept:       {r.concept}", LogType.INFO)
            darwin.log(f"  score:         {r.score:.4f}", LogType.INFO)
            darwin.log(f"  centroid sim:  {r.centroid_sim:.4f}", LogType.INFO)
            darwin.log(f"  keyword hits:  {r.keyword_hits}", LogType.INFO)
            darwin.log(f"  frequency:     {r.frequency}", LogType.INFO)
            darwin.log(f"  keywords:      {', '.join(r.keywords) if r.keywords else '(none)'}", LogType.INFO)
            darwin.log(f"  related:       {', '.join(r.related) if r.related else '(none)'}", LogType.INFO)
            
            if r.summary:
                preview = r.summary[:300].replace("\n", " ").strip()
                if len(r.summary) > 300:
                    preview += "..."
                darwin.log(f"  summary:       {preview}", LogType.INFO)
            
            darwin.log("")
            darwin.log("─" * 40, LogType.INFO)
            darwin.log("")
    
    darwin.log("─" * 60, LogType.INFO)


# ═══════════════════════════════════════════════════════════════════════════
# Wiki validation
# ═══════════════════════════════════════════════════════════════════════════

def _is_wiki_searchable(filename: str | None, status: dict) -> bool:
    """Determine if a file has wiki data available."""
    if filename is None:
        return False
        
    key = Path(filename).name
    file_data = status.get("files", {}).get(key)
    print(file_data)
    if file_data is None:
        return False
    return file_data.get("has_wiki", False)


def _validate_files(
    files: List[str], 
    store: DarwinStore
) -> List[str]:
    """Validate files exist and have wiki data, returning valid filenames."""
    status = store.status()
    valid_files = []
    invalid_files = []
    
    for filename in files:
        if _is_wiki_searchable(filename, status):
            valid_files.append(filename)
        else:
            invalid_files.append(filename)
    
    if invalid_files:
        for f in invalid_files:
            darwin.log(f"'{f}' has no wiki data — skipping", LogType.WARNING)

    return valid_files





def _search_wrapper(
    store: DarwinStore,
    query: str,
    files: Optional[List[str]],
    top_k: int,
    config: WikiGateConfig,
) -> WikiSearchResponse:
    """Execute wiki search and return per-file results with satisfaction status."""
    status = store.status()
    tobe_searched = []
    
    if files:
        darwin.log(f"Searching across {' '.join(files)}", LogType.INFO)
        tobe_searched = _validate_files(files, store)
        
        if not tobe_searched:
            darwin.log("No valid files with wiki data. Search aborted.", LogType.ERROR)
            return WikiSearchResponse(
                file_results={},
                satisfied=False,
                gate_reason="FAIL — no valid files with wiki data"
            )
        
        darwin.log(f"Searching {len(tobe_searched)} valid files for '{query}'", LogType.INFO)
    else:
        tobe_searched = [
            fname for fname, info in status.get("files", {}).items()
            if info.get("has_wiki", False)
        ]
        darwin.log(f"Searching {len(tobe_searched)} files for '{query}'", LogType.INFO)
    
    if not tobe_searched:
        darwin.log("No files with wiki data found. Run wiki extraction first.", LogType.WARNING)
        return WikiSearchResponse(
            file_results={},
            satisfied=False,
            gate_reason="FAIL — no files with wiki data"
        )
    
    start_time = time.perf_counter()

    wiki_resuult_response = wiki_search(query=query, store=store, files=tobe_searched, top_k=top_k, config=config)

    if  not wiki_resuult_response.satisfied:
        darwin.log(f"Wiki search resuts unsatisfied for '{query}'", LogType.ERROR)
        darwin.log(f"Gate reason: {wiki_resuult_response.gate_reason}", LogType.ERROR)
        return wiki_resuult_response

    elapsed = time.perf_counter() - start_time

    if wiki_resuult_response.file_results:
        darwin.log(f"Wiki search complete in {elapsed:.2f}s — {wiki_resuult_response.gate_reason}", LogType.SUCCESS)
    else:
        darwin.log(f"No wiki results found for '{query}'", LogType.WARNING)
    
    return wiki_resuult_response

# ═══════════════════════════════════════════════════════════════════════════
# Results Display
# ═══════════════════════════════════════════════════════════════════════════

def _display_search_results(response: WikiSearchResponse, query: str) -> None:
    """Display wiki results with per-file formatting."""
    
    if not response.file_results:
        darwin.log("")
        darwin.log(f"No wiki results found for '{query}'", LogType.WARNING)
        darwin.log("")
        darwin.log("Try:", LogType.INFO)
        darwin.log("  • Using different keywords", LogType.INFO)
        darwin.log("  • Checking if wiki extraction completed", LogType.INFO)
        darwin.log("  • Lowering gate thresholds with --relaxed", LogType.INFO)
        return
    
    darwin.log("")
    darwin.log(f"Wiki search results for '{query}'", LogType.SUCCESS)
    darwin.log("")
    
    # Per-file results table
    for filename, file_result in response.file_results.items():
        status_icon = "✓" if file_result.satisfied else "✗"
        darwin.log(f"  {status_icon} {filename} — {file_result.gate_reason}", LogType.INFO)
        darwin.log("")
        
        if file_result.results:
            darwin.log("  #    Score   Concept                       KW Hits  Frequency", LogType.INFO)
            darwin.log("  ───  ──────  ────────────────────────────  ───────  ─────────", LogType.INFO)
            
            for i, r in enumerate(file_result.results, 1):
                darwin.log(
                    f"  {i:<4} {r.score:.3f}  {r.concept:<30}  {r.keyword_hits:<7}  {r.frequency}",
                    LogType.INFO
                )
        darwin.log("")
    
    # Summary stats
    darwin.log(f"  Files with wiki data: {len(response.file_results)}", LogType.INFO)
    darwin.log(f"  Files satisfied:      {sum(1 for fr in response.file_results.values() if fr.satisfied)}", LogType.INFO)
    darwin.log(f"  Overall:              {'SATISFIED' if response.satisfied else 'NOT SATISFIED'}", LogType.INFO)
    darwin.log("")


def _format_wiki_results_json(response: WikiSearchResponse) -> str:
    """Format wiki results as JSON string."""
    import json
    
    output = {
        "query_satisfied": response.satisfied,
        "gate_reason": response.gate_reason,
        "files": {}
    }
    
    for filename, file_result in response.file_results.items():
        output["files"][filename] = {
            "satisfied": file_result.satisfied,
            "gate_reason": file_result.gate_reason,
            "results": [
                {
                    "concept": r.concept,
                    "score": r.score,
                    "centroid_sim": r.centroid_sim,
                    "keyword_hits": r.keyword_hits,
                    "frequency": r.frequency,
                    "keywords": r.keywords,
                    "related": r.related,
                    "summary": r.summary,
                    "chunk_indices": r.chunk_indices,
                }
                for r in file_result.results
            ]
        }
    
    return json.dumps(output, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Core handler
# ═══════════════════════════════════════════════════════════════════════════

def handle_wiki_search(args : WikiSearchArgs) -> WikiSearchResponse:
    """
    Handle wiki search — main entry point matching keyword search pattern.
    
    Returns WikiSearchResponse with per-file results and satisfaction status.
    """
    store = DarwinStore()
    
    darwin.log("")
    darwin.log("Wiki Search", LogType.INFO)
    darwin.log(f"Query: {args.query}", LogType.INFO)
    darwin.log(f"Gate:  min_score={args.config.min_top_score}, min_results={args.config.min_results}, min_kw={args.config.min_keyword_hit}", LogType.INFO)
    darwin.log(f"Top-K: {args.top_k}", LogType.INFO)
    darwin.log(f"Scope: {' • '.join(args.files) if args.files else 'all documents'}", LogType.INFO)
    darwin.log("")

    
    response = _search_wrapper(store, args.query, args.files, args.top_k, args.config)
    _display_search_results(response, args.query)
    
    if args.inspect and response.file_results:
        _print_inspection(response, args.query)
    
    if args.json_output and response.file_results:
        darwin.log("── JSON Output ──", LogType.INFO)
        print(_format_wiki_results_json(response))
    
    return response


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wiki concept search over extracted document knowledge"
    )
    parser.add_argument("query", nargs="?", default=None, help="Search query string")
    parser.add_argument("--query", "-q", dest="query_flag", help="Search query (alternative)")
    
    parser.add_argument("--file", "-f", nargs="+", default=None, help="One or more files to search")
    parser.add_argument("--top-k", "-k", type=int, default=5)
    
    # Gate presets
    gate_group = parser.add_mutually_exclusive_group()
    gate_group.add_argument("--strict", action="store_true", help="Use strict gate thresholds")
    gate_group.add_argument("--relaxed", action="store_true", help="Use relaxed gate thresholds")
    
    # Individual gate overrides
    parser.add_argument("--min-score", type=float, default=None, help="Override min top score threshold")
    parser.add_argument("--min-results", type=int, default=None, help="Override min results threshold")
    parser.add_argument("--min-kw", type=int, default=None, help="Override min keyword hit threshold")
    
    parser.add_argument("--inspect", "-i", action="store_true")
    parser.add_argument("--json", "-j", action="store_true")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")

    args = parser.parse_args()
    
    query = args.query or args.query_flag
    
    if not query:
        parser.error("Query required (either as positional or --query)")
    
    # Select gate config
    if args.strict:
        config = GATE_STRICT
    elif args.relaxed:
        config = GATE_RELAXED
    else:
        config = GATE_DEFAULT
    
    # Apply individual overrides
    if args.min_score is not None:
        config.min_top_score = args.min_score
    if args.min_results is not None:
        config.min_results = args.min_results
    if args.min_kw is not None:
        config.min_keyword_hit = args.min_kw

    # Setup WebSocket
    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        wiki_search_args = WikiSearchArgs(
            query=query,
            files=args.file,
            top_k=args.top_k,
            config=config,
            inspect=args.inspect,
            json_output=args.json,
            ws_port=args.ws_port,
            no_ws=args.no_ws,
        )
        handle_wiki_search(wiki_search_args)
        return 0

    except KeyboardInterrupt:
        darwin.log("")
        darwin.log("Goodbye!", LogType.INFO)
        return 130
    except Exception as e:
        import traceback
        darwin.log(f"Fatal error: {e}", LogType.ERROR)
        darwin.log(''.join(traceback.format_exception(e)), LogType.ERROR)
        return 1
    finally:
        shutdown_websocket(ws_server)


if __name__ == "__main__":
    asyncio.run(main())