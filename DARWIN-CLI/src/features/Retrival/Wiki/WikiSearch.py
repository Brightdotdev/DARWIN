"""
src/features/Retrival/WikiSearch.py
-------------------------------------
Pure wiki concept search — scores wiki pages against query and returns 
per-file results with satisfaction status. No hybrid fallback logic.

Usage:
    from src.features.Retrival.WikiSearch import wiki_search

    response = wiki_search(
        query="ATP synthesis",
        files=["bio.pdf", "chem.pdf"],
        store=store,
    )
    
    for filename, file_result in response.file_results.items():
        print(f"{filename}: satisfied={file_result.satisfied}")
        for result in file_result.results:
            print(f"  {result.concept}: {result.score:.3f}")
"""

from __future__ import annotations

from typing import List, Optional, Dict

import numpy as np

from src.features.Knowledge.Wiki.WikiUtils import DocumentWiki
from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Retrival.Wiki.WikiSearchUtils import (
    WikiSearchResult,
    WikiSearchResponse,
    FileWikiResults,
    WikiGateConfig,
    GATE_DEFAULT,
    tokenize,
    score_wiki_page,
    is_gate_satisfied,
    get_gate_reason,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _embed_query(query: str) -> Optional[np.ndarray]:
    """Embed the query. Returns None if embedding is unavailable."""
    try:
        from src.Models import ModelManager, ModelCategory
        loader = ModelManager.get_loader()
        embedder = loader.get(ModelCategory.EMBEDDING)
        vec = embedder.encode([query], show_progress_bar=False)[0]
        return np.asarray(vec, dtype=np.float32)
    except Exception as e:
        darwin.log(f"Embedding unavailable for wiki search: {e}", LogType.WARNING)
        return None


def _load_wikis(
    store: DarwinStore,
    files: Optional[List[str]] = None,
) -> List[DocumentWiki]:
    """Load relevant wikis from the store."""
    all_records = store.registry.list_all()
    target_files = set(files) if files else None

    wikis: List[DocumentWiki] = []
    for record in all_records:
        fname = record.original_name
        if target_files and fname not in target_files:
            continue
        if not store.has_wiki(fname):
            darwin.log(f"  No wiki for '{fname}' — skipping", LogType.WARNING)
            continue
        try:
            raw = store.load_wiki(fname)
            wiki = DocumentWiki.from_dict(raw)
            wikis.append(wiki)
        except Exception as e:
            darwin.log(f"  Failed to load wiki for '{fname}': {e}", LogType.WARNING)

    return wikis


def _score_wiki(
    wiki: DocumentWiki,
    query_tokens: List[str],
    query_vec: Optional[np.ndarray],
    config: WikiGateConfig,
    top_k: int,
) -> List[WikiSearchResult]:
    """Score all pages in a single wiki, return top-k."""
    scored = []
    for page in wiki.pages:
        score, centroid_sim, hits = score_wiki_page(
            page, query_tokens, query_vec, config
        )
        scored.append((score, WikiSearchResult(
            concept=page.concept,
            filename=wiki.filename,
            score=score,
            centroid_sim=centroid_sim,
            keyword_hits=hits,
            frequency=page.frequency,
            related=page.related,
            keywords=page.keywords,
            summary=page.summary,
            chunk_indices=page.chunk_indices,
        )))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def wiki_search(
    query: str,
    store: DarwinStore,
    files: Optional[List[str]] = None,
    top_k: int = 5,
    config: WikiGateConfig = GATE_DEFAULT,
) -> WikiSearchResponse:
    """
    Pure wiki concept search — scores wiki pages per file.
    
    Returns WikiSearchResponse with:
        - file_results: Dict[str, FileWikiResults] — per-file results
        - satisfied:    True only if ALL files passed their gates
        - gate_reason:  Overall summary
    
    Args:
        query:  Natural-language search query
        store:  DarwinStore instance
        files:  Restrict to these filenames; None = all wikis
        top_k:  Max results per file
        config: WikiGateConfig for scoring and thresholds
    """
    darwin.log("", LogType.INFO)
    darwin.log("=" * 60, LogType.INFO)
    darwin.log(f"Wiki Search — '{query}'", LogType.INFO)
    darwin.log("=" * 60, LogType.INFO)

    # ── Load wikis ──
    wikis = _load_wikis(store, files)

    if not wikis:
        darwin.log("No wikis available", LogType.WARNING)
        return WikiSearchResponse(
            file_results={},
            satisfied=False,
            gate_reason="FAIL — no wikis available"
        )

    darwin.log(
        f"Loaded {len(wikis)} wiki(s) covering "
        f"{sum(len(w.pages) for w in wikis)} concepts",
        LogType.INFO
    )

    # ── Embed query ──
    query_vec = _embed_query(query)
    if query_vec is not None:
        darwin.log("Query embedded — centroid similarity active", LogType.INFO)
    else:
        darwin.log("No embedding — keyword-only wiki scoring", LogType.INFO)

    # ── Score each wiki independently ──
    query_tokens = tokenize(query)
    file_results: Dict[str, FileWikiResults] = {}
    
    for wiki in wikis:
        darwin.log(f"\nScoring '{wiki.filename}'...", LogType.INFO)
        
        wiki_results = _score_wiki(wiki, query_tokens, query_vec, config, top_k)
        
        # Log results for this file
        for i, r in enumerate(wiki_results, 1):
            darwin.log(
                f"  [{i}] {r.concept:<30}  score={r.score:.3f}  "
                f"centroid={r.centroid_sim:.3f}  kw_hits={r.keyword_hits}  "
                f"freq={r.frequency}",
                LogType.INFO,
            )
        
        # Check gate per file
        satisfied = is_gate_satisfied(wiki_results, config)
        gate_reason = get_gate_reason(wiki_results, config)
        
        if satisfied:
            darwin.log(f"✓ '{wiki.filename}' gate satisfied — {gate_reason}", LogType.SUCCESS)
        else:
            darwin.log(f"✗ '{wiki.filename}' gate not satisfied — {gate_reason}", LogType.WARNING)
        
        file_results[wiki.filename] = FileWikiResults(
            filename=wiki.filename,
            results=wiki_results,
            satisfied=satisfied,
            gate_reason=gate_reason,
        )

    # ── Overall satisfaction ──
    all_satisfied = all(fr.satisfied for fr in file_results.values())
    summary = (
        "ALL files satisfied" if all_satisfied
        else f"{sum(1 for fr in file_results.values() if fr.satisfied)}/{len(file_results)} files satisfied"
    )
    
    darwin.log(f"\nOverall: {summary}", LogType.SUCCESS if all_satisfied else LogType.WARNING)

    return WikiSearchResponse(
        file_results=file_results,
        satisfied=all_satisfied,
        gate_reason=summary,
    )