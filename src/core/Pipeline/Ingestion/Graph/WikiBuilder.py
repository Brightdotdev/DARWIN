"""
Wiki-style entity summaries.

Tier 1: Template-based (always available, no LLM)
Tier 2: spaCy-based enrichment (noun chunk context)
Tier 3: LLM-based (optional, user-triggered)
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict

import pandas as pd

from core.Logger.DarwinLogger import darwin
from core.Models.Infrence.Spacy.lib import EntityRecord
from core.Pipeline.Ingestion.Graph.GraphConfig import GraphConfig


def enrich_entities(
    nodes_df: pd.DataFrame,
    entities: List[EntityRecord],
    chunks: List[Any],
    config: GraphConfig,
) -> pd.DataFrame:
    """
    Add wiki_summary column to nodes DataFrame.

    Tier 1: Template-based summaries (always available)
    Tier 2: spaCy context enrichment (if entity has enough mentions)
    """
    if nodes_df.empty:
        return nodes_df

    entity_map = {r.entity_id: r for r in entities}
    chunk_texts = {}
    for i, c in enumerate(chunks):
        chunk_texts[i] = c.clean if hasattr(c, "clean") else str(c)

    summaries = []
    for _, row in nodes_df.iterrows():
        node_id = row["id"]
        rec = entity_map.get(node_id)
        if rec is None:
            summaries.append("")
            continue

        summary = _build_tier1(rec, row)
        if config.wiki_tier >= 2:
            summary = _build_tier2(rec, summary, chunk_texts)
        summaries.append(summary)

    nodes_df = nodes_df.copy()
    nodes_df["wiki_summary"] = summaries
    return nodes_df


def _build_tier1(rec: EntityRecord, node_row: pd.Series) -> str:
    """Template-based summary from structured data."""
    name = rec.display_name
    label = rec.type
    mention_count = rec.frequency
    source_count = rec.spread
    source_names = node_row.get("sources", [])
    variants = node_row.get("variants", [])

    source_str = ", ".join(source_names[:3]) if source_names else "multiple sources"
    variant_str = ""
    other_names = [v for v in variants if v.lower() != name.lower()][:3]
    if other_names:
        variant_str = f" Also known as: {', '.join(other_names)}."

    lines = [f"{name} ({label})"]
    lines.append(
        f"Mentioned {mention_count} time(s) across {source_count} document(s) "
        f"({source_str}).{variant_str}"
    )

    return " ".join(lines)


def _build_tier2(
    rec: EntityRecord,
    base_summary: str,
    chunk_texts: Dict[int, str],
) -> str:
    """Enrich summary with surrounding context from mentions."""
    context_snippets = []
    seen = set()

    for mention in rec.mentions[:5]:
        chunk_text = chunk_texts.get(mention.chunk_id, "")
        if not chunk_text:
            continue

        snippet = _extract_context_window(
            chunk_text, mention.text, mention.span, window=80
        )
        if snippet and snippet not in seen:
            seen.add(snippet)
            context_snippets.append(snippet)

    if not context_snippets:
        return base_summary

    context_block = " ".join(f'"{s}"' for s in context_snippets[:3])
    return f"{base_summary} Appears in contexts such as: {context_block}"


def _extract_context_window(
    text: str,
    entity_text: str,
    span: tuple,
    window: int = 80,
) -> str:
    """Extract a text window around an entity mention."""
    start, end = span
    if start >= len(text) or end > len(text):
        idx = text.lower().find(entity_text.lower())
        if idx == -1:
            return ""
        start, end = idx, idx + len(entity_text)

    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)

    snippet = text[ctx_start:ctx_end].strip()
    if ctx_start > 0:
        snippet = "..." + snippet
    if ctx_end < len(text):
        snippet = snippet + "..."

    return snippet
