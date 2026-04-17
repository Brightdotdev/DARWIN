"""
src/core/Graph/ConceptElevator.py
----------------------------------
Elevate raw yake keywords into real semantic concepts using an LLM.

This is an optional enhancement layer — yake runs first, then this
takes the raw keywords and returns meaningful concepts like
"Atonement", "Divine Trinity", "Mosaic Law" instead of
"lord", "god", "jesus", "law".

One LLM call per document. Anthropic used as placeholder —
swap the _call_llm() function for your agnostic wrapper later.

Pipeline
--------
raw yake keywords (Dict[str, List[int]])
    → format into prompt
    → one LLM call
    → parse JSON response
    → return elevated concepts (same Dict shape, better labels)
"""

import json
import re
from typing import Dict, List, Optional
from collections import defaultdict

from src.lib.Terminal import TerminalUI


# ============================================================================
# System prompt
# ============================================================================

SYSTEM_PROMPT = """You are a knowledge graph concept extractor.

Your job is to take a list of raw keywords extracted statistically from a document
and identify the true semantic concepts they represent.

Rules:
- Merge related keywords into single meaningful concepts
- Name concepts abstractly and domain-agnostically (works for any document, not just religious texts)
- Concepts should be meaningful as standalone mindmap nodes
- Each concept should represent a distinct idea, theme, or entity
- Return between 20 and 40 concepts — quality over quantity
- Prefer noun phrases: "Divine Authority" not "is divine"
- Be specific enough to be meaningful, broad enough to cover related keywords

Output ONLY a JSON array, no preamble, no markdown fences, no explanation:
[
  {
    "concept": "Human-readable concept name",
    "keywords": ["keyword1", "keyword2"],
    "frequency": <sum of keyword frequencies>
  }
]"""


# ============================================================================
# LLM call (Anthropic placeholder)
# ============================================================================

def _call_llm(prompt: str) -> Optional[str]:
    """
    Call Anthropic API and return the response text.
    
    Replace this function with your agnostic LLM wrapper later.
    Just needs to return a string (the model's response).
    """
    return f"{prompt} was passed here haha"


# ============================================================================
# Response parsing
# ============================================================================

def _parse_response(response: str) -> Optional[List[dict]]:
    """
    Parse LLM JSON response into a list of concept dicts.
    Handles common LLM response quirks (markdown fences, extra text).
    """
    if not response:
        return None

    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", response).strip()

    # Find the JSON array
    start = cleaned.find("[")
    end   = cleaned.rfind("]")

    if start == -1 or end == -1:
        TerminalUI.error("LLM response did not contain a JSON array")
        return None

    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        TerminalUI.error(f"Failed to parse LLM response: {e}")
        return None


# ============================================================================
# Concept elevation
# ============================================================================

def elevate_concepts(
    raw_concepts: Dict[str, List[int]],
    max_keywords: int = 100,
) -> Dict[str, List[int]]:
    """
    Elevate raw yake keywords into semantic concepts via LLM.

    Takes the same Dict shape as yake output and returns the same shape —
    drops straight into build_semantic_graph() with no changes needed.

    Args:
        raw_concepts:  keyword → list of chunk indices (from yake)
        max_keywords:  how many top keywords to send to the LLM

    Returns:
        elevated concept → list of chunk indices
        Falls back to raw_concepts if LLM call fails.
    """
    if not raw_concepts:
        return raw_concepts

    # Sort by frequency, take top N for the prompt
    sorted_keywords = sorted(
        raw_concepts.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )[:max_keywords]

    # Format keyword list for prompt
    keyword_lines = "\n".join(
        f"- {kw} (appears in {len(indices)} chunks)"
        for kw, indices in sorted_keywords
    )

    prompt = f"""Here are the top keywords extracted from a document:

{keyword_lines}

Identify the true semantic concepts these keywords represent.
Return the JSON array as instructed."""

    TerminalUI.dim(f"Elevating {len(sorted_keywords)} keywords → concepts via LLM...")

    # Call LLM
    response = _call_llm(prompt)
    if not response:
        TerminalUI.warn("LLM elevation failed — using raw yake keywords")
        return raw_concepts

    # Parse response
    elevated = _parse_response(response)
    if not elevated:
        TerminalUI.warn("Could not parse LLM response — using raw yake keywords")
        return raw_concepts

    # Build new concepts dict
    # Map each elevated concept back to chunk indices
    # by unioning the chunk indices of its source keywords
    keyword_to_indices = dict(sorted_keywords)
    result: Dict[str, List[int]] = {}

    for item in elevated:
        concept  = item.get("concept", "").strip()
        keywords = item.get("keywords", [])

        if not concept:
            continue

        # Union chunk indices from all source keywords
        chunk_indices = []
        seen = set()
        for kw in keywords:
            for idx in keyword_to_indices.get(kw, []):
                if idx not in seen:
                    chunk_indices.append(idx)
                    seen.add(idx)

        # If LLM hallucinated keywords not in our set,
        # fall back to frequency from response
        if not chunk_indices:
            freq = item.get("frequency", 1)
            # Assign dummy indices so the concept passes frequency filter
            chunk_indices = list(range(min(freq, len(raw_concepts))))

        result[concept] = sorted(chunk_indices)

    if not result:
        TerminalUI.warn("LLM returned empty concepts — using raw yake keywords")
        return raw_concepts

    TerminalUI.success(f"Elevated {len(sorted_keywords)} keywords → {len(result)} concepts")
    return result