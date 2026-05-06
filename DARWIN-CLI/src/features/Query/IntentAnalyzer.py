"""
src/core/Agent/IntentAnalyzer.py
--------------------------------
Pre-LLM query analysis layer for Darwin.

Classifies intent, detects followups / multi-question / parametric / document-scope,
selects response type, and optionally rewrites queries.

Usage:
    from src.core.Agent.IntentAnalyzer import analyze_query, QueryAnalysis

    analysis = analyze_query(
        query="who is enoch and why did he write a book",
        history=[...],
        store=darwin_store,
        enabled_tiers=["easy"],
        message_metadata={"type": "normal_reply"},
    )
    # analysis.response_type -> "accordion"
    # analysis.requires_search -> True
"""

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from src.features.Query.ResponseTypes import (
    RESPONSE_TYPES,
    ResponseTypeName,
    detect_response_type,
    get_top_k,
)
from src.core.Logger import darwin, LogType


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

# v1: query rewriting disabled. Flip to True when LLM rewriter is ready.
ENABLE_QUERY_REWRITE = False

# Followup heuristics
FOLLOWUP_MAX_WORDS = 6
FOLLOWUP_PRONOUNS = {
    "it", "they", "them", "he", "she", "his", "her", "its", "their",
    "this", "that", "these", "those", "here", "there",
}
FOLLOWUP_PHRASES = [
    "you said", "elaborate", "explain that", "tell me more",
    "what about", "how about", "go on", "continue", "and then",
    "put it in simpler", "simpler words", "help me understand",
    "break it down", "make it clearer", "clarify",
]

# Parametric patterns
PARAMETRIC_KEYWORDS = [
    "what if", "if they were", "suppose", "assume",
    "hypothetically", "if it was", "if there were",
]
PARAMETRIC_REGEX = re.compile(
    r"(?:if|what if|suppose|assume|hypothetically)\s.+?"
    r"(?:were?|is|had|has|was)\s+"
    r"([\d,]+(?:\.\d+)?)\s*"
    r"(feet|meters|times|percent|years|people|days|miles|km|kg|lbs|tons|ft|m|%)?",
    re.IGNORECASE,
)

# Document scope patterns
DOC_SCOPE_PREFIXES = [
    "in the", "from the", "in", "from", "according to",
    "chapter", "book of", "section", "page",
]

DOC_SCOPE_STOPWORDS = {
    "book", "file", "document", "text", "page", "chapter", "section",
    "paper", "article", "report", "note", "guide", "story", "letter",
}
# Intent detection — spaCy-driven
QUESTION_POS = {"WP", "WRB", "WDT"}  # who/what, where/when/how, which
COMPARISON_LEMMAS = {"compare", "contrast", "differ", "versus", "similarity", "difference"}
SUMMARY_LEMMAS = {"summarize", "summarise", "overview", "brief", "outline"}
SEARCH_LEMMAS = {"find", "locate", "look", "search", "seek", "show", "list"}


# ═══════════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════════

IntentName = Literal["question", "comparison", "summary", "search"]


@dataclass
class QueryAnalysis:
    """The single output of the intent analysis pipeline."""

    query: str
    intent: IntentName
    is_followup: bool
    is_multi_question: bool
    is_parametric: bool
    parametric_seed: Optional[dict] = None  # {"value": 450.0, "unit": "feet"}
    document_scope: Optional[str] = None    # matched filename
    response_type: ResponseTypeName = "plain_text"
    rewritten_queries: list[str] = field(default_factory=list)
    top_k: int = 5
    requires_search: bool = True
    signals: dict = field(default_factory=dict)  # raw debug signals


# ═══════════════════════════════════════════════════════════════════════════════
# spaCy loader — lazy, never imported at module level
# ═══════════════════════════════════════════════════════════════════════════════

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        from src.Models import ModelManager, ModelCategory
        loader = ModelManager.get_loader()
        extractor = loader.get(ModelCategory.TOKENIZER)
        _nlp = extractor.nlp
    return _nlp


# ═══════════════════════════════════════════════════════════════════════════════
# Detectors
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_intent(doc, query: str) -> IntentName:
    text = query.lower().strip()
    lemmas = {token.lemma_.lower() for token in doc}

    # Comparison: lemma hit or explicit keywords
    if lemmas & COMPARISON_LEMMAS:
        return "comparison"

    # Summary: lemma hit
    if lemmas & SUMMARY_LEMMAS:
        return "summary"

    # Search: lemma hit, but only if it doesn't look like a question
    if lemmas & SEARCH_LEMMAS:
        if not text.endswith("?") and not any(t.tag_ in QUESTION_POS for t in doc):
            return "search"

    # Question: WH word, trailing ?, or auxiliary inversion
    if text.endswith("?"):
        return "question"
    if any(t.tag_ in QUESTION_POS for t in doc):
        return "question"
    for i, token in enumerate(doc[:-1]):
        nxt = doc[i + 1]
        if token.tag_ in {"VBD", "VBZ", "VBP", "MD"} and nxt.tag_ in {"PRP", "NNP", "NN", "DT"}:
            return "question"

    return "search"


def _detect_multi_question(doc, query: str) -> bool:
    """Detect 2+ questions in a single query."""
    text = query.strip()

    # Two+ explicit question marks
    if text.count("?") >= 2:
        return True

    # Split by conjunctions that typically join questions
    separators = [" and ", " & ", " plus ", " as well as ", " along with "]
    clauses = [text]
    for sep in separators:
        new_clauses = []
        for c in clauses:
            new_clauses.extend(c.split(sep))
        clauses = new_clauses

    wh_starts = {"who", "what", "where", "when", "why", "how", "which", "whom", "whose"}
    question_clauses = 0
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if clause.endswith("?"):
            question_clauses += 1
            continue
        words = clause.lower().split()
        if words and words[0] in wh_starts:
            question_clauses += 1

    return question_clauses >= 2


def _detect_followup(doc, query: str, history: list[dict], message_metadata: Optional[dict]) -> bool:
    """Detect if query is a conversational followup."""

    # Frontend metadata is the strongest signal
    if message_metadata:
        msg_type = message_metadata.get("type", "normal_reply")
        if msg_type in {"highlight_reply", "infographic_interaction", "followup"}:
            return True
        if msg_type != "normal_reply":
            return True

    if not history:
        return False

    text = query.lower().strip()
    words = text.split()

    # Short + pronoun
    if len(words) <= FOLLOWUP_MAX_WORDS:
        if any(token.lower_ in FOLLOWUP_PRONOUNS for token in doc):
            return True

    # Explicit followup phrases
    if any(phrase in text for phrase in FOLLOWUP_PHRASES):
        return True

    # No named entities + short = likely followup
    if len(doc.ents) == 0 and len(words) <= 8:
        return True

    return False


def _detect_parametric(doc, query: str) -> tuple[bool, Optional[dict]]:
    """Detect hypothetical queries with numeric seeds."""

    text = query.lower().strip()

    # Fast keyword gate
    has_keyword = any(kw in text for kw in PARAMETRIC_KEYWORDS)
    if not has_keyword:
        return False, None

    # Extract number + unit
    match = PARAMETRIC_REGEX.search(query)
    if match:
        raw_value = match.group(1).replace(",", "")
        unit = match.group(2)
        try:
            value = float(raw_value)
        except ValueError:
            value = None
        seed = {
            "value": value,
            "unit": unit.lower() if unit else None,
            "raw": match.group(0),
        }
        return True, seed

    # Keyword match but no parseable number — still parametric intent
    return True, None


def _detect_document_scope(doc, query: str, store) -> Optional[str]:
    """Fuzzy match query against known document filenames."""

    try:
        filenames = store.registry.list_all()
    except Exception:
        darwin.log("Document scope: could not list files from store", LogType.WARNING)
        return None

    if not filenames:
        return None

    text = query.lower()

    # Collect candidates: quoted phrases, prefix patterns, noun chunks
    candidates = []

    for q in re.findall(r'"([^"]+)"', query) + re.findall(r"'([^']+)'", query):
        candidates.append(q.lower().strip())

    for prefix in DOC_SCOPE_PREFIXES:
        for m in re.finditer(rf"{prefix}\s+([a-z0-9_\-\s]+)", text, re.IGNORECASE):
            candidates.append(m.group(1).strip())

    for chunk in doc.noun_chunks:
        candidate = chunk.text.lower().strip()
        words = candidate.split()
        # Skip single-word generics — too noisy
        if len(words) == 1 and len(words[0]) < 4:
            continue
        # Skip generic stopwords
        if candidate in DOC_SCOPE_STOPWORDS:
            continue
        candidates.append(candidate)
    # Match against filenames
    best_match = None
    best_score = 0.0

    for filename in filenames:
        fname_lower = filename.lower()
        fname_stem = fname_lower.rsplit(".", 1)[0] if "." in fname_lower else fname_lower
        fname_words = set(fname_stem.replace("_", " ").replace("-", " ").split())

        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue

            # Exact word match = instant win (not loose substring)
            if candidate in fname_words and len(candidate) >= 4:
                return filename

            # Word overlap score
            cand_words = set(candidate.split())
            if cand_words and fname_words:
                overlap = len(cand_words & fname_words) / max(len(cand_words), len(fname_words))
                if overlap > best_score and overlap >= 0.5:
                    best_score = overlap
                    best_match = filename

    return best_match


def _rewrite_query(query: str, is_multi_question: bool) -> list[str]:
    """Stub for query rewriting. v1 disabled — returns raw query as-is."""
    if not ENABLE_QUERY_REWRITE:
        return [query]

    # When enabled: clean fluff, expand acronyms, split multi-questions
    # For now, no-op to keep v1 shipping
    return [query]


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_query(
    query: str,
    history: list[dict],
    store,
    enabled_tiers: Optional[list[str]] = None,
    message_metadata: Optional[dict] = None,
) -> QueryAnalysis:
    """
    Analyze a user query and return a QueryAnalysis dataclass.

    Main entrypoint. Runs in <20ms, zero API calls.
    """

    if enabled_tiers is None:
        enabled_tiers = ["easy", "medium", "hard"]

    # ── spaCy pass ────────────────────────────────────────────────────────────
    nlp = _get_nlp()
    doc = nlp(query)

    # ── Detectors ─────────────────────────────────────────────────────────────
    intent = _detect_intent(doc, query)
    darwin.log(f"Intent: {intent}", LogType.INFO)

    is_multi = _detect_multi_question(doc, query)
    darwin.log(f"Multi-question: {is_multi}", LogType.INFO)

    is_followup = _detect_followup(doc, query, history, message_metadata)
    darwin.log(f"Followup: {is_followup}", LogType.INFO)

    is_parametric, parametric_seed = _detect_parametric(doc, query)
    darwin.log(f"Parametric: {is_parametric} {parametric_seed or ''}", LogType.INFO)

    doc_scope = _detect_document_scope(doc, query, store)
    darwin.log(f"Document scope: {doc_scope}", LogType.INFO)

    # ── Response type ─────────────────────────────────────────────────────────
    response_type = detect_response_type(
        doc=doc,
        query=query,
        multi_question=is_multi,
        is_followup=is_followup,
        enabled_tiers=enabled_tiers,
    )
    darwin.log(f"Response type: {response_type}", LogType.INFO)

    # Followups never search — reuse cached context
    requires_search = RESPONSE_TYPES[response_type].requires_search
    if is_followup:
        requires_search = False
        darwin.log("Followup: search disabled", LogType.INFO)

    # ── Query rewrite ─────────────────────────────────────────────────────────
    rewritten = _rewrite_query(query, is_multi)
    darwin.log(f"Rewritten: {rewritten}", LogType.INFO)

    # ── Assemble ──────────────────────────────────────────────────────────────
    analysis = QueryAnalysis(
        query=query,
        intent=intent,
        is_followup=is_followup,
        is_multi_question=is_multi,
        is_parametric=is_parametric,
        parametric_seed=parametric_seed,
        document_scope=doc_scope,
        response_type=response_type,
        rewritten_queries=rewritten,
        top_k=get_top_k(response_type),
        requires_search=requires_search,
        signals={
            "intent": intent,
            "entity_count": len(doc.ents),
            "token_count": len(doc),
            "pos_tags": [t.tag_ for t in doc],
            "named_entities": [(ent.text, ent.label_) for ent in doc.ents],
            "message_type": message_metadata.get("type") if message_metadata else "normal_reply",
        },
    )

    darwin.log(f"QueryAnalysis: {analysis.response_type} | search={analysis.requires_search}", LogType.INFO)
    return analysis