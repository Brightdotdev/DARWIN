"""
src/core/Agent/IntentAnalyzer.py
--------------------------------
Pre-LLM query analysis layer for Darwin.

Classifies intent, detects followups / multi-question / parametric / document-scope /
garbage / clarification-needed, selects response type, suggests blocks, handles context.

Usage:
    from src.core.Agent.IntentAnalyzer import analyze_query, QueryAnalysis, UserMessage

    msg = UserMessage(
        query="help me with a paper",
        context="",
        attachments=[],
        metadata={"type": "normal_reply"},
    )
    analysis = analyze_query(
        message=msg,
        history=[...],
        store=darwin_store,
        enabled_tiers=["easy", "medium"],
    )
"""

import re
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict

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

# Clarification: "static" = hardcoded options, "dynamic" = cheap LLM generates questions
CLARIFICATION_MODE = "static"

# Garbage detection
ENABLE_GARBAGE_CHECK = True

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
QUESTION_POS = {"WP", "WRB", "WDT"}
COMPARISON_LEMMAS = {"compare", "contrast", "differ", "versus", "similarity", "difference"}
SUMMARY_LEMMAS = {"summarize", "summarise", "overview", "brief", "outline"}
SEARCH_LEMMAS = {"find", "locate", "look", "search", "seek", "show", "list"}

# Clarification triggers — broad, vague, underspecified
CLARIFICATION_TRIGGERS = {
    "help me", "help with", "assist me", "i need help",
    "do something", "make something",
    "prepare", "build", "design",
}
CLARIFICATION_MIN_WORDS = 3
CLARIFICATION_MAX_WORDS = 8

# Block composition hints — maps intent to suggested multi-block layouts
BLOCK_MAP: Dict[str, List[List[str]]] = {
    "question": [
        ["plain_text"],
        ["plain_text", "entity_map"],
        ["plain_text", "timeline"],
        ["network_view", "plain_text"],
    ],
    "comparison": [
        ["comparison_table"],
        ["comparison_table", "plain_text"],
    ],
    "summary": [
        ["summary_card"],
        ["summary_card", "plain_text"],
    ],
    "search": [
        ["plain_text"],
        ["stats_grid", "plain_text"],
        ["network_view"],
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════════

IntentName = Literal["question", "comparison", "summary", "search"]


@dataclass
class UserMessage:
    """Structured input from frontend. Replaces raw string query."""
    query: str
    context: str = ""
    attachments: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryAnalysis:
    """The single output of the intent analysis pipeline."""

    query: str
    intent: IntentName
    is_followup: bool
    is_multi_question: bool
    is_parametric: bool
    parametric_seed: Optional[dict] = None
    document_scope: Optional[str] = None
    response_type: ResponseTypeName = "plain_text"
    rewritten_queries: list[str] = field(default_factory=list)
    top_k: int = 5
    requires_search: bool = True
    
    is_garbage: bool = False
    needs_clarification: bool = False
    clarification_options: list[dict] = field(default_factory=list)
    suggested_blocks: list[str] = field(default_factory=list)
    context_source: str = "search"
    
    signals: dict = field(default_factory=dict)


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

    if lemmas & COMPARISON_LEMMAS:
        return "comparison"
    if lemmas & SUMMARY_LEMMAS:
        return "summary"
    if lemmas & SEARCH_LEMMAS:
        if not text.endswith("?") and not any(t.tag_ in QUESTION_POS for t in doc):
            return "search"

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

    # BUG FIX: Don't count repeated punctuation as multiple questions
    question_marks = [m.start() for m in re.finditer(r'\?', text)]
    if len(question_marks) >= 2:
        # Check if there's actual text between question marks
        has_real_separation = any(
            text[question_marks[i]:question_marks[i+1]].strip('?').strip()
            for i in range(len(question_marks) - 1)
        )
        if has_real_separation:
            return True

    separators = [" and ", " & ", " plus ", " as well as ", " along with "]
    clauses = [text]
    for sep in separators:
        new_clauses = []
        for c in clauses:
            new_clauses.extend(c.split(sep))
        clauses = new_clauses

    wh_starts = {"who", "what", "where", "when", "why", "how", "which", "whom", "whose"}
    
    question_clauses = 0
    has_wh_setup = False
    
    for i, clause in enumerate(clauses):
        clause = clause.strip()
        if not clause:
            continue
        
        if clause.endswith("?"):
            question_clauses += 1
            continue
            
        words = clause.lower().split()
        if words and words[0] in wh_starts:
            question_clauses += 1
            has_wh_setup = True
            continue
        
        # Implicit predicate after WH setup
        if has_wh_setup and i > 0 and len(words) <= 4:
            if any(w.pos_ in {"NOUN", "ADJ", "VERB", "PROPN"} for w in doc):
                question_clauses += 1
                continue

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

    # If query has WH-words or ends with ?, it's a real question — not a followup
    if any(t.tag_ in QUESTION_POS for t in doc) or text.endswith("?"):
        return False

    # BUG FIX: Imperative commands (Verb + Noun) are standalone, not followups
    if len(doc) >= 2:
        first = doc[0]
        if first.pos_ == "VERB" and any(t.pos_ in {"NOUN", "PROPN"} for t in doc[1:]):
            return False

    # BUG FIX: Multiple content nouns = likely a topic query, not followup
    content_nouns = [t for t in doc if t.pos_ in {"NOUN", "PROPN"} and not t.is_stop]
    if len(content_nouns) >= 2:
        return False

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
    has_keyword = any(kw in text for kw in PARAMETRIC_KEYWORDS)
    if not has_keyword:
        return False, None

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
    candidates = []

    for q in re.findall(r'"([^"]+)"', query) + re.findall(r"'([^']+)'", query):
        candidates.append(q.lower().strip())

    for prefix in DOC_SCOPE_PREFIXES:
        for m in re.finditer(rf"{prefix}\s+([a-z0-9_\-\s]+)", text, re.IGNORECASE):
            candidates.append(m.group(1).strip())

    for chunk in doc.noun_chunks:
        candidate = chunk.text.lower().strip()
        words = candidate.split()
        if len(words) == 1 and len(words[0]) < 4:
            continue
        if candidate in DOC_SCOPE_STOPWORDS:
            continue
        candidates.append(candidate)

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

            if candidate in fname_words and len(candidate) >= 4:
                return filename

            cand_words = set(candidate.split())
            if cand_words and fname_words:
                overlap = len(cand_words & fname_words) / max(len(cand_words), len(fname_words))
                if overlap > best_score and overlap >= 0.5:
                    best_score = overlap
                    best_match = filename

    return best_match


def _detect_garbage(doc, query: str) -> tuple[bool, float]:
    """Heuristic garbage detection. Returns (is_garbage, confidence)."""
    if not ENABLE_GARBAGE_CHECK:
        return False, 0.0

    text = query.strip()
    score = 0.0

    # No alphabetic characters
    if not any(c.isalpha() for c in text):
        score += 0.5

    # No content words
    content_tokens = [t for t in doc if t.pos_ in {"NOUN", "VERB", "PROPN", "ADJ"}]
    if not content_tokens:
        score += 0.4

    # All stopwords
    if len(doc) > 0 and all(t.is_stop for t in doc):
        score += 0.3

    # Random character distribution
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5 and len(text) > 10:
        score += 0.3

    # Extremely short
    if len(text) < 3:
        score += 0.3

    # High out-of-vocabulary ratio = likely gibberish
    alpha_tokens = [t for t in doc if t.is_alpha]
    if alpha_tokens:
        oov_ratio = sum(1 for t in alpha_tokens if t.is_oov) / len(alpha_tokens)
        # BUG FIX: Lower threshold from > 2 to > 1
        if oov_ratio > 0.8 and len(alpha_tokens) > 1:
            score += 0.4

    # BUG FIX: Single long gibberish word
    if len(doc) == 1 and doc[0].is_alpha and doc[0].is_oov and len(doc[0].text) > 6:
        score += 0.5

    is_garbage = score >= 0.3
    return is_garbage, round(score, 2)


def _detect_clarification(doc, query: str, intent: str, is_followup: bool) -> tuple[bool, list[dict]]:
    """Detect vague queries that need disambiguation before answering."""
    if is_followup:
        return False, []

    text = query.lower().strip()
    words = text.split()
    lemmas = {token.lemma_.lower() for token in doc}

    # Skip if query has specific entities — it's not vague
    if len(doc.ents) > 0:
        return False, []

    # Too short + broad intent words
    has_trigger = any(trig in text for trig in CLARIFICATION_TRIGGERS)
    is_short = CLARIFICATION_MIN_WORDS <= len(words) <= CLARIFICATION_MAX_WORDS

    # No specific entities + broad verbs
    no_entities = len(doc.ents) == 0
    broad_verbs = lemmas & {"help", "make", "do", "prepare", "build", "design"}

    needs_it = (has_trigger and is_short and no_entities) or (has_trigger and broad_verbs and len(words) < 12)

    if not needs_it:
        return False, []

    options = _build_clarification_options(intent, text)
    return True, options


def _build_clarification_options(intent: str, query: str) -> list[dict]:
    """Static clarification options. Dynamic mode swaps this for LLM call."""
    if CLARIFICATION_MODE == "dynamic":
        return [{"label": "Custom", "value": "custom", "prompt": "Tell me more..."}]

    base_options = {
        "question": [
            {"label": "Get a summary", "value": "summary", "intent": "summary"},
            {"label": "Deep dive with sources", "value": "deep_dive", "intent": "question"},
            {"label": "Compare with related topics", "value": "compare", "intent": "comparison"},
            {"label": "Something else...", "value": "custom", "intent": "question"},
        ],
        "search": [
            {"label": "List key facts", "value": "list", "intent": "search"},
            {"label": "Show statistics", "value": "stats", "intent": "search"},
            {"label": "Find specific quotes", "value": "quotes", "intent": "search"},
            {"label": "Something else...", "value": "custom", "intent": "search"},
        ],
        "comparison": [
            {"label": "Side-by-side table", "value": "table", "intent": "comparison"},
            {"label": "Pros and cons list", "value": "procon", "intent": "comparison"},
            {"label": "Something else...", "value": "custom", "intent": "comparison"},
        ],
        "summary": [
            {"label": "Brief overview", "value": "brief", "intent": "summary"},
            {"label": "Key points only", "value": "bullet", "intent": "summary"},
            {"label": "Full detailed summary", "value": "full", "intent": "summary"},
            {"label": "Something else...", "value": "custom", "intent": "summary"},
        ],
    }

    return base_options.get(intent, base_options["question"])


def _build_suggested_blocks(intent: str, response_type: ResponseTypeName, is_parametric: bool) -> list[str]:
    """Suggest multi-block layout for the planner LLM."""
    if is_parametric:
        return ["parametric", "plain_text"]

    layouts = BLOCK_MAP.get(intent, [["plain_text"]])
    
    for layout in layouts:
        if layout[0] == response_type:
            return layout
    
    return [response_type]


def _rewrite_query(query: str, is_multi_question: bool) -> list[str]:
    if not ENABLE_QUERY_REWRITE:
        return [query]
    return [query]


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_query(
    message: UserMessage,
    history: list[dict],
    store,
    enabled_tiers: Optional[list[str]] = None,
) -> QueryAnalysis:
    """
    Analyze a user message and return QueryAnalysis.

    Main entrypoint. Runs in <20ms, zero API calls.
    """

    if enabled_tiers is None:
        enabled_tiers = ["easy", "medium", "hard"]

    query = message.query.strip()
    context = message.context.strip()
    metadata = message.metadata

    # ── Context source routing ────────────────────────────────────────────────
    context_source = "search"
    if context and len(context) > 500:
        context_source = "user_paste"
        darwin.log(f"Context: user_paste ({len(context)} chars), skipping search", LogType.INFO)
    elif message.attachments:
        context_source = "attachment"
        darwin.log(f"Context: {len(message.attachments)} attachments", LogType.INFO)

    # ── spaCy pass ────────────────────────────────────────────────────────────
    nlp = _get_nlp()
    doc = nlp(query)

    # ── Detectors ─────────────────────────────────────────────────────────────
    intent = _detect_intent(doc, query)
    darwin.log(f"Intent: {intent}", LogType.INFO)

    is_multi = _detect_multi_question(doc, query)
    darwin.log(f"Multi-question: {is_multi}", LogType.INFO)

    is_followup = _detect_followup(doc, query, history, metadata)
    darwin.log(f"Followup: {is_followup}", LogType.INFO)

    is_parametric, parametric_seed = _detect_parametric(doc, query)
    darwin.log(f"Parametric: {is_parametric} {parametric_seed or ''}", LogType.INFO)

    doc_scope = _detect_document_scope(doc, query, store)
    darwin.log(f"Document scope: {doc_scope}", LogType.INFO)

    # BUG FIX: Scoped queries are never followups
    if doc_scope is not None:
        is_followup = False
        darwin.log("Followup overridden: document scope detected", LogType.INFO)

    is_garbage, garbage_score = _detect_garbage(doc, query)
    darwin.log(f"Garbage: {is_garbage} (score: {garbage_score})", LogType.INFO)

    needs_clarification, clarification_options = _detect_clarification(doc, query, intent, is_followup)
    darwin.log(f"Clarification: {needs_clarification}", LogType.INFO)

    # ── Response type ─────────────────────────────────────────────────────────
    response_type = detect_response_type(
        doc=doc,
        query=query,
        multi_question=is_multi,
        is_followup=is_followup,
        enabled_tiers=enabled_tiers,
    )
    darwin.log(f"Response type: {response_type}", LogType.INFO)

    requires_search = RESPONSE_TYPES[response_type].requires_search
    if is_followup or context_source in {"user_paste", "attachment"}:
        requires_search = False
        darwin.log(f"Search disabled: followup={is_followup}, source={context_source}", LogType.INFO)

    # ── Block suggestions ───────────────────────────────────────────────────────
    suggested_blocks = _build_suggested_blocks(intent, response_type, is_parametric)
    darwin.log(f"Suggested blocks: {suggested_blocks}", LogType.INFO)

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
        is_garbage=is_garbage,
        needs_clarification=needs_clarification,
        clarification_options=clarification_options,
        suggested_blocks=suggested_blocks,
        context_source=context_source,
        signals={
            "intent": intent,
            "entity_count": len(doc.ents),
            "token_count": len(doc),
            "pos_tags": [t.tag_ for t in doc],
            "named_entities": [(ent.text, ent.label_) for ent in doc.ents],
            "message_type": metadata.get("type", "normal_reply"),
            "garbage_score": garbage_score,
            "context_length": len(context),
            "attachment_count": len(message.attachments),
        },
    )

    darwin.log(
        f"QueryAnalysis: {analysis.response_type} | "
        f"search={analysis.requires_search} | "
        f"clarify={analysis.needs_clarification} | "
        f"garbage={analysis.is_garbage}",
    )
    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# Backward compat — old string-based API
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_query_legacy(
    query: str,
    history: list[dict],
    store,
    enabled_tiers: Optional[list[str]] = None,
    message_metadata: Optional[dict] = None,
) -> QueryAnalysis:
    """Wrapper for old string-based callers. Wraps query in UserMessage."""
    msg = UserMessage(
        query=query,
        context="",
        attachments=[],
        metadata=message_metadata or {"type": "normal_reply"},
    )
    return analyze_query(message=msg, history=history, store=store, enabled_tiers=enabled_tiers)