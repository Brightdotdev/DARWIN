from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from src.core.Logger.DarwinLogger import darwin, LogType

# ═══════════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════════

ResponseTypeName = Literal[
    "plain_text",
    "accordion",
    "comparison_table",
    "stats_grid",
    "summary_card",
    "network_view",
    "flowchart",
    "timeline",
    "entity_map",
    "bar_chart",
    "stepper",
    "parametric",
    "follow_up_clarification",
    "context_slot_response",
    "hybrid",
]

ConfidenceLevel = Literal["high", "medium", "low"]

PastedTextType = Literal["document", "code", "data", "long_question"]

QuestionType = Literal["WH", "YN", "imperative", "declarative"]

FollowUpReason = Literal[
    "vague_reference",
    "no_scope",
    "url_no_search",
    "intent_unclear",
    "file_not_searchable",
    "ambiguous_file_match",
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpacyFeatures:
    entities: list[dict]
    root_verb: str | None
    comparison_subjects: list[str] | None


@dataclass
class PastedText:
    content: str
    type: PastedTextType


@dataclass
class FollowUpOption:
    id: str
    label: str
    value: str


@dataclass
class FollowUpBlock:
    question: str
    options: list[FollowUpOption]
    allow_custom: bool
    answer: str | None
    reason: FollowUpReason


@dataclass
class OrganizedQuery:
    clean_query: str
    spacy: SpacyFeatures
    pasted_texts: list[PastedText]
    has_urls: bool
    extracted_urls: list[dict]
    search_enabled: bool
    pinned_file_ids: list[str]
    is_followup: bool
    needs_clarification: bool
    followup: FollowUpBlock | None
    candidate_type: ResponseTypeName
    candidate_confidence: ConfidenceLevel
    override_reason: str | None
    retrieval_hint: str
    top_k: int
    comparison_subjects: list[str] | None


@dataclass
class RoutingPrompt:
    """Prepared prompt data ready to be sent to a small LLM router."""
    system_prompt: str
    user_message: str
    clean_query: str
    spacy_features: SpacyFeatures
    has_urls: bool
    extracted_urls: list[dict]
    search_enabled: bool
    is_followup: bool


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "prepare_routing_prompt",
    "parse_routing_response",
    "OrganizedQuery",
    "RoutingPrompt",
    "SpacyFeatures",
    "PastedText",
    "FollowUpBlock",
    "FollowUpOption",
]


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

_COMPARISON_KEYWORDS = [
    "compare",
    "vs",
    "versus",
    "difference between",
    "contrast",
    "similarities",
    "how do they differ",
]

_SYSTEM_PROMPT = """\
You are the query router for Darwin, a local RAG application.
You receive a user query and metadata. You return a single JSON object.
No explanation. No markdown. No preamble. Raw JSON only.

Your job in order:

1. CLASSIFY each pasted blob into one of: document, code, data, long_question
   Each blob gets its own type. Preserve full content.

2. RESOLVE file references against the file registry.
   Fuzzy match by name or alias.
   If confident single match → pin it, set needs_clarification false.
   If 2+ plausible matches → trigger clarification, reason: ambiguous_file_match,
     list all matches as options.
   If no match → trigger clarification, reason: vague_reference.

3. CHECK searchability for any referenced or pinned files using searchable_files.
   If a file is referenced and all search methods are False →
     trigger clarification, reason: file_not_searchable.
     Ask if user wants to try anyway or pick a different file.
   If a file only supports keyword search but query needs semantic understanding →
     surface this as a warning in the followup question, not a hard block.

4. CHECK scope. If no pinned files, no pasted blobs, no session context,
   no search enabled, and the query clearly requires a document →
     trigger clarification, reason: no_scope.
     Ask naturally — not an error, just a question.

5. CHECK URLs. If has_urls is true and search_enabled is false →
     trigger clarification, reason: url_no_search.
     Reference the domain names from placeholders in your question.
     e.g. "I see you linked arxiv and wikipedia — do you want me to
     search these? You will need search enabled for that."

6. CLASSIFY intent and select candidate_type from:
   plain_text, accordion, comparison_table, stats_grid, summary_card,
   network_view, flowchart, timeline, entity_map, bar_chart,
   stepper, parametric, follow_up_clarification,
   context_slot_response, hybrid

7. SET confidence: high / medium / low
   high   — unambiguous trigger, supporting evidence present
   medium — plausible but another type could apply
   low    — fell through to plain_text or signals contradict type

8. SET override_reason if you considered one type and chose another.
   Explain briefly. Null if no override.

9. REWRITE retrieval_hint as an optimized vector search string.
   Keep: nouns, proper nouns, lemmatized verbs, entities, dates.
   Drop: stopwords, question words, filler.
   If comparison_subjects present: ensure both appear in hint.
   Empty string if needs_clarification is true.

10. SET top_k based on complexity:
    Simple factual: 3
    Summary or overview: 8
    Flowchart or timeline: 10–15
    Comparison (per subject): 10
    Parametric or stepper: 5
    Broad entity map: 15–20
    0 if needs_clarification is true.

If is_reply_to_followup is true:
  Read previous_organized_query.followup.question.
  Read the user's answer from clean_query.
  Resolve the previously unresolved field.
  Return a completed OrganizedQuery.
  Do not trigger clarification for the same reason again.

Return this exact JSON schema — no deviations:

{
  "clean_query": "string",
  "spacy": {
    "entities": [{"text": "string", "label": "string"}],
    "root_verb": "string | null",
    "comparison_subjects": ["string"] | null
  },
  "pasted_texts": [
    {"content": "string", "type": "document|code|data|long_question"}
  ],
  "has_urls": bool,
  "extracted_urls": [
    {"original": "string", "domain": "string", "placeholder": "string"}
  ],
  "search_enabled": bool,
  "pinned_file_ids": ["string"],
  "is_followup": bool,
  "needs_clarification": bool,
  "followup": {
    "question": "string",
    "options": [{"id": "string", "label": "string", "value": "string"}],
    "allow_custom": bool,
    "answer": null,
    "reason": "vague_reference|no_scope|url_no_search|intent_unclear|file_not_searchable|ambiguous_file_match"
  } | null,
  "candidate_type": "ResponseTypeName",
  "candidate_confidence": "high|medium|low",
  "override_reason": "string | null",
  "retrieval_hint": "string",
  "top_k": number,
  "comparison_subjects": ["string"] | null
}"""


# ═══════════════════════════════════════════════════════════════════════════════
# PRIVATE — spaCy lazy loader
# ═══════════════════════════════════════════════════════════════════════════════

_spacy_model = None


def _get_spacy_model():
    global _spacy_model
    if _spacy_model is None:
        from src.Models import ModelManager, ModelCategory

        loader = ModelManager.get_loader()
        extractor = loader.get(ModelCategory.TOKENIZER)
        print("loading spacy model")
        _spacy_model = extractor.nlp
    return _spacy_model


# ═══════════════════════════════════════════════════════════════════════════════
# PRIVATE — URL extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_urls(query: str, linked_urls: list[str]) -> tuple[str, bool, list[dict]]:
    seen_originals: set[str] = set()
    extracted: list[dict] = []
    replacements: list[tuple[str, str]] = []

    for match in _URL_RE.finditer(query):
        raw = match.group(0)
        while raw and raw[-1] in ".,;:!?)'>\"]":
            raw = raw[:-1]

        if raw in seen_originals:
            continue
        seen_originals.add(raw)

        domain = _extract_domain(raw)
        placeholder = f"[link:{domain}]"
        extracted.append(
            {"original": raw, "domain": domain, "placeholder": placeholder}
        )
        replacements.append((raw, placeholder))

    for url in linked_urls:
        if url in seen_originals:
            continue
        seen_originals.add(url)
        domain = _extract_domain(url)
        placeholder = f"[link:{domain}]"
        extracted.append(
            {"original": url, "domain": domain, "placeholder": placeholder}
        )

    clean_query = query
    for original, placeholder in sorted(replacements, key=lambda x: len(x[0]), reverse=True):
        clean_query = clean_query.replace(original, placeholder)

    has_urls = len(extracted) > 0
    return clean_query, has_urls, extracted


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if not netloc:
        parts = url.split("/")
        if len(parts) >= 3:
            netloc = parts[2]
        else:
            netloc = "unknown"
    return netloc.split(".")[0]


# ═══════════════════════════════════════════════════════════════════════════════
# PRIVATE — spaCy preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def _run_spacy(clean_query: str) -> SpacyFeatures:
    nlp = _get_spacy_model()
    doc = nlp(clean_query)

    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    root_verb = next(
        (t.lemma_ for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"),
        None,
    )
    print("using spacy")

    comparison_subjects = _extract_comparison_subjects(doc)

    return SpacyFeatures(
        entities=entities,
        root_verb=root_verb,
        comparison_subjects=comparison_subjects,
    )


def _extract_comparison_subjects(doc) -> list[str] | None:
    text_lower = doc.text.lower()

    keyword_start = -1
    keyword_end = -1
    for kw in _COMPARISON_KEYWORDS:
        idx = text_lower.find(kw)
        if idx != -1:
            keyword_start = idx
            keyword_end = idx + len(kw)
            break

    if keyword_start == -1:
        return None

    candidates: list[tuple[int, int, str]] = []
    for chunk in doc.noun_chunks:
        candidates.append((chunk.start_char, chunk.end_char, chunk.text))
    for ent in doc.ents:
        candidates.append((ent.start_char, ent.end_char, ent.text))

    seen: set[tuple[int, int]] = set()
    unique_candidates: list[tuple[int, int, str]] = []
    for start, end, text in sorted(candidates, key=lambda x: x[0]):
        if (start, end) not in seen:
            seen.add((start, end))
            unique_candidates.append((start, end, text))

    left = [c for c in unique_candidates if c[1] <= keyword_start]
    right = [c for c in unique_candidates if c[0] >= keyword_end]

    subjects: list[str] = []

    if left:
        left.sort(key=lambda c: c[1], reverse=True)
        subjects.append(left[0][2])

    if right:
        right.sort(key=lambda c: c[0])
        subjects.append(right[0][2])

    if not subjects:
        return None

    return subjects


# ═══════════════════════════════════════════════════════════════════════════════
# PRIVATE — LLM prompt builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_llm_prompt(
    clean_query: str,
    pasted_texts: list[str],
    extracted_urls: list[dict],
    search_enabled: bool,
    spacy_features: SpacyFeatures,
    pinned_file_ids: list[str],
    file_registry: list[dict],
    searchable_files: dict,
    session_state: dict | None,
    is_reply_to_followup: bool,
    previous_organized_query: OrganizedQuery | None,
) -> tuple[str, str]:
    metadata: dict[str, Any] = {
        "clean_query": clean_query,
        "pasted_texts": pasted_texts,
        "extracted_urls": extracted_urls,
        "search_enabled": search_enabled,
        "spacy_features": {
            "entities": spacy_features.entities,
            "root_verb": spacy_features.root_verb,
            "comparison_subjects": spacy_features.comparison_subjects,
        },
        "pinned_file_ids": pinned_file_ids,
        "file_registry": file_registry,
        "searchable_files": searchable_files,
        "session_state": session_state,
        "is_reply_to_followup": is_reply_to_followup,
        "previous_organized_query": (
            dataclasses.asdict(previous_organized_query)
            if previous_organized_query is not None
            else None
        ),
    }

    metadata_json = json.dumps(metadata, indent=2, default=str)

    system_prompt = f"{_SYSTEM_PROMPT}\n\n--- INPUT METADATA ---\n{metadata_json}"
    user_message = clean_query

    return system_prompt, user_message


# ═══════════════════════════════════════════════════════════════════════════════
# PRIVATE — Response parsing & hydration
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_markdown_code(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _fallback_query(clean_query: str, spacy: SpacyFeatures) -> OrganizedQuery:
    return OrganizedQuery(
        clean_query=clean_query,
        spacy=spacy,
        pasted_texts=[],
        has_urls=False,
        extracted_urls=[],
        search_enabled=False,
        pinned_file_ids=[],
        is_followup=False,
        needs_clarification=False,
        followup=None,
        candidate_type="plain_text",
        candidate_confidence="low",
        override_reason="LLM response parsing failed — safe fallback applied",
        retrieval_hint=clean_query,
        top_k=5,
        comparison_subjects=None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC — Preprocessing & parsing
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_routing_prompt(
    query: str,
    pasted_texts: list[str] = [],
    linked_urls: list[str] = [],
    pinned_file_ids: list[str] = [],
    file_registry: list[dict] = [],
    searchable_files: dict = {},
    session_state: dict | None = None,
    is_reply_to_followup: bool = False,
    previous_organized_query: OrganizedQuery | None = None,
    search_enabled: bool = False,
) -> RoutingPrompt:
    """
    Preprocess a user query for intent routing.

    Steps:
      1. Extract URLs and build clean_query
      2. Run spaCy for entities, root verb, comparison subjects (skipped for follow-ups)
      3. Build the system prompt and user message for the small router LLM.

    Returns a RoutingPrompt dataclass containing everything the caller needs
    to send to an LLM and later parse the response.
    """
    # Step 1 — URL extraction
    clean_query, has_urls, extracted_urls = _extract_urls(query, linked_urls)

    # Step 2 — spaCy
    if is_reply_to_followup:
        spacy_features = SpacyFeatures(
            entities=[],
            root_verb=None,
            comparison_subjects=None,
        )
    else:
        spacy_features = _run_spacy(clean_query)

    # Step 3 — Build prompt
    system_prompt, user_message = _build_llm_prompt(
        clean_query=clean_query,
        pasted_texts=pasted_texts,
        extracted_urls=extracted_urls,
        search_enabled=search_enabled,
        spacy_features=spacy_features,
        pinned_file_ids=pinned_file_ids,
        file_registry=file_registry,
        searchable_files=searchable_files,
        session_state=session_state,
        is_reply_to_followup=is_reply_to_followup,
        previous_organized_query=previous_organized_query,
    )

    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print(system_prompt)
    print(user_message)
    print(spacy_features)
    print(has_urls)
    print(extracted_urls)
    print(search_enabled)
    print(is_reply_to_followup)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    return RoutingPrompt(
        system_prompt=system_prompt,
        user_message=user_message,
        clean_query=clean_query,
        spacy_features=spacy_features,
        has_urls=has_urls,
        extracted_urls=extracted_urls,
        search_enabled=search_enabled,
        is_followup=is_reply_to_followup,
    )


def parse_routing_response(
    response_text: str,
    routing_prompt: RoutingPrompt,
) -> OrganizedQuery:
    """
    Parse the raw LLM JSON response into an OrganizedQuery, enforcing
    spaCy ground-truth for entities/root_verb/comparison_subjects.
    """
    try:
        stripped = _strip_markdown_code(response_text)
        data = json.loads(stripped)

        required_top = [
            "clean_query",
            "spacy",
            "pasted_texts",
            "has_urls",
            "extracted_urls",
            "search_enabled",
            "pinned_file_ids",
            "is_followup",
            "needs_clarification",
            "followup",
            "candidate_type",
            "candidate_confidence",
            "override_reason",
            "retrieval_hint",
            "top_k",
            "comparison_subjects",
        ]
        for key in required_top:
            if key not in data:
                raise ValueError(f"Missing required field in LLM response: {key}")

        spacy_actual = routing_prompt.spacy_features
        spacy_data = data.get("spacy", {})

        final_entities = spacy_actual.entities
        final_root_verb = spacy_actual.root_verb
        final_comparison_subjects = spacy_actual.comparison_subjects

        # Allow LLM to refine comparison_subjects only if spaCy found one or none
        if spacy_actual.comparison_subjects is not None and len(spacy_actual.comparison_subjects) == 1:
            llm_subjects = spacy_data.get("comparison_subjects")
            if isinstance(llm_subjects, list) and len(llm_subjects) >= 2:
                final_comparison_subjects = llm_subjects
        elif spacy_actual.comparison_subjects is None:
            llm_subjects = spacy_data.get("comparison_subjects")
            if isinstance(llm_subjects, list) and len(llm_subjects) > 0:
                final_comparison_subjects = llm_subjects

        spacy_features = SpacyFeatures(
            entities=final_entities,
            root_verb=final_root_verb,
            comparison_subjects=final_comparison_subjects,
        )

        pasted_texts: list[PastedText] = []
        for pt in data.get("pasted_texts", []):
            if isinstance(pt, dict) and "content" in pt and "type" in pt:
                pasted_texts.append(PastedText(content=pt["content"], type=pt["type"]))

        followup: FollowUpBlock | None = None
        fu = data.get("followup")
        if fu is not None and isinstance(fu, dict):
            options = []
            for opt in fu.get("options", []):
                if isinstance(opt, dict):
                    options.append(
                        FollowUpOption(
                            id=opt.get("id", ""),
                            label=opt.get("label", ""),
                            value=opt.get("value", ""),
                        )
                    )
            followup = FollowUpBlock(
                question=fu.get("question", ""),
                options=options,
                allow_custom=fu.get("allow_custom", True),
                answer=fu.get("answer"),
                reason=fu.get("reason", "intent_unclear"),
            )

        top_level_subjects = None
        if data.get("candidate_type") == "comparison_table":
            top_level_subjects = spacy_features.comparison_subjects

        return OrganizedQuery(
            clean_query=routing_prompt.clean_query,
            spacy=spacy_features,
            pasted_texts=pasted_texts,
            has_urls=routing_prompt.has_urls,
            extracted_urls=routing_prompt.extracted_urls,
            search_enabled=routing_prompt.search_enabled,
            pinned_file_ids=data.get("pinned_file_ids", []),
            is_followup=routing_prompt.is_followup,
            needs_clarification=data.get("needs_clarification", False),
            followup=followup,
            candidate_type=data.get("candidate_type", "plain_text"),
            candidate_confidence=data.get("candidate_confidence", "low"),
            override_reason=data.get("override_reason"),
            retrieval_hint=data.get("retrieval_hint", ""),
            top_k=data.get("top_k", 5),
            comparison_subjects=top_level_subjects,
        )

    except Exception:
        return _fallback_query(routing_prompt.clean_query, routing_prompt.spacy_features)