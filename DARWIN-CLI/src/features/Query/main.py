#!/usr/bin/env python3
"""
darwin_query.py
---------------
CLI for the Darwin query orchestration pipeline.

Usage:
    python darwin_query.py "your query here" [options]

Examples:
    python darwin_query.py "explain transformers"

    python darwin_query.py "summarise this doc" \\
        --pinned-files report_q3.pdf notes.txt

    python darwin_query.py "what does this code do" \\
        --pasted-texts "def foo(): return 42" \\
        --pasted-types code \\
        --urls "https://example.com"

    python darwin_query.py "analyse sales data" \\
        --pinned-files sales.csv \\
        --mock-registry

    python darwin_query.py "compare them" \\
        --output result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

# ---------------------------------------------------------------------------
# Darwin imports
# ---------------------------------------------------------------------------
from src.features.LLM.LLMManager import LLMManager


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PastedTextType = Literal["document", "code", "data", "long_question"]

FollowUpReason = Literal[
    "missing_context",
    "ambiguous_query",
    "missing_parameter",
    "tool_selection",
    "confirmation_required",
    "vague_reference",
    "no_scope",
    "url_no_search",
    "file_not_searchable",
    "ambiguous_file_match",
    "other",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PastedText:
    content: str
    type: PastedTextType


@dataclass
class URLRef:
    url: str
    domain: str
    placeholder: str


@dataclass
class FollowUpOption:
    id: str
    label: str
    value: str


@dataclass
class FollowUpBlock:
    question: str
    options: list[FollowUpOption] = field(default_factory=list)
    allow_custom: bool = False
    answer: Optional[str] = None
    reason: FollowUpReason = "other"


@dataclass
class RawQuery:
    query: str
    pasted_texts: list[PastedText] = field(default_factory=list)
    pinned_files: list[str] = field(default_factory=list)
    extracted_urls: list[URLRef] = field(default_factory=list)


@dataclass
class OrganizedQuery:
    clean_query: str
    questionType: list
    suggestedRepresentation: list
    pasted_texts: list
    extracted_urls: list
    pinned_file_ids: list
    is_followup: bool
    needs_clarification: bool
    followup: list | None


# ---------------------------------------------------------------------------
# Session / state machine
# ---------------------------------------------------------------------------

class QueryState(str, Enum):
    CLASSIFYING = "classifying"
    RESOLVING   = "resolving"
    WAITING     = "waiting_for_clarification"
    READY       = "ready"


@dataclass
class QuerySession:
    state: QueryState
    raw: RawQuery
    classified: dict | None = None
    validated: dict | None = None
    pending_followups: list[FollowUpBlock] = field(default_factory=list)
    answers: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """
You are a query classifier. Return ONLY valid JSON, no explanation, no markdown fences.
""".strip()

CLASSIFY_USER = """
## INPUT
Query: {query}
Pasted texts: {pasted_texts}
Extracted URLs: {extracted_urls}

## OUTPUT SCHEMA
{{
  "clean_query": string,
  "questionType": QuestionType[],
  "suggestedRepresentation": ResponsePresentation[],
  "pasted_texts": PastedText[],
  "extracted_urls": URLRef[],
  "is_followup": boolean
}}

## QUESTION TYPES
fact | instruction | reasoning | comparison | opinion | hypothetical | clarification | multi_part | yes_no | statement

## RESPONSE PRESENTATIONS (best-fit first)
plain_text | markdown | accordion | comparison_table | stats_grid | summary_card | timeline | entity_map | flowchart | bar_chart | stepper | graph

## RULES
- clean_query             : normalize the query, strip file references and URLs
- questionType            : pick one or more that best describe the intent
- suggestedRepresentation : order best-fit first based on questionType
- pasted_texts            : pass through unchanged
- extracted_urls          : pass through unchanged
- is_followup             : true if the query references prior context without restating it
""".strip()


VALIDATE_SYSTEM = """
You are a retrieval validator and clarification planner. Return ONLY valid JSON, no explanation, no markdown fences.
""".strip()

VALIDATE_USER = """
## CONTEXT
Classified query: {classified}
Pinned file IDs: {pinned_file_ids}

File registry (filename → capabilities):
{file_registry}

## RETRIEVAL MODE RULES
Each questionType maps to a retrieval mode and rag requirement:
  fact          → hybrid,   requires_rag=true
  instruction   → hybrid,   requires_rag=true
  reasoning     → hybrid,   requires_rag=true
  comparison    → hybrid,   requires_rag=true
  multi_part    → hybrid,   requires_rag=true
  opinion       → hybrid,   requires_rag=false
  hypothetical  → semantic, requires_rag=false
  clarification → keyword,  requires_rag=false
  yes_no        → keyword,  requires_rag=false
  statement     → keyword,  requires_rag=false

File registry capability mapping:
  keyword  → file must have kw=true
  semantic → file must have embedding=true
  hybrid   → file must have hybrid=true

## YOUR JOB
1. For each pinned file:
   - If NOT in registry          → invalid, reason="ambiguous_file_match"
   - If requires_rag=false       → exclude silently (no clarification needed)
   - If file lacks required mode → invalid, reason="file_not_searchable"
   - Otherwise                   → valid, keep it

2. Decide if clarification is needed:
   - Any invalid files            → needs_clarification=true
   - Query too vague to act on    → needs_clarification=true, reason="ambiguous_query"
   - Required parameter missing   → needs_clarification=true, reason="missing_parameter"
   - Otherwise                    → needs_clarification=false

3. If needs_clarification=true  → populate followup[] with one block per issue.
   If needs_clarification=false → set followup=null.

## OUTPUT SCHEMA
{{
  "pinned_file_ids": string[],
  "needs_clarification": boolean,
  "followup": [
    {{
      "id": string,
      "question": string,
      "options": [{{"id": string, "label": string, "value": string}}],
      "allow_custom": boolean,
      "reason": FollowUpReason
    }}
  ] | null
}}

## FOLLOW UP REASONS
missing_context | ambiguous_query | missing_parameter | tool_selection |
confirmation_required | vague_reference | no_scope | url_no_search |
file_not_searchable | ambiguous_file_match | other
""".strip()


# ---------------------------------------------------------------------------
# LLM caller  —  str response → parsed dict
# ---------------------------------------------------------------------------

def _call_llm(system: str, user: str) -> dict:
    """
    Calls LLMManager → call_small, strips accidental markdown fences,
    returns parsed dict.
    """
    LLMManager.initialize()
    client = LLMManager.get_client()
    raw: str = client.call_small(
        [{"role": "user", "content": user}],
        system=system,
    )

    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]
        clean = clean.rsplit("```", 1)[0].strip()

    return json.loads(clean)


# ---------------------------------------------------------------------------
# File registry
# ---------------------------------------------------------------------------

_USE_MOCK_REGISTRY = False

_MOCK_REGISTRY: dict[str, dict[str, bool]] = {
    "report_q3.pdf": {"kw": True,  "hybrid": True,  "embedding": False},
    "notes.txt":     {"kw": True,  "hybrid": False, "embedding": False},
    "data.csv":      {"kw": True,  "hybrid": True,  "embedding": True},
    "slides.pptx":   {"kw": False, "hybrid": False, "embedding": False},
}


def return_searchable_files() -> dict[str, dict[str, bool]]:
    if _USE_MOCK_REGISTRY:
        return _MOCK_REGISTRY

    from src.features.Storage.DarwinStore import DarwinStore
    store = DarwinStore()
    status = store.status()
    return {
        filename: {
            "kw":        info.get("is_kw_searchable", False),
            "hybrid":    info.get("is_hybrid_searchable", False),
            "embedding": info.get("is_embedding_searchable", False),
        }
        for filename, info in status.get("files", {}).items()
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize(obj) -> str:
    if isinstance(obj, list):
        return json.dumps([o.__dict__ if hasattr(o, "__dict__") else o for o in obj])
    if hasattr(obj, "__dict__"):
        return json.dumps(obj.__dict__)
    return json.dumps(obj)


def is_vague(classified: dict) -> bool:
    types = classified.get("questionType", [])
    query = classified.get("clean_query", "")
    return set(types).issubset({"clarification", "statement"}) and len(query.split()) < 6


def inject_answers(raw: RawQuery, answers: dict[str, str]) -> RawQuery:
    answer_block = PastedText(
        content="\n".join(f"{k}: {v}" for k, v in answers.items()),
        type="document",
    )
    return RawQuery(
        query=raw.query,
        pasted_texts=[*raw.pasted_texts, answer_block],
        pinned_files=raw.pinned_files,
        extracted_urls=raw.extracted_urls,
    )


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

async def handle_query(
    session: QuerySession,
    user_input: str | None = None,
) -> QuerySession:

    match session.state:

        case QueryState.CLASSIFYING:
            step("Prompt 1", "classifying query")
            session.classified = _call_llm(
                system=CLASSIFY_SYSTEM,
                user=CLASSIFY_USER.format(
                    query=session.raw.query,
                    pasted_texts=_serialize(session.raw.pasted_texts),
                    extracted_urls=_serialize(session.raw.extracted_urls),
                ),
            )
            session.state = QueryState.RESOLVING
            return await handle_query(session)   # fall through immediately

        case QueryState.RESOLVING:
            needs_validation = bool(session.raw.pinned_files) or is_vague(session.classified)

            if needs_validation:
                step("Prompt 2", "validating files + clarification check")
                session.validated = _call_llm(
                    system=VALIDATE_SYSTEM,
                    user=VALIDATE_USER.format(
                        classified=json.dumps(session.classified),
                        pinned_file_ids=json.dumps(session.raw.pinned_files),
                        file_registry=json.dumps(return_searchable_files()),
                    ),
                )
            else:
                step("Skipping Prompt 2", "no files, query is clear")
                session.validated = {
                    "pinned_file_ids": [],
                    "needs_clarification": False,
                    "followup": None,
                }

            if session.validated["needs_clarification"]:
                raw_followups = session.validated.get("followup") or []
                session.pending_followups = [
                    FollowUpBlock(
                        question=f["question"],
                        options=[
                            FollowUpOption(id=o["id"], label=o["label"], value=o["value"])
                            for o in f.get("options", [])
                        ],
                        allow_custom=f.get("allow_custom", False),
                        reason=f.get("reason", "other"),
                    )
                    for f in raw_followups
                ]
                session.state = QueryState.WAITING
            else:
                session.state = QueryState.READY

            return session

        case QueryState.WAITING:
            if user_input is None:
                raise ValueError("user_input is required when session is in WAITING state")

            current = session.pending_followups[0]
            session.answers[current.question] = user_input
            session.pending_followups.pop(0)

            if not session.pending_followups:
                session.raw = inject_answers(session.raw, session.answers)
                session.state = QueryState.RESOLVING
                return await handle_query(session)

            return session   # more followups → stay in WAITING

        case QueryState.READY:
            return session

        case _:
            raise ValueError(f"Unknown QueryState: {session.state}")


# ---------------------------------------------------------------------------
# Final assembly
# ---------------------------------------------------------------------------

def build_organized_query(session: QuerySession) -> OrganizedQuery:
    if session.state != QueryState.READY:
        raise RuntimeError(
            f"Session is not READY (current state: {session.state}). "
            "Finish the clarification loop before building OrganizedQuery."
        )
    cl = session.classified
    v  = session.validated
    return OrganizedQuery(
        clean_query=cl["clean_query"],
        questionType=cl["questionType"],
        suggestedRepresentation=cl["suggestedRepresentation"],
        pasted_texts=[
            PastedText(content=p["content"], type=p["type"])
            for p in cl.get("pasted_texts", [])
        ],
        extracted_urls=[
            URLRef(url=u["url"], domain=u["domain"], placeholder=u["placeholder"])
            for u in cl.get("extracted_urls", [])
        ],
        pinned_file_ids=v.get("pinned_file_ids", []),
        is_followup=cl["is_followup"],
        needs_clarification=False,
        followup=None,
    )


# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"


def c(color: str, text: str) -> str:
    return f"{color}{text}{C.RESET}"


def banner(text: str, color: str = C.CYAN) -> None:
    width = 60
    print(f"\n{color}{'─' * width}{C.RESET}")
    print(f"{color}{C.BOLD}  {text}{C.RESET}")
    print(f"{color}{'─' * width}{C.RESET}")


def step(label: str, detail: str = "") -> None:
    print(f"  {c(C.DIM, '→')} {c(C.BOLD, label)}", end="")
    if detail:
        print(f"  {c(C.DIM, detail)}")
    else:
        print()


def info(text: str) -> None:
    print(f"  {c(C.DIM, text)}")


def print_state(session: QuerySession) -> None:
    colors = {
        QueryState.CLASSIFYING: C.BLUE,
        QueryState.RESOLVING:   C.YELLOW,
        QueryState.WAITING:     C.MAGENTA,
        QueryState.READY:       C.GREEN,
    }
    col = colors.get(session.state, C.DIM)
    info(f"state → {c(col, session.state.value)}")


# ---------------------------------------------------------------------------
# Interactive followup prompt
# ---------------------------------------------------------------------------

def prompt_user_for_followup(block: FollowUpBlock) -> str:
    print(f"\n  {c(C.MAGENTA, C.BOLD + '?' + C.RESET + C.MAGENTA)}  {c(C.BOLD, block.question)}")
    print(f"  {c(C.DIM, f'reason: {block.reason}')}")

    if block.options:
        print()
        for i, opt in enumerate(block.options, 1):
            print(f"    {c(C.CYAN, f'[{i}]')} {opt.label}")
        if block.allow_custom:
            print(f"    {c(C.DIM, '[or type your own answer]')}")
        print()
        while True:
            raw = input(f"  {c(C.CYAN, '>')} ").strip()
            if not raw:
                continue
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(block.options):
                    chosen = block.options[idx].value
                    info(f"selected → {c(C.GREEN, chosen)}")
                    return chosen
                print(f"  {c(C.RED, 'Invalid option, try again')}")
            elif block.allow_custom:
                return raw
            else:
                print(f"  {c(C.RED, 'Please pick a number from the list')}")
    else:
        print()
        while True:
            raw = input(f"  {c(C.CYAN, '>')} ").strip()
            if raw:
                return raw


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def print_organized_query(oq: OrganizedQuery) -> None:
    banner("✓  Organized Query", C.GREEN)

    def row(label: str, value: Any) -> None:
        print(f"  {c(C.DIM, f'{label:<26}')} {value}")

    row("clean_query",             c(C.BOLD, oq.clean_query))
    row("questionType",            c(C.CYAN,  ", ".join(oq.questionType)))
    row("suggestedRepresentation", c(C.CYAN,  ", ".join(oq.suggestedRepresentation)))
    row("is_followup",             c(C.YELLOW if oq.is_followup else C.GREEN, str(oq.is_followup)))
    row("needs_clarification",     c(C.GREEN, "False"))
    row("pinned_file_ids",         c(C.CYAN,  str(oq.pinned_file_ids) if oq.pinned_file_ids else "[]"))

    if oq.pasted_texts:
        print(f"\n  {c(C.DIM, 'pasted_texts:')}")
        for pt in oq.pasted_texts:
            preview = pt.content[:80] + ("..." if len(pt.content) > 80 else "")
            print(f"    {c(C.DIM, f'[{pt.type}]')} {preview}")

    if oq.extracted_urls:
        print(f"\n  {c(C.DIM, 'extracted_urls:')}")
        for u in oq.extracted_urls:
            print(f"    {c(C.BLUE, u.url)}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    global _USE_MOCK_REGISTRY
    _USE_MOCK_REGISTRY = args.mock_registry

    # build pasted texts
    pasted_texts: list[PastedText] = []
    for i, content in enumerate(args.pasted_texts or []):
        types = args.pasted_types or []
        ptype = types[i] if i < len(types) else "document"
        pasted_texts.append(PastedText(content=content, type=ptype))

    # build url refs
    extracted_urls: list[URLRef] = []
    for url in (args.urls or []):
        domain = url.split("//")[-1].split("/")[0]
        extracted_urls.append(URLRef(url=url, domain=domain, placeholder=f"[{domain}]"))

    raw = RawQuery(
        query=args.query,
        pasted_texts=pasted_texts,
        pinned_files=args.pinned_files or [],
        extracted_urls=extracted_urls,
    )

    # header
    banner("Darwin Query CLI", C.CYAN)
    step("query", c(C.BOLD, raw.query))
    if raw.pinned_files:
        step("pinned files", ", ".join(raw.pinned_files))
    if raw.pasted_texts:
        step("pasted texts", f"{len(raw.pasted_texts)} block(s)")
    if raw.extracted_urls:
        step("urls", ", ".join(u.url for u in raw.extracted_urls))
    if _USE_MOCK_REGISTRY:
        step("registry", c(C.YELLOW, "mock"))

    session = QuerySession(state=QueryState.CLASSIFYING, raw=raw)

    banner("Processing", C.BLUE)

    loop_count = 0
    max_loops  = 10

    while session.state != QueryState.READY and loop_count < max_loops:
        loop_count += 1
        print_state(session)

        if session.state == QueryState.WAITING:
            remaining = len(session.pending_followups)
            info(f"clarification needed ({remaining} question{'s' if remaining > 1 else ''} remaining)")
            answer = prompt_user_for_followup(session.pending_followups[0])
            session = await handle_query(session, user_input=answer)
        else:
            session = await handle_query(session)

    if session.state != QueryState.READY:
        print(f"\n{c(C.RED, 'ERROR: hit loop limit without reaching READY state')}")
        sys.exit(1)

    print_state(session)

    organized = build_organized_query(session)
    print_organized_query(organized)

    if args.output:
        out = {
            "clean_query":             organized.clean_query,
            "questionType":            organized.questionType,
            "suggestedRepresentation": organized.suggestedRepresentation,
            "pinned_file_ids":         organized.pinned_file_ids,
            "is_followup":             organized.is_followup,
            "needs_clarification":     organized.needs_clarification,
            "pasted_texts": [
                {"content": p.content, "type": p.type}
                for p in organized.pasted_texts
            ],
            "extracted_urls": [
                {"url": u.url, "domain": u.domain}
                for u in organized.extracted_urls
            ],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  {c(C.DIM, 'saved to')} {c(C.GREEN, args.output)}\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="darwin_query",
        description="Darwin query classification + clarification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python darwin_query.py "explain transformers"

  python darwin_query.py "summarise this" --pinned-files report_q3.pdf

  python darwin_query.py "compare them" --mock-registry

  python darwin_query.py "what does this code do" \\
      --pasted-texts "def foo(): return 42" --pasted-types code

  python darwin_query.py "analyse trends" \\
      --pinned-files data.csv \\
      --urls https://example.com/ref \\
      --output result.json \\
      --mock-registry
        """,
    )
    parser.add_argument(
        "query",
        help="The user query string",
    )
    parser.add_argument(
        "--pinned-files", nargs="+", metavar="FILE",
        help="Pinned file names to include in retrieval",
    )
    parser.add_argument(
        "--pasted-texts", nargs="+", metavar="TEXT",
        help="Pasted text blocks (one string per arg)",
    )
    parser.add_argument(
        "--pasted-types", nargs="+", metavar="TYPE",
        choices=["document", "code", "data", "long_question"],
        help="Type for each pasted text block (matches order). Default: document",
    )
    parser.add_argument(
        "--urls", nargs="+", metavar="URL",
        help="URLs to include as extracted references",
    )
    parser.add_argument(
        "--output", metavar="FILE",
        help="Write final OrganizedQuery JSON to this file",
    )
    parser.add_argument(
        "--mock-registry", action="store_true",
        help="Use mock file registry instead of DarwinStore",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    asyncio.run(run(args))