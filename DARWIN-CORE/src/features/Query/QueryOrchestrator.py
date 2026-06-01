"""
QueryOrchestrator.py
First-bounce input processor: blob/link extraction, pushback detection,
query rewrite/discovery, lightweight routing, and cache-aware dispatch.
Emits darwin.log structured events for streaming simulation.
"""

import re
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from src.core.Logger.DarwinLogger import darwin, LogType

# Local imports (assumes Darwin project structure)
from src.features.Conversation.ConversationStore import ConversationStore, TurnRecord, BlobRecord
from src.features.Conversation.ConversationCache import ConversationCache


@dataclass
class ExtractedBlob:
    blob_type: str   # "code" | "text" | "link" | "image_ref"
    content: str
    span: tuple      # (start, end) in original message
    language: Optional[str] = None  # for code blocks


@dataclass
class QueryPlan:
    plan_id: str
    conv_id: str
    original_message: str
    rewritten_query: str
    routing_target: str  # "conversation_rag" | "document_rag" | "tool_call" | "hybrid" | "direct"
    extracted_blobs: List[ExtractedBlob] = field(default_factory=list)
    extracted_links: List[str] = field(default_factory=list)
    pushback_detected: bool = False
    pushback_context: Optional[str] = None  # previous turn content if pushback
    cache_hits: List[str] = field(default_factory=list)
    cache_misses: List[str] = field(default_factory=list)
    confidence: float = 1.0  # routing confidence 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


class DarwinLogger:
    """
    Structured event logger simulating streaming updates.
    Writes JSON lines to stdout and optionally to a file.
    """

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = Path(log_file) if log_file else None

    def log(self, event_type: str, payload: Dict[str, Any], 
            conv_id: Optional[str] = None, plan_id: Optional[str] = None):
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "info",
            "event": event_type,
            "conv_id": conv_id,
            "plan_id": plan_id,
            "payload": payload
        }
        line = json.dumps(event, default=str)
        darwin.log(line, flush=True)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(line + "\n")

    def log_stage(self, stage: str, status: str, details: Dict[str, Any],
                  conv_id: Optional[str] = None, plan_id: Optional[str] = None):
        self.log(f"stage.{stage}", {"status": status, **details}, conv_id, plan_id)


class QueryOrchestrator:
    """
    First-bounce processor. Lightweight heuristics only — no heavy LLM calls here.

    Flow:
      1. Extract blobs & links from raw message
      2. Check cache for similar recent queries
      3. Detect pushback / narrowing against recent turns
      4. Rewrite / expand query intent
      5. Route to downstream pipeline
      6. Emit darwin.log events at each step
    """

    def __init__(self,
                 store: ConversationStore,
                 cache: ConversationCache,
                 logger: Optional[DarwinLogger] = None,
                 max_history_for_pushback: int = 5):
        self.store = store
        self.cache = cache
        self.logger = logger or DarwinLogger()
        self.max_history = max_history_for_pushback

        # Regex patterns
        self._code_block_re = re.compile(
            r"```(?P<lang>[a-zA-Z0-9+]*)"
        )
        self._inline_code_re = re.compile(r"`([^`]+)`")
        self._url_re = re.compile(
            r"https?://[^\s\)\]>\"\']+"
        )
        self._pushback_re = re.compile(
            r"\b(what about|how about|instead|no[ ,]I meant|go back|"
            r"wait|actually|correction|sorry|clarify|narrow|specifically|"
            r"but cheaper|but faster|smaller|larger|better|worse|other|"
            r"the (one|query|question) (before|above|earlier)|"
            r"can you|could you|would you)\b",
            re.IGNORECASE
        )
        self._short_query_re = re.compile(r"^.{1,40}$")

    # ── Public API ──────────────────────────────────────────

    def process(self, conv_id: str, raw_message: str,
                attachments: Optional[List[BlobRecord]] = None) -> QueryPlan:
        """
        Main entry point. Returns a QueryPlan for downstream execution.
        """
        plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat()

        self.logger.log_stage(
            "orchestrator_start", "running",
            {"message_length": len(raw_message)},
            conv_id, plan_id
        )

        # 1. Extract inline blobs & links
        extracted_blobs = self._extract_blobs(raw_message)
        extracted_links = self._extract_links(raw_message)

        self.logger.log_stage(
            "extraction", "done",
            {"blobs_found": len(extracted_blobs), "links_found": len(extracted_links),
             "blob_types": list(set(b.blob_type for b in extracted_blobs))},
            conv_id, plan_id
        )

        # 2. Load recent history for pushback detection
        recent_turns = self.store.get_turns(conv_id, limit=self.max_history)

        # 3. Detect pushback
        pushback_detected, pushback_context = self._detect_pushback(
            raw_message, recent_turns
        )

        self.logger.log_stage(
            "pushback_check", "done",
            {"detected": pushback_detected, 
             "history_turns": len(recent_turns)},
            conv_id, plan_id
        )

        # 4. Check cache for similar queries
        cache_key = ConversationCache.make_key(
            conv_id, raw_message, [b.content for b in extracted_blobs]
        )
        cached_plan = self.cache.get(cache_key, conv_id=conv_id, tool_name="query_orchestrator")

        cache_hits = []
        cache_misses = ["query_orchestrator"]

        if cached_plan:
            cache_hits.append("query_orchestrator")
            cache_misses.remove("query_orchestrator")
            self.logger.log_stage(
                "cache", "hit",
                {"cache_key": cache_key, "hits": cached_plan.hit_count},
                conv_id, plan_id
            )
            # Return cached plan with fresh metadata
            plan = QueryPlan(**cached_plan.cache_value)
            plan.plan_id = plan_id
            plan.created_at = now
            plan.cache_hits = cache_hits
            plan.cache_misses = cache_misses
            return plan

        self.logger.log_stage(
            "cache", "miss",
            {"cache_key": cache_key},
            conv_id, plan_id
        )

        # 5. Rewrite / expand query
        rewritten = self._rewrite_query(
            raw_message, extracted_blobs, extracted_links,
            pushback_detected, pushback_context, recent_turns
        )

        self.logger.log_stage(
            "rewrite", "done",
            {"original": raw_message[:80], "rewritten": rewritten[:120]},
            conv_id, plan_id
        )

        # 6. Route
        routing_target, confidence = self._route(
            raw_message, rewritten, extracted_blobs, extracted_links,
            pushback_detected, recent_turns
        )

        self.logger.log_stage(
            "routing", "done",
            {"target": routing_target, "confidence": confidence},
            conv_id, plan_id
        )

        # 7. Build plan
        plan = QueryPlan(
            plan_id=plan_id,
            conv_id=conv_id,
            original_message=raw_message,
            rewritten_query=rewritten,
            routing_target=routing_target,
            extracted_blobs=extracted_blobs,
            extracted_links=extracted_links,
            pushback_detected=pushback_detected,
            pushback_context=pushback_context,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            confidence=confidence,
            metadata={
                "history_turns_loaded": len(recent_turns),
                "has_attachments": bool(attachments),
                "attachment_count": len(attachments) if attachments else 0
            },
            created_at=now
        )

        # 8. Persist attachments as blobs if provided
        if attachments:
            for blob in attachments:
                blob.conv_id = conv_id
                # Link to a placeholder turn — downstream will link properly
                blob.turn_id = "pending"
                self.store.attach_blob(blob)

        # 9. Cache the plan (short TTL — plans are cheap to recompute)
        self.cache.set(
            cache_key,
            value=self._plan_to_dict(plan),
            conv_id=conv_id,
            tool_name="query_orchestrator",
            ttl_seconds=300,  # 5 min TTL for plan cache
            metadata={"confidence": confidence, "routing_target": routing_target}
        )

        self.logger.log_stage(
            "orchestrator_complete", "done",
            {"routing_target": routing_target, "plan_id": plan_id},
            conv_id, plan_id
        )

        return plan

    # ── Extraction ──────────────────────────────────────────

    def _extract_blobs(self, message: str) -> List[ExtractedBlob]:
        blobs = []

        # Code blocks (fenced)
        parts = message.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            lang = ""
            if block.startswith("\n"):
                first_line = block.split("\n")[0]
                lang = first_line.strip()
                block = "\n".join(block.split("\n")[1:])
            elif "\n" in block:
                first_line = block.split("\n")[0]
                if first_line.strip() and not first_line.strip().startswith(" "):
                    lang = first_line.strip()
                    block = "\n".join(block.split("\n")[1:])

            blobs.append(ExtractedBlob(
                blob_type="code",
                content=block.strip(),
                span=(0, 0),  # simplified
                language=lang if lang else None
            ))

        # Inline code
        for match in self._inline_code_re.finditer(message):
            blobs.append(ExtractedBlob(
                blob_type="code",
                content=match.group(1),
                span=match.span(),
                language=None
            ))

        # Large pasted text (>500 chars, no code fence)
        if len(message) > 500 and not blobs:
            blobs.append(ExtractedBlob(
                blob_type="text",
                content=message,
                span=(0, len(message))
            ))

        return blobs

    def _extract_links(self, message: str) -> List[str]:
        return self._url_re.findall(message)

    # ── Pushback Detection ─────────────────────────────────

    def _detect_pushback(self, message: str, 
                         recent_turns: List[TurnRecord]) -> tuple[bool, Optional[str]]:
        """
        Heuristic pushback detection.
        Returns (detected, previous_turn_content_for_context).
        """
        if not recent_turns:
            return False, None

        # Signal 1: explicit pushback keywords
        has_pushback_kw = bool(self._pushback_re.search(message))

        # Signal 2: very short query (likely follow-up)
        is_short = bool(self._short_query_re.match(message.strip()))

        # Signal 3: starts with conjunction / comparative
        comparative_starts = ["but ", "and ", "or ", "also ", "what ", "how "]
        starts_comparative = any(
            message.strip().lower().startswith(s) for s in comparative_starts
        )

        # Signal 4: pronoun-heavy (referring back)
        pronoun_count = len(re.findall(
            r"\b(it|that|this|they|them|those|the former|the latter)\b", 
            message, re.IGNORECASE
        ))

        score = 0
        if has_pushback_kw: score += 2
        if is_short: score += 1
        if starts_comparative: score += 1
        if pronoun_count >= 2: score += 1

        # Need at least 2 signals or explicit keyword
        detected = score >= 2 or has_pushback_kw

        if detected and recent_turns:
            # Find the most relevant previous user turn
            user_turns = [t for t in reversed(recent_turns) if t.role == "user"]
            if user_turns:
                return True, user_turns[0].content

        return False, None

    # ── Query Rewrite ───────────────────────────────────────

    def _rewrite_query(self, message: str,
                       blobs: List[ExtractedBlob],
                       links: List[str],
                       pushback: bool,
                       pushback_context: Optional[str],
                       recent_turns: List[TurnRecord]) -> str:
        """
        Lightweight rewrite. No LLM call — rule-based expansion.
        """
        parts = [message.strip()]

        # Merge pushback context
        if pushback and pushback_context:
            parts.insert(0, f"[Context: {pushback_context[:200]}...]")

        # Inline blob summaries
        for blob in blobs:
            if blob.blob_type == "code" and blob.language:
                parts.append(f"[Code snippet: {blob.language}]")
            elif blob.blob_type == "code":
                parts.append("[Code snippet attached]")
            elif blob.blob_type == "text":
                parts.append("[Pasted text attached]")

        # Inline link summaries
        for link in links:
            parts.append(f"[Link: {link}]")

        # If very short and has history, append last assistant response for context
        if len(message.strip()) < 30 and recent_turns:
            last_assistant = None
            for t in reversed(recent_turns):
                if t.role == "assistant":
                    last_assistant = t.content[:150]
                    break
            if last_assistant:
                parts.append(f"[Previous response: {last_assistant}...]")

        return " ".join(parts)

    # ── Routing ─────────────────────────────────────────────

    def _route(self, original: str, rewritten: str,
               blobs: List[ExtractedBlob], links: List[str],
               pushback: bool, recent_turns: List[TurnRecord]) -> tuple[str, float]:
        """
        Lightweight router. Returns (target, confidence).

        Targets:
          - "tool_call"     : explicit tool request, calculation, live data
          - "document_rag"  : code blobs, links, file references
          - "hybrid"        : pushback with history + new constraints
          - "conversation_rag": default — use conversation memory
          - "direct"        : greeting, meta-query, very simple
        """
        lowered = original.lower()

        # Tool signals
        tool_kws = [
            "calculate", "compute", "search", "find", "price", "cost", 
            "compare", "what is the", "how much", "current", "latest",
            "weather", "stock", "news"
        ]
        tool_score = sum(1 for kw in tool_kws if kw in lowered)

        # Document signals
        has_code = any(b.blob_type == "code" for b in blobs)
        has_links = len(links) > 0
        doc_score = (2 if has_code else 0) + (1 if has_links else 0)

        # Direct signals (greetings, meta)
        direct_patterns = [
            r"^hi\b", r"^hello\b", r"^hey\b", r"^thanks?\b",
            r"^ok\b", r"^got it\b", r"^bye\b", r"^what can you do",
            r"^who are you", r"^help\b"
        ]
        is_direct = any(re.search(p, lowered) for p in direct_patterns)

        # Conversation signals
        conv_score = 0
        if pushback: conv_score += 2
        if len(recent_turns) > 2: conv_score += 1
        if len(original) > 100 and not has_code and not has_links: conv_score += 1

        # Scoring
        scores = {
            "tool_call": tool_score,
            "document_rag": doc_score,
            "conversation_rag": conv_score,
            "direct": 3 if is_direct else 0,
            "hybrid": 2 if (pushback and (has_code or has_links or tool_score > 0)) else 0
        }

        best = max(scores, key=scores.get)
        best_score = scores[best]
        total = sum(scores.values()) or 1
        confidence = min(best_score / 3.0, 1.0)  # normalize

        # Tie-breaking: conversation wins for pushback, tool for explicit tool words
        if best_score == 0:
            best = "conversation_rag"
            confidence = 0.5

        return best, round(confidence, 2)

    # ── Serialization ─────────────────────────────────────────

    @staticmethod
    def _plan_to_dict(plan: QueryPlan) -> Dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "conv_id": plan.conv_id,
            "original_message": plan.original_message,
            "rewritten_query": plan.rewritten_query,
            "routing_target": plan.routing_target,
            "extracted_blobs": [
                {"blob_type": b.blob_type, "content": b.content[:200], 
                 "language": b.language} for b in plan.extracted_blobs
            ],
            "extracted_links": plan.extracted_links,
            "pushback_detected": plan.pushback_detected,
            "pushback_context": plan.pushback_context,
            "confidence": plan.confidence,
            "metadata": plan.metadata
        }