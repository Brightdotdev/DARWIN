"""
ConversationManager.py
Unified conversation manager. Wires together:
  - ConversationStore (SQLite persistence)
  - ConversationCache (tool/query result caching)
  - QueryOrchestrator (first-bounce extraction + routing)
  - MemoryRetriever (NPZ embedding index)
  - TurnExtractor (claim/intent/preference extraction)
  - ContextAssembler (prompt assembly)

No DarwinStore dependency.
"""

import uuid
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# ── Local / project imports with fallbacks ─────────────────
from src.features.Conversation.ConversationConfig import ConversationConfig
from src.features.Conversation.ConversationStore import ConversationStore, TurnRecord, BlobRecord
from src.features.Conversation.ConversationCache import ConversationCache

from src.features.Query.QueryOrchestrator import QueryOrchestrator, DarwinLogger
from src.features.Conversation.MemoryRetriever import MemoryRetriever
from src.features.Conversation.TurnExtractor import TurnExtractor
from src.features.Conversation.ContextAssembler import ContextAssembler

@dataclass
class ConversationAudit:
    conversation_id: str
    total_turns: int
    indexed_memories: int
    estimated_context_tokens: int
    budget_status: str
    last_intent: Optional[str] = None
    last_routing_target: Optional[str] = None
    cache_hit_rate: float = 0.0


class ConversationManager:
    """
    Single entry point for conversation lifecycle.

    Typical flow:
      1. manager.start_conversation(title) -> conv_id
      2. plan, messages, user_turn_id = manager.process_user_turn(msg, tools, prompt)
      3. # ... downstream LLM call using messages ...
      4. manager.save_assistant_turn(response_payload, plan, turn_metadata)
      5. manager.flush()
    """

    def __init__(
        self,
        config: Optional[ConversationConfig] = None,
        db_path: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        self.config = config or ConversationConfig()
        self.store = ConversationStore(db_path=db_path)
        self.cache = ConversationCache(db_path=cache_path)
        self.logger = DarwinLogger()
        self.orchestrator = QueryOrchestrator(
            self.store, self.cache, self.logger,
            max_history_for_pushback=5
        )
        self.memory_retriever = MemoryRetriever(self.config)
        self.turn_extractor = TurnExtractor(self.config)
        self.context_assembler = ContextAssembler(self.config)

        self.conversation_id: Optional[str] = None
        self._turns_cache: List[Dict[str, Any]] = []   # in-memory for fast context assembly
        self._user_turn_count = 0

    # ── Conversation Lifecycle ──────────────────────────────

    def start_conversation(self, title: str = "Untitled",
                           metadata: Optional[Dict] = None) -> str:
        self.conversation_id = self.store.create_conversation(title, metadata)
        self._turns_cache = []
        self._user_turn_count = 0
        self.logger.log("conversation.started", {"title": title}, self.conversation_id)
        return self.conversation_id

    def load_conversation(self, conv_id: str) -> str:
        meta = self.store.get_conversation(conv_id)
        if not meta:
            raise ValueError(f"Conversation {conv_id} not found")
        self.conversation_id = conv_id
        records = self.store.get_turns(conv_id)
        self._turns_cache = [self._record_to_dict(r) for r in records]
        self._user_turn_count = sum(1 for t in self._turns_cache if t.get("role") == "user")
        self.logger.log("conversation.loaded", {"turns": len(self._turns_cache)}, conv_id)
        return conv_id

    def list_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [m.__dict__ for m in self.store.list_conversations(limit)]

    def delete_conversation(self, conv_id: str) -> bool:
        return self.store.delete_conversation(conv_id)

    # ── Core Turn Flow ──────────────────────────────────────

    def process_user_turn(
        self,
        user_message: str,
        available_tools: List[str],
        system_prompt: str,
    ) -> Tuple[Any, List[Dict[str, Any]], str]:
        """
        First-bounce processing for a user message.

        Returns:
            (QueryPlan, assembled_messages, user_turn_id)
        """
        if not self.conversation_id:
            raise RuntimeError("No active conversation. Call start_conversation() or load_conversation() first.")

        # 1. Orchestrator: extract blobs, detect pushback, rewrite, route
        plan = self.orchestrator.process(self.conversation_id, user_message)

        # 2. Prepare turn data for extraction
        turn_data = {
            "user_message": plan.rewritten_query or user_message,
            "assistant_plan": "",
            "final_response": "",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # 3. Extract claims, preferences, intent, embedding
        extracted = self.turn_extractor.extract(turn_data)

        # 4. Build rich metadata dict (#5 fix — stores EVERYTHING)
        user_metadata = {
            "extracted_claims": extracted.get("claims", []),
            "intent": extracted.get("intent", "general"),
            "confidence": extracted.get("confidence", 0.0),
            "extraction_method": extracted.get("extraction_method", "regex"),
            "original_message": user_message,
            "rewritten_query": plan.rewritten_query,
            "routing_target": plan.routing_target,
            "pushback_detected": plan.pushback_detected,
            "extracted_links": plan.extracted_links,
            "blob_count": len(plan.extracted_blobs),
            "blob_types": list(set(b.blob_type for b in plan.extracted_blobs)),
        }

        # 5. Persist user turn to SQLite
        user_turn = TurnRecord(
            turn_id="",
            conv_id=self.conversation_id,
            role="user",
            content=plan.rewritten_query or user_message,
            extraction_method=extracted.get("extraction_method", "regex"),
            preference_signals=extracted.get("preferences", {}),
            metadata=user_metadata,
        )
        saved_user = self.store.append_turn(user_turn)

        # 6. Persist extracted blobs with correct turn_id
        for blob in plan.extracted_blobs:
            blob_record = BlobRecord(
                blob_id="",
                turn_id=saved_user.turn_id,
                conv_id=self.conversation_id,
                blob_type=blob.blob_type,
                content=blob.content,
                metadata={"language": blob.language, "span": blob.span},
            )
            self.store.attach_blob(blob_record)

        # 7. Index to NPZ memory (conversation-as-RAG)
        memory_dict = {
            "turn_id": saved_user.turn_id,
            "conversation_id": self.conversation_id,
            "timestamp": saved_user.timestamp,
            "extracted_claims": extracted.get("claims", []),
            "preference_signals": extracted.get("preferences", {}),
            "intent": extracted.get("intent"),
            "embedding": extracted.get("embedding"),
        }
        self.memory_retriever.index_turn(memory_dict)

        # 8. Update in-memory cache for fast context assembly
        self._turns_cache.append(self._record_to_dict(saved_user))
        self._user_turn_count += 1

        # 9. Retrieve relevant memories
        memories = self.memory_retriever.retrieve(
            query=plan.rewritten_query,
            conversation_id=self.conversation_id,
            top_k=self.config.memory_top_k,
            cross_conversation=self.config.enable_cross_conversation_memory,
        )

        # 10. Assemble context for downstream LLM
        messages = self.context_assembler.assemble(
            system_prompt=system_prompt,
            memories=memories,
            turns=self._turns_cache,
            user_message=plan.rewritten_query,
            available_tools=available_tools,
        )

        self.logger.log_stage(
            "user_turn_complete", "done",
            {"turn_id": saved_user.turn_id, "routing": plan.routing_target},
            self.conversation_id, plan.plan_id,
        )

        return plan, messages, saved_user.turn_id

    def save_assistant_turn(
        self,
        response_payload: Any,
        plan: Any,
        turn_metadata: Optional[Dict[str, Any]] = None,
        index_to_memory: bool = False,
    ) -> TurnRecord:
        """
        Persist assistant / system / tool response.

        Args:
            response_payload: Either a string or a dict with ResponseType fields.
                              If dict, "content" key is extracted for the main text.
            plan: The QueryPlan from process_user_turn.
            turn_metadata: Serialized TurnMetadata from ResponseTypes (tier, model,
                           token_usage, latency_ms, skills_used, tools_invoked, etc.)
            index_to_memory: Rarely True. Assistant responses usually not indexed.
        """
        if not self.conversation_id:
            raise RuntimeError("No active conversation.")

        # Normalize payload
        if isinstance(response_payload, dict):
            content = response_payload.get("content", "")
            response_type = response_payload.get("type", "text")
            blocks = response_payload.get("blocks", [])
            citations = response_payload.get("citations", [])
        else:
            content = str(response_payload)
            response_type = "text"
            blocks = []
            citations = []

        # Merge all metadata (#5 fix — full TurnMetadata lives here)
        assistant_metadata = {
            "response_type": response_type,
            "plan_id": getattr(plan, "plan_id", None),
            "routing_target": getattr(plan, "routing_target", None),
            "has_blocks": len(blocks) > 0,
            "block_types": [b.get("type") for b in blocks] if blocks else [],
            "citation_count": len(citations),
            **(turn_metadata or {}),   # ← TurnMetadata.to_dict() goes here
        }

        assistant_turn = TurnRecord(
            turn_id="",
            conv_id=self.conversation_id,
            role="assistant",
            content=content,
            metadata=assistant_metadata,
        )
        saved = self.store.append_turn(assistant_turn)
        self._turns_cache.append(self._record_to_dict(saved))

        if index_to_memory:
            # Only if you want assistant responses searchable
            mem_dict = {
                "turn_id": saved.turn_id,
                "conversation_id": self.conversation_id,
                "timestamp": saved.timestamp,
                "extracted_claims": [content[:200]],
                "preference_signals": {},
                "intent": "assistant_response",
                "embedding": None,  # Would need separate embedding call
            }
            # self.memory_retriever.index_turn(mem_dict)

        self.logger.log_stage(
            "assistant_turn_saved", "done",
            {"turn_id": saved.turn_id, "content_length": len(content)},
            self.conversation_id, getattr(plan, "plan_id", None),
        )
        return saved

    def save_system_turn(self, content: str,
                         system_metadata: Optional[Dict[str, Any]] = None) -> TurnRecord:
        """Persist a system message (config changes, skill loads, etc.)."""
        if not self.conversation_id:
            raise RuntimeError("No active conversation.")

        turn = TurnRecord(
            turn_id="", conv_id=self.conversation_id, role="system",
            content=content, metadata=system_metadata,
        )
        saved = self.store.append_turn(turn)
        self._turns_cache.append(self._record_to_dict(saved))
        return saved

    def save_tool_turn(self, tool_name: str, tool_input: Dict[str, Any],
                       tool_output: Dict[str, Any]) -> TurnRecord:
        """Persist a tool invocation result."""
        if not self.conversation_id:
            raise RuntimeError("No active conversation.")

        content = f"[{tool_name}] Input: {json.dumps(tool_input)}\nOutput: {json.dumps(tool_output, default=str)}"
        turn = TurnRecord(
            turn_id="", conv_id=self.conversation_id, role="tool",
            content=content[:2000],  # cap length
            metadata={"tool_name": tool_name, "input": tool_input, "output": tool_output},
        )
        saved = self.store.append_turn(turn)
        self._turns_cache.append(self._record_to_dict(saved))
        return saved

    # ── Context & Audit ─────────────────────────────────────

    def get_turns(self) -> List[Dict[str, Any]]:
        """Return in-memory turn cache (fast, for context assembly)."""
        return self._turns_cache

    def get_audit(self) -> ConversationAudit:
        stats = self.context_assembler.get_stats()
        cache_stats = self.cache.get_stats()
        total_hits = sum(t.get("hits", 0) for t in cache_stats.get("top_tools", []))
        total_entries = cache_stats.get("total_entries", 1)

        last_intent = None
        last_routing = None
        if self._turns_cache:
            last_turn = self._turns_cache[-1]
            meta = last_turn.get("metadata", {})
            last_intent = meta.get("intent") if isinstance(meta, dict) else None
            last_routing = meta.get("routing_target") if isinstance(meta, dict) else None

        return ConversationAudit(
            conversation_id=self.conversation_id or "",
            total_turns=len(self._turns_cache),
            indexed_memories=self.memory_retriever.count_indexed(self.conversation_id or ""),
            estimated_context_tokens=stats.get("estimated_tokens", 0),
            budget_status="ok" if stats.get("under_budget") else "compressed",
            last_intent=last_intent,
            last_routing_target=last_routing,
            cache_hit_rate=round(total_hits / max(total_entries, 1), 2),
        )

    def export_conversation(self, format: str = "json") -> Dict[str, Any]:
        if not self.conversation_id:
            raise RuntimeError("No active conversation.")
        return self.store.export_conversation(self.conversation_id, format)

    # ── Internal ────────────────────────────────────────────

    @staticmethod
    def _record_to_dict(record: TurnRecord) -> Dict[str, Any]:
        """
        Convert TurnRecord -> dict format expected by ContextAssembler.
        """
        meta = record.metadata or {}
        return {
            "turn_id": record.turn_id,
            "conversation_id": record.conv_id,
            "timestamp": record.timestamp,
            "role": record.role,
            "content": record.content,
            # ContextAssembler expects these keys:
            "user_message": record.content if record.role == "user" else "",
            "extracted_claims": meta.get("extracted_claims", []),
            "preference_signals": record.preference_signals or {},
            "intent": meta.get("intent", "general"),
            "confidence": meta.get("confidence", 0.0),
            "extraction_method": record.extraction_method,
            "metadata": meta,
        }

    def flush(self) -> None:
        self.memory_retriever.flush()
        self.store.close()
        self.cache.close()

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass