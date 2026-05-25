"""
src/features/Conversation/ConversationModule.py
------------------------------------------------
Thread lifecycle, turn management, working memory, and warm persistence.
Delegates cold storage (embeddings, search, contradiction) to ConversationRAG.
"""

import json
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Any

from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.AppPaths import default_data_dir
from src.features.Conversation.types import (ThreadState, Turn, TurnMetadata, ConversationThread)


# ═══════════════════════════════════════════════════════════════════════════════
# Core types
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Conversation Module
# ═══════════════════════════════════════════════════════════════════════════════

class ConversationModule:
    """
    Manages conversation threads as hot memory.
    Delegates cold storage (embeddings, search, contradiction) to ConversationRAG.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Hot storage: in-memory active threads
    _threads: Dict[str, ConversationThread] = {}
    
    # Paths
    _conversations_dir: Path = Path(default_data_dir()) / "conversations"
    _index_path: Path = _conversations_dir / "index.json"
    
    # RAG submodule (injected)
    _rag = None
    
    # Auto-archive timer
    _auto_archive_minutes: int = 30
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, rag_submodule=None, auto_archive_minutes: int = 30) -> None:
        """Initialize module. Optionally inject ConversationRAG submodule."""
        cls._conversations_dir.mkdir(parents=True, exist_ok=True)
        cls._rag = rag_submodule
        cls._auto_archive_minutes = auto_archive_minutes
        cls._load_index()
        darwin.log("ConversationModule initialized", LogType.SUCCESS)
    
    # ── Thread lifecycle ─────────────────────────────────────────────────────
    
    @classmethod
    def create_thread(cls, title: Optional[str] = None) -> str:
        """Create new thread. Auto-generate title after first exchange."""
        thread_id = str(uuid.uuid4())[:8]
        thread = ConversationThread(
            thread_id=thread_id,
            title=title,
            created_at=datetime.now(),
            last_active=datetime.now(),
        )
        with cls._lock:
            cls._threads[thread_id] = thread
            cls._update_index(thread)
        
        darwin.log(f"Created thread: {thread_id}", LogType.INFO)
        return thread_id
    
    @classmethod
    def continue_thread(cls, thread_id: str) -> Optional[ConversationThread]:
        """Load thread from JSONL into hot memory."""
        # Check hot memory first
        if thread_id in cls._threads:
            cls._threads[thread_id].last_active = datetime.now()
            return cls._threads[thread_id]
        
        # Load from warm storage
        thread = cls._load_thread_from_jsonl(thread_id)
        if thread:
            with cls._lock:
                cls._threads[thread_id] = thread
            darwin.log(f"Resumed thread: {thread_id}", LogType.INFO)
            return thread
        
        darwin.log(f"Thread not found: {thread_id}", LogType.ERROR)
        return None
    
    @classmethod
    def list_threads(cls, limit: int = 10) -> List[Dict[str, Any]]:
        """List threads sorted by last_active descending."""
        cls._load_index()
        
        # Merge hot + index
        all_threads = {}
        for tid, thread in cls._threads.items():
            all_threads[tid] = {
                "thread_id": tid,
                "title": thread.title or "(Untitled)",
                "turn_count": len(thread.turns),
                "last_active": thread.last_active.isoformat(),
                "state": thread.state.value,
                "is_hot": True,
            }
        
        # Add archived threads from index
        index = cls._read_index()
        for tid, meta in index.items():
            if tid not in all_threads:
                all_threads[tid] = {
                    "thread_id": tid,
                    "title": meta.get("title", "(Archived)"),
                    "turn_count": meta.get("turn_count", 0),
                    "last_active": meta.get("last_active", ""),
                    "state": meta.get("state", "archived"),
                    "is_hot": False,
                }
        
        # Sort by last_active
        sorted_threads = sorted(
            all_threads.values(),
            key=lambda x: x["last_active"],
            reverse=True,
        )
        return sorted_threads[:limit]
    
    @classmethod
    def archive_thread(cls, thread_id: str) -> bool:
        """Archive thread to ConversationRAG. Keep JSONL."""
        thread = cls._threads.get(thread_id)
        if not thread:
            thread = cls._load_thread_from_jsonl(thread_id)
        
        if not thread:
            darwin.log(f"Cannot archive: thread {thread_id} not found", LogType.ERROR)
            return False
        
        thread.state = ThreadState.ARCHIVED
        
        # Delegate to RAG submodule
        if cls._rag:
            try:
                npz_path = cls._rag.index_thread(thread_id, thread.turns)
                thread.npz_path = npz_path
                thread.chunk_count = cls._rag.get_chunk_count(thread_id)
                darwin.log(f"Thread {thread_id} indexed to {npz_path}", LogType.SUCCESS)
            except Exception as e:
                darwin.log(f"RAG indexing failed for {thread_id}: {e}", LogType.ERROR)
                thread.state = ThreadState.ERROR
        
        # Save final state
        cls._save_thread_to_jsonl(thread)
        
        with cls._lock:
            if thread_id in cls._threads:
                del cls._threads[thread_id]  # Remove from hot memory
        
        cls._update_index(thread)
        return True
    
    @classmethod
    def delete_thread(cls, thread_id: str) -> bool:
        """Permanently delete thread and all artifacts."""
        # Remove from hot memory
        with cls._lock:
            cls._threads.pop(thread_id, None)
        
        # Delete files
        jsonl_path = cls._jsonl_path(thread_id)
        if jsonl_path.exists():
            jsonl_path.unlink()
        
        # Delete NPZ if exists
        npz_dir = cls._conversations_dir / thread_id
        if npz_dir.exists():
            import shutil
            shutil.rmtree(npz_dir)
        
        # Update index
        index = cls._read_index()
        index.pop(thread_id, None)
        cls._write_index(index)
        
        darwin.log(f"Deleted thread: {thread_id}", LogType.WARNING)
        return True
    
    # ── Turn management ──────────────────────────────────────────────────────
    
    @classmethod
    def add_turn(
        cls,
        thread_id: str,
        role: str,
        content: str,
        metadata: Optional[TurnMetadata] = None,
    ) -> Turn:
        """Add turn to thread. Auto-enrich metadata."""
        thread = cls._ensure_thread_loaded(thread_id)
        
        if metadata is None:
            metadata = TurnMetadata(thread_id=thread_id)
        
        turn = Turn(
            turn_id=str(uuid.uuid4())[:12],
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata,
        )
        
        thread.turns.append(turn)
        thread.last_active = datetime.now()
        
        # Auto-title after first exchange
        if len(thread.turns) == 2 and not thread.title:
            thread.title = cls._generate_title(thread)
        
        # Auto-enrich (background, non-blocking)
        cls._enrich_turn_async(turn)
        
        # Flush to JSONL periodically
        if len(thread.turns) % 5 == 0:
            cls._save_thread_to_jsonl(thread)
        
        return turn
    
    @classmethod
    def get_context_window(
        cls,
        thread_id: str,
        max_tokens: int = 4000,
        tier: str = "normal",
    ) -> List[Dict[str, str]]:
        """
        Get recent turns formatted for LLM prompt injection.
        Budget: 60% recent turns, 40% rolling summary.
        Tier-aware: fast=summaries, smart=full text.
        """
        thread = cls._ensure_thread_loaded(thread_id)
        if not thread:
            return []
        
        messages = []
        recent_budget = int(max_tokens * 0.6)
        summary_budget = int(max_tokens * 0.4)
        
        # Recent turns (newest first, until budget exhausted)
        recent_turns = []
        token_count = 0
        
        for turn in reversed(thread.turns):
            turn_text = cls._format_turn(turn, tier)
            turn_tokens = len(turn_text) // 4  # Rough estimate
            
            if token_count + turn_tokens > recent_budget:
                break
            
            recent_turns.append({
                "role": turn.role if turn.role != "tool" else "assistant",
                "content": turn_text,
            })
            token_count += turn_tokens
        
        # Add in chronological order
        messages.extend(reversed(recent_turns))
        
        # Rolling summary if available and budget allows
        if thread.summary and token_count < max_tokens - summary_budget:
            messages.insert(0, {
                "role": "system",
                "content": f"[Conversation summary so far]: {thread.summary[:summary_budget * 4]}",
            })
        
        return messages
    
    @classmethod
    def update_summary(cls, thread_id: str) -> Optional[str]:
        """Generate rolling summary (call after every 10 turns)."""
        thread = cls._ensure_thread_loaded(thread_id)
        if not thread or len(thread.turns) < 10:
            return None
        
        # Delegate to smart tier via injected client or caller
        # This is a hook — actual LLM call happens in AgentBuilder
        thread.summary = f"({len(thread.turns)} turns discussed: {', '.join(thread.turns[-1].metadata.topics[:3])})"
        cls._save_thread_to_jsonl(thread)
        return thread.summary
    
    # ── Working memory helpers ───────────────────────────────────────────────
    
    @classmethod
    def get_last_n_turns(cls, thread_id: str, n: int = 5) -> List[Turn]:
        thread = cls._ensure_thread_loaded(thread_id)
        return thread.turns[-n:] if thread else []
    
    @classmethod
    def get_turn_by_id(cls, thread_id: str, turn_id: str) -> Optional[Turn]:
        thread = cls._ensure_thread_loaded(thread_id)
        if not thread:
            return None
        for turn in thread.turns:
            if turn.turn_id == turn_id:
                return turn
        return None
    
    @classmethod
    def get_thread_state(cls, thread_id: str) -> Optional[ThreadState]:
        thread = cls._threads.get(thread_id)
        if thread:
            return thread.state
        # Check index
        index = cls._read_index()
        if thread_id in index:
            return ThreadState(index[thread_id].get("state", "archived"))
        return None
    
    # ── Internal helpers ─────────────────────────────────────────────────────
    
    @classmethod
    def _ensure_thread_loaded(cls, thread_id: str) -> Optional[ConversationThread]:
        if thread_id not in cls._threads:
            return cls.continue_thread(thread_id)
        return cls._threads[thread_id]
    
    @classmethod
    def _generate_title(cls, thread: ConversationThread) -> str:
        """Auto-generate title from first exchange (fast-tier hook)."""
        if len(thread.turns) >= 2:
            first_user = next((t.content for t in thread.turns if t.role == "user"), "")
            # Truncate and clean
            title = first_user[:40].replace("\n", " ").strip()
            if title:
                return title + "..." if len(first_user) > 40 else title
        return f"Conversation {thread.thread_id[:6]}"
    
    @classmethod
    def _enrich_turn_async(cls, turn: Turn) -> None:
        """Background enrichment: sentiment, topics, turn_type."""
        # Placeholder — actual implementation uses fast-tier LLM or rules
        # These are non-blocking; stale values are acceptable
        if turn.metadata.sentiment is None:
            turn.metadata.sentiment = 0.0  # Neutral default
        if not turn.metadata.topics:
            turn.metadata.topics = []
    
    @classmethod
    def _format_turn(cls, turn: Turn, tier: str) -> str:
        """Format turn for prompt injection. Tier-aware compression."""
        if tier == "fast" and len(turn.content) > 200:
            return turn.content[:200] + "..."
        return turn.content
    
    # ── Persistence ──────────────────────────────────────────────────────────
    
    @classmethod
    def _jsonl_path(cls, thread_id: str) -> Path:
        return cls._conversations_dir / f"{thread_id}.jsonl"
    
    @classmethod
    def _save_thread_to_jsonl(cls, thread: ConversationThread) -> None:
        path = cls._jsonl_path(thread.thread_id)
        with open(path, "w", encoding="utf-8") as f:
            for turn in thread.turns:
                f.write(json.dumps(turn.to_dict()) + "\n")
    
    @classmethod
    def _load_thread_from_jsonl(cls, thread_id: str) -> Optional[ConversationThread]:
        path = cls._jsonl_path(thread_id)
        if not path.exists():
            return None
        
        turns = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    turns.append(Turn.from_dict(json.loads(line)))
        
        if not turns:
            return None
        
        return ConversationThread(
            thread_id=thread_id,
            turns=turns,
            created_at=turns[0].timestamp,
            last_active=turns[-1].timestamp,
        )
    
    @classmethod
    def _read_index(cls) -> Dict[str, dict]:
        if not cls._index_path.exists():
            return {}
        try:
            with open(cls._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return {}
    
    @classmethod
    def _write_index(cls, index: Dict[str, dict]) -> None:
        with open(cls._index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, default=str)
    
    @classmethod
    def _update_index(cls, thread: ConversationThread) -> None:
        index = cls._read_index()
        index[thread.thread_id] = {
            "title": thread.title,
            "turn_count": len(thread.turns),
            "last_active": thread.last_active.isoformat(),
            "state": thread.state.value,
            "npz_path": str(thread.npz_path) if thread.npz_path else None,
            "chunk_count": thread.chunk_count,
        }
        cls._write_index(index)
    
    @classmethod
    def _load_index(cls) -> None:
        """Load hot threads from index on startup."""
        index = cls._read_index()
        for thread_id, meta in index.items():
            if meta.get("state") == "active":
                # Only load recent active threads
                last_active = datetime.fromisoformat(meta["last_active"])
                if datetime.now() - last_active < timedelta(minutes=cls._auto_archive_minutes):
                    cls.continue_thread(thread_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════════

def get_conversation_module() -> ConversationModule:
    return ConversationModule()


def create_conversation(title: Optional[str] = None) -> str:
    return ConversationModule.create_thread(title)


def continue_conversation(thread_id: str) -> Optional[ConversationThread]:
    return ConversationModule.continue_thread(thread_id)