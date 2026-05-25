from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np


class ThreadState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    ERROR = "error"


@dataclass
class TurnMetadata:
    tier: str = "normal"
    model: str = ""
    skills_used: List[str] = field(default_factory=list)
    tools_invoked: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    thread_id: str = ""
    parent_turn_id: Optional[str] = None
    user_satisfaction: Optional[float] = None
    sentiment: Optional[float] = None
    topics: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TurnMetadata":
        return cls(**d)


@dataclass
class Turn:
    turn_id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime
    metadata: TurnMetadata

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(
            turn_id=d["turn_id"],
            role=d["role"],
            content=d["content"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            metadata=TurnMetadata.from_dict(d["metadata"]),
        )


@dataclass
class ConversationThread:
    thread_id: str
    title: Optional[str] = None
    turns: List[Turn] = field(default_factory=list)
    state: ThreadState = ThreadState.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    summary: Optional[str] = None
    chunk_count: int = 0
    npz_path: Optional[Path] = None
    last_indexed_turn_count: int = 0          # NEW: how many turns were already indexed

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "title": self.title,
            "turns": [t.to_dict() for t in self.turns],
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "summary": self.summary,
            "chunk_count": self.chunk_count,
            "npz_path": str(self.npz_path) if self.npz_path else None,
            "last_indexed_turn_count": self.last_indexed_turn_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationThread":
        return cls(
            thread_id=d["thread_id"],
            title=d.get("title"),
            turns=[Turn.from_dict(t) for t in d.get("turns", [])],
            state=ThreadState(d.get("state", "active")),
            created_at=datetime.fromisoformat(d["created_at"]),
            last_active=datetime.fromisoformat(d["last_active"]),
            summary=d.get("summary"),
            chunk_count=d.get("chunk_count", 0),
            npz_path=Path(d["npz_path"]) if d.get("npz_path") else None,
            last_indexed_turn_count=d.get("last_indexed_turn_count", 0),
        )


@dataclass
class ConvoChunk:
    text: str
    clean: str
    index: int
    source: str              # thread_id
    turn_range: Tuple[int, int]
    role: Literal["user", "assistant", "system", "tool", "mixed"]
    thread_id: str
    timestamp: datetime
    sentiment: float = 0.0
    topics: List[str] = field(default_factory=list)
    turn_ids: List[str] = field(default_factory=list)
    is_preference_statement: bool = False
    is_factual_claim: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "clean": self.clean,
            "index": self.index,
            "source": self.source,
            "turn_range": list(self.turn_range),
            "role": self.role,
            "thread_id": self.thread_id,
            "timestamp": self.timestamp.isoformat(),
            "sentiment": self.sentiment,
            "topics": self.topics,
            "turn_ids": self.turn_ids,
            "is_preference_statement": self.is_preference_statement,
            "is_factual_claim": self.is_factual_claim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConvoChunk":
        return cls(
            text=d["text"],
            clean=d["clean"],
            index=d["index"],
            source=d["source"],
            turn_range=tuple(d["turn_range"]),
            role=d["role"],
            thread_id=d["thread_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            sentiment=d.get("sentiment", 0.0),
            topics=d.get("topics", []),
            turn_ids=d.get("turn_ids", []),
            is_preference_statement=d.get("is_preference_statement", False),
            is_factual_claim=d.get("is_factual_claim", False),
        )


@dataclass
class MemoryDocument:
    doc_id: str
    text: str
    clean: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[ConvoChunk] = None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "clean": self.clean,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }


@dataclass
class Contradiction:
    claim_a: MemoryDocument
    claim_b: MemoryDocument
    contradiction_type: Literal[
        "direct_negation", "preference_reversal", "fact_conflict",
        "goal_change", "temporal_inconsistency"
    ]
    severity: float
    time_delta: timedelta
    resolution: Optional[str] = None
    detected_by: Literal["rule", "llm_judge"] = "rule"

    def to_dict(self) -> dict:
        return {
            "claim_a": self.claim_a.to_dict(),
            "claim_b": self.claim_b.to_dict(),
            "contradiction_type": self.contradiction_type,
            "severity": self.severity,
            "time_delta_seconds": self.time_delta.total_seconds(),
            "resolution": self.resolution,
            "detected_by": self.detected_by,
        }


@dataclass
class SearchFilters:
    date_range: Optional[Tuple[datetime, datetime]] = None
    roles: Optional[List[str]] = None
    thread_ids: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    sentiment_range: Optional[Tuple[float, float]] = None
    min_relevance: float = 0.7


@dataclass
class SearchResult:
    doc: MemoryDocument
    score: float
    context_snippet: str = ""

    def to_dict(self) -> dict:
        return {
            "doc": self.doc.to_dict(),
            "score": self.score,
            "context_snippet": self.context_snippet,
        }