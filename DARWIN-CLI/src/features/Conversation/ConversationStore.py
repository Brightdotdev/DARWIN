"""
ConversationStore.py
SQLite-backed conversation persistence with blob attachments, lifecycle management,
and turn indexing. No DarwinStore dependency.
"""

import sqlite3
import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import threading


# Fallback if AppPaths not available
try:
    from src.lib.AppPaths import conversation_db_path
except ImportError:
    def conversation_db_path() -> Path:
        return Path("darwin_conversations.db")


@dataclass
class TurnRecord:
    turn_id: str
    conv_id: str
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: str = ""
    extraction_method: str = "regex"
    preference_signals: Optional[Dict[str, Any]] = None
    context_xml: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # Holds TurnMetadata, extracted_claims, intent, etc.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TurnRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BlobRecord:
    blob_id: str = ""
    turn_id: str = ""
    conv_id: str = ""
    blob_type: str = "text"  # "text" | "code" | "image" | "file" | "link"
    content: Optional[str] = None
    file_path: Optional[str] = None
    embedding_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationMeta:
    conv_id: str
    title: str
    created_at: str
    updated_at: str
    turn_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
    schema_version: str = "1.0"


class ConversationStore:
    """
    SQLite-backed store. WAL mode. Thread-safe.
    """

    SCHEMA = """
    PRAGMA journal_mode=WAL;
    PRAGMA foreign_keys=ON;

    CREATE TABLE IF NOT EXISTS conversations (
        conv_id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT 'Untitled',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata TEXT,
        schema_version TEXT DEFAULT '1.0'
    );

    CREATE TABLE IF NOT EXISTS turns (
        turn_id TEXT PRIMARY KEY,
        conv_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        extraction_method TEXT DEFAULT 'regex',
        preference_signals TEXT,
        context_xml TEXT,
        metadata TEXT,
        FOREIGN KEY (conv_id) REFERENCES conversations(conv_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conv_id, timestamp);

    CREATE TABLE IF NOT EXISTS blobs (
        blob_id TEXT PRIMARY KEY,
        turn_id TEXT NOT NULL,
        conv_id TEXT NOT NULL,
        blob_type TEXT NOT NULL,
        content TEXT,
        file_path TEXT,
        embedding_id TEXT,
        metadata TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (turn_id) REFERENCES turns(turn_id) ON DELETE CASCADE,
        FOREIGN KEY (conv_id) REFERENCES conversations(conv_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_blobs_conv ON blobs(conv_id, blob_type);
    CREATE INDEX IF NOT EXISTS idx_blobs_turn ON blobs(turn_id);
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else conversation_db_path()
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        with self._write_lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.executescript(self.SCHEMA)
            conn.commit()
            conn.close()

    @contextmanager
    def _transaction(self):
        with self._write_lock:
            conn = self._connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ── Conversation Lifecycle ──────────────────────────────

    def create_conversation(self, title: str = "Untitled",
                          metadata: Optional[Dict] = None) -> str:
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat()
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO conversations (conv_id, title, created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (conv_id, title, now, now, json.dumps(metadata) if metadata else None)
            )
        return conv_id

    def list_conversations(self, limit: int = 100, offset: int = 0) -> List[ConversationMeta]:
        conn = self._connection()
        rows = conn.execute(
            """SELECT c.*, COUNT(t.turn_id) as turn_count
               FROM conversations c
               LEFT JOIN turns t ON c.conv_id = t.conv_id
               GROUP BY c.conv_id
               ORDER BY c.updated_at DESC
               LIMIT ? OFFSET ?""",
            (limit, offset)
        ).fetchall()
        return [
            ConversationMeta(
                conv_id=r["conv_id"], title=r["title"],
                created_at=r["created_at"], updated_at=r["updated_at"],
                turn_count=r["turn_count"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else None,
                schema_version=r["schema_version"]
            ) for r in rows
        ]

    def get_conversation(self, conv_id: str) -> Optional[ConversationMeta]:
        conn = self._connection()
        row = conn.execute(
            """SELECT c.*, COUNT(t.turn_id) as turn_count
               FROM conversations c
               LEFT JOIN turns t ON c.conv_id = t.conv_id
               WHERE c.conv_id = ? GROUP BY c.conv_id""", (conv_id,)
        ).fetchone()
        if not row:
            return None
        return ConversationMeta(
            conv_id=row["conv_id"], title=row["title"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            turn_count=row["turn_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            schema_version=row["schema_version"]
        )

    def update_title(self, conv_id: str, title: str):
        now = datetime.utcnow().isoformat()
        with self._transaction() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE conv_id = ?",
                (title, now, conv_id)
            )

    def delete_conversation(self, conv_id: str) -> bool:
        with self._transaction() as conn:
            cur = conn.execute("DELETE FROM conversations WHERE conv_id = ?", (conv_id,))
            return cur.rowcount > 0

    def export_conversation(self, conv_id: str, format: str = "json") -> Dict[str, Any]:
        meta = self.get_conversation(conv_id)
        if not meta:
            raise ValueError(f"Conversation {conv_id} not found")
        turns = self.get_turns(conv_id)
        blobs = self.get_blobs(conv_id)
        payload = {
            "format": format,
            "exported_at": datetime.utcnow().isoformat(),
            "conversation": meta.__dict__,
            "turns": [t.to_dict() for t in turns],
            "blobs": [b.to_dict() for b in blobs]
        }
        if format == "markdown":
            md_lines = [f"# {meta.title}\n", f"*Exported: {payload['exported_at']}*\n\n"]
            for t in turns:
                md_lines.append(f"## {t.role.capitalize()} ({t.timestamp})\n\n{t.content}\n\n")
                for b in (x for x in blobs if x.turn_id == t.turn_id):
                    if b.blob_type == "code":
                        md_lines.append(f"```\n{b.content}\n```\n\n")
                    elif b.blob_type == "link":
                        md_lines.append(f"[Link]({b.content})\n\n")
            payload["markdown"] = "".join(md_lines)
        return payload

    def import_conversation(self, payload: Dict[str, Any]) -> str:
        conv_data = payload["conversation"]
        new_conv_id = self.create_conversation(
            title=conv_data.get("title", "Imported"),
            metadata=conv_data.get("metadata")
        )
        for t in payload.get("turns", []):
            turn = TurnRecord.from_dict(t)
            turn.conv_id = new_conv_id
            turn.turn_id = f"turn_{uuid.uuid4().hex[:12]}"
            self.append_turn(turn)
        for b in payload.get("blobs", []):
            blob = BlobRecord(**b)
            blob.conv_id = new_conv_id
            blob.blob_id = f"blob_{uuid.uuid4().hex[:12]}"
            self.attach_blob(blob)
        return new_conv_id

    # ── Turn Management ─────────────────────────────────────

    def append_turn(self, turn: TurnRecord) -> TurnRecord:
        now = datetime.utcnow().isoformat()
        turn.timestamp = now
        if not turn.turn_id:
            turn.turn_id = f"turn_{uuid.uuid4().hex[:12]}"
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO turns
                   (turn_id, conv_id, role, content, timestamp, extraction_method,
                    preference_signals, context_xml, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (turn.turn_id, turn.conv_id, turn.role, turn.content, turn.timestamp,
                 turn.extraction_method,
                 json.dumps(turn.preference_signals) if turn.preference_signals else None,
                 turn.context_xml,
                 json.dumps(turn.metadata) if turn.metadata else None)
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conv_id = ?",
                (now, turn.conv_id)
            )
        return turn

    def get_turns(self, conv_id: str, limit: int = 1000) -> List[TurnRecord]:
        conn = self._connection()
        rows = conn.execute(
            "SELECT * FROM turns WHERE conv_id = ? ORDER BY timestamp ASC LIMIT ?",
            (conv_id, limit)
        ).fetchall()
        return [
            TurnRecord(
                turn_id=r["turn_id"], conv_id=r["conv_id"], role=r["role"],
                content=r["content"], timestamp=r["timestamp"],
                extraction_method=r["extraction_method"],
                preference_signals=json.loads(r["preference_signals"]) if r["preference_signals"] else None,
                context_xml=r["context_xml"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else None
            ) for r in rows
        ]

    def get_turn(self, turn_id: str) -> Optional[TurnRecord]:
        conn = self._connection()
        r = conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        if not r:
            return None
        return TurnRecord(
            turn_id=r["turn_id"], conv_id=r["conv_id"], role=r["role"],
            content=r["content"], timestamp=r["timestamp"],
            extraction_method=r["extraction_method"],
            preference_signals=json.loads(r["preference_signals"]) if r["preference_signals"] else None,
            context_xml=r["context_xml"],
            metadata=json.loads(r["metadata"]) if r["metadata"] else None
        )

    def delete_turn(self, turn_id: str) -> bool:
        with self._transaction() as conn:
            cur = conn.execute("DELETE FROM turns WHERE turn_id = ?", (turn_id,))
            return cur.rowcount > 0

    # ── Blob Management ───────────────────────────────────

    def attach_blob(self, blob: BlobRecord) -> BlobRecord:
        now = datetime.utcnow().isoformat()
        blob.timestamp = now
        if not blob.blob_id:
            blob.blob_id = f"blob_{uuid.uuid4().hex[:12]}"
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO blobs
                   (blob_id, turn_id, conv_id, blob_type, content, file_path,
                    embedding_id, metadata, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (blob.blob_id, blob.turn_id, blob.conv_id, blob.blob_type,
                 blob.content, blob.file_path, blob.embedding_id,
                 json.dumps(blob.metadata) if blob.metadata else None, blob.timestamp)
            )
        return blob

    def get_blobs(self, conv_id: str, blob_type: Optional[str] = None) -> List[BlobRecord]:
        conn = self._connection()
        if blob_type:
            rows = conn.execute(
                "SELECT * FROM blobs WHERE conv_id = ? AND blob_type = ? ORDER BY timestamp",
                (conv_id, blob_type)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM blobs WHERE conv_id = ? ORDER BY timestamp", (conv_id,)
            ).fetchall()
        return [
            BlobRecord(
                blob_id=r["blob_id"], turn_id=r["turn_id"], conv_id=r["conv_id"],
                blob_type=r["blob_type"], content=r["content"], file_path=r["file_path"],
                embedding_id=r["embedding_id"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else None,
                timestamp=r["timestamp"]
            ) for r in rows
        ]

    def get_blobs_for_turn(self, turn_id: str) -> List[BlobRecord]:
        conn = self._connection()
        rows = conn.execute(
            "SELECT * FROM blobs WHERE turn_id = ? ORDER BY timestamp", (turn_id,)
        ).fetchall()
        return [
            BlobRecord(
                blob_id=r["blob_id"], turn_id=r["turn_id"], conv_id=r["conv_id"],
                blob_type=r["blob_type"], content=r["content"], file_path=r["file_path"],
                embedding_id=r["embedding_id"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else None,
                timestamp=r["timestamp"]
            ) for r in rows
        ]

    def delete_blob(self, blob_id: str) -> bool:
        with self._transaction() as conn:
            cur = conn.execute("DELETE FROM blobs WHERE blob_id = ?", (blob_id,))
            return cur.rowcount > 0

    def update_blob_embedding(self, blob_id: str, embedding_id: str):
        with self._transaction() as conn:
            conn.execute(
                "UPDATE blobs SET embedding_id = ? WHERE blob_id = ?",
                (embedding_id, blob_id)
            )

    # ── Utility ─────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        conn = self._connection()
        return {
            "conversations": conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
            "turns": conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0],
            "blobs": conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0],
            "db_path": str(self.db_path)
        }

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None