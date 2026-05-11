# src/features/Query/ConversationDB.py
"""
SQLite storage for multi‑turn conversations.
"""
import sqlite3
import json
import time
from pathlib import Path
from typing import List, Optional
from src.features.Query.QueryTypes import RawQuery


class ConversationDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT REFERENCES conversations(id),
                    user_query_json TEXT,
                    classification TEXT,
                    retrieval_policy_json TEXT,
                    retrieved_results_json TEXT,
                    resolved_contexts_json TEXT,
                    artifact_type TEXT,
                    artifact_json TEXT,
                    response_text TEXT,
                    confidence REAL,
                    timestamp REAL
                )
            """)

    def create_conversation(self, title: str = "") -> str:
        conv_id = str(time.time())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
                (conv_id, title, time.time())
            )
        return conv_id

    def add_turn(
        self,
        conversation_id: str,
        raw_query: RawQuery,
        classification: str,
        retrieval_policy_json: str,
        retrieved_results_json: str,
        resolved_contexts_json: str,
        artifact_type: str,
        artifact_json: str,
        response_text: str,
        confidence: Optional[float] = None,
    ):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO turns 
                   (conversation_id, user_query_json, classification, retrieval_policy_json,
                    retrieved_results_json, resolved_contexts_json, artifact_type,
                    artifact_json, response_text, confidence, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    conversation_id,
                    json.dumps(raw_query.__dict__),
                    classification,
                    retrieval_policy_json,
                    retrieved_results_json,
                    resolved_contexts_json,
                    artifact_type,
                    artifact_json,
                    response_text,
                    confidence,
                    time.time(),
                )
            )

    def get_last_turns(self, conversation_id: str, limit: int = 3) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM turns WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit)
            ).fetchall()
        return [dict(zip([d[0] for d in rows[0].__class__.__dict__], row)) for row in reversed(rows)]