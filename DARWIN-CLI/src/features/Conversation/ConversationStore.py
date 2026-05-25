"""
Conversation persistence layer.
Manages JSONL turn storage, thread index, and NPZ embeddings.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Storage.StorePaths import now, read_json, write_json
from src.core.Embedding.Embedder import get_current_model_name
from src.features.Conversation.types import ConvoChunk, ConversationThread, Turn


class ConversationStore:
    """Pure persistence for conversations: JSONL, index, embeddings."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_dir / "index.json"

    # ── Turn persistence (JSONL) ──────────────────────────────────────────

    def _jsonl_path(self, thread_id: str) -> Path:
        return self.base_dir / f"{thread_id}.jsonl"

    def save_turn(self, thread_id: str, turn: Turn) -> None:
        """Append a single turn to JSONL."""
        path = self._jsonl_path(thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn.to_dict()) + "\n")

    def save_thread(self, thread: ConversationThread) -> None:
        """Overwrite entire JSONL with thread turns."""
        path = self._jsonl_path(thread.thread_id)
        with open(path, "w", encoding="utf-8") as f:
            for turn in thread.turns:
                f.write(json.dumps(turn.to_dict()) + "\n")

    def load_turns(self, thread_id: str) -> List[Turn]:
        """Load all turns from JSONL as Turn objects."""
        path = self._jsonl_path(thread_id)
        if not path.exists():
            return []
        turns = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    turns.append(Turn.from_dict(json.loads(line)))
        return turns

    def load_thread(self, thread_id: str) -> Optional[ConversationThread]:
        """Reconstruct a ConversationThread from JSONL (without embeddings)."""
        turns = self.load_turns(thread_id)
        if not turns:
            return None
        # Read last_indexed_turn_count from index if available
        index = self.read_index()
        meta = index.get(thread_id, {})
        return ConversationThread(
            thread_id=thread_id,
            turns=turns,
            created_at=turns[0].timestamp,
            last_active=turns[-1].timestamp,
            last_indexed_turn_count=meta.get("last_indexed_turn_count", 0),
        )

    # ── Thread index (quick listing) ──────────────────────────────────────

    def read_index(self) -> Dict[str, dict]:
        if not self._index_path.exists():
            return {}
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return {}

    def write_index(self, index: Dict[str, dict]) -> None:
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, default=str)

    def update_index(self, thread: ConversationThread) -> None:
        index = self.read_index()
        index[thread.thread_id] = {
            "title": thread.title,
            "turn_count": len(thread.turns),
            "last_active": thread.last_active.isoformat(),
            "state": thread.state.value,
            "npz_path": str(thread.npz_path) if thread.npz_path else None,
            "chunk_count": thread.chunk_count,
            "last_indexed_turn_count": thread.last_indexed_turn_count,   # NEW
        }
        self.write_index(index)

    def list_threads_meta(self) -> List[dict]:
        """Return all threads from index."""
        index = self.read_index()
        return [
            {"thread_id": tid, **meta} for tid, meta in index.items()
        ]

    # ── Embedding NPZ persistence ─────────────────────────────────────────

    def _npz_path(self, thread_id: str, model_name: Optional[str] = None) -> Path:
        if model_name is None:
            model_name = get_current_model_name()
        safe_model = model_name.replace("/", "_").replace("\\", "_")
        npz_dir = self.base_dir / thread_id
        npz_dir.mkdir(parents=True, exist_ok=True)
        return npz_dir / f"embeddings_{safe_model}.npz"

    def save_embeddings(
        self,
        thread_id: str,
        vectors: np.ndarray,
        chunks: List[ConvoChunk],
        model_name: Optional[str] = None,
    ) -> Path:
        """Save conversation embeddings to NPZ (overwrites)."""
        if model_name is None:
            model_name = get_current_model_name()
        path = self._npz_path(thread_id, model_name)

        np.savez_compressed(
            path,
            vectors=vectors.astype(np.float32),
            text=np.array([c.text for c in chunks], dtype=object),
            clean=np.array([c.clean for c in chunks], dtype=object),
            turn_ids=np.array([json.dumps(c.turn_ids) for c in chunks], dtype=object),
            roles=np.array([c.role for c in chunks], dtype=object),
            timestamps=np.array([c.timestamp.isoformat() for c in chunks], dtype=object),
            sentiments=np.array([c.sentiment for c in chunks], dtype=np.float32),
            topics=np.array([json.dumps(c.topics) for c in chunks], dtype=object),
            turn_ranges=np.array([json.dumps(list(c.turn_range)) for c in chunks], dtype=object),
            is_preference=np.array([c.is_preference_statement for c in chunks], dtype=bool),
            is_factual=np.array([c.is_factual_claim for c in chunks], dtype=bool),
        )

        meta_path = path.with_suffix(".meta.json")
        write_json(meta_path, {
            "thread_id": thread_id,
            "embedding_model": model_name,
            "dimensions": int(vectors.shape[1]),
            "chunk_count": len(chunks),
            "created_at": now(),
        })

        darwin.log(f"Saved convo embeddings: {path.name}", LogType.SUCCESS)
        return path

    def load_embeddings(self, thread_id: str) -> Tuple[Optional[np.ndarray], Optional[List[ConvoChunk]]]:
        """Load NPZ embeddings and reconstruct chunks."""
        model_name = get_current_model_name()
        path = self._npz_path(thread_id, model_name)
        if not path.exists():
            return None, None

        try:
            data = np.load(path, allow_pickle=True, mmap_mode='r')
        except ValueError:
            data = np.load(path, allow_pickle=True)

        n = len(data['text'])
        chunks = []
        for i in range(n):
            chunks.append(ConvoChunk(
                text=str(data['text'][i]),
                clean=str(data['clean'][i]),
                index=i,
                source=thread_id,
                turn_range=tuple(json.loads(str(data['turn_ranges'][i]))),
                role=str(data['roles'][i]),
                thread_id=thread_id,
                timestamp=datetime.fromisoformat(str(data['timestamps'][i])),
                sentiment=float(data['sentiments'][i]),
                topics=json.loads(str(data['topics'][i])),
                turn_ids=json.loads(str(data['turn_ids'][i])),
                is_preference_statement=bool(data['is_preference'][i]),
                is_factual_claim=bool(data['is_factual'][i]),
            ))
        return data['vectors'], chunks

    def get_embedding_metadata(self, thread_id: str) -> Optional[dict]:
        path = self._npz_path(thread_id)
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            return None
        return read_json(meta_path)

    def list_indexed_threads(self) -> List[str]:
        if not self.base_dir.exists():
            return []
        indexed = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and any(d.glob("embeddings_*.npz")):
                indexed.append(d.name)
        return indexed

    def delete_thread(self, thread_id: str) -> None:
        jsonl = self._jsonl_path(thread_id)
        if jsonl.exists():
            jsonl.unlink()
        npz_dir = self.base_dir / thread_id
        if npz_dir.exists():
            shutil.rmtree(npz_dir)
        index = self.read_index()
        if thread_id in index:
            del index[thread_id]
            self.write_index(index)
        darwin.log(f"Deleted thread {thread_id} from store", LogType.WARNING)