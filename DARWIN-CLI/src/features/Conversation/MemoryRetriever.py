import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.features.Conversation.ConversationConfig import ConversationConfig
from src.features.Indexing.Embedding.Embedder import  embed_query
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.AppPaths import conversation_memory_path


class MemoryRetriever:
    def __init__(self,config: ConversationConfig):
        self.config = config
        self.index_path = conversation_memory_path()
        self.index_path.mkdir(exist_ok=True)

        self.global_emb_path = self.index_path / "global_memory.npz"
        self.global_meta_path = self.index_path / "global_metadata.jsonl"

        self._embeddings: Optional[np.ndarray] = None
        self._metadata: List[Dict[str, Any]] = []
        self._dirty = False

    def index_turn(self, turn_record: Dict[str, Any]) -> None:
        self._load()

        emb = np.array(turn_record["embedding"], dtype=np.float16).reshape(1, -1)
        if len(self._embeddings) > 0:
            self._embeddings = np.vstack([self._embeddings, emb])
        else:
            self._embeddings = emb

        self._metadata.append({
            "conversation_id": turn_record["conversation_id"],
            "turn_id": turn_record["turn_id"],
            "claims": turn_record.get("extracted_claims", []),
            "preferences": turn_record.get("preference_signals", {}),
            "intent": turn_record.get("intent"),
            "timestamp": turn_record.get("timestamp"),
        })
        self._dirty = True

        self._maybe_rechunk()

    def retrieve(
        self,
        query: str,
        conversation_id: str,
        top_k: int = 3,
        cross_conversation: bool = True,
    ) -> List[Dict[str, Any]]:
        self._load()

        if len(self._embeddings) == 0:
            return []

        query_emb = embed_query(query)

        results = self._search_subset(query_emb, conversation_id, top_k)

        if cross_conversation and len(results) < top_k:
            global_results = self._search_global(query_emb, top_k - len(results), exclude=conversation_id)
            results.extend(global_results)

        return results[:top_k]

    def _search_subset(
        self,
        query_emb: np.ndarray,
        conversation_id: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        indices = [
            i for i, m in enumerate(self._metadata)
            if m["conversation_id"] == conversation_id
        ]
        if not indices:
            return []

        subset_emb = self._embeddings[indices]
        similarities = np.dot(subset_emb, query_emb) / (
            np.linalg.norm(subset_emb, axis=1) * np.linalg.norm(query_emb)
        )
        top_local = np.argsort(similarities)[-top_k:][::-1]

        return [
            {
                "text": self._metadata[indices[i]]["claims"],
                "similarity": float(similarities[i]),
                "timestamp": self._metadata[indices[i]]["timestamp"],
                "conversation_id": conversation_id,
                "preferences": self._metadata[indices[i]].get("preferences", {}),
            }
            for i in top_local
            if similarities[i] > 0.5
        ]

    def _search_global(
        self,
        query_emb: np.ndarray,
        top_k: int,
        exclude: str,
    ) -> List[Dict[str, Any]]:
        similarities = np.dot(self._embeddings, query_emb) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top_global = np.argsort(similarities)[-top_k * 3:][::-1]

        results = []
        for i in top_global:
            meta = self._metadata[i]
            if meta["conversation_id"] == exclude:
                continue
            if similarities[i] <= 0.5:
                continue
            results.append({
                "text": meta["claims"],
                "similarity": float(similarities[i]),
                "timestamp": meta["timestamp"],
                "conversation_id": meta["conversation_id"],
                "preferences": meta.get("preferences", {}),
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def count_indexed(self, conversation_id: str) -> int:
        self._load()
        return sum(1 for m in self._metadata if m["conversation_id"] == conversation_id)

    def _maybe_rechunk(self) -> None:
        if not self.global_emb_path.exists():
            return

        size_mb = self.global_emb_path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_npz_mb:
            darwin.log(f"Memory index {size_mb:.1f}MB > {self.config.max_npz_mb}MB, rechunking...", LogType.INFO)
            self._rechunk()

    def _rechunk(self) -> None:
        sorted_meta = sorted(self._metadata, key=lambda x: x.get("timestamp", ""))
        by_conv: Dict[str, List[Dict[str, Any]]] = {}
        for m in sorted_meta:
            by_conv.setdefault(m["conversation_id"], []).append(m)

        new_metadata = []
        new_embeddings = []

        for conv_id, turns in by_conv.items():
            recent = turns[-self.config.rechunk_keep_recent:]
            old = turns[:-self.config.rechunk_keep_recent]

            if old:
                summary_claims = []
                for t in old:
                    summary_claims.extend(t.get("claims", []))

                summary_text = f"Conversation {conv_id} summary: " + "; ".join(summary_claims[:10])
                summary_emb = embed_query(summary_text[:500])

                new_embeddings.append(np.array(summary_emb, dtype=np.float16).reshape(1, -1))
                new_metadata.append({
                    "conversation_id": conv_id,
                    "turn_id": "summary",
                    "claims": summary_claims[:10],
                    "preferences": {},
                    "intent": "summary",
                    "timestamp": old[-1].get("timestamp"),
                    "is_summary": True,
                })

            for t in recent:
                idx = self._metadata.index(t)
                new_embeddings.append(self._embeddings[idx].reshape(1, -1))
                new_metadata.append(t)

        if new_embeddings:
            self._embeddings = np.vstack(new_embeddings).astype(np.float16)
        else:
            self._embeddings = np.zeros((0, self.config.embedding_dim), dtype=np.float16)
        self._metadata = new_metadata
        self._dirty = True

        darwin.log(f"Rechunked: {len(sorted_meta)} turns -> {len(new_metadata)} entries", LogType.SUCCESS)

    def _load(self) -> None:
        if self._embeddings is not None:
            return

        if self.global_emb_path.exists():
            loaded = np.load(self.global_emb_path)
            self._embeddings = loaded["embeddings"].astype(np.float16)
            self._metadata = []
            with open(self.global_meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._metadata.append(json.loads(line))
        else:
            self._embeddings = np.zeros((0, self.config.embedding_dim), dtype=np.float16)
            self._metadata = []

    def _save(self) -> None:
        if not self._dirty:
            return

        np.savez_compressed(self.global_emb_path, embeddings=self._embeddings)
        with open(self.global_meta_path, "w", encoding="utf-8") as f:
            for m in self._metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        self._dirty = False

    def flush(self) -> None:
        self._save()

    def __del__(self):
        self._save()