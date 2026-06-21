"""
Darwin Wiki / Entity Index Query Interface (v3.1)
Lazy-loading, LRU-bounded, thread-safe merged in-memory index
with robust null-object fallback.

REQUIRES: Full Darwin stack (src.core.Logger)
"""

import re
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Literal, Any, Set

from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Knowledge.lib import EntityRecord, Mention, VALID_ENTITY_TYPES
from src.features.Knowledge.Utils import normalize


class WikiIndex:
    """
    Global query interface backed by the store.
    Lazy-loads per-file wikis into a merged in-memory view.
    Thread-safe with LRU eviction for memory-bounded operation.
    """

    def __init__(self, store: Any = None, max_loaded_files: int = 20):
        self.store = store
        self._entities: Dict[str, EntityRecord] = {}
        self._doc_chunks: Dict[str, Set[int]] = {}
        self._loaded_files: Set[str] = set()
        self._file_load_order: OrderedDict[str, None] = OrderedDict()
        self._max_loaded = max_loaded_files
        self._lock = threading.RLock()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _recompute_salience(self, records: List[EntityRecord]) -> None:
        if not records:
            return
        raw = [r.frequency * r.spread for r in records]
        mn, mx = min(raw), max(raw)
        if mx == mn:
            for r in records:
                r.salience = 1.0 if mx > 0 else 0.0
        else:
            for r in records:
                r.salience = (r.frequency * r.spread - mn) / (mx - mn)

    def _load_file(self, filename: str) -> None:
        """Thread-safe lazy load with LRU eviction."""
        with self._lock:
            if filename in self._loaded_files:
                self._file_load_order.move_to_end(filename)
                return

            try:
                payload = self.store.load_wiki(filename)
            except Exception as e:
                darwin.log(f"Wiki: could not load wiki for {filename}: {e}", LogType.ERROR)
                return

            if payload is None:
                darwin.log(
                    f"Wiki: wiki payload for {filename} is None (file missing on disk). "
                    "Consider rebuilding with rerun=True.",
                    LogType.WARNING
                )
                return

            for eid, rec_dict in payload.get("entities", {}).items():
                rec = EntityRecord.from_dict(rec_dict)
                if eid not in self._entities:
                    self._entities[eid] = rec
                else:
                    existing = self._entities[eid]
                    existing.mentions.extend(rec.mentions)
                    existing.chunk_ids = sorted(set(existing.chunk_ids + rec.chunk_ids))
                    existing.doc_ids = sorted(set(existing.doc_ids + rec.doc_ids))
                    existing.frequency = len(existing.mentions)
                    existing.spread = len(existing.doc_ids)

            chunk_ids = payload.get("doc_chunks", {}).get(filename, [])
            self._doc_chunks[filename] = set(chunk_ids)
            self._loaded_files.add(filename)
            self._file_load_order[filename] = None

            # LRU eviction
            while len(self._file_load_order) > self._max_loaded:
                oldest = self._file_load_order.popitem(last=False)[0]
                self._unload_file(oldest)

    def _unload_file(self, filename: str) -> None:
        """Evict a file from memory, removing its entities from the merged view."""
        with self._lock:
            self._loaded_files.discard(filename)
            self._doc_chunks.pop(filename, None)

            # Remove entities that only existed in this file
            to_remove = []
            for eid, rec in self._entities.items():
                rec.mentions = [m for m in rec.mentions if m.doc_id != filename]
                rec.chunk_ids = sorted({m.chunk_id for m in rec.mentions})
                rec.doc_ids = sorted({m.doc_id for m in rec.mentions})
                rec.frequency = len(rec.mentions)
                rec.spread = len(rec.doc_ids)
                if not rec.mentions:
                    to_remove.append(eid)

            for eid in to_remove:
                del self._entities[eid]

    def _resolve_filenames(self, filenames: Optional[List[str]]) -> List[str]:
        return list(self._loaded_files) if filenames is None else filenames

    def _get_filtered_records(
        self,
        filenames: List[str],
        entity_type: Optional[str] = None,
        min_frequency: int = 1,
    ) -> List[EntityRecord]:
        with self._lock:
            records = []
            for rec in self._entities.values():
                if entity_type and rec.type != entity_type:
                    continue
                scoped = [m for m in rec.mentions if m.doc_id in filenames]
                if not scoped:
                    continue
                freq = len(scoped)
                if freq < min_frequency:
                    continue
                chunk_ids = sorted({m.chunk_id for m in scoped})
                doc_ids = sorted({m.doc_id for m in scoped})
                view = EntityRecord(
                    canonical_name=rec.canonical_name,
                    type=rec.type,
                    mentions=scoped,
                    chunk_ids=chunk_ids,
                    doc_ids=doc_ids,
                    frequency=freq,
                    spread=len(doc_ids),
                )
                records.append(view)
            return records

    # ── Public Query API ─────────────────────────────────────────────────

    def detect_entities(self, query: str, filenames: Optional[List[str]] = None) -> List[EntityRecord]:
        if not query:
            return []
        target = self._resolve_filenames(filenames)
        for f in target:
            self._load_file(f)

        with self._lock:
            qnorm = normalize(query)
            matched = []
            seen = set()
            for rec in self._entities.values():
                if re.search(r"\b" + re.escape(rec.canonical_name) + r"\b", qnorm):
                    if any(m.doc_id in target for m in rec.mentions):
                        if rec.entity_id not in seen:
                            seen.add(rec.entity_id)
                            matched.append(rec)
            self._recompute_salience(matched)
            matched.sort(key=lambda r: (-r.salience, -r.frequency))
            return matched

    def retrieve_by_entity(self, entity_id: str, filenames: Optional[List[str]] = None) -> List[int]:
        target = self._resolve_filenames(filenames)
        for f in target:
            self._load_file(f)

        with self._lock:
            rec = self._entities.get(entity_id)
            if not rec:
                return []
            chunk_counts: Dict[int, int] = {}
            for m in rec.mentions:
                if m.doc_id in target:
                    chunk_counts[m.chunk_id] = chunk_counts.get(m.chunk_id, 0) + 1
            return [cid for cid, _ in sorted(chunk_counts.items(), key=lambda x: -x[1])]

    def list_entities(
        self,
        filenames: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        sort_by: Literal["frequency", "spread", "salience", "name"] = "salience",
        limit: int = 50,
        min_frequency: int = 1,
    ) -> List[EntityRecord]:
        # Enforce valid type if provided
        if entity_type is not None and entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {entity_type}")

        target = self._resolve_filenames(filenames)
        for f in target:
            self._load_file(f)

        with self._lock:
            records = self._get_filtered_records(target, entity_type, min_frequency)
            if not records:
                return []
            if sort_by == "salience":
                self._recompute_salience(records)
                records.sort(key=lambda r: (-r.salience, -r.frequency))
            elif sort_by == "frequency":
                records.sort(key=lambda r: -r.frequency)
            elif sort_by == "spread":
                records.sort(key=lambda r: -r.spread)
            elif sort_by == "name":
                records.sort(key=lambda r: r.canonical_name)
            return records[:limit]

    def get_entity_stats(self, filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        target = self._resolve_filenames(filenames)
        for f in target:
            self._load_file(f)

        with self._lock:
            records = self._get_filtered_records(target)
            if not records:
                return {
                    "total_entities": 0, "total_mentions": 0,
                    "type_distribution": {}, "top_entities": [],
                    "most_connected": [], "niche_entities": [],
                    "document_coverage": 0.0,
                }
            total_mentions = sum(r.frequency for r in records)
            type_dist: Dict[str, int] = {}
            for r in records:
                type_dist[r.type] = type_dist.get(r.type, 0) + 1
            self._recompute_salience(records)
            top = sorted(records, key=lambda r: (-r.salience, -r.frequency))[:10]
            connected = sorted(records, key=lambda r: -r.spread)[:10]
            niche = sorted(
                [r for r in records if r.frequency <= 2 and r.spread == 1],
                key=lambda r: r.canonical_name
            )[:20]

            all_chunks = 0
            chunks_with_ent: Set[Any] = set()
            for fname in target:
                cids = self._doc_chunks.get(fname, set())
                all_chunks += len(cids)
                for r in records:
                    for m in r.mentions:
                        if m.doc_id == fname:
                            chunks_with_ent.add((fname, m.chunk_id))
            coverage = len(chunks_with_ent) / all_chunks if all_chunks else 0.0

            return {
                "total_entities": len(records),
                "total_mentions": total_mentions,
                "type_distribution": type_dist,
                "top_entities": [r.to_dict() for r in top],
                "most_connected": [r.to_dict() for r in connected],
                "niche_entities": [r.to_dict() for r in niche],
                "document_coverage": round(coverage, 4),
            }

    def get_document_profile(self, filename: str) -> Dict[str, Any]:
        self._load_file(filename)

        with self._lock:
            records = self._get_filtered_records([filename])
            if not records:
                return {
                    "filename": filename,
                    "total_entities": 0,
                    "dominant_types": [],
                    "top_entities": [],
                    "entity_density": 0.0,
                    "key_concepts": [],
                    "type_breakdown": {},
                }
            self._recompute_salience(records)
            total_chunks = len(self._doc_chunks.get(filename, set()))
            type_breakdown: Dict[str, int] = {}
            for r in records:
                type_breakdown[r.type] = type_breakdown.get(r.type, 0) + 1
            dominant = [t for t, _ in sorted(type_breakdown.items(), key=lambda x: -x[1])[:3]]
            top = sorted(records, key=lambda r: (-r.salience, -r.frequency))[:10]
            key = sorted([r for r in records if r.salience > 0.5], key=lambda r: -r.salience)
            density = len(records) / total_chunks if total_chunks else 0.0

            return {
                "filename": filename,
                "total_entities": len(records),
                "dominant_types": dominant,
                "top_entities": [r.to_dict() for r in top],
                "entity_density": round(density, 4),
                "key_concepts": [r.to_dict() for r in key],
                "type_breakdown": type_breakdown,
            }

    def compare_entities(self, entity_a_id: str, entity_b_id: str,
                         filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        target = self._resolve_filenames(filenames)
        for f in target:
            self._load_file(f)

        with self._lock:
            a = self._entities.get(entity_a_id)
            b = self._entities.get(entity_b_id)
            if not a or not b:
                missing = []
                if not a:
                    missing.append(entity_a_id)
                if not b:
                    missing.append(entity_b_id)
                return {
                    "error": f"Entity not found: {', '.join(missing)}",
                    "entity_a": None,
                    "entity_b": None,
                    "shared_docs": [],
                    "shared_chunks": [],
                    "cooccurrence_score": 0.0,
                    "stronger_in": None,
                }

            a_mentions = [m for m in a.mentions if m.doc_id in target]
            b_mentions = [m for m in b.mentions if m.doc_id in target]
            a_chunks = {m.chunk_id for m in a_mentions}
            b_chunks = {m.chunk_id for m in b_mentions}
            a_docs = {m.doc_id for m in a_mentions}
            b_docs = {m.doc_id for m in b_mentions}

            shared_docs = sorted(a_docs & b_docs)
            shared_chunks = sorted(a_chunks & b_chunks)
            union = a_chunks | b_chunks
            cooc = len(shared_chunks) / len(union) if union else 0.0

            a_view = EntityRecord(
                canonical_name=a.canonical_name, type=a.type,
                mentions=a_mentions, chunk_ids=sorted(a_chunks),
                doc_ids=sorted(a_docs), frequency=len(a_mentions), spread=len(a_docs)
            )
            b_view = EntityRecord(
                canonical_name=b.canonical_name, type=b.type,
                mentions=b_mentions, chunk_ids=sorted(b_chunks),
                doc_ids=sorted(b_docs), frequency=len(b_mentions), spread=len(b_docs)
            )
            self._recompute_salience([a_view, b_view])
            stronger = entity_a_id if a_view.salience >= b_view.salience else entity_b_id

            return {
                "entity_a": a_view.to_dict(),
                "entity_b": b_view.to_dict(),
                "shared_docs": shared_docs,
                "shared_chunks": shared_chunks,
                "cooccurrence_score": round(cooc, 4),
                "stronger_in": stronger,
            }

    def rank_by_type(self, entity_type: str, filenames: Optional[List[str]] = None) -> List[EntityRecord]:
        if entity_type not in VALID_ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {entity_type}")
        return self.list_entities(filenames=filenames, entity_type=entity_type,
                                  sort_by="salience", limit=999999)

    def find_niche_entities(self, filenames: Optional[List[str]] = None, limit: int = 20) -> List[EntityRecord]:
        records = self.list_entities(filenames=filenames, limit=999999)
        niche = [r for r in records if r.frequency <= 2 and r.spread == 1]
        niche.sort(key=lambda r: r.canonical_name)
        return niche[:limit]


# ── Null Object (true subclass) ────────────────────────────────────────

class NullWikiIndex(WikiIndex):
    """No-op index used when spaCy or wiki data is unavailable."""

    def __init__(self):
        super().__init__(store=None)

    def detect_entities(self, query: str, filenames: Optional[List[str]] = None) -> List[EntityRecord]:
        return []

    def retrieve_by_entity(self, entity_id: str, filenames: Optional[List[str]] = None) -> List[int]:
        return []

    def list_entities(self, **kwargs) -> List[EntityRecord]:
        return []

    def get_entity_stats(self, filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "total_entities": 0, "total_mentions": 0,
            "type_distribution": {}, "top_entities": [],
            "most_connected": [], "niche_entities": [],
            "document_coverage": 0.0,
        }

    def get_document_profile(self, filename: str) -> Dict[str, Any]:
        return {
            "filename": filename,
            "total_entities": 0,
            "dominant_types": [],
            "top_entities": [],
            "entity_density": 0.0,
            "key_concepts": [],
            "type_breakdown": {},
        }

    def compare_entities(self, entity_a_id: str, entity_b_id: str,
                         filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "entity_a": None,
            "entity_b": None,
            "shared_docs": [],
            "shared_chunks": [],
            "cooccurrence_score": 0.0,
            "stronger_in": None,
        }

    def rank_by_type(self, entity_type: str, filenames: Optional[List[str]] = None) -> List[EntityRecord]:
        return []

    def find_niche_entities(self, filenames: Optional[List[str]] = None, limit: int = 20) -> List[EntityRecord]:
        return []