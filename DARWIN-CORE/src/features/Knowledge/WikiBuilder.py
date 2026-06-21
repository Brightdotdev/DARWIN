import threading
import multiprocessing as mp
from typing import Dict, List, Optional, Any, Set, Counter as TypingCounter, Tuple
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Knowledge.Utils import normalize, is_valid_entity, is_garbage_entity
from src.features.Knowledge.lib import ENTITY_STOPWORDS
from src.features.Knowledge.WikiIndex import WikiIndex, NullWikiIndex


# ── Worker function for multiprocessing ────────────────────────────────

def _extract_worker(args: Tuple[int, str, Any]) -> Tuple[int, List[Dict]]:
    """
    ProcessPoolExecutor worker: run spaCy on a single chunk.

    Args:
        args: (chunk_id, chunk_text, spacy_wrapper)

    Returns:
        (chunk_id, list of raw mention dicts)
    """
    chunk_id, chunk_text, spacy_wrapper = args
    mentions = []

    try:
        doc = spacy_wrapper.nlp(chunk_text)
    except Exception:
        return chunk_id, []

    for ent in doc.ents:
        if not is_valid_entity(ent.text, ent.label_):
            continue

        norm = normalize(ent.text)
        if norm in ENTITY_STOPWORDS or is_garbage_entity(ent.text):
            continue

        mentions.append({
            "text": ent.text,
            "label": ent.label_,
            "chunk_id": chunk_id,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        })

    return chunk_id, mentions


class WikiBuilder:
    def __init__(self, store: Any, max_workers: int = 4):
        self.store = store
        self._lock = threading.Lock()
        self.max_workers = max_workers

    def get(self, filename: str):
        """Load cached wiki without rebuilding. Returns NullWikiIndex if absent."""
        if not self.store.has_wiki(filename):
            return NullWikiIndex()

        wiki = WikiIndex(self.store)
        wiki._load_file(filename)
        return wiki

    def build(self, filename: str, rerun: bool = False,
              min_frequency: int = 2, use_multiprocess: bool = True) -> Any:
        """
        Build or retrieve a wiki for *filename*.

        - `rerun=False` uses cached store version if present.
        - `min_frequency` sets the minimum number of mentions required
          for an entity to be kept (after type merging). Default 2 removes
          singletons.
        - `use_multiprocess=True` spawns worker processes for spaCy extraction,
          bypassing the GIL and utilizing multiple CPU cores.
        """
        if not rerun and self.store.has_wiki(filename):
            return self.get(filename)

        spacy = self._load_spacy()
        if spacy is None:
            darwin.log("Tokenizer not available. Wiki indexing disabled.", LogType.ERROR)
            return NullWikiIndex()

        try:
            # Only fetch body records – skip headers, footers, tables
            body_records = [
                r for r in self.store.query_records(
                    filename,
                    columns=["index", "clean", "structural_type"]
                )
                if r.get("structural_type") == "body"
            ]
            darwin.log(f"{len(body_records)} body records loaded for {filename}")
            if not body_records:
                darwin.log(f"No body records found for {filename}")
                return NullWikiIndex()

            if use_multiprocess and len(body_records) > 10:
                payload = self._build_payload_multiprocess(
                    filename, body_records, spacy, min_frequency
                )
            else:
                payload = self._build_payload(
                    filename, body_records, spacy, min_frequency
                )

            self.store.save_wiki(filename, payload)

        except Exception as e:
            darwin.log(f"Wiki: failed to build wiki for {filename}: {e}", LogType.ERROR)
            return NullWikiIndex()

        wiki = WikiIndex(self.store)
        wiki._load_file(filename)
        return wiki

    def _build_payload(self, filename: str, body_records: List[Dict],
                       spacy: Any, min_frequency: int) -> Dict[str, Any]:
        """
        Single-threaded extraction pipeline.
        Use for small documents (< 10 chunks) or when multiprocessing is disabled.
        """
        raw_entities: Dict[str, Dict] = {}
        doc_chunks: Set[int] = set()
        seen_mentions: Set[tuple] = set()

        for row in body_records:
            chunk_id = row.get("index", 0)
            chunk_text = row.get("clean", "")
            doc_chunks.add(chunk_id)

            if not chunk_text or not chunk_text.strip():
                continue

            try:
                doc = spacy.nlp(chunk_text)
            except Exception as e:
                darwin.log(f"Wiki: spaCy failed on chunk {chunk_id} of {filename}: {e}", LogType.ERROR)
                continue

            for ent in doc.ents:
                if not is_valid_entity(ent.text, ent.label_):
                    continue

                norm = normalize(ent.text)
                if norm in ENTITY_STOPWORDS or is_garbage_entity(ent.text):
                    continue

                mention_key = (chunk_id, ent.start_char, ent.end_char)
                if mention_key in seen_mentions:
                    continue
                seen_mentions.add(mention_key)

                if norm not in raw_entities:
                    raw_entities[norm] = {
                        "type_counts": Counter(),
                        "mentions": [],
                        "type_confidence": {},
                    }

                rec = raw_entities[norm]
                rec["type_counts"][ent.label_] += 1
                rec["mentions"].append({
                    "text": ent.text,
                    "chunk_id": chunk_id,
                    "span": [ent.start_char, ent.end_char],
                    "doc_id": filename,
                })

        return self._finalize_payload(raw_entities, doc_chunks, filename, min_frequency)

    def _build_payload_multiprocess(self, filename: str, body_records: List[Dict],
                                     spacy: Any, min_frequency: int) -> Dict[str, Any]:
        """
        Multi-process extraction pipeline.
        Spawns worker processes to bypass Python's GIL and utilize multiple cores.

        Each worker gets a chunk of text and returns raw mentions.
        The main process aggregates, deduplicates, and builds the payload.
        """
        darwin.log(f"Wiki: using {self.max_workers} workers for {len(body_records)} chunks")

        raw_entities: Dict[str, Dict] = {}
        doc_chunks: Set[int] = set()
        seen_mentions: Set[tuple] = set()

        # Prepare work items: (chunk_id, chunk_text, spacy_wrapper)
        # NOTE: spacy_wrapper must be pickleable for ProcessPoolExecutor
        work_items = [
            (row.get("index", 0), row.get("clean", ""), spacy)
            for row in body_records
            if row.get("clean", "").strip()
        ]

        for chunk_id, _, _ in work_items:
            doc_chunks.add(chunk_id)

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_extract_worker, item): item[0] for item in work_items}

            for future in as_completed(futures):
                chunk_id, mentions = future.result()

                for m in mentions:
                    mention_key = (chunk_id, m["start_char"], m["end_char"])
                    if mention_key in seen_mentions:
                        continue
                    seen_mentions.add(mention_key)

                    norm = normalize(m["text"])
                    if norm not in raw_entities:
                        raw_entities[norm] = {
                            "type_counts": Counter(),
                            "mentions": [],
                            "type_confidence": {},
                        }

                    rec = raw_entities[norm]
                    rec["type_counts"][m["label"]] += 1
                    rec["mentions"].append({
                        "text": m["text"],
                        "chunk_id": chunk_id,
                        "span": [m["start_char"], m["end_char"]],
                        "doc_id": filename,
                    })

        return self._finalize_payload(raw_entities, doc_chunks, filename, min_frequency)

    def _finalize_payload(self, raw_entities: Dict[str, Dict],
                          doc_chunks: Set[int], filename: str,
                          min_frequency: int) -> Dict[str, Any]:
        """
        Shared post-processing: merge types, apply frequency floor, build final entities.
        """
        final_entities: Dict[str, Dict] = {}
        for name, data in raw_entities.items():
            total_mentions = len(data["mentions"])
            if total_mentions < min_frequency:
                continue

            dominant_type = data["type_counts"].most_common(1)[0][0]
            entity_id = f"{name}|{dominant_type.lower()}"

            chunk_ids = sorted({m["chunk_id"] for m in data["mentions"]})
            doc_ids = sorted({m["doc_id"] for m in data["mentions"]})

            final_entities[entity_id] = {
                "canonical_name": name,
                "type": dominant_type,
                "mentions": data["mentions"],
                "chunk_ids": chunk_ids,
                "doc_ids": doc_ids,
                "frequency": total_mentions,
                "spread": len(doc_ids),
                "salience": 0.0,
                "type_confidence": dict(data["type_counts"]),
            }

        return {
            "entities": final_entities,
            "doc_chunks": {filename: sorted(doc_chunks)},
        }

    def _load_spacy(self) -> Optional[Any]:
        from src.core.Models.ModelManager import modelManager
        from src.core.Models.Lib.ModelRegistry import ModelCategory
        try:
            spacy_wrapper = modelManager.get(ModelCategory.NLP)
            if spacy_wrapper is None:
                darwin.log("Spacy not available")
                return None
            return spacy_wrapper
        except Exception as e:
            darwin.log(f"Wiki: spaCy load failed: {e}", LogType.ERROR)
            return None