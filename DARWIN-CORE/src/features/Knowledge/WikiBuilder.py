"""
Darwin Wiki / Entity Index Builder (v2.0)
Per‑file construction with dominant‑type merging and aggressive noise filtering.
"""

from typing import Dict, List, Optional, Any, Set, Counter as TypingCounter
from collections import Counter
from src.core.Logger.DarwinLogger import darwin, LogType

from src.features.Knowledge.Utils import normalize, is_valid_entity
from src.features.Knowledge.lib import VALID_ENTITY_TYPES, EntityRecord
from src.features.Knowledge.WikiIndex import WikiIndex

class NullWikiIndex:
    """No‑op index – now a proper subclass (defined later)."""


class WikiBuilder:
    def __init__(self, store: Any):
        self.store = store

    def get(self, filename: str):
        """Load cached wiki without rebuilding. Returns NullWikiIndex if absent."""
        if not self.store.has_wiki(filename):
            return NullWikiIndex()

        wiki = WikiIndex(self.store)
        wiki._load_file(filename)
        return wiki

    def build(self, filename: str, rerun: bool = False,
              min_frequency: int = 2) -> Any:
        """
        Build or retrieve a wiki for *filename*.

        - `rerun=False` uses cached store version if present.
        - `min_frequency` sets the minimum number of mentions required
          for an entity to be kept (after type merging). Default 2 removes
          singletons.
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

            payload = self._build_payload(filename, body_records, spacy, min_frequency)
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
        Core extraction pipeline:
        1. Collect raw mentions per canonical name
        2. Merge types by dominant label
        3. Apply frequency floor & final serialisation
        """
        # Temporary accumulator: { canonical_name -> { type_counts: Counter, mentions: [...] } }
        raw_entities: Dict[str, Dict] = {}
        doc_chunks: Set[int] = set()

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
                # Fast rejection of obvious noise
                if not is_valid_entity(ent.text, ent.label_):
                    continue

                norm = normalize(ent.text)

                # Init record if new name
                if norm not in raw_entities:
                    raw_entities[norm] = {
                        "type_counts": Counter(),
                        "mentions": [],
                    }

                rec = raw_entities[norm]
                rec["type_counts"][ent.label_] += 1
                rec["mentions"].append({
                    "text": ent.text,
                    "chunk_id": chunk_id,
                    "span": [ent.start_char, ent.end_char],
                    "doc_id": filename,
                })

        # --- Post‑processing: merge types, build final entities ---
        final_entities: Dict[str, Dict] = {}
        for name, data in raw_entities.items():
            total_mentions = len(data["mentions"])
            if total_mentions < min_frequency:
                continue

            # Dominant type
            dominant_type = data["type_counts"].most_common(1)[0][0]
            entity_id = f"{name}|{dominant_type.lower()}"

            # Collect chunk & doc sets
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
            }

        return {
            "entities": final_entities,
            "doc_chunks": {filename: sorted(doc_chunks)},
        }

    def _load_spacy(self) -> Optional[Any]:
        try:
            from src.core.Models import ModelManager, ModelCategory
            loader = ModelManager.get_loader()
            spacy_wrapper = loader.get(ModelCategory.TOKENIZER)
            if spacy_wrapper is None:
                darwin.log("Spacy not available")
                return None
            return spacy_wrapper
        except Exception as e:
            darwin.log(f"Wiki: spaCy load failed: {e}", LogType.ERROR)
            return None