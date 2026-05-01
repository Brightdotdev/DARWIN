"""
src/core/Models/SpacyConceptExtractor.py
-----------------------------------------
Lightweight, focused spaCy wrapper for concept extraction.

Design: one model loaded once, cached by ModelLoader, used everywhere.
Graph needs concepts. Wiki needs concepts + context. Both get what they
need from the same object — no duplicate loading, no scattered spacy calls.
"""

from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict


class SpacyConceptExtractor:
    """
    spaCy wrapper optimized for knowledge extraction (graph + wiki).

    Loaded once by ModelLoader, cached for the session lifetime.
    Handles missing components gracefully — always returns something,
    even if spaCy pipes aren't all available.

    Usage:
        extractor = loader.get(ModelCategory.TOKENIZER)
        concepts = extractor.extract_concepts(text)          # → list[dict]
        entities = extractor.extract_entities(text)          # → list[dict]
        nouns    = extractor.extract_noun_chunks(text)       # → list[str]
        sents    = extractor.split_sentences(text)           # → list[str]
        keywords = extractor.extract_keywords_for_graph(chunks)  # → {kw: [idx]}
    """

    # Entity types we care about (skip noise like dates, numbers)
    _SIGNIFICANT_ENTITIES = {
        "PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT",
        "EVENT", "WORK_OF_ART", "LAW", "NORP", "LANGUAGE",
    }

    # POS tags we consider as concepts
    _CONCEPT_POS = {"NOUN", "PROPN", "ADJ"}

    def __init__(self, model_path: str, model_name: str):
        import spacy
        from spacy.util import load_model_from_path

        self.model_path = Path(model_path)
        self.model_name = model_name

        if not self.model_path.exists():
            raise FileNotFoundError(f"spaCy model not found: {model_path}")

        # ── Determine which components actually exist ──
        available = self._detect_components()
        exclude = self._build_exclude_list(available)

        try:
            self.nlp = load_model_from_path(self.model_path, exclude=exclude)
        except Exception:
            # Fallback: minimal load — disable everything non-essential
            minimal_exclude = [
                c for c in self._all_possible_components()
                if c not in {"tokenizer", "vocab"}
            ]
            self.nlp = load_model_from_path(self.model_path, exclude=minimal_exclude)

        self._enabled_pipes = set(self.nlp.pipe_names)
        print(f"   ✅ spaCy loaded: {', '.join(self._enabled_pipes) or 'tokenizer only'}")

    # ──────────────────────────────────────────────────────────────────────
    # Public API — what graph & wiki actually call
    # ──────────────────────────────────────────────────────────────────────

    def extract_concepts(self, text: str, max_length: int = 5000) -> List[Dict]:
        """
        Full concept extraction for a single chunk/passage.
        Returns entities + normalized noun phrases, deduplicated.

        Used by: wiki page generation, concept detail views.
        """
        text = self._prepare(text, max_length)
        if not text:
            return []

        doc = self.nlp(text)
        concepts = []
        seen: Set[str] = set()

        # Named entities first (higher quality)
        for ent in doc.ents:
            if ent.label_ not in self._SIGNIFICANT_ENTITIES:
                continue
            key = self._normalize(ent.text)
            if key and key not in seen and len(key) > 2:
                seen.add(key)
                concepts.append({
                    "text": ent.text.strip(),
                    "lemma": key,
                    "type": ent.label_,
                    "source": "ner",
                })

        # Noun chunks as fallback concepts
        if self.has_parser:
            for chunk in doc.noun_chunks:
                key = self._normalize(chunk.text)
                if key and key not in seen and len(key) > 3:
                    seen.add(key)
                    concepts.append({
                        "text": chunk.text.strip(),
                        "lemma": key,
                        "type": "CONCEPT",
                        "source": "noun_chunk",
                    })

        return concepts

    def extract_entities(self, text: str, max_length: int = 5000) -> List[Dict]:
        """
        Named entities only — fast, focused.
        Used by: graph node extraction, entity linking.
        """
        text = self._prepare(text, max_length)
        if not text:
            return []

        doc = self.nlp(text)
        entities = []
        seen: Set[str] = set()

        for ent in doc.ents:
            if ent.label_ not in self._SIGNIFICANT_ENTITIES:
                continue
            key = self._normalize(ent.text)
            if key and key not in seen and len(key) > 2:
                seen.add(key)
                entities.append({
                    "text": ent.text.strip(),
                    "lemma": key,
                    "type": ent.label_,
                })

        return entities

    def extract_noun_chunks(self, text: str, max_length: int = 5000) -> List[str]:
        """
        Clean noun phrases — used for keyword/concept extraction.
        Returns deduplicated, normalized strings.
        """
        text = self._prepare(text, max_length)
        if not text:
            return []

        doc = self.nlp(text)
        chunks = []
        seen: Set[str] = set()

        # NER first (highest quality)
        for ent in doc.ents:
            if ent.label_ in self._SIGNIFICANT_ENTITIES and ent.label_ != "GPE":
                key = self._normalize(ent.text)
                if key and key not in seen and len(key) > 2:
                    seen.add(key)
                    chunks.append(key)

        # Multi-word noun chunks only
        if self.has_parser:
            for chunk in doc.noun_chunks:
                clean = " ".join(
                    t.text.lower() for t in chunk
                    if t.dep_ not in {"det", "poss"} and t.pos_ not in {"DET", "PRON"}
                ).strip()
                clean = self._normalize(clean)
                if clean and clean not in seen and len(clean.split()) >= 2:
                    seen.add(clean)
                    chunks.append(clean)

        return chunks



    def split_sentences(self, text: str, max_length: int = 100000) -> List[str]:
        """
        Sentence splitting. Falls back to regex if parser unavailable.
        Used by: wiki summaries, chunk splitting.
        """
        text = text.strip()
        if len(text) > max_length:
            text = text[:max_length]
        if not text:
            return []

        try:
            doc = self.nlp(text)
            if doc.has_annotation("SENT_START"):
                return [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception:
            pass

        # Regex fallback
        import re
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def extract_keywords_for_graph(
        self,
        chunks: List,
        max_per_chunk: int = 20,
    ) -> Dict[str, List[int]]:
        """
        Extract keywords from all chunks for graph building.
        Returns {keyword: [chunk_index, ...]} — same shape as yake output.

        """
        keywords: Dict[str, List[int]] = defaultdict(list)

        for chunk in chunks:
            text = chunk.clean[:3000]
            if not text.strip():
                continue
            if text.strip().startswith("[Image:"):
                continue

            for kw in self.extract_noun_chunks(text)[:max_per_chunk]:
                keywords[kw].append(chunk.index)

        return dict(keywords)

    # ──────────────────────────────────────────────────────────────────────
    # Concept elevation (for graph pipeline)
    # ──────────────────────────────────────────────────────────────────────

    def elevate_keywords(
        self,
        raw_keywords: Dict[str, List[int]],
        max_concepts: int = 100,
    ) -> Dict[str, List[int]]:
        """
        Normalize raw keywords into clean concepts.
        - Lemmatizes ("running" → "run")
        - Deduplicates ("CO2" + "carbon dioxide" → "carbon dioxide")
        - Strips determiners ("the Calvin cycle" → "calvin cycle")

        Used by: ConceptElevator when mode="spacy".
        """
        sorted_kw = sorted(raw_keywords.items(), key=lambda x: len(x[1]), reverse=True)
        top_kw = sorted_kw[:max_concepts * 2]

        canonical_map: Dict[str, List[tuple]] = defaultdict(list)

        for kw, indices in top_kw:
            doc = self.nlp(kw[:100])

            if doc.ents:
                canonical = doc.ents[0].text.lower().strip()
            elif len(doc) >= 1:
                canonical = " ".join(
                    t.lemma_.lower() for t in doc
                    if not t.is_stop and t.is_alpha
                )
                if not canonical:
                    canonical = kw.lower().strip()
            else:
                canonical = kw.lower().strip()

            canonical = canonical.strip(".,;:!?\"'()[]{}")
            if len(canonical) < 3:
                continue

            canonical_map[canonical].append((kw, indices))

        elevated: Dict[str, List[int]] = {}
        for canonical, entries in canonical_map.items():
            merged = []
            seen = set()
            for _, idx_list in entries:
                for idx in idx_list:
                    if idx not in seen:
                        merged.append(idx)
                        seen.add(idx)
            elevated[canonical] = sorted(merged)

        return dict(
            sorted(elevated.items(), key=lambda x: len(x[1]), reverse=True)[:max_concepts]
        )

    # ──────────────────────────────────────────────────────────────────────
    # Capability checks
    # ──────────────────────────────────────────────────────────────────────

    @property
    def has_parser(self) -> bool:
        return "parser" in self._enabled_pipes

    @property
    def has_ner(self) -> bool:
        return "ner" in self._enabled_pipes

    @property
    def has_tagger(self) -> bool:
        return "tagger" in self._enabled_pipes

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────

    def _prepare(self, text: str, max_length: int) -> str:
        """Truncate and strip text before processing."""
        text = text.strip()
        if len(text) > max_length:
            text = text[:max_length]
        return text

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize a concept string: lowercase, strip punctuation, dedupe whitespace."""
        text = text.strip().lower()
        text = text.strip(".,;:!?\"'()[]{}")
        return " ".join(text.split())

    def _detect_components(self) -> Set[str]:
        """Check which components exist in the model directory."""
        available = set()
        for item in self.model_path.iterdir():
            if item.is_dir():
                available.add(item.name)
        for essential in {"tokenizer", "vocab"}:
            if (self.model_path / essential).exists():
                available.add(essential)
        return available

    def _all_possible_components(self) -> Set[str]:
        return {
            "tok2vec", "tagger", "parser", "ner", "lemmatizer",
            "senter", "attribute_ruler", "textcat", "textcat_multilabel",
            "entity_linker", "entity_ruler", "sentiment", "transformer",
        }

    def _build_exclude_list(self, available: Set[str]) -> List[str]:
        """Build list of components to exclude on load."""
        all_possible = self._all_possible_components()
        exclude = sorted(all_possible - available)
        # Never exclude tok2vec if parser or tagger is available
        if "tok2vec" in exclude and ("tagger" in available or "parser" in available):
            exclude.remove("tok2vec")
        return exclude

    def __repr__(self) -> str:
        pipes = ", ".join(sorted(self._enabled_pipes)) if self._enabled_pipes else "tokenizer only"
        return f"SpacyConceptExtractor({self.model_name}, pipes=[{pipes}])"