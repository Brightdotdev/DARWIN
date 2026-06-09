from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import unicodedata

import spacy
from spacy.tokens import Doc


class SpacyConceptExtractor:
    """
    spaCy wrapper optimized for knowledge extraction (graph + wiki).
    Loaded once by Darwin's ModelLoader, cached for the session.
    """

    _SIGNIFICANT_ENTITIES = {
        "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
        "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    }

    def __init__(self, model_path: str, model_name: str):
        self.model_path = Path(model_path)
        self.model_name = model_name

        if not self.model_path.exists():
            raise FileNotFoundError(f"spaCy model not found: {model_path}")

        disable = ["textcat", "textcat_multilabel", "entity_linker", "entity_ruler", "sentiment"]

        try:
            self.nlp = spacy.load(str(self.model_path), disable=disable)
        except Exception:
            self.nlp = spacy.load(str(self.model_path))

        self._enabled_pipes = set(self.nlp.pipe_names)

        if "parser" not in self._enabled_pipes and "senter" not in self._enabled_pipes:
            self.nlp.add_pipe("sentencizer")
            self._enabled_pipes.add("sentencizer")

    # ── NEW: Plugin-compatible batch processing ───────────────────────────

    def pipe(self, texts, batch_size=64):
        """Delegate to spaCy Language.pipe for batch processing."""
        return self.nlp.pipe(texts, batch_size=batch_size)

    def split_sentences(self, text: str, max_length: int = 100000) -> List[str]:
        """Return plain sentence strings (List[str]) for chunking."""
        text = text.strip()
        if not text:
            return []
        if len(text) > max_length:
            text = text[:max_length]
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # ── Public API: Knowledge Plugin Entry Points ─────────────────────────

    def extract_entities(self, text: str, max_length: int = 5000) -> List[Dict]:
        text = self._prepare(text, max_length)
        if not text:
            return []

        doc = self.nlp(text)
        entities = []
        seen: Set[str] = set()

        for ent in doc.ents:
            if ent.label_ not in self._SIGNIFICANT_ENTITIES:
                continue

            key = self._canonicalize(ent.text)
            if not key or key in seen or len(key) <= 2:
                continue

            seen.add(key)
            entities.append({
                "text": ent.text.strip(),
                "lemma": key,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "sentence_id": self._get_sentence_id(doc, ent),
            })

        return entities

    def extract_sentences(self, text: str, max_length: int = 100000) -> List[Dict]:
        """Split text into sentences with per-sentence entity lists."""
        text = text.strip()
        if not text:
            return []
        if len(text) > max_length:
            text = text[:max_length]

        doc = self.nlp(text)
        sentences = []

        for sent_idx, sent in enumerate(doc.sents):
            sent_entities = []
            seen_in_sent: Set[str] = set()

            for ent in sent.ents:
                if ent.label_ not in self._SIGNIFICANT_ENTITIES:
                    continue

                lemma = self._canonicalize(ent.text)
                if not lemma or lemma in seen_in_sent or len(lemma) <= 2:
                    continue

                seen_in_sent.add(lemma)
                sent_entities.append({
                    "text": ent.text.strip(),
                    "lemma": lemma,
                    "type": ent.label_,
                    "start": ent.start_char - sent.start_char,
                    "end": ent.end_char - sent.start_char,
                })

            sentences.append({
                "id": sent_idx,
                "text": sent.text.strip(),
                "start": sent.start_char,
                "end": sent.end_char,
                "entities": sent_entities,
                "entity_count": len(sent_entities),
            })

        return sentences

    def extract_concepts(self, text: str, max_length: int = 5000) -> List[Dict]:
        text = self._prepare(text, max_length)
        if not text:
            return []

        doc = self.nlp(text)
        concepts = []
        seen: Set[str] = set()

        for ent in doc.ents:
            if ent.label_ not in self._SIGNIFICANT_ENTITIES:
                continue

            key = self._canonicalize(ent.text)
            if not key or key in seen or len(key) <= 2:
                continue

            seen.add(key)
            concepts.append({
                "text": ent.text.strip(),
                "lemma": key,
                "type": ent.label_,
                "source": "ner",
            })

        if "parser" in self._enabled_pipes:
            for chunk in doc.noun_chunks:
                if all(t.pos_ in {"DET", "PRON"} for t in chunk):
                    continue

                key = self._canonicalize(chunk.text)
                if not key or key in seen or len(key) <= 3:
                    continue

                seen.add(key)
                concepts.append({
                    "text": chunk.text.strip(),
                    "lemma": key,
                    "type": "CONCEPT",
                    "source": "noun_chunk",
                })

        return concepts

    def extract_keywords_for_graph(
        self,
        texts: List[str],
        max_per_text: int = 20,
    ) -> Dict[str, List[int]]:
        keywords: Dict[str, List[int]] = defaultdict(list)

        for idx, text in enumerate(texts):
            if not text or not text.strip() or text.strip().startswith("[Image:"):
                continue

            doc = self.nlp(text[:3000])
            chunks = self._extract_chunks_from_doc(doc, max_per_text)

            for kw in chunks:
                keywords[kw].append(idx)

        return dict(keywords)

    def elevate_keywords(
        self,
        raw_keywords: Dict[str, List[int]],
        max_concepts: int = 100,
    ) -> Dict[str, List[int]]:
        if not raw_keywords:
            return {}

        sorted_kw = sorted(raw_keywords.items(), key=lambda x: len(x[1]), reverse=True)
        top_kw = sorted_kw[:max_concepts * 2]

        canonical_map: Dict[str, List[Tuple[str, List[int]]]] = defaultdict(list)

        for kw, indices in top_kw:
            canonical = self._quick_normalize(kw)
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

    # ── Capability checks ────────────────────────────────────────────────

    @property
    def has_parser(self) -> bool:
        return "parser" in self._enabled_pipes

    @property
    def has_ner(self) -> bool:
        return "ner" in self._enabled_pipes

    @property
    def has_tagger(self) -> bool:
        return "tagger" in self._enabled_pipes

    # ── Internal helpers ─────────────────────────────────────────────────

    def _prepare(self, text: str, max_length: int) -> str:
        text = text.strip()
        if len(text) > max_length:
            text = text[:max_length]
        return text

    @staticmethod
    def _canonicalize(text: str) -> str:
        text = text.strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = text.strip(".,;:!?" + chr(34) + chr(39) + "()[]{}")
        text = " ".join(text.split())
        return text

    @staticmethod
    def _quick_normalize(text: str) -> str:
        text = text.strip().lower()
        text = text.strip(".,;:!?" + chr(34) + chr(39) + "()[]{}")
        text = " ".join(text.split())
        return text

    def _get_sentence_id(self, doc: Doc, ent) -> int:
        for sent_idx, sent in enumerate(doc.sents):
            if sent.start <= ent.start < sent.end:
                return sent_idx
        return 0

    def _extract_chunks_from_doc(self, doc: Doc, max_per_doc: int = 20) -> List[str]:
        chunks = []
        seen: Set[str] = set()

        for ent in doc.ents:
            if ent.label_ in self._SIGNIFICANT_ENTITIES and ent.label_ != "GPE":
                key = self._canonicalize(ent.text)
                if key and key not in seen and len(key) > 2:
                    seen.add(key)
                    chunks.append(key)

        if "parser" not in self._enabled_pipes:
            return chunks[:max_per_doc]

        for chunk in doc.noun_chunks:
            tokens = [
                t for t in chunk
                if t.dep_ not in {"det", "poss"} and t.pos_ not in {"DET", "PRON"}
            ]
            if not tokens:
                continue

            clean = " ".join(t.text.lower() for t in tokens).strip()
            clean = self._canonicalize(clean)
            if clean and clean not in seen and len(clean) >= 3:
                seen.add(clean)
                chunks.append(clean)

        return chunks[:max_per_doc]

    def __repr__(self) -> str:
        pipes = ", ".join(sorted(self._enabled_pipes)) if self._enabled_pipes else "tokenizer only"
        return f"SpacyConceptExtractor({self.model_name}, pipes=[{pipes}])"