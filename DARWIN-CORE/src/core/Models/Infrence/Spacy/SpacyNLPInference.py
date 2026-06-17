# src/core/Models/ModelInference/SpacyNLPInference.py
"""
SpaCy NLP Inference Engine
Loads spaCy model and runs NLP pipeline with metadata.
Enhanced with robust loading, canonicalization, and concept extraction.
"""

import gc
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Iterator, Set

from src.core.Logger.DarwinLogger import darwin, LogType


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NLPMetadata:
    """Metadata about the NLP inference run."""
    model_name: str
    pipeline: List[str]          # pipeline components loaded
    inference_time_ms: float
    num_texts: int
    num_tokens: int
    language: str
    vector_size: Optional[int]   # None if no word vectors
    has_word_vectors: bool
    has_transformer: bool
    spacy_version: str


@dataclass
class Entity:
    """Named entity result."""
    text: str
    label: str
    start: int
    end: int
    score: float


@dataclass
class Token:
    """Token-level result."""
    text: str
    lemma: str
    pos: str
    tag: str
    dep: str
    is_stop: bool
    is_punct: bool
    is_alpha: bool
    is_digit: bool
    ent_type: Optional[str]
    ent_iob: Optional[str]


@dataclass
class NLPResult:
    """Single NLP output."""
    text: str
    tokens: List[Token]
    entities: List[Entity]
    noun_chunks: List[str]
    sentiment: Optional[float]   # if textcat component present
    metadata: NLPMetadata


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Engine
# ═══════════════════════════════════════════════════════════════════════════════

class SpacyNLPInference:
    """
    spaCy NLP inference engine.
    Loads spaCy model and runs full NLP pipeline.

    Expects model directory layout from SpacyNLPModelDownloader:
        model_dir/
            config.cfg
            meta.json
            tokenizer/
            vocab/
            ... (pipeline components)
    """

    # Entity types considered "significant" for extraction (from old model)
    _SIGNIFICANT_ENTITIES = {
        "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
        "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    }

    def __init__(
        self,
        model_dir: Path,
        model_name: str = "en_core_web_sm",
        disable_components: Optional[List[str]] = None,
    ):
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.disable_components = disable_components or []

        self._nlp = None
        self._spacy_version = "unknown"
        self._enabled_pipes: Set[str] = set()

    @property
    def model_path(self) -> Path:
        """Return the path to the loaded model directory."""
        return self.model_dir

    # ── Capability checks (from old model) ──────────────────────────────────

    @property
    def has_parser(self) -> bool:
        return "parser" in self._enabled_pipes

    @property
    def has_ner(self) -> bool:
        return "ner" in self._enabled_pipes

    @property
    def has_tagger(self) -> bool:
        return "tagger" in self._enabled_pipes

    # ── Darwin logging ───────────────────────────────────────────────────────

    def _log(self, msg: str, level: LogType = LogType.INFO):
        darwin.log(f"[SpaCy Inference] {msg}", level)

    # ── Context Manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "SpacyNLPInference":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> "SpacyNLPInference":
        """Load spaCy model. Chainable."""
        if self.is_loaded():
            self._log("Already loaded, skipping.")
            return self

        if not self.model_dir.exists():
            self._log(f"Model directory not found: {self.model_dir}", LogType.ERROR)
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        if not self._check_model_exists():
            self._log("Model files missing. Run downloader first.", LogType.ERROR)
            raise FileNotFoundError("Model files missing. Run downloader first.")

        if not self._check_integrity():
            self._log("Model integrity check failed. Model may be corrupted.", LogType.ERROR)
            raise RuntimeError("Model integrity check failed. Model may be corrupted.")

        self._log(f"Loading spaCy model from {self.model_dir}")
        self._load_spacy()
        self._log("Model loaded successfully.", LogType.SUCCESS)
        return self

    def unload(self) -> None:
        """Unload model and free memory."""
        if not self.is_loaded():
            self._log("Not loaded, nothing to unload.")
            return

        self._log("Unloading model...")
        self._nlp = None
        gc.collect()
        self._log("Unloaded.", LogType.SUCCESS)

    def is_loaded(self) -> bool:
        return self._nlp is not None

    def _check_model_exists(self) -> bool:
        """Check if model files exist at all."""
        return (self.model_dir / "config.cfg").exists()

    def _check_integrity(self) -> bool:
        """
        Quick integrity check: files exist and have reasonable size.
        """
        required = [
            self.model_dir / "config.cfg",
            self.model_dir / "meta.json",
            self.model_dir / "tokenizer",
            self.model_dir / "vocab",
        ]
        for p in required:
            if not p.exists():
                self._log(f"Integrity fail: missing {p.name}", LogType.WARNING)
                return False
            if p.is_file() and p.stat().st_size < 10:
                self._log(f"Integrity fail: {p.name} too small", LogType.WARNING)
                return False
        return True

    # ── Internal: spaCy loading (with fallback) ─────────────────────────────

    def _load_spacy(self) -> None:
        """Load spaCy model from disk, with fallback if disabling fails."""
        try:
            import spacy
        except ImportError:
            self._log("spaCy not installed. pip install spacy", LogType.ERROR)
            raise ImportError("spaCy not installed. pip install spacy")

        self._spacy_version = spacy.__version__

        # Try loading with disable list first
        try:
            self._nlp = spacy.load(self.model_dir, disable=self.disable_components)
        except Exception as first_err:
            self._log(
                f"spaCy load with disable={self.disable_components} failed: {first_err}. "
                "Retrying without disable list...",
                LogType.WARNING,
            )
            try:
                self._nlp = spacy.load(self.model_dir)
            except Exception as second_err:
                self._log(f"spaCy load completely failed: {second_err}", LogType.ERROR)
                raise

        self._enabled_pipes = set(self._nlp.pipe_names)

        # Ensure a sentence segmenter is present
        if "parser" not in self._enabled_pipes and "senter" not in self._enabled_pipes:
            self._nlp.add_pipe("sentencizer")
            self._enabled_pipes.add("sentencizer")

        self._log(
            f"Loaded spaCy {self._spacy_version} | "
            f"Pipeline: {self._nlp.pipe_names} | "
            f"Vectors: {self._nlp.vocab.vectors_length}d"
        )

    # ── Public inference API ─────────────────────────────────────────────────

    def process(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        n_process: int = 1,
    ) -> List[NLPResult]:
        """
        Process text(s) through spaCy pipeline.
        Returns full NLP results with tokens, entities, noun chunks.
        """
        if not self.is_loaded():
            self._log("Model not loaded. Call .load() first.", LogType.ERROR)
            raise RuntimeError("Model not loaded. Call .load() first.")

        if isinstance(texts, str):
            texts = [texts]

        self._log(f"Processing {len(texts)} text(s)")

        spinner_id = f"spacy_{id(texts)}"
        darwin.spinner_start(spinner_id, f"Processing {len(texts)} text(s)")

        start = time.perf_counter()
        results = []
        total_tokens = 0

        for doc in self._nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            total_tokens += len(doc)

            tokens = [
                Token(
                    text=t.text,
                    lemma=t.lemma_,
                    pos=t.pos_,
                    tag=t.tag_,
                    dep=t.dep_,
                    is_stop=t.is_stop,
                    is_punct=t.is_punct,
                    is_alpha=t.is_alpha,
                    is_digit=t.is_digit,
                    ent_type=t.ent_type_ if t.ent_type_ else None,
                    ent_iob=t.ent_iob_ if t.ent_iob_ else None,
                )
                for t in doc
            ]

            entities = [
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=getattr(ent, "score", 0.0),
                )
                for ent in doc.ents
            ]
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]

            sentiment = None
            if doc.cats:
                sentiment = max(doc.cats.values()) if doc.cats else None

            results.append(
                NLPResult(
                    text=doc.text,
                    tokens=tokens,
                    entities=entities,
                    noun_chunks=noun_chunks,
                    sentiment=sentiment,
                    metadata=None,          # filled after all processed
                )
            )

        inference_time_ms = (time.perf_counter() - start) * 1000
        darwin.spinner_stop(spinner_id, success=True)

        meta = self._build_metadata(len(texts), total_tokens, inference_time_ms)
        for r in results:
            r.metadata = meta

        self._log(
            f"Processing complete: {len(texts)} text(s), {total_tokens} tokens "
            f"in {inference_time_ms:.1f}ms",
            LogType.SUCCESS,
        )
        return results

    def process_stream(
        self,
        texts: Iterator[str],
        batch_size: int = 64,
        n_process: int = 1,
    ) -> Iterator[NLPResult]:
        """
        Stream process texts through spaCy pipeline.
        Yields NLP results one at a time.
        """
        if not self.is_loaded():
            self._log("Model not loaded. Call .load() first.", LogType.ERROR)
            raise RuntimeError("Model not loaded. Call .load() first.")

        self._log("Streaming processing...")
        start = time.perf_counter()
        total_tokens = 0
        num_texts = 0

        for doc in self._nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            total_tokens += len(doc)
            num_texts += 1

            tokens = [
                Token(
                    text=t.text,
                    lemma=t.lemma_,
                    pos=t.pos_,
                    tag=t.tag_,
                    dep=t.dep_,
                    is_stop=t.is_stop,
                    is_punct=t.is_punct,
                    is_alpha=t.is_alpha,
                    is_digit=t.is_digit,
                    ent_type=t.ent_type_ if t.ent_type_ else None,
                    ent_iob=t.ent_iob_ if t.ent_iob_ else None,
                )
                for t in doc
            ]

            entities = [
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=0.0,
                )
                for ent in doc.ents
            ]

            noun_chunks = [chunk.text for chunk in doc.noun_chunks]

            sentiment = None
            if doc.cats:
                sentiment = max(doc.cats.values()) if doc.cats else None

            yield NLPResult(
                text=doc.text,
                tokens=tokens,
                entities=entities,
                noun_chunks=noun_chunks,
                sentiment=sentiment,
                metadata=None,
            )

        inference_time_ms = (time.perf_counter() - start) * 1000
        meta = self._build_metadata(num_texts, total_tokens, inference_time_ms)

        self._log(
            f"Stream processing complete: {num_texts} text(s), {total_tokens} tokens "
            f"in {inference_time_ms:.1f}ms",
            LogType.SUCCESS,
        )

    # ── Direct pipe access ──────────────────────────────────────────────────

    def pipe(self, texts, batch_size: int = 64):
        """
        Delegate to spaCy Language.pipe for low‑level batch iteration.
        Useful when you want to process spaCy Doc objects directly.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call .load() first.")
        return self._nlp.pipe(texts, batch_size=batch_size)

    # ── Quick helpers ───────────────────────────────────────────────────────

    def extract_entities(self, text: str, max_length: int = 5000) -> List[Dict]:
        """
        Extract significant entities with canonicalization and filtering.
        Returns list of dicts with text, lemma, type, start, end, sentence_id.
        """
        if not self.is_loaded():
            return []
        text = self._prepare(text, max_length)
        if not text:
            return []
        doc = self._nlp(text)
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

    def extract_noun_chunks(self, text: str, max_length: int = 5000) -> List[str]:
        """Quick noun chunk extraction (returns list of chunk texts)."""
        results = self.process(text, batch_size=1)
        return results[0].noun_chunks if results else []

    def lemmatize(self, text: str) -> List[str]:
        """Quick lemmatization."""
        results = self.process(text)
        return [t.lemma for t in results[0].tokens] if results else []

    def get_vectors(self, text: str) -> Optional[List[float]]:
        """Get document vector if model has word vectors."""
        if not self.is_loaded():
            return None
        doc = self._nlp(text)
        if doc.has_vector:
            return doc.vector.tolist()
        return None

    def extract_concepts(self, text: str, max_length: int = 5000) -> List[Dict]:
        """
        Extract concepts combining NER and noun chunks (from old model).
        Returns list of dicts with text, lemma, type, source.
        """
        if not self.is_loaded():
            return []
        text = self._prepare(text, max_length)
        if not text:
            return []
        doc = self._nlp(text)
        concepts = []
        seen: Set[str] = set()

        # NER concepts
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

        # Noun chunk concepts (if parser available)
        if self.has_parser:
            for chunk in doc.noun_chunks:
                # Skip purely determiner/pronoun chunks
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

    # ── Internal: Metadata ───────────────────────────────────────────────────

    def _build_metadata(
        self,
        num_texts: int,
        num_tokens: int,
        inference_time_ms: float,
    ) -> NLPMetadata:
        """Build metadata from loaded model state."""
        pipeline = self._nlp.pipe_names if self._nlp else []
        has_vectors = self._nlp.vocab.vectors_length > 0 if self._nlp else False

        return NLPMetadata(
            model_name=self.model_name,
            pipeline=pipeline,
            inference_time_ms=round(inference_time_ms, 3),
            num_texts=num_texts,
            num_tokens=num_tokens,
            language=self._nlp.meta.get("lang", "unknown") if self._nlp else "unknown",
            vector_size=self._nlp.vocab.vectors_length if self._nlp and has_vectors else None,
            has_word_vectors=has_vectors,
            has_transformer="transformer" in pipeline if self._nlp else False,
            spacy_version=self._spacy_version,
        )

    # ── Internal helpers (from old model) ───────────────────────────────────

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

    def _get_sentence_id(self, doc, ent) -> int:
        for sent_idx, sent in enumerate(doc.sents):
            if sent.start <= ent.start < sent.end:
                return sent_idx
        return 0

    # ── Info ─────────────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        """Return static info about the loaded model without running inference."""
        if not self.is_loaded():
            return {
                "status": "not_loaded",
                "model_dir": str(self.model_dir),
                "model_name": self.model_name,
            }

        return {
            "status": "loaded",
            "model_dir": str(self.model_dir),
            "model_name": self.model_name,
            "spacy_version": self._spacy_version,
            "pipeline": self._nlp.pipe_names,
            "language": self._nlp.meta.get("lang", "unknown"),
            "vector_size": self._nlp.vocab.vectors_length,
            "has_vectors": self._nlp.vocab.vectors_length > 0,
            "has_transformer": "transformer" in self._nlp.pipe_names,
            "disable_components": self.disable_components,
        }

    # ── Debug ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        pipes = ", ".join(sorted(self._enabled_pipes)) if self._enabled_pipes else "tokenizer only"
        return f"SpacyNLPInference(model_name={self.model_name}, pipes=[{pipes}])"