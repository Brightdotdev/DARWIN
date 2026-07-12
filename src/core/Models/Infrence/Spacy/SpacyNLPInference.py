


import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator, Iterable,Tuple

from src.core.Logger.DarwinLogger import darwin


from src.core.Models.Infrence.Spacy.Utils import canonicalize, is_valid_entity, is_valid_noun_chunk


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NLPMetadata:
    model_name: str
    inference_time_ms: float
    num_texts: int
    num_tokens: int
    pipeline: List[str]


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    score: float = 0.0


@dataclass
class NLPResult:
    text: str
    entities: List[Entity]
    metadata: Optional[NLPMetadata] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Engine
# ═══════════════════════════════════════════════════════════════════════════════

class SpacyNLPInference:
    """spaCy NLP inference engine with NER + noun chunk co-extraction."""

    _SIGNIFICANT_ENTITIES = {
        "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
        "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    }

    DEFAULT_MAX_LENGTH = 5000

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
        self._enabled_pipes: List[str] = []

    @property
    def has_ner(self) -> bool:
        return "ner" in self._enabled_pipes

    @property
    def has_parser(self) -> bool:
        return "parser" in self._enabled_pipes


    def __enter__(self) -> "SpacyNLPInference":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

    def load(self) -> "SpacyNLPInference":
        if self.is_loaded():
            return self
        if not self.model_dir.exists():
            darwin.error(f"Model directory not found: {self.model_dir}")
            return self
        if not (self.model_dir / "config.cfg").exists():
            darwin.error("Model files missing. Run downloader first.")
            return self

        darwin.info(f"Loading spaCy model from {self.model_dir}")
        import spacy
        self._spacy_version = spacy.__version__

        try:
            self._nlp = spacy.load(self.model_dir, disable=self.disable_components)
        except Exception as first_err:
            darwin.warning(f"Load with disable={self.disable_components} failed: {first_err}. Retrying...")
            self._nlp = spacy.load(self.model_dir)

        self._enabled_pipes = list(self._nlp.pipe_names)
        if "parser" not in self._enabled_pipes and "senter" not in self._enabled_pipes:
            self._nlp.add_pipe("sentencizer")
            self._enabled_pipes.append("sentencizer")

        darwin.info(f"Loaded spaCy {self._spacy_version} | Pipeline: {self._nlp.pipe_names} | Vectors: {self._nlp.vocab.vectors_length}d")
        return self

    def unload(self) -> None:
        if not self.is_loaded():
            return
        darwin.info("Unloading model...")
        self._nlp = None
        self._enabled_pipes.clear()
        gc.collect()
        darwin.success("Unloaded.")

    def is_loaded(self) -> bool:
        return self._nlp is not None

    # ── Public inference API ─────────────────────────────────────────────────

    def process(self, texts: str | List[str], batch_size: int = 64) -> List[NLPResult]:
        if not self.is_loaded():
            darwin.error("Model not loaded. Call .load() first.")
            return []
        if isinstance(texts, str):
            texts = [texts]
        texts = [t.strip() for t in texts if t and t.strip()]
        if not texts:
            darwin.error("No valid texts to process")
            return []

        batch_size = max(1, batch_size)
        darwin.info(f"Processing {len(texts)} text(s)")
        spinner_id = f"spacy_{id(texts)}"
        darwin.spinner_start(spinner_id, f"Processing {len(texts)} text(s)")

        start = time.perf_counter()
        results = []
        total_tokens = 0

        for doc in self._nlp.pipe(texts, batch_size=batch_size):
            total_tokens += len(doc)
            entities = [
                Entity(text=ent.text.strip(), label=ent.label_, start=ent.start_char, end=ent.end_char, score=getattr(ent, "score", 0.0))
                for ent in doc.ents
            ]
            results.append(NLPResult(text=doc.text.strip(), entities=entities))

        inference_time_ms = (time.perf_counter() - start) * 1000
        darwin.spinner_stop(spinner_id, success=True)

        meta = NLPMetadata(model_name=self.model_name, inference_time_ms=round(inference_time_ms, 3), num_texts=len(texts), num_tokens=total_tokens, pipeline=self._enabled_pipes)
        for r in results:
            r.metadata = meta

        darwin.success(f"Done: {len(texts)} text(s), {total_tokens} tokens in {inference_time_ms:.1f}ms")
        return results

    # ── FAST PATH: Co-extraction of NER + noun chunks ────────────────────────

    def extract_entities_raw(
        self,
        texts: Iterable[str],
        batch_size: int = 64,
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield validated entities from BOTH NER and noun chunks.
        source="ner" | "noun_chunk" | "both" (if both detect same span).
        """
        if not self.is_loaded():
            darwin.error("Model not loaded. Call .load() first.")
            return iter([])

        for doc in self._nlp.pipe(texts, batch_size=batch_size):
            # Collect all candidate spans with their sources
            span_map: Dict[Tuple[int, int, str], Dict[str, Any]] = {}

            # NER entities
            for ent in doc.ents:
                if not is_valid_entity(ent.text, ent.label_):
                    continue
                key = (ent.start_char, ent.end_char, canonicalize(ent.text))
                span_map[key] = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "ner",
                }

            # Noun chunks (only if parser available)
            if self.has_parser:
                for chunk in doc.noun_chunks:
                    # Skip determiner-only chunks
                    chunk_words = [t.text.lower() for t in chunk if not t.pos_ == "DET"]
                    if not chunk_words:
                        continue

                    # Reconstruct chunk text without determiners
                    chunk_text = " ".join(t.text for t in chunk if not t.pos_ == "DET")
                    if not chunk_text:
                        chunk_text = chunk.text

                    if not is_valid_noun_chunk(chunk_text):
                        continue

                    key = (chunk.start_char, chunk.end_char, canonicalize(chunk_text))
                    if key in span_map:
                        span_map[key]["source"] = "both"
                    else:
                        span_map[key] = {
                            "text": chunk_text,
                            "label": "CONCEPT",  # Noun chunks get generic type
                            "start": chunk.start_char,
                            "end": chunk.end_char,
                            "source": "noun_chunk",
                        }

            # Yield all unique candidates
            for item in span_map.values():
                yield item

    # ── Direct pipe access ──────────────────────────────────────────────────

    def pipe(self, texts: Iterable[str], batch_size: int = 64) -> Iterator:
        if not self.is_loaded():
            darwin.error("Model not loaded. Call .load() first.")
            return iter([])
        return self._nlp.pipe(texts, batch_size=batch_size)

    # ── Info ─────────────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        if not self.is_loaded():
            return {"status": "not_loaded", "model_dir": str(self.model_dir), "model_name": self.model_name}
        return {
            "status": "loaded",
            "model_dir": str(self.model_dir),
            "model_name": self.model_name,
            "spacy_version": self._spacy_version,
            "pipeline": self._enabled_pipes,
            "vector_size": self._nlp.vocab.vectors_length,
            "has_vectors": self._nlp.vocab.vectors_length > 0,
            "disable_components": self.disable_components,
        }

    def __repr__(self) -> str:
        pipes = ", ".join(self._enabled_pipes) if self._enabled_pipes else "tokenizer only"
        return f"SpacyNLPInference(model_name={self.model_name}, pipes=[{pipes}])"