"""
Summarizer — Extractive summarization via spaCy + pytextrank.

Production-grade rewrite with:
  - Isolated per-instance spaCy pipelines (no shared-state pollution)
  - Original-case TextRank (no lowercasing before processing)
  - Robust pytextrank version handling (float vs object)
  - Vector validation before KMeans clustering
  - Proper sentence-aware chunking using spaCy (no regex splitting)
  - Deduplication, boundary-awareness, and safe fallbacks
  - Thread-safe initialization
  - All behaviour controlled by constructor parameters.

Purposes (per user spec):
  1. Blob reduction before embedding
  2. Conversation history compression
  3. Retrieved results classification / response block extraction
  — NOT for chunking. Chunking is sacred and handled elsewhere.
"""

import re
import threading
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

import numpy as np

from src.core.Logger.DarwinLogger import darwin, LogType

# ── Configuration defaults ──────────────────────────────────────────────────
DEFAULT_SENTENCE_COUNT = 5
MIN_CHARS_TO_SUMMARIZE = 300
DEFAULT_CHUNK_SIZE = 20000          # characters per chunk for long texts
LARGE_TEXT_THRESHOLD = 50000        # auto-delegate to chunked above this


# ── Internal: Sentence record ───────────────────────────────────────────────

@dataclass
class _Sentence:
    """Internal representation of a sentence with all metadata."""
    text: str
    lower: str
    vector: Optional[np.ndarray] = None
    rank: float = 0.0
    index: int = 0  # original document order


# ── spaCy pipeline factory (isolated per Summarizer) ──────────────────────────

class _SpacyPipelineFactory:
    """
    Creates isolated spaCy pipelines per Summarizer instance.

    The global SpacyLoader is used only to get the base model.
    Each Summarizer gets its OWN pipeline copy to avoid:
      - race conditions on add_pipe()
      - config leakage between instances
      - cross-module pollution
    """

    _base_lock = threading.Lock()
    _base_nlp = None
    _base_model_name: Optional[str] = None

    @classmethod
    def get_base_nlp(cls):
        """Get the shared base model (read-only, never mutated)."""
        if cls._base_nlp is not None:
            return cls._base_nlp

        with cls._base_lock:
            if cls._base_nlp is not None:
                return cls._base_nlp

            try:
                from src.core.Models.ModelManager import modelManager
                from src.core.Models.Lib.ModelRegistry import ModelCategory

                spacy_wrapper = modelManager.get(ModelCategory.NLP)
                if spacy_wrapper is None:
                    darwin.log("Spacy not available for Summarizer", LogType.ERROR)
                    return None

                cls._base_nlp = spacy_wrapper.nlp
                cls._base_model_name = getattr(spacy_wrapper, 'model_name', 'unknown')
                darwin.log(f"Summarizer base spaCy model: {cls._base_model_name}", LogType.INFO)

            except Exception as e:
                darwin.log(f"[Summarizer] spaCy load failed: {e}", LogType.ERROR)
                return None

        return cls._base_nlp

    @classmethod
    def create_isolated_pipeline(cls, pos_filter: bool = False):
        """
        Create a fresh, isolated spaCy pipeline with textrank.

        Returns a spaCy Language object that the caller owns.
        """
        base = cls.get_base_nlp()
        if base is None:
            return None

        # Deep copy — each Summarizer gets its own pipeline
        try:
            nlp = base.copy()
        except Exception as e:
            darwin.log(f"[Summarizer] Failed to copy spaCy pipeline: {e}", LogType.ERROR)
            return None

        # Add textrank to this isolated pipeline
        if "textrank" not in nlp.pipe_names:
            try:
                nlp.add_pipe("textrank")
            except Exception as e:
                darwin.log(f"[Summarizer] Failed to add textrank: {e}", LogType.ERROR)
                return None

        # Apply POS filter to THIS pipeline only
        if pos_filter:
            try:
                nlp.get_pipe("textrank").pos_keep = {"NOUN", "PROPN", "ADJ"}
                darwin.log("TextRank POS filter set to NOUN, PROPN, ADJ", LogType.INFO)
            except Exception as e:
                darwin.log(f"[Summarizer] Failed to set POS filter: {e}", LogType.WARNING)

        return nlp


# ── Main Summarizer ─────────────────────────────────────────────────────────

class Summarizer:
    """
    Extractive summarizer using spaCy + pytextrank.

    Constructor parameters
    ----------------------
    min_sentence_words : int
        Skip sentences with fewer than this many words.  Default 4.
    chunk_sentences : int
        Number of sentences to keep per chunk when chunking large docs.
        Default 3.
    pos_filter : bool
        If True, only nouns, proper nouns, and adjectives contribute to
        TextRank scores.  Default False.
    clustered : bool
        If True, use KMeans clustering on spaCy sentence vectors instead
        of raw TextRank ranks.  Requires scikit-learn.  Default False.
    chunk_size : int
        Max characters per chunk for long texts.  Default 20000.
    batch_size : int
        Batch size for spaCy pipe() on chunks.  Default 16.
    """

    def __init__(
        self,
        min_sentence_words: int = 4,
        chunk_sentences: int = 3,
        pos_filter: bool = False,
        clustered: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        batch_size: int = 16,
    ):
        if min_sentence_words < 1:
            raise ValueError(f"min_sentence_words must be >= 1, got {min_sentence_words}")
        if chunk_sentences < 1:
            raise ValueError(f"chunk_sentences must be >= 1, got {chunk_sentences}")
        if chunk_size < 1000:
            raise ValueError(f"chunk_size must be >= 1000, got {chunk_size}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.min_sentence_words = min_sentence_words
        self.chunk_sentences = chunk_sentences
        self.pos_filter = pos_filter
        self.clustered = clustered
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        # Isolated pipeline — no shared state
        self._nlp = None
        self._init_nlp()

    # ── spaCy / pytextrank setup ────────────────────────────────────────

    def _init_nlp(self) -> None:
        """Initialize an isolated spaCy pipeline with textrank."""
        try:
            import pytextrank  # noqa: F401 — verifies installation
        except ImportError:
            darwin.log("pytextrank not installed — summarizer disabled", LogType.WARNING)
            self._nlp = None
            return

        self._nlp = _SpacyPipelineFactory.create_isolated_pipeline(
            pos_filter=self.pos_filter
        )

        if self._nlp is None:
            darwin.log("spaCy pipeline not available — summarizer disabled", LogType.WARNING)

    @property
    def available(self) -> bool:
        return self._nlp is not None

    # ── Public summarization methods ───────────────────────────────────

    def summarize(self, text: str, max_sentences: int = DEFAULT_SENTENCE_COUNT) -> str:
        """Basic extractive summarization. Automatically chunks large texts."""
        if not self.available or len(text) < MIN_CHARS_TO_SUMMARIZE:
            return text
        if max_sentences < 1:
            raise ValueError(f"max_sentences must be >= 1, got {max_sentences}")

        if len(text) > LARGE_TEXT_THRESHOLD:
            darwin.log("Large text — switching to chunked summarization", LogType.INFO)
            return self.summarize_long(text, max_sentences)

        return self._textrank(text, max_sentences)

    def summarize_by_target(self, text: str, target_chars: int = 500) -> str:
        """
        Compress text to fit under `target_chars`.
        Large texts first produce a chunked summary, then iteratively compress.
        """
        if not self.available or len(text) <= target_chars:
            return text
        if target_chars < 1:
            raise ValueError(f"target_chars must be >= 1, got {target_chars}")

        if len(text) > LARGE_TEXT_THRESHOLD:
            result = self.summarize_long(text, max_sentences=10)
        else:
            result = text

        for n in [5, 3, 2, 1]:
            if len(result) <= target_chars:
                break
            result = self._textrank(result, n)

        # Final safety: hard truncate if still over target
        if len(result) > target_chars:
            # Truncate at last sentence boundary before target
            truncated = result[:target_chars]
            last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
            if last_period > target_chars // 2:
                result = truncated[:last_period + 1]
            else:
                result = truncated  # fallback: mid-word truncate

        return result

    def summarize_long(
        self,
        text: str,
        max_sentences: int = DEFAULT_SENTENCE_COUNT,
    ) -> str:
        """Chunked summarization for very long documents."""
        if not self.available or len(text) < MIN_CHARS_TO_SUMMARIZE:
            return text
        if max_sentences < 1:
            raise ValueError(f"max_sentences must be >= 1, got {max_sentences}")

        if len(text) <= self.chunk_size:
            return self._textrank(text, max_sentences)

        chunks = self._split_into_sentence_chunks(text, self.chunk_size)
        if not chunks:
            return text

        # Process all chunks in a spaCy batch for speed
        docs = list(self._nlp.pipe(chunks, batch_size=self.batch_size))

        chunk_summaries = []
        seen_sentences = set()  # deduplicate across chunks (overlap handling)

        for doc in docs:
            summary = self._extract_top_sentences(doc, self.chunk_sentences)
            if summary:
                # Deduplicate: skip sentences already seen from overlap
                sents = summary.split('. ')
                filtered = []
                for s in sents:
                    key = s.strip().lower()
                    if key and key not in seen_sentences:
                        seen_sentences.add(key)
                        filtered.append(s)
                if filtered:
                    chunk_summaries.append('. '.join(filtered))

        if not chunk_summaries:
            # Fallback: return first chunk's top sentences rather than full text
            fallback = self._extract_top_sentences(docs[0], max_sentences) if docs else ""
            return fallback or text[:self.chunk_size]

        combined = " ".join(chunk_summaries)
        return self._textrank(combined, max_sentences)

    def summarize_for_conversation(self, text: str) -> str:
        """Short, 3-sentence summary ideal for conversation context."""
        return self.summarize(text, max_sentences=3)

    # ── Core: TextRank on original-cased text ─────────────────────────

    def _textrank(self, text: str, max_sentences: int) -> str:
        """
        Run TextRank on original-cased text.

        No lowercasing — preserves sentence boundaries, POS tags, NER,
        and TextRank graph quality.
        """
        if self.clustered:
            return self._textrank_clustered(text, max_sentences)

        sentences = self._extract_sentences_with_ranks(text)

        if not sentences:
            return self._safe_fallback(text)

        # Sort by rank (descending), take top N, then restore original order
        top_indices = sorted(
            range(len(sentences)),
            key=lambda i: sentences[i].rank,
            reverse=True,
        )[:max_sentences]

        top = [sentences[i] for i in sorted(top_indices, key=lambda i: sentences[i].index)]
        return " ".join(s.text for s in top)

    # ── Clustered summarization (sklearn KMeans) ──────────────────────

    def _textrank_clustered(self, text: str, max_sentences: int) -> str:
        """
        Cluster sentence vectors, pick best from each cluster using TextRank.
        Falls back to standard TextRank if vectors are unavailable or invalid.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            darwin.log("scikit-learn not installed — falling back to TextRank", LogType.WARNING)
            return self._textrank(text, max_sentences)

        sentences = self._extract_sentences_with_ranks(text)

        if not sentences:
            return self._safe_fallback(text)

        if len(sentences) <= max_sentences:
            return " ".join(s.text for s in sentences)

        # Validate vectors
        vectors = []
        for s in sentences:
            if s.vector is None or not np.any(s.vector):
                darwin.log("Sentence vectors unavailable — falling back to TextRank", LogType.WARNING)
                return self._textrank(text, max_sentences)
            vectors.append(s.vector)

        vectors = np.array(vectors)
        n_clusters = min(max_sentences, len(sentences))

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            clusters = kmeans.fit_predict(vectors)
        except Exception as e:
            darwin.log(f"KMeans clustering failed: {e} — falling back to TextRank", LogType.WARNING)
            return self._textrank(text, max_sentences)

        # For each cluster, pick the sentence with highest TextRank score
        selected = []
        for cluster_id in range(n_clusters):
            indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            if not indices:
                continue
            best_idx = max(indices, key=lambda i: sentences[i].rank)
            selected.append((best_idx, sentences[best_idx]))

        # Sort by original document order
        selected.sort(key=lambda x: x[0])
        return " ".join(s.text for _, s in selected)

    # ── Unified sentence extraction with rank parsing ─────────────────

    def _extract_sentences_with_ranks(self, text: str) -> List[_Sentence]:
        """
        Process text through spaCy + pytextrank, extract sentences with ranks.

        Handles both:
          - pytextrank returning objects with .rank attribute
          - pytextrank returning float values directly

        Returns sentences in original document order.
        """
        doc = self._nlp(text)
        sentences = []

        for i, sent in enumerate(doc.sents):
            txt = sent.text.strip()
            if not txt:
                continue

            word_count = len(txt.split())
            if word_count < self.min_sentence_words:
                continue

            # Parse rank — handle both API styles
            rank_raw = getattr(sent._, "textrank", None)
            rank_val = 0.0

            if rank_raw is not None:
                if hasattr(rank_raw, "rank"):
                    rank_val = float(rank_raw.rank)
                elif isinstance(rank_raw, (int, float)):
                    rank_val = float(rank_raw)
                else:
                    # Try to coerce
                    try:
                        rank_val = float(rank_raw)
                    except (TypeError, ValueError):
                        rank_val = 0.0

            sentences.append(_Sentence(
                text=txt,
                lower=txt.lower(),
                vector=sent.vector if hasattr(sent, 'vector') else None,
                rank=rank_val,
                index=i,
            ))

        # Warn if all ranks are zero (indicates pytextrank misconfiguration)
        if sentences and all(s.rank == 0.0 for s in sentences):
            darwin.log(
                "[Summarizer] All TextRank scores are zero — "
                "check pytextrank version and pipeline configuration",
                LogType.WARNING
            )

        return sentences

    # ── Sentence extraction from Doc (for chunked mode) ───────────────

    def _extract_top_sentences(self, doc, max_sentences: int) -> str:
        """Extract top sentences from a spaCy Doc, respecting min_sentence_words."""
        sentences = []

        for i, sent in enumerate(doc.sents):
            txt = sent.text.strip()
            if not txt or len(txt.split()) < self.min_sentence_words:
                continue

            rank_raw = getattr(sent._, "textrank", None)
            rank_val = 0.0

            if rank_raw is not None:
                if hasattr(rank_raw, "rank"):
                    rank_val = float(rank_raw.rank)
                elif isinstance(rank_raw, (int, float)):
                    rank_val = float(rank_raw)
                else:
                    try:
                        rank_val = float(rank_raw)
                    except (TypeError, ValueError):
                        rank_val = 0.0

            sentences.append((txt, rank_val, i))

        if not sentences:
            return ""

        top_indices = sorted(
            range(len(sentences)),
            key=lambda i: sentences[i][1],
            reverse=True,
        )[:max_sentences]

        top = [sentences[i] for i in sorted(top_indices, key=lambda i: sentences[i][2])]
        return " ".join(t[0] for t in top)

    # ── Sentence-intact chunking with spaCy (no regex splitting) ──────

    def _split_into_sentence_chunks(self, text: str, max_chars: int) -> List[str]:
        """
        Split text into chunks of at most `max_chars` characters,
        keeping sentences intact. Overlap last sentence to next chunk.

        Uses spaCy for proper sentence boundaries instead of regex.
        """
        doc = self._nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        chunks = []
        current_chunk = ""
        overlap = ""

        for sent in sentences:
            if current_chunk and len(current_chunk) + len(sent) + 1 > max_chars:
                chunks.append(current_chunk.strip())
                current_chunk = (overlap + " " + sent) if overlap else sent
            else:
                if current_chunk:
                    current_chunk += " " + sent
                else:
                    current_chunk = sent
            overlap = sent

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # ── Safe fallback ─────────────────────────────────────────────────

    @staticmethod
    def _safe_fallback(text: str, max_chars: int = 500) -> str:
        """
        Return a safe fallback when summarization produces nothing valid.
        Never returns the full unprocessed text.
        """
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        # Try to end at a sentence boundary
        last_boundary = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        if last_boundary > max_chars // 2:
            return truncated[:last_boundary + 1]
        return truncated


# ── Module-level convenience (default Summarizer) ────────────────────────────
_summarizer = None
_summarizer_lock = threading.Lock()

def _get_summarizer() -> Summarizer:
    global _summarizer
    if _summarizer is None:
        with _summarizer_lock:
            if _summarizer is None:
                _summarizer = Summarizer()
    return _summarizer


def is_summarizer_available() -> bool:
    return _get_summarizer().available


def summarize(text: str, max_sentences: int = DEFAULT_SENTENCE_COUNT) -> str:
    return _get_summarizer().summarize(text, max_sentences)


def summarize_by_target(text: str, target_chars: int = 500) -> str:
    return _get_summarizer().summarize_by_target(text, target_chars)


def summarize_long(
    text: str,
    max_sentences: int = DEFAULT_SENTENCE_COUNT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    return _get_summarizer().summarize_long(text, max_sentences)


def summarize_for_conversation(text: str) -> str:
    return _get_summarizer().summarize_for_conversation(text)


__all__ = [
    "Summarizer",
    "is_summarizer_available",
    "summarize",
    "summarize_by_target",
    "summarize_long",
    "summarize_for_conversation",
]