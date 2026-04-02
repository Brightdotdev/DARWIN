"""
darwin.core.embedder
--------------------
Converts chunks into dense vector embeddings using e5-small-v2.

We build this as a class so later we can swap the model easily
(e5-small → BGE-M3 → GritLM) without touching anything else in Darwin.

e5-small quirks to know:
  - needs "passage: " prefix on things you're storing
  - needs "query: " prefix on things you're searching with
  - normalize_embeddings=True turns cosine into a dot product (faster)
  - 512 token limit — our chunker keeps us well under this
"""

import numpy as np
from dataclasses import dataclass, field

from src.core.Chunker import Chunk



# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class EmbeddedChunk:
    chunk:     Chunk            # original chunk with text + metadata
    embedding: np.ndarray       # the dense vector — shape (384,) for e5-small


@dataclass
class EmbeddingResult:
    embedded_chunks: list[EmbeddedChunk]
    model_name:      str
    dimension:       int         # vector size — 384 for e5-small
    total_chunks:    int


# ── Embedder ──────────────────────────────────────────────────────────────────

class Embedder:
    """
    Wraps sentence-transformers to embed Darwin chunks.

    Lazy loads the model on first use so importing this module
    doesn't slow down the CLI startup — model only loads when
    the user actually runs an embed operation.

    Usage:
        embedder = Embedder()
        result   = embedder.embed(chunks)

        # access embeddings
        for ec in result.embedded_chunks:
            print(ec.chunk.text)
            print(ec.embedding.shape)  # (384,)
    """

    DEFAULT_MODEL = "intfloat/e5-small-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model     = None   # lazy loaded on first embed call

    def _load_model(self):
        """Load model once and cache it on the instance."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, chunks: list[Chunk]) -> EmbeddingResult:
        """
        Embed a list of chunks into dense vectors.

        Adds the "passage: " prefix e5-small needs before encoding.
        Returns an EmbeddingResult with everything needed for the graph builder.

        Args:
            chunks: output from chunk_document()

        Returns:
            EmbeddingResult containing embedded chunks + metadata
        """
        if not chunks:
            return EmbeddingResult(
                embedded_chunks = [],
                model_name      = self.model_name,
                dimension       = 0,
                total_chunks    = 0,
            )

        model = self._load_model()

        # e5-small was trained with this prefix on stored passages
        # skipping it noticeably hurts retrieval quality
        prefixed_texts = [f"passage: {chunk.text}" for chunk in chunks]

        # normalize=True → vectors have length 1
        # cosine similarity then becomes a simple dot product
        # which is faster and numerically cleaner
        embeddings = model.encode(
            prefixed_texts,
            normalize_embeddings = True,
            show_progress_bar    = len(chunks) > 20,  # only show bar for large docs
            batch_size           = 32,
        )

        embedded_chunks = [
            EmbeddedChunk(chunk=chunk, embedding=embeddings[i])
            for i, chunk in enumerate(chunks)
        ]

        return EmbeddingResult(
            embedded_chunks = embedded_chunks,
            model_name      = self.model_name,
            dimension       = embeddings.shape[1],
            total_chunks    = len(chunks),
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string for similarity search.
        Uses "query: " prefix instead of "passage: " — e5-small
        was trained to distinguish between these two roles.

        Args:
            query: the user's question or search string

        Returns:
            normalized embedding vector, shape (384,)
        """
        model = self._load_model()
        embedding = model.encode(
            f"query: {query}",
            normalize_embeddings = True,
        )
        return embedding