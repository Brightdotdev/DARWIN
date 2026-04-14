"""
src/core/Embedding/EmbedderUtils.py
-----------------------------------
Utility functions and configuration for embedding models.

Provides:
- model registry and resolution
- lazy model loading with caching
- e5 prefix helpers for correct embedding usage

Model Registry
--------------
Maps short names to HuggingFace model IDs:

    "e5-small" → "intfloat/e5-small-v2"

Allows configuration-level abstraction so the system can switch models
without changing embedding code.

If a model name is not found in the registry, it is treated as a full
HuggingFace model ID.

e5 Prefixing (Critical)
----------------------
e5 models require task-specific prefixes:

    prefix_passage(text)
        → "passage: {text}"
        Used for indexing document chunks

    prefix_query(text)
        → "query: {text}"
        Used for embedding user queries

This asymmetry is intentional and required for proper semantic search.

Using incorrect or missing prefixes will significantly degrade
retrieval performance.

Model Loading
-------------
load_model(model_id):
- lazily loads SentenceTransformer models
- caches them in-memory per session
- avoids repeated initialization cost

Cache key:
    model_id (string)

Constants
---------
DEFAULT_MODEL
    Default HuggingFace model ID

DEFAULT_BATCH_SIZE
    Default batch size for embedding

MODEL_REGISTRY
    Mapping of short names → model IDs

Notes
-----
- Uses sentence-transformers backend
- Requires `pip install sentence-transformers`
- Designed to be lightweight and dependency-isolated
"""

from typing import Optional
from dataclasses import dataclass
from src.core.Embedding.EmbeddingModel import (DEFAULT_BATCH_SIZE,DEFAULT_MODEL)


# ============================================================================
# Args
# ============================================================================

@dataclass
class EmbeddingArgs:
    target:     str
    file:       Optional[str] = None
    project:    Optional[str] = None
    model:       str  = DEFAULT_MODEL
    batch_size: int  = DEFAULT_BATCH_SIZE
    inspect:    bool = False
    rerun:      bool = False

# ============================================================================
# e5 prefix helpers
# These must be used consistently — passages during indexing,
# queries during search. Asymmetric by design.
# ============================================================================

def prefix_passage(text: str) -> str:
    """Prepend passage prefix for e5 models. Use when embedding chunks."""
    return f"passage: {text}"


def prefix_query(text: str) -> str:
    """Prepend query prefix for e5 models. Use when embedding search queries."""
    return f"query: {text}"

