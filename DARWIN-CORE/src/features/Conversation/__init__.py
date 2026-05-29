"""
src/features/Conversation/ConversationRAG/
-------------------------------------------
Submodule for conversation cold storage: chunk, embed, index, search, contradict.
Uses existing DarwinStore and Embedder pipeline.
"""

from .types import ConvoChunk, MemoryDocument, Contradiction, SearchResult, SearchFilters
from .chunker import chunk_conversation
from .indexer import ConvoIndexer
from .search import ConvoSearch
from .mixins import ConvoEmbeddingsMixin, ConvoStoreMixin

__all__ = [
    "ConvoChunk",
    "MemoryDocument", 
    "Contradiction",
    "SearchResult",
    "SearchFilters",
    "chunk_conversation",
    "ConvoIndexer",
    "ConvoSearch",
    "ConvoEmbeddingsMixin",
    "ConvoStoreMixin",
]