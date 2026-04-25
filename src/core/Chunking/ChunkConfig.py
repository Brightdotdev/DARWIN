"""
src/core/Chunking/ChunkConfig.py
---------------------------------
Single source of truth for embedding model configuration.

Covers:
  - Chunk sizing (max_tokens, min_tokens, overlap)
  - Cleaning mode (plain_text vs semantic_markdown)
  - Prefix/instruction templates per model
  - Hardware-aware batch sizing
  - Lazy model loading (delegates to ModelLoader)

No hardcoded model names anywhere else in the pipeline.
Everything auto-detects from the currently active model.
"""

from src.core.Models.EmbeddingConfig import get_preset,ModelPreset




# ── Shorthand for code that only needs chunk sizing ──

def get_chunk_config() -> ModelPreset:
    """Alias — returns the same thing as get_preset()."""
    return get_preset()


# ── Legacy dict-style access for backward compat ──

def get_model_config() -> dict:
    """Dict-style access for code expecting get_model_config()["max_tokens"]."""
    preset = get_preset()
    return {
        "max_tokens": preset.max_tokens,
        "min_tokens": preset.min_tokens,
        "overlap": preset.overlap,
        "category": preset.category,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Prefix / instruction formatting
# ═══════════════════════════════════════════════════════════════════════════

def format_passage(text: str) -> str:
    """Apply the model's passage prefix/instruction to text."""
    
    preset = get_preset()
    
    if preset.is_instruction_tuned and preset.instruction_template:
        return preset.instruction_template.format(text=text)
    
    if preset.needs_prefix:
        return preset.passage_prefix + text
    
    return text


def format_query(text: str) -> str:
    """Apply the model's query prefix to a search query."""
    
    preset = get_preset()
    
    if preset.needs_prefix:
        return preset.query_prefix + text
    
    return text

