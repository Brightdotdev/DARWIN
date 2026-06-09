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





def get_current_embedding_model() -> str:
    from src.core.Models import ModelManager, ModelCategory
    loader = ModelManager.get_loader()
    return loader.get_model_key(ModelCategory.EMBEDDING)



# ── Shorthand for code that only needs chunk sizing ──

def get_chunk_config():
    from src.core.Models.EmbeddingConfig import get_preset
    """Alias — returns the same thing as get_preset()."""
    key = get_current_embedding_model()
    return get_preset(key)


# ── Legacy dict-style access for backward compat ──

def get_model_config() -> dict:
    """Dict-style access for code expecting get_model_config()["max_tokens"]."""
    from src.core.Models.EmbeddingConfig import get_preset
    key = get_current_embedding_model()
    preset = get_preset(key)
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
    from src.core.Models.EmbeddingConfig import get_preset
    key = get_current_embedding_model()
    preset = get_preset(key)
    
    if preset.is_instruction_tuned and preset.instruction_template:
        return preset.instruction_template.format(text=text)
    
    if preset.needs_prefix:
        return preset.passage_prefix + text
    
    return text

