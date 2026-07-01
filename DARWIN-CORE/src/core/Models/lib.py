# src/core/Models/lib.py



REQUIRED_FILES: dict[str, list[str]] = {
    "sentence_transformers": [
        "config.json",
        "tokenizer_config.json",
        "sentence_bert_config.json",
    ],
    "embedding_st": [
        # SentenceTransformers-based embeddings (e.g. bge-m3)
        "config.json",
        "tokenizer_config.json",
        "sentence_bert_config.json",
    ],
    "embedding_hf": [
        # Raw HuggingFace AutoModel embeddings (e.g. e5-small-v2)
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ],
    "cross_encoder": [
        "config.json",
        "tokenizer_config.json",
    ],
    "transformers": [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "generation_config.json",
    ],
    
}

# ═══════════════════════════════════════════════════════════════════════════
# Download filters
# ═══════════════════════════════════════════════════════════════════════════





SPACY_URLS = {
    "en_core_web_md": [
        # Try newest versions first (compatible with spaCy 3.8)
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl",
    ],
    "en_core_web_lg": [
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl",
    ],
}

tarball_urls = {
    "en_core_web_md": [
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1.tar.gz",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0.tar.gz",
    ],
    "en_core_web_sm": [
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz",
    ],
    "en_core_web_lg": [
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1.tar.gz",
    ],
}