# src/core/Models/lib.py

# ═══════════════════════════════════════════════════════════════════════════
# Required files per loader type (minimum for inference)
# ═══════════════════════════════════════════════════════════════════════════

# FIX: Added "embedding_hf" and "embedding_st" entries that were previously
# missing, causing is_downloaded() to fall through to a loose config.json
# check that could pass prematurely.
#
# FIX: Removed "spacy" entry — it was dead code. ModelStore.is_downloaded()
# delegates spaCy verification to _verify_spacy_model(), never reaching
# REQUIRED_FILES for that loader. Having it here was misleading.

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
    # "spacy" intentionally omitted — verified by ModelStore._verify_spacy_model()
}

# ═══════════════════════════════════════════════════════════════════════════
# Download filters
# ═══════════════════════════════════════════════════════════════════════════

ALLOW_PATTERNS = [
    "*.safetensors",
    "pytorch_model*.bin",
    "config.json",
    "preprocessor_config.json",
    "generation_config.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "added_tokens.json",
    "image_processor.json",
    "processor_config.json",
    "modules.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "chat_template.json",
]

IGNORE_PATTERNS = [
    "onnx/**", "onnx/*", "*.onnx", "*.h5", "*.keras",
    "tf_model.h5", "tf_model/**", "*.msgpack",
    "flax_model.msgpack", "flax_model/**",
    "*.ot", "rust_model.ot", "rust_model/**",
    "*.gguf", "ggml-model-*",
    "coreml/**", "coreml/*", "*.mlmodel", "*.mlpackage",
    "README.md", "*.md", ".gitattributes", ".gitignore", ".git/**",
    "checkpoint-*/**", "*.ckpt",
    "optimizer.pt", "scheduler.pt",
    "eval/**", "test/**", "tests/**", "*.ipynb", "runs/**",
    "*.tar.gz", "*.tgz", "*.zip",
]


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