from src.core.Models import get_loader, ModelManager, ModelCategory
from src.core.Storage.DarwinStore import DarwinStore

store = DarwinStore()
manager = ModelManager(darwin_store=store)

# ── ModelLoader (lazy-loading, caching) ───────────────────────────────────
loader = get_loader()

# Get models (auto-downloads if missing)
loader.get(ModelCategory.EMBEDDING)    # SentenceTransformer | HF wrapper
loader.get(ModelCategory.RERANKER)     # CrossEncoder
loader.get(ModelCategory.VISION)       # (processor, model, device)
loader.get(ModelCategory.TOKENIZER)    # SpacyConceptExtractor

# Info
loader.get_model_name(ModelCategory.EMBEDDING)   # "BAAI/bge-m3"
loader.get_model_key(ModelCategory.EMBEDDING)    # "BAAI/bge-m3"
loader.preferences                               # {"embedding": "...", ...}

# Lifecycle
loader.set_preference("embedding", "BAAI/bge-m3")           # Switch
loader.set_preference("embedding", "BAAI/bge-m3", cleanup_old=True)  # Switch + delete old
loader.unload(ModelCategory.EMBEDDING)                      # Free RAM

# Migration
loader.migrate_embeddings(store)                   # All stale files
loader.migrate_embeddings(store, filename="x.pdf") # Single file
loader.migrate_embeddings(store, force=True)       # Everything

# ── ModelManager (user orchestration) ────────────────────────────────────
manager.switch_embedding("BAAI/bge-m3")           # Auto-download + migrate
manager.switch_embedding("BAAI/bge-m3", migrate=False)  # No migration
manager.switch_embedding("BAAI/bge-m3", file="x.pdf")   # Migrate one file

manager.switch_reranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
manager.switch_vision("vikhyatk/moondream2")
manager.switch_tokenizer("en_core_web_lg")

# Migration only
manager.migrate_embedding()                    # All stale files
manager.migrate_embedding(file="x.pdf")        # Single file
manager.migrate_embedding(force=True)          # Everything

# Installation
manager.install("BAAI/bge-m3")                 # Download only (not active)
manager.install_category("embedding", tier="light")
manager.uninstall("BAAI/bge-m3")

# Setup
manager.setup_required()                       # Install required models
manager.setup_recommended()                    # Install recommended based on hardware

# Status
manager.status()                               # Show installed models
manager.list_models(category="embedding")      # List available
manager.upgrade("BAAI/bge-m3")                 # Force re-download

# Hardware validation
from src.core.Models import validate_model, validate_category
validate_model("BAAI/bge-m3", system, store).can_run
validate_category("embedding", system, store)

# ── CLI Usage ────────────────────────────────────────────────────────────
# python -m src.core.Models.cli status
# python -m src.core.Models.cli install --embedding
# python -m src.core.Models.cli install --category embedding --tier light
# python -m src.core.Models.cli switch --category embedding --model BAAI/bge-m3
# python -m src.core.Models.cli setup-custom  # Interactive onboarding