from dataclasses import dataclass, field
from typing import Literal, Set, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Document-type presets
# ═══════════════════════════════════════════════════════════════════════════

PRESETS: dict[str, dict] = {
    "academic": {
        # Research papers, arXiv preprints, conference proceedings
        "processing_mode": "sliding_window",
        "chunk_size": 512,
        "chunk_overlap": 0.3,
        "max_concepts": 50,
        "min_freq": 1,
        "relation_confidence_threshold": 0.0,  # keep all explicit SVOs
        "dedup_strategy": "svo_canonical",
        "similarity_threshold": 0.2,
        "max_cooccurrence_edges": 300,
        "require_typed_entities": True,
        "require_parser": True,
        "batch_size_spacy": 128,
    },
    "research": {
        # Deep research reports, white papers, technical docs
        "processing_mode": "chunked",
        "chunk_size": 1024,
        "chunk_overlap": 0.2,
        "max_concepts": 75,
        "min_freq": 2,
        "relation_confidence_threshold": 0.0,
        "dedup_strategy": "svo_canonical",
        "similarity_threshold": 0.15,
        "max_cooccurrence_edges": 500,
        "require_typed_entities": True,
        "require_parser": True,
        "batch_size_spacy": 64,
    },
    "legal": {
        # Contracts, court opinions, statutes
        "processing_mode": "full",
        "chunk_size": 2048,
        "chunk_overlap": 0.0,
        "max_concepts": 40,
        "min_freq": 3,
        "relation_confidence_threshold": 1.0,  # require multiple mentions
        "dedup_strategy": "strict",
        "similarity_threshold": 0.3,
        "max_cooccurrence_edges": 200,
        "require_typed_entities": True,
        "require_parser": True,
        "batch_size_spacy": 32,
    },
    "narrative": {
        # Books, biographies, news articles, biblical texts
        "processing_mode": "sliding_window",
        "chunk_size": 512,
        "chunk_overlap": 0.4,
        "max_concepts": 100,
        "min_freq": 2,
        "relation_confidence_threshold": 0.3,
        "dedup_strategy": "svo_canonical",
        "similarity_threshold": 0.25,
        "max_cooccurrence_edges": 600,
        "require_typed_entities": True,
        "require_parser": True,
        "batch_size_spacy": 128,
    },
    "technical": {
        # API docs, READMEs, engineering specs
        "processing_mode": "chunked",
        "chunk_size": 256,
        "chunk_overlap": 0.1,
        "max_concepts": 60,
        "min_freq": 1,
        "relation_confidence_threshold": 0.0,
        "dedup_strategy": "svo_canonical",
        "similarity_threshold": 0.1,
        "max_cooccurrence_edges": 400,
        "require_typed_entities": False,  # noun chunks often better for code/docs
        "require_parser": True,
        "batch_size_spacy": 128,
    },
}


def get_preset(name: str) -> dict:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return dict(PRESETS[name])


@dataclass
class GraphExtractionConfig:
    """
    Configuration for document-agnostic knowledge graph extraction.

    All fields are tunable. Regenerate graphs with different configs
    until best fit is achieved for your document type.
    """

    # ═══════════════════════════════════════════════════════════════════════
    # Processing mode
    # ═══════════════════════════════════════════════════════════════════════
    processing_mode: Literal["auto", "full", "chunked", "sliding_window"] = "auto"
    """auto: choose based on token count; full: always full-file; chunked: fixed chunks; sliding_window: 50% overlap"""

    chunk_size: int = 512
    """Token count per chunk (for chunked/sliding_window modes)"""

    chunk_overlap: float = 0.2
    """Fraction of overlap between chunks (0.0-1.0). 0.5 for sliding_window."""

    # ═══════════════════════════════════════════════════════════════════════
    # Graph size control
    # ═══════════════════════════════════════════════════════════════════════
    max_concepts: int = 75
    """Hard ceiling on nodes. Drives Pareto pruning."""

    min_freq: int = 2
    """Minimum chunk occurrences for an entity to be included."""

    # ═══════════════════════════════════════════════════════════════════════
    # Relationship extraction
    # ═══════════════════════════════════════════════════════════════════════
    relation_confidence_threshold: float = 0.0
    """Relation confidence threshold. 0.0=keep all explicit SVOs (good for papers), 0.6=moderate, 1.0+=require multiple mentions."""
    """Minimum chunk-vote fraction to keep a relation (0.0-1.0)."""

    dedup_strategy: Literal["svo_canonical", "strict", "none"] = "svo_canonical"
    """svo_canonical: lemmatize+sort SVO; strict: exact match only; none: keep all."""

    similarity_threshold: Optional[float] = None
    """Cosine floor for co-occurrence edges (None = auto-calculated)."""

    max_cooccurrence_edges: int = 500
    """Hard cap on co-occurrence edges to prevent graph explosion."""

    # ═══════════════════════════════════════════════════════════════════════
    # Stopwords
    # ═══════════════════════════════════════════════════════════════════════
    stopword_source: Literal["spacy", "custom"] = "spacy"
    """spacy: use spaCy's built-in is_stop; custom: use provided set."""

    custom_stopwords: Set[str] = field(default_factory=set)
    """Additional stopwords beyond spaCy's defaults."""

    # ═══════════════════════════════════════════════════════════════════════
    # spaCy requirements
    # ═══════════════════════════════════════════════════════════════════════
    require_typed_entities: bool = True
    """Fail if spaCy NER is unavailable. If False, falls back to noun chunks."""

    require_parser: bool = True
    """Fail if spaCy dependency parser is unavailable. If False, no SVO relations."""

    # ═══════════════════════════════════════════════════════════════════════
    # Performance
    # ═══════════════════════════════════════════════════════════════════════
    batch_size_spacy: int = 64
    """Batch size for spaCy nlp.pipe()."""

    batch_size_embed: int = 64
    """Batch size for embedding model encode()."""

    # ═══════════════════════════════════════════════════════════════════════
    # Structural metadata (for wiki generation)
    # ═══════════════════════════════════════════════════════════════════════
    capture_structural_order: bool = True
    """Track first_mention_chunk_idx per entity."""

    # ═══════════════════════════════════════════════════════════════════════
    # Legacy compatibility
    # ═══════════════════════════════════════════════════════════════════════
    visualize: bool = False
    """Generate HTML visualization."""

    inspect: bool = False
    """Print graph details to log."""

    json_output: bool = False
    """Export JSON to stdout."""

    rerun: bool = False
    """Force re-extraction even if fresh graph exists."""


    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        overrides: dict = None,
    ) -> "GraphExtractionConfig":
        """
        Create config from a named preset with optional overrides.

        Args:
            preset_name: One of: academic, research, legal, narrative, technical
            overrides: Dict of field names to override preset values

        Example:
            config = GraphExtractionConfig.from_preset("academic", {"max_concepts": 30})
        """
        preset = get_preset(preset_name)
        if overrides:
            preset.update(overrides)

        # Extract known fields, pass rest as overrides
        known_fields = {
            "processing_mode", "chunk_size", "chunk_overlap", "max_concepts",
            "min_freq", "relation_confidence_threshold", "dedup_strategy",
            "similarity_threshold", "max_cooccurrence_edges", "stopword_source",
            "custom_stopwords", "require_typed_entities", "require_parser",
            "batch_size_spacy", "batch_size_embed", "capture_structural_order",
            "visualize", "inspect", "json_output", "rerun",
        }

        kwargs = {k: v for k, v in preset.items() if k in known_fields}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        """Serialize for logging/debugging."""
        return {
            k: (list(v) if isinstance(v, set) else v)
            for k, v in self.__dict__.items()
        }