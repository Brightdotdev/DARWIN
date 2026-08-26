"""
Entity extraction from chunks using spaCy NER + noun chunks.

Uses ModelManager to load the spaCy model, then runs extract_entities_raw()
for co-extraction of NER entities and noun chunks with deduplication.
"""

from typing import List, Dict, Any

from core.Logger.DarwinLogger import darwin
from core.Models.Infrence.Spacy.lib import EntityRecord, Mention
from core.Models.Infrence.Spacy.Utils import (
    get_base_name, get_display_name,
    compute_entity_signal,
)
from core.Pipeline.Ingestion.Graph.GraphConfig import GraphConfig


def extract_entities(
    chunks: List[Any],
    config: GraphConfig,
) -> List[EntityRecord]:
    """
    Extract and deduplicate entities from a list of Chunks.

    Returns a list of EntityRecord objects sorted by salience (descending).
    """
    if not chunks:
        return []

    spacy_wrapper = _load_spacy()
    if spacy_wrapper is None:
        return []

    raw_entities = list(spacy_wrapper.extract_entities_raw(
        texts=(c.clean for c in chunks),
        batch_size=64,
    ))

    if not raw_entities:
        darwin.info("No entities extracted from chunks")
        return []

    merged = _merge_entities(raw_entities, chunks, config)
    scored = _score_entities(merged, len(chunks), config)

    darwin.info(f"Extracted {len(scored)} unique entities from {len(chunks)} chunks")
    return scored


def _load_spacy():
    """Load spaCy model via ModelManager."""
    try:
        from core.Models.ModelManager import modelManager
        from core.Models.Lib.ModelRegistry import ModelCategory

        wrapper = modelManager.get(ModelCategory.NLP)
        if wrapper is None:
            darwin.error("spaCy model not available for entity extraction")
            return None
        return wrapper
    except Exception as e:
        darwin.error(f"Failed to load spaCy for entity extraction: {e}")
        return None


def _merge_entities(
    raw_entities: List[Dict[str, Any]],
    chunks: List[Any],
    config: GraphConfig,
) -> Dict[str, EntityRecord]:
    """Merge raw extracted entities into deduplicated EntityRecord objects."""
    records: Dict[str, EntityRecord] = {}

    chunk_source_map = {i: c.source for i, c in enumerate(chunks)}

    for item in raw_entities:
        text = item.get("text", "")
        label = item.get("label", "")
        source = item.get("source", "ner")

        if config.entity_types and label not in config.entity_types:
            continue

        base = get_base_name(text)
        if not base:
            continue

        if config.label_filter and label not in config.label_filter:
            continue

        if base not in records:
            records[base] = EntityRecord(
                base_name=base,
                display_name=get_display_name(text),
                type=label,
            )

        rec = records[base]

        chunk_idx = _find_chunk_index(item, chunks)
        doc_id = chunk_source_map.get(chunk_idx, "")

        mention = Mention(
            text=text,
            chunk_id=chunk_idx,
            span=(item.get("start", 0), item.get("end", 0)),
            doc_id=doc_id,
            source=source,
        )
        rec.mentions.append(mention)
        rec.variant_names.add(text)

        if label and rec.type != label:
            rec.type_confidence[label] = rec.type_confidence.get(label, 0) + 1
            dominant = max(rec.type_confidence, key=rec.type_confidence.get)
            rec.type = dominant

    return records


def _find_chunk_index(item: Dict[str, Any], chunks: List[Any]) -> int:
    """Find the chunk index from raw entity output."""
    for i, chunk in enumerate(chunks):
        if hasattr(chunk, "clean") and chunk.clean:
            if item.get("text", "") in chunk.clean:
                return i
    return 0


def _score_entities(
    records: Dict[str, EntityRecord],
    total_chunks: int,
    config: GraphConfig,
) -> List[EntityRecord]:
    """Compute salience scores and filter by thresholds."""
    scored = []
    for rec in records.values():
        if len(rec.mentions) < config.min_mentions:
            continue

        mention_dicts = [m.to_dict() for m in rec.mentions]
        signal = compute_entity_signal(mention_dicts, total_chunks)
        rec.salience = signal.get("signal", 0.0)
        rec.quality_signal = rec.salience

        if rec.salience < config.min_salience:
            continue

        scored.append(rec)

    scored.sort(key=lambda r: r.salience, reverse=True)
    return scored
