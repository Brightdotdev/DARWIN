"""
Build entity graph nodes and edges from extracted entities.

Produces DataFrames for nodes (entities) and edges (relationships)
with co-occurrence, dependency, and document-level connections.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import math

import pandas as pd

from core.Logger.DarwinLogger import darwin
from core.Models.Infrence.Spacy.lib import EntityRecord
from core.Pipeline.Ingestion.Graph.GraphConfig import GraphConfig


@dataclass
class GraphResult:
    """Result of graph construction: nodes and edges as DataFrames."""
    nodes: pd.DataFrame
    edges: pd.DataFrame
    stats: Dict[str, Any] = field(default_factory=dict)


def build_graph(
    entities: List[EntityRecord],
    chunks: List[Any],
    config: GraphConfig,
) -> GraphResult:
    """
    Build a graph from extracted entities and chunks.

    Creates:
      - Node DataFrame with entity attributes
      - Edge DataFrame with co-occurrence + dependency relationships
    """
    if not entities:
        return GraphResult(
            nodes=pd.DataFrame(),
            edges=pd.DataFrame(),
            stats={"nodes": 0, "edges": 0},
        )

    nodes_df = _build_nodes(entities)
    edges_df = _build_edges(entities, chunks, config)

    stats = {
        "nodes": len(nodes_df),
        "edges": len(edges_df),
        "entity_types": nodes_df["label"].value_counts().to_dict() if len(nodes_df) > 0 else {},
        "edge_types": edges_df["relation"].value_counts().to_dict() if len(edges_df) > 0 else {},
    }

    darwin.info(f"Built graph: {stats['nodes']} nodes, {stats['edges']} edges")
    return GraphResult(nodes=nodes_df, edges=edges_df, stats=stats)


def _build_nodes(entities: List[EntityRecord]) -> pd.DataFrame:
    """Build the node DataFrame from EntityRecord objects."""
    rows = []
    for rec in entities:
        sources = sorted(rec.doc_ids)
        rows.append({
            "id": rec.entity_id,
            "name": rec.display_name,
            "label": rec.type,
            "salience": round(rec.salience, 4),
            "mention_count": rec.frequency,
            "chunk_count": len(rec.chunk_ids),
            "source_count": rec.spread,
            "sources": sources,
            "variants": sorted(rec.variant_names)[:10],
            "wiki_summary": "",
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.drop_duplicates(subset=["id"], keep="first")
    return df


def _build_edges(
    entities: List[EntityRecord],
    chunks: List[Any],
    config: GraphConfig,
) -> pd.DataFrame:
    """Build the edge DataFrame from entity co-occurrences."""
    edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

    chunk_entity_map = _map_chunks_to_entities(entities, chunks)

    for chunk_idx, entity_ids in chunk_entity_map.items():
        entity_list = sorted(entity_ids)
        for i, id_a in enumerate(entity_list):
            for id_b in entity_list[i + 1:]:
                key = (id_a, id_b)
                if key not in edges:
                    edges[key] = {
                        "source": id_a,
                        "target": id_b,
                        "relation": "CO_OCCURS",
                        "weight": 0.0,
                        "chunk_ids": [],
                        "evidence": [],
                    }
                edge = edges[key]
                edge["chunk_ids"].append(chunk_idx)
                edge["weight"] += 1.0

    if config.include_dependency_edges:
        dep_edges = _extract_dependency_edges(entities, chunks)
        for dep_edge in dep_edges:
            key = (dep_edge["source"], dep_edge["target"])
            if key in edges:
                existing = edges[key]
                if existing["relation"] != "DEPENDENCY":
                    existing["relation"] = "CO_OCCURS+DEPENDENCY"
                existing["weight"] += dep_edge["weight"]
                existing["evidence"].extend(dep_edge.get("evidence", []))
            else:
                edges[key] = dep_edge

    normalized = _normalize_edges(edges, entities)

    if not normalized:
        return pd.DataFrame(columns=["source", "target", "relation", "weight", "chunk_ids", "evidence"])

    df = pd.DataFrame(normalized)
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    return df


def _map_chunks_to_entities(
    entities: List[EntityRecord],
    chunks: List[Any],
) -> Dict[int, Set[str]]:
    """Map each chunk index to the set of entity IDs it contains."""
    mapping: Dict[int, Set[str]] = defaultdict(set)
    for rec in entities:
        for mention in rec.mentions:
            mapping[mention.chunk_id].add(rec.entity_id)
    return mapping


def _extract_dependency_edges(
    entities: List[EntityRecord],
    chunks: List[Any],
) -> List[Dict[str, Any]]:
    """Extract syntactic dependency edges between entities in the same chunk."""
    edges = []
    try:
        from core.Models.ModelManager import modelManager
        from core.Models.Lib.ModelRegistry import ModelCategory

        wrapper = modelManager.get(ModelCategory.NLP)
        if wrapper is None or not wrapper.is_loaded():
            return edges

        entity_positions = {}
        for rec in entities:
            for mention in rec.mentions:
                if mention.chunk_id not in entity_positions:
                    entity_positions[mention.chunk_id] = []
                entity_positions[mention.chunk_id].append({
                    "id": rec.entity_id,
                    "start": mention.span[0],
                    "end": mention.span[1],
                })

        for chunk_idx, ents in entity_positions.items():
            if chunk_idx >= len(chunks):
                continue
            chunk = chunks[chunk_idx]
            text = chunk.clean if hasattr(chunk, "clean") else str(chunk)

            try:
                doc = wrapper._nlp(text)
                ent_spans = [(e["id"], e["start"], e["end"]) for e in ents]

                for token in doc:
                    if token.dep_ in ("nsubj", "dobj", "pobj", "appos"):
                        head = token.head
                        for ent_id_a, start_a, end_a in ent_spans:
                            if start_a <= token.idx < end_a:
                                for ent_id_b, start_b, end_b in ent_spans:
                                    if start_b <= head.idx < end_b and ent_id_a != ent_id_b:
                                        edges.append({
                                            "source": ent_id_a,
                                            "target": ent_id_b,
                                            "relation": "DEPENDENCY",
                                            "weight": 0.5,
                                            "chunk_ids": [chunk_idx],
                                            "evidence": [f"{token.text} --{token.dep_}--> {head.text}"],
                                        })
            except Exception:
                continue

    except Exception as e:
        darwin.warning(f"Dependency edge extraction failed: {e}")

    return edges


def _normalize_edges(
    edges: Dict[Tuple[str, str], Dict[str, Any]],
    entities: List[EntityRecord],
) -> List[Dict[str, Any]]:
    """Normalize edge weights to 0-1 range and add evidence from chunk text."""
    if not edges:
        return []

    max_weight = max(e["weight"] for e in edges.values())
    if max_weight <= 0:
        max_weight = 1.0

    entity_map = {r.entity_id: r for r in entities}
    result = []

    for edge in edges.values():
        normalized_weight = min(edge["weight"] / max_weight, 1.0)

        if normalized_weight < 0.05:
            continue

        chunk_ids = edge.get("chunk_ids", [])
        evidence = edge.get("evidence", [])

        if not evidence and chunk_ids:
            for cid in chunk_ids[:3]:
                rec_a = entity_map.get(edge["source"])
                rec_b = entity_map.get(edge["target"])
                if rec_a and rec_b:
                    for m in rec_a.mentions:
                        if m.chunk_id == cid:
                            evidence.append(m.text)
                            break

        result.append({
            "source": edge["source"],
            "target": edge["target"],
            "relation": edge["relation"],
            "weight": round(normalized_weight, 4),
            "chunk_ids": sorted(chunk_ids),
            "evidence": evidence[:5],
        })

    return result
