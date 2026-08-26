"""
Entity Graph Pipeline — entry point.

Standalone experimental module. Not wired into PipelineOrchestrator.

Usage:
    from core.Pipeline.Ingestion.Graph.main import build_entity_graph
    from core.Pipeline.Ingestion.Graph.GraphConfig import GraphConfig

    config = GraphConfig(min_mentions=2, wiki_tier=2)
    result = build_entity_graph(chunks, config, store, "filename.pdf")
"""

from typing import List, Any, Optional
from pathlib import Path

import pandas as pd

from core.Logger.DarwinLogger import darwin
from core.Pipeline.Ingestion.Graph.GraphConfig import GraphConfig
from core.Pipeline.Ingestion.Graph.EntityExtractor import extract_entities
from core.Pipeline.Ingestion.Graph.GraphBuilder import build_graph, GraphResult
from core.Pipeline.Ingestion.Graph.WikiBuilder import enrich_entities


def build_entity_graph(
    chunks: List[Any],
    config: Optional[GraphConfig] = None,
    store: Any = None,
    source_name: str = "",
) -> Optional[GraphResult]:
    """
    Full pipeline: extract entities -> build graph -> enrich wiki.

    This is the main public API. Call this directly — do not wire into
    PipelineOrchestrator until this is no longer experimental.

    Args:
        chunks: List of Chunk objects from the chunking stage.
        config: Graph configuration. Uses defaults if None.
        store: Optional DarwinStore for saving results.
        source_name: Document identifier for storage.

    Returns:
        GraphResult with nodes and edges DataFrames, or None on failure.
    """
    if config is None:
        config = GraphConfig()

    if not chunks:
        darwin.warning("No chunks provided for graph extraction")
        return None

    darwin.info(f"Building entity graph from {len(chunks)} chunks")

    # ── Step 1: Extract entities ─────────────────────────────────────────
    entities = extract_entities(chunks, config)
    if not entities:
        darwin.warning("No entities extracted — graph will be empty")
        return GraphResult(
            nodes=pd.DataFrame(),
            edges=pd.DataFrame(),
            stats={"nodes": 0, "edges": 0, "status": "empty"},
        )

    # ── Step 2: Build graph ──────────────────────────────────────────────
    graph_result = build_graph(entities, chunks, config)

    # ── Step 3: Enrich with wiki summaries ───────────────────────────────
    if not graph_result.nodes.empty and config.wiki_tier >= 1:
        graph_result.nodes = enrich_entities(
            graph_result.nodes, entities, chunks, config
        )

    # ── Step 4: Generate visualization ───────────────────────────────────
    if config.visualization and not graph_result.nodes.empty:
        graph_result = _add_visualization(graph_result)

    # ── Step 5: Save to store ────────────────────────────────────────────
    if store is not None and source_name:
        _save_to_store(store, source_name, graph_result)

    darwin.success(
        f"Entity graph complete: {graph_result.stats.get('nodes', 0)} nodes, "
        f"{graph_result.stats.get('edges', 0)} edges"
    )
    return graph_result


def _add_visualization(result: GraphResult) -> GraphResult:
    """Generate interactive HTML visualization with pyvis."""
    try:
        from pyvis.network import Network

        net = Network(
            height="750px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#333333",
            directed=False,
            notebook=False,
            cdn_resources="remote",
        )

        label_colors = {
            "PERSON": "#4a90d9",
            "ORG": "#e74c3c",
            "GPE": "#2ecc71",
            "LOC": "#27ae60",
            "EVENT": "#f39c12",
            "PRODUCT": "#9b59b6",
            "NORP": "#e67e22",
            "FAC": "#1abc9c",
            "WORK_OF_ART": "#e91e63",
            "LAW": "#607d8b",
            "LANGUAGE": "#795548",
        }

        for _, row in result.nodes.iterrows():
            node_id = row["id"]
            name = row["name"]
            label = row["label"]
            salience = row.get("salience", 0.1)
            mentions = row.get("mention_count", 1)
            wiki = row.get("wiki_summary", "")

            size = max(10, min(50, salience * 40 + mentions * 2))
            color = label_colors.get(label, "#95a5a6")

            title = f"<b>{name}</b><br>Type: {label}<br>Mentions: {mentions}<br>Salience: {salience:.3f}"
            if wiki:
                title += f"<br><br>{wiki}"

            net.add_node(
                node_id,
                label=name,
                title=title,
                size=size,
                color=color,
                group=label,
            )

        for _, row in result.edges.iterrows():
            weight = max(1, row.get("weight", 0.1) * 5)
            evidence = row.get("evidence", [])
            title = f"{row['relation']}<br>Weight: {row.get('weight', 0):.3f}"
            if evidence:
                title += "<br><br>" + "<br>".join(evidence[:3])

            net.add_edge(
                row["source"],
                row["target"],
                value=weight,
                title=title,
                label=row.get("relation", ""),
            )

        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.005,
              "springLength": 150,
              "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 200}
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)

        result.stats["pyvis_html"] = net
        result.stats["visualization"] = True

    except ImportError:
        darwin.warning("pyvis not installed — skipping visualization")
    except Exception as e:
        darwin.warning(f"Visualization generation failed: {e}")

    return result


def _save_to_store(store: Any, source_name: str, result: GraphResult) -> None:
    """Save graph results to the store if GraphStore is available."""
    try:
        from core.Storage.GraphStore import GraphStore

        gs = GraphStore(store)
        gs.save_graph(source_name, result)
    except ImportError:
        darwin.debug("GraphStore not available — skipping save")
    except Exception as e:
        darwin.warning(f"Failed to save graph to store: {e}")
