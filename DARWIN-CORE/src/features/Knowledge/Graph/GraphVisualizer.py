"""
src/core/Knowledge/Graph/GraphVisualizer.py
-------------------------------------------
Obsidian-style knowledge graph visualization.

Updated for spaCy-based entity-relationship graphs.
Supports directed semantic edges + undirected co-occurrence edges.
"""

from typing import Optional, List
from src.features.Knowledge.Graph.GraphUtils import GraphData


def render_concept_graph(
    graph: GraphData,
    highlighted_nodes: List[str] = None,
    height: str = "700px",
) -> Optional[str]:
    """
    Build interactive HTML graph using pyvis.

    Distinguishes:
      - Semantic relations (directed, colored by relation type)
      - Co-occurrence edges (undirected, gray)

    Args:
        graph:             The knowledge graph
        highlighted_nodes: Node IDs to pulse/highlight (gold)
        height:            CSS container height

    Returns:
        HTML string or None if pyvis not available
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return None

    highlighted = set(highlighted_nodes or [])

    net = Network(height=height, width="100%", bgcolor="#0f0f0f", font_color="#e0e0e0", directed=True)

    # ── Physics ───────────────────────────────────────────────────────────
    net.set_options("""{
        "nodes": {
            "shape": "dot",
            "scaling": {
                "min": 8, "max": 40,
                "label": {"enabled": true, "min": 10, "max": 22}
            },
            "font": {"size": 13, "face": "Inter, system-ui, sans-serif", "strokeWidth": 0},
            "borderWidth": 1.5,
            "borderWidthSelected": 3,
            "shadow": {"enabled": true, "color": "rgba(0,0,0,0.4)", "size": 4}
        },
        "edges": {
            "smooth": {"type": "continuous", "roundness": 0.5},
            "arrows": {
                "to": {"enabled": true, "scaleFactor": 0.6}
            },
            "color": {"inherit": false}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -120,
                "centralGravity": 0.005,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.4,
                "avoidOverlap": 0.8
            },
            "minVelocity": 0.5,
            "maxVelocity": 25,
            "solver": "forceAtlas2Based",
            "timestep": 0.3,
            "stabilization": {"enabled": true, "iterations": 300, "updateInterval": 25}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 150,
            "hideEdgesOnDrag": false,
            "navigationButtons": true,
            "keyboard": true
        }
    }""")

    # ── Nodes ───────────────────────────────────────────────────────────
    max_freq = max((n.frequency for n in graph.nodes), default=1)

    for node in graph.nodes:
        size = max(8, min(40, (node.frequency / max_freq) * 35))

        # Count edge types
        semantic_count = sum(1 for e in graph.edges
                            if (e.source == node.id or e.target == node.id)
                            and e.relation_type != "co_occurrence")
        cooc_count = sum(1 for e in graph.edges
                        if (e.source == node.id or e.target == node.id)
                        and e.relation_type == "co_occurrence")

        # Color by entity type
        type_colors = {
            "PERSON": "#ff6b6b",
            "ORG": "#4ecdc4",
            "GPE": "#45b7d1",
            "PRODUCT": "#feca57",
            "EVENT": "#ff9ff3",
            "WORK_OF_ART": "#54a0ff",
            "LAW": "#5f27cd",
            "CONCEPT": "#96ceb4",
        }

        if node.id in highlighted:
            color = "#ffd700"
            border_width = 4
        else:
            color = type_colors.get(node.entity_type, "#a8a8a8")
            border_width = 1.5

        title = f"<b>{node.display_name}</b><br>"
        title += f"Type: {node.entity_type}<br>"
        title += f"In {node.frequency} chunks<br>"
        title += f"First @ chunk {node.first_mention_chunk}<br>"
        title += f"{semantic_count} semantic, {cooc_count} co-occurrence"

        net.add_node(
            node.id,
            label=node.display_name,
            title=title,
            size=size,
            color=color,
            borderWidth=border_width,
        )

    # ── Edges ───────────────────────────────────────────────────────────
    max_weight = max((e.weight for e in graph.edges), default=1.0)

    for edge in graph.edges:
        w = float(edge.weight)
        width = 0.3 + (w / max_weight) * 2.5
        opacity = 0.15 + (w / max_weight) * 0.6

        # Semantic vs co-occurrence styling
        if edge.relation_type == "co_occurrence":
            edge_color = f"rgba(150, 180, 220, {opacity})"
            dashes = True
            arrow_scale = 0
        else:
            # Semantic relation: color by type
            relation_colors = {
                "acquire": "#ff6b6b",
                "found": "#4ecdc4",
                "create": "#feca57",
                "lead": "#ff9ff3",
                "develop": "#54a0ff",
            }
            base_color = relation_colors.get(edge.relation_type, "#a8a8a8")
            edge_color = f"rgba({int(base_color[1:3], 16)}, {int(base_color[3:5], 16)}, {int(base_color[5:7], 16)}, {min(opacity + 0.2, 1.0)})"
            dashes = False
            arrow_scale = 0.6

        if edge.source in highlighted and edge.target in highlighted:
            edge_color = f"rgba(255, 215, 0, {min(opacity + 0.3, 1.0)})"
            width *= 1.8

        title = f"{edge.relation_type}<br>Confidence: {w:.3f}<br>Evidence: {len(edge.evidence_chunks)} chunks"

        net.add_edge(
            edge.source, edge.target,
            value=w,
            width=width,
            color={"color": edge_color, "highlight": "#ffffff"},
            title=title,
            dashes=dashes,
            arrows={"to": {"enabled": arrow_scale > 0, "scaleFactor": arrow_scale}},
        )

    return net.generate_html()