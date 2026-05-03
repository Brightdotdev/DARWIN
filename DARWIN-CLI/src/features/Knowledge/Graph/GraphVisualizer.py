"""
src/core/Knowledge/Graph/GraphVisualizer.py
-------------------------------------------
Obsidian-style unipartite concept graph visualization.

Produces HTML with force-directed layout, clear node separation,
auto-threshold edge filtering, and query highlighting support.
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

    Obsidian-like feel:
      - Strong repulsion between all nodes (gravitationalConstant: -120)
      - Weak springs between connected nodes (springConstant: 0.04)
      - Long spring length for separation (springLength: 200)
      - High damping for quick stabilization

    Args:
        graph:             The concept graph
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

    net = Network(height=height, width="100%", bgcolor="#0f0f0f", font_color="#e0e0e0")

    # ── Obsidian-like physics ───────────────────────────────────────────
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
            "arrows": {"to": {"enabled": false}},
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
        edge_count = sum(1 for e in graph.edges
                        if e.source == node.id or e.target == node.id)

        # Color: more connected = warmer
        if node.id in highlighted:
            color = "#ffd700"
            border_width = 4
        elif edge_count >= 5:
            color = "#ff6b6b"
            border_width = 1.5
        elif edge_count >= 3:
            color = "#feca57"
            border_width = 1.5
        else:
            color = "#4ecdc4"
            border_width = 1.5

        net.add_node(
            node.id,
            label=node.label,
            title=f"<b>{node.label}</b><br>In {node.frequency} chunks<br>{edge_count} connections",
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

        if edge.source in highlighted and edge.target in highlighted:
            edge_color = f"rgba(255, 215, 0, {min(opacity + 0.3, 1.0)})"
            edge_width = width * 1.8
        else:
            edge_color = f"rgba(150, 180, 220, {opacity})"
            edge_width = width

        net.add_edge(
            edge.source, edge.target,
            value=w,
            width=edge_width,
            color={"color": edge_color, "highlight": "#ffffff"},
            title=f"Similarity: {w:.3f}",
        )

    return net.generate_html()