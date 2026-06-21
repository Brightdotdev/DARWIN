"""
Darwin Wiki / Graph Builder (v3.3)
Entity co-occurrence graph generation using Pyvis.

Pyvis handles all vis.js embedding, HTML generation, and interactivity.
No manual HTML templates, no CDN dependencies, no inline JS.
"""

from typing import List, Dict, Any, Optional
from pyvis.network import Network
from src.features.Knowledge.WikiIndex import WikiIndex


# ── Color palette by entity type ───────────────────────────────────────

ENTITY_COLORS = {
    "PERSON":      "#e67e22",
    "ORG":         "#3498db",
    "GPE":         "#2ecc71",
    "LOC":         "#2ecc71",
    "WORK_OF_ART": "#9b59b6",
    "NORP":        "#f1c40f",
    "FAC":         "#1abc9c",
    "EVENT":       "#e74c3c",
    "LAW":         "#95a5a6",
    "LANGUAGE":    "#af7ac5",
    "PRODUCT":     "#d35400",
    "DATE":        "#34495e",
    "TIME":        "#7f8c8d",
    "MONEY":       "#27ae60",
    "PERCENT":     "#16a085",
    "QUANTITY":    "#8e44ad",
}

DEFAULT_COLOR = "#4a90d9"
BG_COLOR = "#1e1e1e"
FONT_COLOR = "#e0e0e0"


class GraphBuilder:
    """Build entity co-occurrence graphs using Pyvis."""

    def __init__(self, wiki_index: WikiIndex):
        self.wiki = wiki_index

    def build_graph_data(self, filename: str,
                         max_entities: int = 100,
                         min_cooccurrence: float = 0.1,
                         max_edges_per_node: int = 8) -> Dict[str, list]:
        """Returns filtered nodes and edges as plain dicts."""
        entities = self.wiki.list_entities(
            filenames=[filename], sort_by="salience", limit=max_entities
        )
        if not entities:
            return {"nodes": [], "edges": []}

        nodes = self._make_nodes(entities)
        edge_candidates = self._all_edge_candidates(entities, filename)
        edges = self._filter_edges(edge_candidates, min_cooccurrence, max_edges_per_node)
        return {"nodes": nodes, "edges": edges}

    def build_interactive_html(self, filename: str,
                               max_entities: int = 100,
                               title: str = "Entity Graph") -> str:
        """
        Generate an interactive HTML graph using Pyvis.

        Pyvis handles all vis.js bundling, physics, and UI controls.
        The output is a self-contained HTML file with no external dependencies.
        """
        entities = self.wiki.list_entities(
            filenames=[filename], sort_by="salience", limit=max_entities
        )
        if not entities:
            return "<html><body>No entities found.</body></html>"

        # Build the Pyvis network
        net = Network(
            height="100vh",
            width="100vw",
            bgcolor=BG_COLOR,
            font_color=FONT_COLOR,
            heading=title,
            directed=False,
        )

        # Physics: repulsion solver for clean layout
        net.repulsion(
            node_distance=120,
            central_gravity=0.2,
            spring_length=150,
            spring_strength=0.05,
            damping=0.09,
        )

        # Add nodes
        for e in entities:
            color = ENTITY_COLORS.get(e.type, DEFAULT_COLOR)
            size = max(10, min(60, 10 + e.salience * 50))  # Scale 10-60 by salience
            tooltip = f"<b>{e.canonical_name}</b><br>Type: {e.type}<br>Mentions: {e.frequency}<br>Spread: {e.spread}"

            net.add_node(
                e.entity_id,
                label=e.canonical_name,
                color=color,
                size=size,
                title=tooltip,
                group=e.type,
            )

        # Add all edge candidates (Pyvis lets users filter interactively)
        edge_candidates = self._all_edge_candidates(entities, filename)
        for edge in edge_candidates:
            net.add_edge(
                edge["from"],
                edge["to"],
                value=edge["value"],
                title=edge["title"],
            )

        # Enable UI controls
        net.show_buttons(filter_=["physics"])
        net.toggle_physics(True)

        # Generate HTML string
        return net.generate_html()

    # ── Private helpers ────────────────────────────────────────────────

    def _make_nodes(self, entities: List) -> List[Dict]:
        nodes = []
        for e in entities:
            nodes.append({
                "id": e.entity_id,
                "label": e.canonical_name,
                "value": e.salience,
                "group": e.type,
                "frequency": e.frequency,
                "spread": e.spread,
                "title": f"Type: {e.type} | Mentions: {e.frequency}",
            })
        return nodes

    def _all_edge_candidates(self, entities: List, filename: str) -> List[Dict]:
        """Compute all co-occurrence scores, no filtering."""
        edges = []
        for i in range(len(entities)):
            a = entities[i]
            for j in range(i + 1, len(entities)):
                b = entities[j]
                comp = self.wiki.compare_entities(
                    a.entity_id, b.entity_id, filenames=[filename]
                )
                score = comp.get("cooccurrence_score", 0)
                edges.append({
                    "from": a.entity_id,
                    "to": b.entity_id,
                    "value": score,
                    "title": f"Co-occurrence: {score:.3f}",
                })
        return edges

    def _filter_edges(self, candidates: List[Dict],
                      min_cooc: float, max_edges: int) -> List[Dict]:
        """Server-side filter (for non-interactive data export)."""
        filtered = [e for e in candidates if e["value"] >= min_cooc]
        node_edge_map: Dict[str, List[Dict]] = {}
        for e in filtered:
            node_edge_map.setdefault(e["from"], []).append(e)
            node_edge_map.setdefault(e["to"], []).append(e)
        used = set()
        result = []
        for node, edges in node_edge_map.items():
            edges.sort(key=lambda x: x["value"], reverse=True)
            for e in edges[:max_edges]:
                key = tuple(sorted((e["from"], e["to"])))
                if key not in used:
                    used.add(key)
                    result.append(e)
        return result