import json
from typing import List, Dict, Any, Optional
from src.features.Knowledge.WikiIndex import WikiIndex

class GraphBuilder:
    """Build entity co‑occurrence graph data + HTML for a document."""

    def __init__(self, wiki_index: WikiIndex):
        self.wiki = wiki_index

    def build_graph_data(self, filename: str,
                         max_entities: int = 100,
                         min_cooccurrence: float = 0.1,
                         max_edges_per_node: int = 8) -> Dict[str, list]:
        """Returns filtered nodes and edges (as before)."""
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
        Generate a full HTML page with interactive sliders.
        All candidate edges (score >= 0) are embedded, and two sliders
        let you filter them on the fly.
        """
        entities = self.wiki.list_entities(
            filenames=[filename], sort_by="salience", limit=max_entities
        )
        if not entities:
            return "<html><body>No entities found.</body></html>"

        nodes = self._make_nodes(entities)
        all_edges = self._all_edge_candidates(entities, filename)

        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(all_edges)   # full set, scoring from 0–1

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{
      margin: 0; background: #1e1e1e; font-family: 'Segoe UI', sans-serif;
    }}
    #controls {{
      position: absolute; top: 10px; left: 10px; z-index: 999;
      background: rgba(30,30,30,0.9); padding: 15px; border-radius: 8px;
      color: #e0e0e0; display: flex; flex-direction: column; gap: 8px;
      min-width: 220px;
    }}
    #controls label {{ font-size: 14px; }}
    #mynetwork {{
      width: 100vw; height: 100vh;
    }}
  </style>
</head>
<body>
<div id="controls">
  <label>Min Co‑occurrence: <span id="coocVal">0.10</span></label>
  <input type="range" id="coocSlider" min="0" max="1" step="0.01" value="0.10">

  <label>Max Edges per Node: <span id="maxEdgeVal">8</span></label>
  <input type="range" id="maxEdgeSlider" min="1" max="30" step="1" value="8">

  <div style="margin-top:6px;font-size:13px;color:#aaa">
    Nodes: {len(nodes)} | Edges: <span id="edgeCount">{len(all_edges)}</span>
  </div>
</div>
<div id="mynetwork"></div>
<script type="text/javascript">
  var allNodes = new vis.DataSet({nodes_json});
  var allEdges = {edges_json};   // raw array of {{from, to, value, title}}

  var container = document.getElementById('mynetwork');
  var data = {{
    nodes: allNodes,
    edges: new vis.DataSet()
  }};
  var options = {{
    nodes: {{
      shape: 'dot',
      scaling: {{
        min: 10, max: 60,
        label: {{ enabled: true, min: 14, max: 30 }}
      }},
      font: {{ color: '#e0e0e0', size: 14 }},
      color: {{
        background: '#4a90d9', border: '#2c5f8a',
        highlight: {{ background: '#ffaa00', border: '#cc8800' }}
      }}
    }},
    edges: {{
      color: {{ color: '#5a5a5a', highlight: '#ffaa00', hover: '#aaaaaa' }},
      width: 1, selectionWidth: 2,
      smooth: {{ type: 'continuous' }}
    }},
    physics: {{
      stabilization: {{ iterations: 200 }},
      solver: 'repulsion',
      repulsion: {{
        centralGravity: 0.2, springLength: 150, springConstant: 0.05,
        nodeDistance: 120, damping: 0.09
      }}
    }},
    interaction: {{
      hover: true, tooltipDelay: 200, navigationButtons: true, keyboard: true
    }},
    groups: {{
      PERSON: {{ color: {{ background: '#e67e22', border: '#a85c1a' }} }},
      ORG:    {{ color: {{ background: '#3498db', border: '#1f618d' }} }},
      GPE:    {{ color: {{ background: '#2ecc71', border: '#1f8b4c' }} }},
      LOC:    {{ color: {{ background: '#2ecc71', border: '#1f8b4c' }} }},
      WORK_OF_ART: {{ color: {{ background: '#9b59b6', border: '#6c3483' }} }},
      NORP:   {{ color: {{ background: '#f1c40f', border: '#b7950b' }} }},
      FAC:    {{ color: {{ background: '#1abc9c', border: '#117864' }} }},
      EVENT:  {{ color: {{ background: '#e74c3c', border: '#a93226' }} }},
      LAW:    {{ color: {{ background: '#95a5a6', border: '#5d6d7e' }} }},
      LANGUAGE: {{ color: {{ background: '#af7ac5', border: '#76448a' }} }}
    }}
  }};
  var network = new vis.Network(container, data, options);

  function applyFilters() {{
    var minCooc = parseFloat(document.getElementById('coocSlider').value);
    var maxEdges = parseInt(document.getElementById('maxEdgeSlider').value);

    document.getElementById('coocVal').textContent = minCooc.toFixed(2);
    document.getElementById('maxEdgeVal').textContent = maxEdges;

    // Step 1: filter by co‑occurrence
    var filtered = allEdges.filter(e => e.value >= minCooc);

    // Step 2: keep top N edges per node
    var nodeEdgeCount = {{}};
    filtered.forEach(e => {{
      nodeEdgeCount[e.from] = nodeEdgeCount[e.from] || [];
      nodeEdgeCount[e.from].push(e);
      nodeEdgeCount[e.to] = nodeEdgeCount[e.to] || [];
      nodeEdgeCount[e.to].push(e);
    }});
    var used = new Set();
    var finalEdges = [];
    Object.keys(nodeEdgeCount).forEach(node => {{
      var sorted = nodeEdgeCount[node].sort((a,b) => b.value - a.value);
      sorted.slice(0, maxEdges).forEach(e => {{
        var key = [e.from, e.to].sort().join('---');
        if (!used.has(key)) {{
          used.add(key);
          finalEdges.push(e);
        }}
      }});
    }});

    document.getElementById('edgeCount').textContent = finalEdges.length;
    network.setData({{ nodes: allNodes, edges: new vis.DataSet(finalEdges) }});
  }}

  document.getElementById('coocSlider').addEventListener('input', applyFilters);
  document.getElementById('maxEdgeSlider').addEventListener('input', applyFilters);

  // Initial apply
  applyFilters();
</script>
</body>
</html>"""
        return html

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
                "title": f"<b>{e.canonical_name}</b><br>Type: {e.type}<br>Mentions: {e.frequency}"
            })
        return nodes

    def _all_edge_candidates(self, entities: List, filename: str) -> List[Dict]:
        """Compute all co‑occurrence scores, no filtering."""
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
                    "title": f"Co‑occurrence: {score:.3f}"
                })
        return edges

    def _filter_edges(self, candidates: List[Dict],
                      min_cooc: float, max_edges: int) -> List[Dict]:
        """Server‑side filter (for non‑interactive method)."""
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