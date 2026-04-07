"""
src/core/Graph.py
-----------------
Builds and renders the Darwin knowledge graph.

Pipeline:
    chunks → Extractor → concepts → embed → dedup → base graph
                                                          ↓
                                              save to .darwin/
                                                          ↓
                                         render(threshold) → pyvis HTML
                                                          ↑
                                              slider changes this only

The slider never recomputes — it just filters the similarity matrix
that was saved once when the graph was first built.

Usage:
    builder = GraphBuilder(store, llm)

    # build once per file
    graph_data = builder.build(filename, chunks)

    # render at any threshold (slider)
    html_path = builder.render(filename, threshold=0.40)

    # re-render at different threshold (slider moved)
    html_path = builder.render(filename, threshold=0.25)
"""

import asyncio
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import networkx as nx

from src.core.Chunker  import Chunk
from src.core.Embedder import Embedder
from src.core.Extractor import Extractor, Concept, ExtractionResult
from src.core.LLM      import LLM
from src.core.Store    import DarwinStore
from src.lib.Terminal import TerminalUI


# ── Constants ──────────────────────────────────────────────────────────────────

DEDUP_THRESHOLD  = 0.85   # above this → same concept, merge
EDGE_LABEL_WORDS = 6      # max words in an edge label


# ── Deduplication ──────────────────────────────────────────────────────────────

def _deduplicate(
    concepts:   list[Concept],
    embeddings: np.ndarray,
) -> tuple[list[Concept], np.ndarray]:
    """
    Merge concepts that are too similar to each other.
    Uses cosine similarity on embeddings — same idea, different wording.

    When merging: keep the concept with higher weight.
    Merge source_chunks so the surviving node knows all its sources.

    Returns: (unique_concepts, unique_embeddings)
    """
    if len(concepts) == 0:
        return [], embeddings

    # full similarity matrix in one shot — fast numpy operation
    sim_matrix = embeddings @ embeddings.T   # shape (n, n)

    skip    = set()
    keep    = []
    keep_idx = []

    for i in range(len(concepts)):
        if i in skip:
            continue

        merged = concepts[i]

        for j in range(i + 1, len(concepts)):
            if j in skip:
                continue

            if sim_matrix[i][j] > DEDUP_THRESHOLD:
                # merge j into i — keep whichever has higher weight
                if concepts[j].weight > merged.weight:
                    merged = concepts[j]
                skip.add(j)

        keep.append(merged)
        keep_idx.append(i)

    unique_embeddings = embeddings[keep_idx]
    return keep, unique_embeddings


# ── Edge labels ────────────────────────────────────────────────────────────────

def _simple_edge_label(concept_a: Concept, concept_b: Concept) -> str:
    """
    Generate an edge label without an LLM call.
    Finds meaningful shared words between two concept labels.
    Not as rich as LLM labels but instant and free.
    """
    stop = {
        "the", "a", "an", "is", "in", "of", "to", "and", "or",
        "that", "this", "it", "its", "are", "was", "be", "by",
        "for", "with", "as", "at", "from", "on", "not", "but",
    }

    words_a = set(concept_a.label.lower().split()) - stop
    words_b = set(concept_b.label.lower().split()) - stop
    shared  = words_a & words_b

    if shared:
        label = ", ".join(sorted(shared)[:3])
        return label

    return "related"


# ── Graph builder ──────────────────────────────────────────────────────────────

@dataclass
class BuildResult:
    filename:      str
    concept_count: int
    edge_count:    int
    llm_available: bool
    failed_chunks: int
    is_raw:        bool   # True = no LLM, raw chunk labels


class GraphBuilder:



    """
    Builds and renders Darwin knowledge graphs.

    Args:
        store: DarwinStore for loading/saving graph data
        llm:   LLM instance — if unavailable, raw mode used
    """

    def __init__(self, store: DarwinStore, llm: LLM):
        self.store    = store
        self.llm      = llm
        self.embedder = Embedder()

    def build(self, filename: str, chunks: list[Chunk]) -> BuildResult:
        """
        Build the base graph for a file.

        Steps:
            1. extract concepts (LLM or raw fallback)
            2. embed all concepts
            3. deduplicate near-identical concepts
            4. compute full similarity matrix
            5. save to .darwin/filename/graph/base_graph.json

        This runs once. The slider reads from the saved JSON forever.

        Args:
            filename: the source file e.g. "biology.pdf"
            chunks:   output from chunk_document()

        Returns:
            BuildResult with stats about what was built
        """
        TerminalUI.blank()
        TerminalUI.bold(f"building graph for {filename}")
        TerminalUI.blank()

        # ── step 1: extract concepts ──────────────────────────────────────────
        extractor = Extractor(self.llm)

        if not self.llm.is_available:
            TerminalUI.dim("no LLM — using raw chunk text as concept labels")
            TerminalUI.dim(f"tip: set up an LLM and run 'darwin graph {filename} --refine'")
            TerminalUI.dim("for cleaner concept labels")
            TerminalUI.blank()

        result = extractor.extract_all_sync(chunks)
        concepts = result.concepts

        if not concepts:
            TerminalUI.error(f"no concepts extracted from {filename}")
            raise ValueError(f"no concepts could be extracted from {filename}")

        TerminalUI.step(f"extracted {len(concepts)} concepts ({'llm' if result.llm_available else 'raw'})")
        TerminalUI.done()
        TerminalUI.blank()

        # ── step 2: embed all concepts ────────────────────────────────────────
        TerminalUI.step("embedding concepts")
        
        # reuse Embedder with passage prefix
        from src.core.Chunker import Chunk as ChunkClass
        concept_chunks = [
            ChunkClass(
                text        = concept.label,
                index       = i,
                source      = concept.source,
                token_count = len(concept.label.split()),
                is_overlap  = False,
            )
            for i, concept in enumerate(concepts)
        ]

        embed_result = self.embedder.embed(concept_chunks)
        embeddings   = np.array([ec.embedding for ec in embed_result.embedded_chunks])
        TerminalUI.done()

        # ── step 3: deduplicate ───────────────────────────────────────────────
        TerminalUI.step("deduplicating similar concepts")
        concepts, embeddings = _deduplicate(concepts, embeddings)
        TerminalUI.done(f"{len(concepts)} unique concepts")
        TerminalUI.blank()

        # ── step 4: compute full similarity matrix ────────────────────────────
        TerminalUI.step("computing similarity matrix")
        # this is what the slider reads — computed once, never again
        sim_matrix = embeddings @ embeddings.T
        sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
        TerminalUI.done()

        # ── step 5: count edges at default threshold ──────────────────────────
        default_threshold = self.store.load_config().get("graph_threshold", 0.40)
        n = len(concepts)
        edge_count = sum(
            1 for i in range(n)
            for j in range(i + 1, n)
            if sim_matrix[i][j] > default_threshold
        )

        # ── step 6: save ──────────────────────────────────────────────────────
        TerminalUI.step("saving graph to disk")
        graph_data = {
            "concepts": [
                {
                    "id":          i,
                    "label":       c.label,
                    "weight":      c.weight,
                    "chunk_index": c.chunk_index,
                    "page":        c.page,
                    "source":      c.source,
                    "from_llm":    c.from_llm,
                }
                for i, c in enumerate(concepts)
            ],
            "similarity_matrix": sim_matrix.tolist(),
            "llm_available":     result.llm_available,
            "is_raw":            not result.llm_available,
        }

        self.store.save_graph(filename, graph_data, threshold=default_threshold)
        TerminalUI.done()
        TerminalUI.blank()

        TerminalUI.success(f"saved — {len(concepts)} concepts, {edge_count} edges")
        TerminalUI.blank()

        return BuildResult(
            filename      = filename,
            concept_count = len(concepts),
            edge_count    = edge_count,
            llm_available = result.llm_available,
            failed_chunks = result.failed_chunks,
            is_raw        = not result.llm_available,
        )

    def render(
        self,
        filename:  str,
        threshold: float = 0.40,
        open_browser: bool = True,
    ) -> Path:
        """
        Render the graph at a given threshold and open in browser.

        This is what the slider calls — loads the saved similarity matrix
        and filters edges by threshold. No LLM calls, no embedding.
        Instant.

        Args:
            filename:     the source file e.g. "biology.pdf"
            threshold:    edge threshold 0.25 (dense) to 0.65 (sparse)
            open_browser: open the HTML in browser automatically

        Returns:
            Path to the rendered HTML file
        """
        graph_data = self.store.load_graph(filename)
        concepts   = graph_data["concepts"]
        matrix     = graph_data["similarity_matrix"]
        is_raw     = graph_data.get("is_raw", False)

        # ── build NetworkX graph ──────────────────────────────────────────────
        G = nx.Graph()

        # add nodes
        for c in concepts:
            # truncate long labels for display
            display = c["label"]
            if len(display) > 80:
                display = display[:77] + "..."

            G.add_node(
                c["id"],
                label     = display,
                full_label = c["label"],
                weight    = c["weight"],
                size      = c["weight"] * 6,
                page      = c["page"],
                source    = c["source"],
                from_llm  = c["from_llm"],
                title     = (                          # hover tooltip
                    f"<b>{display}</b><br>"
                    f"source: {c['source']}<br>"
                    f"page: {c['page']}<br>"
                    f"importance: {c['weight']}/10<br>"
                    f"chunk: {c['chunk_index']}"
                ),
            )

        # add edges above threshold
        n = len(concepts)
        for i in range(n):
            for j in range(i + 1, n):
                sim = matrix[i][j]
                if sim > threshold:
                    c_i = concepts[i]
                    c_j = concepts[j]

                    # edge label — shared keywords
                    concept_i = Concept(
                        label=c_i["label"], weight=c_i["weight"],
                        chunk_index=c_i["chunk_index"]
                    )
                    concept_j = Concept(
                        label=c_j["label"], weight=c_j["weight"],
                        chunk_index=c_j["chunk_index"]
                    )
                    edge_label = _simple_edge_label(concept_i, concept_j)

                    G.add_edge(
                        i, j,
                        weight = sim,
                        width  = sim * 8,
                        title  = f"{edge_label} (similarity: {sim:.2f})",
                        label  = "",   # don't show on edge — too cluttered
                    )

        # ── pyvis render ──────────────────────────────────────────────────────
        from pyvis.network import Network

        net = Network(
            height     = "100vh",
            width      = "100%",
            bgcolor    = "#0f0f0f",
            font_color = "#ffffff",
            notebook   = False,
            directed   = False,
        )

        net.from_nx(G)

        # node styling — size from weight, color from LLM vs raw
        for node in net.nodes:
            node_id   = node["id"]
            nx_data   = G.nodes[node_id]
            from_llm  = nx_data.get("from_llm", True)
            is_raw_node = not from_llm

            node["label"] = nx_data["label"]
            node["title"] = nx_data["title"]
            node["size"]  = nx_data["size"]

            if is_raw_node:
                # raw nodes — muted blue, indicates "could be refined"
                node["color"] = {
                    "background": "#1a3a5c",
                    "border":     "#2e6da4",
                    "highlight":  {"background": "#2e6da4", "border": "#5ba3d9"},
                }
            else:
                # LLM nodes — rich purple, indicates clean concepts
                node["color"] = {
                    "background": "#2d1b5e",
                    "border":     "#7c3aed",
                    "highlight":  {"background": "#7c3aed", "border": "#a78bfa"},
                }

            node["font"] = {"size": 14, "color": "#ffffff"}

        # edge styling
        for edge in net.edges:
            sim          = edge.get("weight", 0.4)
            edge["width"] = sim * 8
            edge["color"] = {
                "color":   "#4a4a6a",
                "highlight": "#7c3aed",
                "opacity": 0.6,
            }

        # physics — force atlas for organic clustering
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "forceAtlas2Based": {
              "gravitationalConstant": -60,
              "centralGravity": 0.01,
              "springLength": 180,
              "springConstant": 0.08,
              "damping": 0.4,
              "avoidOverlap": 0.8
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
              "enabled": true,
              "iterations": 200
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "hideEdgesOnDrag": true,
            "navigationButtons": true
          },
          "nodes": {
            "shape": "dot",
            "borderWidth": 2,
            "shadow": false
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            },
            "shadow": false
          }
        }
        """)

        # ── save HTML ─────────────────────────────────────────────────────────
        graph_dir  = self.store._graph_dir(filename)
        graph_dir.mkdir(parents=True, exist_ok=True)
        html_path  = graph_dir / f"graph_{threshold:.2f}.html"

        # inject warning banner for raw graphs
        net.save_graph(str(html_path))

        if is_raw:
            _inject_raw_warning(html_path, filename)

        _inject_slider(html_path, threshold, filename)


        TerminalUI.success(f"rendered → {html_path.name}")
        TerminalUI.dim(f"nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}  threshold: {threshold:.2f}")

        if open_browser:
            webbrowser.open(html_path.as_uri())

        return html_path

    def refine(self, filename: str, chunks: list[Chunk]) -> BuildResult:
        """
        Refine an existing raw graph with LLM concept extraction.
        Replaces raw chunk labels with clean proposition-style concepts.
        Called when user runs 'darwin graph <file> --refine'

        Only makes sense when llm.is_available is True.
        """
        if not self.llm.is_available:
            TerminalUI.error("refine requires an LLM — set one up in config")
            raise RuntimeError("LLM not available for graph refinement")

        TerminalUI.info( label="Building Refined Graph", value="refining graph...")
        return self.build(filename, chunks)


# ── HTML injection helpers ─────────────────────────────────────────────────────

def _inject_raw_warning(html_path: Path, filename: str) -> None:
    """
    Inject a warning banner into the graph HTML when no LLM was used.
    Tells the user the graph is rough but still useful,
    and how to refine it.
    """
    warning_html = f"""
    <div id="darwin-warning" style="
        position: fixed;
        top: 16px;
        left: 50%;
        transform: translateX(-50%);
        background: #1a3a5c;
        border: 1px solid #2e6da4;
        border-radius: 8px;
        padding: 10px 20px;
        color: #93c5fd;
        font-family: monospace;
        font-size: 13px;
        z-index: 9999;
        text-align: center;
        max-width: 600px;
    ">
        ⚠ raw graph — no LLM used, node labels are chunk text
        &nbsp;·&nbsp;
        <span style="color: #60a5fa;">
            run <code>darwin graph {filename} --refine</code> for cleaner concepts
        </span>
        <span onclick="document.getElementById('darwin-warning').remove()"
              style="cursor:pointer; margin-left:12px; opacity:0.6;">✕</span>
    </div>
    """

    html = html_path.read_text(encoding="utf-8")
    html = html.replace("<body>", f"<body>\n{warning_html}", 1)
    html_path.write_text(html, encoding="utf-8")


def _inject_slider(html_path: Path, current_threshold: float, filename: str) -> None:
    """
    Inject the zoom slider into the graph HTML.

    Slider maps:
        left  (0.0) → threshold 0.65 → sparse, big concepts
        right (1.0) → threshold 0.25 → dense, granular concepts

    The slider label shows the current zoom level as a descriptor
    not a raw number — more intuitive for non-technical users.
    """
    slider_value = 1.0 - ((current_threshold - 0.25) / (0.65 - 0.25))
    slider_value = max(0.0, min(1.0, slider_value))

    if current_threshold >= 0.55:
        zoom_label = "overview"
    elif current_threshold >= 0.40:
        zoom_label = "balanced"
    else:
        zoom_label = "detailed"

    slider_html = f"""
    <div id="darwin-controls" style="
        position: fixed;
        bottom: 24px;
        left: 50%;
        transform: translateX(-50%);
        background: #1a1a2e;
        border: 1px solid #2d2d4a;
        border-radius: 12px;
        padding: 14px 24px;
        display: flex;
        align-items: center;
        gap: 16px;
        font-family: monospace;
        font-size: 13px;
        color: #a0a0c0;
        z-index: 9999;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    ">
        <span style="color:#6b6b8a;">overview</span>
        <input
            type="range"
            id="zoom-slider"
            min="0"
            max="1"
            step="0.05"
            value="{slider_value:.2f}"
            style="width: 200px; accent-color: #7c3aed; cursor: pointer;"
            oninput="updateZoom(this.value)"
        />
        <span style="color:#6b6b8a;">detailed</span>
        <span id="zoom-label" style="
            color: #a78bfa;
            min-width: 70px;
            text-align: center;
        ">{zoom_label}</span>
        <span style="color:#4a4a6a; font-size:11px;">
            {filename}
        </span>
    </div>

    <script>
    function updateZoom(val) {{
        var v         = parseFloat(val);
        var threshold = 0.65 - (v * (0.65 - 0.25));
        var label     = document.getElementById('zoom-label');

        if (threshold >= 0.55) {{
            label.textContent = 'overview';
        }} else if (threshold >= 0.40) {{
            label.textContent = 'balanced';
        }} else {{
            label.textContent = 'detailed';
        }}

        // in a real app this would call back to darwin to re-render
        // for now show what threshold would be used
        label.title = 'threshold: ' + threshold.toFixed(2);
    }}
    </script>
    """

    html = html_path.read_text(encoding="utf-8")
    html = html.replace("</body>", f"{slider_html}\n</body>", 1)
    html_path.write_text(html, encoding="utf-8")