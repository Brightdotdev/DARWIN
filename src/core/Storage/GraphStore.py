"""
GraphStore — save/load entity graph artifacts to a DarwinStore.

Uses the existing ARTIFACT_MANIFEST paths:
  graph/nodes.parquet
  graph/edges.parquet
  graph/graph.html
"""

from pathlib import Path
from typing import Optional, Any

import pandas as pd

from src.core.Logger.DarwinLogger import darwin


class GraphStore:
    """Thin wrapper for graph persistence on top of a DarwinStore."""

    def __init__(self, store: Any):
        self._store = store

    def save_graph(self, source_name: str, graph_result: Any) -> None:
        """
        Save nodes, edges, and optional visualization to the store.

        Args:
            source_name: Original document filename.
            graph_result: GraphResult with .nodes, .edges, .stats DataFrames.
        """
        content_hash = self._resolve_hash(source_name)
        if not content_hash:
            darwin.error(f"Cannot save graph: no content hash for {source_name}")
            return

        graph_dir = self._store.dir_for_hash(content_hash) / "graph"
        graph_dir.mkdir(parents=True, exist_ok=True)

        if not graph_result.nodes.empty:
            nodes_path = graph_dir / "nodes.parquet"
            graph_result.nodes.to_parquet(nodes_path, index=False)
            darwin.info(f"Saved {len(graph_result.nodes)} graph nodes to {nodes_path}")

        if not graph_result.edges.empty:
            edges_path = graph_dir / "edges.parquet"
            graph_result.edges.to_parquet(edges_path, index=False)
            darwin.info(f"Saved {len(graph_result.edges)} graph edges to {edges_path}")

        html_content = graph_result.stats.get("pyvis_html")
        if html_content is not None:
            html_path = graph_dir / "graph.html"
            try:
                html_path.write_text(html_content.generate_html(), encoding="utf-8")
                darwin.info(f"Saved graph visualization to {html_path}")
            except Exception as e:
                darwin.warning(f"Failed to save graph HTML: {e}")

        self._mark_stage(content_hash, "graph")

    def load_graph(self, source_name: str) -> Optional[dict]:
        """
        Load nodes and edges DataFrames from the store.

        Returns:
            Dict with 'nodes' and 'edges' DataFrames, or None if not found.
        """
        content_hash = self._resolve_hash(source_name)
        if not content_hash:
            return None

        graph_dir = self._store.dir_for_hash(content_hash) / "graph"
        nodes_path = graph_dir / "nodes.parquet"
        edges_path = graph_dir / "edges.parquet"

        if not nodes_path.exists() and not edges_path.exists():
            return None

        result = {}
        if nodes_path.exists():
            result["nodes"] = pd.read_parquet(nodes_path)
        else:
            result["nodes"] = pd.DataFrame()

        if edges_path.exists():
            result["edges"] = pd.read_parquet(edges_path)
        else:
            result["edges"] = pd.DataFrame()

        return result

    def has_graph(self, source_name: str) -> bool:
        """Check if graph artifacts exist for a source."""
        content_hash = self._resolve_hash(source_name)
        if not content_hash:
            return False

        graph_dir = self._store.dir_for_hash(content_hash) / "graph"
        return (graph_dir / "nodes.parquet").exists()

    def get_graph_html_path(self, source_name: str) -> Optional[Path]:
        """Return path to the HTML visualization if it exists."""
        content_hash = self._resolve_hash(source_name)
        if not content_hash:
            return None

        html_path = self._store.dir_for_hash(content_hash) / "graph" / "graph.html"
        return html_path if html_path.exists() else None

    def list_graphed_sources(self) -> list:
        """Return list of source names that have graph artifacts."""
        result = []
        try:
            registry = self._store._registry
            for source in registry.list_sources():
                if self.has_graph(source.display_name):
                    result.append(source.display_name)
        except Exception:
            pass
        return result

    def _resolve_hash(self, source_name: str) -> Optional[str]:
        """Resolve a source name to its content hash."""
        try:
            content = self._store._registry.get_content_by_name(source_name)
            if content:
                return content.hash
        except Exception:
            pass

        try:
            sources = self._store._registry.list_sources()
            for s in sources:
                if s.display_name == source_name:
                    return s.content_hash
        except Exception:
            pass

        return None

    def _mark_stage(self, content_hash: str, stage: str) -> None:
        """Mark a pipeline stage as completed in the registry."""
        try:
            self._store._registry.mark_stage(content_hash, stage)
        except Exception as e:
            darwin.debug(f"Failed to mark stage {stage}: {e}")
