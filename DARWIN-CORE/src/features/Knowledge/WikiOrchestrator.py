from typing import List, Dict, Any, Optional
from pathlib import Path
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Knowledge.WikiBuilder import WikiBuilder
from src.features.Knowledge.WikiIndex import WikiIndex, NullWikiIndex
from src.features.Knowledge.GraphBuilder import GraphBuilder

class WikiOrchestrator:
    """Central orchestrator for all wiki + graph operations."""

    def __init__(self, store: DarwinStore):
        self.store = store
        self.builder = WikiBuilder(store)

    # ── Core build / load ──────────────────────────────────────────────

    def get_wiki(self, filename: str) -> WikiIndex:
        return self.builder.get(filename)

    def build_wiki(self, filename: str, rerun: bool = False) -> WikiIndex:
        return self.builder.build(filename, rerun=rerun)

    # ── Entity analysis helpers ────────────────────────────────────────

    def entity_card(self, entity_name: str, filenames: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        wiki = self._load_any_wiki(filenames)
        if wiki is None:
            return None
        matches = wiki.detect_entities(entity_name)
        if not matches:
            return None
        e = matches[0]
        return {
            "entity_id": e.entity_id,
            "canonical_name": e.canonical_name,
            "type": e.type,
            "mentions": e.frequency,
            "spread": e.spread,
            "salience": e.salience,
            "doc_ids": e.doc_ids,
            "top_cooccurring": [],
        }

    def expand_query(self, query: str, filenames: Optional[List[str]] = None,
                     top_n: int = 5) -> str:
        wiki = self._load_any_wiki(filenames)
        if wiki is None:
            return query
        entities = wiki.detect_entities(query)
        if not entities:
            return query
        expansions = set()
        for e in entities[:3]:
            stats = wiki.get_entity_stats(filenames)
            for top in stats.get("top_entities", []):
                comp = wiki.compare_entities(e.entity_id, top["entity_id"])
                if comp.get("cooccurrence_score", 0) > 0.2:
                    expansions.add(top["canonical_name"])
                    if len(expansions) >= top_n:
                        break
        if expansions:
            return f"{query} {' '.join(expansions)}"
        return query

    def document_profile(self, filename: str) -> Dict[str, Any]:
        wiki = self.get_wiki(filename)
        if isinstance(wiki, NullWikiIndex):
            return {}
        return wiki.get_document_profile(filename)

    # ── Graph generation & persistence ─────────────────────────────────

    def build_and_save_graph(self, filename: str,
                             max_entities: int = 100,
                             min_cooccurrence: float = 0.1,
                             max_edges_per_node: int = 8,
                             force_rebuild: bool = False) -> bool:
        """Build entity graph data + HTML and persist both to the store."""
        if not force_rebuild and self.store.has_graph(filename):
            darwin.log(f"Graph already complete for {filename}. Use force_rebuild=True.")
            return True

        wiki = self.get_wiki(filename)
        if isinstance(wiki, NullWikiIndex):
            darwin.log(f"No wiki data for {filename} — cannot build graph.", LogType.ERROR)
            return False

        gb = GraphBuilder(wiki)
        graph_data = gb.build_graph_data(
            filename,
            max_entities=max_entities,
            min_cooccurrence=min_cooccurrence,
            max_edges_per_node=max_edges_per_node
        )
        html = gb.build_html(
            filename,
            max_entities=max_entities,
            min_cooccurrence=min_cooccurrence,
            max_edges_per_node=max_edges_per_node,
            title=f"Entity Graph: {filename}"
        )

        try:
            self.store.save_graph(filename, graph_data["nodes"], graph_data["edges"])
            self.store.save_graph_html(filename, html)
            darwin.log(f"Graph and HTML saved for {filename}", LogType.SUCCESS)
            return True
        except Exception as e:
            darwin.log(f"Failed to save graph for {filename}: {e}", LogType.ERROR)
            return False
        
    def build_and_save_interactive_graph(self, filename: str,
                                     max_entities: int = 100,
                                     title: str = "Entity Graph") -> bool:
        """Build the interactive HTML (with sliders) and persist it."""
        wiki = self.get_wiki(filename)
        if isinstance(wiki, NullWikiIndex):
            darwin.log(f"No wiki data for {filename} — cannot build graph.", LogType.ERROR)
            return False

        gb = GraphBuilder(wiki)
        html = gb.build_interactive_html(filename, max_entities=max_entities, title=title)

        try:
            # Save HTML to store (same as before)
            self.store.save_graph_html(filename, html)
            # Also generate basic graph data (nodes/edges) so the store is consistent
            data = gb.build_graph_data(filename, max_entities=max_entities,
                                    min_cooccurrence=0.0, max_edges_per_node=30)
            self.store.save_graph(filename, data["nodes"], data["edges"])

            # Optional: also write a local copy for instant viewing
            with open("enoch_interactive.html", "w", encoding="utf-8") as f:
                f.write(html)
            return True
        except Exception as e:
            darwin.log(f"Failed to save interactive graph: {e}", LogType.ERROR)
            return False


    def get_graph_html(self, filename: str) -> Optional[str]:
        """Return the HTML string of the saved graph, if present."""
        h = self.store._resolve_hash(filename)
        if h is None:
            return None
        path = self.store.paths.graph_html_path(h)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def get_graph_data(self, filename: str) -> Optional[Dict[str, list]]:
        """
        Return nodes and edges as lists of dicts.
        Converts from the store's Parquet columnar format.
        """
        try:
            raw = self.store.load_graph(filename)
        except Exception:
            return None
        if not raw or (not raw.get("nodes") and not raw.get("edges")):
            return None

        nodes = self._columns_to_records(raw["nodes"]) if raw.get("nodes") else []
        edges = self._columns_to_records(raw["edges"]) if raw.get("edges") else []
        return {"nodes": nodes, "edges": edges}

    # ── Internal helpers ───────────────────────────────────────────────

    def _load_any_wiki(self, filenames: Optional[List[str]] = None) -> Optional[WikiIndex]:
        if filenames:
            for f in filenames:
                wiki = self.get_wiki(f)
                if not isinstance(wiki, NullWikiIndex):
                    return wiki
        # If none given, try first file we can get a real wiki for
        # (You may replace this with a method that lists all registered files)
        return None

    @staticmethod
    def _columns_to_records(columns: dict) -> list[dict]:
        """Convert columnar dict (from PyArrow to_pydict) back to list of records."""
        if not columns:
            return []
        keys = list(columns.keys())
        length = len(columns[keys[0]])
        return [{k: columns[k][i] for k in keys} for i in range(length)]