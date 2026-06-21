"""
Darwin Wiki / Knowledge Orchestrator (v3.4)
Central orchestrator for all wiki + graph operations.

NEW: Batch operations, staged builds, multi-file entity overlap detection.

REQUIRES: Full Darwin stack (src.core.Logger, src.core.Storage)
"""

from typing import List, Dict, Any, Optional, Set, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Knowledge.WikiBuilder import WikiBuilder
from src.features.Knowledge.WikiIndex import WikiIndex, NullWikiIndex
from src.features.Knowledge.GraphBuilder import GraphBuilder
from src.features.Knowledge.lib import EntityRecord


# ── Build stage enum ───────────────────────────────────────────────────

BUILD_STAGE = Literal["wiki", "graph", "both"]


# ── Batch result dataclass ─────────────────────────────────────────────

@dataclass
class BatchBuildResult:
    """Result of a single file in a batch build operation."""
    filename: str
    success: bool
    wiki_built: bool = False
    graph_built: bool = False
    error: Optional[str] = None
    entity_count: int = 0
    duration_ms: float = 0.0


@dataclass  
class MultiFileEntityResult:
    """
    Result of listing entities across multiple files.

    Includes per-file views AND merged overlap detection.
    """
    # Per-file entity lists (filename -> list of EntityRecord)
    per_file: Dict[str, List[EntityRecord]] = field(default_factory=dict)

    # Merged view: entities that appear in ANY of the files
    merged_entities: List[EntityRecord] = field(default_factory=list)

    # Overlap analysis: entities that appear in MULTIPLE files
    overlapping: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Entities unique to each file (appear in only one)
    unique_per_file: Dict[str, List[EntityRecord]] = field(default_factory=dict)

    # Cross-file statistics
    total_entities_across_all: int = 0
    total_unique_entities: int = 0
    overlap_count: int = 0


class WikiOrchestrator:
    """Central orchestrator for all wiki + graph operations."""

    def __init__(self, store: DarwinStore, max_workers: int = 4):
        self.store = store
        self.builder = WikiBuilder(store)
        self.max_workers = max_workers

    # ═══════════════════════════════════════════════════════════════════
    # SINGLE-FILE OPERATIONS (existing, preserved)
    # ═══════════════════════════════════════════════════════════════════

    def get_wiki(self, filename: str) -> WikiIndex:
        return self.builder.get(filename)

    def build_wiki(self, filename: str, rerun: bool = False) -> WikiIndex:
        return self.builder.build(filename, rerun=rerun)

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

        stats = wiki.get_entity_stats(filenames)
        top_entities = stats.get("top_entities", [])

        expansions = set()
        for e in entities[:3]:
            for top in top_entities:
                comp = wiki.compare_entities(e.entity_id, top["entity_id"])
                if comp.get("cooccurrence_score", 0) > 0.2:
                    expansions.add(top["canonical_name"])
                    if len(expansions) >= top_n:
                        break
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

    # ═══════════════════════════════════════════════════════════════════
    # BATCH WIKI BUILD
    # ═══════════════════════════════════════════════════════════════════

    def build_wiki_batch(
        self,
        filenames: List[str],
        rerun: bool = False,
        min_frequency: int = 2,
        use_multiprocess: bool = True,
        max_workers: Optional[int] = None,
        on_progress: Optional[Any] = None,  # Callable[[str, bool], None]
    ) -> Dict[str, BatchBuildResult]:
        """
        Build wikis for multiple files in parallel.

        Parameters:
            filenames:      List of files to index
            rerun:          Force rebuild even if cached
            min_frequency:  Minimum mentions to keep an entity
            use_multiprocess: Use ProcessPoolExecutor for spaCy extraction
            max_workers:    Thread pool size (default: self.max_workers)
            on_progress:    Callback(filename, success) for progress tracking

        Returns:
            Dict mapping filename -> BatchBuildResult
        """
        workers = max_workers or self.max_workers
        results: Dict[str, BatchBuildResult] = {}

        import time

        def _build_one(fname: str) -> BatchBuildResult:
            start = time.time()
            try:
                wiki = self.builder.build(
                    fname,
                    rerun=rerun,
                    min_frequency=min_frequency,
                    use_multiprocess=use_multiprocess,
                )
                success = not isinstance(wiki, NullWikiIndex)
                entity_count = len(wiki._entities) if success else 0
                dur = (time.time() - start) * 1000

                if on_progress:
                    on_progress(fname, success)

                return BatchBuildResult(
                    filename=fname,
                    success=success,
                    wiki_built=success,
                    entity_count=entity_count,
                    duration_ms=dur,
                )
            except Exception as e:
                dur = (time.time() - start) * 1000
                if on_progress:
                    on_progress(fname, False)
                return BatchBuildResult(
                    filename=fname,
                    success=False,
                    error=str(e),
                    duration_ms=dur,
                )

        # Use ThreadPoolExecutor for I/O-bound store operations
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_build_one, f): f for f in filenames}
            for future in as_completed(futures):
                result = future.result()
                results[result.filename] = result

        # Log summary
        successes = sum(1 for r in results.values() if r.success)
        darwin.log(
            f"Wiki batch: {successes}/{len(filenames)} succeeded",
            LogType.SUCCESS if successes == len(filenames) else LogType.WARNING
        )

        return results

    # ═══════════════════════════════════════════════════════════════════
    # STAGED KNOWLEDGE BUILD (wiki / graph / both)
    # ═══════════════════════════════════════════════════════════════════

    def build_knowledge(
        self,
        filename: str,
        stage: BUILD_STAGE = "both",
        rerun: bool = False,
        min_frequency: int = 2,
        use_multiprocess: bool = True,
        graph_max_entities: int = 100,
        graph_min_cooccurrence: float = 0.1,
        graph_max_edges_per_node: int = 8,
        graph_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build knowledge artifacts for a single file with stage control.

        Parameters:
            filename:                Target file
            stage:                   "wiki" | "graph" | "both"
            rerun:                   Force rebuild even if cached
            min_frequency:           Minimum mentions for entity retention
            use_multiprocess:        Use multiprocessing for spaCy extraction
            graph_max_entities:      Max entities to include in graph
            graph_min_cooccurrence:  Min co-occurrence score for edges
            graph_max_edges_per_node: Max edges per node (degree cap)
            graph_title:             Custom graph title (default: filename)

        Returns:
            Dict with keys: wiki_index, graph_html, graph_data, stats
        """
        result: Dict[str, Any] = {
            "filename": filename,
            "stage": stage,
            "wiki_index": None,
            "graph_html": None,
            "graph_data": None,
            "stats": {},
            "errors": [],
        }

        # ── Stage 1: Build Wiki ─────────────────────────────────────
        if stage in ("wiki", "both"):
            wiki = self.builder.build(
                filename,
                rerun=rerun,
                min_frequency=min_frequency,
                use_multiprocess=use_multiprocess,
            )
            if isinstance(wiki, NullWikiIndex):
                result["errors"].append("Wiki build failed")
                darwin.log(f"Knowledge build: wiki failed for {filename}", LogType.ERROR)
                return result

            result["wiki_index"] = wiki
            result["stats"]["wiki"] = {
                "entity_count": len(wiki._entities),
                "doc_chunks": len(wiki._doc_chunks.get(filename, set())),
            }
            darwin.log(f"Wiki built for {filename}: {len(wiki._entities)} entities", LogType.SUCCESS)
        else:
            # Graph-only mode: load existing wiki
            wiki = self.get_wiki(filename)
            if isinstance(wiki, NullWikiIndex):
                result["errors"].append("No existing wiki for graph-only build")
                return result
            result["wiki_index"] = wiki

        # ── Stage 2: Build Graph ────────────────────────────────────
        if stage in ("graph", "both"):
            wiki = result["wiki_index"]
            if wiki is None or isinstance(wiki, NullWikiIndex):
                result["errors"].append("Cannot build graph: no wiki data")
                return result

            gb = GraphBuilder(wiki)

            # Build graph data
            graph_data = gb.build_graph_data(
                filename,
                max_entities=graph_max_entities,
                min_cooccurrence=graph_min_cooccurrence,
                max_edges_per_node=graph_max_edges_per_node,
            )
            result["graph_data"] = graph_data

            # Build interactive HTML
            title = graph_title or f"Entity Graph: {filename}"
            html = gb.build_interactive_html(
                filename,
                max_entities=graph_max_entities,
                title=title,
            )
            result["graph_html"] = html

            # Persist
            try:
                self.store.save_graph(filename, graph_data["nodes"], graph_data["edges"])
                self.store.save_graph_html(filename, html)
                darwin.log(f"Graph saved for {filename}", LogType.SUCCESS)
            except Exception as e:
                result["errors"].append(f"Graph persistence failed: {e}")
                darwin.log(f"Graph save failed for {filename}: {e}", LogType.ERROR)

            result["stats"]["graph"] = {
                "node_count": len(graph_data.get("nodes", [])),
                "edge_count": len(graph_data.get("edges", [])),
            }

        return result

    # ═══════════════════════════════════════════════════════════════════
    # BATCH STAGED KNOWLEDGE BUILD
    # ═══════════════════════════════════════════════════════════════════

    def build_knowledge_batch(
        self,
        filenames: List[str],
        stage: BUILD_STAGE = "both",
        rerun: bool = False,
        min_frequency: int = 2,
        use_multiprocess: bool = True,
        graph_max_entities: int = 100,
        graph_min_cooccurrence: float = 0.1,
        graph_max_edges_per_node: int = 8,
        max_workers: Optional[int] = None,
        on_progress: Optional[Any] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build knowledge (wiki + graph) for multiple files in parallel.

        Parameters: (same as build_knowledge + batch params)
            filenames:      List of files to process
            stage:          "wiki" | "graph" | "both"
            rerun:          Force rebuild even if cached
            min_frequency:  Minimum mentions for entity retention
            use_multiprocess: Use multiprocessing for spaCy extraction
            graph_max_entities:      Max entities in graph
            graph_min_cooccurrence:  Min co-occurrence for edges
            graph_max_edges_per_node: Max edges per node
            max_workers:    Thread pool size
            on_progress:    Callback(filename, result_dict)

        Returns:
            Dict mapping filename -> build_knowledge result
        """
        workers = max_workers or self.max_workers
        results: Dict[str, Dict[str, Any]] = {}

        def _build_one(fname: str) -> Dict[str, Any]:
            result = self.build_knowledge(
                filename=fname,
                stage=stage,
                rerun=rerun,
                min_frequency=min_frequency,
                use_multiprocess=use_multiprocess,
                graph_max_entities=graph_max_entities,
                graph_min_cooccurrence=graph_min_cooccurrence,
                graph_max_edges_per_node=graph_max_edges_per_node,
            )
            if on_progress:
                on_progress(fname, result)
            return result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_build_one, f): f for f in filenames}
            for future in as_completed(futures):
                result = future.result()
                fname = result["filename"]
                results[fname] = result

        # Summary logging
        total = len(filenames)
        wiki_ok = sum(1 for r in results.values() if r.get("wiki_index") and not isinstance(r.get("wiki_index"), NullWikiIndex))
        graph_ok = sum(1 for r in results.values() if r.get("graph_html"))
        errors = sum(1 for r in results.values() if r.get("errors"))

        darwin.log(
            f"Knowledge batch: {wiki_ok}/{total} wiki, {graph_ok}/{total} graph, {errors} errors",
            LogType.SUCCESS if errors == 0 else LogType.WARNING
        )

        return results

    # ═══════════════════════════════════════════════════════════════════
    # MULTI-FILE ENTITY LISTING WITH OVERLAP DETECTION
    # ═══════════════════════════════════════════════════════════════════

    def list_entities_for_files(
        self,
        filenames: List[str],
        entity_type: Optional[str] = None,
        sort_by: Literal["frequency", "spread", "salience", "name", "file_count"] = "salience",
        limit: int = 50,
        min_frequency: int = 1,
        min_files_present: int = 1,
        include_overlap_analysis: bool = True,
    ) -> MultiFileEntityResult:
        """
        List entities across multiple files with rich overlap detection.

        Parameters:
            filenames:              List of files to query
            entity_type:            Filter by entity type (e.g., "PERSON")
            sort_by:                "frequency" | "spread" | "salience" | "name" | "file_count"
            limit:                  Max entities to return in merged view
            min_frequency:          Minimum total mentions across all files
            min_files_present:      Minimum number of files an entity must appear in
            include_overlap_analysis: Compute overlapping/unique entity sets

        Returns:
            MultiFileEntityResult with per-file, merged, overlap, and unique views
        """
        result = MultiFileEntityResult()

        # Load all files into a merged WikiIndex
        wiki = WikiIndex(self.store)
        for fname in filenames:
            wiki._load_file(fname)

        # ── Per-file entity lists ─────────────────────────────────────
        for fname in filenames:
            file_entities = wiki.list_entities(
                filenames=[fname],
                entity_type=entity_type,
                sort_by=sort_by if sort_by != "file_count" else "salience",
                limit=999999,  # Get all for analysis
                min_frequency=min_frequency,
            )
            result.per_file[fname] = file_entities

        # ── Merged view: all entities across all files ──────────────
        merged = wiki.list_entities(
            filenames=filenames,
            entity_type=entity_type,
            sort_by=sort_by if sort_by != "file_count" else "salience",
            limit=999999,
            min_frequency=min_frequency,
        )

        # Filter by min_files_present
        if min_files_present > 1:
            merged = [e for e in merged if len(e.doc_ids) >= min_files_present]

        # Apply limit to merged view
        result.merged_entities = merged[:limit]
        result.total_entities_across_all = len(merged)
        result.total_unique_entities = len({e.entity_id for e in merged})

        # ── Overlap analysis ────────────────────────────────────────
        if include_overlap_analysis:
            # Group entities by ID to find which files they appear in
            entity_sources: Dict[str, Set[str]] = {}
            for fname, entities in result.per_file.items():
                for e in entities:
                    if e.entity_id not in entity_sources:
                        entity_sources[e.entity_id] = set()
                    entity_sources[e.entity_id].add(fname)

            # Build overlap map: entities appearing in 2+ files
            for eid, sources in entity_sources.items():
                if len(sources) >= 2:
                    # Find the canonical record from merged view
                    rec = next((e for e in merged if e.entity_id == eid), None)
                    if rec:
                        result.overlapping[eid] = {
                            "canonical_name": rec.canonical_name,
                            "type": rec.type,
                            "sources": sorted(sources),
                            "source_count": len(sources),
                            "total_mentions": rec.frequency,
                            "salience": rec.salience,
                        }

            result.overlap_count = len(result.overlapping)

            # ── Unique per file ─────────────────────────────────────
            for fname in filenames:
                unique = []
                for e in result.per_file[fname]:
                    sources = entity_sources.get(e.entity_id, set())
                    if len(sources) == 1 and fname in sources:
                        unique.append(e)
                result.unique_per_file[fname] = unique

        # Re-sort merged if sort_by is file_count
        if sort_by == "file_count":
            result.merged_entities.sort(key=lambda e: (-len(e.doc_ids), -e.frequency))

        return result

    # ═══════════════════════════════════════════════════════════════════
    # EXISTING GRAPH METHODS (preserved)
    # ═══════════════════════════════════════════════════════════════════

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
        html = gb.build_interactive_html(
            filename,
            max_entities=max_entities,
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
            self.store.save_graph_html(filename, html)
            data = gb.build_graph_data(filename, max_entities=max_entities,
                                    min_cooccurrence=0.0, max_edges_per_node=30)
            self.store.save_graph(filename, data["nodes"], data["edges"])

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
        """
        Load a real wiki from the provided filenames, or fall back to
        the first available indexed file in the store.
        """
        if filenames:
            for f in filenames:
                wiki = self.get_wiki(f)
                if not isinstance(wiki, NullWikiIndex):
                    return wiki

        try:
            all_files = self.store.list_indexed_files()
            for f in all_files:
                wiki = self.get_wiki(f)
                if not isinstance(wiki, NullWikiIndex):
                    return wiki
        except Exception:
            pass

        return None

    @staticmethod
    def _columns_to_records(columns: dict) -> list[dict]:
        """Convert columnar dict (from PyArrow to_pydict) back to list of records."""
        if not columns:
            return []
        keys = list(columns.keys())
        length = len(columns[keys[0]])
        return [{k: columns[k][i] for k in keys} for i in range(length)]