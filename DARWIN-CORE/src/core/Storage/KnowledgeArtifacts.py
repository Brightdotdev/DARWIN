
from src.core.Logger.DarwinLogger import darwin, LogType

from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq
import json
from src.core.Storage.PathResolverMixin import PathResolverMixin
from src.core.Storage.StoreUtils import now

def normalize_record(obj: Any) -> dict[str, Any]:
    """
    Convert any object to a plain dict without importing Chunk, dataclasses,
    pydantic, or attrs.  Duck-typing keeps the storage layer fully decoupled.

    FIX: Added handling for objects with __slots__ (no __dict__) by iterating
    over all slot names and extracting values.
    """
    if isinstance(obj, dict):
        return dict(obj)
    
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()

    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()

    # Pydantic v1
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()

    # Dataclass (duck-typed via __dataclass_fields__)
    dc_fields = getattr(obj, "__dataclass_fields__", None)
    if dc_fields is not None:
        return {
            k: getattr(obj, k)
            for k in dc_fields
            if not k.startswith("_")
        }

    # attrs (duck-typed via __attrs_attrs__)
    attrs_attrs = getattr(obj, "__attrs_attrs__", None)
    if attrs_attrs is not None:
        return {
            a.name: getattr(obj, a.name)
            for a in attrs_attrs
        }

    # FIX: Handle objects with __slots__ (no __dict__)
    if hasattr(obj, "__slots__"):
        slots = obj.__slots__
        # __slots__ can be a string or a list/tuple of strings
        if isinstance(slots, str):
            slots = [slots]
        result = {}
        for slot in slots:
            if not slot.startswith("_"):
                try:
                    result[slot] = getattr(obj, slot)
                except AttributeError:
                    pass
        if result:
            return result

    # Generic Python object with __dict__
    if hasattr(obj, "__dict__"):
        return vars(obj)

    raise TypeError(
        f"Cannot normalize object of type {type(obj).__name__} to dict. "
        "Expected dict, dataclass, pydantic model, attrs, or object with __dict__ or __slots__."
    )


def _deep_serialize(obj: Any) -> Any:
    """Recursively convert any object to a JSON‑serializable type."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_deep_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _deep_serialize(v) for k, v in obj.items()}
    try:
        # Try duck‑typing with normalize_record (handles dataclasses, etc.)
        return _deep_serialize(normalize_record(obj))
    except TypeError:
        # Last resort: string representation
        return str(obj)



def records_to_table(records: list[dict[str, Any]]) -> pa.Table:
    """
    Convert plain dicts to an Arrow Table.
    If schema inference fails, convert **every** value to a JSON string.
    """
    if not records:
        return pa.Table.from_pydict({})
    try:
        return pa.Table.from_pylist(records)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        darwin.log(
            f"Parquet schema inference failed ({exc}); coercing ALL fields to JSON strings",
            LogType.WARNING,
        )

        def to_json_string(obj: Any) -> str:
            return json.dumps(_deep_serialize(obj))

        normalized = [{k: to_json_string(v) for k, v in record.items()} for record in records]
        return pa.Table.from_pylist(normalized)



def table_to_records(table: pa.Table) -> list[dict[str, Any]]:
    """Arrow Table → list[dict].  Symmetric with records_to_table."""
    return table.to_pylist()

# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Graph
# ═══════════════════════════════════════════════════════════════════════════

class GraphMixin(PathResolverMixin):

    def has_graph(self, filename: str) -> bool:
        return self._has_artifact(filename, "graph")

    def has_graph_html(self, filename: str) -> bool:
        return self._has_artifact(filename, "graph_vis")

    def save_graph(self, filename: str, nodes: list[Any], edges: list[Any]) -> None:
        if not nodes and not edges:
            darwin.log(f"No graph data to save for {filename}", LogType.ERROR)
            return

        d = self._dir_for(filename) / "graph"
        d.mkdir(parents=True, exist_ok=True)

        # Write nodes and edges as separate parquet files for modular access.
        # Callers can load only nodes if they don't need edge traversal.
        if nodes:
            nodes_table = records_to_table([normalize_record(n) for n in nodes])
            pq.write_table(nodes_table, d / "nodes.parquet", compression="zstd")
        if edges:
            edges_table = records_to_table([normalize_record(e) for e in edges])
            pq.write_table(edges_table, d / "edges.parquet", compression="zstd")

        self._update_meta(filename, "graph", extras={
            "graph": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "built_at": now(),
            }
        })

        record = self._record_for(filename)
        if record:
            try:
                self.registry.update_pipeline(
                    record.last_seen_path or filename,
                    "graph",
                    node_count=len(nodes),
                    edge_count=len(edges),
                )
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)

        self.invalidate(filename)
        darwin.log(f"Graph saved: {len(nodes)} nodes, {len(edges)} edges", LogType.SUCCESS)

    def save_graph_html(self, filename: str, html: str) -> None:
        if html is None:
            darwin.log(f"No graph HTML to save for {filename}", LogType.ERROR)
            return

        d = self._dir_for(filename) / "graph"
        d.mkdir(parents=True, exist_ok=True)
        (d / "graph.html").write_text(html, encoding="utf-8")
        self._update_meta(filename, "graph_vis")
        record = self._record_for(filename)
        if record:
            try:
                self.registry.update_pipeline(record.last_seen_path or filename, "graph_vis")
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)
        self.invalidate(filename)

    def load_graph(self, filename: str) -> dict:
        """Return graph as {nodes: [...], edges: [...]}."""
        cache_key = f"graph:{filename}"
        if hasattr(self, "_artifact_cache") and cache_key in self._artifact_cache:
            return self._artifact_cache[cache_key]

        d = self._dir_for(filename) / "graph"
        nodes_path = d / "nodes.parquet"
        edges_path = d / "edges.parquet"

        result: dict[str, list] = {"nodes": [], "edges": []}
        if nodes_path.exists():
            result["nodes"] = table_to_records(pq.read_table(nodes_path))
        if edges_path.exists():
            result["edges"] = table_to_records(pq.read_table(edges_path))

        if hasattr(self, "_artifact_cache"):
            self._artifact_cache[cache_key] = result
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Wiki
# ═══════════════════════════════════════════════════════════════════════════

class WikiMixin(PathResolverMixin):

    def has_wiki(self, filename: str) -> bool:
        return self._has_artifact(filename, "wiki")

    def save_wiki(self, filename: str, pages: list[Any]) -> None:
        if not pages:
            darwin.log(f"No wiki pages to save for {filename}", LogType.ERROR)
            return

        d = self._dir_for(filename) / "wiki"
        d.mkdir(parents=True, exist_ok=True)

        dicts = [normalize_record(p) for p in pages]
        table = records_to_table(dicts)
        pq.write_table(table, d / "wiki.parquet", compression="zstd")

        record = self._record_for(filename)
        if record:
            try:
                self.registry.update_pipeline(
                    record.last_seen_path or filename,
                    "wiki",
                    page_num=len(pages),
                )
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)

        self.invalidate(filename)
        darwin.log(f"Wiki saved: {len(pages)} pages", LogType.SUCCESS)

    def load_wiki(self, filename: str) -> list[dict] | None:
        path = self._dir_for(filename) / "wiki" / "wiki.parquet"
        if not path.exists():
            return None
        return table_to_records(pq.read_table(path))


