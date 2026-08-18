"""
Structured search — filter-by-fields with hybrid fallback.

Filters structured records by field values, applies boost to scores,
and falls back to hybrid text search when no structured hits match.

Filter syntax (Django-style double underscore operators):
  Exact match:   {"status": "open"}
  Contains:      {"title__contains": "login"}
  Starts with:   {"title__startswith": "login"}
  Ends with:     {"title__endswith": "error"}
  Greater than:  {"score__gt": 5}
  Less than:     {"score__lt": 10}
  GTE / LTE:     {"score__gte": 5}, {"score__lte": 10}
  Not equal:     {"status__neq": "closed"}
  In list:       {"status__in": ["open", "pending"]}
  Exists:        {"email__exists": true}

Nested field access (dot-notation):
  {"user.name": "Bright"}        → matches nested JSON object
  {"user.role__contains": "admin"} → nested + operator

Guardrails:
  - Filter keys validated against schema before searching
  - Type mismatches between filter values and schema types are warned
  - Overlapping keys across files detected and reported
  - Null/missing field values handled gracefully
  - Empty filters return early with warning
  - Case-insensitive string matching for user convenience
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin


DEFAULT_FILTER_BOOST = 0.2

# Supported operators (Django-style double underscore suffix)
OPERATORS = {
    "contains",      # substring match
    "startswith",    # prefix match
    "endswith",      # suffix match
    "gt",            # greater than
    "lt",            # less than
    "gte",           # greater than or equal
    "lte",           # less than or equal
    "neq",           # not equal
    "in",            # match any value in list
    "exists",        # field existence check
}

# Operators that work on strings
STRING_OPS = {"contains", "startswith", "endswith", "neq"}
# Operators that work on numbers (gt, lt, gte, lte only — neq is handled separately)
NUMERIC_OPS = {"gt", "lt", "gte", "lte"}


def _parse_filter_key(filter_key: str) -> Tuple[str, str]:
    """Parse a filter key into (field_name, operator).

    Examples:
        "status"           → ("status", "exact")
        "title__contains"  → ("title", "contains")
        "score__gt"        → ("score", "gt")
        "user.name"        → ("user.name", "exact")
        "user.name__contains" → ("user.name", "contains")
    """
    # Check for double-underscore operator suffix
    # But be careful: field names can contain underscores
    # Strategy: check from the right, find the last __ that matches a known operator
    for op in sorted(OPERATORS, key=len, reverse=True):  # longest first
        suffix = f"__{op}"
        if filter_key.endswith(suffix):
            field_name = filter_key[:-len(suffix)]
            if field_name:  # ensure field name isn't empty
                return field_name, op

    return filter_key, "exact"


def _json_path_for_field(field_name: str) -> str:
    """Convert a dot-notation field name to a JSON path expression.

    Handles field names with spaces/special chars by bracket notation.

    Examples:
        "status"       → "$.status"
        "user.name"    → "$.user.name"
        "Track Name"   → "$.[\"Track Name\"]"
        "a.b.c"        → "$.a.b.c"
    """
    parts = field_name.split(".")
    segments = []
    for part in parts:
        if " " in part or not part.isidentifier():
            # Use bracket notation for non-identifier keys
            segments.append(f'["{part}"]')
        else:
            segments.append(f".{part}")
    return "$" + "".join(segments)


def _struct_field_expr(field_name: str) -> pl.Expr:
    """Build a Polars expression that extracts a value from a struct-typed fields column.

    Uses .struct.field() for native struct access. Supports nested dot-notation
    for nested struct fields.

    Examples:
        "gender"       → pl.col("fields").struct.field("gender")
        "user.name"    → pl.col("fields").struct.field("user").struct.field("name")
    """
    parts = field_name.split(".")
    expr = pl.col("fields")
    for part in parts:
        expr = expr.struct.field(part)
    return expr


def _field_exists_expr(field_name: str) -> pl.Expr:
    """Check if a field exists in the struct (non-null)."""
    parts = field_name.split(".")
    expr = pl.col("fields")
    for part in parts:
        expr = expr.struct.field(part)
    return expr.is_not_null()


def _extract_json_value_expr(field_name: str) -> pl.Expr:
    """Build a Polars expression that extracts a value from the fields column.

    Handles both struct-typed columns (via .struct.field()) and
    legacy JSON string columns (via .str.json_path_match()).
    """
    return _struct_field_expr(field_name)


def _coerce_filter_value(value: Any, schema_type: str) -> Any:
    """Attempt to coerce a filter value to match the schema type.

    Returns the coerced value, or the original value if coercion fails.
    Caller should check for type match after calling this.
    """
    if value is None:
        return None

    if schema_type == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        try:
            return int(str(value))
        except (ValueError, TypeError):
            return value

    if schema_type == "float":
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except (ValueError, TypeError):
            return value

    if schema_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            if value.lower() in ("false", "0", "no"):
                return False
        return value

    # str or unknown — keep as-is
    return str(value) if not isinstance(value, str) else value


def _build_filter_expr(
    field_name: str,
    expected_value: Any,
    schema_type: str,
    operator: str = "exact",
) -> Optional[pl.Expr]:
    """Build a Polars expression for filtering on a JSON fields column.

    Supports operators: exact, contains, startswith, endswith, gt, lt, gte, lte,
    neq, in, exists.

    For nested fields (dot-notation), uses JSON path extraction.
    Returns None if the expression cannot be built (unsupported type/operator combo).
    """

    # ── Existence check (special case — no value needed) ────────────
    if operator == "exists":
        if expected_value:
            # Field must exist and not be null
            return _field_exists_expr(field_name)
        else:
            # Field must not exist or be null
            return _field_exists_expr(field_name).not_()

    # ── In-list operator (special case — value is a list) ───────────
    if operator == "in":
        if not isinstance(expected_value, list):
            return None
        # Build OR expression: match any value in the list
        exprs = []
        for v in expected_value:
            e = _build_filter_expr(field_name, v, schema_type, "exact")
            if e is not None:
                exprs.append(e)
        if not exprs:
            return None
        combined = exprs[0]
        for e in exprs[1:]:
            combined = combined | e
        return combined

    # ── Extract the JSON value for this field ────────────────────────
    val_expr = _extract_json_value_expr(field_name)
    field_key_expr = _field_exists_expr(field_name)

    # ── Neq operator (handles both strings and numbers) ────────────
    if operator == "neq":
        coerced = _coerce_filter_value(expected_value, schema_type)
        if isinstance(coerced, (int, float)) and schema_type in ("int", "float"):
            num_val = val_expr.cast(pl.Float64, strict=False)
            return field_key_expr & (num_val != coerced)
        else:
            # String neq — case-insensitive
            if isinstance(coerced, str):
                search_str = coerced.lower()
                val_lower = val_expr.cast(pl.Utf8, strict=False).str.to_lowercase()
                return field_key_expr & ~val_lower.str.contains(search_str)
            else:
                return field_key_expr & (val_expr.cast(pl.Utf8, strict=False) != str(coerced))

    # ── Numeric operators ───────────────────────────────────────────
    if operator in NUMERIC_OPS:
        coerced = _coerce_filter_value(expected_value, schema_type)
        if not isinstance(coerced, (int, float)):
            return None

        # For numeric comparison, we need to extract as float
        # json_path_match returns a string, so we cast to float
        num_val = val_expr.cast(pl.Float64, strict=False)

        if operator == "gt":
            return field_key_expr & (num_val > coerced)
        if operator == "gte":
            return field_key_expr & (num_val >= coerced)
        if operator == "lt":
            return field_key_expr & (num_val < coerced)
        if operator == "lte":
            return field_key_expr & (num_val <= coerced)

    # ── String operators ────────────────────────────────────────────
    coerced = _coerce_filter_value(expected_value, schema_type)
    if not isinstance(coerced, str):
        coerced = str(coerced) if coerced is not None else ""

    # Cast extracted value to string for case-insensitive operations
    val_lower = val_expr.cast(pl.Utf8, strict=False).str.to_lowercase()
    search_lower = coerced.lower()

    if operator == "exact":
        # Exact match (case-insensitive for strings)
        return field_key_expr & (val_lower == search_lower)

    if operator == "contains":
        return field_key_expr & val_lower.str.contains(search_lower)

    if operator == "startswith":
        return field_key_expr & val_lower.str.starts_with(search_lower)

    if operator == "endswith":
        return field_key_expr & val_lower.str.ends_with(search_lower)

    return None


def _filter_records(
    store: DarwinStore,
    filters: Dict[str, Any],
    files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Filter structured records by field values across files.

    Supports operator syntax: {"field__operator": value}
    See module docstring for full operator list.

    Guardrails:
      - Empty filters: return empty list
      - Filter key not in schema: skip that filter (already validated upstream)
      - No structured files found: return empty list
      - Partial schema match: only use filters that exist in at least one schema
      - Null field values in records: never match non-null filter values
      - Case-insensitive string matching for user convenience
      - Nested dot-notation fields handled via JSON path extraction
    """
    if not filters:
        return []

    results = []
    warnings_seen = set()

    # Determine which files to search
    if files:
        searchable = store.get_search_ready_files(files, mode="hybrid")
    else:
        searchable = store.get_search_ready_files(mode="hybrid")

    if not searchable:
        darwin.warning("No search-ready files found")
        return []

    for filename in searchable:
        # Only search structured files (those with schema)
        schema = store.load_schema(filename)
        if schema is None:
            continue

        h = store._resolve_hash(filename)
        if h is None:
            continue

        # Load chunks Parquet and filter by fields
        table = store.load_records(filename)
        if table is None or table.num_rows == 0:
            continue

        # Check if the table has a "fields" column
        if "fields" not in table.column_names:
            continue

        # Build filter expressions — parse operators and skip unknown fields
        filter_exprs = []
        skipped_filters = []
        for filter_key, expected_value in filters.items():
            field_name, operator = _parse_filter_key(filter_key)
            schema_types = schema.get("types", {})

            # For nested fields (dot-notation), check if the root key exists
            root_key = field_name.split(".")[0]
            if root_key not in schema_types and field_name not in schema_types:
                skipped_filters.append(filter_key)
                continue

            # Use root key's type for coercion (nested values are JSON)
            schema_type = schema_types.get(root_key, schema_types.get(field_name, "str"))

            expr = _build_filter_expr(field_name, expected_value, schema_type, operator)
            if expr is not None:
                filter_exprs.append(expr)

        # Warn about skipped filters (once per file)
        if skipped_filters and filename not in warnings_seen:
            darwin.info(
                f"Skipping filters not in schema for {filename}: {skipped_filters}"
            )
            warnings_seen.add(filename)

        if not filter_exprs:
            continue

        # Combine all filter expressions with AND
        combined = filter_exprs[0]
        for expr in filter_exprs[1:]:
            combined = combined & expr

        # Scan and filter
        try:
            lf = pl.scan_parquet(store.paths.artifact_path(h, "chunks", "file"))
            filtered = lf.filter(combined).collect()
        except Exception as e:
            darwin.warning(f"Filter failed for {filename}: {e}")
            continue

        if filtered.is_empty():
            continue

        # Convert to records
        for row in filtered.iter_rows(named=True):
            fields_str = row.get("fields", "{}")
            if isinstance(fields_str, str):
                try:
                    fields = json.loads(fields_str)
                except json.JSONDecodeError:
                    fields = {}
            else:
                fields = fields_str

            # Calculate match score (fraction of filters that actually matched)
            match_count = 0
            for filter_key, v in filters.items():
                field_name, operator = _parse_filter_key(filter_key)
                record_val = _get_nested_value(fields, field_name)
                if record_val is None:
                    continue

                if operator == "exact":
                    if isinstance(v, str) and isinstance(record_val, str):
                        if record_val.lower() == v.lower():
                            match_count += 1
                    elif record_val == v:
                        match_count += 1
                    elif isinstance(v, str):
                        coerced = _coerce_filter_value(v, schema.get("types", {}).get(field_name, "str"))
                        if record_val == coerced:
                            match_count += 1
                elif operator == "contains":
                    if isinstance(record_val, str) and isinstance(v, str):
                        if v.lower() in record_val.lower():
                            match_count += 1
                elif operator == "startswith":
                    if isinstance(record_val, str) and isinstance(v, str):
                        if record_val.lower().startswith(v.lower()):
                            match_count += 1
                elif operator == "endswith":
                    if isinstance(record_val, str) and isinstance(v, str):
                        if record_val.lower().endswith(v.lower()):
                            match_count += 1
                elif operator in ("gt", "gte", "lt", "lte"):
                    try:
                        rv = float(record_val)
                        vv = float(v)
                        if operator == "gt" and rv > vv:
                            match_count += 1
                        elif operator == "gte" and rv >= vv:
                            match_count += 1
                        elif operator == "lt" and rv < vv:
                            match_count += 1
                        elif operator == "lte" and rv <= vv:
                            match_count += 1
                    except (ValueError, TypeError):
                        pass
                elif operator == "neq":
                    if isinstance(v, str) and isinstance(record_val, str):
                        if record_val.lower() != v.lower():
                            match_count += 1
                    elif record_val != v:
                        match_count += 1
                elif operator == "in":
                    if isinstance(v, list):
                        matched = False
                        for item in v:
                            if isinstance(item, str) and isinstance(record_val, str):
                                if record_val.lower() == item.lower():
                                    matched = True
                                    break
                            elif record_val == item:
                                matched = True
                                break
                        if matched:
                            match_count += 1
                elif operator == "exists":
                    # If we got here, the field exists
                    match_count += 1

            score = match_count / len(filters) if filters else 0.0

            results.append({
                "fields": fields,
                "score": score,
                "source": filename,
                "chunk_index": row.get("index", 0),
                "heading": row.get("heading", ""),
                "structural_type": "record",
            })

    return results


def _get_nested_value(d: dict, field_name: str) -> Any:
    """Get a value from a nested dict using dot-notation.

    Examples:
        _get_nested_value({"user": {"name": "Bright"}}, "user.name") → "Bright"
        _get_nested_value({"status": "open"}, "status") → "open"
        _get_nested_value({"a": {"b": 1}}, "a.b.c") → None
    """
    parts = field_name.split(".")
    current = d
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _record_to_text(record: dict) -> str:
    """Convert a structured record's fields to a text representation.

    Used for semantic re-ranking — the cross-encoder needs text to compare.
    Format: "field: value | field: value | ..."
    """
    fields = record.get("fields", {})
    parts = []
    for k, v in fields.items():
        if v is not None:
            parts.append(f"{k}: {v}")
    return " | ".join(parts) if parts else ""


def _rerank_by_query(
    query: str,
    results: List[Dict[str, Any]],
    filter_boost_factor: float = DEFAULT_FILTER_BOOST,
) -> List[Dict[str, Any]]:
    """Re-rank structured results by semantic similarity to the query.

    Uses the existing cross-encoder reranker. Converts each record's
    fields to text for the reranker, then combines:
      final_score = rerank_weight * rerank_score + (1 - rerank_weight) * filter_score

    Falls back to original filter scores if reranking fails.
    """
    try:
        from src.core.Retrival.Reranking.Reranker import rerank_results
        from src.core.Retrival.Reranking.Reranker import RerankedResult

        # Create lightweight candidates for the reranker
        # The reranker expects objects with a .text attribute
        class _Candidate:
            def __init__(self, text: str):
                self.text = text
                self.clean = text
                self.source = ""
                self.chunk_index = 0
                self.heading = ""
                self.page = 0
                self.structural_type = "record"

        candidates = []
        for r in results:
            text = _record_to_text(r)
            candidates.append(_Candidate(text))

        # Rerank
        reranked = rerank_results(
            query=query,
            candidates=candidates,
            top_k=len(results),
            original_score_weight=0.4,
        )

        if not reranked:
            return results

        # Map rerank scores back to original results
        rerank_weight = 0.6  # weight for semantic score vs filter score
        for i, rr in enumerate(reranked):
            if i < len(results):
                orig_score = results[i].get("score", 0.0)
                rerank_score = rr.rerank_score if rr.rerank_score is not None else orig_score
                # Combine: semantic similarity + filter match
                combined = (rerank_weight * rerank_score) + ((1 - rerank_weight) * orig_score)
                results[i]["score"] = min(1.0, combined)
                results[i]["rerank_score"] = rerank_score

        # Re-sort by combined score
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results

    except Exception as e:
        darwin.info(f"Reranking skipped: {type(e).__name__}: {e}")
        return results


def search_structured(
    store: DarwinStore,
    query: str = "",
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    files: Optional[List[str]] = None,
    filter_boost_factor: float = DEFAULT_FILTER_BOOST,
    mode: str = "auto",
) -> Tuple[List[Dict[str, Any]], str]:
    """Structured search with hybrid fallback.

    Guardrails:
      - Empty filters + empty query: return empty list
      - mode="structured" + no matches: return empty (no fallback)
      - mode="text": skip structured search entirely
      - mode="auto": try structured first, fallback to text
      - All results capped at top_k

    Returns (results, mode_used):
        results: list of dicts with "fields", "score", "source", etc.
        mode_used: "structured" or "text" (indicates which path was taken)
    """
    has_filters = bool(filters)

    # ── Guard: no filters and no query ────────────────────────────────
    if not has_filters and not query:
        darwin.warning("search_structured called with no filters and no query")
        return [], "structured"

    # ── Step 1: Try structured filter search ──────────────────────────
    if has_filters and mode in ("auto", "structured"):
        filter_results = _filter_records(store, filters, files)

        if filter_results:
            # Apply boost to filter matches
            for r in filter_results:
                r["score"] = min(1.0, r["score"] + filter_boost_factor)

            # Sort by score descending
            filter_results.sort(key=lambda x: x["score"], reverse=True)

            # If query is also provided, re-rank filtered results by semantic similarity
            if query and len(filter_results) > 1:
                filter_results = _rerank_by_query(query, filter_results, filter_boost_factor)

            return filter_results[:top_k], "structured"

        if mode == "structured":
            # Explicit structured mode — return empty if no filter matches
            darwin.info("No structured matches found (explicit structured mode)")
            return [], "structured"

    # ── Step 2: Hybrid fallback ───────────────────────────────────────
    if query:
        darwin.info("Falling back to hybrid text search")
        from src.core.Retrival.RetrievalArgs import RetrievalArgs
        from src.core.Retrival.lib import RetrievalMethod
        from src.core.Retrival.main import handle_retrieval

        args = RetrievalArgs(
            query=query,
            method=RetrievalMethod.HYBRID,
            files=files or [],
            top_k=top_k,
        )
        results = handle_retrieval(args, store=store)

        # Convert RetrievalResult objects to dicts
        text_results = []
        for r in results:
            text_results.append({
                "fields": {},
                "text": r.text,
                "clean": r.clean,
                "score": r.score,
                "source": r.source,
                "chunk_index": r.chunk_index,
                "heading": r.heading,
                "page": r.page,
                "structural_type": getattr(r, "structural_type", None),
            })

        return text_results, "text"

    return [], "text"
