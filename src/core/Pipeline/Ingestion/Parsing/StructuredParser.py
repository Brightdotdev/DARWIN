"""
Structured data parser — extracts records from JSON arrays and CSV/TSV files.

Bypasses markitdown entirely. Returns a StructuredDocument with:
- records: list of dicts (each row/object)
- schema: union of all keys with inferred types
- metadata: file info
"""

from __future__ import annotations

import csv
import io
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class StructuredDocument:
    """Parsed structured data from a JSON array or CSV file."""
    filename: str
    records: List[Dict[str, Any]]
    schema: Dict[str, Optional[str]]   # key -> inferred type name
    record_count: int
    file_type: str                      # "json", "csv", "tsv"
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None


def _infer_type(value: Any) -> str:
    """Infer a type label from a Python value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "str"


def _infer_schema(records: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """
    Extract the union of all keys across records.
    For each key, store the most common non-null type.
    """
    if not records:
        return {}

    type_counts: Dict[str, Dict[str, int]] = {}
    for record in records:
        for key, value in record.items():
            if key not in type_counts:
                type_counts[key] = {}
            t = _infer_type(value)
            type_counts[key][t] = type_counts[key].get(t, 0) + 1

    schema = {}
    for key, counts in type_counts.items():
        # Pick the most common non-null type
        non_null = {t: c for t, c in counts.items() if t != "null"}
        if non_null:
            schema[key] = max(non_null, key=non_null.get)
        else:
            schema[key] = "null"

    return schema


def _has_header(rows: List[List[str]]) -> bool:
    """
    Heuristic: check if the first row looks like a header.
    A header row has no numeric-only cells and differs from the second row.
    """
    if len(rows) < 2:
        return True  # assume header for single-row CSV

    header = rows[0]
    data_row = rows[1]

    # If any cell in the data row is numeric but the header cell is not, likely a header
    for h, d in zip(header, data_row):
        h_stripped = h.strip()
        d_stripped = d.strip()
        # Header is non-numeric, data is numeric -> likely a header
        if h_stripped and d_stripped:
            try:
                float(d_stripped)
                try:
                    float(h_stripped)
                except ValueError:
                    return True  # header is non-numeric, data is numeric
            except ValueError:
                pass

    # If all header cells are unique and not repeated in data, likely a header
    header_set = set(h.strip().lower() for h in header if h.strip())
    data_set = set(d.strip().lower() for d in data_row if d.strip())
    if header_set and header_set.isdisjoint(data_set):
        return True

    return False


def parse_json(path: Path, filename: str) -> StructuredDocument:
    """Parse a JSON file. Expects a top-level array of objects."""
    try:
        raw = path.read_text(encoding="utf-8-sig")
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return StructuredDocument(
            filename=filename, records=[], schema={},
            record_count=0, file_type="json",
            error=f"Invalid JSON: {e}",
        )
    except Exception as e:
        return StructuredDocument(
            filename=filename, records=[], schema={},
            record_count=0, file_type="json",
            error=f"Failed to read JSON: {e}",
        )

    if not isinstance(data, list):
        return StructuredDocument(
            filename=filename, records=[], schema={},
            record_count=0, file_type="json",
            error="JSON root is not an array",
        )

    # Filter to only dict entries (objects)
    records = [r for r in data if isinstance(r, dict)]
    skipped = len(data) - len(records)

    schema = _infer_schema(records)

    metadata = {
        "total_entries": len(data),
        "valid_records": len(records),
        "skipped_non_objects": skipped,
        "file_size_bytes": path.stat().st_size,
        "content_hash": hashlib.sha256(raw.encode()).hexdigest()[:16],
    }

    return StructuredDocument(
        filename=filename,
        records=records,
        schema=schema,
        record_count=len(records),
        file_type="json",
        metadata=metadata,
    )


def parse_csv(path: Path, filename: str, delimiter: str = ",") -> StructuredDocument:
    """Parse a CSV/TSV file into records."""
    try:
        raw = path.read_text(encoding="utf-8-sig")  # utf-8-sig strips BOM automatically
    except Exception as e:
        return StructuredDocument(
            filename=filename, records=[], schema={},
            record_count=0, file_type="csv",
            error=f"Failed to read CSV: {e}",
        )

    try:
        reader = csv.DictReader(io.StringIO(raw), delimiter=delimiter)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return StructuredDocument(
                filename=filename, records=[], schema={},
                record_count=0, file_type="csv",
                error="No header row found in CSV",
            )

        records = []
        for row in reader:
            # Convert empty strings to None, try to parse numbers
            cleaned = {}
            for k, v in row.items():
                if v is None or v.strip() == "":
                    cleaned[k] = None
                else:
                    # Try numeric conversion
                    try:
                        if "." in v:
                            cleaned[k] = float(v)
                        else:
                            cleaned[k] = int(v)
                    except ValueError:
                        # Keep as string (but strip whitespace)
                        cleaned[k] = v.strip()
            records.append(cleaned)

    except Exception as e:
        return StructuredDocument(
            filename=filename, records=[], schema={},
            record_count=0, file_type="csv",
            error=f"Failed to parse CSV: {e}",
        )

    schema = _infer_schema(records)

    metadata = {
        "headers": list(fieldnames),
        "record_count": len(records),
        "file_size_bytes": path.stat().st_size,
        "delimiter": delimiter,
        "content_hash": hashlib.sha256(raw.encode()).hexdigest()[:16],
    }

    file_type = "tsv" if delimiter == "\t" else "csv"

    return StructuredDocument(
        filename=filename,
        records=records,
        schema=schema,
        record_count=len(records),
        file_type=file_type,
        metadata=metadata,
    )


def is_structured_json(path: Path) -> bool:
    """Check if a JSON file is a top-level array of objects."""
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, list) and len(data) > 0:
            return isinstance(data[0], dict)
        return False
    except Exception:
        return False


def is_structured_csv(path: Path) -> Tuple[bool, str]:
    """
    Check if a file is a CSV/TSV with a header row.
    Returns (is_structured, delimiter).
    """
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except Exception:
        return False, ","

    # Sniff delimiter
    try:
        dialect = csv.Sniffer().sniff(raw[:4096])
        delimiter = dialect.delimiter
    except csv.Error:
        # Fallback: check for tab vs comma
        first_line = raw.split("\n", 1)[0]
        if "\t" in first_line:
            delimiter = "\t"
        else:
            delimiter = ","

    try:
        reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= 5:  # sample first 6 rows
                break

        if len(rows) < 1:
            return False, delimiter

        return _has_header(rows), delimiter

    except Exception:
        return False, delimiter


def parse_structured(path: Path, filename: str | None = None) -> StructuredDocument:
    """
    Main entry point: detect format and parse.
    Returns a StructuredDocument with records and schema.
    """
    if filename is None:
        filename = path.name

    suffix = path.suffix.lower()

    if suffix == ".json":
        return parse_json(path, filename)

    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        # Also try sniffing for .csv files that might actually be TSV
        if suffix == ".csv":
            is_struct, sniffed_delim = is_structured_csv(path)
            if not is_struct:
                return StructuredDocument(
                    filename=filename, records=[], schema={},
                    record_count=0, file_type="csv",
                    error="CSV has no detectable header row",
                )
            delimiter = sniffed_delim
        return parse_csv(path, filename, delimiter=delimiter)

    if suffix in (".xlsx", ".xls", ".ods"):
        return _parse_spreadsheet(path, filename)

    return StructuredDocument(
        filename=filename, records=[], schema={},
        record_count=0, file_type="unknown",
        error=f"Unsupported structured format: {suffix}",
    )


def _parse_spreadsheet(path: Path, filename: str) -> StructuredDocument:
    """Parse spreadsheet via markitdown fallback — convert to CSV-like records."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            return StructuredDocument(
                filename=filename, records=[], schema={},
                record_count=0, file_type="xlsx",
                error="No active sheet found",
            )

        rows = list(ws.iter_rows(values_only=True))
        wb.close()

        if len(rows) < 2:
            return StructuredDocument(
                filename=filename, records=[], schema={},
                record_count=0, file_type="xlsx",
                error="Spreadsheet has fewer than 2 rows",
            )

        headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
        records = []
        for row in rows[1:]:
            record = {}
            for h, v in zip(headers, row):
                if v is None:
                    record[h] = None
                elif isinstance(v, (int, float, bool, str)):
                    record[h] = v
                else:
                    record[h] = str(v)
            records.append(record)

        schema = _infer_schema(records)

        return StructuredDocument(
            filename=filename,
            records=records,
            schema=schema,
            record_count=len(records),
            file_type="xlsx",
            metadata={"headers": headers, "sheet": ws.title},
        )

    except ImportError:
        return StructuredDocument(
            filename=filename, records=[], schema={},
            record_count=0, file_type="xlsx",
            error="openpyxl not installed — cannot parse spreadsheets",
        )
    except Exception as e:
        return StructuredDocument(
            filename=filename, records=[], schema={},
            record_count=0, file_type="xlsx",
            error=f"Failed to parse spreadsheet: {e}",
        )
