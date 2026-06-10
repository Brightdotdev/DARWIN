# ============================================================
# src/core/Parsing/Extractors/table_detection.py
# ============================================================

import fitz
from typing import List, Optional, Tuple, Dict
from collections import Counter

from src.features.Ingestion.Parsing.Extractors.Constants import (
    TABLE_X_SPREAD, TABLE_MIN_SPANS, TABLE_MIN_COLS, TABLE_MIN_ROWS,
    TABLE_Y_GAP_RATIO,
)


# ── Heuristic fallback (same logic, improved documentation) ──────────

def _cluster_rows(all_spans: List[dict], y_threshold: float = 3.0) -> List[List[dict]]:
    spans = [s for s in all_spans if s.get("text", "").strip()]
    if not spans:
        return []
    spans.sort(key=lambda s: s["bbox"][1])
    heights = [s["bbox"][3] - s["bbox"][1] for s in spans]
    avg_height = sum(heights) / len(heights) if heights else 10.0
    dynamic_threshold = max(y_threshold, avg_height * 0.4)

    rows: List[List[dict]] = []
    for span in spans:
        y = span["bbox"][1]
        placed = False
        for row in rows:
            row_y = sum(s["bbox"][1] for s in row) / len(row)
            if abs(y - row_y) <= dynamic_threshold:
                row.append(span)
                placed = True
                break
        if not placed:
            rows.append([span])
    return rows


def _is_table_heuristic(all_spans: List[dict], page_w: float) -> bool:
    if len(all_spans) < TABLE_MIN_SPANS:
        return False
    spans = [s for s in all_spans if s.get("text", "").strip()]
    if len(spans) < TABLE_MIN_SPANS:
        return False

    xs = [s["bbox"][0] for s in spans]
    if (max(xs) - min(xs)) / max(page_w, 1.0) < TABLE_X_SPREAD:
        return False

    rows = _cluster_rows(spans)
    if len(rows) < TABLE_MIN_ROWS:
        return False

    y_gaps = []
    for i in range(1, len(rows)):
        prev_max_y = max(s["bbox"][3] for s in rows[i - 1])
        curr_min_y = min(s["bbox"][1] for s in rows[i])
        gap = curr_min_y - prev_max_y
        if gap > 0:
            y_gaps.append(gap)
    if y_gaps and max(y_gaps) > sum(y_gaps) / len(y_gaps) * TABLE_Y_GAP_RATIO:
        return False

    spans_per_row = [len(row) for row in rows]
    avg = sum(spans_per_row) / len(spans_per_row)
    variance = sum((s - avg) ** 2 for s in spans_per_row) / len(spans_per_row)
    if variance > avg * 2:
        return False

    BUCKET = 5.0
    def bucket_x(x):
        return int(x / BUCKET)
    row_x_sets = [tuple(sorted(bucket_x(s["bbox"][0]) for s in row)) for row in rows]
    pattern_counts = Counter(row_x_sets)
    if pattern_counts.most_common(1)[0][1] < len(rows) * 0.5:
        return False

    if len(set(bucket_x(x) for x in xs)) < TABLE_MIN_COLS:
        return False

    return True


def _format_table_markdown(rows: List[List[dict]]) -> str:
    """Build a proper Markdown table from clustered rows."""
    # Sort each row by x
    sorted_rows = []
    for row in rows:
        sorted_row = sorted(row, key=lambda s: s["bbox"][0])
        cells = [s["text"].strip() for s in sorted_row if s["text"].strip()]
        if cells:
            sorted_rows.append(cells)
    if not sorted_rows:
        return ""

    # Determine column count (max across rows)
    max_cols = max(len(r) for r in sorted_rows)
    # Pad shorter rows with empty strings
    for r in sorted_rows:
        r.extend([""] * (max_cols - len(r)))

    header = sorted_rows[0]
    separator = ["---"] * max_cols
    body = sorted_rows[1:] if len(sorted_rows) > 1 else []

    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(separator) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def detect_and_format_table(
    page: fitz.Page,
    block_spans: List[dict],
    page_w: float,
    use_builtin: bool = True,
) -> Optional[str]:
    """
    Try to detect and format a table. If successful, returns a Markdown string;
    otherwise None. Prefers `page.find_tables()` when available.
    """
    # Try built-in table detection (PyMuPDF >= 1.19.0)
    if use_builtin and hasattr(page, "find_tables"):
        tables = page.find_tables()
        if tables.tables:
            # Extract the first table that overlaps the block's bbox (approximate)
            # For simplicity, we take the first table on the page, assuming one per block.
            # A more precise approach would check bounding box overlap.
            table = tables.tables[0]
            rows = table.extract()
            if rows and len(rows) >= TABLE_MIN_ROWS:
                # Convert to Markdown
                header = rows[0] if rows else []
                if not header:
                    return None
                col_count = len(header)
                lines = ["| " + " | ".join(str(c) for c in header) + " |",
                         "| " + " | ".join(["---"] * col_count) + " |"]
                for row in rows[1:]:
                    padded = list(row) + [""] * (col_count - len(row))
                    lines.append("| " + " | ".join(str(c) for c in padded[:col_count]) + " |")
                return "\n".join(lines)
            return None

    # Fallback to heuristic
    if _is_table_heuristic(block_spans, page_w):
        rows = _cluster_rows(block_spans)
        formatted = _format_table_markdown(rows)
        if formatted:
            return formatted
    return None