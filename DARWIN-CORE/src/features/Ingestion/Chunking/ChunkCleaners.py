"""
ChunkCleaners.py — Text normalisation for embedding and BM25 pipelines.

Cleaning levels
───────────────
plain_text       Aggressive: strip all markdown/formatting. For models trained on
                 plain prose (e5-*, ada-002, embed-*).

semantic_markdown Preserve meaningful structure (headings, bold, code, tables).
                 For models trained on mixed corpora (bge-*, GritLM, voyage,
                 text-embedding-3-*).

Both modes share the same pre-pass that removes extractor artefacts and
PyMuPDF4LLM injection tokens before any model-specific logic runs.
"""

import re
import unicodedata

from src.features.Ingestion.Chunking.ChunkConfig import get_chunk_config

# ─────────────────────────────────────────────────────────────────────────────
# Compiled regex library
# ─────────────────────────────────────────────────────────────────────────────

# Markdown structural elements
_HEADING_RE   = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BOLD_RE      = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC_RE    = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_CODE_INLINE_RE = re.compile(r"`[^`\n]+`")
_LINK_RE      = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")   # group 1=text, 2=url
_IMAGE_MD_RE  = re.compile(r"!\[([^\]]*)\]\([^)]*\)")    # markdown images

# HTML
_HTML_TAG_RE  = re.compile(r"</?\w+[^>]*>|<!--.*?-->", re.DOTALL)

# ── PyMuPDF4LLM injection tokens ─────────────────────────────────────────────
# These are emitted by pymupdf4llm when OCR/rendering partially fails.
# Strip them unconditionally in all modes.
_PYMUPDF_OCR_FAIL_RE = re.compile(
    r"\*{0,2}"                            # optional bold markers around it
    r"-{3,}\s*"
    r"(?:"
    r"End\s+of\s+picture\s+text"
    r"|Picture"                           # "**----- Picture -----**"
    r"|End\s+of\s+page"
    r"|Start\s+of\s+page"
    r"|Image\s+text"
    r"|No\s+text\s+(?:found|detected)"
    r")"
    r"\s*-{3,}"
    r"\*{0,2}",
    re.IGNORECASE,
)

# pymupdf4llm wraps OCR attempt text between these sentinels
_PYMUPDF_PICTURE_BLOCK_RE = re.compile(
    r"\*{0,2}-{3,}\s*Picture\s*-{3,}\*{0,2}"   # opening sentinel
    r"[\s\S]*?"
    r"(?:"
    r"\*{0,2}-{3,}\s*End\s+of\s+picture\s+text\s*-{3,}\*{0,2}"  # closing sentinel
    r"|(?=\*{0,2}-{3,})"                         # or next sentinel
    r"|$"
    r")",
    re.IGNORECASE,
)

# pymupdf4llm debug / internal markers  (==>...<==)
_PYMUPDF_ARROW_MARKER_RE = re.compile(r"\*\*==>.*?<==\*\*", re.DOTALL)

# Our own extractor tokens
_PAGE_HEADER_RE    = re.compile(r"^\[Page\s+[^\]]+\]\s*$", re.MULTILINE)


# Generic horizontal rules that pymupdf4llm emits as page separators
_HRULE_RE          = re.compile(r"^-{3,}\s*$", re.MULTILINE)


# Unicode noise
_SOFT_HYPHEN_RE    = re.compile(r"\xad")          # U+00AD soft hyphen
_ZERO_WIDTH_RE     = re.compile(r"[\u200b\u200c\u200d\ufeff]")  # ZWS, ZWJ, BOM
_NULL_RE           = re.compile(r"\x00")
_FORM_FEED_RE      = re.compile(r"\f")

# Spurious artefacts from fitz text extraction
_REPEATED_DASH_RE  = re.compile(r"-{4,}")          # long dash runs not caught above
_EMPTY_BOLD_RE     = re.compile(r"\*\*\s*\*\*")    # empty bold ****


# ─────────────────────────────────────────────────────────────────────────────
# Table normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_table(table_text: str, for_embedding: bool = True) -> str:
    """
    Convert a markdown/pipe table (including the broken multi-line-cell style
    that pymupdf4llm produces) into a flat, tokenizer-friendly representation.

    Input example (pymupdf4llm output):
        |**Algorithm**|**Details**|**MMLU-Pro**|
        |---|---|---|
        |BF16|—|83.49|
        |Default NVFP4|Static per-tensor scales are com-|82.99|
        ||puted using max-value calibration;||
        ||per-block scales are computed dy-||
        ||namically from block maximum val-||
        ||ues.|||

    Output (flat):
        Algorithm | Details | MMLU-Pro
        BF16 | — | 83.49
        Default NVFP4 | Static per-tensor scales are computed using max-value
          calibration; per-block scales are computed dynamically from block
          maximum values. | 82.99
    """
    lines = table_text.strip().splitlines()
    raw_rows: list[list[str]] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Drop pure separator rows
        if re.match(r"^\|[-:\s|]+\|$", line):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            raw_rows.append(cells)
        else:
            # Non-pipe line inside a table block: treat as continuation of last row
            if raw_rows:
                # Append to the last non-empty cell of the last row
                for i in range(len(raw_rows[-1]) - 1, -1, -1):
                    if raw_rows[-1][i]:
                        raw_rows[-1][i] += " " + line
                        break
                else:
                    raw_rows[-1][-1] += " " + line

    if not raw_rows:
        return ""

    # Merge continuation rows: a row where ALL cells are either empty or
    # continuation text (i.e. the first cell is empty) belongs to the row above.
    merged: list[list[str]] = []
    for row in raw_rows:
        is_continuation = row and not row[0].strip()
        if is_continuation and merged:
            for i, cell in enumerate(row):
                if cell.strip() and i < len(merged[-1]):
                    merged[-1][i] = (merged[-1][i] + " " + cell).strip()
        else:
            merged.append([c for c in row])

    # Strip markdown bold/italic from cells
    def _clean_cell(c: str) -> str:
        c = _BOLD_RE.sub(r"\1", c)
        c = _ITALIC_RE.sub(r"\1", c)
        # Rejoin words hyphenated across lines
        c = re.sub(r"-\s+", "", c)
        return c.strip()

    # Determine column count from header row
    n_cols = max((len(r) for r in merged), default=0)

    # Pad/trim all rows to same width and clean cells
    clean_rows = []
    for row in merged:
        row = [row[i] if i < len(row) else "" for i in range(n_cols)]
        row = [_clean_cell(c) for c in row]
        # Skip rows that are entirely empty after cleaning
        if any(c for c in row):
            clean_rows.append(row)

    if not clean_rows:
        return ""

    if for_embedding:
        # Flat prose format: "Col1 | Col2 | Col3\nVal1 | Val2 | Val3"
        return "\n".join(" | ".join(c for c in row if c) for row in clean_rows)
    else:
        # Same — we don't produce HTML or RST here; keep it simple for BM25 too
        return "\n".join(" | ".join(c for c in row if c) for row in clean_rows)


def _process_tables(text: str, for_embedding: bool = True) -> str:
    """
    Find and normalise all table blocks in text.
    Handles three formats:
      1. Our extractor's  [TABLE]...[/TABLE]  wrappers
      2. Our extractor's  [Table]\n...         (embedding variant, no close tag)
      3. Raw pipe tables from pymupdf4llm
    """
    # 1. [TABLE]...[/TABLE]
    def _replace_tagged(m: re.Match) -> str:
        inner = m.group(1).strip()
        flat  = _normalise_table(inner, for_embedding)
        return f"\n{flat}\n" if flat else "\n"

    text = re.sub(
        r"\[TABLE\]([\s\S]*?)\[/TABLE\]",
        _replace_tagged,
        text,
        flags=re.IGNORECASE,
    )

    # 2. [Table]\n... (runs until next blank line or next block marker)
    def _replace_table_marker(m: re.Match) -> str:
        return "\n"   # marker itself already stripped; body handled by pipe-table pass

    text = re.sub(r"\[Table\]\s*\n?", "", text, flags=re.IGNORECASE)

    # 3. Raw pipe tables: consecutive lines starting with |
    def _replace_pipe_table(m: re.Match) -> str:
        flat = _normalise_table(m.group(0), for_embedding)
        return f"\n{flat}\n" if flat else "\n"

    text = re.sub(
        r"(?:^\|.+\n?){2,}",   # 2+ consecutive pipe lines
        _replace_pipe_table,
        text,
        flags=re.MULTILINE,
    )

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Shared pre-pass: remove extractor + PyMuPDF4LLM artefacts
# ─────────────────────────────────────────────────────────────────────────────

def _remove_artefacts(text: str) -> str:
    """
    Strip all known injection tokens, OCR failure markers, and Unicode noise.
    This runs before any model-specific cleaning so both modes start clean.
    """
    # Unicode normalisation + line-ending consistency
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\r\n?", "\n", text)

    # Unicode noise
    text = _SOFT_HYPHEN_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _NULL_RE.sub("", text)
    text = _FORM_FEED_RE.sub("\n", text)

    # PyMuPDF4LLM: full picture blocks (sentinel-wrapped OCR text) first
    text = _PYMUPDF_PICTURE_BLOCK_RE.sub("\n", text)
    # PyMuPDF4LLM: remaining lone sentinels (e.g. closing tag left over)
    text = _PYMUPDF_OCR_FAIL_RE.sub("", text)
    # PyMuPDF4LLM: arrow debug markers
    text = _PYMUPDF_ARROW_MARKER_RE.sub("", text)

    # Our extractor: [Page N] headers — converted to a blank line boundary
    text = _PAGE_HEADER_RE.sub("\n", text)
    # Our extractor: image slots

    # Horizontal rules (page separators from pymupdf4llm and our extractor)
    text = _HRULE_RE.sub("\n", text)

    # Spurious artefacts
    text = _EMPTY_BOLD_RE.sub("", text)
    text = _REPEATED_DASH_RE.sub("—", text)   # collapse long dashes to em-dash

    # Collapse excess whitespace introduced by removals
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text




# ─────────────────────────────────────────────────────────────────────────────
# Mode-specific cleaners
# ─────────────────────────────────────────────────────────────────────────────

def _clean_plain_text(text: str, preserve_line_breaks: bool = True) -> str:
    """
    Aggressive cleaning for models that don't understand formatting.
    Used for: e5-*, embed-english, embed-multilingual, ada-002
    """
    # Tables → flat before stripping pipes
    text = _process_tables(text, for_embedding=False)



    # Code blocks: remove entirely
    text = _CODE_FENCE_RE.sub("", text)
    text = _CODE_INLINE_RE.sub("", text)

    # Images: already removed by artefact pass; handle any remaining markdown images
    text = _IMAGE_MD_RE.sub("", text)

    # Links: keep anchor text, drop URL
    text = _LINK_RE.sub(r"\1", text)

    # Strip formatting markers
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)

    # HTML
    text = _HTML_TAG_RE.sub("", text)

    # Headers: remove # symbols, keep text
    text = _HEADING_RE.sub("", text)

    # Lists: normalise bullets
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "• ", text, flags=re.MULTILINE)

    # Remaining special characters
    text = re.sub(r"[#>~|\\]", " ", text)

    # Whitespace
    if preserve_line_breaks:
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)

    return text.strip()


def _clean_semantic_markdown(text: str, preserve_line_breaks: bool = True) -> str:
    """
    Preserve semantic structure for models that understand markdown.
    Used for: bge-*, GritLM-*, voyage, text-embedding-3-*
    """
    # Tables → normalised flat format (still pipe-separated for readability)
    text = _process_tables(text, for_embedding=True)
    

    # Markdown images: extract alt text as semantic hint
    text = _IMAGE_MD_RE.sub(_extract_image_semantic, text)

    # Links: "anchor text (url)" — fix: _LINK_RE has 2 groups so \2 is valid
    text = _LINK_RE.sub(r"\1 (\2)", text)

    # Keep bold and italic — they signal emphasis in trained models
    # Clean up degenerate spacing inside markers
    text = re.sub(r"\*\*\s+", "**", text)
    text = re.sub(r"\s+\*\*", "**", text)

    # Normalize heading spacing
    text = re.sub(r"^(#{1,6})\s*", r"\1 ", text, flags=re.MULTILINE)

    # Lists: normalise but keep markdown markers
    text = re.sub(r"^\s*[-*]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(\d+)\.\s+", r"\1. ", text, flags=re.MULTILINE)

    # HTML and custom markers
    text = _HTML_TAG_RE.sub("", text)

    # Strip characters that add noise without semantic value
    text = re.sub(r"[>~\\]", " ", text)

    # Whitespace
    if preserve_line_breaks:
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)

    return text.strip()


def _extract_image_semantic(match: re.Match) -> str:
    """Return a short semantic placeholder for an image markdown node."""
    alt = match.group(1).strip() if match.group(1) else ""
    if alt and not alt.isspace():
        return f"[Image: {alt}]"
    return "[Image]"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────



def prepare_embedding_text(
    text: str,
    preserve_line_breaks: bool = True,
) -> str:
    """
    Prepare text for embedding based on the currently loaded model.

    Pipeline:
      1. _remove_artefacts  — shared pre-pass
      2. mode-specific cleaner (plain_text or semantic_markdown)

    Args:
        text:                Raw markdown/formatted text from extractor.
        preserve_line_breaks: Keep paragraph structure (recommended for chunkers).

    Returns:
        Cleaned text optimised for the active embedding model.
    """
    if not text or not text.strip():
        return ""

    text = _remove_artefacts(text)
    config = get_chunk_config()

    if config.category == "plain_text":
        return _clean_plain_text(text, preserve_line_breaks)
    else:
        return _clean_semantic_markdown(text, preserve_line_breaks)
