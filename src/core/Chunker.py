"""
src/core/Chunker.py
-------------------
Converts parsed markdown text into clean overlapping chunks
ready for embedding into e5-small.

Improvements over v1:
  - Strips markdown symbols before embedding (keeps original for display)
  - Tracks nearest heading above each chunk (for citations)
  - Preserves page number metadata from parser
  - Skips pure markdown noise (headings with no content, table rows, etc)
  - Cleans table markup into readable prose before chunking

Strategy:
  1. Pre-process: extract headings, clean markdown, detect page markers
  2. Split by paragraph (respects natural idea boundaries)
  3. Sweet spot (100-400 tokens) → keep as-is
  4. Too big → split into sentences with overlap
  5. Too small → buffer and merge with next paragraph
  6. Post-process: merge fragments, re-index cleanly

Two text fields per chunk:
  text      → original markdown (shown to user, sent to LLM)
  clean     → stripped plain text (used for embedding)
"""

import re
import spacy
from dataclasses import dataclass, field


# ── Constants ──────────────────────────────────────────────────────────────────

MIN_TOKENS   = 100
MAX_TOKENS   = 400
OVERLAP_SIZE = 40


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text:        str          # original text (shown to user, sent to LLM)
    clean:       str          # stripped plain text (used for embedding)
    index:       int          # position in document
    source:      str          # filename
    token_count: int          # token count of clean text
    is_overlap:  bool         # contains overlap from previous chunk
    page:        int  = 0     # page number (0 if unknown)
    page_end:    int  = 0     # last page this chunk touches
    heading:     str  = ""    # nearest section heading above this chunk


# ── Markdown cleaning ──────────────────────────────────────────────────────────

# patterns to strip entirely — pure noise for embeddings
_STRIP_PATTERNS = [
    r"!\[.*?\]\(.*?\)",           # images ![alt](url)
    r"\[.*?\]\(.*?\)",            # links [text](url) → keep text below
    r"^\s*\|.*\|\s*$",            # table rows (handled separately)
    r"^\s*[-|:]+\s*$",            # table separator rows |---|---|
    r"```[\s\S]*?```",            # code blocks
    r"`[^`]+`",                   # inline code
    r"<!--.*?-->",                # html comments
    r"<[^>]+>",                   # html tags
    r"==.*?==",                   # highlight markup
    r"~~.*?~~",                   # strikethrough
    r"\*\*\*.*?\*\*\*",           # bold italic
]

_LINK_PATTERN     = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # extract link text
_BOLD_PATTERN     = re.compile(r"\*\*([^*]+)\*\*")        # extract bold text
_ITALIC_PATTERN   = re.compile(r"\*([^*]+)\*")            # extract italic text
_HEADING_PATTERN  = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_PAGE_PATTERN     = re.compile(r"<!--\s*page[:\s]+(\d+)\s*-->", re.IGNORECASE)
_FIGURE_PATTERN   = re.compile(r"==>.*?<==")              # pymupdf4llm image markers


def _clean_markdown(text: str) -> str:
    """
    Strip markdown formatting to get clean plain text for embedding.
    Preserves actual content — removes only formatting symbols.

    Does NOT modify the original text stored in chunk.text.
    Only used for chunk.clean which feeds the embedding model.
    """
    # extract link text before removing links
    text = _LINK_PATTERN.sub(r"\1", text)

    # extract bold and italic text
    text = _BOLD_PATTERN.sub(r"\1", text)
    text = _ITALIC_PATTERN.sub(r"\1", text)

    # remove pymupdf4llm image markers
    text = _FIGURE_PATTERN.sub("", text)

    # apply strip patterns
    for pattern in _STRIP_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL)

    # remove heading markers but keep heading text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # remove remaining markdown symbols
    text = re.sub(r"[*_#>~|\\]", " ", text)

    # collapse whitespace
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _is_noise_paragraph(text: str) -> bool:
    """
    Returns True if a paragraph is pure markdown noise —
    not worth embedding or including as a concept.

    Examples:
        "## Chapter 3"              → True  (heading with no content)
        "| col1 | col2 |"           → True  (orphan table row)
        "---"                       → True  (horizontal rule)
        "![Figure 3](image.png)"   → True  (image only)
        "See figure 3 above"       → False (actual content)
    """
    stripped = text.strip()

    # empty
    if not stripped:
        return True

    # pure heading with no body (just "## Title" and nothing else)
    if re.match(r"^#{1,6}\s+\S", stripped) and "\n" not in stripped:
        # heading alone — content will appear in next paragraph
        # keep headings as context but don't chunk them alone
        return True

    # pure horizontal rule
    if re.match(r"^[-*_]{3,}$", stripped):
        return True

    # pure table separator
    if re.match(r"^[\s|:-]+$", stripped):
        return True

    # pure image
    if re.match(r"^!\[.*?\]\(.*?\)$", stripped):
        return True

    # pymupdf4llm image marker only
    if re.match(r"^==>.*?<==\s*$", stripped):
        return True

    # after cleaning, is there any real text left?
    cleaned = _clean_markdown(stripped)
    if len(cleaned.split()) < 3:
        return True

    return False


def _extract_tables_as_prose(text: str) -> str:
    """
    Convert markdown tables into readable prose sentences.
    Tables carry real information that should be embedded —
    but pipe-separated rows confuse embedding models.

    | Process    | Location   | Products     |
    | Light rxns | Thylakoid  | ATP, NADPH   |

    becomes:

    "Process is Light rxns. Location is Thylakoid. Products are ATP, NADPH."
    """
    lines  = text.split("\n")
    result = []
    table  = []

    for line in lines:
        if re.match(r"^\s*\|", line):
            table.append(line)
        else:
            if table:
                prose = _table_to_prose(table)
                if prose:
                    result.append(prose)
                table = []
            result.append(line)

    if table:
        prose = _table_to_prose(table)
        if prose:
            result.append(prose)

    return "\n".join(result)


def _table_to_prose(table_lines: list[str]) -> str:
    """Convert a list of table lines into prose sentences."""
    rows = []
    for line in table_lines:
        # skip separator rows (|---|---|)
        if re.match(r"^\s*\|[\s|:-]+\|\s*$", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    headers = rows[0] if rows else []
    data    = rows[1:] if len(rows) > 1 else []

    if not data:
        # single row table — just join cells
        return ". ".join(headers) + "."

    sentences = []
    for row in data:
        parts = []
        for i, cell in enumerate(row):
            if i < len(headers):
                parts.append(f"{headers[i]}: {cell}")
            else:
                parts.append(cell)
        sentences.append(", ".join(parts) + ".")

    return " ".join(sentences)


# ── Page number extraction ─────────────────────────────────────────────────────

def _extract_page_markers(text: str) -> dict[int, int]:
    """
    Extract page markers from pymupdf4llm output.
    pymupdf4llm inserts HTML comments like <!-- page: 12 --> at page breaks.

    Returns dict mapping character position → page number.
    """
    markers = {}
    for match in _PAGE_PATTERN.finditer(text):
        markers[match.start()] = int(match.group(1))
    return markers


def _page_at_position(pos: int, markers: dict[int, int]) -> int:
    """Find which page a character position falls on."""
    if not markers:
        return 0
    page = 0
    for marker_pos, page_num in sorted(markers.items()):
        if marker_pos <= pos:
            page = page_num
        else:
            break
    return page


# ── Heading tracking ───────────────────────────────────────────────────────────

def _extract_headings(text: str) -> list[tuple[int, str]]:
    """
    Extract all headings with their character positions.
    Returns list of (position, heading_text) sorted by position.
    """
    headings = []
    for match in _HEADING_PATTERN.finditer(text):
        headings.append((match.start(), match.group(1).strip()))
    return headings


def _heading_at_position(pos: int, headings: list[tuple[int, str]]) -> str:
    """Find the most recent heading before a character position."""
    current = ""
    for heading_pos, heading_text in headings:
        if heading_pos <= pos:
            current = heading_text
        else:
            break
    return current


# ── spaCy helpers ──────────────────────────────────────────────────────────────

def _load_spacy():
    """Lazy load spaCy — falls back gracefully if not installed."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            return spacy.load("en_core_web_md")
        except OSError:
            return None


def _is_complete_sentence(text: str, nlp) -> bool:
    text = text.strip()
    if not text:
        return False
    if nlp is not None:
        doc       = nlp(text)
        sentences = list(doc.sents)
        if not sentences:
            return False
        last = sentences[-1].text.strip()
        return bool(last) and last[-1] in ".!?"
    return bool(re.search(r'[.!?]["\']?\s*$', text))


def _split_into_sentences(text: str, nlp) -> list[str]:
    if nlp is not None:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    abbreviations = r"(?<!\b(?:Mr|Mrs|Dr|Prof|Sr|Jr|Fig|vs|etc|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec))"
    pattern       = abbreviations + r"(?<=[.!?])\s+"
    sentences     = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _token_count(text: str) -> int:
    return int(len(text.split()) / 0.75)


def _add_overlap(previous_sentences: list[str], current: str) -> str:
    if not previous_sentences:
        return current
    tail     = previous_sentences[-1][-OVERLAP_SIZE:].strip()
    combined = f"{tail} {current}"
    if _token_count(combined) > MAX_TOKENS + 50:
        return current
    return combined


# ── Core chunking ──────────────────────────────────────────────────────────────

def _chunk_paragraph(
    paragraph:   str,
    chunk_index: int,
    source:      str,
    nlp,
    page:        int,
    heading:     str,
) -> list[Chunk]:
    """
    Process one paragraph into chunks.
    Stores original text in .text, cleaned text in .clean.
    """
    paragraph = paragraph.strip()
    if not paragraph:
        return []

    clean  = _clean_markdown(paragraph)
    tokens = _token_count(clean)

    # sweet spot — one clean chunk
    if MIN_TOKENS <= tokens <= MAX_TOKENS and _is_complete_sentence(clean, nlp):
        return [Chunk(
            text        = paragraph,
            clean       = clean,
            index       = chunk_index,
            source      = source,
            token_count = tokens,
            is_overlap  = False,
            page        = page,
            page_end    = page,
            heading     = heading,
        )]

    # too big — sentence split with overlap
    if tokens > MAX_TOKENS:
        sentences     = _split_into_sentences(clean, nlp)
        orig_sentences = _split_into_sentences(paragraph, nlp)
        chunks        = []
        buffer_clean  = ""
        buffer_orig   = ""
        processed     = []

        for idx, (sentence, orig_sentence) in enumerate(
            zip(sentences, orig_sentences or sentences)
        ):
            is_first  = idx == 0
            candidate = sentence if is_first else _add_overlap(processed, sentence)
            orig_cand = orig_sentence

            buf_tokens = _token_count(buffer_clean + " " + candidate)

            if buf_tokens <= MAX_TOKENS:
                buffer_clean = (buffer_clean + " " + candidate).strip()
                buffer_orig  = (buffer_orig  + " " + orig_cand).strip()
            else:
                if buffer_clean:
                    chunks.append(Chunk(
                        text        = buffer_orig,
                        clean       = buffer_clean,
                        index       = chunk_index + len(chunks),
                        source      = source,
                        token_count = _token_count(buffer_clean),
                        is_overlap  = not is_first,
                        page        = page,
                        page_end    = page,
                        heading     = heading,
                    ))
                buffer_clean = candidate
                buffer_orig  = orig_cand

            processed.append(sentence)

        if buffer_clean.strip():
            chunks.append(Chunk(
                text        = buffer_orig,
                clean       = buffer_clean,
                index       = chunk_index + len(chunks),
                source      = source,
                token_count = _token_count(buffer_clean),
                is_overlap  = True,
                page        = page,
                page_end    = page,
                heading     = heading,
            ))

        return chunks

    # too small — return bare for merging
    return [Chunk(
        text        = paragraph,
        clean       = clean,
        index       = chunk_index,
        source      = source,
        token_count = tokens,
        is_overlap  = False,
        page        = page,
        page_end    = page,
        heading     = heading,
    )]


def _merge_small_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Merge consecutive undersized chunks."""
    if not chunks:
        return []

    merged = []
    buffer: Chunk | None = None

    for chunk in chunks:
        if buffer is None:
            buffer = chunk
            continue

        combined_tokens = buffer.token_count + chunk.token_count

        if buffer.token_count < MIN_TOKENS and combined_tokens <= MAX_TOKENS:
            buffer = Chunk(
                text        = buffer.text  + " " + chunk.text,
                clean       = buffer.clean + " " + chunk.clean,
                index       = buffer.index,
                source      = buffer.source,
                token_count = combined_tokens,
                is_overlap  = buffer.is_overlap,
                page        = buffer.page,
                page_end    = chunk.page_end,
                heading     = buffer.heading or chunk.heading,
            )
        else:
            merged.append(buffer)
            buffer = chunk

    if buffer:
        merged.append(buffer)

    return merged


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_document(
    text:   str,
    source: str = "unknown",
    pages:  dict[int, int] | None = None,
) -> list[Chunk]:
    """
    Main entry point. Takes raw markdown from pymupdf4llm/markitdown
    and returns clean overlapping chunks ready for embedding.

    Args:
        text:   raw markdown string from parser
        source: filename for citations
        pages:  optional dict mapping char position → page number
                if None, attempts to extract page markers from text

    Returns:
        list[Chunk] — each has .text (original) and .clean (for embedding)
    """
    nlp = _load_spacy()

    # extract page markers from text if not provided
    page_markers = pages or _extract_page_markers(text)

    # extract heading positions for citation context
    headings = _extract_headings(text)

    # convert tables to prose before splitting
    text = _extract_tables_as_prose(text)

    # split into paragraphs
    paragraphs_raw = text.split("\n\n")

    all_chunks:  list[Chunk] = []
    chunk_index: int         = 0

    # track character position for page/heading lookup
    char_pos = 0

    for paragraph in paragraphs_raw:
        paragraph = paragraph.strip()
        if not paragraph:
            char_pos += len(paragraph) + 2
            continue

        # skip pure noise
        if _is_noise_paragraph(paragraph):
            char_pos += len(paragraph) + 2
            continue

        # resolve page and heading at this position
        page    = _page_at_position(char_pos, page_markers)
        heading = _heading_at_position(char_pos, headings)

        new_chunks = _chunk_paragraph(
            paragraph, chunk_index, source, nlp, page, heading
        )
        all_chunks.extend(new_chunks)
        chunk_index += len(new_chunks)
        char_pos    += len(paragraph) + 2

    # merge fragments
    all_chunks = _merge_small_chunks(all_chunks)

    # re-index cleanly
    for i, chunk in enumerate(all_chunks):
        chunk.index = i

    return all_chunks