
"""
Document chunking pipeline for embedding and retrieval (RAG).

This module converts raw parsed documents into structured, embedding-ready chunks.
It is designed for use in retrieval-augmented generation (RAG) systems where
documents must be split into semantically meaningful, token-bounded segments.

Core responsibilities:
- Clean and normalize markdown-heavy text for embedding models
- Split documents into sections using Markdown headings (#, ##, ###, etc.)
- Further split large sections into token-bounded chunks
- Preserve semantic structure (headings, paragraphs, captions, tables)
- Maintain metadata for retrieval and citation (source, page, heading)
- Add overlap between chunks to preserve context across boundaries
- Merge small chunks to avoid low-quality embeddings

Chunking strategy:
1. Markdown cleaning removes formatting while preserving content meaning.
2. Documents are split into sections using heading structure.
3. Each section is converted into one or more chunks based on token limits.
4. Large paragraphs are split into sentence-level windows with overlap.
5. Special blocks like tables and captions are handled explicitly.
6. Small chunks are merged post-processing to improve embedding quality.

Key design goals:
- Keep chunks within embedding model token limits
- Preserve semantic coherence across boundaries
- Ensure retrieval-friendly metadata (heading, page, source)
- Avoid breaking tables, captions, and structured blocks
- Maintain lightweight token estimation (no external tokenizer dependency)

Outputs:
List[Chunk] where each Chunk contains:
- text: original content for LLM context
- clean: normalized text for embeddings
- index: stable ordering
- source: document identifier
- token_count: estimated size
- heading: nearest section title
- page / page_end: document location tracking
- is_overlap: whether chunk includes context overlap
"""



import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from src.lib.Terminal import TerminalUI




# ============================================================================
# Constants — overridden by caller via chunk_document() parameters
# ============================================================================

DEFAULT_MAX_TOKENS  = 350    # safe for e5-small (512 token window)
DEFAULT_MIN_TOKENS  = 80     # below this we merge with neighbour
DEFAULT_OVERLAP     = 2      # sentences of overlap between chunks

# signals from our parser — never split mid-token
TABLE_TOKEN   = "[Table]"
CAPTION_TOKEN = "[Caption:"
PAGE_SEP      = "---"


# ============================================================================
# Data model
# ============================================================================

@dataclass
class Chunk:
    text:        str     # original text → shown to user, sent to LLM
    clean:       str     # stripped plain text → fed to embedding model
    index:       int     # position in document (0-based)
    source:      str     # filename for citations
    token_count: int     # estimated tokens of clean text
    heading:     str     # nearest section heading above this chunk
    page:        int     # page number (0 if unknown)
    page_end:    int     # last page this chunk touches
    is_overlap:  bool    # True if this chunk contains overlap from previous


# ============================================================================
# Token estimation
# Deliberately simple — no tokenizer dependency.
# word_count / 0.75 approximates subword token count closely enough
# for chunking decisions. Real token count from your embedding model
# will differ by <5% for English academic text.
# ============================================================================

def _tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


# ============================================================================
# Markdown cleaning
# Strips formatting symbols from embedding_text to produce clean text.
# Preserves all actual content — removes only markdown syntax.
# ============================================================================

_LINK_RE    = re.compile(r"\[([^\]]+)\]\([^)]+\)")   # [text](url) → text
_BOLD_RE    = re.compile(r"\*\*([^*]+)\*\*")          # **bold** → bold
_ITALIC_RE  = re.compile(r"\*([^*]+)\*")              # *italic* → italic
_IMAGE_RE   = re.compile(r"!\[.*?\]\(.*?\)")           # images → remove
_CODE_RE    = re.compile(r"```[\s\S]*?```|`[^`]+`")   # code → remove
_HTML_RE    = re.compile(r"<[^>]+>|<!--.*?-->", re.DOTALL)
_MARKER_RE  = re.compile(r"==>.*?<==", re.DOTALL)     # pymupdf4llm markers


def _clean(text: str) -> str:
    """Strip markdown formatting, preserve content."""
    text = _LINK_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _IMAGE_RE.sub("", text)
    text = _CODE_RE.sub("", text)
    text = _HTML_RE.sub("", text)
    text = _MARKER_RE.sub("", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[Table\]\s*", "", text)
    text = re.sub(r"\[Caption:\s*", "", text)
    text = re.sub(r"\[Image[^\]]*\]", "", text)
    text = re.sub(r"[*_#>~|\\]", " ", text)
    text = re.sub(r"-{3,}", " ", text)          # page separators
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()





# ============================================================================
# Args
# ============================================================================

@dataclass
class ChunkingArgs:
    target:     str
    file:       Optional[str] = None
    project:    Optional[str] = None

    min_tokens: int  = DEFAULT_MIN_TOKENS
    overlap:    int  = DEFAULT_OVERLAP
    model:      str  = "e5-small"
    inspect:    bool = False
    rerun:      bool = False
    embedding_text : str = ""



# ============================================================================
# Sentence splitting
# BlingFire for maximum speed + multilingual support, regex fallback.
# ============================================================================

_BLINGFIRE_AVAILABLE = False
_BLINGFIRE_CHECKED = False


def _init_blingfire() -> bool:
    """Lazy-load BlingFire once per session."""
    global _BLINGFIRE_AVAILABLE, _BLINGFIRE_CHECKED
    if not _BLINGFIRE_CHECKED:
        try:
            from blingfire import text_to_sentences
            _BLINGFIRE_AVAILABLE = True
        except ImportError:
            _BLINGFIRE_AVAILABLE = False
        _BLINGFIRE_CHECKED = True
    return _BLINGFIRE_AVAILABLE


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using fastest available method.
    BlingFire handles 100+ languages with near-instant cold start.
    """
    if _init_blingfire():
        from blingfire import text_to_sentences
        # BlingFire returns sentences separated by \n
        result = text_to_sentences(text).strip()
        if result:
            return [s.strip() for s in result.split('\n') if s.strip()]
    
    # Regex fallback — handles common abbreviations
    abbrev = r"(?<!\b(?:Mr|Mrs|Dr|Prof|Sr|Jr|Fig|vs|etc|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec))"
    parts = re.split(abbrev + r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]

# ============================================================================
# Section splitting
# Splits embedding_text into (heading, body) pairs at ## / ### boundaries.
# Falls back to paragraph splitting if no headings found.
# ============================================================================

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def split_into_sections(text: str) -> List[Tuple[str, str, int]]:
    """
    Split text at heading boundaries.

    Returns list of (heading_text, body_text, heading_level) tuples.
    heading_level: 1 for #, 2 for ##, 3 for ### etc.
    If no headings, returns [("", full_text, 0)] — single section.

    Page separators (---) are kept in body text so page tracking works.
    """
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        return [("", text, 0)]

    sections = []

    for i, match in enumerate(matches):
        level   = len(match.group(1))
        heading = match.group(2).strip()

        # body is from end of this heading line to start of next heading
        body_start = match.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()

        sections.append((heading, body, level))

    # text before first heading (preamble) — treat as headingless section
    first_heading_start = matches[0].start()
    if first_heading_start > 0:
        preamble = text[:first_heading_start].strip()
        if preamble:
            sections.insert(0, ("", preamble, 0))

    return sections


# ============================================================================
# Page number tracking
# Our parser inserts --- as page separators.
# We track how many separators appear before each chunk to estimate page.
# ============================================================================

def count_page_seps(text: str) -> int:
    """Count page separator markers in text."""
    return len(re.findall(r"^---\s*$", text, re.MULTILINE))


# ============================================================================
# Table detection
# [Table] blocks from our parser should never be split mid-table.
# ============================================================================

def _is_table_block(text: str) -> bool:
    return text.strip().startswith(TABLE_TOKEN)


def _is_caption_block(text: str) -> bool:
    return text.strip().startswith(CAPTION_TOKEN)


# ============================================================================
# Core chunk builder
# Takes a piece of text that fits within the token budget and wraps it
# into a Chunk with all metadata attached.
# ============================================================================

def _make_chunk(
    text:       str,
    source:     str,
    index:      int,
    heading:    str,
    page:       int,
    page_end:   int,
    is_overlap: bool = False,
) -> Optional[Chunk]:
    """Build one Chunk. Returns None if text has no real content."""
    text = text.strip()
    if not text:
        return None

    clean = _clean(text)

    # skip if nothing real left after cleaning
    if len(clean.split()) < 4:
        return None

    return Chunk(
        text        = text,
        clean       = clean,
        index       = index,
        source      = source,
        token_count = _tokens(clean),
        heading     = heading,
        page        = page,
        page_end    = page_end,
        is_overlap  = is_overlap,
    )


# ============================================================================
# Sentence-level splitting with overlap
# Used when a paragraph/section is too large for the token budget.
# ============================================================================

def _split_with_overlap(
    text:       str,
    source:     str,
    index:      int,
    heading:    str,
    page:       int,
    max_tokens: int,
    overlap:    int,
) -> List[Chunk]:
    """
    Split text into sentence-windowed chunks with overlap.
    overlap = number of sentences to prepend from previous window.
    """
    sentences = _split_sentences(_clean(text))
    orig_sents = _split_sentences(text)   # parallel original text sentences

    # align lengths — fallback if sentence counts differ
    if len(orig_sents) != len(sentences):
        orig_sents = sentences

    chunks  = []
    window_clean: List[str] = []
    window_orig:  List[str] = []

    def _flush(is_overlap: bool = False) -> None:
        if not window_clean:
            return
        c = _make_chunk(
            " ".join(window_orig),
            source, index + len(chunks),
            heading, page, page,
            is_overlap,
        )
        if c:
            chunks.append(c)

    for i, (sent, orig) in enumerate(zip(sentences, orig_sents)):
        trial = window_clean + [sent]
        if _tokens(" ".join(trial)) > max_tokens and window_clean:
            _flush(is_overlap=(i > 0))
            # keep last `overlap` sentences as context for next chunk
            window_clean = window_clean[-overlap:] if overlap else []
            window_orig  = window_orig[-overlap:]  if overlap else []

        window_clean.append(sent)
        window_orig.append(orig)

    _flush(is_overlap=bool(window_clean))
    return chunks


# ============================================================================
# Section → chunks
# Handles one (heading, body) section.
# ============================================================================

def section_to_chunks(
    heading:    str,
    body:       str,
    source:     str,
    index:      int,
    page:       int,
    max_tokens: int,
    min_tokens: int,
    overlap:    int,
) -> List[Chunk]:
    """
    Convert one section into one or more chunks.

    Rules:
    - [Table] paragraphs → always their own chunk, never split
    - [Caption] paragraphs → prepend to previous chunk or own chunk
    - Section fits in budget → one chunk
    - Section too big → split paragraphs, each inherits heading
    - Paragraph still too big → sentence split with overlap
    - Tiny paragraphs → collected and merged at end
    """
    if not body.strip():
        return []

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

    # if section fits whole → one chunk
    body_clean = _clean(body)
    if _tokens(body_clean) <= max_tokens:
        c = _make_chunk(body, source, index, heading, page, page)
        return [c] if c else []

    # section too big → process paragraph by paragraph
    chunks:  List[Chunk]  = []
    pending: List[str]    = []   # accumulate small paragraphs
    pending_orig: List[str] = []

    def _flush_pending() -> None:
        if not pending:
            return
        merged = " ".join(pending_orig)
        c = _make_chunk(merged, source, index + len(chunks), heading, page, page)
        if c:
            chunks.append(c)
        pending.clear()
        pending_orig.clear()

    for para in paragraphs:
        # table block — always its own chunk, flush pending first
        if _is_table_block(para):
            _flush_pending()
            c = _make_chunk(para, source, index + len(chunks), heading, page, page)
            if c:
                chunks.append(c)
            continue

        # caption — try to attach to previous chunk, else own chunk
        if _is_caption_block(para):
            if chunks:
                last = chunks[-1]
                combined = last.text + "\n" + para
                if _tokens(_clean(combined)) <= max_tokens:
                    chunks[-1] = _make_chunk(
                        combined, source, last.index,
                        last.heading, last.page, last.page_end,
                        last.is_overlap,
                    ) or last
                    continue
            c = _make_chunk(para, source, index + len(chunks), heading, page, page)
            if c:
                chunks.append(c)
            continue

        para_tokens = _tokens(_clean(para))

        # paragraph fits in budget
        if para_tokens <= max_tokens:
            # can we merge with pending?
            pending_tokens = _tokens(" ".join(pending)) if pending else 0
            if pending_tokens + para_tokens <= max_tokens:
                pending.append(_clean(para))
                pending_orig.append(para)
            else:
                _flush_pending()
                pending.append(_clean(para))
                pending_orig.append(para)

            # if pending is big enough, flush it
            if _tokens(" ".join(pending)) >= min_tokens:
                _flush_pending()

        # paragraph too big → sentence split
        else:
            _flush_pending()
            sub = _split_with_overlap(
                para, source, index + len(chunks),
                heading, page, max_tokens, overlap,
            )
            chunks.extend(sub)

    _flush_pending()
    return chunks


# ============================================================================
# Small chunk merging
# After section processing, merge any chunks still under min_tokens.
# ============================================================================

def merge_small(
    chunks:     List[Chunk],
    min_tokens: int,
    max_tokens: int,
) -> List[Chunk]:
    """Merge consecutive undersized chunks that share the same heading."""
    if not chunks:
        return []

    merged = []
    buf: Optional[Chunk] = None

    for chunk in chunks:
        if buf is None:
            buf = chunk
            continue

        # only merge if both are small and same heading
        both_small = buf.token_count < min_tokens
        same_section = buf.heading == chunk.heading
        fits = buf.token_count + chunk.token_count <= max_tokens

        if both_small and same_section and fits:
            buf = Chunk(
                text        = buf.text  + "\n\n" + chunk.text,
                clean       = buf.clean + " "    + chunk.clean,
                index       = buf.index,
                source      = buf.source,
                token_count = buf.token_count + chunk.token_count,
                heading     = buf.heading,
                page        = buf.page,
                page_end    = chunk.page_end,
                is_overlap  = buf.is_overlap,
            )
        else:
            merged.append(buf)
            buf = chunk

    if buf:
        merged.append(buf)

    return merged