"""
ChunkerUtils.py — Document chunking pipeline for embedding and retrieval (RAG).

This module converts raw parsed documents into structured, embedding-ready chunks.
Designed for retrieval-augmented generation (RAG) systems where documents must be
split into semantically meaningful, token-bounded segments.

Core responsibilities:
- Clean and normalize markdown-heavy text for embedding models
- Split documents into sections using Markdown headings (#, ##, ###, etc.)
- Further split large sections into token-bounded chunks
- Preserve semantic structure (headings, paragraphs, captions, tables)
- Maintain metadata for retrieval and citation (source, page, heading)
- Add overlap between chunks to preserve context across boundaries
- Merge small chunks to avoid low-quality embeddings
- Attach per-chunk link metadata extracted from source document

Key design goals:
- Keep chunks within embedding model token limits
- Preserve semantic coherence across boundaries
- Ensure retrieval-friendly metadata (heading, page, source)
- Avoid breaking tables, captions, and structured blocks
- Maintain lightweight token estimation (no external tokenizer dependency)

Outputs:
    List[Chunk] where each Chunk contains:
    - text:        original content for LLM context
    - clean:       normalized text for embeddings
    - index:       stable ordering
    - source:      document identifier
    - token_count: estimated size
    - heading:     nearest section title
    - page / page_end: document location tracking
    - is_overlap:  whether chunk includes context overlap
    - links:       hyperlinks whose page falls within this chunk's page range
"""

import re
import functools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


from src.core.Chunking.ChunkCleaners import prepare_embedding_text, get_model_config

DEFAULT_MODEL = "intfloat/e5-small-v2"

# ── Block-type sentinels (must match ChunkCleaners + extractor) ───────────────
# These are present in the *raw* extractor output (before cleaning).
# The chunker reads them to make split decisions, then they are cleaned away
# by prepare_embedding_text before chunks reach the embedding model.
TABLE_OPEN_TOKEN  = "[TABLE]"          # fitz extractor display text
TABLE_CLOSE_TOKEN = "[/TABLE]"
TABLE_EMB_TOKEN   = "[Table]"          # fitz extractor embedding text
CAPTION_TOKEN     = "[Caption:"        # fitz extractor embedding text
PAGE_HEADER_RE    = re.compile(r"^\[Page\s+[^\]]+\]\s*$", re.MULTILINE)

# Legacy separator kept for backwards compat with pymupdf4llm output
_LEGACY_PAGE_SEP_RE = re.compile(r"^---\s*$", re.MULTILINE)


# ─────────────────────────────────────────────────────────────────────────────
# Token counting
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_tiktoken_encoder():
    """Load the tiktoken encoder once and cache it for the process lifetime."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def tokens(text: str) -> int:
    """
    Estimate token count.

    Uses tiktoken (cl100k_base) when available. Falls back to word-count / 0.75
    which handles Latin text well. The fallback under-counts for CJK / code-heavy
    content; treat it as a lower-bound estimate.
    """
    if not text:
        return 0
    enc = _get_tiktoken_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, int(len(text.split()) / 0.75))


def dynamic_chunk_params(model_name: str, doc_tokens: int) -> dict:
    """
    Scale chunk size to document length.
    Longer docs need smaller chunks for retrieval precision.
    """
    base         = get_model_config(model_name)
    max_capacity = base["max_tokens"]

    if doc_tokens < 2_000:
        chunk_ratio, overlap_add = 1.0, 0
    elif doc_tokens < 10_000:
        chunk_ratio, overlap_add = 0.9, 0
    elif doc_tokens < 50_000:
        chunk_ratio, overlap_add = 0.75, 1
    elif doc_tokens < 200_000:
        chunk_ratio, overlap_add = 0.6, 2
    else:
        chunk_ratio, overlap_add = 0.5, 3

    min_chunk_size = max(150, int(max_capacity * 0.4))
    optimal_max    = max(min_chunk_size, int(max_capacity * chunk_ratio))

    return {
        "max_tokens": optimal_max,
        "min_tokens": base["min_tokens"],
        "overlap":    base["overlap"] + overlap_add,
        "base_max":   max_capacity,
        "strategy":   f"{optimal_max}/{max_capacity} tokens per chunk",
    }


def get_dynamic_chunk_params_for_text(
    text: str,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """Analyse text and return optimal chunking parameters + metadata."""
    total_tokens = tokens(text)
    params       = dynamic_chunk_params(model_name, total_tokens)
    params.update({
        "doc_tokens":       total_tokens,
        "estimated_chunks": max(1, total_tokens // params["max_tokens"]),
        "model":            model_name,
    })
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkLink:
    """A hyperlink scoped to a chunk."""
    url:  str
    text: str = ""


@dataclass
class Chunk:
    text:        str             # original text → shown to user / sent to LLM
    clean:       str             # stripped text → fed to embedding model
    index:       int             # position in document (0-based)
    source:      str             # filename for citations
    token_count: int             # estimated tokens of clean text
    heading:     str             # nearest section heading above this chunk
    page:        int             # first page this chunk touches
    page_end:    int             # last page this chunk touches
    is_overlap:  bool            # True if chunk carries overlap from previous
    links:       List[ChunkLink] = field(default_factory=list)


@dataclass
class ChunkingArgs:
    target:         str
    file:           Optional[str] = None
    project:        Optional[str] = None
    inspect:        bool = False
    rerun:          bool = False
    embedding_text: str  = ""


# ─────────────────────────────────────────────────────────────────────────────
# Link extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_links(text: str) -> List[ChunkLink]:
    """
    Extract markdown and bare-URL links from text.
    Returns a deduplicated list of ChunkLink objects.
    """
    links: List[ChunkLink] = []

    for link_text, url in re.findall(r"\[([^\]]*)\]\(([^)]*)\)", text):
        links.append(ChunkLink(text=link_text, url=url))

    existing_urls = {l.url for l in links}
    for url in re.findall(r"(?<![(\[])(https?://[^\s<>\"')\]]+)", text):
        if url not in existing_urls:
            links.append(ChunkLink(text="", url=url))
            existing_urls.add(url)

    return links


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

_BLINGFIRE_AVAILABLE = False
_BLINGFIRE_CHECKED   = False


def _init_blingfire() -> bool:
    global _BLINGFIRE_AVAILABLE, _BLINGFIRE_CHECKED
    if not _BLINGFIRE_CHECKED:
        try:
            from blingfire import text_to_sentences  # noqa: F401
            _BLINGFIRE_AVAILABLE = True
        except ImportError:
            _BLINGFIRE_AVAILABLE = False
        _BLINGFIRE_CHECKED = True
    return _BLINGFIRE_AVAILABLE


def _split_sentences(text: str) -> List[str]:
    if _init_blingfire():
        from blingfire import text_to_sentences
        result = text_to_sentences(text).strip()
        if result:
            return [s.strip() for s in result.split("\n") if s.strip()]

    abbrev = r"(?<!\b(?:Mr|Mrs|Dr|Prof|Sr|Jr|Fig|vs|etc|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec))"
    parts  = re.split(abbrev + r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Page tracking
# ─────────────────────────────────────────────────────────────────────────────

def count_page_seps(text: str) -> int:
    """
    Count page boundaries in a text fragment.

    Supports both formats:
    - New extractor:  [Page N]  lines
    - Legacy pymupdf4llm / old extractor:  ---  lines
    """
    new_style    = len(PAGE_HEADER_RE.findall(text))
    legacy_style = len(_LEGACY_PAGE_SEP_RE.findall(text))
    return new_style + legacy_style


# ─────────────────────────────────────────────────────────────────────────────
# Section splitting
# ─────────────────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def split_into_sections(text: str) -> List[Tuple[str, str, int]]:
    """
    Split text at heading boundaries.

    Page headers ([Page N]) are treated as soft boundaries: they do not start
    a new section but their presence is preserved in the body text so that
    count_page_seps() can track pages correctly inside each section.

    Returns list of (heading_text, body_text, heading_level).
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text, 0)]

    sections: List[Tuple[str, str, int]] = []
    for i, match in enumerate(matches):
        level      = len(match.group(1))
        heading    = match.group(2).strip()
        body_start = match.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()
        sections.append((heading, body, level))

    # Preamble before first heading
    first_start = matches[0].start()
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections.insert(0, ("", preamble, 0))

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Block-type detection (operates on raw extractor text, before cleaning)
# ─────────────────────────────────────────────────────────────────────────────

def _is_table_block(text: str) -> bool:
    """True if this paragraph is a table block from the extractor."""
    t = text.strip()
    return (
        t.startswith(TABLE_OPEN_TOKEN)
        or t.startswith(TABLE_EMB_TOKEN)
        # Raw pipe table (pymupdf4llm output not yet normalised)
        or (t.startswith("|") and "\n|" in t)
    )


def _is_caption_block(text: str) -> bool:
    """True if this paragraph is a caption from the extractor."""
    return text.strip().startswith(CAPTION_TOKEN)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_chunk(
    text:        str,
    source:      str,
    index:       int,
    heading:     str,
    page:        int,
    page_end:    int,
    model_name:  str = DEFAULT_MODEL,
    is_overlap:  bool = False,
    links:       Optional[List[ChunkLink]] = None,
) -> Optional[Chunk]:
    """Build one Chunk. Returns None if text has no real content."""
    text = text.strip()
    if not text:
        return None

    clean = prepare_embedding_text(text, model_name)
    if len(clean.split()) < 4:
        return None
    
    
    caption_patterns = [
        r'^\s*\[Caption:',           # Our internal format
        r'^\s*Figure\s+\d+[:.]',     # Figure 1: 
        r'^\s*Table\s+\d+[:.]',      # Table 1:
        r'^\s*Fig\.\s*\d+[:.]',      # Fig. 1:
    ]
    
    for pattern in caption_patterns:
        if re.match(pattern, clean, re.IGNORECASE):
            # It's a caption - check if it has enough meaningful content
            # Remove the caption prefix and see what's left
            remaining = re.sub(pattern, '', clean, flags=re.IGNORECASE).strip()
            if len(remaining.split()) < 3:
                # Caption with almost no description - skip it
                return None
    

    return Chunk(
        text        = text,
        clean       = clean,
        index       = index,
        source      = source,
        token_count = tokens(clean),
        heading     = heading,
        page        = page,
        page_end    = page_end,
        is_overlap  = is_overlap,
        links       = links if links is not None else [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-level splitting with overlap
# ─────────────────────────────────────────────────────────────────────────────

def _split_with_overlap(
    text:        str,
    source:      str,
    index:       int,
    heading:     str,
    page:        int,
    max_tokens:  int,
    overlap:     int,
    model_name:  str = DEFAULT_MODEL,
) -> List[Chunk]:
    """
    Split text into sentence-windowed chunks with overlap.

    Strategy: walk sentences in parallel — original text sentences for the
    chunk `text` field, cleaned sentences for token counting and `clean`.
    If blingfire gives different sentence counts for raw vs. cleaned text,
    we fall back to splitting only on the cleaned version and use its
    sentence boundaries to slice the original text.
    """
    links = extract_links(text)

    orig_sents  = _split_sentences(text)
    clean_text  = prepare_embedding_text(text, model_name)
    clean_sents = _split_sentences(clean_text)

    # If counts differ, align by using orig_sents as both (rare but safe fallback)
    if len(orig_sents) != len(clean_sents):
        clean_sents = [prepare_embedding_text(s, model_name) for s in orig_sents]

    chunks:       List[Chunk] = []
    window_orig:  List[str]   = []
    window_clean: List[str]   = []

    def _flush(is_overlap_flag: bool = False) -> None:
        nonlocal window_orig, window_clean
        if not window_orig:
            return
        raw_text   = " ".join(window_orig)
        clean_join = " ".join(window_clean)
        c = Chunk(
            text        = raw_text,
            clean       = clean_join,
            index       = index + len(chunks),
            source      = source,
            token_count = tokens(clean_join),
            heading     = heading,
            page        = page,
            page_end    = page,
            is_overlap  = is_overlap_flag,
            links       = links,
        )
        # Discard near-empty chunks
        if len(clean_join.split()) >= 4:
            chunks.append(c)

    for i, (sent_orig, sent_clean) in enumerate(zip(orig_sents, clean_sents)):
        trial_clean = window_clean + [sent_clean]
        if tokens(" ".join(trial_clean)) > max_tokens and window_clean:
            _flush(is_overlap_flag=(i > 0))
            window_orig  = window_orig[-overlap:]  if overlap else []
            window_clean = window_clean[-overlap:] if overlap else []

        window_orig.append(sent_orig)
        window_clean.append(sent_clean)

    _flush(is_overlap_flag=bool(chunks))   # last window is overlap if others exist
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Section → chunks
# ─────────────────────────────────────────────────────────────────────────────

def section_to_chunks(
    heading:     str,
    body:        str,
    source:      str,
    index:       int,
    page:        int,
    max_tokens:  int,
    min_tokens:  int,
    overlap:     int,
    model_name:  str = DEFAULT_MODEL,
) -> List[Chunk]:
    """
    Convert one (heading, body) section into one or more chunks.

    Page tracking inside a section:
    We walk paragraphs in order and count any [Page N] / --- markers we pass
    so page_end advances correctly even within a multi-page section.
    """
    if not body.strip():
        return []

    body_links  = extract_links(body)
    body_clean  = prepare_embedding_text(body, model_name)

    if tokens(body_clean) <= max_tokens:
        c = _make_chunk(
            body, source, index, heading,
            page, page + count_page_seps(body),
            model_name=model_name,
            links=body_links,
        )
        return [c] if c else []

    paragraphs:   List[str]   = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks:       List[Chunk] = []
    pending_orig: List[str]   = []
    pending_clean: List[str]  = []
    current_page: int         = page

    def _flush_pending() -> None:
        """Flush accumulated pending paragraphs as one chunk."""
        if not pending_orig:
            return
        merged_orig  = "\n\n".join(pending_orig)
        merged_clean = " ".join(pending_clean)
        c = Chunk(
            text        = merged_orig,
            clean       = merged_clean,
            index       = index + len(chunks),
            source      = source,
            token_count = tokens(merged_clean),
            heading     = heading,
            page        = page,
            page_end    = current_page,
            is_overlap  = False,
            links       = body_links,
        )
        if len(merged_clean.split()) >= 4:
            chunks.append(c)
        pending_orig.clear()
        pending_clean.clear()

    for para in paragraphs:
        current_page += count_page_seps(para)

        # ── Table block: always its own chunk ────────────────────────────────
        if _is_table_block(para):
            _flush_pending()
            c = _make_chunk(
                para, source, index + len(chunks),
                heading, current_page, current_page,
                model_name=model_name,
                links=extract_links(para),
            )
            if c:
                chunks.append(c)
            continue

        # ── Caption block: try to attach to the preceding chunk ───────────────
        if _is_caption_block(para):
            if chunks:
                last     = chunks[-1]
                combined = last.text + "\n" + para
                combined_clean = prepare_embedding_text(combined, model_name)
                if tokens(combined_clean) <= max_tokens:
                    chunks[-1] = Chunk(
                        text        = combined,
                        clean       = combined_clean,
                        index       = last.index,
                        source      = last.source,
                        token_count = tokens(combined_clean),
                        heading     = last.heading,
                        page        = last.page,
                        page_end    = current_page,
                        is_overlap  = last.is_overlap,
                        links       = last.links + extract_links(para),
                    )
                    continue
            # No preceding chunk — treat as standalone
            c = _make_chunk(
                para, source, index + len(chunks),
                heading, current_page, current_page,
                model_name=model_name,
                links=extract_links(para),
            )
            if c:
                chunks.append(c)
            continue

        # ── Normal paragraph ─────────────────────────────────────────────────
        para_clean  = prepare_embedding_text(para, model_name)
        para_tokens = tokens(para_clean)

        if para_tokens > max_tokens:
            # Paragraph itself is too large — flush pending and sub-split
            _flush_pending()
            sub = _split_with_overlap(
                para, source, index + len(chunks),
                heading, current_page, max_tokens, overlap,
                model_name=model_name,
            )
            chunks.extend(sub)
            continue

        pending_tokens = tokens(" ".join(pending_clean)) if pending_clean else 0

        if pending_tokens + para_tokens > max_tokens:
            # Adding this para would overflow — flush first
            _flush_pending()

        pending_orig.append(para)
        pending_clean.append(para_clean)

        # Eagerly flush once we've reached min_tokens to avoid overfilling
        if tokens(" ".join(pending_clean)) >= min_tokens:
            _flush_pending()

    _flush_pending()
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Small chunk merging
# ─────────────────────────────────────────────────────────────────────────────

def merge_small(
    chunks:     List[Chunk],
    min_tokens: int,
    max_tokens: int,
) -> List[Chunk]:
    """
    Merge consecutive undersized chunks.

    Normal case  (both small, same heading): merge if combined fits.
    Cross-heading (both tiny, < min_tokens/2): merge if fits; first heading kept.

    Token counts are re-estimated after merge to account for the joining
    whitespace (naive addition would undercount slightly).
    """
    if not chunks:
        return []

    tiny_threshold = min_tokens // 2
    merged:  List[Chunk]      = []
    buf:     Optional[Chunk]  = None

    for chunk in chunks:
        if buf is None:
            buf = chunk
            continue

        combined_text  = buf.text  + "\n\n" + chunk.text
        combined_clean = buf.clean + " "    + chunk.clean
        combined_toks  = tokens(combined_clean)   # re-estimate, don't just sum

        fits       = combined_toks <= max_tokens
        both_small = buf.token_count < min_tokens  and chunk.token_count < min_tokens
        both_tiny  = buf.token_count < tiny_threshold and chunk.token_count < tiny_threshold
        same_sec   = buf.heading == chunk.heading

        should_merge = fits and ((both_small and same_sec) or both_tiny)

        if should_merge:
            buf = Chunk(
                text        = combined_text,
                clean       = combined_clean,
                index       = buf.index,
                source      = buf.source,
                token_count = combined_toks,
                heading     = buf.heading,
                page        = buf.page,
                page_end    = chunk.page_end,
                is_overlap  = buf.is_overlap,
                links       = buf.links + chunk.links,
            )
        else:
            merged.append(buf)
            buf = chunk

    if buf:
        merged.append(buf)

    return merged