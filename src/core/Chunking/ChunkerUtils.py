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
from typing import Dict, List, Optional, Tuple

from src.lib.Terminal import TerminalUI
from src.core.Chunking.ChunkCleaners import prepare_embedding_text, get_model_config

DEFAULT_MODEL = "intfloat/e5-small-v2"

# signals from our parser — never split mid-token
TABLE_TOKEN   = "[Table]"
CAPTION_TOKEN = "[Caption:"
PAGE_SEP      = "---"


# ── Token counting ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_tiktoken_encoder():
    """
    Load the tiktoken encoder once and cache it for the process lifetime.
    Returns None if tiktoken is not installed.
    """
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def tokens(text: str) -> int:
    """
    Estimate token count.

    Uses tiktoken (cl100k_base) when available. Falls back to words / 0.75
    which handles Latin text well; note it under-counts for CJK / code-heavy
    content, so callers should treat the fallback as a lower-bound estimate.
    """
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


def get_dynamic_chunk_params_for_text(text: str, model_name: str = DEFAULT_MODEL) -> dict:
    """Analyse text and return optimal chunking parameters + metadata."""
    total_tokens = tokens(text)
    params       = dynamic_chunk_params(model_name, total_tokens)
    params.update({
        "doc_tokens":        total_tokens,
        "estimated_chunks":  max(1, total_tokens // params["max_tokens"]),
        "model":             model_name,
    })
    return params


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class ChunkLink:
    """
    A hyperlink scoped to a chunk. Mirrors ExtractedLink but is kept
    separate so ChunkerUtils has no hard import dependency on ParserUtils.
    """
    url:  str
    text: str = ""


@dataclass
class Chunk:
    text:        str               # original text → shown to user / sent to LLM
    clean:       str               # stripped plain text → fed to embedding model
    index:       int               # position in document (0-based)
    source:      str               # filename for citations
    token_count: int               # estimated tokens of clean text
    heading:     str               # nearest section heading above this chunk
    page:        int               # first page this chunk touches
    page_end:    int               # last page this chunk touches
    is_overlap:  bool              # True if chunk contains overlap from previous
    links:       List[ChunkLink]   = field(default_factory=list)


@dataclass
class ChunkingArgs:
    target:         str
    file:           Optional[str] = None
    project:        Optional[str] = None
    inspect:        bool = False
    rerun:          bool = False
    embedding_text: str  = ""


# ── Link extraction ────────────────────────────────────────────────────────────

def extract_links(text: str) -> List[ChunkLink]:
    """
    Extract markdown and bare-URL links from text.
    Returns a deduplicated list of ChunkLink objects.
    """
    links: List[ChunkLink] = []

    # Markdown links: [text](url)
    for link_text, url in re.findall(r"\[(.*?)\]\((.*?)\)", text):
        links.append(ChunkLink(text=link_text, url=url))

    # Bare URLs — skip any already captured above
    existing_urls = {l.url for l in links}
    for url in re.findall(r"(?<![(\[])(https?://[^\s<>\"')\]]+)", text):
        if url not in existing_urls:
            links.append(ChunkLink(text="", url=url))
            existing_urls.add(url)

    return links


# ── Sentence splitting ─────────────────────────────────────────────────────────

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


# ── Section splitting ──────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def split_into_sections(text: str) -> List[Tuple[str, str, int]]:
    """
    Split text at heading boundaries.
    Returns list of (heading_text, body_text, heading_level).
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text, 0)]

    sections = []
    for i, match in enumerate(matches):
        level      = len(match.group(1))
        heading    = match.group(2).strip()
        body_start = match.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()
        sections.append((heading, body, level))

    first_heading_start = matches[0].start()
    if first_heading_start > 0:
        preamble = text[:first_heading_start].strip()
        if preamble:
            sections.insert(0, ("", preamble, 0))

    return sections


# ── Page tracking ──────────────────────────────────────────────────────────────

def count_page_seps(text: str) -> int:
    """Count page separator markers (---) in text."""
    return len(re.findall(r"^---\s*$", text, re.MULTILINE))


# ── Table / caption detection ──────────────────────────────────────────────────

def _is_table_block(text: str) -> bool:
    return text.strip().startswith(TABLE_TOKEN)


def _is_caption_block(text: str) -> bool:
    return text.strip().startswith(CAPTION_TOKEN)


# ── Chunk builder ──────────────────────────────────────────────────────────────

def _make_chunk(
    text:       str,
    source:     str,
    index:      int,
    heading:    str,
    page:       int,
    page_end:   int,
    is_overlap: bool = False,
    links:      Optional[List[ChunkLink]] = None,   # FIX: no mutable default
) -> Optional[Chunk]:
    """Build one Chunk. Returns None if text has no real content."""
    text = text.strip()
    if not text:
        return None

    clean = prepare_embedding_text(text, DEFAULT_MODEL)
    if len(clean.split()) < 4:
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


# ── Sentence-level splitting with overlap ─────────────────────────────────────

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
    """
    links      = extract_links(text)
    sentences  = _split_sentences(prepare_embedding_text(text, DEFAULT_MODEL))
    orig_sents = _split_sentences(text)

    # align lengths — fallback if sentence counts differ
    if len(orig_sents) != len(sentences):
        orig_sents = sentences

    chunks:       List[Chunk] = []
    window_orig:  List[str]   = []
    window_clean: List[str]   = []

    def _flush(is_overlap_flag: bool = False) -> None:
        if not window_orig:
            return
        c = _make_chunk(
            " ".join(window_orig),
            source, index + len(chunks),
            heading, page, page,
            is_overlap_flag,
            links,
        )
        if c:
            chunks.append(c)

    for i, (sent_clean, sent_orig) in enumerate(zip(sentences, orig_sents)):
        trial_clean = window_clean + [sent_clean]
        if tokens(" ".join(trial_clean)) > max_tokens and window_clean:
            _flush(is_overlap_flag=(i > 0))
            window_orig  = window_orig[-overlap:]  if overlap else []
            window_clean = window_clean[-overlap:] if overlap else []

        window_orig.append(sent_orig)
        window_clean.append(sent_clean)

    _flush(is_overlap_flag=bool(window_orig))
    return chunks


# ── Section → chunks ───────────────────────────────────────────────────────────

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
    Convert one (heading, body) section into one or more chunks.

    Page tracking inside a section:
    We walk paragraphs in order and count any --- markers we pass so
    page_end advances correctly even within a multi-page section.
    """
    if not body.strip():
        return []

    body_links = extract_links(body)
    body_clean = prepare_embedding_text(body, DEFAULT_MODEL)

    if tokens(body_clean) <= max_tokens:
        c = _make_chunk(
            body, source, index, heading,
            page, page + count_page_seps(body),
            links=body_links,
        )
        return [c] if c else []

    paragraphs   = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks:       List[Chunk] = []
    pending:      List[str]   = []
    pending_orig: List[str]   = []
    current_page = page

    def _flush_pending() -> None:
        if not pending:
            return
        merged = " ".join(pending_orig)
        # FIX: pass body_links so pending-flush chunks keep their links
        c = _make_chunk(
            merged, source, index + len(chunks),
            heading, current_page, current_page,
            links=body_links,
        )
        if c:
            chunks.append(c)
        pending.clear()
        pending_orig.clear()

    for para in paragraphs:
        current_page += count_page_seps(para)

        if _is_table_block(para):
            _flush_pending()
            c = _make_chunk(
                para, source, index + len(chunks),
                heading, current_page, current_page,
                links=extract_links(para),
            )
            if c:
                chunks.append(c)
            continue

        if _is_caption_block(para):
            if chunks:
                last     = chunks[-1]
                combined = last.text + "\n" + para
                if tokens(prepare_embedding_text(combined, DEFAULT_MODEL)) <= max_tokens:
                    chunks[-1] = _make_chunk(
                        combined, source, last.index,
                        last.heading, last.page, current_page,
                        last.is_overlap,
                        links=last.links,
                    ) or last
                    continue
            c = _make_chunk(
                para, source, index + len(chunks),
                heading, current_page, current_page,
                links=extract_links(para),
            )
            if c:
                chunks.append(c)
            continue

        para_tokens    = tokens(prepare_embedding_text(para, DEFAULT_MODEL))
        pending_tokens = tokens(" ".join(pending)) if pending else 0

        if para_tokens <= max_tokens:
            if pending_tokens + para_tokens <= max_tokens:
                pending.append(prepare_embedding_text(para, DEFAULT_MODEL))
                pending_orig.append(para)
            else:
                _flush_pending()
                pending.append(prepare_embedding_text(para, DEFAULT_MODEL))
                pending_orig.append(para)

            if tokens(" ".join(pending)) >= min_tokens:
                _flush_pending()
        else:
            _flush_pending()
            sub = _split_with_overlap(
                para, source, index + len(chunks),
                heading, current_page, max_tokens, overlap,
            )
            chunks.extend(sub)

    _flush_pending()
    return chunks


# ── Small chunk merging ────────────────────────────────────────────────────────

def merge_small(
    chunks:     List[Chunk],
    min_tokens: int,
    max_tokens: int,
) -> List[Chunk]:
    """
    Merge consecutive undersized chunks.

    Normal case (both small, same heading): always merge if fits.
    Cross-heading case (both tiny, < min_tokens/2): merge if fits, but
      the heading of the *first* chunk is kept so citations still point
      to the correct section.
    """
    if not chunks:
        return []

    tiny_threshold = min_tokens // 2
    merged         = []
    buf: Optional[Chunk] = None

    for chunk in chunks:
        if buf is None:
            buf = chunk
            continue

        fits       = buf.token_count + chunk.token_count <= max_tokens
        both_small = buf.token_count < min_tokens and chunk.token_count < min_tokens
        both_tiny  = buf.token_count < tiny_threshold and chunk.token_count < tiny_threshold
        same_sec   = buf.heading == chunk.heading

        should_merge = fits and (
            (both_small and same_sec) or
            (both_tiny)
        )

        if should_merge:
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
                links       = buf.links + chunk.links,   # FIX: merge links
            )
        else:
            merged.append(buf)
            buf = chunk

    if buf:
        merged.append(buf)

    return merged