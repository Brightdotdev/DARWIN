import re
import functools
from typing import List, Optional, Tuple
from src.features.Ingestion.Chunking.ChunkConfig import get_chunk_config
from src.features.Ingestion.Chunking.ChunkCleaners import prepare_embedding_text
from src.features.Ingestion.Chunking.lib import (OverlapDirection, ChunkLink, Chunk,count_page_seps, TABLE_OPEN_TOKEN, TABLE_EMB_TOKEN, CAPTION_TOKEN)

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

def _is_table_token_block(text: str) -> bool:
    """
    True if this paragraph is a table block from the extractor.
    Checks for token-prefix markers — NOT geometry (that lives in Extractors.py).
    Renamed from _is_table_block to avoid collision with Extractors.py's
    geometry-based function of the same name.
    """
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

    Uses tiktoken (cl100k_base) when available.  Falls back to
    character-count / 4, which handles Latin, CJK, and code-heavy content
    more reliably than the old word-count / 0.75 approximation.
    The fallback is still a lower-bound estimate for very dense code.
    """
    if not text:
        return 0
    enc = _get_tiktoken_encoder()
    if enc is not None:
        return len(enc.encode(text))
    # Character-based fallback — more reliable than word-split for CJK / code
    return max(1, len(text) // 4)




def dynamic_chunk_params(doc_tokens: int) -> dict:
    """
    Scale chunk size to document length using the active model's config.
    Auto-detects model from loader — no model_name parameter.
    """
    config = get_chunk_config()
    max_capacity = config.max_tokens

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
        "min_tokens": config.min_tokens,
        "overlap":    config.overlap + overlap_add,
        "base_max":   max_capacity,
        "strategy":   f"{optimal_max}/{max_capacity} tokens per chunk",
    }




# ── Update _make_chunk — no model_name parameter ──

def _make_chunk(
    text:             str,
    source:           str,
    index:            int,
    heading:          str,
    page:             int,
    page_end:         int,
    overlap_direction: "OverlapDirection" = None,
    links:            Optional[List["ChunkLink"]] = None,
) -> Optional["Chunk"]:
    """Build one Chunk. Returns None if text has no real content."""
    if overlap_direction is None:
        overlap_direction = OverlapDirection.NONE

    text = text.strip()
    if not text:
        return None

    # Auto-detect cleaning mode from active model
    clean = prepare_embedding_text(text)
    if len(clean.split()) < 4:
        return None

    caption_patterns = [
        r'^\s*\[Caption:',
        r'^\s*Figure\s+\d+[:.]',
        r'^\s*Table\s+\d+[:.]',
        r'^\s*Fig\.\s*\d+[:.]',
    ]
    for pattern in caption_patterns:
        if re.match(pattern, clean, re.IGNORECASE):
            remaining = re.sub(pattern, '', clean, flags=re.IGNORECASE).strip()
            if len(remaining.split()) < 3:
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
        overlap_direction = overlap_direction,
        links       = links if links is not None else [],
    )




def _split_with_overlap(
    text:        str,
    source:      str,
    index:       int,
    heading:     str,
    page:        int,
    max_tokens:  int,
    overlap:     int,
) -> List["Chunk"]:
    """Split text into sentence-windowed chunks with overlap."""
    links = extract_links(text)

    orig_sents  = _split_sentences(text)
    clean_text  = prepare_embedding_text(text)
    clean_sents = _split_sentences(clean_text)

    if len(orig_sents) != len(clean_sents):
        clean_sents = [prepare_embedding_text(s) for s in orig_sents]

    chunks:       List["Chunk"] = []
    window_orig:  List[str]     = []
    window_clean: List[str]     = []
    window_has_left_overlap: bool = False

    def _flush(will_continue: bool) -> None:
        nonlocal window_orig, window_clean, window_has_left_overlap
        if not window_orig:
            return
        raw_text   = " ".join(window_orig)
        clean_join = " ".join(window_clean)
        has_left  = window_has_left_overlap
        has_right = will_continue and overlap > 0 and len(window_orig) > overlap
        if has_left and has_right:
            direction = OverlapDirection.BOTH
        elif has_left:
            direction = OverlapDirection.LEFT
        elif has_right:
            direction = OverlapDirection.RIGHT
        else:
            direction = OverlapDirection.NONE
        c = Chunk(
            text=raw_text, clean=clean_join, index=index + len(chunks),
            source=source, token_count=tokens(clean_join), heading=heading,
            page=page, page_end=page, overlap_direction=direction, links=links,
        )
        if len(clean_join.split()) >= 4:
            chunks.append(c)

    for i, (sent_orig, sent_clean) in enumerate(zip(orig_sents, clean_sents)):
        trial_clean = window_clean + [sent_clean]
        if tokens(" ".join(trial_clean)) > max_tokens and window_clean:
            _flush(will_continue=True)
            window_orig  = window_orig[-overlap:]  if overlap else []
            window_clean = window_clean[-overlap:] if overlap else []
            window_has_left_overlap = bool(overlap and window_orig)
        window_orig.append(sent_orig)
        window_clean.append(sent_clean)
    _flush(will_continue=False)
    return chunks


# ── Update section_to_chunks — no model_name ──

def section_to_chunks(
    heading:     str,
    body:        str,
    source:      str,
    index:       int,
    page:        int,
    max_tokens:  int,
    min_tokens:  int,
    overlap:     int,
) -> List["Chunk"]:
    """Convert one (heading, body) section into one or more chunks."""
    if not body.strip():
        return []

    body_links = extract_links(body)
    body_clean = prepare_embedding_text(body)

    if tokens(body_clean) <= max_tokens:
        c = _make_chunk(
            body, source, index, heading,
            page, page + count_page_seps(body),
            links=body_links,
        )
        return [c] if c else []

    paragraphs   = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks       = []
    pending_orig = []
    pending_clean = []
    current_page = page

    def _flush_pending():
        if not pending_orig:
            return
        merged_orig  = "\n\n".join(pending_orig)
        merged_clean = " ".join(pending_clean)
        c = Chunk(
            text=merged_orig, clean=merged_clean, index=index + len(chunks),
            source=source, token_count=tokens(merged_clean), heading=heading,
            page=page, page_end=current_page,
            overlap_direction=OverlapDirection.NONE, links=body_links,
        )
        if len(merged_clean.split()) >= 4:
            chunks.append(c)
        pending_orig.clear()
        pending_clean.clear()

    for para in paragraphs:
        current_page += count_page_seps(para)

        if _is_table_token_block(para):
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
                last = chunks[-1]
                combined = last.text + "\n" + para
                combined_clean = prepare_embedding_text(combined)
                if tokens(combined_clean) <= max_tokens:
                    chunks[-1] = Chunk(
                        text=combined, clean=combined_clean, index=last.index,
                        source=last.source, token_count=tokens(combined_clean),
                        heading=last.heading, page=last.page, page_end=current_page,
                        overlap_direction=last.overlap_direction,
                        links=last.links + extract_links(para),
                    )
                    continue
            c = _make_chunk(
                para, source, index + len(chunks),
                heading, current_page, current_page,
                links=extract_links(para),
            )
            if c:
                chunks.append(c)
            continue

        para_clean  = prepare_embedding_text(para)
        para_tokens = tokens(para_clean)

        if para_tokens > max_tokens:
            _flush_pending()
            sub = _split_with_overlap(
                para, source, index + len(chunks),
                heading, current_page, max_tokens, overlap,
            )
            chunks.extend(sub)
            continue

        pending_tokens = tokens(" ".join(pending_clean)) if pending_clean else 0
        if pending_tokens + para_tokens > max_tokens:
            _flush_pending()

        pending_orig.append(para)
        pending_clean.append(para_clean)

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
    Cross-heading (both tiny, < min_tokens/2): merge ONLY if fits AND the
    heading difference is minor (one of them has no heading) to avoid silently
    destroying citation metadata across unrelated sections.

    When chunks are merged:
    - The combined overlap_direction is the union of both directions, so no
      overlap information is silently lost.
    - Token counts are re-estimated after merge to account for the joining
      whitespace (naive addition would undercount slightly).
    """
    if not chunks:
        return []

    tiny_threshold = min_tokens // 2
    merged: List[Chunk]     = []
    buf:    Optional[Chunk] = None

    def _merge_directions(a: OverlapDirection, b: OverlapDirection) -> OverlapDirection:
        """
        Union the overlap directions of two chunks being merged.
        We keep LEFT from `a` (it opened with borrowed sentences) and
        RIGHT from `b` (it donated its tail forward).
        """
        has_left  = a in (OverlapDirection.LEFT,  OverlapDirection.BOTH)
        has_right = b in (OverlapDirection.RIGHT, OverlapDirection.BOTH)
        if has_left and has_right:
            return OverlapDirection.BOTH
        if has_left:
            return OverlapDirection.LEFT
        if has_right:
            return OverlapDirection.RIGHT
        return OverlapDirection.NONE

    for chunk in chunks:
        if buf is None:
            buf = chunk
            continue

        combined_text  = buf.text  + "\n\n" + chunk.text
        combined_clean = buf.clean + " "    + chunk.clean
        combined_toks  = tokens(combined_clean)

        fits       = combined_toks <= max_tokens
        both_small = buf.token_count < min_tokens  and chunk.token_count < min_tokens
        both_tiny  = buf.token_count < tiny_threshold and chunk.token_count < tiny_threshold
        same_sec   = buf.heading == chunk.heading

        # Cross-heading tiny merge: only allow when at least one side has no
        # heading (e.g. a preamble paragraph), not between two named sections.
        cross_heading_ok = both_tiny and (not buf.heading or not chunk.heading)

        should_merge = fits and ((both_small and same_sec) or cross_heading_ok)

        if should_merge:
            buf = Chunk(
                text             = combined_text,
                clean            = combined_clean,
                index            = buf.index,
                source           = buf.source,
                token_count      = combined_toks,
                heading          = buf.heading,
                page             = buf.page,
                page_end         = chunk.page_end,
                overlap_direction = _merge_directions(buf.overlap_direction,
                                                      chunk.overlap_direction),
                links            = buf.links + chunk.links,
            )
        else:
            merged.append(buf)
            buf = chunk

    if buf:
        merged.append(buf)

    return merged