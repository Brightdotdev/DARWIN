"""
darwin.core.chunker
-------------------
Converts raw parsed text into clean overlapping chunks
ready for embedding into e5-small.

Strategy:
  1. Split by paragraph first  (respects natural idea boundaries)
  2. If paragraph is in sweet spot (100-400 tokens) → keep as-is
  3. If paragraph is too big  → split into sentences with overlap
  4. If paragraph is too small → buffer and merge with next paragraph
"""

import re
import spacy
from dataclasses import dataclass


# ── Constants ────────────────────────────────────────────────────────────────

MIN_TOKENS   = 100   # below this → too little context for LLM
MAX_TOKENS   = 400   # above this → LLM loses focus, also e5-small limit is 512
OVERLAP_SIZE = 40    # characters to carry over between sentence chunks


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text:       str        # the actual chunk content
    index:      int        # position in document (for citations later)
    source:     str        # source filename
    token_count: int       # rough token count
    is_overlap: bool       # whether this chunk contains overlap from previous


# ── Helpers ──────────────────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    """
    Rough token estimate without loading a full tokenizer.
    Words / 0.75 approximates subword tokenization well enough
    for chunking decisions — we don't need exactness here.
    """
    return int(len(text.split()) / 0.75)


def _load_spacy():
    """
    Load spacy model lazily — only when first needed.
    Falls back to a regex heuristic if spacy model not installed
    so darwin doesn't hard crash on a missing model.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


def _is_complete_sentence(text: str, nlp) -> bool:
    """
    Returns True if text ends with a complete sentence.
    Uses spacy when available, falls back to punctuation check.
    Handles edge cases like "Dr. Smith" and "Fig. 3" correctly
    because spacy understands abbreviations — raw regex does not.
    """
    text = text.strip()
    if not text:
        return False

    if nlp is not None:
        doc = nlp(text)
        sentences = list(doc.sents)
        if not sentences:
            return False
        last = sentences[-1].text.strip()
        return last[-1] in ".!?" if last else False

    # fallback: naive punctuation check
    # won't handle abbreviations perfectly but good enough
    return bool(re.search(r'[.!?]["\']?\s*$', text))


def _split_into_sentences(text: str, nlp) -> list[str]:
    """
    Split a paragraph into individual sentences.
    Uses spacy's sentence boundary detection when available.
    Falls back to a regex sentence splitter that handles
    common abbreviations (Mr., Dr., Fig., etc.)
    """
    if nlp is not None:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # fallback regex — handles common abbreviations
    abbreviations = r"(?<!\b(?:Mr|Mrs|Dr|Prof|Sr|Jr|Fig|vs|etc|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec))"
    pattern = abbreviations + r"(?<=[.!?])\s+"
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _add_overlap(previous_sentences: list[str], current: str) -> str:
    """
    Prepend the tail of the previous sentence to the current one.
    We take the END of the previous sentence (not the start)
    because the end provides the context the current sentence needs.

    Example:
        previous → "...the electron is excited and jumps to a higher energy state."
        overlap  → "...jumps to a higher energy state."
        current  → "This energy is then used to produce ATP."
        result   → "...jumps to a higher energy state. This energy is then used to produce ATP."

    Now the embedding model understands what "this energy" refers to.
    """
    if not previous_sentences:
        return current

    tail = previous_sentences[-1][-OVERLAP_SIZE:].strip()

    # avoid overlap if it would push us way over the token limit
    combined = f"{tail} {current}"
    if _token_count(combined) > MAX_TOKENS + 50:
        return current

    return combined


# ── Core chunking logic ───────────────────────────────────────────────────────

def _chunk_paragraph(
    paragraph:  str,
    chunk_index: int,
    source:     str,
    nlp,
) -> list[Chunk]:
    """
    Process a single paragraph into one or more chunks.

    Cases:
      - Sweet spot (100-400 tokens) + complete sentence → one chunk
      - Too big → split into sentences with overlap
      - Too small → return as-is (caller will buffer and merge)
    """
    paragraph = paragraph.strip()
    if not paragraph:
        return []

    tokens = _token_count(paragraph)

    # ── Sweet spot ────────────────────────────────────────────────────────────
    if MIN_TOKENS <= tokens <= MAX_TOKENS and _is_complete_sentence(paragraph, nlp):
        return [Chunk(
            text        = paragraph,
            index       = chunk_index,
            source      = source,
            token_count = tokens,
            is_overlap  = False,
        )]

    # ── Too big → sentence split with overlap ─────────────────────────────────
    if tokens > MAX_TOKENS:
        sentences = _split_into_sentences(paragraph, nlp)
        chunks    = []
        buffer    = ""
        processed = []   # track processed sentences for overlap

        for idx, sentence in enumerate(sentences):
            is_first = idx == 0

            if is_first:
                candidate = sentence
            else:
                # carry tail of previous sentence into this one
                candidate = _add_overlap(processed, sentence)

            buffer_tokens = _token_count(buffer + " " + candidate)

            if buffer_tokens <= MAX_TOKENS:
                buffer = (buffer + " " + candidate).strip()
            else:
                # flush current buffer as a chunk
                if buffer:
                    chunks.append(Chunk(
                        text        = buffer.strip(),
                        index       = chunk_index + len(chunks),
                        source      = source,
                        token_count = _token_count(buffer),
                        is_overlap  = not is_first,
                    ))
                buffer = candidate

            processed.append(sentence)

        # flush remaining buffer
        if buffer.strip():
            chunks.append(Chunk(
                text        = buffer.strip(),
                index       = chunk_index + len(chunks),
                source      = source,
                token_count = _token_count(buffer),
                is_overlap  = True,
            ))

        return chunks

    # ── Too small → return bare, caller will merge ────────────────────────────
    return [Chunk(
        text        = paragraph,
        index       = chunk_index,
        source      = source,
        token_count = tokens,
        is_overlap  = False,
    )]


def _merge_small_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """
    Second pass: merge consecutive undersized chunks into one.
    Runs after initial paragraph chunking to clean up fragments.

    Two small chunks next to each other → merge them into one
    if combined they're still under MAX_TOKENS.
    """
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
            # merge into buffer
            buffer = Chunk(
                text        = buffer.text + " " + chunk.text,
                index       = buffer.index,
                source      = buffer.source,
                token_count = combined_tokens,
                is_overlap  = buffer.is_overlap,
            )
        else:
            merged.append(buffer)
            buffer = chunk

    if buffer:
        merged.append(buffer)

    return merged


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(text: str, source: str = "unknown") -> list[Chunk]:
    """
    Main entry point. Takes raw parsed text from pymupdf4llm
    and returns a list of clean overlapping chunks ready for embedding.

    Args:
        text:   raw text string from the PDF parser
        source: filename or identifier for citation purposes

    Returns:
        list of Chunk objects, each ready to embed with e5-small
    """
    nlp        = _load_spacy()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    all_chunks: list[Chunk] = []
    chunk_index = 0

    for paragraph in paragraphs:
        new_chunks = _chunk_paragraph(paragraph, chunk_index, source, nlp)
        all_chunks.extend(new_chunks)
        chunk_index += len(new_chunks)

    # second pass: clean up any remaining fragments
    all_chunks = _merge_small_chunks(all_chunks)

    # re-index cleanly after merging
    for i, chunk in enumerate(all_chunks):
        chunk.index = i

    return all_chunks