"""
src/core/Extractor.py
---------------------
Extracts concepts from chunks for the knowledge graph.

Two modes:
    with LLM  → rich proposition-style concepts
                 "chlorophyll absorbs red and blue light to drive reactions"
    without   → raw chunk text truncated to 120 chars
                 graph still works, just less clean labels

Async extraction fires all LLM calls simultaneously
with a semaphore to respect rate limits.

Usage:
    extractor = Extractor(llm)
    concepts  = await extractor.extract_all(chunks)

    # each concept has:
    concept.label        # the idea as a sentence
    concept.weight       # importance 1-10
    concept.chunk_index  # which chunk it came from
    concept.page         # page number for citations
"""

import re
import asyncio
from dataclasses import dataclass, field
from src.core.Chunker import Chunk
from src.core.LLM import LLM


# ── Constants ──────────────────────────────────────────────────────────────────

MAX_CONCURRENT_LLM = 5     # max simultaneous LLM calls (rate limit protection)
MAX_CONCEPTS_PER_CHUNK = 4 # LLM extracts at most this many per chunk
MIN_LABEL_WORDS = 5        # reject concepts shorter than this


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Concept:
    label:       str          # the idea as a full sentence
    weight:      float        # importance 1-10 → becomes node size
    chunk_index: int          # source chunk index → for citations
    page:        int = 0      # page number → for citations
    source:      str = ""     # filename → for citations
    from_llm:    bool = True  # False = raw chunk fallback


@dataclass
class ExtractionResult:
    concepts:      list[Concept]
    total_chunks:  int
    llm_available: bool
    failed_chunks: int = 0   # chunks where LLM failed → fell back to raw


# ── XML parsing ────────────────────────────────────────────────────────────────

def _parse_concepts_xml(xml: str, chunk: Chunk) -> list[Concept]:
    """
    Parse LLM XML response into Concept objects.

    Expected format:
        <concepts>
          <concept>
            <label>chlorophyll absorbs red and blue light</label>
            <weight>8</weight>
          </concept>
        </concepts>

    Robust — handles malformed XML gracefully.
    Falls back to raw chunk if parsing fails completely.
    """
    concepts = []

    # find all concept blocks
    blocks = re.findall(r"<concept>(.*?)</concept>", xml, re.DOTALL)

    for block in blocks:
        # extract label
        label_match = re.search(r"<label>(.*?)</label>", block, re.DOTALL)
        if not label_match:
            continue
        label = label_match.group(1).strip()

        # reject if too short — probably a topic word not a proposition
        if len(label.split()) < MIN_LABEL_WORDS:
            continue

        # extract weight
        weight_match = re.search(r"<weight>(\d+(?:\.\d+)?)</weight>", block)
        weight = float(weight_match.group(1)) if weight_match else 5.0
        weight = max(1.0, min(10.0, weight))  # clamp to 1-10

        concepts.append(Concept(
            label       = label,
            weight      = weight,
            chunk_index = chunk.index,
            page        = getattr(chunk, "page", 0),
            source      = chunk.source,
            from_llm    = True,
        ))

    return concepts[:MAX_CONCEPTS_PER_CHUNK]


def _raw_concept(chunk: Chunk) -> Concept:
    """
    Fallback when LLM is unavailable or fails.
    Uses truncated chunk text as the concept label.
    Still useful — embeddings make the graph meaningful
    even without clean labels.
    """
    # take first 120 chars, stop at last complete word
    text = chunk.text.strip()
    if len(text) > 120:
        text = text[:120].rsplit(" ", 1)[0] + "..."

    # rough importance from token count — longer chunks = more content
    weight = min(10.0, max(3.0, chunk.token_count / 40))

    return Concept(
        label       = text,
        weight      = weight,
        chunk_index = chunk.index,
        page        = getattr(chunk, "page", 0),
        source      = chunk.source,
        from_llm    = False,
    )


# ── Extraction prompt ──────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a knowledge extraction system.
Extract key ideas from document chunks as complete propositions.
Always respond with valid XML exactly as specified. Nothing else."""

def _extraction_prompt(chunk: Chunk) -> str:
    return f"""Extract 2-4 key ideas from this text chunk.

Rules:
- Each idea must be a COMPLETE SENTENCE expressing a claim, mechanism, or cause-effect
- NOT a topic heading or single word — a full proposition
- NOT meta-commentary ("this section discusses...") — the actual idea
- NOT definitions ("X is a...") unless the definition IS the key insight
- Rate importance 1-10 based on how central this idea is to the document

Respond ONLY with this XML, no other text:
<concepts>
  <concept>
    <label>the complete idea as one sentence</label>
    <weight>8</weight>
  </concept>
</concepts>

Chunk (index={chunk.index}, source={chunk.source}):
{chunk.text}"""


# ── Async extraction ───────────────────────────────────────────────────────────

async def _extract_chunk_async(
    chunk: Chunk,
    llm:   LLM,
    sem:   asyncio.Semaphore,
) -> tuple[list[Concept], bool]:
    """
    Extract concepts from one chunk asynchronously.
    Uses semaphore to limit concurrent LLM calls.

    Returns: (concepts, success)
        success=False means LLM failed → caller uses raw fallback
    """
    async with sem:
        # run the synchronous LLM call in a thread pool
        # so it doesn't block the event loop
        loop     = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: llm.complete(
                prompt     = _extraction_prompt(chunk),
                system     = EXTRACTION_SYSTEM,
                max_tokens = 512,
            )
        )

        if response is None:
            return [], False

        concepts = _parse_concepts_xml(response.text, chunk)

        if not concepts:
            return [], False

        return concepts, True


# ── Main extractor ─────────────────────────────────────────────────────────────

class Extractor:
    """
    Extracts concepts from chunks for the knowledge graph.

    With LLM:    fires async calls for all chunks simultaneously
                 max 5 concurrent (semaphore) to respect rate limits
    Without LLM: immediately returns raw chunk text as concepts
                 graph still renders, labels are just less clean

    Args:
        llm: LLM instance — if not available, falls back to raw mode
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    async def extract_all(self, chunks: list[Chunk]) -> ExtractionResult:
        """
        Extract concepts from all chunks.

        With LLM available:
            - fires all chunks concurrently (max 5 at once)
            - failed chunks fall back to raw text
            - returns mix of LLM concepts + raw fallbacks

        Without LLM:
            - immediately returns raw chunk concepts
            - no async overhead
            - notifies user graph might be rough

        Args:
            chunks: output from chunk_document()

        Returns:
            ExtractionResult with all concepts ready for graph builder
        """
        if not self.llm.is_available:
            # no LLM — raw mode, instant
            concepts = [_raw_concept(chunk) for chunk in chunks]
            return ExtractionResult(
                concepts      = concepts,
                total_chunks  = len(chunks),
                llm_available = False,
                failed_chunks = 0,
            )

        # LLM available — async extraction
        sem     = asyncio.Semaphore(MAX_CONCURRENT_LLM)
        tasks   = [
            _extract_chunk_async(chunk, self.llm, sem)
            for chunk in chunks
        ]
        results = await asyncio.gather(*tasks)

        all_concepts  = []
        failed_chunks = 0

        for (concepts, success), chunk in zip(results, chunks):
            if success and concepts:
                all_concepts.extend(concepts)
            else:
                # LLM failed for this chunk → raw fallback
                all_concepts.append(_raw_concept(chunk))
                failed_chunks += 1

        return ExtractionResult(
            concepts      = all_concepts,
            total_chunks  = len(chunks),
            llm_available = True,
            failed_chunks = failed_chunks,
        )

    def extract_all_sync(self, chunks: list[Chunk]) -> ExtractionResult:
        """
        Synchronous wrapper around extract_all.
        Use this when you're not already in an async context
        (e.g. from the CLI).
        """
        return asyncio.run(self.extract_all(chunks))