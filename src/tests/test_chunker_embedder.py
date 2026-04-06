"""
tests for darwin chunker and embedder
run with: pytest tests/test_chunker_embedder.py -v
"""

import pytest
import numpy as np
from src.core.Chunker import chunk_document, Chunk, MIN_TOKENS, MAX_TOKENS
from src.core.Embedder import Embedder


# ── Sample texts ──────────────────────────────────────────────────────────────

# short paragraph — should get buffered and merged
SHORT_PARA = "Chlorophyll is a green pigment."

# sweet spot paragraph — should chunk as-is
SWEET_PARA = """
Chlorophyll absorbs red and blue light but reflects green light, which is why plants
appear green to our eyes. This absorption is not accidental — red and blue wavelengths
carry the right amount of energy to excite electrons in the chlorophyll molecule to a
higher energy state. These excited electrons are the starting point of the entire
photosynthesis process, eventually driving the production of ATP and NADPH.
""".strip()

# long paragraph — should split into sentences with overlap
LONG_PARA = """
During the light-dependent reactions of photosynthesis, chlorophyll molecules in the
thylakoid membrane absorb photons of light. This absorption excites electrons to a
higher energy level, and these high-energy electrons are passed along the electron
transport chain. As electrons move through this chain, their energy is used to pump
protons across the thylakoid membrane, creating a concentration gradient. This gradient
drives ATP synthase to produce ATP from ADP and inorganic phosphate. The electrons
ultimately reduce NADP+ to NADPH, which along with ATP will power the Calvin cycle.
Water molecules are split in a process called photolysis to replace the electrons lost
by chlorophyll, releasing oxygen as a byproduct into the atmosphere.
""".strip()

MULTI_PARA_DOC = f"{SHORT_PARA}\n\n{SWEET_PARA}\n\n{LONG_PARA}"


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestChunker:

    def test_returns_list_of_chunks(self):
        chunks = chunk_document(SWEET_PARA, source="test.pdf")
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_sweet_spot_paragraph_stays_whole(self):
        chunks = chunk_document(SWEET_PARA, source="test.pdf")
        # one clean paragraph in sweet spot should produce one chunk
        assert len(chunks) == 1
        assert chunks[0].text == SWEET_PARA

    def test_long_paragraph_splits(self):
        chunks = chunk_document(LONG_PARA, source="test.pdf")
        # long paragraph must produce more than one chunk
        assert len(chunks) > 1
        # each chunk must respect the max token limit (with some buffer for overlap)
        for chunk in chunks:
            assert chunk.token_count <= MAX_TOKENS + 60, (
                f"chunk too large: {chunk.token_count} tokens\n{chunk.text[:80]}"
            )

    def test_short_paragraph_merges(self):
        # two short paragraphs should merge into one chunk
        doc = f"{SHORT_PARA}\n\n{SHORT_PARA}"
        chunks = chunk_document(doc, source="test.pdf")
        assert len(chunks) == 1

    def test_overlap_present_in_sentence_chunks(self):
        chunks = chunk_document(LONG_PARA, source="test.pdf")
        overlap_chunks = [c for c in chunks if c.is_overlap]
        # at least some chunks from a long paragraph should have overlap
        assert len(overlap_chunks) > 0

    def test_source_preserved(self):
        chunks = chunk_document(SWEET_PARA, source="biology.pdf")
        assert all(c.source == "biology.pdf" for c in chunks)

    def test_index_sequential(self):
        chunks = chunk_document(MULTI_PARA_DOC, source="test.pdf")
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_string_returns_empty(self):
        chunks = chunk_document("", source="test.pdf")
        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        chunks = chunk_document("   \n\n   \n\n   ", source="test.pdf")
        assert chunks == []

    def test_multi_paragraph_doc(self):
        chunks = chunk_document(MULTI_PARA_DOC, source="test.pdf")
        assert len(chunks) >= 2  # at least short+sweet merged + long split

    def test_no_empty_chunks(self):
        chunks = chunk_document(MULTI_PARA_DOC, source="test.pdf")
        assert all(c.text.strip() for c in chunks)

    def test_all_text_preserved(self):
        """make sure no words are silently dropped during chunking"""
        chunks = chunk_document(SWEET_PARA, source="test.pdf")
        reconstructed = " ".join(c.text for c in chunks)
        # every word in the original should appear somewhere in the chunks
        for word in SWEET_PARA.split():
            assert word in reconstructed, f"word '{word}' lost during chunking"


# ── Embedder tests ────────────────────────────────────────────────────────────

class TestEmbedder:

    @pytest.fixture(scope="class")
    def embedder(self):
        """shared embedder instance — model loads once for all tests"""
        return Embedder()

    @pytest.fixture(scope="class")
    def chunks(self):
        return chunk_document(MULTI_PARA_DOC, source="test.pdf")

    @pytest.fixture(scope="class")
    def result(self, embedder, chunks):
        return embedder.embed(chunks)

    def test_returns_embedding_result(self, result):
        from src.core.Embedder import EmbeddingResult
        assert isinstance(result, EmbeddingResult)

    def test_correct_number_of_embeddings(self, result, chunks):
        assert result.total_chunks == len(chunks)
        assert len(result.embedded_chunks) == len(chunks)

    def test_embedding_shape(self, result):
        # e5-small-v2 produces 384-dimensional vectors
        for ec in result.embedded_chunks:
            assert ec.embedding.shape == (384,), (
                f"expected (384,) got {ec.embedding.shape}"
            )

    def test_embeddings_normalized(self, result):
        # normalized vectors have length 1.0
        # this is what enables fast dot-product cosine similarity
        for ec in result.embedded_chunks:
            norm = float(np.linalg.norm(ec.embedding))
            assert abs(norm - 1.0) < 1e-5, f"embedding not normalized: norm={norm}"

    def test_different_chunks_have_different_embeddings(self, result):
        if len(result.embedded_chunks) < 2:
            pytest.skip("need at least 2 chunks")
        e1 = result.embedded_chunks[0].embedding
        e2 = result.embedded_chunks[1].embedding
        # dot product of two different normalized vectors should not be 1.0
        similarity = float(np.dot(e1, e2))
        assert similarity < 0.999, "two different chunks have identical embeddings"

    def test_similar_text_has_high_similarity(self, embedder):
        # two chunks about the same topic should have high cosine similarity
        chunks_a = chunk_document("Chlorophyll absorbs light to drive photosynthesis.", "a.pdf")
        chunks_b = chunk_document("Chlorophyll captures light energy for photosynthesis.", "b.pdf")

        result_a = embedder.embed(chunks_a)
        result_b = embedder.embed(chunks_b)

        if not result_a.embedded_chunks or not result_b.embedded_chunks:
            pytest.skip("chunks too short after chunking")

        sim = float(np.dot(
            result_a.embedded_chunks[0].embedding,
            result_b.embedded_chunks[0].embedding,
        ))
        assert sim > 0.80, f"similar sentences should be close: similarity={sim:.3f}"

    def test_unrelated_text_has_low_similarity(self, embedder):
        chunks_a = chunk_document(SWEET_PARA, "a.pdf")
        chunks_b = chunk_document(
            "The French Revolution began in 1789 when economic hardship and "
            "political inequality pushed citizens to revolt against the monarchy.",
            "b.pdf"
        )

        result_a = embedder.embed(chunks_a)
        result_b = embedder.embed(chunks_b)

        if not result_a.embedded_chunks or not result_b.embedded_chunks:
            pytest.skip("chunks too short")

        sim = float(np.dot(
            result_a.embedded_chunks[0].embedding,
            result_b.embedded_chunks[0].embedding,
        ))
        assert sim < 0.75, f"unrelated sentences should be far apart: similarity={sim:.3f}"

    def test_empty_chunks_returns_empty_result(self, embedder):
        result = embedder.embed([])
        assert result.total_chunks == 0
        assert result.embedded_chunks == []
        assert result.dimension == 0

    def test_embed_query_returns_correct_shape(self, embedder):
        vec = embedder.embed_query("what is photosynthesis?")
        assert vec.shape == (384,)

    def test_embed_query_normalized(self, embedder):
        vec = embedder.embed_query("what is chlorophyll?")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_chunk_text_preserved_in_result(self, result, chunks):
        for ec, chunk in zip(result.embedded_chunks, chunks):
            assert ec.chunk.text == chunk.text