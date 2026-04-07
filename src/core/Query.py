"""
darwin.core.query
-----------------
Full hybrid search query pipeline.

Assumes chunks are already built and indexed.
Takes a raw user query, returns top_k chunks ready for LLM context.

Pipeline:
    query → normalize → expand → hybrid search (semantic + BM25)
          → RRF merge → cross-encoder rerank → top_k
"""

import re
import asyncio
import numpy as np
from dataclasses import dataclass, field
from src.core.Chunker import Chunk
from src.core.Embedder import Embedder
from src.lib.Terminal import TerminalUI


# ── Constants ─────────────────────────────────────────────────────────────────

TOP_K          = 5     # chunks sent to LLM
RRF_K          = 60    # smoothing constant for RRF — 60 is standard
RERANK_TOP_N   = 20    # rerank only the top 20 from RRF (speed vs accuracy)
BM25_MAX_DF    = 0.85  # words in >85% of chunks → domain stopwords → excluded
BM25_MIN_DF    = 2     # word must appear in at least 2 chunks to matter


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    chunk:           Chunk
    semantic_score:  float = 0.0   # cosine similarity from e5-small
    bm25_score:      float = 0.0   # BM25 score from keyword search
    rrf_score:       float = 0.0   # merged rank score
    rerank_score:    float = 0.0   # cross-encoder score (final authority)
    rank:            int   = 0     # final rank after reranking


@dataclass
class QueryResult:
    query:        str
    variations:   list[str]          # expanded query variations
    results:      list[SearchResult] # top_k final results
    total_found:  int                # how many chunks were considered


# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Index:
    """
    BM25 index built from chunks.
    Discovers domain stopwords automatically via max_df.
    No hardcoded domain lists — the document teaches us what's boilerplate.

    Usage:
        index = BM25Index()
        index.build(chunks)          # call once after chunking
        scores = index.search(query) # call on every query
    """

    def __init__(self, max_df: float = BM25_MAX_DF, min_df: int = BM25_MIN_DF):
        self.max_df          = max_df
        self.min_df          = min_df
        self._bm25           = None
        self._tokenized_docs = []
        self._domain_stopwords: set[str] = set()
        self._chunks: list[Chunk] = []

    def _discover_stopwords(self, chunk_texts: list[str]) -> set[str]:
        """
        Use TF-IDF max_df to discover domain-specific stopwords.
        Words appearing in more than max_df fraction of chunks
        are boilerplate for THIS document — auto-discovered, not hardcoded.

        TF-IDF is only used here for stopword discovery.
        Actual search uses BM25.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            max_df    = self.max_df,
            min_df    = self.min_df,
            stop_words = "english",   # remove standard stopwords first
        )

        try:
            vectorizer.fit(chunk_texts)
            kept_vocab = set(vectorizer.vocabulary_.keys())

            # get all words across all chunks
            analyzer  = vectorizer.build_analyzer()
            all_words = set()
            for text in chunk_texts:
                all_words.update(analyzer(text))

            # words not kept = too common = domain boilerplate
            domain_stopwords = all_words - kept_vocab
            return domain_stopwords

        except ValueError:
            # too few documents to fit — skip domain stopword discovery
            return set()

    def _tokenize(self, text: str) -> list[str]:
        """
        Clean and tokenize text for BM25.
        Lowercase, remove punctuation, remove discovered stopwords.
        This is ONLY for the keyword search path.
        Embeddings never go through this.
        """
        text   = text.lower()
        text   = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
        tokens = text.split()
        tokens = [
            t for t in tokens
            if t not in self._domain_stopwords
            and len(t) > 2                         # skip single chars and 2-letter noise
        ]
        return tokens

    def build(self, chunks: list[Chunk]) -> None:
        """
        Build BM25 index from chunks.
        Discovers domain stopwords first, then tokenizes, then indexes.

        Args:
            chunks: output from chunk_document()
        """
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        chunk_texts = [c.clean for c in chunks]

        # step 1 — discover domain stopwords from this document
        self._domain_stopwords = self._discover_stopwords(chunk_texts)

        # step 2 — tokenize every chunk (with domain stopwords now known)
        self._tokenized_docs = [self._tokenize(text) for text in chunk_texts]

        # step 3 — build BM25 index
        # BM25Okapi is the standard variant — handles term frequency
        # saturation and document length normalization
        self._bm25 = BM25Okapi(self._tokenized_docs)

    def search(self, query: str) -> list[tuple[Chunk, float]]:
        """
        BM25 keyword search for a single query string.
        Returns all chunks with their BM25 scores, sorted descending.

        Args:
            query: raw query string (we tokenize it the same way as index)

        Returns:
            list of (chunk, score) sorted by score descending
        """
        if self._bm25 is None:
            raise RuntimeError("call build() before search()")

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        results = [
            (self._chunks[i], float(scores[i]))
            for i in range(len(self._chunks))
        ]

        # sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ── Semantic Index ─────────────────────────────────────────────────────────────

class SemanticIndex:
    """
    Semantic search using e5-small embeddings.
    Stores all chunk embeddings in memory as a numpy matrix.
    Fast dot-product search (cosine on normalized vectors).

    For Darwin's scale (personal documents) in-memory is fine.
    ChromaDB swap is trivial later if needed.

    Usage:
        index = SemanticIndex()
        index.build(chunks)
        results = index.search(query)
    """

    def __init__(self):
        self._embedder        = Embedder()
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None   # shape (n_chunks, 384)

    def build(self, chunks: list[Chunk]) -> None:
        """
        Embed all chunks and store as matrix.
        Called once after chunking — expensive but happens only once.
        """
        self._chunks    = chunks
        result          = self._embedder.embed(chunks)
        self._embeddings = np.array([
            ec.embedding for ec in result.embedded_chunks
        ])

    def search(self, query: str) -> list[tuple[Chunk, float]]:
        """
        Semantic search for a single query string.
        Uses "query: " prefix (not "passage: ") as e5-small requires.

        Returns:
            list of (chunk, cosine_similarity) sorted descending
        """
        if self._embeddings is None:
            raise RuntimeError("call build() before search()")

        query_vec = self._embedder.embed_query(query)

        # dot product = cosine similarity (vectors are normalized)
        # shape: (n_chunks,)
        scores = self._embeddings @ query_vec

        results = [
            (self._chunks[i], float(scores[i]))
            for i in range(len(self._chunks))
        ]

        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ── Query expansion ────────────────────────────────────────────────────────────

def _expand_query(query: str) -> list[str]:
    """
    Generate 2-3 variations of the query for better recall.
    Uses simple WordNet synonyms — no LLM call, works offline, free.

    Variations cover:
      - original query (always included)
      - synonym-expanded version (different words, same meaning)
      - keyword-only version (stripped down for BM25)

    LLM expansion is v2 — costs tokens, diminishing returns for v1.
    """
    variations = [query]   # always include original

    # variation 2 — synonym expansion via wordnet
    try:
        from nltk.corpus import wordnet
        import nltk

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)

        words    = query.lower().split()
        expanded = list(words)

        for word in words:
            synsets = wordnet.synsets(word)
            if synsets:
                # take first synonym from first synset that isn't the word itself
                for lemma in synsets[0].lemmas():
                    synonym = lemma.name().replace("_", " ")
                    if synonym.lower() != word.lower():
                        expanded.append(synonym)
                        break

        synonym_variation = " ".join(expanded)
        if synonym_variation != query:
            variations.append(synonym_variation)

    except ImportError:
        pass   # nltk not installed — skip synonym expansion silently

    # variation 3 — keyword only (remove question words for BM25)
    question_words = {"what", "how", "why", "when", "where", "who",
                      "does", "is", "are", "do", "did", "can", "could"}
    keywords = [w for w in query.lower().split() if w not in question_words]
    if keywords and " ".join(keywords) != query.lower():
        variations.append(" ".join(keywords))

    return variations[:4]   # cap at 4 variations


# ── RRF merge ─────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Chunk, float]]],
    k: int = RRF_K,
) -> list[tuple[Chunk, float]]:
    """
    Merge multiple ranked lists into one using Reciprocal Rank Fusion.

    RRF only cares about RANK POSITION not score values.
    This makes BM25 scores (arbitrary scale) and cosine scores (0-1)
    directly comparable — no normalization needed.

    Formula: rrf_score = sum over all lists of: 1 / (k + rank)
    k=60 is the standard constant — smooths the impact of top ranks.

    A chunk appearing at rank 1 in semantic AND rank 1 in BM25
    scores higher than a chunk appearing at rank 1 in only one list.
    Consensus across lists wins.
    """
    rrf_scores: dict[int, float] = {}   # chunk index → rrf score
    chunk_map:  dict[int, Chunk] = {}   # chunk index → chunk object

    for ranked_list in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked_list):
            chunk_id = chunk.index

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
                chunk_map[chunk_id]  = chunk

            # rank is 0-indexed so we use rank+1
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)

    # sort by rrf score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

    return [(chunk_map[i], rrf_scores[i]) for i in sorted_ids]


# ── Cross-encoder reranker ────────────────────────────────────────────────────

def _rerank(
    query:   str,
    results: list[tuple[Chunk, float]],
    top_n:   int = RERANK_TOP_N,
) -> list[tuple[Chunk, float]]:
    """
    Rerank top_n results using a cross-encoder.

    Cross-encoder sees query AND chunk together in one pass.
    Much more accurate than bi-encoder (e5-small) for final ranking
    because it understands how query terms relate to chunk content directly.

    We only rerank top_n (default 20) — reranking everything is too slow.
    The bi-encoder + RRF already filtered to good candidates.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - small and fast
      - runs locally, no API call
      - free
      - trained specifically for passage reranking
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        # cross-encoder not available — return as-is
        return results

    candidates = results[:top_n]
    remainder  = results[top_n:]

    try:
        model  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs  = [(query, chunk.text) for chunk, _ in candidates]
        scores = model.predict(pairs)

        reranked = sorted(
            zip([c for c, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # remainder stays at the bottom, unsorted
        return list(reranked) + [(c, 0.0) for c, _ in remainder]

    except Exception:
        # reranker failed for any reason — fall back to RRF order
        return results


# ── Main query pipeline ───────────────────────────────────────────────────────

class QueryPipeline:
    """
    Full hybrid search pipeline.
    Assumes chunks are already built and passed in.

    Usage:
        pipeline = QueryPipeline()
        pipeline.build(chunks)        # index once

        result = pipeline.query("how does chlorophyll work?")
        for r in result.results:
            print(r.chunk.text)
            print(r.rerank_score)
    """

    def __init__(self):
        self._semantic = SemanticIndex()
        self._bm25     = BM25Index()
        self._built    = False

    def build(self, chunks: list[Chunk]) -> None:
        """
        Index all chunks for both semantic and keyword search.
        Call this once after chunking — not on every query.
        """
        self._semantic.build(chunks)
        self._bm25.build(chunks)
        self._built = True

        # printing
        TerminalUI.bold(f"  indexed {len(chunks)} chunks")
        TerminalUI.dim(f" domain stopwords discovered: {len(self._bm25._domain_stopwords)})")

    def query(self, raw_query: str, top_k: int = TOP_K) -> QueryResult:
        """
        Run the full pipeline for a user query.

        Steps:
          1. expand query into variations
          2. semantic search each variation
          3. BM25 search each variation
          4. RRF merge all result lists
          5. cross-encoder rerank top 20
          6. return top_k

        Args:
            raw_query: the user's raw input string
            top_k:     how many chunks to return for LLM context

        Returns:
            QueryResult with ranked chunks and metadata
        """
        if not self._built:
            raise RuntimeError("call build() before query()")

        # ── step 1 — expand query ─────────────────────────────────────────────
        variations = _expand_query(raw_query)

        # ── step 2 + 3 — search each variation with both methods ──────────────
        all_ranked_lists = []

        for variation in variations:
            semantic_results = self._semantic.search(variation)
            bm25_results     = self._bm25.search(variation)

            all_ranked_lists.append(semantic_results)
            all_ranked_lists.append(bm25_results)

        # ── step 4 — RRF merge ────────────────────────────────────────────────
        merged = _reciprocal_rank_fusion(all_ranked_lists)

        # ── step 5 — cross-encoder rerank ─────────────────────────────────────
        # use the ORIGINAL query for reranking — not variations
        # cross-encoder is precise enough that original is better
        reranked = _rerank(raw_query, merged, top_n=RERANK_TOP_N)

        # ── step 6 — build final results ──────────────────────────────────────
        final = []
        for rank, (chunk, score) in enumerate(reranked[:top_k]):
            # find original scores for this chunk
            sem_score = next(
                (s for c, s in self._semantic.search(raw_query) if c.index == chunk.index),
                0.0,
            )
            bm25_score = next(
                (s for c, s in self._bm25.search(raw_query) if c.index == chunk.index),
                0.0,
            )
            rrf_score = next(
                (s for c, s in merged if c.index == chunk.index),
                0.0,
            )

            final.append(SearchResult(
                chunk          = chunk,
                semantic_score = sem_score,
                bm25_score     = bm25_score,
                rrf_score      = rrf_score,
                rerank_score   = score,
                rank           = rank,
            ))

        return QueryResult(
            query        = raw_query,
            variations   = variations,
            results      = final,
            total_found  = len(merged),
        )

    def format_context(self, result: QueryResult) -> str:
        """
        Format query results as XML context block for LLM.
        Source citations are embedded in the XML so the LLM
        can reference them in its answer.

        Output format:
            <context>
              <chunk index="3" source="biology.pdf" score="0.92">
                ...chunk text...
              </chunk>
            </context>
        """
        chunks_xml = "\n".join([
            f'  <chunk index="{r.chunk.index}" '
            f'source="{r.chunk.source}" '
            f'score="{r.rerank_score:.3f}">\n'
            f'    {r.chunk.text}\n'
            f'  </chunk>'
            for r in result.results
        ])

        return f"<context>\n{chunks_xml}\n</context>"