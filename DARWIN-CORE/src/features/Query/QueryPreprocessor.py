"""
Query Preprocessor / Router
---------------------------
Extracts links and blobs, classifies intent, scores desperation,
optimizes text for RAG, and provides entity recognition hooks.
Designed to be LLM‑agnostic – rule‑based first, with optional LLM override.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse


from src.features.Query.lib import (
    ExtractedLink,
    ExtractedBlob,
    PreprocessedQuery,
    PresentationTypeWithScore,
    PresentationType,
)

# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Preprocesses raw user input: link/blob extraction, intent classification,
    desperation scoring, RAG query optimisation, and named entity recognition.
    """

    def __init__(
        self,
        blob_threshold: int = 300,
        spacy_model: Optional[Any] = None,
        classification_model: Optional[Any] = None,
    ):
        """
        Args:
            blob_threshold: minimum character length for a text block to be considered a blob.
            spacy_model: optional spaCy language model for NER. If None, a mock is used.
            classification_model: optional callable for LLM‑based classification.
        """
        self.blob_threshold = blob_threshold
        self.spacy_model = spacy_model
        self.classification_model = classification_model

        self.link_pattern = re.compile(r'https?://\S+')
        self.stopwords = {
            "the", "is", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
            "but", "with", "that", "this", "it", "from", "by", "as", "be", "has", "have",
            "had", "not", "are", "was", "were", "been", "will", "would", "could", "should",
            "may", "might", "can", "shall", "do", "does", "did",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, raw_text: str) -> PreprocessedQuery:
        """Main entry point: run all preprocessing steps and return a PreprocessedQuery."""
        # 1. Extract links
        text_no_links, links = self._extract_links(raw_text)

        # 2. Extract blobs
        cleaned_text, blobs = self._extract_blobs(text_no_links)

        # 3. Classify intents
        presentation_types, requires_llm = self._classify(cleaned_text)

        # 4. Desperation score
        desperation_score = self._detect_desperation(cleaned_text)

        # 5. Build RAG query
        rag_query = self._build_rag_query(cleaned_text, presentation_types)

        # 6. Build placeholder map
        placeholder_map = self._build_placeholders(links, blobs)

        # 7. Named entities (mock or real spaCy)
        named_entities = self._ner(cleaned_text)

        # 8. Sub‑questions (optional)
        sub_questions = self._detect_sub_questions(cleaned_text)

        return PreprocessedQuery(
            original_text=raw_text,
            cleaned_text=cleaned_text,
            rag_query=rag_query,
            presentation_types=presentation_types,
            requires_llm_classification=requires_llm,
            desperation_score=desperation_score,
            extracted_links=links,
            extracted_blobs=blobs,
            placeholders=placeholder_map,
            sub_questions=sub_questions,
            named_entities=named_entities,
        )

    # ------------------------------------------------------------------
    # Link extraction
    # ------------------------------------------------------------------

    def _extract_links(self, text: str) -> Tuple[str, List[ExtractedLink]]:
        """Find all URLs, replace with [link::domain], return cleaned text + links."""
        links: List[ExtractedLink] = []

        def repl(match: re.Match) -> str:
            url = match.group(0)
            domain = urlparse(url).netloc or "unknown"
            placeholder = f"[link::{domain}]"
            links.append(ExtractedLink(raw_url=url, domain=domain, placeholder=placeholder))
            return placeholder

        cleaned = self.link_pattern.sub(repl, text)
        return cleaned, links

    # ------------------------------------------------------------------
    # Blob extraction
    # ------------------------------------------------------------------

    def _extract_blobs(self, text: str) -> Tuple[str, List[ExtractedBlob]]:
        """Detect large monolithic text blocks and replace them with [blob::<id>]."""
        blobs: List[ExtractedBlob] = []
        parts = re.split(r'\n\n+', text)
        cleaned_parts = []

        for part in parts:
            stripped = part.strip()
            if len(stripped) > self.blob_threshold or self._is_code_or_log(stripped):
                blob_id = uuid.uuid4().hex[:8]
                placeholder = f"[blob::{blob_id}]"
                blobs.append(ExtractedBlob(
                    blob_id=blob_id,
                    content_preview=stripped[:200],
                    placeholder=placeholder,
                    token_count=len(stripped.split()),
                    is_truncated=len(stripped) > 5000,
                ))
                cleaned_parts.append(placeholder)
            else:
                cleaned_parts.append(stripped)

        return '\n\n'.join(cleaned_parts), blobs

    def _is_code_or_log(self, text: str) -> bool:
        """Heuristic for code blocks, logs, and stack traces."""
        indicators = [
            '{', '}', 'def ', 'import ', '=>', '||',
            'Traceback', 'ERROR:', 'WARN:', 'DEBUG:', 'Exception',
        ]
        return any(ind in text for ind in indicators)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, text: str) -> Tuple[List[PresentationTypeWithScore], bool]:
        """Hybrid classification: rule‑based first, LLM fallback if confidence is low."""
        rule_types, avg_confidence = self._rule_classify(text)
        needs_llm = avg_confidence < 0.5 or self._detect_multiple_intents(text)
        # If LLM model is available and we need it, caller will handle the actual LLM call.
        # We simply flag `requires_llm_classification`.
        return rule_types, needs_llm

    def _rule_classify(self, text: str) -> Tuple[List[PresentationTypeWithScore], float]:
        """Fast keyword/pattern matching. Returns types and average confidence."""
        t = text.lower()
        matches: List[PresentationTypeWithScore] = []

        # High confidence patterns
        if any(p in t for p in ["should i", "is it worth", "pros and cons", "pros & cons"]):
            matches.append(PresentationTypeWithScore(PresentationType.PROS_CONS, "elaborator", 0.9))
        if any(p in t for p in ["compare", "vs", "versus", "difference between"]):
            matches.append(PresentationTypeWithScore(PresentationType.COMPARISON_TABLE, "elaborator", 0.9))
        if any(p in t for p in ["how do i", "how to", "steps to", "guide me"]):
            matches.append(PresentationTypeWithScore(PresentationType.HOW_TO, "elaborator", 0.85))
        if any(p in t for p in ["tell me the story", "history of", "what happened"]):
            matches.append(PresentationTypeWithScore(PresentationType.NARRATIVE, "elaborator", 0.8))
        if any(p in t for p in ["who is", "what is", "profile of", "tell me about"]):
            matches.append(PresentationTypeWithScore(PresentationType.PROFILE_CARD, "elaborator", 0.75))
        if any(p in t for p in ["swot"]):
            matches.append(PresentationTypeWithScore(PresentationType.SWOT, "elaborator", 0.95))
        if any(p in t for p in ["quiz", "test me", "exam"]):
            matches.append(PresentationTypeWithScore(PresentationType.MINI_TEST, "elaborator", 0.85))
        if any(p in t for p in ["flashcard", "study"]):
            matches.append(PresentationTypeWithScore(PresentationType.FLASHCARD, "elaborator", 0.8))
        if any(p in t for p in ["formula", "equation", "calculate"]):
            matches.append(PresentationTypeWithScore(PresentationType.FORMULA, "elaborator", 0.9))
        if any(p in t for p in ["timeline", "chronology"]):
            matches.append(PresentationTypeWithScore(PresentationType.TIMELINE, "elaborator", 0.9))
        if any(p in t for p in ["visualize", "chart", "graph", "plot"]):
            matches.append(PresentationTypeWithScore(PresentationType.VISUALIZATION, "elaborator", 0.85))
        if any(p in t for p in ["faq", "common questions"]):
            matches.append(PresentationTypeWithScore(PresentationType.QA, "elaborator", 0.8))

        # Generic / fallback
        if not matches:
            if len(text.split()) < 5:
                matches.append(PresentationTypeWithScore(PresentationType.CLARIFICATION, "elaborator", 0.6))
            else:
                matches.append(PresentationTypeWithScore(PresentationType.MARKDOWN, "elaborator", 0.3))

        # Always add starter and finisher wrappers (markdown)
        starter = PresentationTypeWithScore(PresentationType.MARKDOWN, "starter", 0.95)
        finisher = PresentationTypeWithScore(PresentationType.MARKDOWN, "finisher", 0.9)

        # Combine: starter + elaborators + finisher (avoid duplicate adjacent markdown)
        all_types = [starter] + matches + [finisher]

        avg_conf = sum(m.confidence for m in matches) / len(matches) if matches else 0.0
        return all_types, avg_conf

    def _detect_multiple_intents(self, text: str) -> bool:
        """Heuristic: multiple questions or compound requests."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        questions = [s for s in sentences if s.endswith('?')]
        return len(questions) > 1

    # ------------------------------------------------------------------
    # Desperation scoring
    # ------------------------------------------------------------------

    def _detect_desperation(self, text: str) -> float:
        """Return a desperation score between 0.0 and 1.0."""
        score = 0.0
        t = text.lower()

        keywords = {
            "help": 0.3, "urgent": 0.6, "emergency": 0.7, "asap": 0.5,
            "desperate": 0.7, "noooo": 0.4, "still not": 0.5, "doesn't work": 0.6,
            "wont work": 0.6, "i don't get it": 0.4, "please help": 0.5,
            "not working": 0.5, "can't": 0.3, "cant": 0.3, "broken": 0.4,
        }
        for phrase, weight in keywords.items():
            if phrase in t:
                score = max(score, weight)

        # All‑caps sentences
        for sentence in text.split('.'):
            stripped = sentence.strip()
            if stripped.isupper() and len(stripped) > 10:
                score = max(score, 0.5)

        # Multiple punctuation
        if re.search(r'[?!]{2,}', text):
            score = max(score, 0.3)

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # RAG query builder
    # ------------------------------------------------------------------

    def _build_rag_query(
        self, text: str, types: List[PresentationTypeWithScore]
    ) -> str:
        """Remove stopwords and append type‑hint keywords."""
        words = text.lower().split()
        meaningful = [w for w in words if w not in self.stopwords and not w.startswith('[')]
        hints = self._get_type_hints(types)
        return ' '.join(meaningful + hints)

    def _get_type_hints(self, types: List[PresentationTypeWithScore]) -> List[str]:
        """Return keywords relevant to the requested presentation types."""
        hints = []
        for pt in types:
            t = pt.type
            if t == PresentationType.PROS_CONS:
                hints += ["advantages", "disadvantages", "benefits", "drawbacks"]
            elif t == PresentationType.COMPARISON_TABLE:
                hints += ["comparison", "differences", "similarities", "features"]
            elif t == PresentationType.HOW_TO:
                hints += ["tutorial", "steps", "guide", "instructions"]
            elif t == PresentationType.PROFILE_CARD:
                hints += ["biography", "details", "background", "history"]
            elif t == PresentationType.EVIDENCE_SYNTHESIS:
                hints += ["evidence", "studies", "data", "research"]
            elif t == PresentationType.MINI_TEST:
                hints += ["quiz", "questions", "exam", "assessment"]
            elif t == PresentationType.FLASHCARD:
                hints += ["definitions", "key terms", "concepts"]
            elif t == PresentationType.FORMULA:
                hints += ["equation", "calculation", "math"]
            elif t == PresentationType.TIMELINE:
                hints += ["chronology", "events", "history", "dates"]
            elif t == PresentationType.VISUALIZATION:
                hints += ["chart", "graph", "data", "statistics"]
            elif t == PresentationType.NARRATIVE:
                hints += ["story", "history", "account"]
        return list(set(hints))

    # ------------------------------------------------------------------
    # Named Entity Recognition (spaCy mock)
    # ------------------------------------------------------------------

    def _ner(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy (or a mock if not available)."""
        if self.spacy_model is not None:
            doc = self.spacy_model(text)
            return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        return self._mock_ner(text)

    def _mock_ner(self, text: str) -> List[Dict[str, Any]]:
        """Simple regex‑based NER for common patterns (fallback)."""
        entities = []
        # Dates
        for m in re.finditer(r'\b\d{4}-\d{2}-\d{2}\b', text):
            entities.append({"text": m.group(), "label": "DATE"})
        # Money
        for m in re.finditer(r'\$\d[\d,]*', text):
            entities.append({"text": m.group(), "label": "MONEY"})
        # Emails
        for m in re.finditer(r'\S+@\S+\.\S+', text):
            entities.append({"text": m.group(), "label": "EMAIL"})
        return entities

    # ------------------------------------------------------------------
    # Helper: placeholders map
    # ------------------------------------------------------------------

    def _build_placeholders(
        self, links: List[ExtractedLink], blobs: List[ExtractedBlob]
    ) -> Dict[str, str]:
        """Build a mapping from placeholder to original content or preview."""
        pmap = {}
        for link in links:
            pmap[link.placeholder] = link.raw_url
        for blob in blobs:
            pmap[blob.placeholder] = blob.content_preview
        return pmap

    # ------------------------------------------------------------------
    # Sub‑question detection
    # ------------------------------------------------------------------

    def _detect_sub_questions(self, text: str) -> List[str]:
        """Split text into individual sub‑questions if multiple exist."""
        raw = re.split(r'[?.!]', text)
        sub_qs = [q.strip() + '?' for q in raw if q.strip() and q.strip()[-1] != '?']
        return sub_qs if len(sub_qs) > 1 else []