import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.features.Query.QueryConfig import OrchestratorConfig
from src.features.Query.lib import (ExtractedLink, ExtractedBlob, PresentationTypeWithScore, PreprocessedQuery,stopwords)




class QueryRouter:
    def __init__(self, config: OrchestratorConfig, spacy_model=None, llm_client=None):
        self.config = config
        self.spacy = spacy_model  # SpacyConceptExtractor instance or mock
        self.llm_client = llm_client
        self.link_pattern = re.compile(r'https?://\S+')
        self.stopwords = stopwords

    def preprocess(self, raw_text: str, pinned_files: Optional[List[str]] = None) -> PreprocessedQuery:
        # 1. Extract links and replace with placeholders
        text_no_links, links = self._extract_links(raw_text)
        # 2. Extract blobs (code, logs)
        cleaned_text, blobs = self._extract_blobs(text_no_links)
        # 3. Classify intent and tiers
        presentation_types, requires_llm = self._classify(cleaned_text)
        # 4. Desperation score
        desperation_score = self._detect_desperation(cleaned_text)
        # 5. Quality assessment (spaCy)
        quality_score, quality_flags = self._assess_quality(cleaned_text)
        # 6. RAG query
        rag_query = self._build_rag_query(cleaned_text, presentation_types)
        # 7. Build placeholder map
        placeholder_map = {}
        for link in links:
            placeholder_map[link.placeholder] = link.raw_url
        for blob in blobs:
            placeholder_map[blob.placeholder] = blob.content_preview
        # 8. Detect sub-questions and multi-intent
        sub_questions = self._detect_sub_questions(cleaned_text)
        is_multi = len(sub_questions) > 1
        # 9. Determine tier
        tier = self._classify_tier(cleaned_text, links, blobs, presentation_types, is_multi)
        # 10. Named entities (spaCy)
        named_entities = self._ner(cleaned_text)

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
            quality_score=quality_score,
            quality_flags=quality_flags,
            tier=tier,
            is_multi_intent=is_multi,
        )

    def _extract_links(self, text: str) -> Tuple[str, List[ExtractedLink]]:
        links = []
        def repl(m):
            url = m.group(0)
            domain = urlparse(url).netloc or "unknown"
            placeholder = f"[link::{domain}]"
            links.append(ExtractedLink(url, domain, placeholder))
            return placeholder
        cleaned = self.link_pattern.sub(repl, text)
        return cleaned, links

    def _extract_blobs(self, text: str) -> Tuple[str, List[ExtractedBlob]]:
        blobs = []
        parts = re.split(r'\n\n+', text)
        cleaned_parts = []
        for part in parts:
            stripped = part.strip()
            if len(stripped) > self.config.blob_threshold or self._is_code_or_log(stripped):
                blob_id = uuid.uuid4().hex[:8]
                placeholder = f"[blob::{blob_id}]"
                blobs.append(ExtractedBlob(
                    blob_id=blob_id,
                    content_preview=stripped[:200],
                    placeholder=placeholder,
                    token_count=len(stripped.split()),
                    is_truncated=len(stripped) > 5000
                ))
                cleaned_parts.append(placeholder)
            else:
                cleaned_parts.append(stripped)
        return '\n\n'.join(cleaned_parts), blobs

    def _is_code_or_log(self, text: str) -> bool:
        indicators = ['{', '}', 'def ', 'import ', '=>', 'Traceback', 'ERROR:', 'WARN:', 'DEBUG:']
        return any(ind in text for ind in indicators)

    def _classify(self, text: str) -> Tuple[List[PresentationTypeWithScore], bool]:
        """Rule-based classification; returns types and whether LLM override needed."""
        # Use simple keyword matching (abbreviated)
        types = []
        t = text.lower()
        if any(p in t for p in ["should i", "pros and cons", "is it worth"]):
            types.append(PresentationTypeWithScore("pros_cons", "elaborator", 0.9))
        elif any(p in t for p in ["compare", "vs", "versus"]):
            types.append(PresentationTypeWithScore("comparison_table", "elaborator", 0.9))
        elif any(p in t for p in ["how do i", "how to", "steps"]):
            types.append(PresentationTypeWithScore("how_to", "elaborator", 0.85))
        elif any(p in t for p in ["tell me the story", "history of"]):
            types.append(PresentationTypeWithScore("narrative", "elaborator", 0.8))
        elif any(p in t for p in ["who is", "what is", "profile"]):
            types.append(PresentationTypeWithScore("profile_card", "elaborator", 0.75))
        elif any(p in t for p in ["quiz", "test me", "exam"]):
            types.append(PresentationTypeWithScore("mini_test", "elaborator", 0.85))
        elif any(p in t for p in ["flashcard", "study"]):
            types.append(PresentationTypeWithScore("flashcard", "elaborator", 0.8))
        elif any(p in t for p in ["formula", "equation"]):
            types.append(PresentationTypeWithScore("formula", "elaborator", 0.9))
        elif any(p in t for p in ["timeline", "chronology"]):
            types.append(PresentationTypeWithScore("timeline", "elaborator", 0.9))
        elif any(p in t for p in ["visualize", "chart", "graph"]):
            types.append(PresentationTypeWithScore("visualization", "elaborator", 0.85))
        else:
            types.append(PresentationTypeWithScore("markdown", "elaborator", 0.3))

        # Add starter/finisher
        starter = PresentationTypeWithScore("markdown", "starter", 0.95)
        finisher = PresentationTypeWithScore("markdown", "finisher", 0.9)
        all_types = [starter] + types + [finisher]

        avg_conf = sum(t.confidence for t in types) / len(types) if types else 0.0
        needs_llm = avg_conf < 0.5 or self._detect_multiple_intents(text)
        return all_types, needs_llm

    def _detect_multiple_intents(self, text: str) -> bool:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        questions = [s for s in sentences if s.endswith('?')]
        return len(questions) > 1

    def _detect_desperation(self, text: str) -> float:
        score = 0.0
        t = text.lower()
        keywords = {
            "help": 0.3, "urgent": 0.6, "emergency": 0.7, "asap": 0.5,
            "noooo": 0.4, "still not": 0.5, "doesn't work": 0.6,
            "wont work": 0.6, "i don't get it": 0.4, "please help": 0.5,
        }
        for phrase, weight in keywords.items():
            if phrase in t:
                score = max(score, weight)
        # All caps sentences
        for sentence in text.split('.'):
            stripped = sentence.strip()
            if stripped.isupper() and len(stripped) > 10:
                score = max(score, 0.5)
        if re.search(r'[?!]{2,}', text):
            score = max(score, 0.3)
        return min(score, 1.0)

    def _assess_quality(self, text: str) -> Tuple[float, List[str]]:
        flags = []
        score = 1.0

        # Meaningful token check using spaCy
        if self.spacy and hasattr(self.spacy, 'nlp'):
            doc = self.spacy.nlp(text[:500])
            meaningful = [t for t in doc if not t.is_stop and not t.is_punct and t.pos_ != "X"]
            if len(meaningful) < self.config.min_meaningful_tokens:
                flags.append("too_short")
                score -= 0.4
            # Junk token ratio
            total_tokens = len(doc)
            if total_tokens > 0:
                junk = sum(1 for t in doc if t.pos_ == "X")
                if junk / total_tokens > self.config.max_junk_token_ratio:
                    flags.append("high_junk_ratio")
                    score -= 0.3
        else:
            # Fallback: simple word count
            words = [w for w in text.split() if w.lower() not in self.stopwords]
            if len(words) < self.config.min_meaningful_tokens:
                flags.append("too_short")
                score -= 0.4

        # Profanity check (simple list)
        if self.config.enable_profanity_filter and self.config.profanity_list:
            lower = text.lower()
            if any(word in lower for word in self.config.profanity_list):
                flags.append("profanity")
                score -= 0.2

        # Entity presence (optional) — if no entities and query is vague
        # We'll just note it; doesn't heavily penalize unless domain-specific.

        return max(0.0, score), flags

    def _build_rag_query(self, text: str, types: List[PresentationTypeWithScore]) -> str:
        words = text.lower().split()
        meaningful = [w for w in words if w not in self.stopwords and not w.startswith('[')]
        hints = []
        for pt in types:
            if pt.type == "pros_cons":
                hints += ["advantages", "disadvantages"]
            elif pt.type == "comparison_table":
                hints += ["comparison", "differences"]
            elif pt.type == "how_to":
                hints += ["tutorial", "steps", "guide", "instructions"]
            elif pt.type == "profile_card":
                hints += ["biography", "details", "background", "history"]
            elif pt.type == "evidence_synthesis":
                hints += ["evidence", "studies", "data", "research"]
            elif pt.type == "mini_test":
                hints += ["quiz", "questions", "exam", "assessment"]
            elif pt.type == "flashcard":
                hints += ["definitions", "key terms", "concepts"]
            elif pt.type == "formula":
                hints += ["equation", "calculation", "math"]
            elif pt.type == "timeline":
                hints += ["chronology", "events", "history", "dates"]
            elif pt.type == "visualization":
                hints += ["chart", "graph", "data", "statistics"]
            elif pt.type == "narrative":
                hints += ["story", "history", "account"]
        
        return ' '.join(meaningful + hints)

    def _ner(self, text: str) -> List[Dict[str, Any]]:
        if self.spacy and hasattr(self.spacy, 'extract_entities'):
            return self.spacy.extract_entities(text[:3000])
        return []

    def _detect_sub_questions(self, text: str) -> List[str]:
        raw = re.split(r'[?.!]', text)
        return [q.strip() + '?' for q in raw if q.strip() and not q.strip().endswith('?')]

    def _classify_tier(self, text: str, links, blobs, types, is_multi) -> str:
        if is_multi:
            return "simple"  # We'll answer concisely, then let user choose
        if text.lower() in ["?", "??", "thanks", "ok", "okay", "cool"]:
            return "simple"
        if any(t.type in ["mini_test", "flashcard", "evidence_synthesis", "diagnostic_tree"] for t in types):
            return "complex"
        if links or blobs:
            return "complex"
        # Default moderate for most questions
        return "moderate"