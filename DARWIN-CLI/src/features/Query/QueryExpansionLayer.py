"""
src/features/Query/QueryExpansionLayer.py
-----------------------------------------
Transforms raw user input into ResolvedQuery(s).

Domain agnostic. Uses spaCy via ModelManager for NER and intent signals.
No hardcoded terms, concepts, or file hints.
"""

from typing import List, Dict, Optional, Literal, Set, Tuple
from dataclasses import dataclass, field
import re

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType

# ═══════════════════════════════════════════════════════════════════════════
# Canonical types (single source of truth)
# ═══════════════════════════════════════════════════════════════════════════
from src.features.Query.Response.ResponseTypes import ResolvedQuery, Entity


# ═══════════════════════════════════════════════════════════════════════════
# spaCy Loader
# ═══════════════════════════════════════════════════════════════════════════

class SpacyLoader:
    """Lazy loader for spaCy nlp."""
    _nlp = None

    @classmethod
    def get_nlp(cls):
        if cls._nlp is None:
            try:
                from src.core.Models import ModelManager, ModelCategory
                loader = ModelManager.get_loader()
                spacy_wrapper = loader.get(ModelCategory.TOKENIZER)
                cls._nlp = spacy_wrapper.nlp
            except Exception as e:
                darwin.log(f"[QueryExpansion] spaCy load failed: {e}", LogType.ERROR)
                cls._nlp = None
        return cls._nlp


# ═══════════════════════════════════════════════════════════════════════════
# Intent Classifier (pattern-based, no domain terms)
# ═══════════════════════════════════════════════════════════════════════════

INTENT_PATTERNS = {
    "broad": [
        r"^\s*(what is|what are|explain|overview|describe|tell me about|how does|what do we know)",
        r"\b(summary|summarize|gist|main points|big picture)\b",
    ],
    "dense": [
        r"\b(find|where|quote|exactly|show me|locate|passage|verse|page|line|cite)\b",
        r"\b(what does it say about|where does it mention|where is|passage about)\b",
    ],
    "both": [
        r"\b(compare|contrast|difference|versus|vs\b|similarities|how are|relationship between)\b",
        r"\b(why|what if|suppose|how would|should|explain why|difference between)\b",
    ],
}


def classify_intent(query: str) -> Literal["broad", "dense", "both"]:
    query_lower = query.lower()
    scores = {"broad": 0, "dense": 0, "both": 0}

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                scores[intent] += 1

    if scores["both"] > 0:
        return "both"
    if scores["dense"] > scores["broad"]:
        return "dense"
    return "broad"


# ═══════════════════════════════════════════════════════════════════════════
# Entity Extractor (spaCy, domain agnostic)
# ═══════════════════════════════════════════════════════════════════════════

def extract_entities(query: str) -> List[Entity]:
    """Extract entities using spaCy. Domain agnostic."""
    nlp = SpacyLoader.get_nlp()
    if nlp is None:
        return []

    doc = nlp(query)
    entities = []

    for ent in doc.ents:
        entities.append(Entity(
            text=ent.text,
            type=ent.label_,
            start=ent.start_char,
            end=ent.end_char,
            label=ent.label_,
        ))

    # Also extract noun chunks as candidate concepts
    for chunk in doc.noun_chunks:
        if not any(e.start == chunk.start_char for e in entities):
            entities.append(Entity(
                text=chunk.text,
                type="NOUN_CHUNK",
                start=chunk.start_char,
                end=chunk.end_char,
                label="NP",
            ))

    return entities


# ═══════════════════════════════════════════════════════════════════════════
# Term Expansion (spaCy, no hardcoded map)
# ═══════════════════════════════════════════════════════════════════════════

def expand_terms(query: str, entities: List[Entity]) -> List[str]:
    """Expand query terms using spaCy. No hardcoded synonyms."""
    nlp = SpacyLoader.get_nlp()
    if nlp is None:
        return query.split()

    doc = nlp(query)
    terms = set()

    # Add lemmatized content words
    stops = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", 
             "is", "are", "was", "were", "be", "been", "have", "has", "had",
             "do", "does", "did", "will", "would", "could", "should", "may", "might"}

    for token in doc:
        if not token.is_stop and not token.is_punct and len(token.lemma_) > 2:
            terms.add(token.lemma_.lower())

    # Add entity texts (lemmatized)
    for e in entities:
        ent_doc = nlp(e.text)
        for token in ent_doc:
            if not token.is_stop and len(token.lemma_) > 2:
                terms.add(token.lemma_.lower())

    return list(terms)


# ═══════════════════════════════════════════════════════════════════════════
# File Router (domain agnostic)
# ═══════════════════════════════════════════════════════════════════════════

def route_files(
    query: str,
    entities: List[Entity],
    pinned_files: List[str],
    store: DarwinStore,
) -> List[str]:
    """
    Infer files from query + store.
    Checks store for files matching entity names.
    """
    inferred = []

    # Get all files from store
    all_files = []
    if hasattr(store, 'list_files'):
        all_files = store.list_files()
    elif hasattr(store, '_chunks'):
        all_files = list(store._chunks.keys())

    # Check if any entity matches a filename
    for e in entities:
        ent_clean = re.sub(r"[^\w\s]", "", e.text.lower())
        for f in all_files:
            file_clean = re.sub(r"[^\w\s]", "", f.lower().replace('.pdf', ''))
            if ent_clean in file_clean or file_clean in ent_clean:
                if f not in pinned_files and f not in inferred:
                    inferred.append(f)

    # Combine: pinned first, then inferred
    seen = set()
    result = []
    for f in pinned_files + inferred:
        if f not in seen:
            seen.add(f)
            result.append(f)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Comparison Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_comparison(query: str, entities: List[Entity], file_count: int) -> bool:
    """Detect if query is a comparison. Uses linguistic patterns only."""
    query_lower = query.lower()

    # Explicit comparison words
    comparison_words = {"compare", "contrast", "versus", "vs", "difference", "similarities",
                        "relationship between", "difference between", "how are", "how do"}

    for word in comparison_words:
        if word in query_lower:
            return True

    # Multiple files + coordinating conjunction
    if file_count > 1:
        if re.search(r"\b(and|or|vs|versus)\b", query_lower):
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

class QueryExpansionLayer:
    """
    Transforms raw user input into one or more ResolvedQueries.
    Domain agnostic. Uses spaCy. No hardcoded terms.
    """

    def __init__(self, store: DarwinStore):
        self.store = store

    def expand(
        self,
        user_input: str,
        pinned_files: Optional[List[str]] = None,
        excluded_files: Optional[List[str]] = None,
    ) -> List[ResolvedQuery]:
        """
        Main entry point.
        Returns list of ResolvedQueries.
        """
        darwin.log(f"[QueryExpansion] Input: '{user_input}'", LogType.INFO)

        pinned = pinned_files or []
        excluded = excluded_files or []

        # 1. spaCy NER
        entities = extract_entities(user_input)
        darwin.log(f"[QueryExpansion] Entities: {[(e.text, e.type) for e in entities]}", LogType.INFO)

        # 2. Classify intent
        intent = classify_intent(user_input)

        # 3. Expand terms (spaCy lemmatization)
        expanded = expand_terms(user_input, entities)

        # 4. Route files
        inferred = route_files(user_input, entities, pinned, self.store)
        all_files = list(dict.fromkeys(pinned + inferred))
        all_files = [f for f in all_files if f not in excluded]

        # 5. Detect comparison
        is_comparison = detect_comparison(user_input, entities, len(all_files))

        if is_comparison and len(all_files) > 1:
            return self._decompose_comparison(
                user_input, intent, expanded, entities, all_files, pinned, excluded
            )

        # 6. Single query
        rq = ResolvedQuery(
            intent=intent,
            original_query=user_input,
            expanded_terms=expanded,
            query_embedding=self._embed(user_input),
            pinned_files=pinned,
            inferred_files=inferred,
            excluded_files=excluded,
            entities=entities,
            requires_comparison=is_comparison,
        )

        darwin.log(f"[QueryExpansion] Output: 1 query, intent={intent}, files={rq.all_files}", LogType.INFO)
        return [rq]

    def _decompose_comparison(
        self,
        user_input: str,
        intent: str,
        expanded_terms: List[str],
        entities: List[Entity],
        all_files: List[str],
        pinned: List[str],
        excluded: List[str],
    ) -> List[ResolvedQuery]:
        """Decompose comparison into per-file queries."""
        darwin.log(f"[QueryExpansion] Decomposing comparison: {all_files}", LogType.INFO)

        # Extract shared subject from noun chunks
        nlp = SpacyLoader.get_nlp()
        shared_subject = user_input
        if nlp:
            doc = nlp(user_input)
            # Remove comparison words and file names to get subject
            subject_tokens = []
            for token in doc:
                if token.lemma_.lower() not in {"compare", "contrast", "versus", "vs", "difference", "and", "or"}:
                    if not any(token.text.lower() in f.lower() for f in all_files):
                        if not token.is_stop and not token.is_punct:
                            subject_tokens.append(token.lemma_.lower())
            if subject_tokens:
                shared_subject = " ".join(subject_tokens)

        queries = []

        for filename in all_files:
            file_entities = [e for e in entities if e.type != "document"]

            rq = ResolvedQuery(
                intent="both",
                original_query=f"{shared_subject} {filename}",
                expanded_terms=expanded_terms,
                query_embedding=self._embed(shared_subject),
                pinned_files=[filename],
                inferred_files=[],
                excluded_files=excluded,
                entities=file_entities,
                requires_comparison=True,
            )
            queries.append(rq)

        darwin.log(f"[QueryExpansion] Decomposed into {len(queries)} queries", LogType.INFO)
        return queries

    def _embed(self, text: str) -> List[float]:
        """Generate embedding. Lazy import to avoid circular deps."""
        try:
            from src.features.Indexing.Embedding.Embedder import embed_query
            return embed_query(text)
        except Exception as e:
            darwin.log(f"[QueryExpansion] Embed failed: {e}", LogType.ERROR)
            return []