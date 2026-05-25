"""
src/core/Knowledge/Graph/GraphExtractor.py
------------------------------------------
Knowledge graph extraction using spaCy NER + dependency parsing.
Document-agnostic. Now gates structural chunks BEFORE extraction.
Includes dynamic batch sizing with pre-emptive text-length estimation.
"""

import gc
import re
import math
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field

from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Ingestion.Chunking.lib import Chunk
from src.features.Knowledge.Graph.GraphUtils import (
    GraphData,
    ConceptNode,
    Relationship,
    normalize_entity,
    lemmatize_verb,
    detect_communities_louvain,
    extract_narrative_keyphrase,
    extract_community_keywords,
    CommunityCluster,
)
from src.features.Knowledge.Graph.GraphConfig import GraphExtractionConfig
from src.features.Storage.DarwinStore import DarwinStore
from src.Models import ModelManager, ModelCategory


# ═══════════════════════════════════════════════════════════════════════════
# Document-agnostic structural chunk detection
# ═══════════════════════════════════════════════════════════════════════════

STRUCTURAL_SIGNALS = {
    "cover", "title", "copyright", "table of contents", "contents",
    "acknowledgments", "preface", "foreword", "introduction", "about",
    "appendix", "index", "glossary", "bibliography", "references",
    "page", "chapter", "book", "part", "section", "volume",
    "ebook", "edition", "published", "publisher", "isbn",
    "all rights reserved", "printed in", "first edition",
    "dedication", "prologue", "epilogue", "notes", "footnote",
    "abstract", "summary", "overview", "list of", "figure", "table",
    "extras", "chart", "works cited", "front matter", "back matter",
}

SECTION_PREFIX_RE = re.compile(
    r'^(Book|Chapter|Ch\.|Part|Section|Vol\.|Volume|Appendix|Excerpt|'
    r'Reference|Note|Figure|Table|Chart|Map|Illustration)s?\s*[\dIVX]+',
    re.IGNORECASE
)

TOC_ENTRY_RE = re.compile(
    r'([\.•\-–—]\s*\d+\s*$)|(^\d+[\.\)]\s+\w+)',
    re.IGNORECASE
)

VERSE_CITATION_RE = re.compile(r'\b\w+\s+\d+:\d+\b')
CROSS_REF_RE = re.compile(r'\b(?:cf\.|see|compare|v\.|vv\.|verses?|ch\.|chap\.|pp\.|p\.)\b', re.IGNORECASE)


def _is_structural_chunk(chunk: Chunk, all_chunks: List[Chunk]) -> bool:
    """Detect structural chunks by content — document-agnostic."""
    text = chunk.clean
    text_lower = text.lower()
    words = set(re.findall(r'\b[a-z]+\b', text_lower))
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if len(text) < 200:
        return True

    structural_found = words & STRUCTURAL_SIGNALS
    if len(words) > 0 and len(structural_found) / len(words) > 0.30:
        return True

    if len(lines) > 3:
        short_numbered = sum(1 for l in lines if len(l) < 60 and re.search(r'\d', l))
        if short_numbered / len(lines) > 0.5:
            return True

    if len(lines) > 2:
        citation_lines = sum(1 for l in lines if VERSE_CITATION_RE.search(l) or CROSS_REF_RE.search(l))
        if citation_lines / len(lines) > 0.3:
            return True

    if len(lines) > 4:
        book_ref_lines = sum(1 for l in lines if re.match(r'^[A-Za-z]+\s+\d+[:.\-]\d*$', l.strip()))
        if book_ref_lines / len(lines) > 0.4:
            return True

    text_hash = hash(text_lower[:200])
    repeat_count = sum(1 for c in all_chunks if hash(c.clean.lower()[:200]) == text_hash)
    if repeat_count > 2:
        return True

    if chunk.index <= len(all_chunks) * 0.05 and len(words) < 20:
        return True

    section_headers = sum(1 for l in lines if SECTION_PREFIX_RE.match(l))
    if section_headers >= 3:
        return True

    if len(lines) > 3:
        toc_entries = sum(1 for l in lines if TOC_ENTRY_RE.search(l))
        if toc_entries / len(lines) > 0.40:
            return True

    if chunk.index <= len(all_chunks) * 0.10 and len(lines) > 0:
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        if (len(words) > 0 and
            len(structural_found) / len(words) > 0.15 and
            avg_line_len < 80):
            return True

    if len(lines) >= 3:
        prefixes = []
        for l in lines:
            m = re.match(r'^([A-Za-z]+)[\s\-–—]', l)
            if m:
                prefixes.append(m.group(1).lower())
        if prefixes:
            most_common = Counter(prefixes).most_common(1)[0]
            if most_common[1] >= 3:
                return True

    return False


def _is_obvious_structural(chunk: Chunk, all_chunks: Optional[List[Chunk]] = None) -> bool:
    """Light-mode filter: only obviously structural chunks (>50% signals)."""
    text = chunk.clean.lower()
    words = set(re.findall(r'\b[a-z]+\b', text))
    structural_found = words & STRUCTURAL_SIGNALS
    return len(words) > 0 and len(structural_found) / len(words) > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Verb synonym groups
# ═══════════════════════════════════════════════════════════════════════════

VERB_SYNONYMS = {
    "develop": {"develop", "create", "build", "design", "implement", "construct"},
    "propose": {"propose", "suggest", "introduce", "present", "advocate"},
    "train": {"train", "fine-tune", "finetune", "optimize", "calibrate"},
    "evaluate": {"evaluate", "assess", "measure", "test", "benchmark", "validate"},
    "outperform": {"outperform", "exceed", "surpass", "beat", "top"},
    "compare": {"compare", "contrast", "versus", "vs", "against"},
    "use": {"use", "utilize", "employ", "apply", "leverage", "adopt"},
    "show": {"show", "demonstrate", "illustrate", "reveal", "indicate"},
    "achieve": {"achieve", "attain", "reach", "obtain", "accomplish"},
    "analyze": {"analyze", "examine", "investigate", "study", "explore"},
    "improve": {"improve", "enhance", "boost", "increase", "augment"},
    "include": {"include", "contain", "comprise", "incorporate", "encompass"},
    "require": {"require", "need", "demand", "necessitate"},
    "support": {"support", "enable", "facilitate", "allow", "permit"},
    "depend": {"depend", "rely", "hinge", "rest"},
}

_VERB_TO_CANONICAL = {}
for canonical, synonyms in VERB_SYNONYMS.items():
    for syn in synonyms:
        _VERB_TO_CANONICAL[syn] = canonical


def canonicalize_svo(subject: str, verb: str, obj: str) -> tuple:
    s = normalize_entity(subject)
    v = lemmatize_verb(verb)
    o = normalize_entity(obj)
    v = _VERB_TO_CANONICAL.get(v, v)
    return (s, v, o)


# ═══════════════════════════════════════════════════════════════════════════
# SpecificityScorer
# ═══════════════════════════════════════════════════════════════════════════

class SpecificityScorer:
    def __init__(self, n_chunks: int):
        self.n_chunks = max(n_chunks, 1)
        self.doc_freq: Counter = Counter()
        self.chunk_bitsets: dict = {}

    def observe(self, candidate: str, chunk_idx: int):
        c = candidate.lower().strip()
        self.doc_freq[c] += 1
        if c not in self.chunk_bitsets:
            self.chunk_bitsets[c] = 0
        self.chunk_bitsets[c] |= (1 << min(chunk_idx, 63))

    def score(self, text: str, spacy_tokens=None) -> float:
        canonical = text.lower().strip()
        words = canonical.split()
        score = 0.0

        score += min(len(text) / 25, 0.15)
        score += min(len(words) / 5, 0.15)

        if text and text[0].isupper():
            score += 0.10
        if text.isupper() and 2 <= len(text) <= 6:
            score += 0.10

        if len(words) >= 2:
            score += 0.10
        if len(words) >= 3:
            score += 0.05

        df = bin(self.chunk_bitsets.get(canonical, 0)).count("1")
        if df > 0 and self.n_chunks > 1:
            idf = math.log(self.n_chunks / df)
            max_idf = math.log(self.n_chunks)
            score += (idf / max_idf) * 0.25

        stops = {
            "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
            "with", "by", "from", "as", "is", "was", "are", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "not", "no", "so", "if",
            "then", "than", "that", "this", "these", "those", "it", "its",
            "he", "she", "they", "we", "you", "i", "me", "him", "her", "them",
        }
        if words:
            stop_ratio = sum(1 for w in words if w in stops) / len(words)
            score -= stop_ratio * 0.20

        if spacy_tokens:
            noun_ratio = sum(1 for t in spacy_tokens if t.pos_ in ("NOUN", "PROPN")) / max(len(spacy_tokens), 1)
            score += noun_ratio * 0.10

        return max(0.0, min(1.0, score))


# ═══════════════════════════════════════════════════════════════════════════
# EntityRegistry
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EntityRegistry:
    canonical_to_id: Dict[str, str] = field(default_factory=dict)
    id_to_node: Dict[str, ConceptNode] = field(default_factory=dict)
    mention_to_canonical: Dict[str, str] = field(default_factory=dict)
    next_id: int = 0

    def get_or_create(self, text: str, entity_type: str, chunk_idx: int) -> str:
        canonical = normalize_entity(text)
        if not canonical or len(canonical) < 2:
            return None

        if canonical in self.canonical_to_id:
            node_id = self.canonical_to_id[canonical]
            node = self.id_to_node[node_id]
            if chunk_idx not in node.chunk_indices:
                node.chunk_indices.append(chunk_idx)
                node.frequency += 1
            if text not in node.mention_forms:
                node.mention_forms.append(text)
            return node_id

        for mention, known_canonical in self.mention_to_canonical.items():
            if self._is_alias(canonical, known_canonical):
                node_id = self.canonical_to_id[known_canonical]
                node = self.id_to_node[node_id]
                if chunk_idx not in node.chunk_indices:
                    node.chunk_indices.append(chunk_idx)
                    node.frequency += 1
                if text not in node.mention_forms:
                    node.mention_forms.append(text)
                self.mention_to_canonical[canonical] = known_canonical
                return node_id

        node_id = f"e{self.next_id}"
        self.next_id += 1

        node = ConceptNode(
            id=node_id,
            label=canonical,
            display_name=text,
            entity_type=entity_type,
            frequency=1,
            chunk_indices=[chunk_idx],
            first_mention_chunk=chunk_idx,
            mention_forms=[text],
        )

        self.canonical_to_id[canonical] = node_id
        self.id_to_node[node_id] = node
        self.mention_to_canonical[canonical] = canonical

        return node_id

    @staticmethod
    def _is_alias(a: str, b: str) -> bool:
        if a == b:
            return True
        if a in b or b in a:
            return True
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return False
        intersection = words_a & words_b
        union = words_a | words_b
        jaccard = len(intersection) / len(union)
        return jaccard >= 0.5


@dataclass
class RelationAccumulator:
    canonical_relations: Dict[tuple, List[Tuple[int, float, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, subject_text: str, verb_text: str, object_text: str,
            chunk_idx: int, weight: float = 1.0, spacy_doc=None) -> None:
        s, v, o = canonicalize_svo(subject_text, verb_text, object_text)
        if not s or not v or not o:
            return
        if s == o:
            return
        key = (s, v, o)
        self.canonical_relations[key].append((chunk_idx, weight, verb_text))

    def get_confident_relations(self, entity_registry: EntityRegistry,
                                threshold: float = 0.6) -> List[Relationship]:
        edges = []
        for (s_canon, v_canon, o_canon), occurrences in self.canonical_relations.items():
            if o_canon == "passive":
                continue
            src_id = entity_registry.canonical_to_id.get(s_canon)
            tgt_id = entity_registry.canonical_to_id.get(o_canon)
            if not src_id or not tgt_id:
                continue

            src_node = entity_registry.id_to_node[src_id]
            tgt_node = entity_registry.id_to_node[tgt_id]

            chunk_indices = sorted(list(set(idx for idx, _, _ in occurrences)))
            surface_verbs = sorted(list(set(vb for _, _, vb in occurrences)))
            occurrence_count = len(chunk_indices)

            if threshold <= 0:
                confidence = 1.0
                keep = True
            elif threshold < 1.0:
                shared_chunks = set(src_node.chunk_indices) & set(tgt_node.chunk_indices)
                if not shared_chunks:
                    continue
                confidence = occurrence_count / len(shared_chunks)
                keep = confidence >= threshold
            else:
                confidence = min(occurrence_count / 3.0, 1.0)
                keep = occurrence_count >= threshold

            if keep:
                edges.append(Relationship(
                    source=src_id,
                    target=tgt_id,
                    weight=round(confidence, 4),
                    relation_type=v_canon,
                    evidence_chunks=chunk_indices,
                    surface_forms=surface_verbs,
                ))
        return edges


# ═══════════════════════════════════════════════════════════════════════════
# spaCy processing
# ═══════════════════════════════════════════════════════════════════════════

def _load_spacy(config: GraphExtractionConfig):
    try:
        loader = ModelManager.get_loader()
        extractor = loader.get(ModelCategory.TOKENIZER)
    except Exception as e:
        darwin.log(f"Failed to load spaCy tokenizer: {e}", LogType.ERROR)
        return None

    darwin.log(f"spaCy pipes available: {sorted(extractor._enabled_pipes)}", LogType.INFO)
    darwin.log(f"has_ner={extractor.has_ner}, has_parser={extractor.has_parser}", LogType.INFO)

    if config.require_typed_entities and not extractor.has_ner:
        darwin.log("NER required but unavailable in spaCy model", LogType.ERROR)
        return None

    if config.require_parser and not extractor.has_parser:
        darwin.log("Dependency parser required but unavailable", LogType.ERROR)
        return None

    return extractor


def _is_bad_entity(ent_text: str) -> bool:
    clean = ent_text.strip()
    lower = clean.lower()

    if len(clean) < 3:
        return True
    if not any(c.isalpha() for c in clean):
        return True

    reject_patterns = [
        r"^section\s+\d+",
        r"^chapter\s+\d+",
        r"^figure\s+\d+",
        r"^table\s+\d+",
        r"^appendix\s+\w+",
        r"^part\s+\d+",
        r"^eq\.?\s*\d+",
        r"^equation\s*\d+",
        r"^page\s+\d+",
    ]
    if any(re.match(p, lower) for p in reject_patterns):
        return True

    if lower in {"et al", "et.", "al.", "et al.", "ibid", "op. cit."}:
        return True

    if re.match(r"^[a-z]+\s+\d+$", lower):
        return True

    if len(clean.replace(".", "").replace(" ", "")) <= 2:
        return True

    if re.match(r"^\d{4}$", clean):
        return True

    if lower.count(".") >= 3 or lower.count(" - ") >= 2 or lower.count(" ch") >= 2:
        return True

    structural_fragments = {
        "cover", "copyright", "title page", "table of contents",
        "acknowledgments", "preface", "foreword", "introduction - about",
        "appendix", "index of", "glossary", "bibliography",
    }
    if any(frag in lower for frag in structural_fragments):
        return True

    if re.search(r"\b(ch|chapter|book|part|section)\b.*\b(ch|chapter|book|part|section)\b", lower):
        return True

    return False


def _extract_entities_from_doc(doc, chunk_idx: int, registry: EntityRegistry,
                                config: GraphExtractionConfig,
                                scorer: SpecificityScorer = None) -> List[str]:
    candidates = []

    if config.require_typed_entities and doc.ents:
        for ent in doc.ents:
            if ent.label_ in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}:
                continue
            if _is_bad_entity(ent.text):
                continue
            if scorer:
                scorer.observe(ent.text, chunk_idx)
            spec = scorer.score(ent.text, ent) if scorer else 0.5
            candidates.append((ent.text, ent.label_, spec, ent))

    if hasattr(doc, "noun_chunks"):
        for chunk in doc.noun_chunks:
            clean_tokens = []
            spacy_tokens = []
            for t in chunk:
                if t.is_stop or t.is_punct or len(t.text) <= 2:
                    continue
                clean_tokens.append(t.text)
                spacy_tokens.append(t)
            clean_text = " ".join(clean_tokens)
            if len(clean_text) < 3:
                continue
            if not any(c.isalpha() for c in clean_text):
                continue
            if _is_bad_entity(clean_text):
                continue
            if scorer:
                scorer.observe(clean_text, chunk_idx)
            spec = scorer.score(clean_text, spacy_tokens) if scorer else 0.3
            candidates.append((clean_text, "CONCEPT", spec, spacy_tokens))

    if hasattr(doc, "sents"):
        for sent in doc.sents:
            tokens = [t for t in sent if t.pos_ in ("PROPN", "NOUN", "ADJ")]
            i = 0
            while i < len(tokens):
                if tokens[i].pos_ == "PROPN":
                    phrase_tokens = [tokens[i]]
                    j = i + 1
                    while j < len(tokens) and tokens[j].i == tokens[j - 1].i + 1 and tokens[j].pos_ in ("PROPN", "NOUN", "ADJ"):
                        phrase_tokens.append(tokens[j])
                        j += 1
                    if len(phrase_tokens) >= 2:
                        phrase = " ".join(t.text for t in phrase_tokens)
                        if not _is_bad_entity(phrase):
                            if scorer:
                                scorer.observe(phrase, chunk_idx)
                            spec = scorer.score(phrase, phrase_tokens) if scorer else 0.4
                            candidates.append((phrase, "PROPN_PHRASE", spec, phrase_tokens))
                    i = j
                else:
                    i += 1

    seen = set()
    unique = []
    for text, etype, spec, tokens in candidates:
        c = text.lower().strip()
        if c in seen:
            continue
        seen.add(c)
        unique.append((text, etype, spec, tokens))

    unique.sort(key=lambda x: x[2], reverse=True)

    entity_ids = []
    for text, etype, spec, tokens in unique[:8]:
        if spec < 0.15:
            continue
        node_id = registry.get_or_create(text=text, entity_type=etype, chunk_idx=chunk_idx)
        if node_id:
            entity_ids.append(node_id)

    return entity_ids


def _get_full_phrase(token) -> str:
    if token is None:
        return None
    tokens = [token]
    def _collect_modifiers(t, collected):
        for child in t.children:
            if child.dep_ in ("compound", "amod", "nmod", "nummod", "det", "poss"):
                if child not in collected:
                    collected.append(child)
                    _collect_modifiers(child, collected)
    _collect_modifiers(token, tokens)
    tokens.sort(key=lambda t: t.i)
    phrase = " ".join(t.text for t in tokens)
    phrase = phrase.strip()
    if len(phrase) < 2:
        return None
    return phrase


def _extract_active_svo(sent, root, chunk_idx, relation_acc):
    subjects, objects = [], []
    for token in sent:
        if token.head == root:
            if token.dep_ in ("nsubj", "nsubj:pass"):
                subj = _get_full_phrase(token)
                if subj:
                    subjects.append(subj)
            elif token.dep_ in ("dobj", "attr", "oprd"):
                obj = _get_full_phrase(token)
                if obj:
                    objects.append(obj)
    for subj in subjects:
        for obj in objects:
            relation_acc.add(subj, root.lemma_, obj, chunk_idx)


def _extract_passive_svo(sent, root, chunk_idx, relation_acc):
    passive_subjs, agents = [], []
    for token in sent:
        if token.head == root:
            if token.dep_ == "nsubj:pass":
                ps = _get_full_phrase(token)
                if ps:
                    passive_subjs.append(ps)
            elif token.dep_ in ("agent",) and token.head.lemma_ in ("by", "via"):
                ag = _get_full_phrase(token)
                if ag:
                    agents.append(ag)
    if passive_subjs and not agents:
        for ps in passive_subjs:
            relation_acc.add(ps, root.lemma_, "passive", chunk_idx)
    elif passive_subjs and agents:
        for ag in agents:
            for ps in passive_subjs:
                relation_acc.add(ag, root.lemma_, ps, chunk_idx)


def _extract_prepositional_rel(sent, root, chunk_idx, relation_acc):
    for token in sent:
        if token.head == root and token.dep_ == "prep":
            prep = token.lemma_
            for pobj in sent:
                if pobj.head == token and pobj.dep_ == "pobj":
                    obj_text = _get_full_phrase(pobj)
                    if obj_text:
                        subj = _get_full_phrase(root)
                        if subj:
                            relation_acc.add(subj, prep, obj_text, chunk_idx)


def _extract_copular_rel(sent, root, chunk_idx, relation_acc):
    subjects, complements = [], []
    for token in sent:
        if token.head == root:
            if token.dep_ in ("nsubj", "nsubj:pass"):
                subj = _get_full_phrase(token)
                if subj:
                    subjects.append(subj)
            elif token.dep_ in ("attr", "dobj", "acomp", "pobj"):
                comp = _get_full_phrase(token)
                if comp:
                    complements.append(comp)
    for subj in subjects:
        for comp in complements:
            relation_acc.add(subj, "is", comp, chunk_idx)


def _extract_nominal_rel(sent, root, chunk_idx, relation_acc):
    for token in sent:
        if token.head == root and token.dep_ == "prep":
            prep = token.lemma_
            for pobj in sent:
                if pobj.head == token and pobj.dep_ == "pobj":
                    obj_text = _get_full_phrase(pobj)
                    if obj_text:
                        subj = _get_full_phrase(root)
                        if subj:
                            relation_acc.add(subj, prep, obj_text, chunk_idx)


def _extract_svo_from_doc(doc, chunk_idx: int, relation_acc: RelationAccumulator) -> None:
    for sent in doc.sents:
        root = None
        for token in sent:
            if token.dep_ == "ROOT":
                root = token
                break
        if not root:
            continue
        if root.pos_ == "VERB":
            _extract_active_svo(sent, root, chunk_idx, relation_acc)
            _extract_passive_svo(sent, root, chunk_idx, relation_acc)
            _extract_prepositional_rel(sent, root, chunk_idx, relation_acc)
        elif root.lemma_ in ("be", "seem", "appear", "become"):
            _extract_copular_rel(sent, root, chunk_idx, relation_acc)
        elif root.pos_ in ("NOUN", "PROPN"):
            _extract_nominal_rel(sent, root, chunk_idx, relation_acc)


# ═══════════════════════════════════════════════════════════════════════════
# Co-occurrence edges
# ═══════════════════════════════════════════════════════════════════════════

def _build_cooccurrence_edges(chunk_entity_map: Dict[int, List[str]],
                               similarity_threshold: Optional[float] = None,
                               max_edges: int = 500) -> List[Relationship]:
    cooccurrence_counts: Dict[Tuple[str, str], int] = Counter()
    evidence: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for chunk_idx, entity_ids in chunk_entity_map.items():
        if len(entity_ids) < 2 or len(entity_ids) > 15:
            continue
        for i, e1 in enumerate(entity_ids):
            for e2 in entity_ids[i + 1:]:
                if e1 == e2:
                    continue
                key = tuple(sorted((e1, e2)))
                cooccurrence_counts[key] += 1
                evidence[key].append(chunk_idx)

    if not cooccurrence_counts:
        return []

    sorted_pairs = cooccurrence_counts.most_common(max_edges)
    max_count = sorted_pairs[0][1] if sorted_pairs else 1

    edges = []
    for (e1, e2), count in sorted_pairs:
        weight = count / max_count
        if similarity_threshold is not None and weight < similarity_threshold:
            continue
        edges.append(Relationship(
            source=e1,
            target=e2,
            weight=round(weight, 4),
            relation_type="co_occurrence",
            evidence_chunks=sorted(list(set(evidence[(e1, e2)]))),
            surface_forms=[],
        ))
    return edges


# ═══════════════════════════════════════════════════════════════════════════
# Pareto pruning
# ═══════════════════════════════════════════════════════════════════════════

def _pareto_prune(registry: EntityRegistry, relations: List[Relationship],
                  cooccurrence_edges: List[Relationship], max_concepts: int):
    nodes = list(registry.id_to_node.values())
    if not nodes:
        return [], []

    max_freq = max(n.frequency for n in nodes)
    freq_scores = {n.id: n.frequency / max_freq for n in nodes}

    edge_counts: Dict[str, int] = Counter()
    for e in relations + cooccurrence_edges:
        edge_counts[e.source] += 1
        edge_counts[e.target] += 1
    max_edges = max(edge_counts.values()) if edge_counts else 1
    density_scores = {n.id: edge_counts.get(n.id, 0) / max_edges for n in nodes}

    all_edges = relations + cooccurrence_edges
    degree: Dict[str, int] = Counter()
    for e in all_edges:
        degree[e.source] += 1
        degree[e.target] += 1
    max_degree = max(degree.values()) if degree else 1
    centrality_scores = {n.id: degree.get(n.id, 0) / max_degree for n in nodes}

    combined = []
    for node in nodes:
        score = (
            freq_scores[node.id] * 0.4 +
            density_scores[node.id] * 0.3 +
            centrality_scores[node.id] * 0.3
        )
        combined.append((node, score))

    combined.sort(key=lambda x: x[1], reverse=True)
    keep_ids = {n.id for n, _ in combined[:max_concepts]}

    pruned_nodes = [n for n in nodes if n.id in keep_ids]
    pruned_relations = [e for e in relations if e.source in keep_ids and e.target in keep_ids]
    pruned_cooccurrence = [e for e in cooccurrence_edges if e.source in keep_ids and e.target in keep_ids]

    seen_pairs: Set[Tuple[str, str]] = set()
    final_edges = []
    for e in pruned_relations:
        key = tuple(sorted((e.source, e.target)))
        seen_pairs.add(key)
        final_edges.append(e)
    for e in pruned_cooccurrence:
        key = tuple(sorted((e.source, e.target)))
        if key not in seen_pairs:
            seen_pairs.add(key)
            final_edges.append(e)

    return pruned_nodes, final_edges


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic batch sizing for spaCy
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_batch_size(texts: List[str], base_size: int) -> int:
    """
    Pre-emptively lower batch size for very long texts to prevent OOM.
    """
    if not texts:
        return base_size
    avg_len = sum(len(t) for t in texts) / len(texts)
    if avg_len > 4000:
        return max(1, base_size // 8)
    elif avg_len > 2000:
        return max(1, base_size // 4)
    elif avg_len > 1000:
        return max(1, base_size // 2)
    return base_size


def _process_with_dynamic_batch(
    extractor,
    texts: List[str],
    initial_batch_size: int,
) -> List[Any]:
    """
    Run spaCy nlp.pipe with automatic batch size reduction on OOM.
    Falls back to sequential (batch_size=1) with GC if needed.
    """
    batch_size = max(1, initial_batch_size)
    last_error = None

    while batch_size >= 1:
        try:
            darwin.log(
                f"spaCy processing {len(texts)} texts with batch_size={batch_size}",
                LogType.INFO,
            )
            return list(extractor.nlp.pipe(texts, batch_size=batch_size))
        except MemoryError as e:
            last_error = e
            if batch_size == 1:
                break
            new_batch = max(1, batch_size // 2)
            darwin.log(
                f"MemoryError at batch_size={batch_size}, halving to {new_batch}...",
                LogType.WARNING,
            )
            batch_size = new_batch
            gc.collect()
        except Exception:
            raise

    # Final fallback: sequential with explicit GC
    darwin.log(
        "MemoryError even at batch_size=1. Falling back to sequential "
        "processing with garbage collection.",
        LogType.WARNING,
    )

    docs = []
    for i, text in enumerate(texts):
        try:
            docs.append(extractor.nlp(text))
        except MemoryError:
            darwin.log(f"MemoryError on doc {i}/{len(texts)}. Skipping.", LogType.ERROR)
            docs.append(extractor.nlp(""))
        if i % 10 == 0:
            gc.collect()

    return docs


# ═══════════════════════════════════════════════════════════════════════════
# Main extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════

def extract_graph_from_chunks(
    chunks: List[Chunk],
    filename: str,
    config: GraphExtractionConfig = None,
    chunk_vectors: Optional[np.ndarray] = None,
) -> GraphData:
    if config is None:
        config = GraphExtractionConfig()

    # Filter structural chunks before any processing
    n_all = len(chunks)
    valid_chunks = [c for c in chunks if not _is_structural_chunk(c, chunks)]
    n_filtered = n_all - len(valid_chunks)

    if n_filtered > 0:
        darwin.log(
            f"[GraphExtractor] Filtered {n_filtered}/{n_all} structural chunks",
            LogType.INFO,
        )

    n_chunks = len(valid_chunks)
    darwin.log(
        f"Extracting graph: {filename} ({n_chunks} valid / {n_all} total chunks)",
        LogType.INFO,
    )

    extractor = _load_spacy(config)
    if extractor is None:
        darwin.log("spaCy unavailable — cannot extract graph", LogType.ERROR)
        return GraphData(
            filename=filename,
            chunk_count=n_all,
            total_entities_extracted=0,
            total_relations_extracted=0,
            config_snapshot=config.to_dict(),
            processing_mode="failed",
        )

    registry = EntityRegistry()
    relation_acc = RelationAccumulator()
    chunk_entity_map: Dict[int, List[str]] = {}

    scorer = SpecificityScorer(n_chunks=len(valid_chunks))

    darwin.log(f"[1/3] Extracting entities and relations...", LogType.INFO)

    texts = [chunk.clean[:5000] for chunk in valid_chunks]

    # Dynamic batch sizing with pre-emptive estimation
    estimated_batch = _estimate_batch_size(texts, config.batch_size_spacy)
    docs = _process_with_dynamic_batch(extractor, texts, estimated_batch)

    valid_index_map = {i: valid_chunks[i].index for i in range(len(valid_chunks))}

    for local_idx, (chunk, doc) in enumerate(zip(valid_chunks, docs)):
        original_chunk_idx = valid_index_map[local_idx]
        entity_ids = _extract_entities_from_doc(doc, local_idx, registry, config, scorer)
        chunk_entity_map[local_idx] = entity_ids

        if config.require_parser and extractor.has_parser:
            _extract_svo_from_doc(doc, local_idx, relation_acc)

    darwin.log(
        f"[1/3] Done — {len(registry.id_to_node)} raw entities, "
        f"{len(relation_acc.canonical_relations)} raw relations",
        LogType.SUCCESS,
    )

    darwin.log(
        f"[2/3] Deduplicating relations (threshold={config.relation_confidence_threshold})...",
        LogType.INFO,
    )

    semantic_relations = relation_acc.get_confident_relations(
        registry,
        threshold=config.relation_confidence_threshold,
    )

    darwin.log(f"[2/3] Done — {len(semantic_relations)} confident relations", LogType.SUCCESS)

    darwin.log("[3/3] Building co-occurrence edges...", LogType.INFO)

    cooccurrence_edges = _build_cooccurrence_edges(
        chunk_entity_map,
        config.similarity_threshold,
        max_edges=min(config.max_cooccurrence_edges, len(registry.id_to_node) * 10),
    )

    darwin.log(f"[3/3] Done — {len(cooccurrence_edges)} co-occurrence edges", LogType.SUCCESS)

    darwin.log(f"Pruning to max {config.max_concepts} concepts...", LogType.INFO)

    pruned_nodes, pruned_edges = _pareto_prune(
        registry,
        semantic_relations,
        cooccurrence_edges,
        config.max_concepts,
    )

    darwin.log(f"Pruned: {len(pruned_nodes)} nodes, {len(pruned_edges)} edges", LogType.SUCCESS)

    if chunk_vectors is not None and len(chunk_vectors) > 0:
        for node in pruned_nodes:
            valid_indices = [i for i in node.chunk_indices if i < len(chunk_vectors)]
            if valid_indices:
                node.centroid = chunk_vectors[valid_indices].mean(axis=0).tolist()
            else:
                node.centroid = []

    cluster_map = {}
    for node in pruned_nodes:
        cluster_map[node.label] = node.chunk_indices

    graph = GraphData(
        filename=filename,
        nodes=pruned_nodes,
        edges=pruned_edges,
        chunk_count=n_all,
        total_entities_extracted=len(registry.id_to_node),
        total_relations_extracted=len(relation_acc.canonical_relations),
        cluster_map=cluster_map,
        config_snapshot=config.to_dict(),
        processing_mode=config.processing_mode,
    )

    darwin.log(
        f"Graph complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges",
        LogType.SUCCESS,
    )

    if graph.nodes:
        preview = ", ".join(n.label for n in graph.nodes[:5])
        darwin.log(f"Top entities: {preview}{'...' if len(graph.nodes) > 5 else ''}", LogType.INFO)

    return graph


def run_graph_extraction_pipeline(
    filename: str,
    store: DarwinStore,
    config: GraphExtractionConfig = None,
) -> Optional[GraphData]:
    if config is None:
        config = GraphExtractionConfig()

    if not config.rerun and store.is_fresh(filename) and store.has_graph(filename):
        try:
            darwin.log(f"Skipping graph: fresh exists for {filename}", LogType.INFO)
            return store.load_graph(filename)
        except Exception as e:
            darwin.log(f"Cached graph corrupted ({e}) — re-extracting...", LogType.WARNING)

    if store.has_chunks(filename):
        try:
            chunks = store.load_chunks(filename)
            if not chunks:
                darwin.log(f"No chunks found for {filename}", LogType.ERROR)
                return None
        except Exception as e:
            darwin.log(f"Failed to load chunks for {filename}: {e}", LogType.ERROR)
            return None
    else:
        darwin.log(f"No chunks found for {filename}. Run chunking first.", LogType.ERROR)
        return None

    vectors = None
    if store.has_embeddings(filename):
        try:
            vectors, _ = store.load_embeddings(filename)
        except Exception as e:
            darwin.log(f"Embeddings unavailable ({e}), proceeding without", LogType.WARNING)

    graph = extract_graph_from_chunks(
        chunks=chunks,
        filename=filename,
        config=config,
        chunk_vectors=vectors,
    )

    store.save_graph(filename, graph)
    darwin.log(f"Graph saved for {filename}", LogType.SUCCESS)

    if config.visualize and graph.nodes:
        try:
            from src.features.Knowledge.Graph.GraphVisualizer import render_concept_graph
            html = render_concept_graph(graph)
            if html:
                store.save_graph_html(filename, html)
                darwin.log(f"Visualization saved for {filename}", LogType.SUCCESS)
        except Exception as e:
            darwin.log(f"Visualization error: {e}", LogType.WARNING)

    return graph