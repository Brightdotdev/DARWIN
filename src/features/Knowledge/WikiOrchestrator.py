"""
Darwin WikiOrchestrator — Per-File Entity Wiki & Graph
Pure math + NLP. No LLM. Logs everything via DarwinLogger.
Optimized: batch NLP, parallel post-processing, pre-compiled regexes,
auto-chunking for long documents, sentence-boundary overlap.
"""

import html
import json
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.Logger.DarwinLogger import darwin
from src.core.Models.ModelManager import modelManager
from src.core.Models.Lib.ModelRegistry import ModelCategory
from src.core.Storage.DarwinStore import DarwinStore


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-compiled regexes
# ═══════════════════════════════════════════════════════════════════════════════

_RE_WHITESPACE = re.compile(r"\s+")
_RE_TITLE = re.compile(r"^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+", re.I)
_RE_INITIAL = re.compile(r"(\b\w)\.\s+")
_RE_LAST_FIRST = re.compile(r"^([\w\s]+),\s+(\w)\.?$")
_RE_WORDS = re.compile(r"\b\w+\b")
_NUMERIC_RE = re.compile(r"^\d+([.,]\d+)?%?$")
_GARBAGE_RE = re.compile(r"^[0-9\s\W]+$")


# ═══════════════════════════════════════════════════════════════════════════════
# Tunable Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractionConfig:
    """All tweakable parameters. Sensible defaults work across any domain."""
    similarity_threshold: float = 0.82
    window_size: int = 5
    min_mentions: int = 1
    min_entropy: float = 0.3
    max_entity_len: int = 60
    pmi_threshold: Optional[float] = None
    max_nodes_html: int = 150
    html_title: str = "Knowledge Graph"
    skip_labels: Set[str] = field(default_factory=lambda: {
        "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY", "DATE", "TIME"
    })
    adaptive_floor: bool = True
    max_workers: int = 1
    batch_size: int = 8
    chunk_size: int = 32
    # NEW: long-document chunking
    max_doc_chars: int = 900_000          # ~90% of spaCy 1M default, with margin
    sentence_overlap: int = 2             # sentences to overlap between chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Internal Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _ExtractedEntity:
    text: str
    label: str
    start: int
    end: int
    source: str = "ner"
    sentence: str = ""


@dataclass
class _WikiEntry:
    canonical_name: str
    aliases: Set[str] = field(default_factory=set)
    types: Set[str] = field(default_factory=set)
    mention_count: int = 0
    contexts: List[str] = field(default_factory=list)
    first_seen: str = ""
    _norm_aliases: Set[str] = field(default_factory=set, repr=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Stopword / garbage filter
# ═══════════════════════════════════════════════════════════════════════════════

_STOPWORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their",
    "what", "which", "who", "whom", "whose",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could", "must", "ought",
    "and", "but", "or", "yet", "so", "if", "because", "as", "than", "though", "although", "while",
    "where", "when", "after", "before", "until",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "into", "through", "during",
    "above", "below", "between", "under", "again", "further", "then", "once", "here", "there",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just",
    "way", "part", "time", "kind", "sort", "type", "thing", "things", "stuff", "lot", "lots", "bit",
    "number", "numbers", "amount", "amounts", "one", "two", "first", "second", "last", "next", "many", "much",
    "well", "also", "still", "even", "back", "place", "point", "case", "fact",
    "group", "area", "system", "company", "government", "hand", "life", "world", "year", "day",
})


# ═══════════════════════════════════════════════════════════════════════════════
# Canonicalizer
# ═══════════════════════════════════════════════════════════════════════════════

class _EntityCanonicalizer:
    def __init__(self, threshold: float = 0.82):
        self.threshold = threshold
        self._entries: Dict[str, _WikiEntry] = {}

    def _normalize(self, text: str) -> str:
        text = text.strip()
        text = _RE_WHITESPACE.sub(" ", text)
        text = _RE_TITLE.sub("", text)
        text = _RE_INITIAL.sub(r"\1 ", text)
        text = _RE_LAST_FIRST.sub(r"\2 \1", text)
        text = text.rstrip('.,;:!?\'"()[]{}')
        return text.lower().strip()

    def _similarity(self, a: str, b: str) -> float:
        na, nb = self._normalize(a), self._normalize(b)
        if na == nb:
            return 1.0
        lev = SequenceMatcher(None, na, nb).ratio()
        def tri(s):
            return {s[i:i + 3] for i in range(max(0, len(s) - 2))}
        ta, tb = tri(na), tri(nb)
        if not ta or not tb:
            return lev
        jac = len(ta & tb) / len(ta | tb)
        return max(lev, jac)

    def resolve(self, mention: str, entity_type: str) -> str:
        norm = self._normalize(mention)
        for canonical, entry in self._entries.items():
            if self._normalize(canonical) == norm:
                entry.aliases.add(mention)
                entry.types.add(entity_type)
                entry._norm_aliases.add(norm)
                return canonical
            if norm in entry._norm_aliases:
                entry.aliases.add(mention)
                entry.types.add(entity_type)
                entry._norm_aliases.add(norm)
                return canonical
        best_match, best_score = None, 0.0
        for canonical in self._entries:
            score = self._similarity(mention, canonical)
            if score > best_score:
                best_score = score
                best_match = canonical
        if best_match and best_score >= self.threshold:
            entry = self._entries[best_match]
            entry.aliases.add(mention)
            entry.types.add(entity_type)
            entry._norm_aliases.add(norm)
            return best_match
        entry = _WikiEntry(
            canonical_name=mention,
            aliases={mention},
            types={entity_type},
            first_seen=datetime.utcnow().isoformat(),
            _norm_aliases={norm},
        )
        self._entries[mention] = entry
        return mention

    def update(self, canonical: str, context: str):
        entry = self._entries[canonical]
        entry.mention_count += 1
        ctx = context[:300]
        if ctx not in entry.contexts:
            entry.contexts.append(ctx)
            if len(entry.contexts) > 15:
                entry.contexts.pop(0)

    def to_dict(self) -> dict:
        return {
            name: {
                "canonical_name": e.canonical_name,
                "aliases": sorted(e.aliases),
                "types": sorted(e.types),
                "mention_count": e.mention_count,
                "contexts": e.contexts,
                "first_seen": e.first_seen,
            }
            for name, e in self._entries.items()
        }

    @property
    def entries(self) -> Dict[str, _WikiEntry]:
        return dict(self._entries)


# ═══════════════════════════════════════════════════════════════════════════════
# Three-Tier Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class _EntityExtractor:
    def __init__(self, config: ExtractionConfig):
        self.cfg = config

    def extract(self, doc) -> List[_ExtractedEntity]:
        candidates = []
        skip = self.cfg.skip_labels
        max_len = self.cfg.max_entity_len
        for ent in doc.ents:
            if ent.label_ in skip:
                continue
            if len(ent.text) > max_len:
                continue
            sent = ent.sent.text[:400] if ent.sent else ""
            candidates.append(_ExtractedEntity(
                text=ent.text.strip(), label=ent.label_,
                start=ent.start_char, end=ent.end_char,
                source="ner", sentence=sent,
            ))
        if doc.has_annotation("DEP"):
            for chunk in doc.noun_chunks:
                overlap = any(
                    ent.start <= chunk.start < ent.end or ent.start < chunk.end <= ent.end
                    for ent in doc.ents
                )
                if overlap:
                    continue
                words = [t.text for t in chunk if t.pos_ not in ("DET", "PUNCT")]
                if not words:
                    continue
                chunk_text = " ".join(words)
                if len(chunk_text) > max_len:
                    continue
                if all(w.lower() in _STOPWORDS for w in words):
                    continue
                if _NUMERIC_RE.match(chunk_text) or _GARBAGE_RE.match(chunk_text):
                    continue
                sent = chunk.sent.text[:400] if chunk.sent else ""
                candidates.append(_ExtractedEntity(
                    text=chunk_text, label="CONCEPT",
                    start=chunk.start_char, end=chunk.end_char,
                    source="noun_chunk", sentence=sent,
                ))
        return self._filter(candidates)

    def _filter(self, candidates: List[_ExtractedEntity]) -> List[_ExtractedEntity]:
        if not candidates:
            return []
        counts = Counter(c.text for c in candidates)
        floor = self.cfg.min_mentions
        if self.cfg.adaptive_floor and len(counts) > 10:
            sorted_counts = sorted(counts.values(), reverse=True)
            idx = max(0, int(len(sorted_counts) * 0.7) - 1)
            adaptive = sorted_counts[min(idx, len(sorted_counts) - 1)]
            floor = max(floor, min(adaptive, max(floor * 2, 3)))
        contexts = defaultdict(list)
        for c in candidates:
            contexts[c.text].append(c.sentence)
        def word_diversity(texts: List[str]) -> float:
            if not texts:
                return 0.0
            words = []
            for t in texts:
                words.extend(_RE_WORDS.findall(t.lower()))
            if len(words) < 5:
                return 1.0
            unique = len(set(words))
            return unique / len(words)
        filtered = []
        for c in candidates:
            if counts[c.text] < floor:
                continue
            if self.cfg.min_entropy > 0 and word_diversity(contexts[c.text]) < self.cfg.min_entropy:
                continue
            filtered.append(c)
        return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Builder  — SPAN-AWARE
# ═══════════════════════════════════════════════════════════════════════════════

class _GraphBuilder:
    def __init__(self, config: ExtractionConfig):
        self.cfg = config
        self._cooccur: Dict[Tuple[str, str], int] = defaultdict(int)
        self._dep_edges: List[dict] = []
        self._window_counts: Dict[str, int] = defaultdict(int)
        self._total_windows: int = 0

    def process(self, doc, canonical_map: Dict[str, str], entity_spans: List[Tuple[str, int, int]]):
        sentences = list(doc.sents)
        if not sentences:
            return
        sent_entities: List[Set[str]] = []
        for sent in sentences:
            sent_start, sent_end = sent.start_char, sent.end_char
            found: Set[str] = set()
            for canonical, estart, eend in entity_spans:
                if estart < sent_end and eend > sent_start:
                    found.add(canonical)
            sent_entities.append(found)
        for i in range(len(sentences)):
            self._total_windows += 1
            found: Set[str] = set()
            for j in range(i, min(i + self.cfg.window_size, len(sentences))):
                found |= sent_entities[j]
            for ent in found:
                self._window_counts[ent] += 1
            for a in found:
                for b in found:
                    if a >= b:
                        continue
                    key = tuple(sorted((a, b)))
                    self._cooccur[key] += 1
        if doc.has_annotation("DEP"):
            token_to_canonical: Dict[int, str] = {}
            for token in doc:
                t_start, t_end = token.idx, token.idx + len(token.text)
                best_can, best_len = None, float("inf")
                for canonical, estart, eend in entity_spans:
                    if estart <= t_start and t_end <= eend:
                        span_len = eend - estart
                        if span_len < best_len:
                            best_len = span_len
                            best_can = canonical
                if best_can:
                    token_to_canonical[token.i] = best_can
            for token in doc:
                if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                    subj_can = token_to_canonical.get(token.i)
                    if not subj_can:
                        continue
                    for child in token.head.children:
                        if child.dep_ in ("dobj", "pobj", "attr", "iobj"):
                            obj_can = token_to_canonical.get(child.i)
                            if obj_can and obj_can != subj_can:
                                self._dep_edges.append({
                                    "source": subj_can, "target": obj_can,
                                    "label": token.head.lemma_,
                                    "type": "dependency", "weight": 1.0,
                                })

    def build(self, canonicalizer: _EntityCanonicalizer) -> Tuple[List[dict], List[dict]]:
        entries = canonicalizer.entries
        if not entries:
            return [], []
        def pick_type(entry: _WikiEntry) -> str:
            if not entry.types:
                return "UNKNOWN"
            type_counts = Counter(entry.types)
            max_count = max(type_counts.values())
            candidates = [t for t, c in type_counts.items() if c == max_count]
            return sorted(candidates)[0]
        nodes = []
        for name, entry in entries.items():
            nodes.append({
                "id": name, "label": name,
                "type": pick_type(entry),
                "mentions": entry.mention_count,
                "aliases": sorted(entry.aliases),
            })
        total_windows = max(self._total_windows, 1)
        edges = []
        for (a, b), count in self._cooccur.items():
            if a not in self._window_counts or b not in self._window_counts:
                continue
            p_a = self._window_counts[a] / total_windows
            p_b = self._window_counts[b] / total_windows
            p_ab = count / total_windows
            if p_ab > 0 and p_a > 0 and p_b > 0:
                pmi = math.log2(p_ab / (p_a * p_b))
            else:
                pmi = 0.0
            edges.append({
                "source": a, "target": b,
                "weight": round(pmi, 4), "count": count,
                "type": "cooccurrence", "label": "",
            })
        if edges:
            if self.cfg.pmi_threshold is None:
                pmi_vals = [e["weight"] for e in edges]
                mean_pmi = sum(pmi_vals) / len(pmi_vals)
                std_pmi = math.sqrt(sum((x - mean_pmi) ** 2 for x in pmi_vals) / len(pmi_vals))
                threshold = mean_pmi + 0.5 * std_pmi
            else:
                threshold = self.cfg.pmi_threshold
            edges = [e for e in edges if e["weight"] >= threshold]
        pair_edges: Dict[Tuple[str, str], dict] = {}
        for e in edges:
            key = tuple(sorted((e["source"], e["target"])))
            existing = pair_edges.get(key)
            if existing is None or (e["type"] == "dependency" and existing["type"] == "cooccurrence"):
                pair_edges[key] = e
        for e in self._dep_edges:
            key = tuple(sorted((e["source"], e["target"])))
            existing = pair_edges.get(key)
            if existing is None or existing["type"] == "cooccurrence":
                pair_edges[key] = e
        deduped = list(pair_edges.values())
        return nodes, deduped


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Visualizer  — XSS-SAFE
# ═══════════════════════════════════════════════════════════════════════════════

class _GraphVisualizer:
    TYPE_COLORS = {
        "PERSON": "#e74c3c", "ORG": "#3498db", "GPE": "#2ecc71",
        "LOC": "#f39c12", "CONCEPT": "#9b59b6", "PRODUCT": "#1abc9c",
        "EVENT": "#e67e22", "WORK_OF_ART": "#34495e", "LAW": "#16a085",
        "LANGUAGE": "#d35400", "UNKNOWN": "#95a5a6",
    }

    @classmethod
    def generate(cls, nodes: List[dict], edges: List[dict], title: str = "Knowledge Graph", max_nodes: int = 150) -> str:
        if len(nodes) > max_nodes:
            degree = Counter()
            for e in edges:
                degree[e["source"]] += 1
                degree[e["target"]] += 1
            top = sorted(degree, key=degree.get, reverse=True)[:max_nodes]
            top_set = set(top)
            nodes = [n for n in nodes if n["id"] in top_set]
            edges = [e for e in edges if e["source"] in top_set and e["target"] in top_set]
        max_mentions = max((n.get("mentions", 1) for n in nodes), default=1)
        cy_nodes = []
        for n in nodes:
            t = n.get("type", "UNKNOWN")
            color = cls.TYPE_COLORS.get(t, "#95a5a6")
            size = 20 + 40 * math.log2(n.get("mentions", 1) + 1) / math.log2(max_mentions + 1)
            cy_nodes.append({
                "data": {
                    "id": n["id"], "label": n["label"], "type": t,
                    "mentions": n.get("mentions", 1), "aliases": n.get("aliases", []),
                },
                "style": {"background-color": color, "width": size, "height": size}
            })
        cy_edges = []
        for e in edges:
            cy_edges.append({
                "data": {
                    "source": e["source"], "target": e["target"],
                    "weight": e.get("weight", 1), "label": e.get("label", ""),
                    "type": e.get("type", "cooccurrence"),
                }
            })
        legend_html = "".join(
            f'<div class="legend-item"><span class="type" style="background:{html.escape(c)}"></span>{html.escape(t)}</div>'
            for t, c in cls.TYPE_COLORS.items()
        )
        elements_json = json.dumps(cy_nodes + cy_edges, ensure_ascii=True)
        elements_json = elements_json.replace("<", "\u003c").replace(">", "\u003e")
        safe_title = html.escape(title)
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{safe_title}</title>
<script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
<style>
body{{margin:0;font-family:system-ui,sans-serif;background:#1a1a2e;color:#eee}}
#cy{{width:100vw;height:100vh}}
#panel{{position:absolute;top:10px;right:10px;width:280px;background:rgba(0,0,0,0.8);
padding:15px;border-radius:8px;font-size:13px;max-height:80vh;overflow-y:auto}}
#panel h3{{margin:0 0 10px;color:#fff}}
#panel .stat{{margin:5px 0;color:#aaa}}
#panel .type{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px}}
.legend{{margin-top:10px}}
.legend-item{{display:flex;align-items:center;margin:3px 0}}
</style>
</head>
<body>
<div id="cy"></div>
<div id="panel">
<h3>{safe_title}</h3>
<div class="stat">Nodes: {len(nodes)}</div>
<div class="stat">Edges: {len(edges)}</div>
<div class="legend">{legend_html}</div>
<div id="info" style="margin-top:15px;"></div>
</div>
<script>
var cy=cytoscape({{
  container:document.getElementById('cy'),
  elements:{elements_json},
  style:[
    {{selector:'node',style:{{'label':'data(label)','color':'#fff','font-size':'10px',
      'text-valign':'center','text-halign':'center','border-width':2,'border-color':'#fff'}}}},
    {{selector:'edge',style:{{'width':2,'line-color':'#888','target-arrow-color':'#888',
      'target-arrow-shape':'triangle','curve-style':'bezier','label':'data(label)',
      'color':'#ccc','font-size':'9px'}}}},
    {{selector:'edge[type="cooccurrence"]',style:{{'line-style':'dotted','target-arrow-shape':'none'}}}}
  ],
  layout:{{name:'cose',padding:10,nodeRepulsion:400000,idealEdgeLength:100,animate:true}}
}});
cy.on('tap','node',function(evt){{
  var n=evt.target;
  var info='<strong>'+n.data('label')+'</strong><br>Type: '+n.data('type')+
           '<br>Mentions: '+n.data('mentions')+
           '<br>Aliases: '+(n.data('aliases')||[]).join(', ');
  document.getElementById('info').innerHTML=info;
}});
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# Long-document chunker (sentence-boundary aware)
# ═══════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, max_chars: int, overlap_sentences: int) -> List[str]:
    """
    Split text into chunks at sentence boundaries, with optional overlap.
    Uses a simple regex sentence splitter as a fast fallback.
    Preserves sentence integrity — never cuts mid-sentence.
    """
    if len(text) <= max_chars:
        return [text]
    # Fast sentence split: split on .!? followed by whitespace and uppercase
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        # Fallback: hard split at max_chars if no sentence boundaries found
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent) + 1  # +1 for separator
        if current_len + sent_len > max_chars and current:
            chunks.append(" ".join(current))
            # Overlap: keep last N sentences for context continuity
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = overlap + [sent]
            current_len = sum(len(s) + 1 for s in current)
        else:
            current.append(sent)
            current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class WikiOrchestrator:
    VALID_STAGES = {"wiki", "graph", "html"}

    def __init__(self, store: DarwinStore, model_manager=None):
        self.store = store
        self.model_manager = model_manager or modelManager
        self._nlp_engine = None
        self._nlp_available: Optional[bool] = None
        self._lock = threading.Lock()
        self._store_lock = threading.Lock()

    def _ensure_nlp(self) -> bool:
        if self._nlp_available is True:
            return True
        with self._lock:
            if self._nlp_available is True:
                return True
            try:
                engine = self.model_manager.get(ModelCategory.NLP)
                if engine is None:
                    engine = self.model_manager.ensure(ModelCategory.NLP, auto_download=False)
                if engine is None:
                    darwin.warning("No NLP model registered in ModelManager. Knowledge extraction disabled.")
                    self._nlp_available = False
                    return False
                self._nlp_engine = engine
                self._nlp_available = True
                model_name = getattr(engine, "model_name", "unknown")
                darwin.success(f"NLP model ready: {model_name}")
                return True
            except Exception as e:
                darwin.error(f"NLP initialization failed: {e}")
                self._nlp_available = False
                return False

    def reset_nlp(self):
        with self._lock:
            self._nlp_available = None
            self._nlp_engine = None

    def _get_doc(self, text: str):
        if self._nlp_engine is None:
            darwin.error("NLP not loaded")
            return None
        return next(self._nlp_engine.pipe([text], batch_size=1))

    def _process_doc(self, filename: str, doc, stages: Set[str], config: ExtractionConfig) -> dict:
        result = {"ok": False, "stages_completed": [], "error": None}
        extractor = _EntityExtractor(config)
        candidates = extractor.extract(doc)
        canonicalizer = _EntityCanonicalizer(threshold=config.similarity_threshold)
        raw_to_canonical: Dict[str, str] = {}
        entity_spans: List[Tuple[str, int, int]] = []
        for c in candidates:
            canonical = canonicalizer.resolve(c.text, c.label)
            raw_to_canonical[c.text] = canonical
            entity_spans.append((canonical, c.start, c.end))
            canonicalizer.update(canonical, c.sentence)
        if "wiki" in stages:
            try:
                wiki_data = {
                    "filename": filename,
                    "generated_at": datetime.utcnow().isoformat(),
                    "entities": canonicalizer.to_dict(),
                    "stats": {
                        "total_mentions": sum(e.mention_count for e in canonicalizer.entries.values()),
                        "unique_entities": len(canonicalizer.entries),
                    },
                }
                with self._store_lock:
                    self.store.save_wiki(filename, wiki_data)
                darwin.info(f"[{filename}] Wiki saved: {len(canonicalizer.entries)} entities, {wiki_data['stats']['total_mentions']} total mentions")
                result["stages_completed"].append("wiki")
            except Exception as e:
                darwin.error(f"[{filename}] Wiki build failed: {e}")
                result["error"] = str(e)
                return result
        if "graph" in stages:
            try:
                builder = _GraphBuilder(config)
                builder.process(doc, raw_to_canonical, entity_spans)
                nodes, edges = builder.build(canonicalizer)
                with self._store_lock:
                    self.store.save_graph(filename, nodes, edges)
                dep_count = sum(1 for e in edges if e.get("type") == "dependency")
                cooc_count = sum(1 for e in edges if e.get("type") == "cooccurrence")
                darwin.info(f"[{filename}] Graph saved: {len(nodes)} nodes, {len(edges)} edges ({dep_count} dependency, {cooc_count} co-occurrence)")
                result["stages_completed"].append("graph")
            except Exception as e:
                darwin.error(f"[{filename}] Graph build failed: {e}")
                result["error"] = str(e)
                return result
        if "html" in stages:
            try:
                if "graph" in result["stages_completed"]:
                    pass
                else:
                    with self._store_lock:
                        graph_data = self.store.load_graph(filename)
                    nodes = graph_data.get("nodes", [])
                    edges = graph_data.get("edges", [])
                if nodes:
                    html_title = config.html_title or f"KG: {filename}"
                    html = _GraphVisualizer.generate(
                        nodes, edges,
                        title=html_title,
                        max_nodes=config.max_nodes_html,
                    )
                    with self._store_lock:
                        self.store.save_graph_html(filename, html)
                    darwin.info(f"[{filename}] HTML visualization saved")
                    result["stages_completed"].append("html")
                else:
                    darwin.warning(f"[{filename}] No graph nodes — skipping HTML")
            except Exception as e:
                darwin.warning(f"[{filename}] HTML generation failed: {e}")
        result["ok"] = True
        return result

    def _process_single(self, filename: str, text: str, stages: Set[str], config: ExtractionConfig) -> dict:
        """Process a single text blob, auto-chunking if too long for spaCy."""
        if len(text) <= config.max_doc_chars:
            try:
                doc = self._get_doc(text)
            except Exception as e:
                darwin.error(f"[{filename}] NLP parsing failed: {e}")
                return {"ok": False, "stages_completed": [], "error": str(e)}
            return self._process_doc(filename, doc, stages, config)
        # Long document: chunk at sentence boundaries, process each, merge results
        darwin.info(f"[{filename}] Document length {len(text)} exceeds {config.max_doc_chars} chars — chunking into sentence-boundary chunks")
        chunks = _chunk_text(text, config.max_doc_chars, config.sentence_overlap)
        darwin.info(f"[{filename}] Split into {len(chunks)} chunks")
        # Process each chunk independently
        all_candidates: List[_ExtractedEntity] = []
        all_dep_edges: List[dict] = []
        all_cooccur: Dict[Tuple[str, str], int] = defaultdict(int)
        all_window_counts: Dict[str, int] = defaultdict(int)
        total_windows = 0
        canonicalizer = _EntityCanonicalizer(threshold=config.similarity_threshold)
        for idx, chunk_text in enumerate(chunks):
            try:
                doc = self._get_doc(chunk_text)
            except Exception as e:
                darwin.error(f"[{filename}] Chunk {idx + 1}/{len(chunks)} NLP parsing failed: {e}")
                continue
            extractor = _EntityExtractor(config)
            candidates = extractor.extract(doc)
            for c in candidates:
                canonical = canonicalizer.resolve(c.text, c.label)
                canonicalizer.update(canonical, c.sentence)
            all_candidates.extend(candidates)
            # Build graph per-chunk and merge
            raw_to_canonical = {c.text: canonicalizer.resolve(c.text, c.label) for c in candidates}
            entity_spans = [(canonicalizer.resolve(c.text, c.label), c.start, c.end) for c in candidates]
            builder = _GraphBuilder(config)
            builder.process(doc, raw_to_canonical, entity_spans)
            for key, count in builder._cooccur.items():
                all_cooccur[key] += count
            for ent, count in builder._window_counts.items():
                all_window_counts[ent] += count
            total_windows += builder._total_windows
            all_dep_edges.extend(builder._dep_edges)
        # Build merged wiki
        if "wiki" in stages:
            try:
                wiki_data = {
                    "filename": filename,
                    "generated_at": datetime.utcnow().isoformat(),
                    "entities": canonicalizer.to_dict(),
                    "stats": {
                        "total_mentions": sum(e.mention_count for e in canonicalizer.entries.values()),
                        "unique_entities": len(canonicalizer.entries),
                    },
                }
                with self._store_lock:
                    self.store.save_wiki(filename, wiki_data)
                darwin.info(f"[{filename}] Wiki saved: {len(canonicalizer.entries)} entities, {wiki_data['stats']['total_mentions']} total mentions")
            except Exception as e:
                darwin.error(f"[{filename}] Wiki build failed: {e}")
                return {"ok": False, "stages_completed": [], "error": str(e)}
        # Build merged graph
        if "graph" in stages:
            try:
                entries = canonicalizer.entries
                if not entries:
                    nodes, edges = [], []
                else:
                    def pick_type(entry: _WikiEntry) -> str:
                        if not entry.types:
                            return "UNKNOWN"
                        type_counts = Counter(entry.types)
                        max_count = max(type_counts.values())
                        candidates = [t for t, c in type_counts.items() if c == max_count]
                        return sorted(candidates)[0]
                    nodes = []
                    for name, entry in entries.items():
                        nodes.append({
                            "id": name, "label": name,
                            "type": pick_type(entry),
                            "mentions": entry.mention_count,
                            "aliases": sorted(entry.aliases),
                        })
                    total_w = max(total_windows, 1)
                    edges = []
                    for (a, b), count in all_cooccur.items():
                        if a not in all_window_counts or b not in all_window_counts:
                            continue
                        p_a = all_window_counts[a] / total_w
                        p_b = all_window_counts[b] / total_w
                        p_ab = count / total_w
                        if p_ab > 0 and p_a > 0 and p_b > 0:
                            pmi = math.log2(p_ab / (p_a * p_b))
                        else:
                            pmi = 0.0
                        edges.append({
                            "source": a, "target": b,
                            "weight": round(pmi, 4), "count": count,
                            "type": "cooccurrence", "label": "",
                        })
                    if edges:
                        if config.pmi_threshold is None:
                            pmi_vals = [e["weight"] for e in edges]
                            mean_pmi = sum(pmi_vals) / len(pmi_vals)
                            std_pmi = math.sqrt(sum((x - mean_pmi) ** 2 for x in pmi_vals) / len(pmi_vals))
                            threshold = mean_pmi + 0.5 * std_pmi
                        else:
                            threshold = config.pmi_threshold
                        edges = [e for e in edges if e["weight"] >= threshold]
                    pair_edges: Dict[Tuple[str, str], dict] = {}
                    for e in edges:
                        key = tuple(sorted((e["source"], e["target"])))
                        existing = pair_edges.get(key)
                        if existing is None or (e["type"] == "dependency" and existing["type"] == "cooccurrence"):
                            pair_edges[key] = e
                    for e in all_dep_edges:
                        key = tuple(sorted((e["source"], e["target"])))
                        existing = pair_edges.get(key)
                        if existing is None or existing["type"] == "cooccurrence":
                            pair_edges[key] = e
                    edges = list(pair_edges.values())
                with self._store_lock:
                    self.store.save_graph(filename, nodes, edges)
                dep_count = sum(1 for e in edges if e.get("type") == "dependency")
                cooc_count = sum(1 for e in edges if e.get("type") == "cooccurrence")
                darwin.info(f"[{filename}] Graph saved: {len(nodes)} nodes, {len(edges)} edges ({dep_count} dependency, {cooc_count} co-occurrence)")
            except Exception as e:
                darwin.error(f"[{filename}] Graph build failed: {e}")
                return {"ok": False, "stages_completed": [], "error": str(e)}
        # HTML
        if "html" in stages:
            try:
                with self._store_lock:
                    graph_data = self.store.load_graph(filename)
                nodes = graph_data.get("nodes", [])
                edges = graph_data.get("edges", [])
                if nodes:
                    html_title = config.html_title or f"KG: {filename}"
                    html = _GraphVisualizer.generate(
                        nodes, edges,
                        title=html_title,
                        max_nodes=config.max_nodes_html,
                    )
                    with self._store_lock:
                        self.store.save_graph_html(filename, html)
                    darwin.info(f"[{filename}] HTML visualization saved")
                else:
                    darwin.warning(f"[{filename}] No graph nodes — skipping HTML")
            except Exception as e:
                darwin.warning(f"[{filename}] HTML generation failed: {e}")
        return {"ok": True, "stages_completed": list({"wiki", "graph", "html"} & stages), "error": None}

    def handle_knowledge(
        self,
        filename: str,
        *,
        rerun: bool = False,
        stages: Optional[List[str]] = None,
        similarity_threshold: float = 0.82,
        window_size: int = 5,
        min_mentions: int = 1,
        min_entropy: float = 0.3,
        max_entity_len: int = 60,
        pmi_threshold: Optional[float] = None,
        max_nodes_html: int = 150,
        html_title: str = "Knowledge Graph",
        skip_labels: Optional[Set[str]] = None,
        adaptive_floor: bool = True,
        max_doc_chars: int = 900_000,
        sentence_overlap: int = 2,
    ) -> dict:
        if stages is None:
            stages = {"wiki", "graph", "html"}
        stages = {s.lower() for s in stages} & self.VALID_STAGES
        if not stages:
            darwin.warning(f"[{filename}] No valid stages requested")
            return {"ok": False, "stages_completed": [], "error": "No valid stages requested"}
        if not self._ensure_nlp():
            darwin.error(f"[{filename}] Cannot process — NLP engine unavailable")
            return {"ok": False, "stages_completed": [], "error": "NLP engine unavailable"}
        text = self.store.load_markdown(filename)
        if text is None:
            darwin.error(f"[{filename}] Not registered in DarwinStore — cannot load markdown")
            return {"ok": False, "stages_completed": [], "error": "File not registered in store"}
        if not text.strip() or len(text.strip()) < 20:
            darwin.warning(f"[{filename}] Text too short ({len(text.strip())} chars) — skipping")
            return {"ok": False, "stages_completed": [], "error": "Text too short"}
        if not rerun:
            missing = []
            if "wiki" in stages and not self.store.has_wiki(filename):
                missing.append("wiki")
            if "graph" in stages and not self.store.has_graph(filename):
                missing.append("graph")
            if "html" in stages and not self.store.has_graph_html(filename):
                missing.append("html")
            if not missing:
                darwin.info(f"[{filename}] Skipped — artifacts already exist (rerun=False)")
                return {"ok": True, "stages_completed": [], "error": None}
        config = ExtractionConfig(
            similarity_threshold=similarity_threshold,
            window_size=window_size,
            min_mentions=min_mentions,
            min_entropy=min_entropy,
            max_entity_len=max_entity_len,
            pmi_threshold=pmi_threshold,
            max_nodes_html=max_nodes_html,
            html_title=html_title,
            skip_labels=skip_labels if skip_labels is not None else ExtractionConfig().skip_labels,
            adaptive_floor=adaptive_floor,
            max_doc_chars=max_doc_chars,
            sentence_overlap=sentence_overlap,
        )
        darwin.info(f"[{filename}] Starting knowledge extraction — stages: {sorted(stages)}")
        result = self._process_single(filename, text, stages, config)
        if result["ok"]:
            darwin.success(f"[{filename}] Knowledge extraction complete")
        return result

    def handle_knowledge_batch(
        self,
        filenames: List[str],
        *,
        rerun: bool = False,
        stages: Optional[List[str]] = None,
        max_workers: int = 1,
        **kwargs,
    ) -> dict:
        if not filenames:
            darwin.warning("Batch knowledge called with empty file list")
            return {"processed": 0, "skipped": 0, "failed": 0, "results": []}
        if stages is None:
            stages = {"wiki", "graph", "html"}
        stages = {s.lower() for s in stages} & self.VALID_STAGES
        if not self._ensure_nlp():
            darwin.error("Batch aborted — NLP engine unavailable")
            return {"processed": 0, "skipped": 0, "failed": 0, "results": []}
        darwin.info(f"Batch knowledge starting: {len(filenames)} file(s) — stages: {sorted(stages)}")
        config = ExtractionConfig(
            similarity_threshold=kwargs.get("similarity_threshold", 0.82),
            window_size=kwargs.get("window_size", 5),
            min_mentions=kwargs.get("min_mentions", 1),
            min_entropy=kwargs.get("min_entropy", 0.3),
            max_entity_len=kwargs.get("max_entity_len", 60),
            pmi_threshold=kwargs.get("pmi_threshold", None),
            max_nodes_html=kwargs.get("max_nodes_html", 150),
            html_title=kwargs.get("html_title", "Knowledge Graph"),
            skip_labels=kwargs.get("skip_labels") if kwargs.get("skip_labels") is not None else ExtractionConfig().skip_labels,
            adaptive_floor=kwargs.get("adaptive_floor", True),
            max_workers=max_workers,
            batch_size=kwargs.get("batch_size", 8),
            chunk_size=kwargs.get("chunk_size", 32),
            max_doc_chars=kwargs.get("max_doc_chars", 900_000),
            sentence_overlap=kwargs.get("sentence_overlap", 2),
        )
        todo = []
        for filename in filenames:
            text = self.store.load_markdown(filename)
            if text is None:
                darwin.error(f"[{filename}] Not registered — skipped")
                continue
            if not text.strip() or len(text.strip()) < 20:
                darwin.warning(f"[{filename}] Text too short — skipped")
                continue
            if not rerun:
                missing = []
                if "wiki" in stages and not self.store.has_wiki(filename):
                    missing.append("wiki")
                if "graph" in stages and not self.store.has_graph(filename):
                    missing.append("graph")
                if "html" in stages and not self.store.has_graph_html(filename):
                    missing.append("html")
                if not missing:
                    darwin.info(f"[{filename}] Skipped — artifacts exist")
                    continue
            todo.append((filename, text))
        if not todo:
            darwin.info("Batch complete: all files skipped or invalid")
            return {"processed": 0, "skipped": len(filenames), "failed": 0, "results": []}
        processed = 0
        skipped = len(filenames) - len(todo)
        failed = 0
        results = []
        # Expand long documents into (filename, chunk_text, is_chunk, total_chunks) tuples
        expanded = []
        for filename, text in todo:
            if len(text) <= config.max_doc_chars:
                expanded.append((filename, text, False, 1))
            else:
                chunks = _chunk_text(text, config.max_doc_chars, config.sentence_overlap)
                darwin.info(f"[{filename}] Split into {len(chunks)} chunks for batch processing")
                for chunk in chunks:
                    expanded.append((filename, chunk, True, len(chunks)))
        # Batch NLP: process all texts through pipe()
        all_texts = [t for _, t, _, _ in expanded]
        try:
            docs = list(self._nlp_engine.pipe(all_texts, batch_size=config.batch_size))
        except Exception as e:
            darwin.error(f"Batch NLP pipe failed: {e}")
            for filename, _, _, _ in expanded:
                failed += 1
                results.append({"filename": filename, "ok": False, "error": f"NLP pipe failed: {e}"})
            return {"processed": 0, "skipped": skipped, "failed": failed, "results": results}
        # Group docs back by filename
        from collections import defaultdict as _dd
        file_docs: Dict[str, List] = _dd(list)
        for (filename, _, is_chunk, total), doc in zip(expanded, docs):
            file_docs[filename].append((doc, is_chunk, total))
        # Process each file: single-chunk files go straight through, multi-chunk files get merged
        for filename, doc_list in file_docs.items():
            if len(doc_list) == 1:
                doc, _, _ = doc_list[0]
                result = self._process_doc(filename, doc, stages, config)
                results.append({"filename": filename, **result})
                if result["ok"]:
                    processed += 1
                else:
                    failed += 1
            else:
                # Multi-chunk: merge results from all chunks
                all_candidates: List[_ExtractedEntity] = []
                all_dep_edges: List[dict] = []
                all_cooccur: Dict[Tuple[str, str], int] = defaultdict(int)
                all_window_counts: Dict[str, int] = defaultdict(int)
                total_windows = 0
                canonicalizer = _EntityCanonicalizer(threshold=config.similarity_threshold)
                for doc, _, _ in doc_list:
                    extractor = _EntityExtractor(config)
                    candidates = extractor.extract(doc)
                    for c in candidates:
                        canonical = canonicalizer.resolve(c.text, c.label)
                        canonicalizer.update(canonical, c.sentence)
                    all_candidates.extend(candidates)
                    raw_to_canonical = {c.text: canonicalizer.resolve(c.text, c.label) for c in candidates}
                    entity_spans = [(canonicalizer.resolve(c.text, c.label), c.start, c.end) for c in candidates]
                    builder = _GraphBuilder(config)
                    builder.process(doc, raw_to_canonical, entity_spans)
                    for key, count in builder._cooccur.items():
                        all_cooccur[key] += count
                    for ent, count in builder._window_counts.items():
                        all_window_counts[ent] += count
                    total_windows += builder._total_windows
                    all_dep_edges.extend(builder._dep_edges)
                # Build merged wiki
                if "wiki" in stages:
                    try:
                        wiki_data = {
                            "filename": filename,
                            "generated_at": datetime.utcnow().isoformat(),
                            "entities": canonicalizer.to_dict(),
                            "stats": {
                                "total_mentions": sum(e.mention_count for e in canonicalizer.entries.values()),
                                "unique_entities": len(canonicalizer.entries),
                            },
                        }
                        with self._store_lock:
                            self.store.save_wiki(filename, wiki_data)
                        darwin.info(f"[{filename}] Wiki saved: {len(canonicalizer.entries)} entities, {wiki_data['stats']['total_mentions']} total mentions")
                    except Exception as e:
                        darwin.error(f"[{filename}] Wiki build failed: {e}")
                        results.append({"filename": filename, "ok": False, "stages_completed": [], "error": str(e)})
                        failed += 1
                        continue
                # Build merged graph
                if "graph" in stages:
                    try:
                        entries = canonicalizer.entries
                        if not entries:
                            nodes, edges = [], []
                        else:
                            def pick_type(entry: _WikiEntry) -> str:
                                if not entry.types:
                                    return "UNKNOWN"
                                type_counts = Counter(entry.types)
                                max_count = max(type_counts.values())
                                candidates = [t for t, c in type_counts.items() if c == max_count]
                                return sorted(candidates)[0]
                            nodes = []
                            for name, entry in entries.items():
                                nodes.append({
                                    "id": name, "label": name,
                                    "type": pick_type(entry),
                                    "mentions": entry.mention_count,
                                    "aliases": sorted(entry.aliases),
                                })
                            total_w = max(total_windows, 1)
                            edges = []
                            for (a, b), count in all_cooccur.items():
                                if a not in all_window_counts or b not in all_window_counts:
                                    continue
                                p_a = all_window_counts[a] / total_w
                                p_b = all_window_counts[b] / total_w
                                p_ab = count / total_w
                                if p_ab > 0 and p_a > 0 and p_b > 0:
                                    pmi = math.log2(p_ab / (p_a * p_b))
                                else:
                                    pmi = 0.0
                                edges.append({
                                    "source": a, "target": b,
                                    "weight": round(pmi, 4), "count": count,
                                    "type": "cooccurrence", "label": "",
                                })
                            if edges:
                                if config.pmi_threshold is None:
                                    pmi_vals = [e["weight"] for e in edges]
                                    mean_pmi = sum(pmi_vals) / len(pmi_vals)
                                    std_pmi = math.sqrt(sum((x - mean_pmi) ** 2 for x in pmi_vals) / len(pmi_vals))
                                    threshold = mean_pmi + 0.5 * std_pmi
                                else:
                                    threshold = config.pmi_threshold
                                edges = [e for e in edges if e["weight"] >= threshold]
                            pair_edges: Dict[Tuple[str, str], dict] = {}
                            for e in edges:
                                key = tuple(sorted((e["source"], e["target"])))
                                existing = pair_edges.get(key)
                                if existing is None or (e["type"] == "dependency" and existing["type"] == "cooccurrence"):
                                    pair_edges[key] = e
                            for e in all_dep_edges:
                                key = tuple(sorted((e["source"], e["target"])))
                                existing = pair_edges.get(key)
                                if existing is None or existing["type"] == "cooccurrence":
                                    pair_edges[key] = e
                            edges = list(pair_edges.values())
                        with self._store_lock:
                            self.store.save_graph(filename, nodes, edges)
                        dep_count = sum(1 for e in edges if e.get("type") == "dependency")
                        cooc_count = sum(1 for e in edges if e.get("type") == "cooccurrence")
                        darwin.info(f"[{filename}] Graph saved: {len(nodes)} nodes, {len(edges)} edges ({dep_count} dependency, {cooc_count} co-occurrence)")
                    except Exception as e:
                        darwin.error(f"[{filename}] Graph build failed: {e}")
                        results.append({"filename": filename, "ok": False, "stages_completed": [], "error": str(e)})
                        failed += 1
                        continue
                # HTML
                if "html" in stages:
                    try:
                        with self._store_lock:
                            graph_data = self.store.load_graph(filename)
                        nodes = graph_data.get("nodes", [])
                        edges = graph_data.get("edges", [])
                        if nodes:
                            html_title = config.html_title or f"KG: {filename}"
                            html = _GraphVisualizer.generate(
                                nodes, edges,
                                title=html_title,
                                max_nodes=config.max_nodes_html,
                            )
                            with self._store_lock:
                                self.store.save_graph_html(filename, html)
                            darwin.info(f"[{filename}] HTML visualization saved")
                        else:
                            darwin.warning(f"[{filename}] No graph nodes — skipping HTML")
                    except Exception as e:
                        darwin.warning(f"[{filename}] HTML generation failed: {e}")
                results.append({"filename": filename, "ok": True, "stages_completed": list({"wiki", "graph", "html"} & stages), "error": None})
                processed += 1
        darwin.success(f"Batch complete: {processed} processed, {skipped} skipped, {failed} failed")
        return {"processed": processed, "skipped": skipped, "failed": failed, "results": results}

    def delete_knowledge(self, filename: str) -> bool:
        with self._store_lock:
            ok_wiki = self.store.delete_wiki(filename)
            ok_graph = self.store.delete_graph(filename)
            ok_html = self.store.delete_graph_html(filename)
        if ok_wiki or ok_graph or ok_html:
            darwin.info(f"[{filename}] Knowledge artifacts deleted")
            return True
        else:
            darwin.warning(f"[{filename}] No knowledge artifacts found to delete")
            return False

    def get_wiki(self, filename: str) -> dict | None:
        return self.store.load_wiki(filename)

    def get_graph(self, filename: str) -> dict:
        return self.store.load_graph(filename)

    def get_graph_html(self, filename: str) -> str | None:
        return self.store.load_graph_html(filename)

    def list_knowledge(self, filename: str) -> dict:
        return {
            "wiki": self.store.has_wiki(filename),
            "graph": self.store.has_graph(filename),
            "html": self.store.has_graph_html(filename),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_wiki_orchestrator(store: DarwinStore, model_manager=None) -> WikiOrchestrator:
    return WikiOrchestrator(store, model_manager=model_manager)