"""
Darwin WikiOrchestrator — Per-File Entity Wiki & Graph
Pure math + NLP. No LLM. Logs everything via DarwinLogger.
"""

import gc
import json
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.Logger.DarwinLogger import darwin
from src.core.Models.ModelManager import modelManager
from src.core.Models.Lib.ModelRegistry import ModelCategory
from src.core.Storage.DarwinStore import DarwinStore


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


# ═══════════════════════════════════════════════════════════════════════════════
# Canonicalizer  (pure math)
# ═══════════════════════════════════════════════════════════════════════════════

class _EntityCanonicalizer:
    def __init__(self, threshold: float = 0.82):
        self.threshold = threshold
        self._entries: Dict[str, _WikiEntry] = {}
        self._lock = threading.RLock()

    def _normalize(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+', '', text, flags=re.I)
        text = re.sub(r'^(\w)\.\s+', r'\1 ', text)
        text = re.sub(r'^([\w\s]+),\s+(\w)\.?$', r'\2 \1', text)
        return text.lower().strip()

    def _similarity(self, a: str, b: str) -> float:
        lev = SequenceMatcher(None, self._normalize(a), self._normalize(b)).ratio()

        def tri(s):
            s = s.lower()
            return {s[i:i + 3] for i in range(len(s) - 2)}
        ta, tb = tri(a), tri(b)
        jac = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
        return max(lev, jac)

    def resolve(self, mention: str, entity_type: str) -> str:
        with self._lock:
            norm = self._normalize(mention)

            for canonical, entry in self._entries.items():
                if self._normalize(canonical) == norm:
                    entry.aliases.add(mention)
                    entry.types.add(entity_type)
                    return canonical
                if norm in {self._normalize(a) for a in entry.aliases}:
                    entry.aliases.add(mention)
                    entry.types.add(entity_type)
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
                return best_match

            self._entries[mention] = _WikiEntry(
                canonical_name=mention,
                aliases={mention},
                types={entity_type},
                first_seen=datetime.utcnow().isoformat(),
            )
            return mention

    def update(self, canonical: str, context: str):
        with self._lock:
            entry = self._entries[canonical]
            entry.mention_count += 1
            if len(entry.contexts) < 15:
                entry.contexts.append(context[:300])

    def to_dict(self) -> dict:
        return {
            name: {
                "canonical_name": e.canonical_name,
                "aliases": list(e.aliases),
                "types": list(e.types),
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

        # Tier 1: NER
        for ent in doc.ents:
            if ent.label_ in self.cfg.skip_labels:
                continue
            if len(ent.text) > self.cfg.max_entity_len:
                continue
            sent = ent.sent.text[:400] if ent.sent else ""
            candidates.append(_ExtractedEntity(
                text=ent.text.strip(), label=ent.label_,
                start=ent.start_char, end=ent.end_char,
                source="ner", sentence=sent,
            ))

        # Tier 2: Noun chunks
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
                if len(chunk_text) > self.cfg.max_entity_len:
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

        if len(counts) > 10:
            sorted_counts = sorted(counts.values(), reverse=True)
            idx = max(0, int(len(sorted_counts) * 0.3) - 1)
            floor = sorted_counts[min(idx, len(sorted_counts) - 1)]
        else:
            floor = 1
        floor = max(floor, self.cfg.min_mentions)

        contexts = defaultdict(list)
        for c in candidates:
            contexts[c.text].append(c.sentence)

        def entropy(texts: List[str]) -> float:
            if not texts:
                return 0.0
            joined = " ".join(texts)
            if len(joined) < 10:
                return 1.0
            freqs = Counter(joined)
            total = len(joined)
            probs = [f / total for f in freqs.values()]
            return -sum(p * math.log2(p) for p in probs if p > 0)

        filtered = []
        for c in candidates:
            if counts[c.text] < floor:
                continue
            if self.cfg.min_entropy > 0 and entropy(contexts[c.text]) < self.cfg.min_entropy:
                continue
            filtered.append(c)

        return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Builder  (PMI + Dependency)
# ═══════════════════════════════════════════════════════════════════════════════

class _GraphBuilder:
    def __init__(self, config: ExtractionConfig):
        self.cfg = config
        self._cooccur: Dict[Tuple[str, str], int] = defaultdict(int)
        self._dep_edges: List[dict] = []

    def process(self, doc, canonical_map: Dict[str, str]):
        sentences = list(doc.sents)

        # Co-occurrence
        for i in range(len(sentences)):
            window = sentences[i:i + self.cfg.window_size]
            window_text = " ".join(s.text for s in window)

            found = set()
            for raw, canonical in canonical_map.items():
                if raw in window_text:
                    found.add(canonical)

            for a in found:
                for b in found:
                    if a >= b:
                        continue
                    key = tuple(sorted((a, b)))
                    self._cooccur[key] += 1

        # Dependency paths
        if doc.has_annotation("DEP"):
            for token in doc:
                if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                    subj_can = None
                    for raw, canonical in canonical_map.items():
                        if raw in token.text or token.text in raw:
                            subj_can = canonical
                            break

                    for child in token.head.children:
                        if child.dep_ in ("dobj", "pobj", "attr"):
                            obj_can = None
                            for raw, canonical in canonical_map.items():
                                if raw in child.text or child.text in raw:
                                    obj_can = canonical
                                    break

                            if subj_can and obj_can and subj_can != obj_can:
                                self._dep_edges.append({
                                    "source": subj_can, "target": obj_can,
                                    "label": token.head.lemma_,
                                    "type": "dependency", "weight": 1.0,
                                })

    def build(self, canonicalizer: _EntityCanonicalizer) -> Tuple[List[dict], List[dict]]:
        entries = canonicalizer.entries
        if not entries:
            return [], []

        nodes = []
        for name, entry in entries.items():
            nodes.append({
                "id": name, "label": name,
                "type": list(entry.types)[0] if entry.types else "UNKNOWN",
                "mentions": entry.mention_count,
                "aliases": list(entry.aliases),
            })

        total_cooccur = sum(self._cooccur.values()) or 1
        entity_counts = {name: e.mention_count for name, e in entries.items()}
        total_mentions = sum(entity_counts.values()) or 1

        edges = []
        for (a, b), count in self._cooccur.items():
            if a not in entity_counts or b not in entity_counts:
                continue
            p_a = entity_counts[a] / total_mentions
            p_b = entity_counts[b] / total_mentions
            p_ab = count / total_cooccur

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

        edges.extend(self._dep_edges)

        seen = set()
        deduped = []
        for e in edges:
            key = tuple(sorted((e["source"], e["target"])))
            if key in seen and e["type"] == "cooccurrence":
                continue
            if e["type"] == "dependency":
                seen.add(key)
            deduped.append(e)

        return nodes, deduped


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Visualizer
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
            f'<div class="legend-item"><span class="type" style="background:{c}"></span>{t}</div>'
            for t, c in cls.TYPE_COLORS.items()
        )

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
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
<h3>{title}</h3>
<div class="stat">Nodes: {len(nodes)}</div>
<div class="stat">Edges: {len(edges)}</div>
<div class="legend">{legend_html}</div>
<div id="info" style="margin-top:15px;"></div>
</div>
<script>
var cy=cytoscape({{
  container:document.getElementById('cy'),
  elements:{json.dumps(cy_nodes + cy_edges)},
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
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class WikiOrchestrator:
    VALID_STAGES = {"wiki", "graph", "html"}

    def __init__(self, store: DarwinStore, model_manager=None):
        self.store = store
        self.model_manager = model_manager or modelManager
        self._nlp_engine = None
        self._nlp_available: Optional[bool] = None
        self._lock = threading.RLock()

    # ── NLP Lifecycle ─────────────────────────────────────────────────────────

    def _ensure_nlp(self) -> bool:
        if self._nlp_available is not None:
            return self._nlp_available

        with self._lock:
            if self._nlp_available is not None:
                return self._nlp_available

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

    def _get_doc(self, text: str):
        if self._nlp_engine is None:
            raise RuntimeError("NLP not loaded")
        return next(self._nlp_engine.pipe([text], batch_size=1))

    # ── Per-File Processing ───────────────────────────────────────────────────

    def _process_single(self, filename: str, text: str, stages: Set[str], config: ExtractionConfig):
        """Internal workhorse. Logs everything. Returns nothing."""
        try:
            doc = self._get_doc(text)
        except Exception as e:
            darwin.error(f"[{filename}] NLP parsing failed: {e}")
            return

        canonicalizer = _EntityCanonicalizer(threshold=config.similarity_threshold)
        raw_to_canonical: Dict[str, str] = {}

        # ── Stage: Wiki ─────────────────────────────────────────────────────
        if "wiki" in stages:
            try:
                extractor = _EntityExtractor(config)
                candidates = extractor.extract(doc)

                for c in candidates:
                    canonical = canonicalizer.resolve(c.text, c.label)
                    raw_to_canonical[c.text] = canonical
                    canonicalizer.update(canonical, c.sentence)

                wiki_data = {
                    "filename": filename,
                    "generated_at": datetime.utcnow().isoformat(),
                    "entities": canonicalizer.to_dict(),
                    "stats": {
                        "total_mentions": sum(e.mention_count for e in canonicalizer.entries.values()),
                        "unique_entities": len(canonicalizer.entries),
                    },
                }

                self.store.save_wiki(filename, wiki_data)
                darwin.info(f"[{filename}] Wiki saved: {len(canonicalizer.entries)} entities, {wiki_data['stats']['total_mentions']} total mentions")

            except Exception as e:
                darwin.error(f"[{filename}] Wiki build failed: {e}")
                return

        # ── Stage: Graph ────────────────────────────────────────────────────
        if "graph" in stages:
            try:
                if "wiki" not in stages:
                    extractor = _EntityExtractor(config)
                    candidates = extractor.extract(doc)
                    for c in candidates:
                        canonical = canonicalizer.resolve(c.text, c.label)
                        raw_to_canonical[c.text] = canonical
                        canonicalizer.update(canonical, c.sentence)

                builder = _GraphBuilder(config)
                builder.process(doc, raw_to_canonical)
                nodes, edges = builder.build(canonicalizer)

                self.store.save_graph(filename, nodes, edges)
                dep_count = sum(1 for e in edges if e.get("type") == "dependency")
                cooc_count = sum(1 for e in edges if e.get("type") == "cooccurrence")
                darwin.info(f"[{filename}] Graph saved: {len(nodes)} nodes, {len(edges)} edges ({dep_count} dependency, {cooc_count} co-occurrence)")

            except Exception as e:
                darwin.error(f"[{filename}] Graph build failed: {e}")
                return

        # ── Stage: HTML ─────────────────────────────────────────────────────
        if "html" in stages and "graph" in stages:
            try:
                graph_data = self.store.load_graph(filename)
                nodes = graph_data.get("nodes", [])
                edges = graph_data.get("edges", [])
                if nodes:
                    html = _GraphVisualizer.generate(
                        nodes, edges,
                        title=f"KG: {filename}",
                        max_nodes=config.max_nodes_html,
                    )
                    self.store.save_graph_html(filename, html)
                    darwin.info(f"[{filename}] HTML visualization saved")
                else:
                    darwin.warning(f"[{filename}] No graph nodes — skipping HTML")
            except Exception as e:
                darwin.warning(f"[{filename}] HTML generation failed: {e}")

    # ── Public API ──────────────────────────────────────────────────────────────

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
    ) -> None:
        """
        Extract wiki and/or graph for a single file. Returns nothing.
        All status is reported via DarwinLogger.
        """
        if stages is None:
            stages = {"wiki", "graph", "html"}
        stages = {s.lower() for s in stages} & self.VALID_STAGES

        if not stages:
            darwin.warning(f"[{filename}] No valid stages requested")
            return

        if not self._ensure_nlp():
            darwin.error(f"[{filename}] Cannot process — NLP engine unavailable")
            return

        text = self.store.load_markdown(filename)
        if text is None:
            darwin.error(f"[{filename}] Not registered in DarwinStore — cannot load markdown")
            return

        if not text.strip() or len(text.strip()) < 20:
            darwin.warning(f"[{filename}] Text too short ({len(text.strip())} chars) — skipping")
            return

        # Rerun guard
        if not rerun:
            missing = []
            if "wiki" in stages and not self.store.has_wiki(filename):
                missing.append("wiki")
            if "graph" in stages and not self.store.has_graph(filename):
                missing.append("graph")
            if "html" in stages and not self.store.has_graph(filename):
                missing.append("graph (for html)")

            if not missing:
                darwin.info(f"[{filename}] Skipped — artifacts already exist (rerun=False)")
                return

        config = ExtractionConfig(
            similarity_threshold=similarity_threshold,
            window_size=window_size,
            min_mentions=min_mentions,
            min_entropy=min_entropy,
            max_entity_len=max_entity_len,
            pmi_threshold=pmi_threshold,
            max_nodes_html=max_nodes_html,
            html_title=html_title,
            skip_labels=skip_labels or ExtractionConfig().skip_labels,
        )

        darwin.info(f"[{filename}] Starting knowledge extraction — stages: {sorted(stages)}")
        self._process_single(filename, text, stages, config)
        darwin.success(f"[{filename}] Knowledge extraction complete")

    def handle_knowledge_batch(
        self,
        filenames: List[str],
        *,
        rerun: bool = False,
        stages: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Process multiple files with the NLP model kept hot in memory.
        Logs per-file progress. Returns nothing.
        """
        if not filenames:
            darwin.warning("Batch knowledge called with empty file list")
            return

        if stages is None:
            stages = {"wiki", "graph", "html"}
        stages = {s.lower() for s in stages} & self.VALID_STAGES

        if not self._ensure_nlp():
            darwin.error("Batch aborted — NLP engine unavailable")
            return

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
            skip_labels=kwargs.get("skip_labels") or ExtractionConfig().skip_labels,
        )

        processed = 0
        skipped = 0
        failed = 0

        for filename in filenames:
            text = self.store.load_markdown(filename)
            if text is None:
                darwin.error(f"[{filename}] Not registered — skipped")
                failed += 1
                continue

            if not text.strip() or len(text.strip()) < 20:
                darwin.warning(f"[{filename}] Text too short — skipped")
                failed += 1
                continue

            # Rerun guard
            if not rerun:
                missing = []
                if "wiki" in stages and not self.store.has_wiki(filename):
                    missing.append("wiki")
                if "graph" in stages and not self.store.has_graph(filename):
                    missing.append("graph")
                if "html" in stages and not self.store.has_graph(filename):
                    missing.append("graph")

                if not missing:
                    darwin.info(f"[{filename}] Skipped — artifacts exist")
                    skipped += 1
                    continue

            self._process_single(filename, text, stages, config)
            processed += 1

        darwin.success(
            f"Batch complete: {processed} processed, {skipped} skipped, {failed} failed"
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def delete_knowledge(self, filename: str) -> None:
        """Remove all knowledge artifacts for a file."""
        ok_wiki = self.store.delete_wiki(filename)
        ok_graph = self.store.delete_graph(filename)
        ok_html = self.store.delete_graph_html(filename)

        if ok_wiki or ok_graph or ok_html:
            darwin.info(f"[{filename}] Knowledge artifacts deleted")
        else:
            darwin.warning(f"[{filename}] No knowledge artifacts found to delete")

    def get_wiki(self, filename: str) -> dict | None:
        return self.store.load_wiki(filename)

    def get_graph(self, filename: str) -> dict:
        return self.store.load_graph(filename)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_wiki_orchestrator(store: DarwinStore) -> WikiOrchestrator:
    return WikiOrchestrator(store)