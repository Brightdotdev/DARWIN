# Darwin Knowledge Graph Extraction Pipeline

## Overview

Document-agnostic knowledge graph extraction using **spaCy only** — no LLM required. Extracts typed entities (PERSON, ORG, GPE, PRODUCT, etc.) and explicit subject-verb-object relations from chunked documents. Built for downstream wiki generation with full evidence tracking.

---

## Architecture

```
Document → Chunks → spaCy nlp.pipe() → [Doc per chunk]
    │
    ├──→ NER → EntityRegistry (canonicalization + deduplication)
    │       └──→ ConceptNode: id, label, type, frequency, first_mention_chunk
    │
    ├──→ Dependency Parser → SVO Extraction (5 strategies)
    │       └──→ RelationAccumulator (deduplication + confidence scoring)
    │       └──→ Relationship: source, target, relation_type, evidence_chunks
    │
    └──→ Co-occurrence Edges (entities in same chunk)
            └──→ Relationship: source, target, "co_occurrence", weight

    → Pareto Pruning (frequency × density × centrality) → max_concepts
    → GraphData with cluster_map for wiki generation
```

---

## Files

| File | Purpose | Status |
|------|---------|--------|
| `GraphConfig.py` | Tunable configuration + 5 document-type presets | **NEW** |
| `GraphExtractor.py` | Main extraction pipeline (NER + SVO + pruning) | **REPLACED** |
| `GraphUtils.py` | Data models (ConceptNode, Relationship, GraphData) | **MODIFIED** |
| `GraphVisualizer.py` | Interactive HTML visualization (pyvis) | **MODIFIED** |
| `main.py` | CLI entry point with WebSocket logging | **MODIFIED** |
| `ConceptClustering.py` | Legacy HDBSCAN clustering (backward compat) | **DEPRECATED** |

---

## Quick Start

### Basic Usage

```bash
# Use document-type preset (recommended)
uv run src/features/Knowledge/Graph/main.py --file paper.pdf --preset academic --rerun

# Override preset settings
uv run src/features/Knowledge/Graph/main.py --file paper.pdf --preset academic --max-concepts 30

# Full manual control
uv run src/features/Knowledge/Graph/main.py --file book.txt --processing-mode sliding_window --chunk-overlap 0.5 --relation-threshold 0.0
```

### Document-Type Presets

| Preset | Best For | Key Settings |
|--------|----------|--------------|
| `academic` | Research papers, arXiv, conference proceedings | `threshold=0.0`, `sliding_window`, `max_concepts=50` |
| `research` | Deep reports, white papers, technical docs | `threshold=0.0`, `chunked`, `max_concepts=75` |
| `legal` | Contracts, court opinions, statutes | `threshold=1.0`, `full`, `strict` dedup, `max_concepts=40` |
| `narrative` | Books, biographies, news, biblical texts | `threshold=0.3`, `sliding_window`, `max_concepts=100` |
| `technical` | API docs, READMEs, engineering specs | `threshold=0.0`, `chunked`, `require_typed_entities=False` |

### CLI Arguments

```
--file, -f              One or more files to process
--preset, -p            Document-type preset (academic|research|legal|narrative|technical)
--processing-mode, -m   auto|full|chunked|sliding_window
--chunk-size            Tokens per chunk (default: 512)
--chunk-overlap         Overlap fraction 0.0-1.0 (default: 0.2)
--max-concepts, -n      Hard ceiling on nodes (default: 75)
--min-freq              Minimum chunk occurrences per entity (default: 2)
--relation-threshold    Confidence: 0.0=keep all, <1.0=fractional, >=1.0=absolute count
--dedup-strategy        svo_canonical|strict|none
--similarity-threshold  Co-occurrence edge floor (default: None=auto)
--require-ner           Fail if NER unavailable (default: True)
--require-parser        Fail if parser unavailable (default: True)
--spacy-batch           Batch size for nlp.pipe() (default: 128)
--visualize, -v         Generate HTML graph
--inspect, -i           Print graph details to log
--json, -j              Export JSON to stdout
--rerun, -R             Force re-extraction
--ws-port               WebSocket server port (default: 8765)
--no-ws                 Disable WebSocket server
```

---

## Configuration

### Programmatic

```python
from src.features.Knowledge.Graph.GraphConfig import GraphExtractionConfig

# From preset
config = GraphExtractionConfig.from_preset("academic", overrides={"max_concepts": 30})

# Manual
config = GraphExtractionConfig(
    processing_mode="sliding_window",
    chunk_size=512,
    chunk_overlap=0.3,
    max_concepts=50,
    relation_confidence_threshold=0.0,  # keep all explicit SVOs
    dedup_strategy="svo_canonical",
    require_typed_entities=True,
    require_parser=True,
)
```

### Tunable Parameters

| Parameter | Description | Tuning Advice |
|-----------|-------------|---------------|
| `processing_mode` | How to split document | `full` for small docs, `sliding_window` for best quality, `chunked` for speed |
| `chunk_overlap` | Fraction of overlap between chunks | 0.0 = no overlap (fast), 0.5 = 50% overlap (best entity linking) |
| `max_concepts` | Hard node ceiling | Lower = tighter focus, higher = more comprehensive |
| `relation_confidence_threshold` | Relation filtering | **0.0** for academic papers (keep all), **0.6** for general, **1.0+** to require multiple mentions |
| `dedup_strategy` | How to merge duplicate relations | `svo_canonical` = lemmatize+sort (recommended), `strict` = exact match, `none` = keep all |
| `max_cooccurrence_edges` | Cap on co-occurrence edges | Lower = sparser graph, higher = more connections |

---

## SVO Extraction Strategies

The pipeline extracts relations using **5 complementary strategies**:

| Strategy | Example | Handles |
|----------|---------|---------|
| **Active SVO** | "We propose CursorBench" | Standard subject-verb-object |
| **Passive SVO** | "CursorBench was evaluated on..." | Passive voice with optional agent |
| **Prepositional** | "outperforms GPT-4 on benchmarks" | Verb + prep + object |
| **Copular** | "CursorBench is a benchmark" | "is/was/become" + complement |
| **Nominal** | "performance of CursorBench on tasks" | Noun + prep + object |

---

## Entity Canonicalization

Entities are merged across chunks using:

1. **Exact match** (case-insensitive, punctuation-stripped)
2. **Substring match** ("Apple" ↔ "Apple Inc.")
3. **Jaccard ≥ 0.5** on word sets ("Steve Jobs" ↔ "Jobs, Steve")

No embedding-based linking (spaCy-only constraint). Deterministic and fast.

---

## Relation Deduplication

SVO triples are canonicalized:
- **Subject/Object**: `normalize_entity()` — lowercase, strip punctuation
- **Verb**: `lemmatize_verb()` + synonym normalization

**Verb synonym groups** (examples):
- `develop` ← develop, create, build, design, implement
- `propose` ← propose, suggest, introduce, present
- `outperform` ← outperform, exceed, surpass, beat
- `show` ← show, demonstrate, illustrate, reveal

---

## Confidence Scoring

| Threshold Mode | Behavior | Best For |
|----------------|----------|----------|
| `≤ 0.0` | Keep all explicit relations | Academic papers (sparse relations) |
| `< 1.0` | Fractional: occurrences / shared_chunks | General documents |
| `≥ 1.0` | Absolute count: require N mentions | Legal, high-confidence only |

---

## Pareto Pruning

Nodes scored on three axes (weighted):
- **Frequency** (40%): how many chunks mention the entity
- **Relation density** (30%): how many edges it participates in
- **Centrality** (30%): degree in the full graph

Top `max_concepts` nodes kept. Edges filtered to only include kept nodes.

---

## Output Schema

```json
{
  "filename": "paper.pdf",
  "nodes": [
    {
      "id": "e0",
      "label": "cursorbench",
      "display_name": "CursorBench",
      "type": "ORG",
      "freq": 15,
      "first_mention_chunk": 0,
      "chunks": [0, 1, 3, 5, 8],
      "mention_forms": ["CursorBench", "Cursor Bench"]
    }
  ],
  "edges": [
    {
      "source": "e0",
      "target": "e1",
      "weight": 1.0,
      "relation_type": "outperform",
      "evidence_chunks": [3, 8],
      "surface_forms": ["outperforms", "outperformed"]
    },
    {
      "source": "e0",
      "target": "e2",
      "weight": 0.8,
      "relation_type": "co_occurrence",
      "evidence_chunks": [1, 3, 5]
    }
  ],
  "chunk_count": 98,
  "total_entities_extracted": 635,
  "total_relations_extracted": 443,
  "processing_mode": "auto"
}
```

---

## Entity Filtering

The pipeline automatically rejects:
- Section/chapter/figure/table references ("Section 5", "Figure 3")
- Citation markers ("et al", "ibid")
- Number + label patterns ("Composer 2", "Agent 3")
- Pure numbers/symbols
- Single letters or initials

---

## spaCy Requirements

**Core engine**: spaCy with NER and dependency parser.

Supported models from your registry:
- `en_core_web_md` (45MB, recommended)
- `en_core_web_lg` (560MB, higher accuracy)

**Fail-fast behavior**: If `require_typed_entities=True` and NER is unavailable, or `require_parser=True` and parser is unavailable, the pipeline returns an error graph immediately.

**Installation**:
```bash
uv add spacy
uv run python -m spacy download en_core_web_md
```

Or via your `ModelStore` download system (ensure all components present in `nlp.pipe_names`).

---

## Performance

| Step | Time (98 chunks, CPU) | Bottleneck |
|------|----------------------|------------|
| spaCy NER + parsing | ~30-60s | `nlp.pipe(batch_size=128)` |
| Entity canonicalization | ~1s | In-memory dict ops |
| SVO extraction | ~2s | Per-sentence traversal |
| Relation deduplication | ~1s | Counter operations |
| Co-occurrence edges | ~1s | Pair counting |
| Pareto pruning | ~1s | Sort + slice |
| **Total** | **~40-70s** | spaCy parsing |

**Speed tips**:
- Use `batch_size_spacy=256` for large documents
- Use `processing_mode=chunked` instead of `sliding_window` for 2x speed
- Lower `max_concepts` reduces pruning time negligibly

---

## Visualization

```bash
uv run src/features/Knowledge/Graph/main.py --file paper.pdf --preset academic --visualize
```

Generates `graph.html` with:
- **Node colors** by entity type (PERSON=red, ORG=teal, GPE=blue, etc.)
- **Directed semantic edges** with arrows
- **Dashed co-occurrence edges** (undirected)
- **Edge tooltips** showing relation type, confidence, evidence chunks

---

## Troubleshooting

### "0 semantic relations"

**Cause**: `relation_confidence_threshold` too high for document type.

**Fix**: Use `--preset academic` (threshold=0.0) or explicitly `--relation-threshold 0.0`

### "646 raw entities, mostly noise"

**Cause**: spaCy NER misclassifying technical terms.

**Fix**: Entity filtering is active. Check `_is_bad_entity()` in `GraphExtractor.py` if specific patterns slip through.

### "spaCy loaded: pipes=[]"

**Cause**: Model loaded without components.

**Fix**: Verify model integrity:
```bash
uv run python -c "import spacy; nlp = spacy.load('path/to/en_core_web_md'); print(nlp.pipe_names)"
```
Should print: `['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']`

### "GraphData has no attribute total_concepts_extracted"

**Cause**: Legacy store code expects old field name.

**Fix**: `GraphData` now has `total_concepts_extracted` as a property alias. Update your `GraphUtils.py`.

---

## Limitations (By Design)

1. **No LLM**: No implicit inference, no multi-hop reasoning, no coreference beyond word-overlap aliases
2. **No temporal ordering**: `first_mention_chunk` is structural, not chronological
3. **No acronym expansion**: "LLM" and "large language model" are separate entities unless they share canonical form
4. **Shallow relations**: Only explicit SVO from dependency parse; no "is-a", "part-of", or domain ontologies
5. **English-centric**: spaCy `en_core_web_md` assumed; multilingual requires different spaCy model
6. **No cross-document linking**: Entity canonicalization is per-document only

---

## Migration from Old Pipeline

Old cached graphs (cluster-based) are still loadable via `store.load_graph()`. New extractions use the spaCy pipeline.

To force re-extraction:
```bash
uv run src/features/Knowledge/Graph/main.py --file old_doc.pdf --rerun
```

Key differences:
| Old | New |
|-----|-----|
| HDBSCAN clusters | spaCy NER entities |
| Noun phrase "concepts" | Typed entities (PERSON, ORG, etc.) |
| Centroid similarity edges | SVO semantic relations |
| Undirected "context_similarity" | Directed, typed relations |
| No evidence tracking | `evidence_chunks` on every edge |

---

## License

Internal Darwin component. Not for external distribution.
