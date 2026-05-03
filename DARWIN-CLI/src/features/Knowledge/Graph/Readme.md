# Darwin . Graph Extraction Module (AI-Enhanced)

The graph module converts chunked documents into a **semantic concept graph**. It extracts keywords, optionally upgrades them into real concepts using an LLM, and connects them using meaning, not just co-occurrence.

---

## How to run it

```bash id="g1">
# standard semantic graph (recommended baseline)
uv run src/core/Knowledge/Graph/main.py --file biology.pdf

# enable AI concept elevation(this only works if you have an llm config else you will be prompted to)
uv run src/core/Knowledge/Graph/main.py --file biology.pdf --ai

# combine with visualization
uv run src/core/Knowledge/Graph/main.py --file biology.pdf --ai --visualize

# inspect graph structure
uv run src/core/Knowledge/Graph/main.py --file biology.pdf --ai --inspect

# adjust similarity threshold
uv run src/core/Knowledge/Graph/main.py --file biology.pdf --ai --threshold 0.55

# force re-extraction
uv run src/core/Knowledge/Graph/main.py --file biology.pdf --ai --rerun
```

Example:

```bash id="g2">
uv run src/core/Knowledge/Graph/main.py --file theology.pdf --ai --threshold 0.5 --visualize --inspect
```

---

## What this module does

```id="g3">
chunks
  → extract keywords (regex / spacy)
  → optionally elevate into real concepts (LLM ✨)
  → filter + rank
  → embed concept labels
  → connect via cosine similarity
  → save graph
```

---

## The key upgrade: AI Concept Elevation ✨

Without AI:

```id="g4">
"god", "lord", "jesus", "spirit"
```

With AI:

```id="g5">
"Divine Trinity", "Salvation", "Covenant Law"
```

The LLM:

* merges related keywords
* removes noise
* creates human-readable concepts
* outputs structured JSON

This turns keyword lists into **actual knowledge graphs**.

---

## When to use `--ai`

Use it when:

* documents are abstract (philosophy, religion, law)
* keywords are noisy or repetitive
* you want mindmap-level concepts

Avoid it when:

* you need strict reproducibility
* you want fully offline pipelines
* latency matters (adds one LLM call per document)

---

## Concept extraction modes



### spaCy (default)

* better linguistic structure

```bash id="g7">
uv run python -m spacy download en_core_web_sm
```

---

### Regex (fallback)

* no dependencies
* lowest quality

---

---

## Semantic vs AI-enhanced graphs

### Standard semantic

```id="g8">
"neural network" ↔ "deep learning"
```

---

### AI-enhanced semantic

```id="g9">
"Representation Learning" ↔ "Artificial Intelligence Systems"
```

The second is what humans actually reason with.

---

## How AI elevation works

```id="g10">
top keywords
  → formatted prompt
  → single LLM call
  → JSON response
  → map back to chunk indices
```

Important detail:

* LLM does NOT see raw text
* only sees keywords + frequencies
* keeps it fast and cheap

---

## Output

```id="g11">
.darwin/
  biology_pdf/
    graph.json
    graph.html   ← if --visualize
```

---

## Visualization

When `--visualize` is used:

* node size = frequency
* node color = importance
* edge thickness = similarity
* hover = metadata

---

## Inspect mode

```bash id="g12">
uv run src/core/Knowledge/Graph/main.py --file biology.pdf --ai --inspect
```

Shows:

* total concepts
* relationships
* top nodes
* strongest edges

---

## File structure

```id="g13">
src/core/Knowledge/Graph/
  main.py               CLI
  GraphExtractor.py     pipeline + graph building
  ConceptElevator.py    LLM concept abstraction ✨
  GraphUtils.py         models, export, visualization
```

---

## Data flow

```id="g14">
CLI args
  → handle_graph_extraction()
    → load chunks
    → extract keywords
    → elevate concepts (optional ✨)
    → build semantic graph
    → save graph
    → optional visualization
```

---

## Key data models

### `ConceptNode`

```python id="g15">
@dataclass
class ConceptNode:
    id: str
    label: str
    frequency: int
    chunk_indices: List[int]
```

---

### `Relationship`

```python id="g16">
@dataclass
class Relationship:
    source: str
    target: str
    weight: float
    relation_type: str
```

---

### `GraphData`

```python id="g17">
@dataclass
class GraphData:
    filename: str
    nodes: List[ConceptNode]
    edges: List[Relationship]
    chunk_count: int
    total_concepts_extracted: int
```

---

## Embedding model

Default:

```id="g18">
intfloat/e5-small-v2
```

Loaded automatically via `SentenceTransformer`.

If it fails, system falls back to co-occurrence.

---

## Tuning constants

| Constant                       | Default | Effect                    |
| ------------------------------ | ------- | ------------------------- |
| `DEFAULT_MIN_CONCEPT_FREQ`     | 3       | Minimum concept frequency |
| `DEFAULT_MAX_CONCEPTS`         | 100     | Max nodes                 |
| `DEFAULT_SIMILARITY_THRESHOLD` | 0.45    | Edge cutoff               |

---

## Special behavior with `--ai`

When AI is enabled:

```id="g19">
min_freq → forced to 1
```

Reason:

* LLM needs full keyword visibility
* filtering happens implicitly via abstraction

---

## Performance notes

* YAKE is fast
* embedding is cheap (short phrases)
* AI adds:

  * 1 LLM call per document
  * biggest latency cost

---

## When things go wrong

### LLM returns garbage

System falls back to raw keywords automatically.

---

### JSON parse fails

Handled by:

```id="g20">
_parse_response()
```

Strips markdown, extracts valid JSON.

---

### Concepts look bad

Fix in order:

1. enable `--ai`
2. increase `--max-concepts`
3. adjust `--threshold`

---

## When to tweak settings

### Too many concepts

```bash id="g21">
--max-concepts 50
```

---

### Graph too dense

```bash id="g22">
--threshold 0.6
```

---

### Graph too sparse

```bash id="g23">
--threshold 0.35
```

---

### Weak concept quality

```bash id="g24">
--ai
```

---

## Typical pipeline

```id="g25">
Parsing → Chunking → Graph Extraction (AI ✨) → Embeddings → Retrieval
```

---

## Bottom line

* YAKE gives you keywords
* embeddings give you relationships
* AI gives you meaning

Without AI, you get a **keyword graph**
With AI, you get a **conceptual map of the document**

That difference is not small. It is the difference between noise and understanding.
