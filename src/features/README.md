# Features — Knowledge, Summarization, Images

**What it is:** Optional feature modules that extend Darwin's core RAG capabilities — knowledge graph construction, extractive summarization, and image processing (OCR + vision captioning).

**Why it exists:** RAG alone isn't always enough. These features add entity extraction + relationship graphs, text summarization for chunk reduction, and image OCR/description for multi-modal documents.

## Files

| File | Lines | Module | Purpose |
|---|---|---|---|
| `Knowledge/WikiOrchestrator.py` | 1171 | Knowledge | Entity extraction + knowledge graph construction |
| `Summarization/Summarizer.py` | 580 | Summarization | Extractive text summarization |
| `Images/ImageHandler.py` | 745 | Images | OCR text extraction + vision captioning |

---

## Knowledge — WikiOrchestrator

**What it is:** Extracts entities and builds knowledge graphs from document text using pure NLP (spaCy NER + PMI scoring + co-occurrence analysis). No LLM required.

**Usage:**

```python
from src.features.Knowledge.WikiOrchestrator import WikiOrchestrator

orch = WikiOrchestrator(store)
result = orch.build_knowledge_graph(file_hash)

# Access entities and relationships
for entity in result.entities:
    print(f"{entity.name} ({entity.label}) x{entity.frequency}")

for rel in result.relationships:
    print(f"{rel.source} --[{rel.label}]--> {rel.target}")

# Export as PyVis HTML graph
html = result.graph_html
```

**Key features:**
- Batch NLP processing with spaCy
- PMI (Pointwise Mutual Information) scoring for relationship strength
- Co-occurrence analysis within sliding windows
- Parallel post-processing for large documents
- Auto-chunking for long documents (>100K chars)
- PyVis HTML graph export

---

## Summarization — Summarizer

**What it is:** Extractive summarization using spaCy + pytextrank. Identifies the most important sentences in a document based on text rank algorithm.

**Usage:**

```python
from src.features.Summarization.Summarizer import Summarizer

summarizer = Summarizer()
summary = summarizer.summarize(
    text="Long document text...",
    num_sentences=5,
    chunk_size=100000,
    overlap=200,
)
print(summary)
```

**Purposes:**
- Blob reduction before embedding
- Conversation history compression
- Retrieved results classification
- Pre-chunking content analysis

**Thread-safe:** Each instance has its own isolated spaCy pipeline.

---

## Images — ImageHandler

**What it is:** Unified image processing — OCR text extraction (RapidOCR) and vision captioning/description (ONNX vision model).

**Usage:**

```python
from src.features.Images.ImageHandler import ImageHandler

handler = ImageHandler()

# OCR: extract text from images
text = handler.extract_text("page1_img0.png")
print(text)

# Caption: generate description
caption = handler.describe("photo.jpg")
print(caption)

# Custom prompts
caption = handler.describe(
    "chart.png",
    vision_prompt="Describe this chart in detail including all data points."
)
```

**Batching:** Thread pool with configurable worker count for bulk processing.
