# Features — Summarization, Images

**What it is:** Optional feature modules that extend Darwin's core RAG capabilities — extractive summarization and image processing (OCR + vision captioning).

**Why it exists:** RAG alone isn't always enough. These features add text summarization for chunk reduction and image OCR/description for multi-modal documents.

## Files

| File | Lines | Module | Purpose |
|---|---|---|---|
| `Summarization/Summarizer.py` | 580 | Summarization | Extractive text summarization |
| `Images/ImageHandler.py` | 745 | Images | OCR text extraction + vision captioning |

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
