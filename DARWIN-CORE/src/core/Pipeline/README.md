# Pipeline — Ingestion & Indexing

**What it is:** The document processing pipeline. Ingestion parses documents into markdown and chunks them into records. Indexing builds BM25 keyword indexes and embedding vectors for search.

**Why it exists:** Raw documents can't be searched directly. They need to be parsed (PDF→markdown), chunked (split into semantic segments), and indexed (BM25 for keyword search + embeddings for semantic search). This pipeline automates that end-to-end.

## Architecture

```
Document ──> Ingestion ──> Indexing ──> Search-ready
                │              │
            ┌───┴───┐      ┌──┴──┐
            │Parse  │      │BM25 │
            │Chunk  │      │Embed│
            └───────┘      └─────┘
```

## Ingestion

### Parsing

**Entry:** `src/core/Pipeline/Ingestion/Parsing/main.py :: handle_parsing()`

Supported formats (from `lib.py :: SUPPORTED_TYPES`):

| Extension | Extractor |
|---|---|
| `.pdf` | PyMuPDF, PyMuPDF4LLM |
| `.docx`, `.pptx` | python-docx, python-pptx |
| `.txt`, `.md` | Plain text |
| `.html`, `.htm` | Trafilatura |
| `.csv`, `.xlsx`, `.ods` | Tabular parsers |
| `.epub` | EPUB parser |
| `.eml` | Email parser |
| URLs | `UrlExtractor` + `UrlFetcher` (via httpx) |

```python
from src.core.Pipeline.Ingestion.Parsing.main import handle_parsing
from src.core.Pipeline.Ingestion.Parsing.lib import ParsingArgs

args = ParsingArgs(strategy="auto", ocr=True)
result = handle_parsing("document.pdf", args)
# result.markdown, result.metadata, result.images
```

### Chunking

**Entry:** `src/core/Pipeline/Ingestion/Chunking/main.py :: handle_chunking()`

Chunks are sentence-aware with structural classification (header, body, table, list).

```python
from src.core.Pipeline.Ingestion.Chunking.main import handle_chunking
from src.core.Pipeline.Ingestion.Chunking.lib import ChunkingArgs

args = ChunkingArgs(chunk_size=512, overlap=64, overlap_direction="right")
chunks = handle_chunking(markdown_text, args)
# List of Chunk objects with text, metadata, structural_type
```

## Indexing

### BM25

**Entry:** `src/core/Pipeline/Indexing/BM25/main.py :: handle_bm25_indexing()`

Builds sparse BM25 matrix (SciPy NPZ) + token metadata (Parquet).

```python
from src.core.Pipeline.Indexing.BM25.main import handle_bm25_indexing

handle_bm25_indexing(store, file_hash, chunks_df, args)
```

### Embedding

**Entry:** `src/core/Pipeline/Indexing/Embedding/main.py :: handle_embedding()`

Generates dense embeddings via ONNX, stores NumPy float16 + USearch vector index.

```python
from src.core.Pipeline.Indexing.Embedding.main import handle_embedding

handle_embedding(store, file_hash, chunks_df, args)
```

## End-to-End

```python
from src.core.Pipeline.Ingestion.main import handle_ingestion
from src.core.Pipeline.Indexing.main import handle_indexing

# Ingest
ingest_result = handle_ingestion(store, file_hash, file_path, parsing_args, chunking_args)

# Index
index_result = handle_indexing(store, file_hash, chunk_df, indexing_args)
```
