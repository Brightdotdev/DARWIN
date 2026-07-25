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

**Entry:** `src/core/Pipeline/Ingestion/main.py :: handle_ingestion()`

```python
from src.core.Pipeline.Ingestion.main import handle_ingestion
from src.core.Pipeline.Ingestion.Parsing.lib import ParsingArgs
from src.core.Storage.DarwinStore import DarwinStore

store = DarwinStore()
args = ParsingArgs(
    target=".",
    files=["document.pdf"],
    images=True,
    caption=True,
    ocr=False,
    precise=False,
    recursive=False,
    rerun=False,
    save=True,
)
handle_ingestion(args, store)
```

### Parsing

**Entry:** `src/core/Pipeline/Ingestion/Parsing/main.py :: handle_parsing()`

Supported formats (from `Parsing/lib.py :: SUPPORTED_TYPES`):

| Extension | Extractor backend |
|---|---|
| `.pdf` | PyMuPDF / PyMuPDF4LLM (precise mode) |
| `.docx`, `.pptx`, `.txt`, `.md` | MarkItDown |
| `.html`, `.htm` | Trafilatura (html processor) |
| `.xml` | XML parser |
| `.csv`, `.tsv` | CSV/tabular parser |
| `.xlsx`, `.xls`, `.ods` | Spreadsheet parser |
| `.epub` | Ebook parser |
| `.eml` | Email parser |
| URLs | `UrlExtractor` + `UrlFetcher` (httpx) |

```python
from src.core.Pipeline.Ingestion.Parsing.main import handle_parsing
from src.core.Pipeline.Ingestion.Parsing.lib import ParsingArgs
from src.core.Storage.DarwinStore import DarwinStore

store = DarwinStore()
args = ParsingArgs(target=".", files=["report.pdf"], images=True, caption=False)
docs = handle_parsing(args, store)
# List[ParsedDocument] — each has:  embedding_text, filename,
#   page_count, file_type, metadata, error, images
```

### Chunking

**Entry:** `src/core/Pipeline/Ingestion/Chunking/main.py :: handle_chunking()`

Chunks are sentence-aware with structural classification (header, body, table, list) and directional overlap (none, left, right, both).

```python
from src.core.Pipeline.Ingestion.Chunking.main import handle_chunking
from src.core.Pipeline.Ingestion.Chunking.lib import ChunkingArgs
from src.core.Storage.DarwinStore import DarwinStore

store = DarwinStore()
args = ChunkingArgs(files=["report.pdf"], rerun=False)
handle_chunking(args, store)
# Chunks are saved to the store via store.save_records()
```

## Indexing

**Entry:** `src/core/Pipeline/Indexing/main.py :: handle_indexing()`

```python
from src.core.Pipeline.Indexing.main import handle_indexing
from src.core.Pipeline.Indexing.lib import IndexingArgs
from src.core.Storage.DarwinStore import DarwinStore

store = DarwinStore()
args = IndexingArgs(
    path=".",
    threshold=0.5,
    idf_smoothing=0.1,
    files=["report.pdf"],  # None = all chunked files
    rerun=False,
    only_kw=False,
    only_semantic=False,
)
handle_indexing(args, store)
```

### BM25

**Entry:** `src/core/Pipeline/Indexing/BM25/main.py :: handle_bm25_indexing()`

Builds sparse BM25 matrix (SciPy NPZ) + token metadata (Parquet).

```python
from src.core.Pipeline.Indexing.BM25.main import handle_bm25_indexing
from src.core.Pipeline.Indexing.BM25.lib import BM25IndexingArgs
from src.core.Storage.DarwinStore import DarwinStore

store = DarwinStore()
args = BM25IndexingArgs(
    files=["report.pdf"],  # None = all chunked files
    threshold=0.5,
    idf_smoothing=0.1,
    rerun=False,
)
handle_bm25_indexing(args, store)
```

### Embedding

**Entry:** `src/core/Pipeline/Indexing/Embedding/main.py :: handle_embedding()`

Generates dense embeddings via the active embedding model, stores NumPy float16 + USearch vector index.

```python
from src.core.Pipeline.Indexing.Embedding.main import handle_embedding
from src.core.Pipeline.Indexing.Embedding.lib import EmbeddingArgs
from src.core.Storage.DarwinStore import DarwinStore

store = DarwinStore()
args = EmbeddingArgs(
    files=["report.pdf"],  # None = all chunked files
    rerun=False,
)
handle_embedding(args, store)
```
