# Document Parsing Module

Extract text from documents with automatic format detection and parallel processing.

## Overview

The parsing module converts documents into structured markdown text, preserving layout information, headings, tables, and optionally extracting images.

It is the first stage of the ingestion pipeline and produces clean text for downstream chunking and indexing.

## 📥 Supported File Types

* `.pdf`
* `.docx`
* `.pptx`
* `.txt`
* `.md`
* `.json`

## Features

- **Multi-Format Support**: Handles supported document types
- **Auto-Detection**: Detects file type automatically
- **Parallel Processing**: Multi-CPU parsing for batches
- **Dual Modes**: Fast or precise extraction
- **Image Extraction**: Optional image extraction and captioning
- **Smart Caching**: Skips already parsed files
- **Progress Tracking**: Real-time logging per file
- **Cluster Support**: Parse selected files only
- **Error Recovery**: Continues even if some files fail

## Pipeline Context

```

INGESTION PIPELINE
│
▼
PARSING (this)
│
▼
CHUNKING
│
▼
INDEXING

```

### Parsing Flow

```

Files (.pdf, .docx, .pptx, .txt, .md, .json)
│
▼
[1] Detect file type
│
▼
[2] Extract content
- text
- structure (headings, pages)
- tables
- images (optional)
│
▼
[3] Save as ParsedDocument

````

## Usage

### Basic

```bash
# Parse everything in current directory
uv run src/core/Ingestion/Parsing/main.py

# Parse a directory
uv run src/core/Ingestion/Parsing/main.py ~/documents/

# Parse recursively
uv run src/core/Ingestion/Parsing/main.py ~/documents/ --recursive
````

### Single File

```bash
uv run src/core/Ingestion/Parsing/main.py --file biology.pdf

# With images
uv run src/core/Ingestion/Parsing/main.py --file report.pdf --images

# With captions
uv run src/core/Ingestion/Parsing/main.py --file textbook.pdf --images --caption

# Precise mode
uv run src/core/Ingestion/Parsing/main.py --file complex.pdf --precise
```

### File Cluster

```bash
uv run src/core/Ingestion/Parsing/main.py --cluster bio.pdf chem.pdf physics.pdf
```

### Cache Control

```bash
# Default skips cached
uv run src/core/Ingestion/Parsing/main.py

# Force re-run
uv run src/core/Ingestion/Parsing/main.py --rerun

# Preview only
uv run src/core/Ingestion/Parsing/main.py --file doc.pdf --no-save
```

## Parameters

### Core

| Parameter   | Description          |
| ----------- | -------------------- |
| `path`      | Directory to scan    |
| `--file`    | Parse a single file  |
| `--cluster` | Parse specific files |

### Processing

| Flag          | Description                         |
| ------------- | ----------------------------------- |
| `--rerun`     | Reprocess cached files              |
| `--recursive` | Scan subdirectories                 |
| `--images`    | Extract images                      |
| `--caption`   | Generate captions (requires images) |
| `--precise`   | High-accuracy parsing               |
| `--no-save`   | Do not save output                  |

### Config

| Flag        | Default | Description       |
| ----------- | ------- | ----------------- |
| `--ws-port` | 8765    | WebSocket port    |
| `--no-ws`   | False   | Disable WebSocket |

## Extraction Modes

### Fast (default)

* Fast execution
* Low memory usage
* Basic layout preservation

### Precise (`--precise`)

* Better layout accuracy
* Improved table handling
* Slower and higher memory usage

## Output

### ParsedDocument

```python
@dataclass
class ParsedDocument:
    display_text: str
    embedding_text: str
    filename: str
    page_count: int
    file_type: str
    error: str = ""
```

### Example

```
✓ biology.pdf — 42 pages · 125k chars
✓ chemistry.pdf — 38 pages · 98k chars
✗ corrupt.pdf — error
```

## Image Extraction

```bash
uv run src/core/Ingestion/Parsing/main.py --file doc.pdf --images
```

Output:

```
.darwin/images/doc.pdf/
├── page_001_img_001.png
├── page_002_img_001.png
└── captions.json
```

## Performance

* Uses multiple CPU cores
* Large files processed more carefully
* Image extraction reduces parallelism

## Error Handling

Parsing continues even if some files fail:

```
✓ file1.pdf
✓ file2.docx
✗ file3.pdf — error

Parsed: 2
Failed: 1
```

## Integration

### CLI

```bash
uv run src/core/Ingestion/main.py --file document.pdf
```

### Programmatic

```python
from src.features.Ingestion.Parsing.main import handle_parsing
from src.features.Ingestion.Parsing.lib import ParsingArgs

args = ParsingArgs(
    target="~/docs",
    file=None,
    cluster=None,
    recursive=True,
    rerun=False,
    images=True,
    caption=False,
    precise=True,
    save=True,
)

docs = handle_parsing(args)
```

## Structure

```
src/core/Ingestion/Parsing/
├── main.py
├── README.md
└── lib.py

src/features/Ingestion/Parsing/
├── Parser.py
├── lib.py
```

## Troubleshooting

### Unsupported file

Only these formats work:

```
.pdf .docx .pptx .txt .md .json
```

### Encrypted PDF

Decrypt before parsing.

### Large file issues

Avoid `--precise` for very large PDFs.

### Slow parsing

Disable images if not needed.

```
uv run src/core/Ingestion/Parsing/main.py --file doc.pdf
```


