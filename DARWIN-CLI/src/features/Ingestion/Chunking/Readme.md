# Document Chunking Module

Split parsed documents into structured chunks for embedding and retrieval.

## Overview

The chunking module takes parsed documents and divides them into smaller, semantically meaningful pieces optimized for embedding models and search systems.

It sits between parsing and indexing in the ingestion pipeline.

## Pipeline Context

```

INGESTION PIPELINE
│
▼
PARSING
│
▼
CHUNKING (this)
│
▼
INDEXING

```

### Chunking Flow

```

Parsed Documents (.darwin markdown)
│
▼
[1] Load parsed text
│
▼
[2] Split into chunks
- token-aware
- heading-aware
- overlap handling
│
▼
[3] Save chunks + metadata

````

## Features

- **Token-Aware Splitting**: Respects model token limits
- **Smart Overlap**: Maintains context across chunks
- **Heading Awareness**: Preserves document structure
- **Link Extraction**: Captures hyperlinks inside chunks
- **Inspection Mode**: View chunk stats and samples
- **Cluster Support**: Chunk selected files only
- **Smart Caching**: Skip already chunked files

## Usage

### Basic

```bash
# Chunk everything parsed in current directory
uv run src/core/Ingestion/Chunking/main.py
````

### Single File

```bash id="c1"
uv run src/core/Ingestion/Chunking/main.py --file biology.pdf
```

### Cluster (multiple files)

```bash id="c2"
uv run src/core/Ingestion/Chunking/main.py --cluster bio.pdf chem.pdf physics.pdf
```

### Inspection

```bash id="c3"
uv run src/core/Ingestion/Chunking/main.py --file biology.pdf --inspect
```

### Cache Control

```bash id="c4"
# Skip already chunked (default)
uv run src/core/Ingestion/Chunking/main.py

# Force rechunk
uv run src/core/Ingestion/Chunking/main.py --rerun
```

## Parameters

### Core

| Parameter   | Description                            |
| ----------- | -------------------------------------- |
| `path`      | Project directory                      |
| `--file`    | Chunk a single file                    |
| `--cluster` | Chunk specific files (space-separated) |

### Processing

| Flag        | Description                    |
| ----------- | ------------------------------ |
| `--rerun`   | Rechunk even if chunks exist   |
| `--inspect` | Print chunk stats and previews |

### Config

| Flag        | Default | Description       |
| ----------- | ------- | ----------------- |
| `--ws-port` | 8765    | WebSocket port    |
| `--no-ws`   | False   | Disable WebSocket |

## Rules

* `--file` and `--cluster` cannot be used together
* Files must already be parsed before chunking
* Uses `.darwin` store as source of truth

## Output

### Chunk Structure

```python id="c5"
class Chunk:
    index: int
    text: str
    heading: str | None
    page: int
    page_end: int
    token_count: int
    is_overlap: bool
    links: list
```

### Example

```id="c6"
✓ biology.pdf — 120 chunks · avg 450 tokens · 32 links
✓ chemistry.pdf — 98 chunks · avg 420 tokens · 18 links
✗ physics.pdf — error
```

## Inspection Mode

Shows:

* total chunks
* token distribution
* headings coverage
* overlap usage
* link statistics
* sample chunks (first, middle, last)

Example:

```
total chunks        120
avg tokens          450
with heading        80 (67%)
overlap chunks      30
```

## Performance

* Fast processing (pure text)
* Scales linearly with document size
* Minimal memory overhead

## Error Handling

```
✓ file1.pdf
✓ file2.pdf
✗ file3.pdf — no markdown found

done — 2 chunked · 1 failed
```

## Integration

### CLI Pipeline

```bash id="c7"
uv run src/core/Ingestion/main.py --file document.pdf
```

### Programmatic

```python id="c8"
from src.features.Ingestion.Chunking.main import handle_chunking
from src.features.Ingestion.Chunking.lib import ChunkingArgs

args = ChunkingArgs(
    target=".",
    file=None,
    cluster=None,
    inspect=False,
    rerun=False,
)

handle_chunking(args)
```

## Structure

```
src/core/Ingestion/Chunking/
├── main.py
├── README.md

src/features/Ingestion/Chunking/
├── Chunker.py
├── lib.py
```

## Troubleshooting

### "No parsed files found"

Run parsing first:

```bash id="c9"
uv run src/core/Ingestion/Parsing/main.py
```

### "Unexpected markdown type"

Indicates corrupted or invalid stored data. Re-run parsing:

```bash id="c10"
uv run src/core/Ingestion/Parsing/main.py --rerun
```

### "Already chunked"

Use:

```bash id="c11"
uv run src/core/Ingestion/Chunking/main.py --rerun
```

## Notes

* Chunk size and overlap are controlled internally by `chunk_for_model`
* Output is optimized for embedding models and retrieval systems
* This stage is required before indexing (BM25 or embeddings)

