
# Knowledge Pipeline CLI

This module runs the full knowledge pipeline. Graph extraction + wiki generation in one command.

## What It Does

* Processes chunked documents from storage
* **Pass 1**. Extracts concept graphs
* **Pass 2**. Generates wiki-style pages from those graphs
* Handles caching. Skips already processed files unless forced
* Supports running each stage independently

## Pipeline Flow

1. **Graph Extraction**

   * Extract concepts and relationships
   * Build node-edge graph

2. **Wiki Generation**

   * Convert graph into structured pages
   * Optional LLM summaries

## Requirements

* Documents must already be **chunked**
* Uses:

  * `GraphExtractor`
  * `WikiGenerator`
  * `DarwinStore`

## Usage

### Full pipeline

```bash
uv run src/features/Knowledge/main.py -f biology.pdf chemistry.pdf
```

### Process all files

```bash
uv run src/features/Knowledge/main.py
```

### With summaries (LLM)

```bash
uv run src/features/Knowledge/main.py \
  -f biology.pdf \
  --wiki-mode summary \
  --llm ollama/llama3.2
```

### Graph only

```bash
uv run src/features/Knowledge/main.py -f biology.pdf --graph-only
```

### Wiki only

```bash
uv run src/features/Knowledge/main.py -f biology.pdf --wiki-only
```

### Force full rerun

```bash
uv run src/features/Knowledge/main.py -f biology.pdf --rerun
```

## Key Options

### Pipeline Control

| Flag           | Description               |
| -------------- | ------------------------- |
| `--graph-only` | Run only graph extraction |
| `--wiki-only`  | Run only wiki generation  |

### Graph Options

| Flag                   | Description               |
| ---------------------- | ------------------------- |
| `--extraction-mode`    | `spacy` or `regex`        |
| `--elevation`          | `none`, `spacy`, `llm`    |
| `--threshold`          | Similarity threshold      |
| `--min-freq`           | Minimum concept frequency |
| `--graph-max-concepts` | Max graph nodes           |
| `--visualize`          | Generate HTML graph       |

### Wiki Options

| Flag                   | Description             |
| ---------------------- | ----------------------- |
| `--wiki-mode`          | `index` or `summary`    |
| `--llm`                | LLM model for summaries |
| `--wiki-max-concepts`  | Max wiki pages          |
| `--chunks-per-concept` | Context per concept     |

### Shared

| Flag         | Description                    |
| ------------ | ------------------------------ |
| `--file, -f` | Files to process               |
| `--inspect`  | Print detailed output          |
| `--rerun`    | Ignore cache                   |
| `--ws-port`  | WebSocket port (default: 8765) |
| `--no-ws`    | Disable WebSocket              |

## Output

* **Console logs** for each pass
* **Graphs**:

  * Concept nodes
  * Relationships
* **Wiki pages**:

  * Concept summaries
  * Keywords
  * Related concepts

## Notes

* Runs graph pass before wiki unless `--wiki-only`
* Wiki pass requires valid graphs
* Skips existing outputs unless `--rerun`
* Automatically processes all chunked files if none specified
* WebSocket logging enabled by default

