Here’s a concise README for this module:

---

# Wiki Generation CLI

This module generates structured, Wikipedia-style pages from previously extracted knowledge graphs. It transforms graph concepts into readable documentation with optional summaries.

## What It Does

* Loads concept graphs from storage
* Creates a page per concept
* Adds:

  * Keywords
  * Related concepts
  * Frequency data
* Optional:

  * LLM-generated summaries
  * Manifest (overview of all concepts)
* Outputs structured “wiki” data for each document

## Requirements

* Graphs must already exist. Run graph extraction first
* Uses:

  * `WikiGenerator`
  * `DarwinStore`
  * `DarwinLogger`

## Usage

### Basic

```bash id="w1k1b1"
uv run src/features/Knowledge/Wiki/main.py --file biology.pdf
```

### Inspect output

```bash id="w1k1b2"
uv run src/features/Knowledge/Wiki/main.py --file biology.pdf --inspect
```

### Generate summaries with LLM

```bash id="w1k1b3"
uv run src/features/Knowledge/Wiki/main.py \
  --file biology.pdf \
  --mode summary \
  --llm ollama/llama3.2
```

### Limit concepts

```bash id="w1k1b4"
uv run src/features/Knowledge/Wiki/main.py --file biology.pdf --max-concepts 20
```

### Force regeneration

```bash id="w1k1b5"
uv run src/features/Knowledge/Wiki/main.py --file biology.pdf --rerun
```

## Key Options

| Flag                   | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| `--file, -f`           | Files to process                                           |
| `--mode, -m`           | `index` (structure only) or `summary` (with LLM summaries) |
| `--llm`                | LLM model for summaries                                    |
| `--max-concepts, -n`   | Max number of concept pages                                |
| `--chunks-per-concept` | Context size per concept                                   |
| `--inspect, -i`        | Print wiki details                                         |
| `--rerun, -R`          | Regenerate even if cached                                  |
| `--ws-port`            | WebSocket port (default: 8765)                             |
| `--no-ws`              | Disable WebSocket logging                                  |

## Output

* **Console logs**: progress and results
* **Wiki pages**:

  * Concept title
  * Keywords
  * Related concepts
  * Optional summaries
* **Manifest**: overview of all concepts (if generated)

## Notes

* Skips files without graphs
* Automatically processes all valid files if none specified
* WebSocket server runs by default for real-time logs
* Summary mode depends on LLM availability

---
