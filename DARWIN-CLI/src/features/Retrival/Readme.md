# Unified Retrieval Module

Single entry point for all search methods: Hybrid, Semantic, and Keyword with optional cross-encoder reranking.

## Features

- **Three Search Methods**: Hybrid (RRF fusion), Semantic (dense), Keyword (BM25)
- **Optional Reranking**: Cross-encoder reranking for improved precision
- **Multiple Scopes**: Search all documents, single files, or file clusters
- **Smart Defaults**: Hybrid search by default with sensible parameters
- **Unified Interface**: Consistent CLI across all search methods

## Quick Start

```bash
# Default hybrid search
uv run src/core/Retrieval/main.py "photosynthesis"

# With reranking
uv run src/core/Retrieval/main.py "photosynthesis" --rerank

# Specific method
uv run src/core/Retrieval/main.py "cell division" --method keyword
uv run src/core/Retrieval/main.py "mitochondria function" --method semantic
```

## Pipeline Context

```
                    ┌─────────────────────────────┐
                    │   UNIFIED RETRIEVAL          │
                    │   (This Module)              │
                    └─────────────┬───────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
        ┌───────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
        │   HYBRID     │  │  SEMANTIC   │  │  KEYWORD    │
        │   (Default)  │  │  (Dense)    │  │  (BM25)     │
        └───────┬──────┘  └──────┬──────┘  └──────┬──────┘
                │                 │                 │
                └─────────────────┼─────────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   OPTIONAL RERANKING         │
                    │   (Cross-Encoder)            │
                    └─────────────────────────────┘
```

## Usage Examples

### Basic Usage
```bash
# Default hybrid search (all documents)
uv run src/core/Retrieval/main.py "photosynthesis process"

# Query with flag
uv run src/core/Retrieval/main.py --query "DNA replication"
```

### Method Selection
```bash
# Hybrid (default) - combines semantic + keyword
uv run src/core/Retrieval/main.py "photosynthesis" --method hybrid

# Semantic - for conceptual queries
uv run src/core/Retrieval/main.py "how do plants make energy" --method semantic

# Keyword - for exact term matching
uv run src/core/Retrieval/main.py "chlorophyll thylakoid" --method keyword
```

### Search Scope
```bash
# Search specific file
uv run src/core/Retrieval/main.py "ATP synthesis" --file biology.pdf

# Search cluster of files
uv run src/core/Retrieval/main.py "protein folding" --cluster bio.pdf chem.pdf

# Search all documents (default)
uv run src/core/Retrieval/main.py "mitosis"
```

### Reranking
```bash
# Enable reranking
uv run src/core/Retrieval/main.py "photosynthesis" --rerank

# Custom rerank weight
uv run src/core/Retrieval/main.py "cell biology" --rerank --rerank-weight 0.2

# Semantic + reranking
uv run src/core/Retrieval/main.py "mitochondria" --method semantic --rerank
```

### Method-Specific Parameters
```bash
# Keyword with custom BM25 params
uv run src/core/Retrieval/main.py "enzymes" --method keyword --k1 1.2 --b 0.5

# Semantic with score threshold
uv run src/core/Retrieval/main.py "mitosis" --method semantic --min-score 0.5

# Hybrid with custom RRF
uv run src/core/Retrieval/main.py "DNA" --method hybrid --rrf-k 30 --semantic-weight 1.5
```

### Output Options
```bash
# More results
uv run src/core/Retrieval/main.py "ATP" --top-k 10

# Detailed inspection
uv run src/core/Retrieval/main.py "enzymes" --inspect

# JSON export
uv run src/core/Retrieval/main.py "cell biology" --json

# Full example
uv run src/core/Retrieval/main.py "photosynthesis" --method hybrid --file bio.pdf --rerank --top-k 10 --inspect
```

## Parameters

### Core Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | Required | Search query (positional or --query) |
| `--method` | hybrid | Retrieval method: hybrid, semantic, keyword |
| `--file` | None | Search specific file |
| `--cluster` | None | Search cluster of files |
| `--top-k` | 5 | Number of final results |

### Hybrid Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pool-size` | 50 | Candidates per retriever |
| `--rrf-k` | 60 | RRF dampening constant |
| `--semantic-weight` | 1.0 | Semantic score weight |
| `--keyword-weight` | 1.0 | Keyword score weight |
| `--alpha` | 0.5 | Alpha for hybrid scoring |
| `--beta` | 0.5 | Beta for hybrid scoring |

### Semantic Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-score` | 0.3 | Minimum similarity threshold |

### Keyword Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--k1` | 1.5 | BM25 term frequency saturation |
| `--b` | 0.75 | BM25 length normalization |

### Reranking Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rerank` | False | Enable cross-encoder reranking |
| `--rerank-weight` | 0.3 | Original score weight (0=pure rerank, 1=pure original) |
| `--rerank-top-k` | 100 | Number of candidates to rerank |

## Output Format

### Standard Output
```
======================================================================
Search Results: 'photosynthesis'
======================================================================

  #    Score       Source                  Page  Preview
  ───  ──────────  ──────────────────────  ────  ───────
  1    0.934       biology.pdf             5     Photosynthesis is the process by which...
  2    0.891       botany.pdf              12    Light-dependent reactions occur in...
  
======================================================================
Found: 10 results
Sources: 3 files
Top score: 0.9340
======================================================================
```

### JSON Output
```json
[
  {
    "final_score": 0.934,
    "rerank_score": 8.452,
    "original_score": 0.856,
    "text": "Photosynthesis is the process...",
    "source": "biology.pdf",
    "page": 5,
    "heading": "Chapter 4: Photosynthesis"
  }
]
```

## Prerequisites

- **Keyword**: BM25 indexing pipeline completed
- **Semantic**: Embedding pipeline completed
- **Hybrid**: Both BM25 and embedding pipelines completed
- **Reranking**: Cross-encoder model available via ModelManager

