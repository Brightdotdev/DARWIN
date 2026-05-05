
## `src/features/Retrival/Sematic/README.md`

```markdown
# Semantic Search Module

Dense vector-based semantic search using embeddings for conceptual understanding.

## Overview

This module provides neural semantic search using document embeddings and vector similarity. Unlike keyword search, it understands meaning and context, making it ideal for conceptual queries, paraphrased content, and natural language questions.

## Features

- **Vector Similarity Search**: Cosine similarity over dense embeddings
- **Score Thresholding**: Filter results by minimum relevance score
- **Multiple Search Scopes**: Search all documents, single files, or clusters
- **Detailed Inspection**: View full chunk content with similarity scores
- **JSON Export**: Output results in machine-readable format

## Usage

### Basic Search
```bash
# Search all embedded documents
uv run src/features/Retrival/Sematic/main.py "photosynthesis process"

# Alternative query syntax
uv run src/features/Retrival/Sematic/main.py --query "how do plants make energy"
```

### Search Specific Documents
```bash
# Search a single document
uv run src/features/Retrival/Sematic/main.py "cell division" --file biology.pdf

# Search a cluster of documents
uv run src/features/Retrival/Sematic/main.py "protein folding" --file biochem.pdf mol_bio.pdf
```

### Advanced Options
```bash
# Get more results with filtering
uv run src/features/Retrival/Sematic/main.py "mitochondria function" --top-k 10

# Set minimum score threshold (0-1)
uv run src/features/Retrival/Sematic/main.py "ATP synthesis" --min-score 0.5

# Inspect detailed results
uv run src/features/Retrival/Sematic/main.py "DNA replication" --inspect

# Export as JSON
uv run src/features/Retrival/Sematic/main.py "enzyme kinetics" --json
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top-k` | 5 | Number of results to return |
| `--min-score` | 0.3 | Minimum similarity score threshold (0-1) |
| `--file` | None | Search specific file(s) |
| `--inspect` | False | Show detailed chunk information |
| `--json` | False | Output results as JSON |

## Score Threshold Guide

The `--min-score` parameter filters results by relevance (0-1 scale):

- **0.7-1.0**: Very high relevance, near-exact matches
- **0.5-0.7**: Good relevance, strong conceptual match
- **0.3-0.5**: Moderate relevance, loosely related content
- **0.0-0.3**: Low relevance, tangentially related (default cutoff)

Adjust based on your precision/recall needs:
- Higher thresholds (0.6+): Fewer but more precise results
- Lower thresholds (0.2+): More results, may include loosely related content

## Output Format

### Standard Output
```
  #    Score       Source                Page  Preview
  ───  ──────────  ────────────────────  ────  ───────
  1    0.892       biology.pdf           5     Photosynthesis is the process by which...
  2    0.756       botany.pdf            12    Light-dependent reactions occur in...
```

### JSON Output
```json
[
  {
    "score": 0.892,
    "source": "biology.pdf",
    "page": 5,
    "text": "Photosynthesis is the process by which...",
    "heading": "Chapter 4: Photosynthesis"
  }
]
```

## Prerequisites

- Documents must be processed through the embedding pipeline
- Embedding model must be configured in your project settings
- Check embedding status with: `uv run src/features/Retrival/main.py`

## Notes

- Semantic search excels at conceptual understanding and paraphrased queries
- Best for natural language questions and abstract concepts
- Combine with keyword search for hybrid retrieval
- Cluster search enables focused semantic exploration of related documents
- Embedding quality depends on the underlying model
```
