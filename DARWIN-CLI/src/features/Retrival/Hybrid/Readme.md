
## `src/features/Retrival/HybridSearch/README.md`

```markdown
# Hybrid Search Module

Combined semantic and keyword retrieval using Reciprocal Rank Fusion (RRF).

## Overview

This module merges the strengths of both semantic (dense) and keyword (sparse) search using Reciprocal Rank Fusion. By combining exact term matching with conceptual understanding, it provides more robust and relevant results across diverse query types.

## Features

- **Reciprocal Rank Fusion**: Sophisticated algorithm combining multiple retrieval signals
- **Weighted Combination**: Adjustable weights for semantic vs keyword importance
- **Configurable Pool**: Control how many candidates each retriever contributes
- **Overlap Detection**: Identifies potentially duplicated content across retrievers
- **Multiple Search Scopes**: Search all documents, single files or multiple flse
- **Detailed Scoring**: See RRF, semantic, and keyword scores individually
- **JSON Export**: Output results in machine-readable format

## Usage

### Basic Search
```bash
# Search all documents with hybrid retrieval
uv run src/features/Retrival/HybridSearch/main.py "photosynthesis"

# Alternative query syntax
uv run src/features/Retrival/HybridSearch/main.py --query "cell division mechanism"
```

### Search Specific Documents
```bash
# Search a single document
uv run src/features/Retrival/HybridSearch/main.py "david covenant" --file bible.pdf

# Search a cluster of documents
uv run src/features/Retrival/HybridSearch/main.py "quantum mechanics" --file physics.pdf modern_physics.pdf
```

### Advanced Options
```bash
# Tune RRF parameters
uv run src/features/Retrival/HybridSearch/main.py "cell division" --rrf-k 30 --semantic-weight 1.5

# More candidates from each retriever
uv run src/features/Retrival/HybridSearch/main.py "mitochondria" --pool-size 100 --top-k 10

# Inspect detailed scoring
uv run src/features/Retrival/HybridSearch/main.py "ATP synthesis" --inspect

# Export as JSON
uv run src/features/Retrival/HybridSearch/main.py "enzyme mechanisms" --json
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top-k` | 5 | Final number of results to return |
| `--rrf-k` | 60 | RRF dampening constant |
| `--semantic-weight` | 1.0 | Weight multiplier for semantic results |
| `--keyword-weight` | 1.0 | Weight multiplier for keyword results |
| `--pool-size` | 50 | Candidates per retriever before fusion |
| `--alpha` | 0.5 | Alpha parameter for hybrid scoring |
| `--beta` | 0.5 | Beta parameter for hybrid scoring |
| `--file` | None | Search specific file(s) |
| `--inspect` | False | Show detailed chunk information |
| `--json` | False | Output results as JSON |

## Parameter Tuning Guide

### RRF Parameters
- **rrf-k (60)**: Higher values reduce the impact of rank differences
  - k=0: Only first rank matters
  - k=60: Standard setting for most use cases
  - k=100+: Ranks become more uniformly weighted

### Weights
- **semantic-weight (1.0)**: Increase for conceptual/natural language queries
- **keyword-weight (1.0)**: Increase for exact term matching queries

Example weight combinations:
- Conceptual queries: `--semantic-weight 1.5 --keyword-weight 0.5`
- Technical queries: `--semantic-weight 0.5 --keyword-weight 1.5`
- Balanced: `--semantic-weight 1.0 --keyword-weight 1.0` (default)

### Pool Size
- **pool-size (50)**: Larger pools increase recall but decrease speed
  - 20-30: Fast, suitable for simple queries
  - 50-100: Balanced performance for most use cases
  - 100-200: High recall for complex queries

## Output Format

### Standard Output
```
[1] Score: 0.892 | Source: biology.pdf (p5)
    Photosynthesis is the process by which...
    (RRF: 0.891 | Sem: 0.823 | KW: 12.345)

[2] Score: 0.756 | Source: botany.pdf (p12)
    Light-dependent reactions occur in...
    (RRF: 0.754 | Sem: 0.712 | KW: 8.921)
```

### JSON Output
```json
[
  {
    "final_score": 0.892,
    "rrf_score": 0.891,
    "semantic_score": 0.823,
    "keyword_score": 12.345,
    "source": "biology.pdf",
    "page": 5,
    "text": "Photosynthesis is the process by which...",
    "heading": "Chapter 4: Photosynthesis",
    "matched_terms": ["photosynthesis", "chlorophyll"],
    "overlap_direction": "none"
  }
]
```

## How RRF Works

Reciprocal Rank Fusion combines rankings from multiple retrievers:

1. **Semantic Retriever**: Returns top `pool_size` results by embedding similarity
2. **Keyword Retriever**: Returns top `pool_size` results by BM25 score
3. **Fusion**: Each result gets an RRF score: `∑ 1/(k + rank_i)` for each retriever
4. **Weighting**: Weights are applied to bias semantic vs keyword importance
5. **Final Ranking**: Results are sorted by final RRF score

## Overlap Detection

The system detects when the same chunk appears in both semantic and keyword results, marking it with the overlap direction. This helps identify:
- Chunks that are conceptually AND terminologically relevant
- Potential duplicate content in the result set

## Prerequisites

- Documents must be indexed with both BM25 and embedding pipelines
- Both keyword and embedding search must be operational
- Check hybrid search status with: `uv run src/features/Retrival/Status/main.py`

## Notes

- Hybrid search provides the best overall retrieval quality
- Slightly slower than single-method search due to dual retrieval
- Ideal for production systems requiring robust search
- Cluster search with hybrid retrieval is particularly powerful for focused research
- Monitor overlap rates to understand your document structure
```
