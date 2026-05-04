## `src/core/KeywordSearch/README.md`

```markdown
# Keyword Search Module

BM25-based keyword search over indexed documents using sparse retrieval.

## Overview

This module provides pure BM25 keyword search functionality, allowing you to search through indexed documents using traditional term-frequency based retrieval. It's ideal for finding exact term matches and works well with technical documents, legal texts, and other domain-specific content.

## Features

- **BM25 Retrieval**: Industry-standard probabilistic relevance scoring
- **Tunable Parameters**: Adjust k1 and b parameters for fine-tuned relevance
- **Multiple Search Scopes**: Search all documents, single files, or clusters
- **Detailed Inspection**: View full chunk content with metadata
- **JSON Export**: Output results in machine-readable format

## Usage

### Basic Search
```bash
# Search all indexed documents
uv run src/core/KeywordSearch/main.py "photosynthesis"

# Alternative query syntax
uv run src/core/KeywordSearch/main.py --query "cell division"
```

### Search Specific Documents
```bash
# Search a single document
uv run src/core/KeywordSearch/main.py "mitochondria" --file biology.pdf

# Search a cluster of documents
uv run src/core/KeywordSearch/main.py "DNA replication" --cluster genetics.pdf biology.pdf cell_bio.pdf
```

### Advanced Options
```bash
# Get more results
uv run src/core/KeywordSearch/main.py "enzymes" --top-k 10

# Tune BM25 parameters
uv run src/core/KeywordSearch/main.py "ATP synthesis" --k1 1.2 --b 0.5

# Inspect detailed results
uv run src/core/KeywordSearch/main.py "photosynthesis" --inspect

# Export as JSON
uv run src/core/KeywordSearch/main.py "cell biology" --json
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top-k` | 5 | Number of results to return |
| `--k1` | 1.5 | BM25 term frequency saturation parameter |
| `--b` | 0.75 | BM25 length normalization parameter |
| `--file` | None | Search specific file |
| `--cluster` | None | Search cluster of files (space-separated) |
| `--inspect` | False | Show detailed chunk information |
| `--json` | False | Output results as JSON |

## BM25 Parameters Explained

- **k1**: Controls term frequency saturation. Higher values give more weight to term frequency.
  - k1 = 0: No term frequency component (binary model)
  - k1 = 1.2-1.5: Standard settings for most collections
  - k1 = 2.0+: Stronger emphasis on frequent terms

- **b**: Controls document length normalization. Range is 0 to 1.
  - b = 0: No length normalization (longer docs favored)
  - b = 0.75: Standard setting (moderate normalization)
  - b = 1: Full length normalization (shorter docs favored)

## Output Format

### Standard Output
```
  #    Score       Source                  Page  Preview
  ───  ──────────  ──────────────────────  ────  ───────
  1    12.345      biology.pdf             5     Photosynthesis is the process by which...
  2    8.921       botany.pdf              12    Light-dependent reactions occur in...
```

### JSON Output
```json
[
  {
    "score": 12.345,
    "source": "biology.pdf",
    "page": 5,
    "text": "Photosynthesis is the process by which...",
    "heading": "Chapter 4: Photosynthesis",
    "matched_terms": ["photosynthesis", "chlorophyll"]
  }
]
```

## Prerequisites

- Documents must be indexed using the BM25 indexing pipeline
- Check searchable status with: `uv run src/core/Status/main.py`

## Notes

- Keyword search excels at exact term matching
- Best for technical/scientific documents with specific terminology
- Combine with semantic search for better recall on conceptual queries
- Cluster search allows focused retrieval across related documents
```
