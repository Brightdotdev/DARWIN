
# 🌿 Darwin Design System

## Overview

Darwin is a **structured, terminal-aware interface** designed for clarity, trust, and intelligent interaction.

It combines:

* **Notion**. predictable layout, spatial clarity
* **Claude**. soft motion, calm intelligence
* **Terminal systems**. truth, feedback, system visibility

The result:

> A system that feels reliable, readable, and alive without being distracting

---

# 1. Core Principles

### 1. Structure First

* Everything aligns to a grid
* No floating or ambiguous UI
* Users always know where they are

### 2. Signal Over Noise

* Color is used intentionally
* Motion is subtle and purposeful
* No visual clutter

### 3. Terminal as Truth Layer

* Terminal is a **component**, not the whole app
* Used for:

  * logs
  * progress
  * system feedback

### 4. Calm Intelligence

* No aggressive animations
* No loud colors
* UI feels composed and confident

---

# 2. Color System

## Base

```css
--bg: #0b0f14;
--surface-1: #0f1620;
--surface-2: #121c28;
--surface-3: #162230;
```

## Text

```css
--text-primary: #e6edf3;
--text-secondary: #9fb0c0;
--text-muted: #6b7c8f;
```

## Borders

```css
--border: rgba(255,255,255,0.06);
--border-strong: rgba(255,255,255,0.12);
```

## Brand (Muted Teal)

```css
--primary: #2f8f89;
--primary-hover: #3aa6a1;
--primary-soft: rgba(47,143,137,0.12);
--primary-glow: rgba(47,143,137,0.35);
```

### Usage Rules

* Primary is **never dominant**
* Used for:

  * active states
  * focus
  * confirmations
* Never used as large backgrounds

---

# 3. Typography

## Fonts

```css
--font-sans: Inter, system-ui, sans-serif;
--font-mono: JetBrains Mono, monospace;
```

## Scale

```css
--text-sm: 13px;
--text-md: 14px;
--text-lg: 16px;
--text-xl: 20px;
```

## Rules

* UI = sans-serif
* System / logs = monospace
* Small metadata may use mono

---

# 4. Spacing System

8pt grid:

```css
4px, 8px, 12px, 16px, 24px, 32px
```

Rules:

* Consistent vertical rhythm
* No arbitrary spacing
* Padding scales with component size

---

# 5. Surfaces & Elevation

### Surface Layers

```css
surface-1 → base cards
surface-2 → inputs, nested areas
surface-3 → deeper containers
```

### Elevation

* Achieved with contrast, not heavy shadows
* Borders define structure

---

# 6. Borders & Radius

```css
--radius-sm: 8px;
--radius-md: 12px;
--radius-lg: 14px;
```

Rules:

* No sharp edges
* No overly rounded UI
* Consistency across components

---

# 7. Motion

## Timing

* Fast: 120ms
* Normal: 180ms
* Max: 240ms

## Behavior

* Ease-out only
* No bounce
* No exaggerated movement

## Types

* Fade in → new content
* Slide up → emergence
* Soft highlight → attention

---

# 8. Components

## Cards

### Variants

* Small
* Medium
* Large

### Structure

* Surface background
* Subtle border
* Rounded corners

### States

* Default: neutral
* Hover: slight teal tint
* Active: stronger tint

---

## Buttons

### Primary

* Background: primary
* Text: white

### Ghost

* Transparent background
* Hover: primary-soft

### Rules

* No heavy shadows
* Clear interaction states

---

## Inputs

### Default

* Background: surface-2
* Border: subtle

### Focus

* Border → primary
* Soft glow

### Error

* Border → error color

---

## Terminal Component

### Role

* Displays system-level feedback
* Used for logs and progress

### Style

* Monospace font
* Slightly brighter accent color
* Structured output (not chaotic)

### Rules

* Never full-screen by default
* Always contained in panel/card

---

## Sidebar

### Behavior

* Always visible
* Collapsible, not hidden

### Active Item

* Left border accent
* Soft background tint

---

# 9. Interaction Design

### Hover

* Subtle highlight
* Never dramatic

### Focus

* Clear but soft
* Uses primary color

### Feedback

* Immediate
* Minimal animation

---

# 10. Feedback States

## Success

* Soft green
* Calm confirmation

## Warning

* Muted amber
* Non-alarming

## Error

* Desaturated red
* Clear but not aggressive

---

# 11. Layout System

### Rules

* Grid-based
* Consistent alignment
* No floating elements

### Feel

> Everything has a place
> Nothing feels accidental

---

# 12. Terminal Integration Philosophy

The terminal:

* enhances trust
* shows system transparency
* provides real-time feedback

It should feel:

* integrated, not separate
* structured, not chaotic
* useful, not decorative

---

# 13. What This System Avoids

* Overuse of color
* Loud animations
* Visual clutter
* “hacker aesthetic” terminal UIs
* Emotionally manipulative design

---

# 14. Final Experience

Darwin should feel like:

* a thinking tool
* a reliable system
* a place to get answers
* a calm, intelligent assistant
















{
  "research_agent_architecture": {
    "version": "2.0",
    "design_goal": "Deterministic multi-source academic retrieval and evidence evaluation pipeline for AI research agents",
    "principles": [
      "Separate retrieval from reasoning",
      "Do not treat all sources equally",
      "Prefer metadata graphs over raw search",
      "Rank evidence quality explicitly",
      "Avoid LLM summarization during routing",
      "Use structured retrieval indexes before semantic reasoning"
    ]
  },
  "source_registry": {
    "Computer Science": [
      {
        "name": "arXiv",
        "type": "preprint_repository",
        "roles": [
          "frontier_research",
          "full_text",
          "freshness"
        ],
        "peer_reviewed": false,
        "trust_level": 0.68,
        "freshness_score": 0.98,
        "citation_quality": 0.65,
        "api_quality": 0.9,
        "domains": [
          "computer_science",
          "mathematics",
          "physics",
          "machine_learning"
        ],
        "strengths": [
          "very recent papers",
          "excellent ML coverage",
          "high availability"
        ],
        "weaknesses": [
          "not peer reviewed",
          "contains speculative work"
        ]
      },
      {
        "name": "Semantic Scholar",
        "type": "citation_graph",
        "roles": [
          "metadata",
          "citation_graph",
          "ranking",
          "paper_similarity",
          "author_graph"
        ],
        "peer_reviewed": "mixed",
        "trust_level": 0.9,
        "freshness_score": 0.86,
        "citation_quality": 0.95,
        "api_quality": 0.95,
        "domains": [
          "multidisciplinary"
        ],
        "strengths": [
          "excellent citation graph",
          "strong embeddings",
          "paper relationships"
        ],
        "weaknesses": [
          "not full-text complete"
        ]
      },
      {
        "name": "CORE",
        "type": "open_access_aggregator",
        "roles": [
          "full_text",
          "oa_discovery"
        ],
        "peer_reviewed": "mixed",
        "trust_level": 0.74,
        "freshness_score": 0.78,
        "citation_quality": 0.62,
        "api_quality": 0.8,
        "domains": [
          "multidisciplinary"
        ]
      },
      {
        "name": "BASE",
        "type": "academic_search_engine",
        "roles": [
          "repository_discovery",
          "oa_search"
        ],
        "peer_reviewed": "mixed",
        "trust_level": 0.72,
        "freshness_score": 0.74,
        "citation_quality": 0.58,
        "api_quality": 0.7,
        "domains": [
          "multidisciplinary"
        ]
      }
    ],
    "Biomedicine": [
      {
        "name": "PubMed",
        "type": "biomedical_index",
        "roles": [
          "medical_truth",
          "citation_index",
          "biomedical_search"
        ],
        "peer_reviewed": true,
        "trust_level": 0.96,
        "freshness_score": 0.88,
        "citation_quality": 0.94,
        "api_quality": 0.92,
        "domains": [
          "medicine",
          "biology",
          "healthcare"
        ]
      },
      {
        "name": "PubMed Central",
        "type": "full_text_archive",
        "roles": [
          "full_text",
          "biomedical_oa"
        ],
        "peer_reviewed": true,
        "trust_level": 0.95,
        "freshness_score": 0.85,
        "citation_quality": 0.9,
        "api_quality": 0.9,
        "domains": [
          "medicine",
          "biology"
        ]
      },
      {
        "name": "bioRxiv",
        "type": "preprint_repository",
        "roles": [
          "frontier_biology",
          "freshness"
        ],
        "peer_reviewed": false,
        "trust_level": 0.7,
        "freshness_score": 0.97,
        "citation_quality": 0.63,
        "api_quality": 0.86,
        "domains": [
          "biology"
        ]
      },
      {
        "name": "medRxiv",
        "type": "medical_preprint_repository",
        "roles": [
          "medical_freshness"
        ],
        "peer_reviewed": false,
        "trust_level": 0.58,
        "freshness_score": 0.96,
        "citation_quality": 0.55,
        "api_quality": 0.82,
        "domains": [
          "medicine"
        ],
        "warning": "Medical claims must not be treated as consensus evidence"
      }
    ],
    "Social Sciences": [
      {
        "name": "ERIC",
        "type": "education_database",
        "roles": [
          "education_research"
        ],
        "peer_reviewed": "mixed",
        "trust_level": 0.84,
        "freshness_score": 0.78,
        "citation_quality": 0.75,
        "api_quality": 0.72,
        "domains": [
          "education",
          "social_science"
        ]
      },
      {
        "name": "SSRN",
        "type": "social_science_preprints",
        "roles": [
          "economics",
          "law",
          "policy",
          "fresh_research"
        ],
        "peer_reviewed": false,
        "trust_level": 0.69,
        "freshness_score": 0.91,
        "citation_quality": 0.68,
        "api_quality": 0.76,
        "domains": [
          "economics",
          "law",
          "social_science"
        ]
      },
      {
        "name": "OpenAlex",
        "type": "research_graph",
        "roles": [
          "metadata_graph",
          "citation_graph",
          "institution_graph",
          "author_disambiguation"
        ],
        "peer_reviewed": "mixed",
        "trust_level": 0.91,
        "freshness_score": 0.84,
        "citation_quality": 0.93,
        "api_quality": 0.94,
        "domains": [
          "multidisciplinary"
        ]
      }
    ],
    "Humanities": [
      {
        "name": "JSTOR",
        "type": "journal_archive",
        "roles": [
          "historical_research",
          "humanities"
        ],
        "peer_reviewed": true,
        "trust_level": 0.9,
        "freshness_score": 0.45,
        "citation_quality": 0.88,
        "api_quality": 0.6,
        "domains": [
          "history",
          "philosophy",
          "literature"
        ]
      },
      {
        "name": "Project MUSE",
        "type": "humanities_database",
        "roles": [
          "humanities",
          "social_science"
        ],
        "peer_reviewed": true,
        "trust_level": 0.87,
        "freshness_score": 0.62,
        "citation_quality": 0.84,
        "api_quality": 0.65,
        "domains": [
          "humanities"
        ]
      }
    ],
    "Open Access Journals": [
      {
        "name": "DOAJ",
        "type": "oa_journal_index",
        "roles": [
          "open_access_validation"
        ],
        "peer_reviewed": true,
        "trust_level": 0.88,
        "freshness_score": 0.75,
        "citation_quality": 0.8,
        "api_quality": 0.8,
        "domains": [
          "multidisciplinary"
        ]
      },
      {
        "name": "PLOS",
        "type": "oa_publisher",
        "roles": [
          "peer_reviewed_oa"
        ],
        "peer_reviewed": true,
        "trust_level": 0.91,
        "freshness_score": 0.82,
        "citation_quality": 0.89,
        "api_quality": 0.83,
        "domains": [
          "science",
          "medicine",
          "biology"
        ]
      },
      {
        "name": "BioMed Central",
        "type": "oa_publisher",
        "roles": [
          "biomedical_oa"
        ],
        "peer_reviewed": true,
        "trust_level": 0.89,
        "freshness_score": 0.8,
        "citation_quality": 0.86,
        "api_quality": 0.8,
        "domains": [
          "medicine",
          "biology"
        ]
      }
    ]
  },
  "evidence_model": {
    "fields": {
      "peer_reviewed": "boolean",
      "citation_strength": "0.0-1.0",
      "publisher_reputation": "0.0-1.0",
      "methodology_confidence": "0.0-1.0",
      "replication_confidence": "0.0-1.0",
      "recency_score": "0.0-1.0",
      "consensus_score": "0.0-1.0",
      "novelty_score": "0.0-1.0"
    }
  },
  "retrieval_index_design": {
    "description": "Deterministic routing layer before semantic reasoning",
    "schema": {
      "file": "string",
      "concepts": [
        "string"
      ],
      "entity_types": [
        "string"
      ],
      "concept_frequency": {
        "concept": "integer"
      },
      "top_relations": [
        "source -> target (relation)"
      ],
      "embedding_centroid": [
        "float"
      ]
    }
  },
  "routing_flow": {
    "step_1_query_analysis": {
      "actions": [
        "extract concepts",
        "detect domain",
        "detect intent",
        "estimate specificity"
      ]
    },
    "step_2_source_selection": {
      "rules": [
        "medicine -> prioritize PubMed",
        "cutting-edge AI -> prioritize arXiv",
        "citation tracing -> prioritize Semantic Scholar/OpenAlex",
        "open access required -> prioritize CORE/DOAJ",
        "humanities -> prioritize JSTOR/Project MUSE"
      ]
    },
    "step_3_retrieval_index_scan": {
      "actions": [
        "match concepts against retrieval indexes",
        "rank files by overlap",
        "boost by concept frequency",
        "penalize low-confidence matches"
      ]
    },
    "step_4_wiki_search": {
      "actions": [
        "retrieve structured pages",
        "expand neighboring concepts",
        "collect candidate evidence"
      ]
    },
    "step_5_graph_search": {
      "actions": [
        "traverse relationships",
        "follow citation chains",
        "identify consensus clusters",
        "detect contradictions"
      ]
    },
    "step_6_evidence_ranking": {
      "ranking_formula": {
        "citation_strength": 0.25,
        "publisher_reputation": 0.2,
        "methodology_confidence": 0.2,
        "consensus_score": 0.15,
        "recency_score": 0.1,
        "replication_confidence": 0.1
      }
    },
    "step_7_response_generation": {
      "rules": [
        "separate consensus from speculation",
        "flag preprints clearly",
        "cite strongest evidence first",
        "mention contradictory findings explicitly",
        "avoid treating freshness as correctness"
      ]
    }
  },
  "agent_modes": {
    "quick_lookup": {
      "uses": [
        "retrieval_index",
        "wiki_pages"
      ],
      "skips": [
        "deep_graph_traversal"
      ]
    },
    "deep_research": {
      "uses": [
        "retrieval_index",
        "wiki_search",
        "graph_traversal",
        "citation_analysis",
        "evidence_ranking"
      ]
    },
    "consensus_mode": {
      "focus": [
        "systematic_reviews",
        "highly_cited_work",
        "replicated_findings"
      ]
    },
    "frontier_mode": {
      "focus": [
        "recent_preprints",
        "novel_work",
        "emerging_topics"
      ],
      "warning": "Lower evidence confidence expected"
    }
  }
}










Here is a build prompt for parsing, indexing, and querying codebases using the AST-based approach discussed:

**1. Parse**
*   Use **Tree-sitter** to parse all code files (`.js`, `.py`, etc.) into Abstract Syntax Trees (ASTs).
*   Traverse the AST to extract semantic units (functions, classes, methods) as chunks, preserving their complete structure and context.

**2. Index**
*   For each chunk, generate a vector embedding using a model like `text-embedding-3-small` or a local model.
*   Store these embeddings in a vector database (e.g., LanceDB, Milvus) alongside rich metadata:
    *   File path and language.
    *   AST-derived info (function name, parameters, dependencies, imported symbols).
    *   Resolved cross-file references (e.g., "calls `getUser` in `user.js`").
*   Implement an incremental updater (e.g., via `git diff`) to re-parse only changed files and update the graph/index.

**3. Query**
*   On a natural language query (e.g., "How is user authentication handled?"):
    *   **(Optional Graph RAG)**: Translate the query into a graph query (e.g., Cypher) to find relevant nodes/edges (e.g., all functions calling `authenticateUser`).
    *   **Vector Search**: Use the query to find the top-k most similar code chunks from the vector database.
    *   **Hybrid**: Combine results from the graph and vector search for maximum relevance.
*   Return the matching code chunks with their metadata to the LLM for contextual reasoning.

This pipeline creates a powerful, real-time system for understanding large codebases.

