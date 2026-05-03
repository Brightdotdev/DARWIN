# Darwin — Wiki Generation Module

The wiki generation module transforms extracted knowledge graphs into human-readable documentation. It produces interlinked markdown files—either as a phonebook-style concept index (no LLM) or Wikipedia-style summaries (LLM-powered). The output works seamlessly with Obsidian, VS Code, or any markdown viewer.

---

## How to run it

```bash
# Generate concept index (phonebook style, no LLM)
uv run src/core/Knowledge/Wiki/main.py --file biology.pdf

# Generate for all documents with extracted graphs
uv run src/core/Knowledge/Wiki/main.py

# Generate Wikipedia-style summaries (requires LLM)
uv run src/core/Knowledge/Wiki/main.py --file bible.pdf --mode summary --llm ollama/llama3.2

# Limit number of concepts
uv run src/core/Knowledge/Wiki/main.py --file bible.pdf --max-concepts 30

# Save to specific output directory
uv run src/core/Knowledge/Wiki/main.py --file bible.pdf --output ./my_wiki

# Inspect generated content before saving
uv run src/core/Knowledge/Wiki/main.py --file bible.pdf --inspect

# Force regeneration (ignore cache)
uv run src/core/Knowledge/Wiki/main.py --file bible.pdf --rerun
```

All flags can be combined. Example — generate a compact index with inspection:

```bash
uv run src/core/Knowledge/Wiki/main.py --file bible.pdf --mode index --max-concepts 25 --output ./bible_wiki --inspect
```

---

## How it works

```
Graph + Chunks → Select top concepts → Gather relevant chunks → Generate pages → Save markdown
```

1. **Concept selection**: Top N concepts are selected from the knowledge graph (ranked by frequency)
2. **Context gathering**: For each concept, relevant chunks are collected from the original document
3. **Page generation**: 
   - **Index mode**: Markdown templates with page references and text snippets
   - **Summary mode**: LLM synthesizes chunks into flowing prose with citations
4. **Linking**: Pages are interlinked based on graph relationships
5. **Output**: Folder of `.md` files ready for Obsidian or any markdown reader

---

## Generation modes

| Mode | Flag | Uses LLM? | Output Style | Best For |
|------|------|-----------|--------------|----------|
| **Index** | `--mode index` | ❌ No | Phonebook with page numbers | Quick reference, citation lookup |
| **Summary** | `--mode summary` | ✅ Yes | Wikipedia-style prose | Reading, studying, synthesis |

**Index mode** is instant, free, and works completely offline. **Summary mode** produces beautiful prose but requires an LLM (local or API).

---

## Output structure

```
output_directory/
├── biology_pdf_index.md    # Master index of all concepts
├── Photosynthesis.md       # Concept page
├── Chloroplast.md          # Concept page
├── Calvin_Cycle.md         # Concept page
├── ATP.md                  # Concept page
└── ... (one file per concept)
```

### Master index example

```markdown
# biology.pdf — Concept Index

*50 concepts extracted*

## All Concepts

- **[Photosynthesis](#photosynthesis)** — 47 mentions | Related: Chloroplast, Light, Calvin Cycle
- **[Chloroplast](#chloroplast)** — 38 mentions | Related: Photosynthesis, Thylakoid, Stroma
- **[ATP](#atp)** — 42 mentions | Related: Mitochondria, Energy, Cellular Respiration
```

### Concept page example (Index mode)

```markdown
# Photosynthesis

*Frequency: 47 mentions across 32 chunks*

**Related concepts:** Chloroplast, Light reactions, Calvin cycle, Chlorophyll, ATP

## Document References

### Page 42 — Light-Dependent Reactions
> ...Photosynthesis occurs in chloroplasts, converting light energy to chemical energy stored in ATP and NADPH...
*Chunk 87*

### Page 45 — Calvin Cycle
> ...The Calvin cycle uses ATP and NADPH from the light reactions to fix carbon dioxide into glucose...
*Chunk 92*
```

### Concept page example (Summary mode)

```markdown
# Photosynthesis

Photosynthesis is the process by which plants convert light energy into chemical energy [Page 42]. The process occurs in specialized organelles called chloroplasts [Page 42].

## Light-Dependent Reactions
The light-dependent reactions take place in the thylakoid membrane [Page 43]. Photons strike photosystem II, exciting electrons that travel down the electron transport chain [Page 43-44].

## Calvin Cycle
The Calvin cycle uses ATP and NADPH from the light reactions to fix carbon dioxide into glucose [Page 45-46]. This process is also known as the dark reactions, though they can occur in light.

**See also:** Chloroplast, Chlorophyll, ATP, NADPH, Light reactions, Calvin cycle
```

---

## Core functions

| Function | Purpose |
|----------|---------|
| `generate_wiki()` | Main orchestrator — routes to index or summary mode |
| `generate_concept_index()` | Phonebook-style index (no LLM) |
| `generate_concept_summary()` | Wikipedia-style prose (LLM required) |
| `run_wiki_pipeline()` | Full pipeline with store integration |

All functions return a `WikiIndex` object containing all generated pages.

---

## Data models

```python
@dataclass
class WikiPage:
    title: str                    # Page title (concept name or "Index")
    content: str                  # Full markdown content
    concept: Optional[str]        # Concept name (None for index page)
    chunk_count: int              # Number of chunks referenced
    related: List[str]            # Related concepts

@dataclass
class WikiIndex:
    filename: str                 # Source document name
    pages: List[WikiPage]         # All generated pages
    mode: str                     # "index" or "summary"
    concept_count: int            # Number of concepts covered
```

---

## Integration with other modules

```python
from src.features.Wiki import run_wiki_pipeline

# Generate wiki from existing graph and chunks
wiki = run_wiki_pipeline(
    filename="biology.pdf",
    store=store,
    mode="index",
    max_concepts=50,
    output_dir=Path("./my_wiki"),
)

# Access generated content
for page in wiki.pages:
    print(f"Generated: {page.title} ({page.chunk_count} references)")
```

---

## CLI Arguments

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--file` | `-f` | None | Generate for specific file |
| `--project` | `-p` | cwd | Project directory |
| `--mode` | `-m` | index | Generation mode: `index` or `summary` |
| `--llm` | | ollama/llama3.2 | LLM model for summary mode |
| `--max-concepts` | | 50 | Maximum concepts to include |
| `--chunks-per-concept` | | 10 | Max chunks per concept for context |
| `--output` | `-o` | None | Output directory for `.md` files |
| `--inspect` | `-i` | False | Show preview of generated content |
| `--rerun` | `-R` | False | Force regeneration |

---

## Performance characteristics

| Mode | Time (50 concepts) | Cost |
|------|-------------------|------|
| Index | <1 second | Free |
| Summary (local LLM) | 2-5 minutes | Free (your hardware) |
| Summary (API LLM) | 1-2 minutes | ~$0.05-0.20 per document |

**Index mode** is instant because it only formats existing data. **Summary mode** makes one LLM call per concept plus one for the master index.

---

## File structure

```
src/core/Wiki/
├── __init__.py           Module exports
├── WikiGenerator.py      Core generation logic, mode routing
├── WikiUtils.py          Templates, formatting, file output
└── main.py               CLI entry point, inspection
```

### Data flow

```
CLI args
  → handle_wiki_generation()              main.py
    → run_wiki_pipeline()                 WikiGenerator.py
      → DarwinStore.load_graph()          Load extracted concepts
      → DarwinStore.load_chunks()         Load original chunks
      → generate_concept_index()          Index mode (no LLM)
      → generate_concept_summary()        Summary mode (LLM)
        → litellm.completion()            One call per concept
      → save_wiki_files()                 Write .md files
    → DarwinStore.save_wiki()             Cache for future use
```

---

## Constants

| Constant | Default | Effect |
|----------|---------|--------|
| `DEFAULT_MODE` | `"index"` | Default generation mode |
| `DEFAULT_MAX_CONCEPTS` | 50 | Max concepts to include |
| `DEFAULT_CHUNKS_PER_CONCEPT` | 10 | Max chunks per concept |
| `SUMMARY_PROMPT_TEMPLATE` | (see code) | LLM prompt for summaries |
| `INDEX_PROMPT_TEMPLATE` | (see code) | LLM prompt for master index |

---

## Obsidian Integration

The generated markdown files are Obsidian-ready:

1. Generate wiki with `--output ./vault/`
2. Open Obsidian and select `./vault/` as your vault
3. Use **Graph View** to see the concept network
4. Use `[[wikilinks]]` syntax for navigation (already included)
5. Backlinks panel shows all connections automatically

---

## Use cases

| Scenario | Recommended Mode |
|----------|-----------------|
| Quick citation lookup | Index |
| Studying a textbook | Summary |
| Research literature review | Index (then Summary for key papers) |
| Building a personal knowledge base | Index |
| Preparing lecture notes | Summary |
| Offline/air-gapped environment | Index |

---
