Here’s a clean, usable README for your file. No fluff, just what someone needs to actually run and integrate it.

---

# Wiki Concept Search

Scores extracted wiki concepts against a query with **per-file satisfaction gating**.
Designed to work like `KeywordSearch`, but over structured concept graphs.

---

## What it does

* Searches **wiki-extracted concepts** across documents
* Scores results using:

  * semantic similarity (centroid)
  * keyword hits
  * frequency
* Applies **per-file gating**:

  * each file must meet thresholds
* Returns:

  * per-file results
  * overall satisfaction status

---

## Requirements

Before running this, you must have:

* A populated `DarwinStore`
* Wiki extraction already done (`has_wiki=True` in store status)

If not, you’ll get:

```
FAIL — no files with wiki data
```

---

## CLI Usage

### Basic

```bash
uv run src/features/Retrival/Wiki/Main.py "ATP synthesis"
```

---

### Specify files

```bash
uv run src/features/Retrival/Wiki/Main.py "ATP synthesis" --file bio.pdf chem.pdf
```

---

### Top-K results

```bash
uv run src/features/Retrival/Wiki/Main.py "ATP synthesis" -k 10
```

---

### Gate presets

Strict (harder to satisfy):

```bash
uv run src/features/Retrival/Wiki/Main.py "ATP synthesis" --strict
```

Relaxed (easier to pass):

```bash
uv run src/features/Retrival/Wiki/Main.py "ATP synthesis" --relaxed
```

---

### Override gate manually

```bash
uv run src/features/Retrival/Wiki/Main.py "ATP synthesis" \
  --min-score 0.6 \
  --min-results 3 \
  --min-kw 2
```

---

### Inspection mode (debug)

Shows full scoring breakdown:

```bash
uv run src/features/Retrival/Wiki/Main.py "ATP synthesis" --inspect
```

---

### JSON output

```bash
uv run src/features/Retrival/Wiki/Main.py "ATP synthesis" --json
```

---

### Disable WebSocket

```bash
uv run src/features/Retrival/Wiki/Main.py "ATP synthesis" --no-ws
```

---

## CLI Arguments

| Flag            | Description           |
| --------------- | --------------------- |
| `query`         | Search string         |
| `--file`, `-f`  | Files to search       |
| `--top-k`, `-k` | Results per file      |
| `--strict`      | Strict gate config    |
| `--relaxed`     | Relaxed gate config   |
| `--min-score`   | Min top score         |
| `--min-results` | Min results per file  |
| `--min-kw`      | Min keyword hits      |
| `--inspect`     | Detailed debug output |
| `--json`        | JSON output           |
| `--no-ws`       | Disable websocket     |

---

## Programmatic Usage

### Basic

```python
from src.features.Retrival.WikiSearch import handle_wiki_search
from src.features.Retrival.Wiki.WikiSearchUtils import WikiSearchArgs, GATE_DEFAULT

args = WikiSearchArgs(
    query="ATP synthesis",
    files=["bio.pdf"],
    top_k=5,
    config=GATE_DEFAULT,
    inspect=False,
    json_output=False,
    ws_port=8765,
    no_ws=True,
)

response = handle_wiki_search(args)

print(response.satisfied)
```

---

### Access results

```python
for filename, file_result in response.file_results.items():
    print(filename, file_result.satisfied)

    for r in file_result.results:
        print(r.concept, r.score)
```

---

### JSON output programmatically

```python
from src.features.Retrival.WikiSearch import _format_wiki_results_json

json_str = _format_wiki_results_json(response)
print(json_str)
```

---

## Output Structure

### `WikiSearchResponse`

```python
response.satisfied          # overall bool
response.gate_reason        # string
response.file_results       # dict
```

---

### Per-file result

```python
file_result.satisfied
file_result.gate_reason
file_result.results
```

---

### Concept result

```python
r.concept
r.score
r.centroid_sim
r.keyword_hits
r.frequency
r.keywords
r.related
r.summary
r.chunk_indices
```

---

## Gating Logic

Each file must pass:

* `min_top_score`
* `min_results`
* `min_keyword_hit`

If any file fails → overall may fail depending on config.

---

## Common Failures

### 1. No wiki data

```
FAIL — no files with wiki data
```

Fix:

* Run wiki extraction first

---

### 2. No valid files

```
No valid files with wiki data. Search aborted.
```

Fix:

* Check filenames
* Ensure `has_wiki=True`

---

### 3. Unsatisfied results

```
Wiki search results unsatisfied
```

Fix:

* Use `--relaxed`
* Lower thresholds
* Improve query wording

---

## Debug Tips

Use this combo:

```bash
--inspect --relaxed --top-k 10
```

This shows:

* why files failed
* scoring breakdown
* keyword coverage

---

## Design Notes

* Deterministic scoring
* Works per-file, not global blending
* Gating prevents weak matches from leaking through

---
