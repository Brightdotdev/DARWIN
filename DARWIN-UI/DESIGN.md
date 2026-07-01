# Darwin UI Design System
## "Soft Amber" — Research Terminal Aesthetic

---

## Philosophy

Darwin is a research tool built for sustained use. The UI should feel like a terminal that was refined over decades — not retro for nostalgia's sake, but because terminal aesthetics prioritize clarity, density, and zero distraction. The "Soft Amber" palette is warm, muted, and designed for hours of reading without eye fatigue.

> *"It should feel like a tool built in 1984, maintained by someone who cared in 2004, and refined by someone with taste in 2026."*

---

## Component Primitives

### Panel
```
┌ PANEL TITLE ───────────────────────────────────┐
│                                                │
│  Content goes here. Monospace for data.        │
│  Sans-serif for labels and descriptions.       │
│                                                │
└────────────────────────────────────────────────┘
```
- Background: `bg-surface`
- Border: `1px solid` `accent-amber-dim` (or ASCII `─│┌┐└┘`)
- Title: `text-lg`, `font-sans`, uppercase, letter-spacing `0.05em`
- Padding: `1.5rem`
- Gap between panels: `1rem`

### Button (Primary)
```
[ Run Pipeline ]
```
- Background: `accent-amber`
- Text: `text-inverse`, `font-sans`, `text-sm`, weight 600
- Padding: `0.5rem 1rem`
- Hover: background `accent-amber-dim`, text `text-primary`
- **No border. No ASCII box. Let the color speak.**

### Button (Secondary / Ghost)
```
[ Cancel ]
```
- Background: transparent
- Border: `1px solid` `text-dim`
- Text: `text-secondary`
- Hover: background `bg-surface-hover`, text `text-primary`

### Input Field
```
╭────────────────────────────────╮
│  search query...               │
╰────────────────────────────────╯
```
- Background: `bg-input`
- Border: `1px solid` `text-dim`
- Focus border: `accent-amber`
- Text: `text-primary`, `font-mono` for query content, `font-sans` for label
- Cursor: blinking block `▮`

### Table / Data Grid
```
  ID    │ Filename          │ Status    │ Score
────────┼───────────────────┼───────────┼───────
  001   │ doc_2024.pdf      │ indexed   │ 0.94
  002   │ notes.md          │ failed    │ 0.00
  003   │ paper_v2.txt      │ queued    │ —
```
- Header: `font-sans`, `text-xs`, uppercase, `text-secondary`
- Separator: `─` in `text-dim`
- Row hover: background `bg-active`
- Monospace for IDs, scores, filenames
- Sans-serif for status labels

### Progress / Status Line
```
Indexing  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░  67%  (14.2s)
```
- Label: `font-sans`, `text-sm`, `text-secondary`
- Bar: `▓` in `accent-amber`, `░` in `text-dim`
- Percentage: `font-mono`, `text-sm`, `text-primary`
- **One line. No box around it.**

### Tag / Pill
```
  [NER]  [PERSON]  [0.89]
```
- Background: `bg-active`
- Border: `1px solid` `accent-amber-dim`
- Text: `font-mono`, `text-xs`, `accent-amber`
- Padding: `0.25rem 0.5rem`

---

## Panel Layout: Darwin Pipeline Dashboard

3-column layout. Density varies by stage.

```
╭──────────────────────────────────────────────────────────────────────────────╮
│  DARWIN  v1.5                                              session: 0x7a3f    │
╰──────────────────────────────────────────────────────────────────────────────╯

┌ INGESTION ─────────────────┐  ┌ KNOWLEDGE GRAPH ───────────────┐  ┌ RETRIEVAL ───────────────┐
│                            │  │                                │  │                            │
│  Drop files or paste text  │  │  Entities extracted: 1,247     │  │  Query:                    │
│                            │  │                                │  │  ╭──────────────────────╮  │
│  [document.pdf]            │  │  ┌ Top Entities ─────────┐    │  │  │ "neural architecture"│  │
│    2.4 MB  ━━━━━━━━━━ 100% │  │  │                       │    │  │  ╰──────────────────────╯  │
│  [notes.md]                │  │  │  Einstein      89      │    │  │                            │
│    14 KB   ━━━━━━━━━━ 100% │  │  │  relativity    76      │    │  │  Results:                  │
│  [paper.txt]               │  │  │  photon        62      │    │  │                            │
│    89 KB   ▓▓▓▓▓▓▓░░░ 67%  │  │  │  ...                    │    │  │  ▸ doc_2024.pdf           │
│                            │  │  │                       │    │  │    score: 0.94             │
│  ▸ 3 queued                │  │  └───────────────────────┘    │  │    excerpt: "...neural..." │
│                            │  │                                │  │                            │
└────────────────────────────┘  │  Relationships: 3,891          │  │  ▸ notes.md                 │
                                │                                │  │    score: 0.87             │
┌ INDEXING ──────────────────┐  │  ┌ Recent Relations ─────┐    │  │                            │
│                            │  │  │                       │    │  │  [ Regenerate ]  [ Export ]│
│  Stage: embedding          │  │  │  Einstein → physics   │    │  │                            │
│  ───────────────────────── │  │  │  photon → wave        │    │  └────────────────────────────┘
│  Batch 12/48               │  │  │  ...                    │    │
│  ▓▓▓▓▓▓▓▓▓▓░░░░░░░░ 25%   │  │  │                       │    │
│  ETA: 4m 12s               │  │  └───────────────────────┘    │
│                            │  │                                │
│  [ Pause ]  [ Cancel ]     │  │  [ Explore Graph ]             │
│                            │  │                                │
└────────────────────────────┘  └────────────────────────────────┘

╭──────────────────────────────────────────────────────────────────────────────╮
│  GENERATION                                                                  │
│                                                                              │
│  Based on the retrieved documents, the neural architecture described in the   │
│  2024 paper represents a significant advancement over prior transformer     │
│  models. The key innovation lies in...                                       │
│                                                                              │
│  Sources:                                                                    │
│    ▸ doc_2024.pdf     [p. 3, 7, 12]                                         │
│    ▸ notes.md         [section 4]                                           │
│                                                                              │
│  [ Copy ]  [ Regenerate ]  [ Save to Session ]                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Layout Notes

| Region | Density | Treatment |
|--------|---------|-----------|
| **Header** | Low | Single line, session ID in `text-dim`, version in `accent-amber` |
| **Ingestion** | Medium | File list with inline progress. No boxes around individual files. |
| **Indexing** | Medium | Stage label + progress bar. Buttons at bottom, separated by space. |
| **Knowledge** | High | Two sub-panels inside one panel. Entity list is dense, monospace. |
| **Retrieval** | Medium | Query input prominent. Results list with excerpts. Scores in `accent-sage`. |
| **Generation** | Low | Maximum whitespace. Proportional font for readability. Sources as subtle tags. |

### Responsive Behavior
- **< 1200px**: 2 columns — Ingestion+Indexing stack, Knowledge+Retrieval stack, Generation full-width below
- **< 800px**: Single column stack. Panels become full-width cards.
- **Mobile**: Hide ASCII borders, use CSS borders with `accent-amber-dim`. Keep monospace for data.

---

## CSS Quick Start

```css
:root {
  --bg-base: #0d0d0d;
  --bg-surface: #141414;
  --bg-surface-hover: #1a1a1a;
  --bg-input: #111111;
  --bg-active: #1e1a12;

  --text-primary: #d4c5a9;
  --text-secondary: #8a7e6a;
  --text-dim: #5a5245;
  --text-inverse: #0d0d0d;

  --accent-amber: #c4953a;
  --accent-amber-dim: #8a6a28;
  --accent-amber-glow: rgba(196, 149, 58, 0.25);
  --accent-sage: #6b8a7a;
  --accent-sage-dim: #4a6054;
  --accent-rose: #d4574a;
  --accent-rose-dim: #8a3a32;

  --font-mono: "SF Mono", "Fira Code", "JetBrains Mono", monospace;
  --font-sans: "Inter", "SF Pro Display", system-ui, sans-serif;
}

body {
  background: var(--bg-base);
  color: var(--text-primary);
  font-family: var(--font-sans);
  line-height: 1.5;
}

::selection {
  background: var(--accent-amber-glow);
  color: var(--text-primary);
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--text-dim); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-amber-dim); }
```

---

## Design Principles

1. **One loud thing per screen.** If the border is heavy, the text is quiet. If the text is bold, the border is thin.
2. **Monospace = content.** Proportional = chrome. Your eye learns the difference instantly.
3. **Single-line ASCII only.** No double lines, no blocks as borders. Reserve `▓` for progress.
4. **Amber is for action, sage is for confidence, rose is for errors.** Everything else is warm gray.
5. **Density shifts with the pipeline stage.** Ingestion is sparse, indexing is dense, generation breathes.
6. **A 1984 terminal should render the core layout.** If it needs CSS grid and blur to communicate, simplify.
7. **Whitespace is not emptiness — it's dead air for the eye.** Use it generously in generation, sparingly in indexing.
