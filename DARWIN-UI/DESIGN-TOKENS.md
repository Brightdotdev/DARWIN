# Darwin Design Tokens — UNBREAKABLE

## Colors
| Token | Hex | Usage |
|-------|-----|-------|
| bg-base | `#0d0d0d` | Root background |
| bg-surface | `#141414` | Panels, cards |
| bg-surface-hover | `#1a1a1a` | Hover state |
| bg-input | `#111111` | Input fields |
| bg-active | `#1e1a12` | Active/selected (subtle amber tint) |
| text-primary | `#d4c5a9` | Body text, labels |
| text-secondary | `#8a7e6a` | Metadata, timestamps |
| text-dim | `#5a5245` | Disabled, placeholders, borders |
| text-inverse | `#0d0d0d` | Text on amber backgrounds |
| accent-amber | `#c4953a` | Primary actions, active nav, highlights |
| accent-amber-dim | `#8a6a28` | Secondary amber, dividers |
| accent-amber-glow | `rgba(196,149,58,0.25)` | Subtle glow/backdrop |
| accent-sage | `#6b8a7a` | Success, confidence scores |
| accent-sage-dim | `#4a6054` | Secondary sage |
| accent-rose | `#d4574a` | Errors, alerts |
| accent-rose-dim | `#8a3a32` | Secondary rose |

## Typography
| Token | Value | Rule |
|-------|-------|------|
| font-mono | `"SF Mono", "Fira Code", "JetBrains Mono", monospace` | **DATA ONLY** — tables, code, entity names, scores |
| font-sans | `"Inter", "SF Pro Display", system-ui, sans-serif` | **UI CHROME ONLY** — labels, buttons, nav, headers |
| Headers | uppercase, letter-spacing `0.05em`, sans-serif | Panel titles, section labels |
| Body | `1rem / 1.5`, sans-serif | Descriptions, generated text |
| Data | `0.875rem / 1.4`, monospace | IDs, scores, filenames, logs |

## Spacing Scale
`0.25` / `0.5` / `0.75` / `1` / `1.5` / `2` / `3` / `4` rem

## ASCII Rules — SINGLE LINE ONLY
| OK | NEVER |
|----|-------|
| `─ │ ┌ ┐ └ ┘ ├ ┤ ┬ ┴ ┼` | `═ ║ ╔ ╗ ╚ ╝ ╠ ╣ ╦ ╩ ╬` |
| `╭ ╮ ╰ ╯` (soft corners, rare) | `█ ▀ ▄` for borders |
| `▸ ▾ ▮ › … → ✓` (decorative) | `█ ▀ ▄` anywhere except progress bars |

- **One loud thing per screen.** Heavy border = quiet text. Bold text = thin border.
- Reserve `▓` and `░` for **progress bars only**.
- Borders: `1px solid` with `accent-amber-dim` or `text-dim`. Never `accent-amber`.

## Density by Stage
| Stage | Mood | Density | Key Visual |
|-------|------|---------|------------|
| Ingestion | Anticipatory, quiet | Medium | Empty drop zone, dashed border |
| Indexing | Busy, technical | High | Progress bars, batch counters, monospace logs |
| Knowledge | Dense, connected | High | Graph nodes as tags, relationship arrows |
| Retrieval | Focused, searching | Medium | Prominent query input, ranked list |
| Generation | Calm, readable | Low | Generous line-height (`1.75`), minimal chrome |

## Selection & Scrollbar
- `::selection`: `accent-amber-glow` background, `text-primary` color
- Scrollbar: `6px` width, `text-dim` thumb, `bg-base` track
