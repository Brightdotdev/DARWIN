
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