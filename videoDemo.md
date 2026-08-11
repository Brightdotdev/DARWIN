# Darwin RAG — Hero Demo Video Script

Looping, silent, ~25s hero video for LinkedIn + portfolio, plus a short GIF for the repo README.

## Verified setup facts (from the codebase, Aug 2026)

- Install: `pip install darwin-rag`
- Setup: `darwin-admin setup --preset recommended` (one-shot, downloads models) or `darwin-admin setup interactive`
- Start server: `darwin mcp --http` (Streamable HTTP, default `http://127.0.0.1:8765`)
- Demo UI: `http://127.0.0.1:8765/demo` (interactive search)
- Log dashboard: `http://127.0.0.1:8765/dashboard` (live hybrid-search + rerank logs)
- Demo doc: `src/pdfs/cursor_paper.pdf` (Composer 2 Technical Report, 23pp) — indexed as store `933e9f27…`
- All 8 indexed files share the default store, so `/demo` searches everything
- **v0.2.16+ file filter:** `/demo` file cards are multi-select — click one (or more) and search runs only inside those files. This is how you keep `book_of_enoch.pdf` out of the shot.

## Tested hero query (verified against the live store)

```
What improvements does Composer 2 show over its base model Kimi K2.5?
```

Result (reranked hybrid search): top 2 hits are `cursor_paper.pdf` page 26 at
scores **0.855 / 0.844**; only one weak extra source. Snippet shown:

> "…these evaluations led us to select Kimi K2.5, a 1.04T parameter / 32B active parameter
> Mixture-of-Experts model as our base model for Composer 2."

Alternative if that feels off: `How is Composer 2 trained with reinforcement learning?`

## Loop structure (25s, 16:9, no audio)

**Frame 0 — the seam (first AND last frame identical, so the cut is invisible):**
Clean terminal, big font, no history above the prompt. Cursor blinking at an empty prompt.

| # | Time | Scene |
|---|------|-------|
| 1 | 0–6s | Run `darwin mcp --http`. Server boots and prints both URLs — `http://127.0.0.1:8765/dashboard` and `/demo`. |
| 2 | 6–14s | Cut to browser on `/demo`. Click the `cursor_paper.pdf` card (filter chip appears). Type the hero query, press Enter. |
| 3 | 14–21s | Results render — `cursor_paper.pdf` top hits with scores/highlights. Quick cut to `/dashboard` showing the live hybrid-search + rerank log lines. |
| 4 | 21–25s | Cut back to the exact terminal from Frame 0 → seamless loop. |

## How to record it, step by step

### Step 1 — Pre-flight (before any recording)
1. Boot the server: `darwin mcp --http` and confirm it prints both URLs.
2. Open `http://127.0.0.1:8765/demo` in a fresh browser tab.
3. Click the `cursor_paper.pdf` card, type the hero query, and confirm the top 2 results are the paper (if not, stop and re-index / re-verify — don't record a weak result).
4. Open `http://127.0.0.1:8765/dashboard` once so the log view is warmed up.
5. Close the browser tab. You'll reopen it fresh for Scene 2.

### Step 2 — Set up your recording surface
1. Terminal window: dark theme, **large font** (16pt+), a terminal theme with no prompt history above the prompt line. Hide anything noisy from your `PS1` (path, git branch, etc.).
2. Browser: use a **clean browser profile** — no extensions/toolbars visible. Zoom to ~110% so text is readable. Hide the mouse cursor (or keep it subtle and slow).
3. Screen recorder: 1080p+ or Retina, 30fps, **system audio off** (video is silent).
4. Decide the loop seam **first**: your terminal must be pixel-identical at the start and end of the clip, so make the terminal the first AND last thing you record.

### Step 3 — Record Scene 1 (terminal boot, 0–6s)
1. Frame the terminal, cursor blinking at an empty prompt. Let it sit still for **2 full seconds** (this becomes your loop seam).
2. Press Enter on `darwin mcp --http`.
3. Wait for the model-ready banner + `Dashboard: http://127.0.0.1:8765/dashboard` line.
4. Let it sit still **1 second**, then **stop recording**.

### Step 4 — Record Scene 2 (the filtered query, 6–14s)
1. New recording (or same file if your recorder supports pause/resume). Open `http://127.0.0.1:8765/demo` fresh.
2. Pause so the page renders. Then click the **`cursor_paper.pdf`** file card — watch for the ✓ highlight and the "Filtering: cursor_paper.pdf" chip.
3. Click into the search box and **type slowly** (each keystroke visible):
   `What improvements does Composer 2 show over its base model Kimi K2.5?`
4. Press Enter. Wait for the results to render.

### Step 5 — Record Scene 3 (results + dashboard, 14–21s)
1. Let the result cards render (top 2 should be `cursor_paper.pdf`, pp. 26).
2. Click/hold for ~2s, then open a second tab to `http://127.0.0.1:8765/dashboard` and show the live search/rerank log lines for 3–4s.
3. Stop recording.

### Step 6 — Record Scene 4 (the seam, 21–25s)
1. Switch back to the terminal from Scene 1 — it must still be at the prompt **exactly** as Scene 1 opened it (same cursor position, same text).
2. Record **2 seconds** of it sitting idle. This frame is identical to Scene 1's first frame.

### Step 7 — Cut the loop
1. Assemble in order: Scene 1 → 2 → 3 → 4.
2. Trim Scene 1's opening idle and Scene 4's idle so both are exactly the same duration, then cut from Scene 4's last frame straight back to Scene 1's first frame.
3. Because both frames are identical, the loop seam is invisible.
4. If your editor snaps to 25s and the cut looks abrupt, trim the idle in Scene 4 to match Scene 1's leading idle and re-export.
5. Export: silent, 16:9, H.264 (LinkedIn + portfolio), ~25s.

### Step 8 — Make the GIF (repo README)
1. Use the same scenes but keep only: boot → card click → query → results render (drop the dashboard and most idle time) → ~8–10s total.
2. Export at 12–15 fps, ~480px wide, no audio.
3. Filename something like `darwin-demo.gif`, commit to the repo root, and reference in README:
   `![Darwin demo](darwin-demo.gif)`

## GIF version (repo README)

Same loop trimmed to ~8–10s: boot → query → results → back to terminal.
12–15 fps, ~480px wide, muted (GitHub renders inline).

## Recording checklist

Run through once before you press record:

- [ ] Models ready + `darwin mcp --http` boots, prints `/dashboard` and `/demo` URLs
- [ ] Hero query returns the paper's pp. 26 passage first (re-run the verify search above)
- [ ] File filter works: clicking `cursor_paper.pdf` shows the "Filtering:" chip
- [ ] Terminal + browser framed, cursor hidden/slow, audio off, 30fps
- [ ] Loop seam frames (Scene 1 start / Scene 4 end) are pixel-identical

## Notes / gotchas

- The setup clip you recorded used `darwin --setupp --recomended` — that command
  does **not** exist. The real one is `darwin-admin setup --preset recommended`.
  Re-record if the setup clip appears in the video, or leave setup out entirely
  (the search loop is the hero).
- Biblical docs (book_of_enoch.pdf) and personal files (resumes, prompts) are in
  the same store — **use the file filter** to select only `cursor_paper.pdf` so
  they never appear in the shot. If you ingest more docs, re-verify the top
  results before recording.
- Reranking takes ~4s on CPU; `/dashboard` shows those latency numbers, which reads
  well ("hybrid + rerank, local, no cloud") — don't edit them out.
