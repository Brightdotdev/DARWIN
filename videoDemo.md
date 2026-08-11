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
| 2 | 6–14s | Cut to browser on `/demo`. Type the hero query, press Enter. |
| 3 | 14–21s | Results render — `cursor_paper.pdf` top hits with scores/highlights. Quick cut to `/dashboard` showing the live hybrid-search + rerank log lines. |
| 4 | 21–25s | Cut back to the exact terminal from Frame 0 → seamless loop. |

## GIF version (repo README)

Same loop trimmed to ~8–10s: boot → query → results → back to terminal.
12–15 fps, ~480px wide, muted (GitHub renders inline).

## Recording checklist

- [ ] Terminal: large font, dark theme, no prompt history above, hide PATH noise
- [ ] Browser: clean tab, `/demo` loaded fresh, mouse cursor hidden or subtle
- [ ] Confirm the store still returns the tested query first (re-run:
      `darwin-admin search "What improvements does Composer 2 show over its base model Kimi K2.5?" --rerank`)
- [ ] Type slowly enough that the query is readable on screen
- [ ] Keep scene 1 and scene 4 terminal frames pixel-identical

## Notes / gotchas

- The setup clip you recorded used `darwin --setupp --recomended` — that command
  does **not** exist. The real one is `darwin-admin setup --preset recommended`.
  Re-record if the setup clip appears in the video, or leave setup out entirely
  (the search loop is the hero).
- Biblical docs (book_of_enoch.pdf) and personal files (resumes, prompts) are in
  the same store — the tested query surfaces only the paper + one weak hit, but if
  you ingest more docs, re-verify the top results before recording.
- Reranking takes ~4s on CPU; `/dashboard` shows those latency numbers, which reads
  well ("hybrid + rerank, local, no cloud") — don't edit them out.
