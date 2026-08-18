# Changelog

## v0.2.17 — 2026-08-18

Structured data indexing and search.

### What's new
- **Structured list indexing** — CSVs, JSON arrays, and spreadsheets are now auto-detected and indexed as structured records (one chunk per row/record). Each record's fields are stored as a Polars struct for native filtering.
- **`search_lists` MCP tool** — filter structured data by field values with 10 operators (`exact`, `__contains`, `__startswith`, `__endswith`, `__gt`, `__lt`, `__gte`, `__lte`, `__neq`, `__in`, `__exists`). Supports nested dot-notation fields, case-insensitive matching, and hybrid text fallback.
- **`get_schema` MCP tool** — inspect schemas for structured files (keys, types, record counts). Four modes: single file, multiple files (overlapping key detection), list all, validate filters.
- **`RetrievalOrchestrator.search_lists()`** — orchestrator-backed structured search with pre-filter validation, re-ranking, and session logging.
- **Admin CLI `search-lists` subcommand** — `--query`, `--filters`, `--mode`, `--files`, `--top-k`, `--filter-boost-factor`, `--rerank`, `--validate-filters`, `--list-all`.
- **`--index-mode` flag** on `pipeline run` — force `text` or `structured` indexing (default `auto`).
- **Filter validation guardrails** — filter keys validated against schemas before searching; overlapping keys across files detected; type mismatches warned.

### Fixes
- **BOM handling** — CSV/JSON parsers now use `utf-8-sig` to strip byte-order marks from Excel-exported files.
- **Polars struct extraction** — `StructuredSearch` now uses `.struct.field()` instead of `str.json_path_match()` for native struct column access.
- **Exact match semantics** — filter `exact` operator now uses `==` instead of `str.contains()`, preventing false positives.

### Docs
- Added full documentation for `search_lists` and `get_schema` in `docs/mcp/search.md`.
- Updated tool counts (9 → 11) in `README.md`, `docs/mcp.md`, `docs/architecture.md`.

## v0.2.16 — 2026-08-11

Demo UI release.

### What's new
- **File filtering in the demo UI** — `/demo` file cards are now a real multi-select: pick one or more indexed documents and search only those. Active filters show as chips with per-file removal and a "Clear all".
- **`/api/search` supports `files`** — comma-separated `?files=` param restricts retrieval to the named documents (backed by `RetrievalParameters.files`), so the same filtering works for any client hitting the HTTP API, not just the web UI.
- **`/api/search` response includes the active `files` scope** so callers can confirm the filter applied.

## v0.2.15 — 2026-08-08

Discoverability release.

### What's new
- **PyPI search optimization** — added `keywords` (rag, knowledge-base, mcp, bm25, hybrid-search, local-ai, offline, …), Information Retrieval classifiers, and a keyword-rich summary so Darwin surfaces for "local RAG", "MCP knowledge base", and related searches.
- **MCP registry republished** at 0.2.15 with the updated metadata.

## v0.2.14 — 2026-08-08

Stability and accuracy release.

### What's new
- **Honest config layout** — `darwin.json` is now written by `darwin-admin setup` as a setup-completion record. `setup validate` no longer reports it as missing on fresh installs (nothing was ever writing it before).
- **Docs accuracy** — setup/admin/architecture docs now state where settings really live: `models.json` for model selections, `logging.json` for logging, and the OS keychain for API keys. Removed the incorrect claim that `darwin.json` stores models/keys/environment.
- **MCP registry manifest cleanup** — removed the placeholder `https://your-host.com/mcp` remotes entry from `server.json`; the registry entry is now pip stdio via `darwin mcp`.
- **setup.md** — marked spaCy NLP support as experimental; clarified Docker networking wording.

## v0.2.13 — 2026-07-29

MCP Registry release.

### What's new
- **MCP Registry publication** — Listed on the official MCP Registry at `io.github.Brightdotdev/darwin-rag`
- **9 MCP tools** — `search_darwin`, `get_search_results`, `get_search_result_by_id`, `run_pipeline`, `purge_artifacts`, `create_store`, `list_files`, `file_status`, `get_logs`
- **Three transports** — stdio (AI agent subprocess), SSE (legacy), Streamable HTTP (recommended)
- **Hybrid search** — BM25 keyword + dense embedding retrieval with reranking, diversity rerank, structural penalties
- **Multi-format ingestion** — PDF, Markdown, HTML, images (OCR), CSV, Excel, ODS, URLs
- **Observability** — Structured logging with per-session SQLite history, queryable via MCP
- **Multi-store isolation** — Independent data stores for tenant/project separation
- **Docker support** — Hardened multi-stage build with auto-setup on first run
- **Web demo UI** — Dark-themed search interface at `/demo` with file browser
- **Deployment configs** — Dockerfile, Railway, and Render one-click deployments
- **Client integration** — Claude Desktop, Cursor, VS Code config generation via `darwin config`
- **Local-first** — Fully offline-capable after initial model download
