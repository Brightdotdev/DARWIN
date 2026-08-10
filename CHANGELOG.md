# Changelog

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
