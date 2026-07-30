# Changelog

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
