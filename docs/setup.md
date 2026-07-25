# Setup

Three ways to get Darwin running. Pick whichever fits your workflow.

---

## 1. Install from PyPI

```bash
pip install darwin-rag
# or
uv add darwin-rag
```

Then run the interactive setup:

```bash
darwin-admin setup interactive
```

This detects your hardware, picks the best models for your system, downloads them, and configures logging. Takes about 2 minutes.

After setup, start the MCP server:

```bash
darwin mcp                     # stdio (for Claude Desktop, Cursor, etc.)
darwin mcp --http --port 8765  # HTTP (for remote agents, Docker, browser)
```

### Optional: spaCy NLP support

spaCy is an optional extra for NLP features (entity recognition, summarization):

```bash
pip install "darwin-rag[nlp]"
```

This is **not required** for core RAG functionality (search, pipeline, store).

---

## 2. From source

```bash
git clone https://github.com/Brightdotdev/DARWIN.git
cd DARWIN
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
darwin-admin setup interactive
```

---

## 3. Docker

```bash
docker compose up -d
```

The container auto-runs setup on first start. MCP is exposed on port 8080.

### Connect your agent

```bash
# Generate config for Claude Desktop
darwin config claude

# Generate config for Cursor
darwin config cursor

# Or point any MCP client to:
# SSE:  http://localhost:8080/sse
# HTTP: http://localhost:8080/mcp
```

### Dashboard

Open `http://localhost:8080/dashboard` in your browser to watch logs in real time.

### Networking

By default, the MCP server is bound to `127.0.0.1:8080` — accessible only from the host machine. This is the right setup when your AI agent (Claude Desktop, Cursor, etc.) runs on the same machine as Docker.

If your agent runs in a **different container**, you need container-to-container networking. The compose file includes a `darwin-net` bridge network for this:

```bash
docker compose --profile network up -d
```

Then connect your agent container to the same network and use `http://darwin:8080/sse` or `http://darwin:8080/mcp` as the endpoint — Docker resolves the service name automatically.

If you need the MCP server accessible from other machines on your network, change the port binding in `docker-compose.yml`:

```yaml
ports:
  - "0.0.0.0:8080:8080"  # accessible from any machine on the network
```

---

## What setup does

1. **Detects hardware** — RAM, disk, CPU, GPU → assigns a tier (light/medium/large/overkill)
2. **Picks models** — Best embedding, reranker, OCR, and vision models for your hardware
3. **Downloads models** — Local ONNX models (no API keys needed for core search)
4. **Configures logging** — Terminal output, SSE streaming, retention

## After setup

```bash
# Ingest documents
darwin-admin pipeline run /path/to/docs

# Search
darwin-admin search "your query"

# Start MCP server for AI agents
darwin mcp --http --port 8765
```

## Configuration files

All config lives in your platform's data directory:

| OS | Path |
|---|---|
| Windows | `%LOCALAPPDATA%/Darwin/config/` |
| macOS | `~/Library/Application Support/Darwin/config/` |
| Linux | `~/.local/share/darwin/config/` |

| File | Purpose |
|---|---|
| `darwin.json` | Models, API keys, environment |
| `logging.json` | Log mode, SSE port, retention |
| `models.json` | Active model per category |
