# Remote HTTP / SSE Integration

## Start the server in network mode

```bash
# Streamable HTTP (recommended for production)
darwin mcp --http --host 0.0.0.0 --port 8765

# SSE transport (legacy)
darwin mcp --sse --host 0.0.0.0 --port 8765
```

## Client configuration

Point any MCP client to the network endpoint:

```json
{
  "mcpServers": {
    "darwin": {
      "url": "http://your-host:8765/mcp"
    }
  }
}
```

For SSE transport, use the `/sse` endpoint:

```json
{
  "mcpServers": {
    "darwin": {
      "url": "http://your-host:8765/sse"
    }
  }
}
```

## Generate automatically

```bash
# SSE config
darwin config sse --host 0.0.0.0 --port 8765

# HTTP config
darwin config http --host 0.0.0.0 --port 8765
```

## Docker deployment

```bash
docker run -d \
  --name darwin \
  -e DARWIN_BASE_DIR=/data \
  -e OPENAI_API_KEY=sk-... \
  -v darwin_data:/data \
  -p 8765:8765 \
  brightdotdev/darwin-rag
```

## Notes

- Use `0.0.0.0` to listen on all interfaces for network access.
- For production, add a reverse proxy (nginx, Caddy) with TLS.
- Persistent storage is required for models and indexes — mount a volume.
- Health check: `GET http://your-host:8765/health`
