# Claude Desktop Integration

## Config Location

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

## Configuration

```json
{
  "mcpServers": {
    "darwin": {
      "command": "darwin",
      "args": ["mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## Generate automatically

```bash
# Prints the JSON (pipe it to your config file)
darwin config claude

# Or generate all client configs at once
darwin config all
```

## Notes

- Replace `OPENAI_API_KEY` with your preferred provider key (or add multiple).
- No LLM key is needed if you only want search/indexing.
- After editing the config, restart Claude Desktop.
- To switch to network mode, use `darwin config sse --host 127.0.0.1 --port 8765` instead.
