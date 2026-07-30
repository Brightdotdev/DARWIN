# Cursor Integration

## Config Location

Create `.cursor/mcp.json` in your project root:

```
your-project/
├── .cursor/
│   └── mcp.json
├── ... (your project files)
```

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

## Alternative: Cursor Settings UI

1. Open Cursor Settings → Features → MCP Servers
2. Click **+ Add New MCP Server**
3. Set **Name**: `darwin`
4. Set **Type**: `command`
5. Set **Command**: `darwin mcp`
6. Add environment variables as needed

## Generate automatically

```bash
darwin config cursor
```

## Notes

- Replace `OPENAI_API_KEY` with your preferred provider key (or add multiple).
- No LLM key is needed if you only want search/indexing.
- Restart Cursor or reload the window after adding the config.
