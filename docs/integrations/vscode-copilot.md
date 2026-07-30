# VS Code / GitHub Copilot Integration

## Prerequisites

- [VS Code MCP Extension](https://marketplace.visualstudio.com/items?itemName=Continue.continue) (Continue.dev) or
- [GitHub Copilot](https://github.com/features/copilot) with MCP support

## Config Location

Create `.vscode/mcp.json` in your project root:

```
your-project/
├── .vscode/
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

## Generate automatically

```bash
darwin config vscode
```

## Notes

- Replace `OPENAI_API_KEY` with your preferred provider key (or add multiple).
- The Continue extension must be installed and enabled.
- Restart VS Code after adding or changing the config.
- For GitHub Copilot's built-in MCP support, use the same JSON format in the VS Code settings.
