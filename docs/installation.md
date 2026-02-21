---
title: Installation Guide
---

# Installation Guide

The ARM Reference MCP Suite is distributed via GitHub. It installs as a standard Python package and runs as one or more stdio-based MCP servers.

## Prerequisites

- **Python 3.10+**
- **uv** ([install guide](https://docs.astral.sh/uv/)) -- recommended, or use pip

---

## Claude Code

### Route 1: Standalone MCP servers

The fastest way to add individual servers. No permanent install -- `uvx` runs them in an isolated environment:

```bash
# ARM Register Reference (23 tools)
claude mcp add --transport stdio arm-reference -- \
  uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-reference-mcp

# ARM Documentation RAG (7 tools)
claude mcp add --transport stdio arm-docs-rag -- \
  uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-docs-rag-mcp

# ARM Cloud Migration Advisor (7 tools)
claude mcp add --transport stdio arm-cloud-migration -- \
  uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-cloud-migration-mcp

# ARM TinyML & Edge AI (7 tools)
claude mcp add --transport stdio arm-tinyml -- \
  uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-tinyml-mcp
```

Or install once with pip, then add the servers:

```bash
pip install "git+https://github.com/yerry262/arm-reference-mcp.git"

claude mcp add --transport stdio arm-reference -- arm-reference-mcp
claude mcp add --transport stdio arm-docs-rag -- arm-docs-rag-mcp
claude mcp add --transport stdio arm-cloud-migration -- arm-cloud-migration-mcp
claude mcp add --transport stdio arm-tinyml -- arm-tinyml-mcp
```

### Route 2: Claude Code plugin

Install all 4 servers at once through the plugin marketplace:

```
/plugin marketplace add yerry262/arm-reference-mcp
/plugin install arm-reference-mcp
```

This reads `.mcp.json` and `.claude-plugin/plugin.json` from the repo and auto-configures everything. No manual server setup needed.

---

## VS Code

Works with **GitHub Copilot Chat** (native MCP support) and [Continue.dev](https://continue.dev/).

Create or edit `.vscode/mcp.json` in your workspace:

```json
{
  "mcpServers": {
    "arm-reference": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/yerry262/arm-reference-mcp.git",
        "arm-reference-mcp"
      ]
    },
    "arm-docs-rag": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/yerry262/arm-reference-mcp.git",
        "arm-docs-rag-mcp"
      ]
    },
    "arm-cloud-migration": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/yerry262/arm-reference-mcp.git",
        "arm-cloud-migration-mcp"
      ]
    },
    "arm-tinyml": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/yerry262/arm-reference-mcp.git",
        "arm-tinyml-mcp"
      ]
    }
  }
}
```

Or add to your user `settings.json` under the `"mcp"` key for global availability.

---

## Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "arm-reference": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/yerry262/arm-reference-mcp.git",
        "arm-reference-mcp"
      ]
    }
  }
}
```

Restart Cursor after saving. Add additional servers using the same pattern with the other entry points (`arm-docs-rag-mcp`, `arm-cloud-migration-mcp`, `arm-tinyml-mcp`).

---

## Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "arm-reference": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/yerry262/arm-reference-mcp.git",
        "arm-reference-mcp"
      ]
    }
  }
}
```

Restart Windsurf to pick up the new server.

---

## OpenAI Codex CLI

Add to your Codex MCP configuration file (see [Codex CLI docs](https://github.com/openai/codex) for platform-specific paths):

```json
{
  "mcpServers": {
    "arm-reference": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/yerry262/arm-reference-mcp.git",
        "arm-reference-mcp"
      ]
    }
  }
}
```

---

## Generic / Other MCP Clients

Any MCP client that supports **stdio** transport can connect. The launch command:

```bash
uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-reference-mcp
```

Or install globally and run:

```bash
pip install "git+https://github.com/yerry262/arm-reference-mcp.git"
arm-reference-mcp
```

No HTTP server, no port configuration. Each server communicates over stdin/stdout.

### Available entry points

| Entry Point | Server |
|-------------|--------|
| `arm-reference-mcp` | ARM Register Reference (23 tools) |
| `arm-docs-rag-mcp` | ARM Documentation RAG (7 tools) |
| `arm-cloud-migration-mcp` | ARM Cloud Migration Advisor (7 tools) |
| `arm-tinyml-mcp` | ARM TinyML & Edge AI (7 tools) |

---

## Running from source

For development or if you want to modify the tools:

```bash
git clone https://github.com/yerry262/arm-reference-mcp.git
cd arm-reference-mcp
pip install -e .

# Run any server
arm-reference-mcp
python -m arm_reference_mcp.docs_rag_server
python -m arm_reference_mcp.cloud_migration_server
python -m arm_reference_mcp.tinyml_server
```

---

## Verifying installation

After adding a server, test it by asking your AI assistant a question:

- *"Look up the X0 register"* -- should trigger `lookup_register`
- *"Explain AArch64 exception levels"* -- should trigger `explain_exception_levels`
- *"Check if numpy works on ARM"* -- should trigger `scan_x86_dependencies`

If the tools don't appear, check that the server process starts cleanly:

```bash
uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-reference-mcp
```

You should see no output (the server is waiting for MCP messages on stdin). Press Ctrl+C to exit.
