# ARM Register Reference MCP

A Model Context Protocol (MCP) server that provides quick-reference information for ARM registers across **AArch32** (ARMv7) and **AArch64** (ARMv8/v9) architectures.

## Tools

### `lookup_register`
Look up a specific register by name or alias.

```
lookup_register("CPSR")
lookup_register("X0", architecture="aarch64")
lookup_register("SP")  # returns both AArch32 and AArch64 entries
```

### `list_registers`
Browse registers by architecture and optional category.

```
list_registers("aarch64")
list_registers("aarch32", category="status")
list_registers("aarch64", category="general_purpose")
```

Categories: `general_purpose`, `status`, `system`, `floating_point`

## Setup

### Option A: Any MCP Client (VSCode, Claude Code, etc.)

Install and run with uvx (no install needed):

```bash
uvx arm-reference-mcp
```

Or install with pip:

```bash
pip install arm-reference-mcp
```

Then add to your MCP client config. For example, in Claude Code:

```bash
claude mcp add --transport stdio arm-reference -- uvx arm-reference-mcp
```

For VSCode, add to your MCP settings:

```json
{
  "mcpServers": {
    "arm-reference": {
      "type": "stdio",
      "command": "uvx",
      "args": ["arm-reference-mcp"]
    }
  }
}
```

### Option B: Claude Code Plugin (Marketplace)

Add this repo as a marketplace:

```
/plugin marketplace add <username>/arm-reference-mcp
```

Then install the plugin:

```
/plugin install arm-reference-mcp
```

### Option C: Local Development

```bash
git clone https://github.com/<username>/arm-reference-mcp.git
cd arm-reference-mcp
pip install -e .
arm-reference-mcp
```

## Register Coverage

### AArch32
- General purpose: R0-R12, R13/SP, R14/LR, R15/PC
- Status: CPSR, SPSR
- Floating point: S0-S31, D0-D31, FPSCR

### AArch64
- General purpose: X0-X30 (with W0-W30 aliases), SP, PC, XZR/WZR
- Status: NZCV, DAIF, CurrentEL
- Floating point: FPCR, FPSR
- System: VBAR_EL1, TTBR0_EL1, SCTLR_EL1

## License

MIT
