# ARM Register Reference MCP

A [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that gives AI assistants instant access to ARM architecture reference data. It covers **AArch32** (ARMv7) and **AArch64** (ARMv8/v9) with 17 tools spanning registers, instruction decoding, condition codes, calling conventions, exception levels, security models, page tables, memory attributes, architecture extensions, assembly patterns, memory barriers, and core/IP reference -- all without leaving your editor or CLI.

Built in Python. Works with any MCP-compatible client over stdio transport.

---

## Tools

The server exposes **17 tools** organized into five categories:

### Register tools

| Tool | Description | Example Input |
|------|-------------|---------------|
| `lookup_register` | Detailed info on a specific register by name or alias | `lookup_register("CPSR")`, `lookup_register("X0", architecture="aarch64")` |
| `list_registers` | Browse registers by architecture and optional category | `list_registers("aarch64", category="system")` |
| `search_registers` | Keyword search across all register data | `search_registers("cache")`, `search_registers("stack", architecture="aarch32")` |
| `decode_register_value` | Decode a hex value against a register's named bit fields | `decode_register_value("CPSR", "0x600001D3")` |

### Instruction & convention tools

| Tool | Description | Example Input |
|------|-------------|---------------|
| `decode_instruction` | Decode a 32-bit AArch32 instruction from hex encoding | `decode_instruction("0xE3A01005")` |
| `explain_condition_code` | Explain an ARM condition code suffix (EQ, NE, GT, etc.) | `explain_condition_code("GE")` |
| `explain_calling_convention` | Full AAPCS32/AAPCS64 calling convention reference | `explain_calling_convention("aarch64")` |

### System architecture tools

| Tool | Description | Example Input |
|------|-------------|---------------|
| `explain_exception_levels` | EL0-EL3 exception levels (AArch64) or processor modes (AArch32) | `explain_exception_levels("aarch64")` |
| `explain_security_model` | TrustZone, RME, Arm CCA security model reference | `explain_security_model("aarch64")` |
| `explain_page_table_format` | AArch64 page table translation for a given granule/VA size | `explain_page_table_format("4KB", va_bits=48)` |
| `explain_memory_attributes` | Memory attributes: cacheability, shareability, permissions | `explain_memory_attributes("device")` |

### Architecture & extensions tools

| Tool | Description | Example Input |
|------|-------------|---------------|
| `explain_extension` | Detailed reference for an ARM architecture extension | `explain_extension("SVE")`, `explain_extension("MTE")` |
| `compare_architecture_versions` | Features for an ARMv8/v9 version, or compare two versions | `compare_architecture_versions("armv9.0", compare_to="armv8.0")` |

### Core/IP & programming tools

| Tool | Description | Example Input |
|------|-------------|---------------|
| `lookup_core` | ARM core/IP reference card (Cortex-A/R/M/X, Neoverse) | `lookup_core("Cortex-A78")`, `lookup_core("N2")` |
| `compare_cores` | Side-by-side comparison of two ARM cores | `compare_cores("Cortex-A78", "Cortex-X4")` |
| `show_assembly_pattern` | Annotated assembly for common ARM patterns | `show_assembly_pattern("spinlock_acquire", architecture="aarch64")` |
| `explain_barrier` | Memory barrier and synchronization instruction reference | `explain_barrier("DMB")`, `explain_barrier("LDAR")` |

### Tool details

**`lookup_register(name, architecture?)`** -- Returns the register's bit width, architecture, category, description, bit field layout (if any), and usage notes. Supports aliases (e.g., `FP`, `LR`, `IP0`, `WZR`). If `architecture` is omitted, matches from both AArch32 and AArch64 are returned.

**`list_registers(architecture, category?)`** -- Lists all registers for a given architecture in a table format. Categories: `general_purpose`, `status`, `system`, `floating_point`. Omit `category` to list everything.

**`search_registers(query, architecture?)`** -- Case-insensitive keyword search across register names, aliases, descriptions, and usage notes. Returns matching registers with context snippets showing where the keyword was found.

**`decode_register_value(register_name, hex_value, architecture?)`** -- Takes a hex value and a register name (e.g., `CPSR`, `SCTLR_EL1`, `FPSCR`, `DAIF`), then extracts each named bit field from the value and displays its decoded contents. Useful for interpreting raw register dumps from debuggers.

**`decode_instruction(hex_value)`** -- Parses a 32-bit AArch32 instruction encoding and breaks it into fields: condition code, instruction type (data processing, branch, load/store, etc.), opcode, registers, immediates, and shift operands. Accepts hex strings with or without `0x` prefix.

**`explain_condition_code(suffix)`** -- Given a condition code mnemonic like `EQ`, `NE`, `CS`, `HI`, `GT`, or `AL`, returns the full name, which NZCV flags are tested, the flag condition expression, the opposite condition, and a usage example.

**`explain_calling_convention(architecture)`** -- Returns a comprehensive AAPCS reference for `aarch32` or `aarch64`, covering argument registers, return value registers, caller-saved vs callee-saved registers, stack alignment requirements, frame pointer convention, and special notes (e.g., PAC, platform register, HFA rules).

**`explain_exception_levels(architecture?)`** -- Describes EL0-EL3 exception levels for AArch64 (with registers, typical software, and transition mechanisms at each level) or the seven processor modes for AArch32 (User, FIQ, IRQ, Supervisor, Abort, Undefined, System). Defaults to AArch64.

**`explain_security_model(architecture)`** -- Reference for ARM's security architecture: TrustZone (Secure/Non-secure worlds), RME (Realm Management Extension, four worlds), and Arm CCA (Confidential Compute Architecture). Includes hardware enforcement, transition mechanisms, and memory partitioning.

**`explain_page_table_format(granule_size, va_bits?)`** -- Shows the AArch64 page table translation scheme for a given granule (4KB, 16KB, or 64KB) and VA width (39, 48, or 52 bits). Includes level structure, VA bit layout diagram, PTE field layouts, OA ranges, and TCR_EL1 configuration fields.

**`explain_memory_attributes(topic?)`** -- Detailed reference on ARM memory attributes. Topics: `"cacheability"`, `"shareability"`, `"device"`, `"permissions"`, `"mair"`, `"stage2"`. Omit for an overview of all concepts.

**`explain_extension(extension_name)`** -- Returns the full name, introducing architecture version, purpose, key registers and instructions, detection method, and practical use cases for 17 ARM extensions including SVE, MTE, PAC, BTI, TME, RME, SME, GCS, FEAT_THE, FEAT_NV2, DIT, MPAM, RAS, SPE, AMU, and BRBE.

**`compare_architecture_versions(version, compare_to?)`** -- Lists mandatory and optional features for an architecture version (ARMv8.0-A through ARMv9.5-A). If `compare_to` is provided, shows a side-by-side diff of all features added between the two versions.

**`lookup_core(core_name)`** -- Returns a detailed reference card for an ARM core/IP: architecture version, pipeline details, decode width, key features, target market, and notable SoCs. Covers Cortex-A, Cortex-X, Cortex-R, Cortex-M, and Neoverse series. Accepts short forms like "A78", "X4", "M55", "N2".

**`compare_cores(core_a, core_b)`** -- Shows a side-by-side comparison table of two ARM cores covering architecture, pipeline, decode width, features, market, and generation.

**`show_assembly_pattern(pattern_name, architecture?)`** -- Returns annotated assembly code for 12 common ARM patterns: function_prologue, function_epilogue, atomic_add, atomic_cas, spinlock_acquire, spinlock_release, context_switch, syscall, tlb_invalidate, cache_clean, enable_mmu, exception_vector. Available for both AArch32 and AArch64.

**`explain_barrier(barrier_type)`** -- Explains an ARM barrier or synchronization instruction: DMB, DSB, ISB, LDAR, STLR, LDAPR, CAS/CASA/CASAL, SB, CSDB, SSBB/PSSBB. Covers ordering semantics, domain options, acquire/release patterns, and Spectre mitigation barriers. Pass `"overview"` for a summary of all barrier types.

---

## Setup / Installation

### Prerequisites

You need Python 3.10+ and either [uv](https://docs.astral.sh/uv/) (recommended) or pip installed.

---

### a. Claude Code (CLI)

**Route 1: Add as a standalone MCP server**

Using `uvx` (no permanent install required):

```bash
claude mcp add --transport stdio arm-reference -- uvx arm-reference-mcp
```

Or, if you have cloned the repo locally and want to run from source:

```bash
claude mcp add --transport stdio arm-reference -- python -m arm_reference_mcp.server
```

After adding, all 17 tools are immediately available in your Claude Code session. Ask Claude about ARM registers, instructions, calling conventions, exception levels, page tables, memory barriers, core comparisons, and more -- it will call the tools automatically.

**Route 2: Add as a plugin via Claude Code marketplace**

First, add this repository as a marketplace source:

```
/plugin marketplace add yerry262/arm-reference-mcp
```

Then install the plugin:

```
/plugin install arm-reference-mcp
```

This auto-configures the MCP server for you based on the `.mcp.json` and `.claude-plugin/plugin.json` files in the repo. No manual server configuration needed.

---

### b. VS Code (GitHub Copilot / Continue.dev)

Recent versions of VS Code support MCP servers natively through GitHub Copilot Chat. You can also use them with the [Continue.dev](https://continue.dev/) extension.

Create or edit `.vscode/mcp.json` in your workspace root:

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

Alternatively, add it to your VS Code user settings JSON (`settings.json`):

```json
{
  "mcp": {
    "servers": {
      "arm-reference": {
        "type": "stdio",
        "command": "uvx",
        "args": ["arm-reference-mcp"]
      }
    }
  }
}
```

Once configured, the tools appear in Copilot Chat or Continue.dev as available MCP tools.

---

### c. Cursor

Add to `.cursor/mcp.json` in your project root, or configure in Cursor's global MCP settings:

```json
{
  "mcpServers": {
    "arm-reference": {
      "command": "uvx",
      "args": ["arm-reference-mcp"]
    }
  }
}
```

Restart Cursor after saving. The tools will be available in Cursor's AI chat.

---

### d. Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "arm-reference": {
      "command": "uvx",
      "args": ["arm-reference-mcp"]
    }
  }
}
```

Restart Windsurf to pick up the new server.

---

### e. OpenAI Codex CLI

Codex CLI supports MCP servers via its configuration. Add the server to your Codex MCP config:

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

Refer to the [Codex CLI documentation](https://github.com/openai/codex) for the exact config file location on your platform.

---

### f. Generic / Other MCP Clients

Any MCP-compatible client can connect to this server over **stdio** transport. The command to launch the server is:

```bash
uvx arm-reference-mcp
```

Or, if installed via pip:

```bash
arm-reference-mcp
```

Or, run directly from a cloned checkout:

```bash
python -m arm_reference_mcp.server
```

Point your client at whichever command above works for your setup. The transport is stdio (stdin/stdout). No HTTP server, no port configuration.

---

## Local Development

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yerry262/arm-reference-mcp.git
cd arm-reference-mcp
pip install -e .
```

Run the server directly:

```bash
arm-reference-mcp
```

Or via the module entry point:

```bash
python -m arm_reference_mcp.server
```

To add it to Claude Code for testing during development:

```bash
claude mcp add --transport stdio arm-reference -- python -m arm_reference_mcp.server
```

### Project structure

```
arm-reference-mcp/
  src/
    arm_reference_mcp/
      __init__.py
      server.py          # MCP server with all 17 tools and inline reference data
      data.py            # Register definitions for AArch32 and AArch64
  tests/
    test_tools.py        # 84 test cases covering all 17 tools
  .claude-plugin/
    plugin.json          # Claude Code plugin metadata
    marketplace.json     # Plugin marketplace manifest
  .mcp.json              # Default MCP server config (used by plugins)
  pyproject.toml         # Package metadata and build config
  README.md
```

### Running tests

```bash
# With the built-in runner (no dependencies beyond the package itself)
python tests/test_tools.py

# Or with pytest
python -m pytest tests/test_tools.py -v
```

### Dependencies

- Python >= 3.10
- `mcp[cli]` >= 1.0.0 (the Model Context Protocol SDK)

---

## Register Coverage

### AArch32 (ARMv7)

| Category | Registers |
|----------|-----------|
| General purpose | R0-R12, R13/SP, R14/LR, R15/PC |
| Status | CPSR (with full bit field decode: N, Z, C, V, Q, J, E, A, I, F, T, M), SPSR |
| Floating point | S0-S31, D0-D31, FPSCR (with bit field decode: N, Z, C, V, RMode, DN, FZ) |

### AArch64 (ARMv8/v9)

| Category | Registers |
|----------|-----------|
| General purpose | X0-X30 (with W0-W30 aliases), SP/WSP, PC, XZR/WZR |
| Status | NZCV (with bit field decode), DAIF (D, A, I, F masks), CurrentEL |
| Floating point | FPCR (DN, FZ, RMode), FPSR (N, Z, C, V, IDC, IXC, UFC, OFC, IOC) |
| System | VBAR_EL1, TTBR0_EL1 (ASID, BADDR, CnP), SCTLR_EL1 (M, C, I, A, EE) |

Registers with defined bit fields support the `decode_register_value` tool for interpreting raw hex dumps.

### Additional reference data

Beyond registers, the server includes:

- **AArch32 instruction decoding** -- data processing, branch, load/store, SWI, with full field breakdowns
- **All 15 ARM condition codes** -- EQ, NE, CS/HS, CC/LO, MI, PL, VS, VC, HI, LS, GE, LT, GT, LE, AL
- **AAPCS32 and AAPCS64 calling conventions** -- argument/return registers, caller/callee-saved sets, stack alignment, frame pointer, platform-specific notes, HFA/HVA rules, PAC
- **Exception levels** -- EL0-EL3 with registers, typical software, and transition mechanisms (AArch64); seven processor modes (AArch32)
- **Security models** -- TrustZone (Secure/Non-secure), RME (four worlds), Arm CCA
- **Page table translation** -- 4KB/16KB/64KB granules, multi-level translation, VA layout diagrams, PTE fields, TCR_EL1 config
- **Memory attributes** -- MAIR, cacheability policies, device types, shareability domains, access permissions, Stage 1/2 combination
- **17 architecture extensions** -- SVE, MTE, PAC, BTI, TME, RME, SME, GCS, FEAT_THE, FEAT_NV2, DIT, MPAM, RAS, SPE, AMU, BRBE, with detection methods and use cases
- **Architecture versions** -- ARMv8.0-A through ARMv9.5-A with mandatory/optional features and example cores
- **ARM core/IP reference** -- Cortex-A/R/M/X and Neoverse series with pipeline details, features, and comparison
- **12 assembly patterns** -- function prologues, atomics, spinlocks, context switches, syscalls, TLB/cache maintenance, MMU enable, exception vectors (both AArch32 and AArch64)
- **10 memory barriers** -- DMB, DSB, ISB, LDAR, STLR, LDAPR, CAS variants, SB, CSDB, SSBB/PSSBB with domain options and Spectre mitigation

---

## License

MIT
