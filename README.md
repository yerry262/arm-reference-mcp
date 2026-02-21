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

This package is distributed via GitHub (not PyPI). All install methods below pull directly from the repository.

---

### a. Claude Code (CLI)

**Route 1: Add as a standalone MCP server**

Using `uvx` to install directly from GitHub (no permanent install required):

```bash
claude mcp add --transport stdio arm-reference -- uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-reference-mcp
```

Or, install with pip first, then add the server:

```bash
pip install "git+https://github.com/yerry262/arm-reference-mcp.git"
claude mcp add --transport stdio arm-reference -- arm-reference-mcp
```

Or, if you have cloned the repo locally and want to run from source:

```bash
git clone https://github.com/yerry262/arm-reference-mcp.git
cd arm-reference-mcp
pip install -e .
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
      "args": ["--from", "git+https://github.com/yerry262/arm-reference-mcp.git", "arm-reference-mcp"]
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
        "args": ["--from", "git+https://github.com/yerry262/arm-reference-mcp.git", "arm-reference-mcp"]
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
      "args": ["--from", "git+https://github.com/yerry262/arm-reference-mcp.git", "arm-reference-mcp"]
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
      "args": ["--from", "git+https://github.com/yerry262/arm-reference-mcp.git", "arm-reference-mcp"]
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
      "args": ["--from", "git+https://github.com/yerry262/arm-reference-mcp.git", "arm-reference-mcp"]
    }
  }
}
```

Refer to the [Codex CLI documentation](https://github.com/openai/codex) for the exact config file location on your platform.

---

### f. Generic / Other MCP Clients

Any MCP-compatible client can connect to this server over **stdio** transport. The command to launch the server is:

```bash
uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-reference-mcp
```

Or, install with pip and run directly:

```bash
pip install "git+https://github.com/yerry262/arm-reference-mcp.git"
arm-reference-mcp
```

Or, run from a cloned checkout:

```bash
git clone https://github.com/yerry262/arm-reference-mcp.git
cd arm-reference-mcp
pip install -e .
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

## Summary of Each Tool

1. **`lookup_register`** (`X0`)

   Returns register details: architecture (aarch64), size (64-bit), aliases (W0), category (general_purpose), description, and usage notes (caller-saved, W0 zero-extends into X0).

2. **`list_registers`** (`aarch64`, `general_purpose`)

   Lists all 34 general-purpose AArch64 registers (X0-X30, SP, PC, XZR) in a table with width, aliases, and description.

3. **`decode_instruction`** (`0xE3A01005`)

   Decodes an AArch32 instruction into fields: condition code (AL), instruction type (Data Processing), mnemonic (MOV), I bit, opcode, S flag, Rn, Rd (R1), and immediate operand (5). Result: `MOV R1, #5`.

4. **`explain_condition_code`** (`EQ`)

   Returns full name (Equal), encoding (0b0000), flags tested (Z), flag condition (Z==1), opposite (NE), and an example with CMP/BEQ.

5. **`explain_calling_convention`** (`aarch64`)

   Comprehensive AAPCS64 reference: argument registers (X0-X7), return value registers (X0-X1, V0-V3), caller/callee-saved registers, stack alignment (16 bytes), frame pointer (X29), and special notes on X8, X16/X17, X18, PAC.

6. **`search_registers`** (`"stack"`)

   Found 3 matches across both architectures: R13/SP (aarch32), X29/FP (aarch64), SP (aarch64).

7. **`decode_register_value`** (`CPSR`, `0x600001D3`)

   Decoded all bit fields: N=0, Z=1, C=1, V=0, Q=0, J=0, E=0 (little-endian), A=1 (abort masked), I=1 (IRQ masked), F=1 (FIQ masked), T=0 (ARM state), M=0x13 (SVC mode).

8. **`explain_exception_levels`** (`aarch64`)

   Detailed reference on EL0-EL3: what runs at each level, accessible registers, stack pointer selection, key registers (SPSR, ELR, VBAR, ESR, HCR, SCR), exception entry/return mechanism, vector table layout (16 entries with offsets), SPSel, and all four security states (NS, Secure, Realm, Root).

9. **`explain_security_model`** (`aarch64`)

   Covers TrustZone (SCR_EL3.NS bit, world-switching mechanism), Secure ELs (S-EL0/1/2), and ARMv9 RME (four-world model, GPT structure, GPC checks, CCA architecture with RMM).

10. **`lookup_core`** (`Cortex-A78`)

    Returns: ARMv8.2-A, Hercules generation, 13-stage OoO pipeline, 4-wide decode, key features (FP16, dot-product, L1/L2/L3 sizes), target market (mobile), notable SoCs (Snapdragon 888, Exynos 2100).

11. **`compare_cores`** (`Cortex-A78` vs `Cortex-X4`)

    Side-by-side comparison table: architecture (v8.2 vs v9.2), decode width (4 vs 6), generation, features, and SoCs. Highlights the generational leap.

12. **`explain_page_table_format`** (`4KB`, 48-bit VA)

    Full translation scheme: 4 levels (L0-L3), VA bit layout diagram, descriptor types at each level (block/table/page), PTE field layout (UXN, PXN, AF, SH, AP, AttrIndx), OA range table, and TCR_EL1 configuration (T0SZ, TG0/TG1, IPS).

13. **`explain_memory_attributes`** (`cacheability`)

    Covers WB/WT/NC policies with MAIR nibble encodings, inner vs outer cache domains, all four device memory types (nGnRnE through GRE) with G/R/E explanations, and cache maintenance operations (DC CIVAC, DC CVAC, IC IALLU, etc.).

14. **`explain_extension`** (`SVE`)

    Scalable Vector Extension: purpose (vector-length-agnostic SIMD), key registers (Z0-Z31, P0-P15, FFR, ZCR), key instructions (LD1/ST1, WHILELT, gather/scatter), detection method (ID_AA64PFR0_EL1), and use cases (HPC, ML, bioinformatics).

15. **`compare_architecture_versions`** (`ARMv9.0`)

    Lists mandatory features (SVE2, ETE, TRBE, RME), optional features (TME, MTE, BTI, SME), notable changes (first major version since ARMv8), and example cores (A710, X2, N2).

16. **`show_assembly_pattern`** (`function_prologue`, `aarch64`)

    Annotated assembly: STP X29/X30 with pre-index, MOV X29/SP, STP for callee-saved registers X19-X28, with comments on alignment, PAC, and frame records.

17. **`explain_barrier`** (`DMB`)

    Explains Data Memory Barrier: what it does (orders memory accesses), what it doesn't do (not DSB/ISB), when to use it (locks, producer-consumer, MMIO), and all 12 domain options (SY, ST, LD, ISH, ISHST, ISHLD, NSH, OSH, etc.).

---

## Example Usage

Once the MCP server is connected, you can ask your AI assistant natural-language questions about ARM and it will call the tools automatically. Below is one example per tool showing the prompt you'd type and the actual response the tool returns.

---

### 1. lookup_register

**Prompt:** "Tell me about the X0 register"

**Response** (via `lookup_register("X0")`):

```
## X0 (aliases: W0)
Architecture: aarch64  |  64-bit  |  Category: general_purpose

General-purpose register. First argument and return value in AAPCS64.

Usage: Caller-saved. W0 is the lower 32 bits. Writing W0 zero-extends into X0.
```

---

### 2. list_registers

**Prompt:** "List all AArch64 system registers"

**Response** (via `list_registers("aarch64", "system")`):

```
# aarch64 Registers — system

Name         Width  Aliases          Description
------------------------------------------------------------------------
VBAR_EL1      64-bit  —                Vector Base Address Register (EL1). Holds the base addres...
TTBR0_EL1     64-bit  —                Translation Table Base Register 0 (EL1). Holds the base a...
SCTLR_EL1     64-bit  —                System Control Register (EL1). Controls MMU enable, cache...
```

---

### 3. search_registers

**Prompt:** "Which registers are related to the stack?"

**Response** (via `search_registers("stack")`):

```
# Search results for 'stack' across all architectures
3 register(s) matched.
R13 (SP)  [aarch32  |  32-bit  |  general_purpose]
  Matched in description: "Stack Pointer. Points to the top of the curre..."

X29 (W29, FP)  [aarch64  |  64-bit  |  general_purpose]
  Matched in description: "...nter. Points to the base of the current stack frame."

SP (WSP)  [aarch64  |  64-bit  |  general_purpose]
  Matched in description: "Stack Pointer. Points to the top of the curre..."
```

---

### 4. decode_register_value

**Prompt:** "My debugger shows CPSR = 0x600001D3. What does that mean?"

**Response** (via `decode_register_value("CPSR", "0x600001D3")`):

```
# CPSR  [aarch32  |  32-bit]
Raw value : 0x600001D3
Binary    : 0110 0000 0000 0000 0000 0001 1101 0011

### Bit Field Decode
  Field         Bits     Hex    Dec  Bin           Description
  --------------------------------------------------------------------------------
  N          [   31]  0x00      0  0             Negative flag
  Z          [   30]  0x01      1  1             Zero flag
  C          [   29]  0x01      1  1             Carry flag
  V          [   28]  0x00      0  0             Overflow flag
  Q          [   27]  0x00      0  0             Sticky overflow (saturating arithmetic)
  J          [   24]  0x00      0  0             Jazelle state bit
  E          [    9]  0x00      0  0             Endianness (0=little, 1=big)
  A          [    8]  0x01      1  1             Asynchronous abort mask
  I          [    7]  0x01      1  1             IRQ mask
  F          [    6]  0x01      1  1             FIQ mask
  T          [    5]  0x00      0  0             Thumb state bit (0=ARM, 1=Thumb)
  M          [  4:0]  0x13     19  10011         Processor mode (e.g. 0x10=User, 0x13=SVC, ...)
```

---

### 5. decode_instruction

**Prompt:** "What does the instruction 0xE3A01005 do?"

**Response** (via `decode_instruction("0xE3A01005")`):

```
# ARM AArch32 Instruction Decode: 0xE3A01005
Binary: 11100011101000000001000000000101

### Condition Code
  [31:28]  0xE (14)  →  Condition = AL (Always (unconditional))

Instruction type: Data Processing (bits [27:26] = 00)
Mnemonic: MOV

### Field Breakdown
  [   25]  0x1 (1)  →  I (Immediate) = 1 (Immediate operand2)
  [24:21]  0xD (13)  →  Opcode = MOV
  [   20]  0x0 (0)  →  S (Set flags) = 0 (No)
  [19:16]  0x0 (0)  →  Rn (first operand register) = R0
  [15:12]  0x1 (1)  →  Rd (destination register) = R1
  [ 11:0]  0x5 (5)  →  Operand2 (immediate): rotate=0, imm8=0x05 -> value = 5
```

---

### 6. explain_condition_code

**Prompt:** "What does the EQ condition code mean?"

**Response** (via `explain_condition_code("EQ")`):

```
# Condition Code: EQ  (0x0 / 0000)

Full name: Equal
Encoding: bits [31:28] = 0b0000 (0x0)
Flags tested: Z
Flag condition: Z == 1
Opposite condition: NE
Common use case: Check if two values are equal (after CMP) or if a result is zero.

### Example
  CMP R0, R1         ; Compare R0 and R1, updating CPSR flags
  BEQ label        ; Branch to 'label' if Z == 1
```

---

### 7. explain_calling_convention

**Prompt:** "What's the AArch64 calling convention?"

**Response** (via `explain_calling_convention("aarch64")`):

```
# Calling Convention: AARCH64
## AAPCS64 (ARM Architecture Procedure Call Standard for AArch64)
Specification: IHI0055

## Argument Registers
Registers (8):  X0, X1, X2, X3, X4, X5, X6, X7
Notes: Integer/pointer arguments in X0-X7. 32-bit values use W-form (W0-W7).
       Arguments beyond X7 are passed on the stack, 8-byte aligned.
       FP/SIMD arguments use V0-V7.

## Return Value Registers
Integer/pointer:    X0, X1
Floating-point:     V0, V1, V2, V3

## Caller-Saved (Volatile)
Integer:         X0-X17, X18 (platform-dependent)
Floating-point:  V0-V7 (argument/result), V16-V31 (scratch)

## Callee-Saved (Non-Volatile)
Integer:         X19-X28, X29 (FP), X30 (LR) when saved
Floating-point:  V8-V15 (low 64 bits only — D8-D15)

## Stack Alignment
At call boundary:  16 bytes
Internal:          16 bytes (SP must always be 16-byte aligned)

## Frame Pointer
Register: X29 (FP)
Notes: Frame record: STP X29, X30, [SP, #-16]! at entry.

## Special Notes
  1. X8: Indirect result register — caller passes struct return address here.
  2. X16/IP0, X17/IP1: Linker veneer scratch.
  3. X18: Platform register — avoid in portable code.
  4. PAC (ARMv8.3+): Return addresses may be signed with PACIASP.
```

---

### 8. explain_exception_levels

**Prompt:** "Explain AArch64 exception levels"

**Response** (via `explain_exception_levels("aarch64")`):

```
# AArch64 Exception Levels (EL0-EL3)

ARM AArch64 defines four Exception Levels (EL0-EL3), with EL3 being the most
privileged. Higher ELs control and can trap operations from lower ELs.

## EL0 — User / Application
  Privilege: Unprivileged (lowest)
  What runs here: User-space applications, unprivileged library code
  Stack pointer: SP_EL0 (always used at EL0)

## EL1 — OS Kernel
  Privilege: Privileged (OS level)
  What runs here: Operating system kernels (Linux, Windows, etc.)
  Key registers: SCTLR_EL1, TTBR0_EL1, VBAR_EL1, ESR_EL1, ELR_EL1, SPSR_EL1

## EL2 — Hypervisor
  Privilege: Hypervisor (higher than OS)
  What runs here: Hypervisors / VMMs (KVM, Xen, Hyper-V)
  Key registers: HCR_EL2 (controls trapping, stage-2 translation), VTTBR_EL2

## EL3 — Secure Monitor
  Privilege: Highest privilege (firmware level)
  What runs here: ARM Trusted Firmware (TF-A / BL31), SMC dispatcher
  Key registers: SCR_EL3 (NS bit controls Secure/Non-secure world)

## Vector Table Layout
  16 entries at VBAR_ELx offsets (0x000-0x780), each 128 bytes.
  Organized by: source (Current EL SP_EL0, Current EL SP_ELx,
  Lower EL AArch64, Lower EL AArch32) × type (Sync, IRQ, FIQ, SError).

## Security States
  Non-secure: SCR_EL3.NS=1       — normal world
  Secure:     SCR_EL3.NS=0       — TrustZone secure world
  Realm:      SCR_EL3.NS=1,NSE=1 — ARMv9 RME confidential VMs
  Root:       SCR_EL3.NS=0,NSE=1 — EL3 firmware (ARMv9 RME)
```

---

### 9. explain_security_model

**Prompt:** "How does ARM TrustZone and RME work?"

**Response** (via `explain_security_model("aarch64")`):

```
# AArch64 Security Model

## TrustZone: Secure vs Non-secure Worlds
  SCR_EL3.NS (bit 0) selects the world:
    0 -> Lower ELs are in Secure state
    1 -> Lower ELs are in Non-secure state

  World-switching is mediated by EL3 firmware via SMC instruction.

## Secure Exception Levels
  S-EL0: Trusted Applications (DRM, key management)
  S-EL1: Trusted OS (OP-TEE, Trusty)
  S-EL2: Secure Partition Manager (ARMv8.4+, Hafnium-based)

## ARMv9 Realm Management Extension (RME)
  Four worlds: Non-secure, Secure, Realm, Root
  Granule Protection Table (GPT): hardware-enforced per-page world assignment
  Granule Protection Checks (GPC): MMU checks PA against GPT on every access

## Confidential Compute Architecture (CCA)
  RMM (Realm Management Monitor) at R-EL2 manages confidential Realm VMs
  Realm VMs are isolated from hypervisor, Secure world, and other Realms
  Hardware attestation proves Realm integrity to remote verifiers
```

---

### 10. lookup_core

**Prompt:** "Tell me about the Cortex-A78"

**Response** (via `lookup_core("Cortex-A78")`):

```
# ARM Cortex-A78
Series: Cortex-A  |  Architecture: ARMv8.2-A  |  Year: 2020

## Microarchitecture
  Generation:    Hercules
  Pipeline:      13-stage out-of-order
  Decode width:  4-wide

## Key Features
  - Out-of-order execution, 4-wide decode
  - ARMv8.2-A with FP16, dot-product, optional RAS
  - L1I 32-64 KB, L1D 32-64 KB
  - Private L2 up to 512 KB, shared L3 via DSU-110
  - ~20% sustained performance uplift over A77 at same power
  - DynamIQ pairing with Cortex-A55 or Cortex-A78C

## Target Market
  Mobile (premium performance core)

## Notable Products / SoCs
  - Qualcomm Snapdragon 888
  - Samsung Exynos 2100
  - MediaTek Dimensity 1200
```

---

### 11. compare_cores

**Prompt:** "Compare the Cortex-A78 and Cortex-X4"

**Response** (via `compare_cores("Cortex-A78", "Cortex-X4")`):

```
# Core Comparison: ARM Cortex-A78 vs ARM Cortex-X4

Attribute               ARM Cortex-A78                   ARM Cortex-X4
---------------------------------------------------------------------------
Series                  Cortex-A                         Cortex-X
Architecture            ARMv8.2-A                        ARMv9.2-A
Year                    2020                             2023
Pipeline Depth          13-stage out-of-order            13+ stage out-of-order
Decode Width            4-wide                           6-wide
Target Market           Mobile (premium)                 Mobile (prime/peak)

## Key Features Comparison

### ARM Cortex-A78
  - 4-wide decode, FP16, dot-product
  - Private L2 up to 512 KB, shared L3 via DSU-110

### ARM Cortex-X4
  - 6-wide decode, SVE2, MTE, BRBE
  - Private L2 up to 2 MB, shared L3 via DSU-120 (up to 16 MB)
  - ~15% IPC uplift over X3

## Notable Products / SoCs
  Cortex-A78: Snapdragon 888, Exynos 2100, Dimensity 1200
  Cortex-X4:  Snapdragon 8 Gen 3, Dimensity 9300
```

---

### 12. explain_page_table_format

**Prompt:** "Explain the 4KB granule page table format with 48-bit VA"

**Response** (via `explain_page_table_format("4KB", 48)`):

```
# AArch64 Page Table Format: 4KB Granule, 48-bit VA

## Overview
  Translation granule:       4KB (4096 bytes)
  Virtual address bits:      48
  Page table entry size:     8 bytes (64 bits)
  Entries per table:         512
  Active translation levels: 4 (L0, L1, L2, L3)

## Virtual Address Bit Layout

  47                                                            0
  +-----------+-----------+-----------+-----------+--------------+
  |    L0     |    L1     |    L2     |    L3     |    Offset    |
  |  [47:39]  |  [38:30]  |  [29:21]  |  [20:12]  |    [11:0]    |
  +-----------+-----------+-----------+-----------+--------------+

## Translation Level Details
  L0: VA[47:39], 512 entries — Table descriptor only
  L1: VA[38:30], 512 entries — Block (1GB) or Table
  L2: VA[29:21], 512 entries — Block (2MB) or Table
  L3: VA[20:12], 512 entries — Page (4KB)

## PTE Descriptor Fields
  [54] UXN, [53] PXN, [52] Contiguous, [10] AF, [9:8] SH,
  [7:6] AP, [4:2] AttrIndx, [1:0] Type

## TCR_EL1 Configuration
  T0SZ/T1SZ = 16 → 48-bit VA
  TG0 = 0b00 → 4KB granule
  IPS: controls max physical address width (32-52 bits)
```

---

### 13. explain_memory_attributes

**Prompt:** "Explain ARM memory cacheability"

**Response** (via `explain_memory_attributes("cacheability")`):

```
# ARM Memory Cacheability

## Normal Memory Cache Policies

### Write-Back (WB)
  Writes update only the cache; dirty lines written to memory on eviction.
  Best performance. MAIR nibble: 0b1011 = WB, Read+Write Allocate.

### Write-Through (WT)
  Writes update both cache and memory simultaneously.
  MAIR nibble: 0b0110 = WT, Read-Allocate.

### Non-cacheable (NC)
  All accesses go directly to memory. MAIR encoding: 0x44.

## Device Memory Types (Always Non-cacheable)

  | Type           | Gathering | Reordering | Early Ack | MAIR  | Use Case              |
  |----------------|-----------|------------|-----------|-------|-----------------------|
  | Device-nGnRnE  | No        | No         | No        | 0x00  | Status registers      |
  | Device-nGnRE   | No        | No         | Yes       | 0x04  | General MMIO          |
  | Device-nGRE    | No        | Yes        | Yes       | 0x08  | PCIe config, DMA      |
  | Device-GRE     | Yes       | Yes        | Yes       | 0x0C  | Framebuffers          |

## Cache Maintenance Operations
  DC CIVAC: Clean+Invalidate by VA to Point of Coherency
  DC CVAC:  Clean by VA to Point of Coherency
  IC IALLU: Invalidate All I-caches to PoU
```

---

### 14. explain_extension

**Prompt:** "Tell me about the SVE extension"

**Response** (via `explain_extension("SVE")`):

```
# Scalable Vector Extension (SVE)
Introduced in: ARMv8.2-A (optional)

## Purpose
Provides a vector-length-agnostic SIMD programming model for HPC and
scientific workloads. Supports vector lengths from 128 to 2048 bits in
128-bit increments, chosen by the hardware implementation.

## Key Registers
  - Z0-Z31: Scalable vector registers (VL bits wide, 128-2048)
  - P0-P15: Predicate registers (VL/8 bits) for per-lane masking
  - FFR: First Fault Register (for speculative memory access)
  - ZCR_EL1/2/3: SVE Control Register (controls effective vector length)

## Key Instructions
  - LD1B/LD1H/LD1W/LD1D: Contiguous predicated loads
  - ST1B/ST1H/ST1W/ST1D: Contiguous predicated stores
  - WHILELT/WHILELE: Generate predicate for loop tails
  - GATHER/SCATTER loads/stores for indirect access

## Detection
  Register: ID_AA64PFR0_EL1, Field: SVE (bits [35:32])
  Linux: HWCAP_SVE in getauxval(AT_HWCAP)

## Practical Use Cases
  - HPC: Dense linear algebra, FFT, weather/climate simulation
  - ML inference: Vectorized activation functions, quantized dot products
  - Bioinformatics: Sequence alignment, genome assembly
```

---

### 15. compare_architecture_versions

**Prompt:** "What features did ARMv9.0 introduce?"

**Response** (via `compare_architecture_versions("armv9.0")`):

```
# ARMv9.0-A (2021)

## Mandatory Features
  - Everything in ARMv8.5-A (ARMv9.0 is based on ARMv8.5)
  - FEAT_SVE2: SVE2 is mandatory (scalable vectors for all workloads)
  - FEAT_ETE: Embedded Trace Extension (successor to ETM for trace)
  - FEAT_TRBE: Trace Buffer Extension (self-hosted trace to memory buffer)
  - FEAT_RME: Realm Management Extension (hardware isolation for confidential compute)

## Optional Features
  - FEAT_TME: Transactional Memory Extension
  - FEAT_MTE / FEAT_MTE2: Memory Tagging Extension
  - FEAT_BTI: Branch Target Identification
  - FEAT_SME: Scalable Matrix Extension (introduced at v9.2)

## Notable Changes
ARMv9 is the first major version change since ARMv8 in 2011. Makes SVE2
mandatory, adds RME for confidential computing (ARM CCA), and introduces
new trace infrastructure (ETE/TRBE).

## Example Cores
  Cortex-A710, Cortex-X2, Neoverse N2 (ARMv9.0 baseline)
```

---

### 16. show_assembly_pattern

**Prompt:** "Show me how to write a spinlock in AArch64"

**Response** (via `show_assembly_pattern("spinlock_acquire", "aarch64")`):

```
# Assembly Pattern: spinlock_acquire (aarch64)

// === AArch64 Spinlock Acquire ===
// X0 = lock address (0=unlocked, 1=locked). Acquire semantics.

    MOV   W2, #1                // W2 = 1 (locked value).
    SEVL                        // Send Event Locally: first WFE won't stall.
1:  WFE                          // Wait For Event: low-power spin wait.
    LDAXR  W3, [X0]             // Load-Exclusive with Acquire: read lock.
                                 //   Acquire ensures we see all writes from
                                 //   the previous lock holder.
    CBNZ   W3, 1b               // If locked, spin.
    STXR   W3, W2, [X0]         // Store-Exclusive: try to set lock=1.
    CBNZ   W3, 1b               // If store failed, retry.

// Critical section begins. All subsequent accesses ordered after LDAXR.
//
// ARMv8.1 LSE alternative:
// 1:  SWPA W2, W3, [X0]        // Atomic swap with Acquire
//     CBNZ W3, 1b              // If old!=0, lock was held
```

---

### 17. explain_barrier

**Prompt:** "Explain the DMB barrier"

**Response** (via `explain_barrier("DMB")`):

```
# Data Memory Barrier (DMB)
Encoding: DMB <option>  (AArch64/AArch32)

## What It Does
DMB ensures that all explicit memory accesses that appear in program order
before the DMB are observed before any explicit memory accesses after the DMB.

CRITICAL: DMB only orders memory accesses relative to each other. It does NOT:
  - Wait for memory accesses to complete (use DSB for that)
  - Flush the pipeline (use ISB for that)

## When To Use It
  1. Between a flag/lock write and subsequent data reads.
  2. Implementing acquire/release when LDAR/STLR are not available.
  3. Producer-consumer: write data, DMB, write flag.
  4. Memory-mapped I/O: ordering a register write before a status read.

## Domain / Shareability Options
  SY:     Full System (default, strongest)
  ST:     Full System, Store only
  LD:     Full System, Load only (ARMv8.1+)
  ISH:    Inner Shareable (most common for SMP)
  ISHST:  Inner Shareable, Store only
  ISHLD:  Inner Shareable, Load only (ARMv8.1+)
  NSH:    Non-shareable (local PE only)
  OSH:    Outer Shareable (full system including GPUs/DMA)
  ... and NSHST, NSHLD, OSHST, OSHLD variants
```

---

## License

MIT
