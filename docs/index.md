---
title: Home
---

# ARM Reference MCP Suite

**44 tools across 4 MCP servers** that give AI coding assistants instant access to ARM architecture knowledge.

Built on the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), this toolkit turns any compatible AI assistant into an ARM architecture expert -- covering everything from register bit fields to cloud migration planning to edge AI deployment.

---

## What's Inside

| Server | Tools | What It Does |
|--------|-------|--------------|
| [ARM Register Reference](servers/arm-register-reference) | 23 | Core architecture reference: registers, instructions, calling conventions, exception levels, security models, page tables, NEON/SME, optimization, system registers, PMU events, and x86-to-ARM porting |
| [ARM Documentation RAG](servers/arm-docs-rag) | 7 | Search ARM manuals, explain concepts, find register documentation, retrieve core errata, compare architecture versions, and browse instruction encodings |
| [ARM Cloud Migration Advisor](servers/arm-cloud-migration) | 7 | Scan dependencies for ARM compatibility, recommend cloud instances, check Docker arm64 support, generate CI configs and Dockerfiles, estimate migration effort, and compare ARM vs x86 benchmarks |
| [ARM TinyML & Edge AI](servers/arm-tinyml) | 7 | Check ML operator support on Ethos-U/CMSIS-NN/ARM NN, suggest quantization, estimate inference performance, compare hardware targets, generate deployment configs, and recommend model architectures |

---

## Quick Start

Install in one command with any MCP-compatible AI coding tool:

**Claude Code:**
```bash
claude mcp add --transport stdio arm-reference -- \
  uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-reference-mcp
```

**Or install all 4 servers at once as a Claude Code plugin:**
```
/plugin marketplace add yerry262/arm-reference-mcp
/plugin install arm-reference-mcp
```

See the full [Installation Guide](installation) for VS Code, Cursor, Windsurf, Codex CLI, and other clients.

---

## How It Works

Once connected, you just ask questions in natural language. The AI assistant calls the right tool automatically.

**You ask:** *"My debugger shows CPSR = 0x600001D3. What does that mean?"*

**The tool decodes every bit field:**

```
# CPSR  [aarch32  |  32-bit]
Raw value : 0x600001D3

### Bit Field Decode
  N  [31]  0  Negative flag
  Z  [30]  1  Zero flag
  C  [29]  1  Carry flag
  V  [28]  0  Overflow flag
  A  [ 8]  1  Asynchronous abort mask
  I  [ 7]  1  IRQ mask
  F  [ 6]  1  FIQ mask
  T  [ 5]  0  Thumb state (0=ARM)
  M [4:0]  0x13  SVC mode
```

**You ask:** *"What's the ARM equivalent of `_mm256_fmadd_ps`?"*

**The tool maps x86 to ARM:**

```
# Intrinsic Translation: _mm256_fmadd_ps -> 2x vfmaq_f32

## ARM NEON Equivalent
  Intrinsic: 2x vfmaq_f32 (split into high/low)
  Instruction: FMLA Vd.4S, Vn.4S, Vm.4S (x2)

## Gotchas
  Operand order differs! x86 FMA: fmadd(a,b,c) = a*b+c.
  NEON FMA: vfmaq_f32(acc,a,b) = acc+a*b. Accumulator is FIRST.
```

**You ask:** *"Check if my Python deps work on ARM"*

**The tool scans your dependency list:**

```
# x86 Dependency Scan: python (5 packages)

## Compatible (4)
  numpy, scipy, tensorflow, pandas -- all have arm64 wheels

## x86-Only (1)
  intel-mkl -- Alternatives: OpenBLAS, ArmPL, BLIS

Migration Readiness: 80/100 (HIGH)
```

More examples in each [server's documentation](servers/arm-register-reference).

---

## Who It's For

- **Embedded / firmware engineers** working with ARM Cortex-M, Cortex-A, or Cortex-R processors
- **Systems programmers** writing kernel code, bootloaders, or hypervisors on AArch64
- **Performance engineers** optimizing with NEON, SVE, or SME intrinsics
- **Cloud engineers** migrating workloads to AWS Graviton, Azure Cobalt, GCP Axion, or Oracle Ampere
- **ML engineers** deploying models on Cortex-M + Ethos-U for edge inference
- **Anyone porting code** from x86 to ARM who needs intrinsic translation and compatibility checks

---

## Architecture

All reference data is stored as **inline Python dictionaries** -- no database, no external API calls, no network dependency. Each server is a standalone Python process communicating over stdio.

```
arm-reference-mcp/
  src/arm_reference_mcp/
    server.py                  # 23 tools, ~12,300 lines of ARM reference data
    docs_rag_server.py         # 7 tools, curated ARM manual snippets
    cloud_migration_server.py  # 7 tools, instance/dependency/benchmark data
    tinyml_server.py           # 7 tools, operator/model/framework data
    data.py                    # 113 ARM register definitions (AArch32 + AArch64)
```

**Zero external dependencies** beyond the MCP SDK (`mcp[cli]>=1.0.0`). Python 3.10+.

---

## Links

- [GitHub Repository](https://github.com/yerry262/arm-reference-mcp)
- [Installation Guide](installation)
- [ARM Register Reference (23 tools)](servers/arm-register-reference)
- [ARM Documentation RAG (7 tools)](servers/arm-docs-rag)
- [ARM Cloud Migration Advisor (7 tools)](servers/arm-cloud-migration)
- [ARM TinyML & Edge AI (7 tools)](servers/arm-tinyml)

---

*MIT License. Built with [FastMCP](https://modelcontextprotocol.io/).*
