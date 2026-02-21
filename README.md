# ARM Reference MCP Suite

A collection of [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) servers that give AI assistants instant access to ARM architecture reference data, cloud migration guidance, documentation search, and edge AI deployment planning.

The suite includes **4 MCP servers** with **44 tools** total:

| MCP Server | Tools | Description |
|------------|-------|-------------|
| **ARM Register Reference** | 23 | Core architecture reference: registers, instructions, calling conventions, exception levels, security, page tables, NEON/SME, optimization, and more |
| **ARM Documentation RAG** | 7 | Search and explain ARM architecture documentation, errata, manual references, and instruction encodings |
| **ARM Cloud Migration Advisor** | 7 | Analyze dependencies, Docker images, Dockerfiles, benchmarks, and infrastructure for x86-to-ARM cloud migration |
| **ARM TinyML & Edge AI** | 7 | Plan and optimize ML model deployment on Cortex-M, Cortex-A, Ethos NPU, and edge AI accelerators |

Built in Python. Works with any MCP-compatible client over stdio transport.

---

## Installation

### Prerequisites

Python 3.10+ and either [uv](https://docs.astral.sh/uv/) (recommended) or pip. Distributed via GitHub (not PyPI).

### Claude Code (CLI)

**Route 1: Add as standalone MCP servers**

Using `uvx` (no permanent install required):

```bash
# ARM Register Reference (23 tools)
claude mcp add --transport stdio arm-reference -- uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-reference-mcp

# ARM Documentation RAG (7 tools)
claude mcp add --transport stdio arm-docs-rag -- uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-docs-rag-mcp

# ARM Cloud Migration Advisor (7 tools)
claude mcp add --transport stdio arm-cloud-migration -- uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-cloud-migration-mcp

# ARM TinyML & Edge AI (7 tools)
claude mcp add --transport stdio arm-tinyml -- uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-tinyml-mcp
```

Or install with pip first, then add:

```bash
pip install "git+https://github.com/yerry262/arm-reference-mcp.git"
claude mcp add --transport stdio arm-reference -- arm-reference-mcp
claude mcp add --transport stdio arm-docs-rag -- arm-docs-rag-mcp
claude mcp add --transport stdio arm-cloud-migration -- arm-cloud-migration-mcp
claude mcp add --transport stdio arm-tinyml -- arm-tinyml-mcp
```

**Route 2: Add as a plugin via Claude Code marketplace**

```
/plugin marketplace add yerry262/arm-reference-mcp
/plugin install arm-reference-mcp
```

This auto-configures all MCP servers based on `.mcp.json` and `.claude-plugin/plugin.json`.

### VS Code (GitHub Copilot / Continue.dev)

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

Or add to your VS Code user `settings.json`:

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

### Cursor

Add to `.cursor/mcp.json` in your project root:

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

### Windsurf

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

### OpenAI Codex CLI

Add to your Codex MCP config (see [Codex CLI docs](https://github.com/openai/codex) for the config file location):

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

### Generic / Other MCP Clients

Any MCP-compatible client can connect over **stdio** transport:

```bash
uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-reference-mcp
```

Or install and run directly:

```bash
pip install "git+https://github.com/yerry262/arm-reference-mcp.git"
arm-reference-mcp
```

No HTTP server or port configuration needed.

---

## ARM Register Reference MCP

The primary MCP server providing **23 tools** for ARM architecture reference, organized into six categories.

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

### SIMD, optimization & porting tools

| Tool | Description | Example Input |
|------|-------------|---------------|
| `explain_neon_intrinsic` | NEON/ASIMD intrinsic reference with instruction, types, latency, and usage example | `explain_neon_intrinsic("vfmaq_f32")`, `explain_neon_intrinsic("vld1q_f32")` |
| `explain_sme_tile` | SME tile operations, ZA storage, streaming SVE mode, outer products | `explain_sme_tile("outer_product")`, `explain_sme_tile("za_storage")` |
| `suggest_optimization` | ARM-specific optimization suggestions for code patterns | `suggest_optimization("matrix_multiply", target_core="cortex-a78")` |
| `lookup_system_register` | Full AArch64 system register reference (beyond the basic set) | `lookup_system_register("TCR_EL1")`, `lookup_system_register("list", el="EL1")` |
| `explain_performance_counter` | ARM PMU performance counter event reference | `explain_performance_counter("L1D_CACHE_REFILL")`, `explain_performance_counter("topdown")` |
| `translate_intrinsic` | Translate between x86 SSE/AVX and ARM NEON/SVE intrinsics | `translate_intrinsic("_mm_add_ps", "x86", "neon")` |

---

## ARM Documentation RAG MCP

Search and explain ARM architecture documentation, errata, and manual references. Provides **7 tools**.

| Tool | Description | Example Input |
|------|-------------|---------------|
| `search_arm_docs` | Search ARM documentation entries by keyword across manuals, guides, and specs | `search_arm_docs("NEON")`, `search_arm_docs("page tables", doc_scope="architecture")` |
| `explain_arm_concept` | Explain an ARM architecture concept in detail | `explain_arm_concept("trustzone")`, `explain_arm_concept("cache_coherency")` |
| `find_register_in_manual` | Find which ARM manual section documents a specific system register | `find_register_in_manual("SCTLR_EL1")`, `find_register_in_manual("TCR_EL1", context="memory")` |
| `get_errata` | Get known errata for an ARM core with severity, workarounds, and impact | `get_errata("cortex-a72")`, `get_errata("cortex-a53", category="functional")` |
| `compare_manual_sections` | Compare how a topic is covered across different ARM architecture versions | `compare_manual_sections("memory_management")`, `compare_manual_sections("exception_handling")` |
| `list_arm_documents` | Browse the ARM documentation catalog, optionally filtered by scope or architecture | `list_arm_documents()`, `list_arm_documents(doc_scope="architecture", architecture="aarch64")` |
| `explain_instruction_encoding` | Explain A64, T32, or A32 instruction set encoding format in detail | `explain_instruction_encoding("a64")`, `explain_instruction_encoding("thumb")` |

---

## ARM Cloud Migration Advisor MCP

Analyze codebases, dependencies, and infrastructure for x86-to-ARM cloud migration readiness. Provides **7 tools**.

| Tool | Description | Example Input |
|------|-------------|---------------|
| `scan_x86_dependencies` | Scan dependencies for x86-only packages and ARM compatibility | `scan_x86_dependencies("python", "numpy,scipy,intel-mkl,tensorflow")` |
| `suggest_arm_cloud_instance` | Map a workload type to recommended ARM instances across providers | `suggest_arm_cloud_instance("web_server")`, `suggest_arm_cloud_instance("database", provider="aws")` |
| `check_docker_arm_support` | Check whether a Docker base image supports arm64/aarch64 | `check_docker_arm_support("postgres")`, `check_docker_arm_support("node")` |
| `generate_ci_matrix` | Generate cross-architecture CI config for building on x86 and ARM | `generate_ci_matrix("github_actions", language="python")` |
| `estimate_migration_effort` | Estimate effort and complexity of migrating a codebase from x86 to ARM | `estimate_migration_effort("python_web")`, `estimate_migration_effort("java_enterprise")` |
| `generate_arm_dockerfile` | Generate a multi-stage ARM-optimized Dockerfile for a given language | `generate_arm_dockerfile("python")`, `generate_arm_dockerfile("go")` |
| `compare_arm_vs_x86_perf` | Compare ARM vs x86 performance benchmarks for a workload type | `compare_arm_vs_x86_perf("web_server")`, `compare_arm_vs_x86_perf("database")` |

---

## ARM TinyML & Edge AI MCP

Plan and optimize ML model deployment on ARM Cortex-M, Cortex-A, Ethos NPU, and edge AI accelerators. Provides **7 tools**.

| Tool | Description | Example Input |
|------|-------------|---------------|
| `check_operator_support` | Check ML operator compatibility on an ARM accelerator or runtime | `check_operator_support("ethos-u55", "conv2d,relu,lstm,softmax,gelu")` |
| `suggest_quantization` | Suggest a quantization strategy for deploying an ML model on an ARM target | `suggest_quantization("image_classification", "ethos-u55")` |
| `estimate_inference` | Estimate inference time and memory requirements for a model on an ARM target | `estimate_inference("mobilenetv2_1.0_224", "cortex-m55_cmsis_nn")` |
| `compare_tinyml_targets` | Compare two ARM TinyML hardware targets side by side | `compare_tinyml_targets("cortex-m55", "ethos-u55")` |
| `generate_deployment_config` | Generate deployment configuration for an ARM edge AI framework | `generate_deployment_config("vela")`, `generate_deployment_config("tflite_micro_cmsis")` |
| `suggest_model_architecture` | Suggest ML model architectures that fit given SRAM/Flash constraints | `suggest_model_architecture("image_classification", sram_kb=256, flash_kb=512)` |
| `explain_tinyml_framework` | Explain a TinyML/edge AI framework in detail | `explain_tinyml_framework("cmsis_nn")`, `explain_tinyml_framework("vela")` |

---

## Example Usage

Once connected, ask your AI assistant natural-language questions about ARM and it will call the tools automatically. Here are a few representative examples.

### lookup_register

**Prompt:** "Tell me about the X0 register"

```
## X0 (aliases: W0)
Architecture: aarch64  |  64-bit  |  Category: general_purpose

General-purpose register. First argument and return value in AAPCS64.

Usage: Caller-saved. W0 is the lower 32 bits. Writing W0 zero-extends into X0.
```

### decode_register_value

**Prompt:** "My debugger shows CPSR = 0x600001D3. What does that mean?"

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

### explain_exception_levels

**Prompt:** "Explain AArch64 exception levels"

```
# AArch64 Exception Levels (EL0-EL3)

## EL0 — User / Application
  Privilege: Unprivileged (lowest)
  What runs here: User-space applications, unprivileged library code

## EL1 — OS Kernel
  Privilege: Privileged (OS level)
  What runs here: Operating system kernels (Linux, Windows, etc.)
  Key registers: SCTLR_EL1, TTBR0_EL1, VBAR_EL1, ESR_EL1, ELR_EL1, SPSR_EL1

## EL2 — Hypervisor
  What runs here: Hypervisors / VMMs (KVM, Xen, Hyper-V)
  Key registers: HCR_EL2, VTTBR_EL2

## EL3 — Secure Monitor
  Privilege: Highest privilege (firmware level)
  What runs here: ARM Trusted Firmware (TF-A / BL31), SMC dispatcher
  Key registers: SCR_EL3 (NS bit controls Secure/Non-secure world)

## Security States
  Non-secure: SCR_EL3.NS=1       — normal world
  Secure:     SCR_EL3.NS=0       — TrustZone secure world
  Realm:      SCR_EL3.NS=1,NSE=1 — ARMv9 RME confidential VMs
  Root:       SCR_EL3.NS=0,NSE=1 — EL3 firmware (ARMv9 RME)
```

### translate_intrinsic

**Prompt:** "What's the ARM equivalent of _mm256_fmadd_ps?"

```
# Intrinsic Translation: _mm256_fmadd_ps -> 2x vfmaq_f32

## x86 (FMA/AVX2)
  Intrinsic: _mm256_fmadd_ps
  Instruction: VFMADD132PS / VFMADD213PS / VFMADD231PS (ymm)

## ARM NEON Equivalent
  Intrinsic: 2x vfmaq_f32 (split into high/low)
  Instruction: FMLA Vd.4S, Vn.4S, Vm.4S (x2)

## ARM SVE Equivalent
  Intrinsic: svmla_f32_x(pg, acc, a, b)
  Instruction: FMLA Zd.S, Pg/M, Zn.S, Zm.S

## Gotchas
Operand order differs! x86 FMA: fmadd(a,b,c) = a*b+c.
NEON FMA: vfmaq_f32(acc,a,b) = acc+a*b. Accumulator is FIRST in NEON.
```

### scan_x86_dependencies

**Prompt:** "Check if my Python deps work on ARM"

```
# x86 Dependency Scan: python (5 packages)

## Compatible (4)
  numpy, scipy, tensorflow, pandas — all have arm64 wheels

## x86-Only (1)
  intel-mkl — Alternatives: OpenBLAS, ArmPL, BLIS

Migration Readiness: 80/100 (HIGH)
```

### suggest_model_architecture

**Prompt:** "What ML models fit in 256 KB SRAM and 512 KB Flash?"

```
# Model Recommendations: image_classification
Constraints: 256 KB SRAM, 512 KB Flash

## Recommended Models
  1. MobileNetV2 0.35 (96x96) — 0.4M params, ~200 KB SRAM, ~400 KB Flash
  2. MCUNet (64x64) — 0.7M params, ~180 KB SRAM, ~350 KB Flash

Models filtered by hardware constraints. Headroom percentages shown.
```

---

## Development

### Project structure

```
arm-reference-mcp/
  src/arm_reference_mcp/
    __init__.py
    server.py                  # ARM Register Reference MCP (23 tools)
    data.py                    # Register definitions for AArch32 and AArch64
    docs_rag_server.py         # ARM Documentation RAG MCP (7 tools)
    cloud_migration_server.py  # ARM Cloud Migration Advisor MCP (7 tools)
    tinyml_server.py           # ARM TinyML & Edge AI MCP (7 tools)
  tests/
    test_tools.py              # 128 tests for ARM Register Reference
    test_docs_rag.py           # 69 tests for ARM Documentation RAG
    test_cloud_migration.py    # 44 tests for ARM Cloud Migration Advisor
    test_tinyml.py             # 73 tests for ARM TinyML & Edge AI
  .claude-plugin/
    plugin.json                # Claude Code plugin metadata
    marketplace.json           # Plugin marketplace manifest
  .mcp.json                    # Default MCP server config (used by plugins)
  pyproject.toml               # Package metadata and build config
```

### Running locally

```bash
git clone https://github.com/yerry262/arm-reference-mcp.git
cd arm-reference-mcp
pip install -e .

# Run any server directly
arm-reference-mcp
python -m arm_reference_mcp.docs_rag_server
python -m arm_reference_mcp.cloud_migration_server
python -m arm_reference_mcp.tinyml_server
```

### Running tests

```bash
# Individual test suites (each has a built-in runner)
python tests/test_tools.py
python tests/test_docs_rag.py
python tests/test_cloud_migration.py
python tests/test_tinyml.py

# Or run everything with pytest
python -m pytest tests/ -v
```

### Dependencies

- Python >= 3.10
- `mcp[cli]` >= 1.0.0 (the Model Context Protocol SDK)

---

## License

MIT
