---
title: ARM Documentation RAG MCP
---

# ARM Documentation RAG MCP

Search and understand ARM architecture documentation without leaving your editor. **7 tools** for navigating manuals, explaining concepts, finding register documentation, retrieving errata, and understanding instruction encodings.

**Server entry point:** `arm-docs-rag-mcp`

---

## Tool Reference

| Tool | Description |
|------|-------------|
| `search_arm_docs(query, doc_scope?, architecture?)` | Search ARM documentation entries by keyword. Searches across Architecture Reference Manuals, Technical Reference Manuals, Programming Guides, Security specs, and Optimization Guides. Optional filters by scope and architecture. |
| `explain_arm_concept(concept)` | Deep-dive explanation of an ARM architecture concept. Supports: `exception_levels`, `trustzone`, `vmsa`, `memory_ordering`, `cache_coherency`, `sve`, `mte`, `pac`, `gic`, `nvic`, `psci`, and common aliases. |
| `find_register_in_manual(register_name, context?)` | Locate which ARM manual section documents a system register. Returns document ID, section number, summary, and related registers. Optional context filter: `system`, `memory`, `exception`, `virtualization`, `security`. |
| `get_errata(core_name, category?)` | Known errata for an ARM core with severity, description, workaround, and impact. Optional category filter: `functional`, `performance`, `security`. |
| `compare_manual_sections(topic)` | Side-by-side comparison of how a topic differs across ARMv7-A, ARMv8-A AArch64, ARMv7-M, and ARMv9-A. Topics: `exception_handling`, `memory_management`, `simd_extensions`, `security_model`, `interrupt_handling`. |
| `list_arm_documents(doc_scope?, architecture?)` | Browse the ARM documentation catalog. Filter by scope (`architecture`, `processor_trm`, `programming_guide`, `security`, `firmware`, `optimization`) and/or architecture. |
| `explain_instruction_encoding(encoding_format)` | Explain instruction set encoding formats. Supports `a64` (AArch64), `t32`/`thumb` (Thumb-2), `a32`/`arm` (ARM), or `overview` for a comparison of all three. Shows encoding groups, key fields, and notes. |

---

## Examples

### Searching documentation

```
> search_arm_docs("NEON", doc_scope="programming_guide")

Found 4 matching sections:
  1. DEN0024 - AArch64 NEON programming guide
     Scope: programming_guide | Architecture: aarch64
  2. DEN0013 - ARMv7 NEON/VFP programming guide
     Scope: programming_guide | Architecture: aarch32
  3. SWOG - Neoverse N2 NEON/SVE optimization
     Scope: optimization | Architecture: aarch64
  ...
```

### Explaining concepts

```
> explain_arm_concept("cache_coherency")

# Cache Coherency

## Overview
ARM uses a MOESI-based cache coherency protocol managed by the
Snoop Control Unit (SCU) or DynamIQ Shared Unit (DSU)...

## Inner vs Outer Shareable
  Inner Shareable: Coherent between cores in same cluster
  Outer Shareable: Coherent across clusters, GPUs, DMA engines

## Key Points
  - Point of Coherency (PoC): where all agents see the same data
  - Point of Unification (PoU): where I-cache and D-cache are coherent
  - DSB + IC IALLU after code modification to ensure I-cache sees new code
```

### Finding register documentation

```
> find_register_in_manual("SCTLR_EL1")

# SCTLR_EL1 in ARM Documentation
  Document: DDI0487 (ARM Architecture Reference Manual)
  Section: D13.2.118

## Summary
  Controls: MMU enable (M), alignment checking (A), cache enable (C/I),
  WXN, endianness (EE), and more.

## Related Registers
  SCTLR_EL2, SCTLR_EL3, TCR_EL1, MAIR_EL1
```

### Retrieving errata

```
> get_errata("cortex-a72", category="functional")

# Cortex-A72 Errata (functional)

## Erratum 859971 [HIGH]
  LDNP/STNP may not maintain ordering with respect to other accesses
  Workaround: Use LDP/STP instead, or insert DMB before LDNP/STNP
  Impact: Data corruption possible under specific access patterns

## Erratum 853709 [MEDIUM]
  STXR may report false failure in rare timing conditions
  Workaround: Standard retry loop (LDXR/STXR) handles this naturally
```

### Comparing across architecture versions

```
> compare_manual_sections("memory_management")

# Memory Management Comparison

## ARMv7-A
  Short-descriptor (32-bit entries) or Long-descriptor (64-bit, LPAE)
  TTBR0/TTBR1, Domain Access Control Register (DACR)

## ARMv8-A AArch64
  4-level 64-bit descriptors, 3 granule sizes (4KB/16KB/64KB)
  Stage 1 + Stage 2 translation, no DACR

## ARMv7-M
  MPU with 8-16 regions, no virtual memory
  Deterministic memory protection for real-time
```

---

## Quick Setup

```bash
claude mcp add --transport stdio arm-docs-rag -- \
  uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-docs-rag-mcp
```

See the full [Installation Guide](../installation) for other editors and clients.
