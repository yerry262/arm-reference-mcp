---
title: ARM Register Reference MCP
---

# ARM Register Reference MCP

The primary server in the suite. **23 tools** covering the full ARM architecture reference landscape for AArch32 (ARMv7) and AArch64 (ARMv8/v9).

**Server entry point:** `arm-reference-mcp`

---

## Tool Reference

### Register Tools

| Tool | Description |
|------|-------------|
| `lookup_register(name, architecture?)` | Detailed info on a register by name or alias. Supports aliases like FP, LR, IP0, WZR. Returns bit width, category, description, bit fields, and usage notes. |
| `list_registers(architecture, category?)` | Browse all registers for an architecture. Categories: `general_purpose`, `status`, `system`, `floating_point`. Omit category to list all. |
| `search_registers(query, architecture?)` | Keyword search across register names, aliases, descriptions, and usage notes. Returns matching registers with context snippets. |
| `decode_register_value(register_name, hex_value, architecture?)` | Decode a raw hex value against a register's bit fields. Works with CPSR, SCTLR_EL1, FPSCR, DAIF, NZCV, CurrentEL, TTBR0_EL1, FPSR, FPCR, and more. |

**Register coverage:**

| Architecture | Registers |
|-------------|-----------|
| AArch32 | R0-R12, R13/SP, R14/LR, R15/PC, CPSR (12 bit fields), SPSR, FPSCR (7 bit fields), S0-S31, D0-D31 |
| AArch64 | X0-X30 (with W aliases), SP/WSP, PC, XZR/WZR, NZCV, DAIF, CurrentEL, FPCR, FPSR, VBAR_EL1, TTBR0_EL1, SCTLR_EL1 |

113 register definitions total, all with structured metadata and many with full bit field decode support.

---

### Instruction & Convention Tools

| Tool | Description |
|------|-------------|
| `decode_instruction(hex_value)` | Decode a 32-bit AArch32 instruction from hex. Breaks it into condition code, instruction type (data processing, branch, load/store, SWI), opcode, registers, immediates, and shift operands. |
| `explain_condition_code(suffix)` | Reference for all 15 ARM condition codes: EQ, NE, CS/HS, CC/LO, MI, PL, VS, VC, HI, LS, GE, LT, GT, LE, AL. Returns the full name, NZCV flags tested, flag condition, opposite code, and an example. |
| `explain_calling_convention(architecture)` | Complete AAPCS32 or AAPCS64 reference: argument registers, return value registers, caller/callee-saved sets, stack alignment, frame pointer, and special notes (X8 indirect result, X16/X17 linker scratch, X18 platform register, PAC). |

**Example -- decode an instruction:**

```
> decode_instruction("0xE3A01005")

# ARM AArch32 Instruction Decode: 0xE3A01005
  Condition: AL (Always)
  Type: Data Processing
  Mnemonic: MOV
  Rd: R1
  Operand2: immediate value = 5

  Result: MOV R1, #5
```

---

### System Architecture Tools

| Tool | Description |
|------|-------------|
| `explain_exception_levels(architecture?)` | EL0-EL3 for AArch64 (with key registers, vector table layout, security states) or the seven processor modes for AArch32. Defaults to AArch64. |
| `explain_security_model(architecture)` | TrustZone (Secure/Non-secure worlds, SCR_EL3.NS), Secure ELs, ARMv9 RME (four-world model, GPT, GPC), and Arm CCA (RMM, Realm VMs, hardware attestation). |
| `explain_page_table_format(granule_size, va_bits?)` | Page table translation for 4KB, 16KB, or 64KB granules with 39, 48, or 52-bit VAs. Shows level structure, VA bit layout diagram, PTE fields, OA ranges, and TCR_EL1 configuration. |
| `explain_memory_attributes(topic?)` | Memory attributes reference. Topics: `cacheability` (WB/WT/NC, MAIR encodings), `shareability` (ISH/OSH/NSH), `access_permissions` (AP, PXN, UXN), `mair` (register encoding), or omit for overview. Also covers all 4 device memory types (nGnRnE through GRE). |

---

### Architecture & Extensions Tools

| Tool | Description |
|------|-------------|
| `explain_extension(extension_name)` | Detailed reference for 17 ARM extensions: **SVE**, **MTE**, **PAC**, **BTI**, **TME**, **RME**, **SME**, **GCS**, FEAT_THE, FEAT_NV2, **DIT**, **MPAM**, **RAS**, **SPE**, **AMU**, **BRBE**. Returns architecture version, purpose, key registers/instructions, detection method, and use cases. |
| `compare_architecture_versions(version, compare_to?)` | List features for ARMv8.0-A through ARMv9.5-A. Pass two versions for a side-by-side diff of added features. Shows mandatory/optional features, notable changes, and example cores. |

**Example -- extension lookup:**

```
> explain_extension("MTE")

# Memory Tagging Extension (MTE)
Introduced in: ARMv8.5-A (optional)

## Purpose
Hardware-assisted memory safety. Tags every 16-byte granule of memory
with a 4-bit tag. Pointer tags must match memory tags on access.
Catches use-after-free, buffer overflow, and other memory bugs.

## Key Registers
  TFSR_EL1: Tag Fault Status Register
  TFSRE0_EL1: Tag Fault Status (EL0)
  GCR_EL1: Tag randomization control
  RGSR_EL1: Random Allocation Tag Seed Register

## Detection
  ID_AA64PFR1_EL1, MTE field (bits [11:8])
  Linux: HWCAP2_MTE
```

---

### Core/IP & Programming Tools

| Tool | Description |
|------|-------------|
| `lookup_core(core_name)` | Reference card for any ARM core: architecture version, pipeline depth, decode width, key features, target market, and notable SoCs. Covers Cortex-A, Cortex-X, Cortex-R, Cortex-M, and Neoverse. Accepts short names like "A78", "X4", "M55", "N2". |
| `compare_cores(core_a, core_b)` | Side-by-side comparison of two cores: architecture, pipeline, decode width, features, market, generation, and SoCs. |
| `show_assembly_pattern(pattern_name, architecture?)` | Annotated assembly for 12 patterns: `function_prologue`, `function_epilogue`, `atomic_add`, `atomic_cas`, `spinlock_acquire`, `spinlock_release`, `context_switch`, `syscall`, `tlb_invalidate`, `cache_clean`, `enable_mmu`, `exception_vector`. Available for both AArch32 and AArch64. |
| `explain_barrier(barrier_type)` | Memory barrier reference: **DMB**, **DSB**, **ISB**, **LDAR**, **STLR**, **LDAPR**, **SB**, **CSDB**, **SSBB**, **PSSBB**. Covers ordering semantics, domain options, acquire/release patterns, and Spectre mitigation. Pass `"overview"` for a comparison of all barrier types. |

**Example -- assembly pattern:**

```
> show_assembly_pattern("spinlock_acquire", "aarch64")

// === AArch64 Spinlock Acquire ===
    MOV   W2, #1                // W2 = locked value
    SEVL                        // Send Event Locally
1:  WFE                          // Wait For Event: low-power spin
    LDAXR  W3, [X0]             // Load-Exclusive with Acquire
    CBNZ   W3, 1b               // If locked, spin
    STXR   W3, W2, [X0]         // Store-Exclusive: try to acquire
    CBNZ   W3, 1b               // If store failed, retry

// ARMv8.1 LSE alternative:
//   SWPA W2, W3, [X0]          // Atomic swap with Acquire
//   CBNZ W3, 1b
```

---

### SIMD, Optimization & Porting Tools

| Tool | Description |
|------|-------------|
| `explain_neon_intrinsic(intrinsic_name)` | Look up a NEON/ASIMD intrinsic: C signature, assembly instruction, data types, category, latency, throughput, and usage example. 40+ intrinsics covering arithmetic, loads/stores, comparisons, conversions, shuffles, and dot products. Pass `"list"` for all intrinsics by category. |
| `explain_sme_tile(operation)` | SME (Scalable Matrix Extension) reference. 10 topics: `overview`, `za_storage`, `outer_product` (FMOPA/FMOPS), `streaming_mode`, `sme2`, `programming_model`, `tile_load_store`, `mopa_fmopa`, `context_switching`, `detection`. |
| `suggest_optimization(code_pattern, target_core?)` | ARM-specific optimization suggestions for 12 code patterns: `matrix_multiply`, `memcpy`, `memset`, `dot_product`, `sort`, `string_search`, `crc32`, `aes_encrypt`, `sha256`, `linked_list_traversal`, `atomic_counter`, `simd_reduction`. Optional core-specific tips. |
| `lookup_system_register(register, el?)` | Full AArch64 system register reference. Look up individual registers (TCR_EL1, HCR_EL2, MAIR_EL1, etc.) or browse by category: `"list"`, `"memory"`, `"timer"`, `"id"`, `"perf"`. 50+ registers covered. Optional EL filter. |
| `explain_performance_counter(event_name)` | ARM PMU event reference. Look up by name (L1D_CACHE_REFILL, CPU_CYCLES) or hex (0x03). Returns what it measures, when to use it, formulas, Linux perf commands, and AI/ML tips. Pass `"topdown"` for the ARM Top-Down methodology guide. 25+ events across cache, branch, pipeline, memory, and TLB categories. |
| `translate_intrinsic(intrinsic, from_arch, to_arch)` | Translate between x86 SSE/AVX and ARM NEON/SVE. Supports both directions. 30+ translations covering SSE, SSE2, AVX, AVX2, AVX-512 to NEON/SVE. Highlights porting gotchas (operand order, width differences, missing equivalents). |

**Example -- intrinsic translation:**

```
> translate_intrinsic("_mm_add_ps", "x86", "neon")

# Intrinsic Translation: _mm_add_ps -> vaddq_f32

## x86 (SSE)
  Intrinsic: _mm_add_ps
  Instruction: ADDPS xmm, xmm

## ARM NEON Equivalent
  Intrinsic: vaddq_f32
  Instruction: FADD Vd.4S, Vn.4S, Vm.4S

## Data Type
  4x float32 (128-bit)

## Porting Notes
  Direct 1:1 mapping. Same width, same semantics.
```

---

## Quick Setup

```bash
claude mcp add --transport stdio arm-reference -- \
  uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-reference-mcp
```

See the full [Installation Guide](../installation) for other editors and clients.
