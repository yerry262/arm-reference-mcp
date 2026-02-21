"""ARM Documentation RAG MCP Server.

Provides seven tools for searching and understanding ARM architecture documentation:
  - search_arm_docs:              Search ARM documentation entries by keyword and scope.
  - explain_arm_concept:          Explain ARM architecture concepts in detail.
  - find_register_in_manual:      Find which manual section documents a register.
  - get_errata:                   Get known errata for ARM cores.
  - compare_manual_sections:      Compare how a topic differs across architecture versions.
  - list_arm_documents:           Browse the ARM documentation catalog by scope/architecture.
  - explain_instruction_encoding: Explain A64, T32, or A32 instruction set encoding formats.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "ARM Documentation RAG",
    instructions="Search and explain ARM architecture documentation, errata, and manual references.",
)


# ---------------------------------------------------------------------------
# Tool 1: search_arm_docs — inline documentation database
# ---------------------------------------------------------------------------

ARM_DOCS_DB: list[dict] = [
    # -- Architecture Reference Manuals --
    {
        "doc_id": "DDI0487",
        "title": "ARM Architecture Reference Manual for A-profile architecture",
        "scope": "architecture",
        "architecture": "aarch64",
        "sections": [
            {"section": "A1", "topic": "Introduction to the ARM Architecture",
             "keywords": ["overview", "profiles", "a-profile", "architecture versions"],
             "summary": "Introduces ARM architecture profiles (A, R, M), versioning from ARMv8.0-A through ARMv9.x-A, and the AArch64/AArch32 execution states."},
            {"section": "B2", "topic": "The AArch64 Application Level Memory Model",
             "keywords": ["memory model", "memory ordering", "normal memory", "device memory", "shareability"],
             "summary": "Defines memory types (Normal, Device), cacheability attributes, shareability domains (Non-shareable, Inner Shareable, Outer Shareable, Full System), and the memory ordering rules for AArch64."},
            {"section": "C5", "topic": "The A64 Instruction Set",
             "keywords": ["a64", "instructions", "encoding", "data processing", "branch", "load", "store"],
             "summary": "Complete A64 instruction set encoding and behavior. Covers data processing, branch, load/store, SIMD/FP, SVE, and system instructions with their binary encodings."},
            {"section": "D1", "topic": "The AArch64 System Level Programmers Model",
             "keywords": ["exception levels", "EL0", "EL1", "EL2", "EL3", "system registers", "privilege"],
             "summary": "Defines exception levels EL0-EL3, the register file at each EL, system register access rules, and the privilege model for AArch64 execution."},
            {"section": "D4", "topic": "The AArch64 Virtual Memory System Architecture",
             "keywords": ["vmsa", "page tables", "translation", "TLB", "granule", "TTBR", "TCR"],
             "summary": "AArch64 VMSA: translation table formats for 4KB/16KB/64KB granules, multi-level page tables, translation regimes (EL1&0, EL2&0, EL2, EL3), TLB management, and address tagging."},
            {"section": "D5", "topic": "The AArch64 Virtual Memory System Architecture — Address Translation",
             "keywords": ["address translation", "IPA", "PA", "stage 1", "stage 2", "secure", "non-secure"],
             "summary": "Stage 1 and Stage 2 address translation, Intermediate Physical Addresses (IPA), Secure and Non-secure translation regimes, and VMID/ASID tagging."},
            {"section": "D7", "topic": "The Performance Monitors Extension",
             "keywords": ["PMU", "performance counters", "PMCCNTR", "PMCR", "events", "profiling"],
             "summary": "ARM Performance Monitors extension: programmable event counters, cycle counter (PMCCNTR), event types, counter overflow interrupts, and profiling support."},
            {"section": "D8", "topic": "The AArch64 Debug Architecture",
             "keywords": ["debug", "breakpoints", "watchpoints", "self-hosted debug", "external debug", "EDSCR"],
             "summary": "Debug architecture for AArch64: breakpoint/watchpoint registers, self-hosted vs external debug, debug exceptions, Debug Communications Channel (DCC), and trace support."},
            {"section": "E1", "topic": "The AArch32 Application Level Programmers Model",
             "keywords": ["aarch32", "ARM state", "Thumb state", "T32", "A32", "registers"],
             "summary": "AArch32 execution state: ARM (A32) and Thumb (T32) instruction sets, general-purpose registers R0-R15, CPSR, banked registers, and interworking."},
            {"section": "G1", "topic": "The AArch32 System Level Programmers Model",
             "keywords": ["aarch32", "processor modes", "USR", "FIQ", "IRQ", "SVC", "ABT", "UND", "MON", "HYP"],
             "summary": "AArch32 processor modes, mode-specific banked registers, coprocessor access (CP14, CP15), and the relationship between AArch32 modes and AArch64 exception levels."},
        ],
    },
    {
        "doc_id": "DDI0406",
        "title": "ARM Architecture Reference Manual ARMv7-A and ARMv7-R edition",
        "scope": "architecture",
        "architecture": "aarch32",
        "sections": [
            {"section": "A2", "topic": "Application Level Programmers Model",
             "keywords": ["registers", "CPSR", "APSR", "data types", "ARM state", "Thumb state"],
             "summary": "ARMv7 application-level model: register file (R0-R15), CPSR/APSR, data types, ARM and Thumb execution states, and the Jazelle extension."},
            {"section": "A3", "topic": "Application Level Memory Model",
             "keywords": ["memory model", "alignment", "endianness", "memory types", "ordering"],
             "summary": "ARMv7 memory model: alignment requirements, big-endian/little-endian support, Normal/Device/Strongly-ordered memory types, and memory ordering rules."},
            {"section": "A4", "topic": "The Instruction Sets",
             "keywords": ["ARM instructions", "Thumb instructions", "A32", "T32", "encoding"],
             "summary": "ARMv7 A32 and T32 instruction set encodings, conditional execution, IT blocks for Thumb-2, and instruction set state switching."},
            {"section": "B1", "topic": "System Level Programmers Model",
             "keywords": ["processor modes", "exceptions", "vectors", "banked registers", "security extensions"],
             "summary": "ARMv7 system-level model: seven processor modes (USR/FIQ/IRQ/SVC/ABT/UND/SYS) plus Monitor and Hyp modes, exception handling, vector tables, and Security Extensions (TrustZone)."},
            {"section": "B3", "topic": "Virtual Memory System Architecture",
             "keywords": ["vmsa", "page tables", "sections", "supersections", "TTBR", "short descriptor", "long descriptor"],
             "summary": "ARMv7 VMSA: Short-descriptor (32-bit) and Long-descriptor (LPAE, 40-bit PA) translation table formats, sections, supersections, and TLB management."},
        ],
    },
    # -- Cortex-M Technical Reference Manuals --
    {
        "doc_id": "DDI0553",
        "title": "ARM Cortex-M33 Processor Technical Reference Manual",
        "scope": "processor_trm",
        "architecture": "armv8-m",
        "sections": [
            {"section": "Chapter 2", "topic": "Programmers Model",
             "keywords": ["cortex-m33", "registers", "MSP", "PSP", "CONTROL", "PRIMASK", "BASEPRI", "FAULTMASK"],
             "summary": "Cortex-M33 programmers model: core registers (R0-R12, SP, LR, PC, xPSR), dual stack pointers (MSP/PSP), special registers, and privilege levels."},
            {"section": "Chapter 3", "topic": "Memory Model",
             "keywords": ["cortex-m33", "memory map", "MPU", "SAU", "IDAU", "memory protection"],
             "summary": "Cortex-M33 memory model: default memory map (Code, SRAM, Peripheral, External RAM/Device, PPB, System), MPU, Security Attribution Unit (SAU), and IDAU."},
            {"section": "Chapter 5", "topic": "TrustZone for ARMv8-M",
             "keywords": ["trustzone", "secure", "non-secure", "SAU", "NSC", "SG instruction", "BXNS", "BLXNS"],
             "summary": "TrustZone for Cortex-M: Secure/Non-secure states, SAU/IDAU configuration, Non-secure Callable (NSC) regions, SG instruction for secure gateways, and transition mechanisms."},
        ],
    },
    {
        "doc_id": "DDI0403",
        "title": "ARMv7-M Architecture Reference Manual",
        "scope": "architecture",
        "architecture": "armv7-m",
        "sections": [
            {"section": "B1", "topic": "System Level Programmers Model",
             "keywords": ["exception model", "NVIC", "vector table", "priority", "tail-chaining", "preemption"],
             "summary": "ARMv7-M exception model: Nested Vectored Interrupt Controller (NVIC), priority levels, tail-chaining, late arrival optimization, vector table relocation, and fault exceptions."},
            {"section": "B3", "topic": "System Address Map",
             "keywords": ["memory map", "SCS", "NVIC registers", "SysTick", "MPU", "debug"],
             "summary": "ARMv7-M fixed memory map: Code (0x00000000), SRAM (0x20000000), Peripheral (0x40000000), External (0x60000000-0x9FFFFFFF), Private Peripheral Bus (0xE0000000), and System Control Space."},
        ],
    },
    # -- Software/Optimization Guides --
    {
        "doc_id": "DEN0024",
        "title": "ARM Cortex-A Series Programmer's Guide for ARMv8-A",
        "scope": "programming_guide",
        "architecture": "aarch64",
        "sections": [
            {"section": "Chapter 5", "topic": "AArch64 Floating-point and NEON",
             "keywords": ["NEON", "SIMD", "floating-point", "FPCR", "FPSR", "vector", "intrinsics"],
             "summary": "AArch64 NEON/ASIMD programming: 32x 128-bit V registers, scalar and vector operations, FPCR/FPSR control, data types, and C intrinsics for NEON."},
            {"section": "Chapter 9", "topic": "Caches",
             "keywords": ["cache", "L1", "L2", "cache maintenance", "PoC", "PoU", "VIPT", "PIPT", "coherency"],
             "summary": "ARM cache architecture: L1 I-cache/D-cache, L2 unified cache, VIPT/PIPT tagging, cache maintenance operations (by VA to PoC/PoU, by Set/Way), and cache coherency protocols."},
            {"section": "Chapter 10", "topic": "Memory Management Unit",
             "keywords": ["MMU", "TLB", "page tables", "address spaces", "ASID", "VMID"],
             "summary": "MMU programming guide: translation table setup, ASID/VMID usage, TLB invalidation, break-before-make sequences, and contiguous bit optimization."},
            {"section": "Chapter 12", "topic": "Memory Ordering",
             "keywords": ["memory ordering", "barriers", "DMB", "DSB", "ISB", "acquire", "release", "observer"],
             "summary": "Memory ordering in ARMv8-A: weakly-ordered model, barrier instructions (DMB, DSB, ISB), acquire/release semantics (LDAR/STLR), and multi-copy atomicity."},
            {"section": "Chapter 14", "topic": "Multi-core Processors",
             "keywords": ["multi-core", "SMP", "coherency", "snoop", "MESI", "GIC", "SGI", "IPI"],
             "summary": "Multi-core ARM programming: cache coherency (ACE/CHI), snoop control, MESI protocol, Generic Interrupt Controller (GIC), Software Generated Interrupts (SGI), and spin-lock patterns."},
        ],
    },
    {
        "doc_id": "DEN0013",
        "title": "ARM Cortex-A Series Programmer's Guide for ARMv7-A",
        "scope": "programming_guide",
        "architecture": "aarch32",
        "sections": [
            {"section": "Chapter 5", "topic": "Introduction to Floating-point and NEON",
             "keywords": ["VFP", "NEON", "SIMD", "floating-point", "D registers", "Q registers"],
             "summary": "ARMv7 NEON/VFP: 16x 128-bit Q registers (or 32x 64-bit D registers), VFPv3/VFPv4 floating-point, NEON data processing, and intrinsics usage."},
            {"section": "Chapter 7", "topic": "Caches",
             "keywords": ["cache", "L1", "L2", "write-back", "write-through", "cache maintenance"],
             "summary": "ARMv7 cache model: L1 I/D caches, L2 unified cache, write-back/write-through policies, cache maintenance by MVA/set/way, and cache lockdown."},
        ],
    },
    # -- Security Documents --
    {
        "doc_id": "DEN0115",
        "title": "ARM Realm Management Extension (RME) Specification",
        "scope": "security",
        "architecture": "aarch64",
        "sections": [
            {"section": "Chapter 2", "topic": "RME Architecture Overview",
             "keywords": ["RME", "realm", "granule protection", "GPT", "PAS", "root", "realm world"],
             "summary": "Realm Management Extension: four Physical Address Spaces (Secure, Non-secure, Realm, Root), Granule Protection Tables (GPT), and the Realm Management Monitor (RMM)."},
            {"section": "Chapter 3", "topic": "Granule Protection Tables",
             "keywords": ["GPT", "GPCCR", "GPTBR", "granule protection check", "GPC fault"],
             "summary": "GPT structure and operation: two-level lookup (L0/L1), GPCCR_EL3 configuration, GPTBR_EL3 base address, GPC faults, and transition between PAS."},
        ],
    },
    {
        "doc_id": "DEN0022",
        "title": "ARM Power State Coordination Interface (PSCI)",
        "scope": "firmware",
        "architecture": "aarch64",
        "sections": [
            {"section": "Chapter 5", "topic": "PSCI Functions",
             "keywords": ["PSCI", "CPU_ON", "CPU_OFF", "CPU_SUSPEND", "SYSTEM_RESET", "AFFINITY_INFO"],
             "summary": "PSCI function interface: CPU_ON/OFF/SUSPEND for CPU power management, SYSTEM_RESET/SYSTEM_OFF, AFFINITY_INFO for topology queries, and SMC/HVC calling conventions."},
        ],
    },
    # -- Performance/Optimization --
    {
        "doc_id": "SWOG",
        "title": "ARM Neoverse N2 Software Optimization Guide",
        "scope": "optimization",
        "architecture": "aarch64",
        "sections": [
            {"section": "Chapter 3", "topic": "Instruction Latencies and Throughput",
             "keywords": ["latency", "throughput", "pipeline", "dispatch", "execution units", "Neoverse N2"],
             "summary": "Neoverse N2 micro-architecture: instruction latencies for integer/FP/NEON/SVE, pipeline stages, dispatch width, execution unit mapping, and throughput bottleneck analysis."},
            {"section": "Chapter 4", "topic": "NEON and SVE Optimization",
             "keywords": ["NEON", "SVE", "vectorization", "predication", "gather", "scatter", "auto-vectorize"],
             "summary": "NEON/SVE optimization for Neoverse N2: vector length agnostic programming (SVE), predicated operations, gather/scatter loads, and compiler auto-vectorization hints."},
            {"section": "Chapter 5", "topic": "Branch Prediction and Code Layout",
             "keywords": ["branch prediction", "BTB", "TAGE", "code layout", "alignment", "hot path"],
             "summary": "Branch prediction on Neoverse N2: BTB structure, TAGE predictor, indirect branch prediction, code layout optimization, function alignment, and hot/cold path splitting."},
        ],
    },
    # -- SVE/SME Programming Guides --
    {
        "doc_id": "DEN0065",
        "title": "ARM Scalable Vector Extension (SVE) Programming Guide",
        "scope": "programming_guide",
        "architecture": "aarch64",
        "sections": [
            {"section": "Chapter 2", "topic": "SVE Overview and Vector Length Agnostic Programming",
             "keywords": ["SVE", "VLA", "predicate", "Z registers", "P registers", "FFR", "vector length"],
             "summary": "SVE fundamentals: Vector Length Agnostic (VLA) programming model, Z registers (128-2048 bits), predicate registers (P0-P15), First Fault Register (FFR), and RDVL/CNTB instructions."},
            {"section": "Chapter 4", "topic": "SVE Memory Access Patterns",
             "keywords": ["SVE", "gather", "scatter", "first-fault", "non-fault", "contiguous", "predicated load"],
             "summary": "SVE memory access: contiguous/non-contiguous loads and stores, gather/scatter operations, first-fault and non-faulting loads for speculative access, and predicated memory operations."},
        ],
    },
    # -- GIC Specification --
    {
        "doc_id": "IHI0069",
        "title": "ARM Generic Interrupt Controller Architecture Specification (GICv3/GICv4)",
        "scope": "architecture",
        "architecture": "aarch64",
        "sections": [
            {"section": "Chapter 2", "topic": "GIC Architecture Overview",
             "keywords": ["GIC", "distributor", "redistributor", "CPU interface", "SPI", "PPI", "SGI", "LPI"],
             "summary": "GICv3 architecture: Distributor (GICD), Redistributor (GICR), CPU Interface (ICC), interrupt types (SPI, PPI, SGI, LPI), affinity routing, and system register access."},
            {"section": "Chapter 4", "topic": "Interrupt Handling and Priority",
             "keywords": ["interrupt priority", "preemption", "priority drop", "EOI", "running priority", "binary point"],
             "summary": "GIC interrupt handling: priority levels, preemption, group 0/1 interrupts, priority drop vs deactivation split, End of Interrupt (EOI) modes, and interrupt masking."},
        ],
    },
]


_VALID_SCOPES = {"architecture", "processor_trm", "programming_guide", "security", "firmware", "optimization"}


@mcp.tool()
def search_arm_docs(query: str, doc_scope: str | None = None, architecture: str | None = None) -> str:
    """Search ARM documentation entries by keyword.

    Searches across ARM Architecture Reference Manuals, Technical Reference
    Manuals, Programming Guides, Security specs, and Optimization Guides.

    Args:
        query: Search keyword or phrase (e.g. "page tables", "NEON", "TrustZone",
               "exception levels", "cache coherency"). Case-insensitive.
        doc_scope: Optional scope filter. One of: "architecture", "processor_trm",
                   "programming_guide", "security", "firmware", "optimization".
        architecture: Optional architecture filter (e.g. "aarch64", "aarch32",
                      "armv7-m", "armv8-m").
    """
    if doc_scope and doc_scope not in _VALID_SCOPES:
        return f"Error: doc_scope must be one of {', '.join(sorted(_VALID_SCOPES))}."

    query_lower = query.lower()
    query_terms = query_lower.split()

    results = []
    for doc in ARM_DOCS_DB:
        if doc_scope and doc["scope"] != doc_scope:
            continue
        if architecture and doc["architecture"] != architecture.lower():
            continue
        for sec in doc["sections"]:
            # Score: count how many query terms match in keywords, topic, or summary
            score = 0
            searchable = (
                " ".join(sec["keywords"]).lower() + " " +
                sec["topic"].lower() + " " +
                sec["summary"].lower()
            )
            for term in query_terms:
                if term in searchable:
                    score += 1
                    # Bonus for keyword exact match
                    if term in [k.lower() for k in sec["keywords"]]:
                        score += 2
            if score > 0:
                results.append((score, doc, sec))

    if not results:
        return (
            f"No documentation found matching '{query}'"
            + (f" in scope '{doc_scope}'" if doc_scope else "")
            + (f" for {architecture}" if architecture else "")
            + ".\n\nTry broader terms or remove filters. Available scopes: "
            + ", ".join(sorted(_VALID_SCOPES))
        )

    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)

    lines = [f"# ARM Documentation Search: \"{query}\""]
    if doc_scope:
        lines.append(f"Scope: {doc_scope}")
    if architecture:
        lines.append(f"Architecture: {architecture}")
    lines.append(f"Found {len(results)} matching section(s)\n")

    for score, doc, sec in results[:10]:  # Top 10
        lines.append(f"## [{doc['doc_id']}] {doc['title']}")
        lines.append(f"Section {sec['section']}: {sec['topic']}")
        lines.append(f"Architecture: {doc['architecture']}  |  Scope: {doc['scope']}")
        lines.append(f"Keywords: {', '.join(sec['keywords'])}")
        lines.append(f"\n{sec['summary']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2: explain_arm_concept — inline concept database
# ---------------------------------------------------------------------------

ARM_CONCEPTS: dict[str, dict] = {
    "exception_levels": {
        "name": "Exception Levels (EL0-EL3)",
        "architecture": "aarch64",
        "category": "system",
        "explanation": (
            "ARM AArch64 defines four exception levels (ELs) that form a privilege hierarchy:\n\n"
            "**EL0 (User/Application):**\n"
            "- Lowest privilege level. Runs user-space applications.\n"
            "- Cannot access most system registers. Uses virtual addresses.\n"
            "- Generates Supervisor Call (SVC) exceptions to request OS services.\n\n"
            "**EL1 (OS Kernel):**\n"
            "- Runs the operating system kernel.\n"
            "- Controls Stage 1 page tables (TTBR0_EL1, TTBR1_EL1), exception handling (VBAR_EL1).\n"
            "- Manages EL0 processes, handles IRQ/FIQ (if not routed to EL2/EL3).\n\n"
            "**EL2 (Hypervisor):**\n"
            "- Runs hypervisors (KVM, Xen, Hyp-V for ARM).\n"
            "- Controls Stage 2 address translation (VTTBR_EL2) for VM isolation.\n"
            "- Can trap EL1 system register accesses (HCR_EL2), intercept instructions.\n"
            "- VHE (Virtualization Host Extensions) allows a host OS kernel to run at EL2 efficiently.\n\n"
            "**EL3 (Secure Monitor / Firmware):**\n"
            "- Highest privilege. Runs secure firmware (ARM Trusted Firmware / TF-A).\n"
            "- Controls transitions between Secure and Non-secure worlds (SCR_EL3.NS bit).\n"
            "- With RME: also controls Realm world transitions.\n"
            "- Handles Secure Monitor Calls (SMC) from lower ELs.\n\n"
            "**Key principles:**\n"
            "- Higher ELs can access all state visible to lower ELs.\n"
            "- Exception entry always goes to the same or a higher EL.\n"
            "- Exception return (ERET) can go to the same or a lower EL.\n"
            "- Each EL has its own SP (SP_ELx), SPSR, ELR, and dedicated system registers."
        ),
        "related_registers": ["CurrentEL", "SCR_EL3", "HCR_EL2", "SPSR_ELx", "ELR_ELx", "VBAR_ELx"],
        "doc_references": ["DDI0487 D1", "DEN0024 Chapter 3"],
    },
    "trustzone": {
        "name": "TrustZone Security Technology",
        "architecture": "aarch64",
        "category": "security",
        "explanation": (
            "TrustZone partitions the system into two isolated worlds:\n\n"
            "**Secure World:**\n"
            "- Runs trusted firmware, secure OS (OP-TEE), and Trusted Applications (TAs).\n"
            "- Has access to all memory and peripherals.\n"
            "- Controlled by EL3 firmware and Secure EL1.\n\n"
            "**Non-secure (Normal) World:**\n"
            "- Runs the main OS (Linux, Windows, etc.) and user applications.\n"
            "- Cannot access Secure world memory or peripherals.\n"
            "- Uses SMC (Secure Monitor Call) to request secure services.\n\n"
            "**Hardware enforcement:**\n"
            "- TZASC (TrustZone Address Space Controller): partitions DRAM regions.\n"
            "- TZPC (TrustZone Protection Controller): partitions peripheral access.\n"
            "- TZC-400: memory controller-level security with region-based access control.\n"
            "- NS bit propagates on the bus — hardware blocks non-secure access to secure resources.\n\n"
            "**Transition mechanism:**\n"
            "- SMC instruction from Non-secure EL1 -> traps to EL3 Secure Monitor.\n"
            "- Monitor saves Non-secure state, restores Secure state, switches SCR_EL3.NS.\n"
            "- ERET returns to appropriate world.\n\n"
            "**TrustZone for Cortex-M (ARMv8-M):**\n"
            "- Uses Security Attribution Unit (SAU) and Implementation Defined Attribution Unit (IDAU).\n"
            "- Memory is divided into Secure, Non-secure, and Non-secure Callable (NSC) regions.\n"
            "- SG (Secure Gateway) instruction for secure entry points.\n"
            "- Hardware stacks context automatically on Secure/Non-secure transitions."
        ),
        "related_registers": ["SCR_EL3", "TZASC", "SAU_CTRL", "SAU_RNR", "SAU_RBAR", "SAU_RLAR"],
        "doc_references": ["DDI0487 D1.5", "DDI0553 Chapter 5", "DEN0115"],
    },
    "vmsa": {
        "name": "Virtual Memory System Architecture (VMSA)",
        "architecture": "aarch64",
        "category": "memory",
        "explanation": (
            "The AArch64 VMSA defines how virtual addresses are translated to physical addresses:\n\n"
            "**Translation Table Formats:**\n"
            "- 4KB granule: 4 levels of lookup (L0-L3), supports up to 48-bit or 52-bit VA/PA.\n"
            "- 16KB granule: up to 4 levels, 47-bit or 52-bit VA.\n"
            "- 64KB granule: 3 levels (L1-L3), 48-bit or 52-bit VA.\n\n"
            "**Translation Regimes:**\n"
            "- EL1&0: used by OS. TTBR0_EL1 (user space, bottom of VA), TTBR1_EL1 (kernel, top of VA).\n"
            "- EL2 or EL2&0 (with VHE): used by hypervisor. TTBR0_EL2, VTTBR_EL2 (Stage 2).\n"
            "- EL3: TTBR0_EL3 (secure firmware).\n\n"
            "**Stage 1 & Stage 2 Translation:**\n"
            "- Stage 1: VA -> IPA (Intermediate Physical Address). Controlled by guest OS.\n"
            "- Stage 2: IPA -> PA (Physical Address). Controlled by hypervisor for VM isolation.\n"
            "- Two-stage translation enables hardware-assisted virtualization.\n\n"
            "**Page Table Entry Attributes:**\n"
            "- AP (Access Permission): read/write, read-only, EL0 access control.\n"
            "- SH (Shareability): Non-shareable, Inner Shareable, Outer Shareable.\n"
            "- AttrIndx: indexes into MAIR_ELx for memory type (Normal, Device).\n"
            "- nG (non-Global): if set, entry is tagged with ASID.\n"
            "- AF (Access Flag): set by hardware or generates fault on first access.\n"
            "- Contiguous bit: hint that adjacent entries map contiguous physical memory.\n"
            "- PXN/UXN/XN: Privileged/Unprivileged/Execute-never controls."
        ),
        "related_registers": ["TTBR0_EL1", "TTBR1_EL1", "TCR_EL1", "MAIR_EL1", "VTTBR_EL2", "SCTLR_EL1"],
        "doc_references": ["DDI0487 D4-D5", "DEN0024 Chapter 10"],
    },
    "memory_ordering": {
        "name": "ARM Memory Ordering Model",
        "architecture": "aarch64",
        "category": "memory",
        "explanation": (
            "ARM uses a weakly-ordered memory model. Loads and stores to Normal memory can be "
            "reordered by the processor for performance. Barriers enforce ordering.\n\n"
            "**Memory Types:**\n"
            "- Normal memory: cacheable, speculative reads allowed, write-combining possible.\n"
            "- Device memory: non-cacheable, no speculation, four sub-types:\n"
            "  - Device-nGnRnE: most restrictive (like x86 UC). No gathering, no reordering, no early ack.\n"
            "  - Device-nGnRE: no gathering, no reordering, early write acknowledgement.\n"
            "  - Device-nGRE: no gathering, reordering allowed, early ack.\n"
            "  - Device-GRE: gathering, reordering, early ack.\n\n"
            "**Barriers:**\n"
            "- DMB (Data Memory Barrier): ensures ordering of memory accesses before/after the barrier.\n"
            "  - DMB ISH/OSH/NSH/SY: scope variants (Inner Shareable, Outer Shareable, Non-shareable, Full System).\n"
            "  - DMB LD/ST: only order loads or stores (not both).\n"
            "- DSB (Data Synchronization Barrier): like DMB but also waits for all pending accesses to complete.\n"
            "- ISB (Instruction Synchronization Barrier): flushes the pipeline and refetches.\n\n"
            "**Acquire/Release Semantics (ARMv8.0+):**\n"
            "- LDAR (Load-Acquire): all subsequent memory accesses are ordered after this load.\n"
            "- STLR (Store-Release): all prior memory accesses are ordered before this store.\n"
            "- LDAPR (Load-AcquirePC, ARMv8.3): weaker acquire, only orders against prior STLR to same address.\n\n"
            "**Multi-copy Atomicity:**\n"
            "- ARMv8.0: other-multi-copy atomic (a write becomes visible to all other observers simultaneously).\n"
            "- ARMv8.4+: full multi-copy atomicity with FEAT_LSMAOC."
        ),
        "related_registers": ["SCTLR_EL1", "MAIR_EL1", "TCR_EL1"],
        "doc_references": ["DDI0487 B2", "DEN0024 Chapter 12"],
    },
    "cache_coherency": {
        "name": "Cache Coherency in ARM",
        "architecture": "aarch64",
        "category": "memory",
        "explanation": (
            "ARM multi-core systems maintain cache coherency through hardware protocols:\n\n"
            "**Coherency Protocols:**\n"
            "- MOESI (Modified, Owned, Exclusive, Shared, Invalid): used by most ARM implementations.\n"
            "- Snoop-based: cores snoop each other's caches via the interconnect.\n"
            "- Directory-based (AMBA CHI): used in larger systems (Neoverse, DynamIQ clusters).\n\n"
            "**Cache Levels and Points:**\n"
            "- PoC (Point of Coherency): point where all agents (CPU, DMA, GPU) see a unified view.\n"
            "- PoU (Point of Unification): point where I-cache and D-cache see the same data (per PE).\n"
            "- PoDP (Point of Deep Persistence): for persistent memory support.\n\n"
            "**Cache Maintenance Operations (AArch64):**\n"
            "- DC CIVAC: Clean and Invalidate by VA to PoC (flush to main memory).\n"
            "- DC CVAC: Clean by VA to PoC (write back dirty data).\n"
            "- DC IVAC: Invalidate by VA to PoC (discard cache line).\n"
            "- DC ZVA: Zero a cache line by VA (fast zeroing without read-for-ownership).\n"
            "- IC IVAU: Invalidate I-cache by VA to PoU (needed after code modification).\n\n"
            "**Self-modifying Code Pattern:**\n"
            "1. Write new instructions to memory (store).\n"
            "2. DC CVAU addr — clean D-cache to PoU.\n"
            "3. DSB ISH — ensure clean completes.\n"
            "4. IC IVAU addr — invalidate I-cache to PoU.\n"
            "5. DSB ISH — ensure invalidation completes.\n"
            "6. ISB — synchronize instruction fetch pipeline.\n\n"
            "**DMA Coherency:**\n"
            "- ACE (AXI Coherency Extension): allows DMA masters to participate in coherency.\n"
            "- ACE-Lite: one-way coherency (DMA can snoop but doesn't maintain cache).\n"
            "- IO-coherent ports: some SoCs provide hardware-coherent DMA ports."
        ),
        "related_registers": ["CTR_EL0", "CLIDR_EL1", "CCSIDR_EL1", "CSSELR_EL1"],
        "doc_references": ["DEN0024 Chapter 9, 14"],
    },
    "sve": {
        "name": "Scalable Vector Extension (SVE/SVE2)",
        "architecture": "aarch64",
        "category": "simd",
        "explanation": (
            "SVE is ARM's scalable SIMD extension, designed for HPC and general-purpose vectorization:\n\n"
            "**Key Design Principles:**\n"
            "- Vector Length Agnostic (VLA): code works across different SVE implementations.\n"
            "- Vector length is 128 to 2048 bits, in 128-bit increments.\n"
            "- Runtime vector length discovery: RDVL, CNTB, CNTW, CNTD instructions.\n\n"
            "**Register File:**\n"
            "- Z0-Z31: scalable vector registers (128-2048 bits each).\n"
            "- P0-P15: predicate registers (1 bit per byte of vector length).\n"
            "- FFR: First Fault Register (for speculative memory access patterns).\n"
            "- ZCR_ELx: vector length control at each exception level.\n\n"
            "**Programming Model:**\n"
            "- Predication: all operations can be predicated with P registers.\n"
            "  - Merging predication (/M): inactive elements retain old values.\n"
            "  - Zeroing predication (/Z): inactive elements set to zero.\n"
            "- WHILELT/WHILELO: loop control instructions for VLA loops.\n"
            "- Gather/scatter: arbitrary indexed memory access.\n"
            "- First-fault loads: speculative access with hardware fault suppression.\n\n"
            "**SVE2 (ARMv9.0-A):**\n"
            "- Adds fixed-point, complex number, and cryptographic operations.\n"
            "- Superset of NEON functionality in SVE encoding.\n"
            "- New histogram, cross-lane permute, and polynomial multiply instructions.\n"
            "- Required in all ARMv9 implementations (SVE1 was optional in ARMv8.2+)."
        ),
        "related_registers": ["ZCR_EL1", "ZCR_EL2", "ZCR_EL3", "ID_AA64ZFR0_EL1"],
        "doc_references": ["DEN0065", "DDI0487 C5 (SVE instructions)"],
    },
    "mte": {
        "name": "Memory Tagging Extension (MTE)",
        "architecture": "aarch64",
        "category": "security",
        "explanation": (
            "MTE (FEAT_MTE, ARMv8.5-A) adds hardware-assisted memory safety:\n\n"
            "**How It Works:**\n"
            "- Every 16 bytes of memory can be assigned a 4-bit tag (stored in dedicated tag RAM).\n"
            "- Pointers carry a 4-bit tag in bits [59:56] (using Top Byte Ignore).\n"
            "- On every memory access, hardware compares pointer tag vs memory tag.\n"
            "- Tag mismatch generates a Tag Check Fault (synchronous or asynchronous).\n\n"
            "**Tag Operations:**\n"
            "- IRG Xd, Xn: Insert Random Tag into pointer.\n"
            "- ADDG Xd, Xn, #uimm, #uimm: Add tag and offset to pointer.\n"
            "- STG Xt, [Xn]: Store Allocation Tag to memory.\n"
            "- LDG Xt, [Xn]: Load Allocation Tag from memory.\n"
            "- STZGM: Store Tag and Zero Multiple (bulk tag+zero).\n\n"
            "**Modes of Operation (SCTLR_EL1.TCF):**\n"
            "- Synchronous (mode 1): immediate fault on mismatch. Best for debugging.\n"
            "- Asynchronous (mode 2): deferred reporting via TFSR_EL1. Lower overhead.\n"
            "- Asymmetric (mode 3): sync for reads, async for writes.\n\n"
            "**Use Cases:**\n"
            "- Detecting use-after-free: freed memory gets a new random tag.\n"
            "- Detecting buffer overflow: adjacent allocations get different tags.\n"
            "- Android uses MTE in production for heap memory protection.\n"
            "- Linux kernel MTE support for kmalloc/kfree."
        ),
        "related_registers": ["SCTLR_EL1", "TCR_EL1", "TFSR_EL1", "TFSRE0_EL1", "GCR_EL1", "RGSR_EL1"],
        "doc_references": ["DDI0487 D8 (MTE)", "Android MTE documentation"],
    },
    "pac": {
        "name": "Pointer Authentication (PAC)",
        "architecture": "aarch64",
        "category": "security",
        "explanation": (
            "PAC (FEAT_PAuth, ARMv8.3-A) adds cryptographic signatures to pointers to "
            "prevent code-reuse attacks:\n\n"
            "**How It Works:**\n"
            "- PAC uses unused upper bits of 64-bit pointers to store a Pointer Authentication Code.\n"
            "- The PAC is a cryptographic hash of: pointer value + 64-bit context + 128-bit key.\n"
            "- On pointer use, the PAC is verified. Mismatch corrupts the pointer -> fault on dereference.\n\n"
            "**Keys (5 keys):**\n"
            "- APIAKey, APIBKey: Instruction Address keys (for return addresses, function pointers).\n"
            "- APDAKey, APDBKey: Data Address keys (for data pointers).\n"
            "- APGAKey: Generic Authentication key (for arbitrary data).\n\n"
            "**Key Instructions:**\n"
            "- PACIA/PACIB: Add PAC to instruction address using Key A/B.\n"
            "- PACDA/PACDB: Add PAC to data address using Key A/B.\n"
            "- AUTIA/AUTIB: Authenticate instruction address (verify and strip PAC).\n"
            "- AUTDA/AUTDB: Authenticate data address.\n"
            "- XPACI/XPACD: Strip PAC without authentication.\n"
            "- RETAA/RETAB: Authenticate-and-return (combined AUT + RET).\n\n"
            "**Compiler Support:**\n"
            "- GCC/Clang: -mbranch-protection=pac-ret signs/authenticates return addresses.\n"
            "- -mbranch-protection=pac-ret+leaf also protects leaf functions.\n"
            "- Linux kernel supports PAC for user-space and kernel-space.\n"
            "- Apple uses PAC extensively in iOS/macOS (ARMv8.3 required on Apple Silicon)."
        ),
        "related_registers": ["APIAKeyLo_EL1", "APIAKeyHi_EL1", "SCTLR_EL1.EnIA", "SCTLR_EL1.EnDA"],
        "doc_references": ["DDI0487 D5 (PAC)", "ARM Learn: Pointer Authentication"],
    },
    "gic": {
        "name": "Generic Interrupt Controller (GICv3/v4)",
        "architecture": "aarch64",
        "category": "system",
        "explanation": (
            "The GIC is ARM's standard interrupt controller for A-profile processors:\n\n"
            "**Components:**\n"
            "- Distributor (GICD): manages SPI routing, enable, priority, and configuration.\n"
            "- Redistributor (GICR): one per PE, handles PPI/SGI/LPI configuration.\n"
            "- CPU Interface (ICC): accessed via system registers (ICC_*), handles interrupt acknowledge/EOI.\n"
            "- ITS (Interrupt Translation Service, GICv3): translates MSI/MSI-X to LPIs.\n\n"
            "**Interrupt Types:**\n"
            "- SGI (Software Generated Interrupt, 0-15): inter-processor interrupts.\n"
            "- PPI (Private Peripheral Interrupt, 16-31): per-PE private (e.g., timer, PMU overflow).\n"
            "- SPI (Shared Peripheral Interrupt, 32-1019): global, routable to any PE.\n"
            "- LPI (Locality-specific Peripheral Interrupt, 8192+): message-based, for PCIe MSI.\n\n"
            "**Key GICv3 Features:**\n"
            "- Affinity routing: use Affinity (Aff3.Aff2.Aff1.Aff0) to target PEs.\n"
            "- System register access: no memory-mapped CPU interface needed.\n"
            "- Security groups: Group 0 (Secure/FIQ), Group 1 Secure, Group 1 Non-secure.\n"
            "- 8-bit priority with binary point for preemption control.\n\n"
            "**GICv4 Additions:**\n"
            "- Direct injection of virtual interrupts (vLPIs) to VMs.\n"
            "- Eliminates hypervisor trap for virtual interrupt delivery.\n"
            "- Doorbell interrupts for signaling blocked vCPUs."
        ),
        "related_registers": ["ICC_IAR1_EL1", "ICC_EOIR1_EL1", "ICC_PMR_EL1", "ICC_SRE_EL1", "ICC_CTLR_EL1"],
        "doc_references": ["IHI0069 (GICv3/v4 spec)"],
    },
    "nvic": {
        "name": "Nested Vectored Interrupt Controller (NVIC)",
        "architecture": "armv7-m",
        "category": "system",
        "explanation": (
            "The NVIC is the interrupt controller for ARM Cortex-M processors:\n\n"
            "**Features:**\n"
            "- Tightly coupled to the processor core (1-cycle latency to start exception entry).\n"
            "- Configurable number of interrupt lines (up to 496 external interrupts).\n"
            "- Configurable priority levels (3-8 bits, implementation-defined).\n"
            "- Automatic state saving: hardware pushes R0-R3, R12, LR, PC, xPSR on entry.\n\n"
            "**Optimization Features:**\n"
            "- Tail-chaining: back-to-back interrupts skip unstacking/restacking.\n"
            "- Late arrival: higher-priority interrupt arriving during stacking pre-empts.\n"
            "- Lazy FP stacking (Cortex-M4/M33): defers FP context save until FP use in ISR.\n\n"
            "**Key Registers (System Control Space):**\n"
            "- NVIC_ISER[n]: Interrupt Set-Enable Registers.\n"
            "- NVIC_ICER[n]: Interrupt Clear-Enable Registers.\n"
            "- NVIC_ISPR[n]: Interrupt Set-Pending Registers.\n"
            "- NVIC_IPR[n]: Interrupt Priority Registers (8-bit per interrupt).\n"
            "- SHPR[n]: System Handler Priority Registers (for faults, SysTick, PendSV).\n\n"
            "**Priority Model:**\n"
            "- Lower numerical value = higher priority.\n"
            "- PRIMASK: masks all interrupts except NMI and HardFault.\n"
            "- BASEPRI: masks interrupts at or below a priority threshold.\n"
            "- FAULTMASK: masks all interrupts except NMI (escalates faults to HardFault)."
        ),
        "related_registers": ["PRIMASK", "BASEPRI", "FAULTMASK", "CONTROL", "SCB_VTOR"],
        "doc_references": ["DDI0403 B1, B3", "DDI0553 Chapter 2"],
    },
    "psci": {
        "name": "Power State Coordination Interface (PSCI)",
        "architecture": "aarch64",
        "category": "firmware",
        "explanation": (
            "PSCI is ARM's standard interface for CPU and system power management:\n\n"
            "**Purpose:**\n"
            "- Provides a consistent API for OS/hypervisor to manage CPU power states.\n"
            "- Abstracts platform-specific power management hardware.\n"
            "- Called via SMC (from EL1) or HVC (from EL1 with EL2 present).\n\n"
            "**Core Functions:**\n"
            "- PSCI_VERSION: query supported PSCI version.\n"
            "- CPU_ON: bring a secondary CPU online at a specified entry point.\n"
            "- CPU_OFF: power down the calling CPU.\n"
            "- CPU_SUSPEND: enter a low-power state (standby or power-down).\n"
            "- AFFINITY_INFO: query power state of a CPU or cluster.\n"
            "- SYSTEM_OFF: shut down the entire system.\n"
            "- SYSTEM_RESET: reset the entire system.\n"
            "- SYSTEM_SUSPEND (PSCI 1.0+): suspend the entire system to RAM.\n\n"
            "**Power State Encoding:**\n"
            "- StateType (bit 16): 0 = standby (WFI-like), 1 = power-down (state lost).\n"
            "- StateID (bits [15:0]): platform-specific power state identifier.\n"
            "- PowerLevel (bits [25:24]): 0 = core, 1 = cluster, 2 = system.\n\n"
            "**Boot Flow:**\n"
            "1. Primary CPU boots from reset, runs firmware/BL1/BL2/BL31.\n"
            "2. BL31 (EL3 firmware) implements PSCI and waits.\n"
            "3. OS calls CPU_ON for each secondary CPU.\n"
            "4. Secondary CPUs start at the specified entry point in EL2 or EL1."
        ),
        "related_registers": ["SCR_EL3", "MPIDR_EL1"],
        "doc_references": ["DEN0022 (PSCI spec)"],
    },
}

# Build lookup index (case-insensitive, supporting aliases)
_CONCEPT_ALIASES: dict[str, str] = {}
for _key in ARM_CONCEPTS:
    _CONCEPT_ALIASES[_key] = _key
    # Common aliases
_CONCEPT_ALIASES["el"] = "exception_levels"
_CONCEPT_ALIASES["exception levels"] = "exception_levels"
_CONCEPT_ALIASES["exception level"] = "exception_levels"
_CONCEPT_ALIASES["el0"] = "exception_levels"
_CONCEPT_ALIASES["el1"] = "exception_levels"
_CONCEPT_ALIASES["el2"] = "exception_levels"
_CONCEPT_ALIASES["el3"] = "exception_levels"
_CONCEPT_ALIASES["tz"] = "trustzone"
_CONCEPT_ALIASES["trust zone"] = "trustzone"
_CONCEPT_ALIASES["tee"] = "trustzone"
_CONCEPT_ALIASES["virtual memory"] = "vmsa"
_CONCEPT_ALIASES["page tables"] = "vmsa"
_CONCEPT_ALIASES["page table"] = "vmsa"
_CONCEPT_ALIASES["mmu"] = "vmsa"
_CONCEPT_ALIASES["translation tables"] = "vmsa"
_CONCEPT_ALIASES["memory order"] = "memory_ordering"
_CONCEPT_ALIASES["barriers"] = "memory_ordering"
_CONCEPT_ALIASES["dmb"] = "memory_ordering"
_CONCEPT_ALIASES["dsb"] = "memory_ordering"
_CONCEPT_ALIASES["isb"] = "memory_ordering"
_CONCEPT_ALIASES["cache"] = "cache_coherency"
_CONCEPT_ALIASES["caches"] = "cache_coherency"
_CONCEPT_ALIASES["coherency"] = "cache_coherency"
_CONCEPT_ALIASES["coherence"] = "cache_coherency"
_CONCEPT_ALIASES["scalable vector"] = "sve"
_CONCEPT_ALIASES["sve2"] = "sve"
_CONCEPT_ALIASES["memory tagging"] = "mte"
_CONCEPT_ALIASES["pointer authentication"] = "pac"
_CONCEPT_ALIASES["pauth"] = "pac"
_CONCEPT_ALIASES["interrupt controller"] = "gic"
_CONCEPT_ALIASES["gicv3"] = "gic"
_CONCEPT_ALIASES["gicv4"] = "gic"
_CONCEPT_ALIASES["interrupts"] = "nvic"
_CONCEPT_ALIASES["power management"] = "psci"
_CONCEPT_ALIASES["cpu_on"] = "psci"


@mcp.tool()
def explain_arm_concept(concept: str) -> str:
    """Explain an ARM architecture concept in detail.

    Provides comprehensive explanations of key ARM concepts including
    exception levels, TrustZone, VMSA, memory ordering, cache coherency,
    SVE, MTE, PAC, GIC, NVIC, and PSCI.

    Args:
        concept: The concept to explain. Examples: "exception_levels",
                 "trustzone", "vmsa", "memory_ordering", "cache_coherency",
                 "sve", "mte", "pac", "gic", "nvic", "psci".
                 Also accepts common aliases like "el", "tz", "page tables",
                 "barriers", "pointer authentication".
    """
    concept_lower = concept.lower().strip()
    key = _CONCEPT_ALIASES.get(concept_lower)
    if key is None:
        # Try partial match
        for alias, k in _CONCEPT_ALIASES.items():
            if concept_lower in alias or alias in concept_lower:
                key = k
                break

    if key is None:
        available = sorted(set(ARM_CONCEPTS.keys()))
        return (
            f"No concept found matching '{concept}'.\n\n"
            f"Available concepts: {', '.join(available)}\n\n"
            "Also accepts aliases like: el, tz, page tables, barriers, "
            "cache, sve, mte, pac, gic, nvic, psci."
        )

    entry = ARM_CONCEPTS[key]
    lines = [f"# {entry['name']}"]
    lines.append(f"Architecture: {entry['architecture']}  |  Category: {entry['category']}")
    lines.append("")
    lines.append(entry["explanation"])
    lines.append("")
    if entry["related_registers"]:
        lines.append("## Related Registers")
        lines.append(", ".join(entry["related_registers"]))
    if entry["doc_references"]:
        lines.append("\n## Documentation References")
        for ref in entry["doc_references"]:
            lines.append(f"- {ref}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: find_register_in_manual — register-to-manual mapping
# ---------------------------------------------------------------------------

REGISTER_MANUAL_MAP: dict[str, dict] = {
    # AArch64 system registers
    "SCTLR_EL1": {
        "full_name": "System Control Register (EL1)",
        "manual": "DDI0487",
        "section": "D13.2.118",
        "context": "system",
        "architecture": "aarch64",
        "summary": "Controls EL1&0 behavior: MMU enable (M bit), alignment checking (A), data/instruction cache enable (C/I), WXN, endianness (EE), UMA, and numerous feature controls.",
        "related": ["SCTLR_EL2", "SCTLR_EL3", "TCR_EL1", "MAIR_EL1"],
    },
    "TCR_EL1": {
        "full_name": "Translation Control Register (EL1)",
        "manual": "DDI0487",
        "section": "D13.2.131",
        "context": "memory",
        "architecture": "aarch64",
        "summary": "Controls EL1&0 address translation: T0SZ/T1SZ (VA range), TG0/TG1 (granule size 4K/16K/64K), SH/ORGN/IRGN (cacheability/shareability), IPS (physical address size), and EPD0/EPD1 (table walk disable).",
        "related": ["TTBR0_EL1", "TTBR1_EL1", "MAIR_EL1"],
    },
    "TTBR0_EL1": {
        "full_name": "Translation Table Base Register 0 (EL1)",
        "manual": "DDI0487",
        "section": "D13.2.144",
        "context": "memory",
        "architecture": "aarch64",
        "summary": "Holds the base address of the translation table for the lower VA range (typically user space, 0x0000...). Also contains ASID field (if TCR_EL1.A1==0).",
        "related": ["TTBR1_EL1", "TCR_EL1"],
    },
    "TTBR1_EL1": {
        "full_name": "Translation Table Base Register 1 (EL1)",
        "manual": "DDI0487",
        "section": "D13.2.145",
        "context": "memory",
        "architecture": "aarch64",
        "summary": "Holds the base address of the translation table for the upper VA range (typically kernel space, 0xFFFF...). Contains ASID field (if TCR_EL1.A1==1).",
        "related": ["TTBR0_EL1", "TCR_EL1"],
    },
    "MAIR_EL1": {
        "full_name": "Memory Attribute Indirection Register (EL1)",
        "manual": "DDI0487",
        "section": "D13.2.97",
        "context": "memory",
        "architecture": "aarch64",
        "summary": "Defines 8 memory attribute encodings (Attr0-Attr7). Each 8-bit field encodes memory type: Normal (cacheable inner/outer write-back/write-through/non-cacheable) or Device (nGnRnE/nGnRE/nGRE/GRE). Page table entries index into MAIR via AttrIndx.",
        "related": ["TCR_EL1", "SCTLR_EL1"],
    },
    "VBAR_EL1": {
        "full_name": "Vector Base Address Register (EL1)",
        "manual": "DDI0487",
        "section": "D13.2.143",
        "context": "exception",
        "architecture": "aarch64",
        "summary": "Holds the base address of the exception vector table for EL1. The table has 16 entries (4 types x 4 sources: current EL SP0, current EL SPx, lower EL AArch64, lower EL AArch32). Each entry is 128 bytes (32 instructions).",
        "related": ["VBAR_EL2", "VBAR_EL3", "ESR_EL1"],
    },
    "ESR_EL1": {
        "full_name": "Exception Syndrome Register (EL1)",
        "manual": "DDI0487",
        "section": "D13.2.37",
        "context": "exception",
        "architecture": "aarch64",
        "summary": "Contains the syndrome (cause) information for exceptions taken to EL1. EC field (bits [31:26]) identifies exception class (SVC, data abort, instruction abort, FP, SVE, etc.). ISS field provides class-specific details.",
        "related": ["FAR_EL1", "VBAR_EL1", "ELR_EL1", "SPSR_EL1"],
    },
    "FAR_EL1": {
        "full_name": "Fault Address Register (EL1)",
        "manual": "DDI0487",
        "section": "D13.2.38",
        "context": "exception",
        "architecture": "aarch64",
        "summary": "Holds the faulting virtual address for instruction aborts, data aborts, PC alignment faults, and watchpoint exceptions taken to EL1.",
        "related": ["ESR_EL1", "VBAR_EL1"],
    },
    "HCR_EL2": {
        "full_name": "Hypervisor Configuration Register (EL2)",
        "manual": "DDI0487",
        "section": "D13.2.48",
        "context": "virtualization",
        "architecture": "aarch64",
        "summary": "Controls hypervisor behavior: VM bit (enable Stage 2 translation), SWIO (set/way invalidation override), FMO/IMO/AMO (routing of FIQ/IRQ/SError to EL2), TGE (trap general exceptions), E2H (VHE enable), and numerous trap controls.",
        "related": ["VTTBR_EL2", "VTCR_EL2", "SCTLR_EL2"],
    },
    "SCR_EL3": {
        "full_name": "Secure Configuration Register (EL3)",
        "manual": "DDI0487",
        "section": "D13.2.113",
        "context": "security",
        "architecture": "aarch64",
        "summary": "Controls Secure/Non-secure state: NS bit (Non-secure state for lower ELs), EEL2 (Secure EL2 enable), FIQ/IRQ routing to EL3, RW bit (EL1 execution state AArch64/AArch32), and feature trap controls.",
        "related": ["HCR_EL2", "SCTLR_EL3"],
    },
    "SPSR_EL1": {
        "full_name": "Saved Program Status Register (EL1)",
        "manual": "DDI0487",
        "section": "D13.2.127",
        "context": "exception",
        "architecture": "aarch64",
        "summary": "Holds the saved PSTATE (condition flags N/Z/C/V, exception mask bits D/A/I/F, execution state, exception level, SP select) from the point where an exception was taken to EL1. Restored by ERET.",
        "related": ["ELR_EL1", "ESR_EL1", "SPSR_EL2", "SPSR_EL3"],
    },
    "ELR_EL1": {
        "full_name": "Exception Link Register (EL1)",
        "manual": "DDI0487",
        "section": "D13.2.36",
        "context": "exception",
        "architecture": "aarch64",
        "summary": "Holds the return address for exceptions taken to EL1. Set by hardware on exception entry. Used by ERET to return to the instruction that caused the exception (or the next instruction).",
        "related": ["SPSR_EL1", "ESR_EL1", "ELR_EL2"],
    },
    "CPACR_EL1": {
        "full_name": "Architectural Feature Access Control Register (EL1)",
        "manual": "DDI0487",
        "section": "D13.2.30",
        "context": "system",
        "architecture": "aarch64",
        "summary": "Controls access to FP/NEON/SVE from EL0 and EL1. FPEN field (bits [21:20]) enables/traps SIMD and FP. ZEN field (bits [17:16]) enables/traps SVE instructions.",
        "related": ["CPTR_EL2", "CPTR_EL3", "ZCR_EL1"],
    },
    "DAIF": {
        "full_name": "Interrupt Mask Bits",
        "manual": "DDI0487",
        "section": "D13.2.32",
        "context": "exception",
        "architecture": "aarch64",
        "summary": "Controls exception masking. D=Debug, A=SError (asynchronous abort), I=IRQ, F=FIQ. Setting a bit masks (disables) that exception type. Accessible via MSR DAIFSet/DAIFClr.",
        "related": ["SPSR_EL1", "ICC_PMR_EL1", "SCR_EL3"],
    },
    # AArch32 key registers
    "CPSR": {
        "full_name": "Current Program Status Register",
        "manual": "DDI0406",
        "section": "A2.5",
        "context": "system",
        "architecture": "aarch32",
        "summary": "Holds condition flags (N/Z/C/V), interrupt masks (I/F/A), processor mode (M[4:0]), execution state (T for Thumb, J for Jazelle), endianness (E), and GE bits for SIMD.",
        "related": ["SPSR", "APSR"],
    },
    "SCTLR": {
        "full_name": "System Control Register (AArch32)",
        "manual": "DDI0406",
        "section": "B4.1.130",
        "context": "system",
        "architecture": "aarch32",
        "summary": "AArch32 system control: MMU enable (M), alignment check (A), D-cache enable (C), I-cache enable (I), branch prediction enable (Z), TEX remap enable (TRE), and exception endianness (EE).",
        "related": ["TTBCR", "TTBR0", "TTBR1"],
    },
    "VTOR": {
        "full_name": "Vector Table Offset Register",
        "manual": "DDI0403",
        "section": "B3.2.5",
        "context": "exception",
        "architecture": "armv7-m",
        "summary": "Sets the base address of the Cortex-M exception vector table. Default is 0x00000000. Can be relocated to any 128-byte aligned address (bits [31:7]).",
        "related": ["NVIC_ISER", "NVIC_ICER", "SHPR"],
    },
    "MPU_TYPE": {
        "full_name": "MPU Type Register",
        "manual": "DDI0403",
        "section": "B3.5.1",
        "context": "memory",
        "architecture": "armv7-m",
        "summary": "Read-only register indicating MPU capabilities. DREGION field indicates the number of supported MPU regions (typically 8 or 16). IREGION is always 0 (unified MPU).",
        "related": ["MPU_CTRL", "MPU_RNR", "MPU_RBAR", "MPU_RASR"],
    },
}


@mcp.tool()
def find_register_in_manual(register_name: str, context: str | None = None) -> str:
    """Find which ARM manual section documents a specific system register.

    Returns the manual document ID, section number, a summary of what the
    register does, and related registers.

    Args:
        register_name: System register name (e.g. "SCTLR_EL1", "TCR_EL1",
                       "VBAR_EL1", "CPSR", "VTOR"). Case-insensitive.
        context: Optional filter to narrow results. One of: "system", "memory",
                 "exception", "virtualization", "security".
    """
    name_upper = register_name.upper().strip()

    # Try exact match first
    entry = REGISTER_MANUAL_MAP.get(name_upper)

    # Try case-insensitive match
    if entry is None:
        for k, v in REGISTER_MANUAL_MAP.items():
            if k.upper() == name_upper:
                entry = v
                name_upper = k
                break

    if entry is None:
        # Try partial match
        matches = []
        for k, v in REGISTER_MANUAL_MAP.items():
            if name_upper in k.upper() or k.upper() in name_upper:
                matches.append((k, v))
        if matches:
            if context:
                filtered = [(k, v) for k, v in matches if v["context"] == context]
                if filtered:
                    matches = filtered
            if len(matches) == 1:
                name_upper, entry = matches[0]
            else:
                lines = [f"Multiple registers match '{register_name}':"]
                for k, v in matches:
                    lines.append(f"  - {k}: {v['full_name']} ({v['manual']} {v['section']})")
                lines.append("\nSpecify the full register name or add a context filter.")
                return "\n".join(lines)

    if entry is None:
        available = sorted(REGISTER_MANUAL_MAP.keys())
        return (
            f"Register '{register_name}' not found in manual mapping.\n\n"
            f"Available registers: {', '.join(available[:20])}"
            + (f"... and {len(available) - 20} more." if len(available) > 20 else ".")
        )

    if context and entry["context"] != context:
        return (
            f"Register '{name_upper}' found but its context is '{entry['context']}', "
            f"not '{context}'. Showing result anyway.\n\n"
            + _format_register_manual_entry(name_upper, entry)
        )

    return _format_register_manual_entry(name_upper, entry)


def _format_register_manual_entry(name: str, entry: dict) -> str:
    lines = [f"# {name} — {entry['full_name']}"]
    lines.append(f"Architecture: {entry['architecture']}  |  Context: {entry['context']}")
    lines.append(f"Manual: {entry['manual']}  |  Section: {entry['section']}")
    lines.append(f"\n{entry['summary']}")
    if entry["related"]:
        lines.append(f"\n**Related registers:** {', '.join(entry['related'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: get_errata — per-core errata database
# ---------------------------------------------------------------------------

CORE_ERRATA_DB: dict[str, dict] = {
    "cortex-a53": {
        "core": "Cortex-A53",
        "revision": "r0p4",
        "doc_id": "EPM048406",
        "errata": [
            {
                "id": "843419",
                "category": "functional",
                "severity": "high",
                "title": "ADRP instruction followed by LDR at specific page offsets may generate incorrect address",
                "description": "An ADRP instruction followed by an LDR/STR instruction within a specific page offset range (0xFF8-0xFFF) may generate an incorrect address. This occurs due to an erratum in the address generation pipeline.",
                "affected_conditions": "ADRP+LDR/STR sequence where the LDR/STR is within 4KB page offset 0xFF8-0xFFF of the ADRP target.",
                "workaround": "Linker workaround: use --fix-cortex-a53-843419 flag in GNU ld/lld. The linker inserts a veneer (thunk) to avoid the problematic page offset. GCC and Clang enable this by default for aarch64-linux targets.",
                "impact": "Silent data corruption or crash if the wrong address is computed. Critical for correctness.",
            },
            {
                "id": "835769",
                "category": "functional",
                "severity": "high",
                "title": "Multiply-accumulate instruction may generate incorrect result",
                "description": "A multiply-accumulate instruction (MADD, MSUB, SMADDL, SMSUBL, UMADDL, UMSUBL) may produce incorrect results when the preceding instruction is a specific memory access instruction.",
                "affected_conditions": "Multiply-accumulate instruction immediately after certain load/store instructions in the pipeline.",
                "workaround": "Compiler workaround: use -mfix-cortex-a53-835769 (GCC/Clang). Inserts a NOP between the affected instruction sequences.",
                "impact": "Incorrect computation results. Critical for numerical workloads.",
            },
            {
                "id": "826319",
                "category": "functional",
                "severity": "medium",
                "title": "DMB instruction may not work correctly before certain store instructions",
                "description": "A DMB instruction may not correctly order memory accesses when followed by certain store instructions in specific pipeline conditions.",
                "affected_conditions": "DMB followed by store with specific timing in the pipeline.",
                "workaround": "Use DSB instead of DMB in critical sections. Alternatively, ensure a dependent instruction between DMB and the store.",
                "impact": "Memory ordering violation in multi-core scenarios. Can cause data races.",
            },
        ],
    },
    "cortex-a55": {
        "core": "Cortex-A55",
        "revision": "r1p0",
        "doc_id": "EPM130014",
        "errata": [
            {
                "id": "1530923",
                "category": "functional",
                "severity": "medium",
                "title": "Prefetch may cross a page boundary and cause a translation fault",
                "description": "Hardware prefetch may cross a page boundary and trigger a translation fault for the prefetched address, even though the actual instruction access does not cross the boundary.",
                "affected_conditions": "Code executing near the end of a page with the next page unmapped or with different permissions.",
                "workaround": "Ensure that code pages are mapped with adjacent guard pages having valid translations, or disable hardware prefetch via CPUECTLR.SMPEN clearing (not recommended for SMP systems).",
                "impact": "Unexpected translation faults. May cause spurious kernel panics if fault handler doesn't account for it.",
            },
            {
                "id": "1024718",
                "category": "performance",
                "severity": "low",
                "title": "Indirect branch predictor may not train correctly after context switch",
                "description": "The indirect branch predictor may retain stale predictions after a context switch, leading to branch mispredictions until the predictor warms up.",
                "affected_conditions": "After context switch or ASID change with indirect branches.",
                "workaround": "No workaround needed for correctness. Performance impact is temporary (predictor self-corrects). Consider pinning latency-sensitive threads to avoid frequent context switches.",
                "impact": "Temporary performance degradation after context switches. Typically a few hundred cycles of mispredictions.",
            },
        ],
    },
    "cortex-a72": {
        "core": "Cortex-A72",
        "revision": "r0p3",
        "doc_id": "EPM058020",
        "errata": [
            {
                "id": "859971",
                "category": "functional",
                "severity": "high",
                "title": "LDNP/STNP instructions may not enforce ordering in certain conditions",
                "description": "Non-temporal load/store pair instructions (LDNP/STNP) may not correctly enforce memory ordering with respect to other memory accesses in specific pipeline conditions.",
                "affected_conditions": "LDNP/STNP instructions followed by regular loads/stores to the same cache line.",
                "workaround": "Use regular LDP/STP instructions instead of LDNP/STNP, or insert a DMB barrier after LDNP/STNP when ordering is required.",
                "impact": "Memory ordering violations. Can cause data corruption in concurrent algorithms.",
            },
            {
                "id": "853709",
                "category": "functional",
                "severity": "medium",
                "title": "Store-Exclusive may return incorrect status in rare conditions",
                "description": "STXR/STLXR instructions may falsely report failure (return 1) even though the store was actually performed, in rare pipeline timing conditions.",
                "affected_conditions": "STXR/STLXR with specific memory access patterns from other cores hitting the same exclusive monitor region.",
                "workaround": "Standard LDXR/STXR retry loops already handle this correctly (retry on failure). Ensure exclusive access loops have a retry mechanism.",
                "impact": "Slightly increased CAS/atomic failure rate. Correctness is maintained if standard retry loops are used.",
            },
        ],
    },
    "cortex-a76": {
        "core": "Cortex-A76",
        "revision": "r4p0",
        "doc_id": "EPM134028",
        "errata": [
            {
                "id": "1490853",
                "category": "security",
                "severity": "high",
                "title": "Speculative execution vulnerability (Spectre variant 4 - SSB)",
                "description": "Speculative Store Bypass (SSB): a speculative load may bypass a preceding store to the same address and read stale data from cache, potentially leaking information across security boundaries.",
                "affected_conditions": "Load that depends on a recent store to the same address, where the store address is resolved late.",
                "workaround": "Enable SSBS (Speculative Store Bypass Safe) via PSTATE.SSBS or MSR SSBS. Linux kernel enables mitigations via prctl(PR_SET_SPECULATION_CTRL). Firmware can set SMCCC_ARCH_WORKAROUND_2.",
                "impact": "Potential information disclosure across privilege boundaries. Mitigated in all major OS kernels.",
            },
            {
                "id": "1165522",
                "category": "functional",
                "severity": "medium",
                "title": "TLB invalidation may not apply to all entries in certain conditions",
                "description": "A TLBI instruction followed by a DSB may not invalidate all matching TLB entries if a concurrent translation table walk is in progress.",
                "affected_conditions": "TLBI instruction concurrent with page table walks from other cores.",
                "workaround": "Use a two-step invalidation: 1) TLBI, 2) DSB ISH, 3) TLBI again, 4) DSB ISH. Linux kernel implements this workaround for affected cores.",
                "impact": "Stale TLB entries may persist, causing incorrect address translation. Critical for correctness of page unmapping.",
            },
        ],
    },
    "neoverse-n1": {
        "core": "Neoverse N1",
        "revision": "r4p1",
        "doc_id": "EPM146010",
        "errata": [
            {
                "id": "1542419",
                "category": "functional",
                "severity": "medium",
                "title": "System register read may return incorrect value after SMC/HVC",
                "description": "A system register read (MRS) immediately after returning from an SMC or HVC exception may return the value from before the exception, if the register was modified by the higher exception level.",
                "affected_conditions": "MRS instruction in the first few instructions after ERET from EL3/EL2.",
                "workaround": "Insert an ISB after the ERET landing point before reading system registers that may have been modified by the higher EL.",
                "impact": "Incorrect system register values visible briefly after exception return.",
            },
            {
                "id": "1315703",
                "category": "performance",
                "severity": "low",
                "title": "SVE gather loads may have higher latency than expected",
                "description": "SVE gather load instructions (LD1 with scalar+vector addressing) may have higher latency than the documented pipeline latency due to a micro-architectural bottleneck in the address generation unit.",
                "affected_conditions": "SVE LD1 gather loads with scalar+vector addressing mode, especially with 32-bit index elements.",
                "workaround": "For performance-critical gather operations, consider using scalar loads in a predicated loop if the access pattern is known. Or use contiguous loads with data reorganization.",
                "impact": "Higher-than-expected latency for SVE gather loads. Throughput unaffected.",
            },
        ],
    },
    "cortex-m33": {
        "core": "Cortex-M33",
        "revision": "r0p4",
        "doc_id": "EPM146011",
        "errata": [
            {
                "id": "837070",
                "category": "functional",
                "severity": "medium",
                "title": "Lazy FP state preservation may not save all FP registers on Secure-to-NS transition",
                "description": "When lazy FP stacking is enabled and a Non-secure exception interrupts Secure code using FP, the automatic FP state preservation may not save all FP registers correctly.",
                "affected_conditions": "Secure code using FP with lazy stacking, interrupted by Non-secure exception.",
                "workaround": "Disable lazy FP stacking (set FPCCR.LSPEN=0) in Secure firmware, or manually save FP context before enabling Non-secure exception handling.",
                "impact": "FP register corruption when Secure code is interrupted. Data integrity issue.",
            },
        ],
    },
}


@mcp.tool()
def get_errata(core_name: str, category: str | None = None) -> str:
    """Get known errata for an ARM core.

    Returns errata entries with severity, description, workarounds, and impact.

    Args:
        core_name: ARM core name (e.g. "cortex-a53", "cortex-a72", "cortex-a76",
                   "cortex-a55", "neoverse-n1", "cortex-m33"). Case-insensitive.
        category: Optional filter. One of: "functional", "performance", "security".
    """
    key = core_name.lower().strip().replace(" ", "-")
    core_data = CORE_ERRATA_DB.get(key)

    if core_data is None:
        available = sorted(CORE_ERRATA_DB.keys())
        return (
            f"Core '{core_name}' not found in errata database.\n\n"
            f"Available cores: {', '.join(available)}"
        )

    valid_categories = {"functional", "performance", "security"}
    if category and category not in valid_categories:
        return f"Error: category must be one of {', '.join(sorted(valid_categories))}."

    errata = core_data["errata"]
    if category:
        errata = [e for e in errata if e["category"] == category]

    if not errata:
        return (
            f"No errata found for {core_data['core']}"
            + (f" in category '{category}'." if category else ".")
        )

    lines = [f"# Errata: {core_data['core']} ({core_data['revision']})"]
    lines.append(f"Document: {core_data['doc_id']}")
    if category:
        lines.append(f"Category filter: {category}")
    lines.append(f"Showing {len(errata)} errata\n")

    for e in errata:
        lines.append(f"## Erratum {e['id']}: {e['title']}")
        lines.append(f"Category: {e['category']}  |  Severity: {e['severity'].upper()}")
        lines.append(f"\n**Description:** {e['description']}")
        lines.append(f"\n**Affected conditions:** {e['affected_conditions']}")
        lines.append(f"\n**Workaround:** {e['workaround']}")
        lines.append(f"\n**Impact:** {e['impact']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5: compare_manual_sections — cross-version comparison
# ---------------------------------------------------------------------------

CROSS_VERSION_COMPARISONS: dict[str, dict] = {
    "exception_handling": {
        "topic": "Exception Handling",
        "versions": {
            "armv7-a": {
                "architecture": "ARMv7-A (AArch32)",
                "manual": "DDI0406",
                "model": (
                    "Seven processor modes (USR, FIQ, IRQ, SVC, ABT, UND, SYS) plus Monitor (TrustZone) "
                    "and Hyp (Virtualization Extensions). Exception vector table at address set by VBAR or "
                    "HVBAR. Each vector is 4 bytes (one instruction, typically B to handler). "
                    "Banked registers per mode (R13_svc, R14_svc, SPSR_svc, etc.). "
                    "Exception entry saves CPSR to SPSR and LR, switches mode. "
                    "Return via MOVS PC, LR or SUBS PC, LR, #offset."
                ),
                "key_registers": ["CPSR", "SPSR", "VBAR", "HVBAR", "DFSR", "IFSR", "DFAR", "IFAR"],
            },
            "armv8-a_aarch64": {
                "architecture": "ARMv8-A (AArch64)",
                "manual": "DDI0487",
                "model": (
                    "Four exception levels (EL0-EL3) replace processor modes. Exception vector table at "
                    "VBAR_ELx with 16 entries: 4 types (Synchronous, IRQ, FIQ, SError) x 4 sources "
                    "(Current EL SP0, Current EL SPx, Lower EL AArch64, Lower EL AArch32). "
                    "Each entry is 128 bytes (32 instructions). "
                    "Exception entry saves PSTATE to SPSR_ELx, return PC to ELR_ELx, syndrome to ESR_ELx. "
                    "Return via ERET. Exception can only go to same or higher EL."
                ),
                "key_registers": ["VBAR_EL1", "ESR_EL1", "FAR_EL1", "ELR_EL1", "SPSR_EL1"],
            },
            "armv7-m": {
                "architecture": "ARMv7-M (Cortex-M)",
                "manual": "DDI0403",
                "model": (
                    "NVIC-based exception model. Vector table contains addresses (not instructions). "
                    "Hardware automatically stacks R0-R3, R12, LR, PC, xPSR on exception entry. "
                    "Two modes: Thread (normal) and Handler (exception). Two stack pointers: MSP (Main) "
                    "and PSP (Process). Exception return via special EXC_RETURN value in LR (0xFFFFFFF1/F9/FD). "
                    "Tail-chaining and late-arrival optimizations for fast back-to-back interrupts. "
                    "Configurable priority levels with priority grouping (PRIGROUP in AIRCR)."
                ),
                "key_registers": ["VTOR", "PRIMASK", "BASEPRI", "FAULTMASK", "CONTROL", "xPSR"],
            },
        },
    },
    "memory_management": {
        "topic": "Memory Management (MMU/MPU)",
        "versions": {
            "armv7-a": {
                "architecture": "ARMv7-A (AArch32)",
                "manual": "DDI0406",
                "model": (
                    "Two translation table formats:\n"
                    "1. Short-descriptor (default): 32-bit entries, 2-level tables, up to 32-bit PA (4GB).\n"
                    "   - First level: 4096 entries x 4B = 16KB table. Maps 1MB sections or points to L2.\n"
                    "   - Second level: 256 entries x 4B = 1KB table. Maps 4KB small pages or 64KB large pages.\n"
                    "2. Long-descriptor (LPAE, ARMv7+): 64-bit entries, 3-level tables, up to 40-bit PA (1TB).\n"
                    "   - Supports 1GB, 2MB, 4KB page sizes.\n"
                    "   - Required for Large Physical Address Extension.\n\n"
                    "TTBR0 for lower VA range, TTBR1 for upper range. TTBCR.N controls split."
                ),
                "key_registers": ["TTBR0", "TTBR1", "TTBCR", "SCTLR", "DACR", "DFSR", "IFSR"],
            },
            "armv8-a_aarch64": {
                "architecture": "ARMv8-A (AArch64)",
                "manual": "DDI0487",
                "model": (
                    "Single translation table format with 64-bit descriptors and up to 4 levels:\n"
                    "- 4KB granule: L0->L1->L2->L3, supports 48-bit VA (256TB) or 52-bit with LVA.\n"
                    "- 16KB granule: L0->L1->L2->L3, supports 47-bit VA.\n"
                    "- 64KB granule: L1->L2->L3, supports 48-bit VA or 52-bit with LVA.\n\n"
                    "Block descriptors for huge pages (L1: 1GB/2MB, L2: 512MB/32MB/2MB depending on granule).\n"
                    "Stage 1 + Stage 2 translation for virtualization.\n"
                    "TTBR0_EL1 (user, lower VA), TTBR1_EL1 (kernel, upper VA).\n"
                    "TCR_EL1 controls granule, VA size, cacheability, shareability.\n"
                    "No Domain Access Control (DACR is AArch32 only)."
                ),
                "key_registers": ["TTBR0_EL1", "TTBR1_EL1", "TCR_EL1", "MAIR_EL1", "SCTLR_EL1"],
            },
            "armv7-m": {
                "architecture": "ARMv7-M (Cortex-M)",
                "manual": "DDI0403",
                "model": (
                    "Memory Protection Unit (MPU) — no virtual memory, no address translation:\n"
                    "- 8 or 16 configurable regions (implementation-defined).\n"
                    "- Each region: base address, size (32B-4GB, power of 2), permissions (RO/RW/No access), "
                    "  memory type (Normal/Device/Strongly-ordered), and 8 sub-regions.\n"
                    "- Regions can overlap; higher-numbered region takes priority.\n"
                    "- No TLB, no page tables — deterministic, single-cycle protection check.\n"
                    "- Default memory map applies when MPU is disabled or for unprivileged background region."
                ),
                "key_registers": ["MPU_TYPE", "MPU_CTRL", "MPU_RNR", "MPU_RBAR", "MPU_RASR"],
            },
        },
    },
    "simd_extensions": {
        "topic": "SIMD/Vector Extensions",
        "versions": {
            "armv7-a": {
                "architecture": "ARMv7-A (AArch32 NEON)",
                "manual": "DDI0406",
                "model": (
                    "NEON (Advanced SIMD) with 128-bit vectors:\n"
                    "- 16x 128-bit Q registers (Q0-Q15), aliased as 32x 64-bit D registers (D0-D31).\n"
                    "- VFPv3/VFPv4 for scalar floating-point, sharing the D register file.\n"
                    "- Data types: 8/16/32/64-bit integer, 16/32-bit float (VFPv4 adds FP16 conversion).\n"
                    "- No predication (all-or-nothing vector operations).\n"
                    "- Limited to 128-bit vector width.\n"
                    "- Conditional execution via IT blocks only (no per-element predication)."
                ),
                "key_registers": ["FPSCR", "FPEXC", "MVFR0", "MVFR1"],
            },
            "armv8-a_aarch64": {
                "architecture": "ARMv8-A (AArch64 NEON/ASIMD)",
                "manual": "DDI0487",
                "model": (
                    "Advanced SIMD (NEON) with 128-bit vectors, significantly enhanced:\n"
                    "- 32x 128-bit V registers (V0-V31), no D-register aliasing issues.\n"
                    "- IEEE 754 compliant FP (unlike ARMv7 NEON which had flush-to-zero only).\n"
                    "- FP16 data processing (FEAT_FP16, ARMv8.2-A).\n"
                    "- BFloat16 support (FEAT_BF16, ARMv8.6-A).\n"
                    "- Dot product instructions (FEAT_DotProd, ARMv8.2-A): SDOT/UDOT.\n"
                    "- FCMA: complex number multiply-accumulate (ARMv8.3-A).\n"
                    "- Still limited to 128-bit width. No predication."
                ),
                "key_registers": ["FPCR", "FPSR", "ID_AA64ISAR0_EL1", "ID_AA64PFR0_EL1"],
            },
            "armv9-a_sve2": {
                "architecture": "ARMv9-A (SVE2)",
                "manual": "DDI0487 / DEN0065",
                "model": (
                    "Scalable Vector Extension 2 (SVE2) — mandatory in ARMv9:\n"
                    "- 32x Z registers: 128-2048 bits (implementation chooses width, 128-bit increments).\n"
                    "- 16x predicate registers (P0-P15): per-element predication.\n"
                    "- Vector Length Agnostic (VLA): single binary works across all SVE widths.\n"
                    "- SVE2 adds: fixed-point, complex number, crypto, histogram operations.\n"
                    "- Superset of NEON functionality (SVE2 can replace all NEON operations).\n"
                    "- Gather/scatter loads, first-fault loads, speculative access.\n"
                    "- WHILE* loop control for VLA programming.\n"
                    "- SME (Scalable Matrix Extension, ARMv9.2-A): outer-product tiles for matrix ops."
                ),
                "key_registers": ["ZCR_EL1", "ID_AA64ZFR0_EL1", "SMCR_EL1", "ID_AA64SMFR0_EL1"],
            },
        },
    },
    "security_model": {
        "topic": "Security Architecture",
        "versions": {
            "armv7-a": {
                "architecture": "ARMv7-A (TrustZone)",
                "manual": "DDI0406",
                "model": (
                    "TrustZone with two worlds (Secure / Non-secure):\n"
                    "- Monitor mode (via SMC instruction) mediates world switches.\n"
                    "- SCR.NS bit in CP15 controls the current security state.\n"
                    "- TZASC and TZPC partition memory and peripherals.\n"
                    "- Secure and Non-secure translation tables are separate.\n"
                    "- No Realm concept. Two-world model only."
                ),
                "key_registers": ["SCR", "MVBAR", "NSACR"],
            },
            "armv8-a_aarch64": {
                "architecture": "ARMv8-A (TrustZone + optional RME)",
                "manual": "DDI0487 / DEN0115",
                "model": (
                    "TrustZone with EL3 Secure Monitor:\n"
                    "- EL3 controls Secure/Non-secure transitions via SCR_EL3.NS.\n"
                    "- Secure EL2 support (FEAT_SEL2, ARMv8.4-A): hypervisor in Secure world.\n"
                    "- Stage 2 translation for Secure world VMs.\n\n"
                    "RME (Realm Management Extension, ARMv9.2-A):\n"
                    "- Four Physical Address Spaces: Secure, Non-secure, Realm, Root.\n"
                    "- Granule Protection Tables (GPT) at EL3 control PAS assignment.\n"
                    "- Realm Management Monitor (RMM) runs at Realm EL2.\n"
                    "- Arm CCA (Confidential Compute Architecture): hardware-enforced VM isolation."
                ),
                "key_registers": ["SCR_EL3", "GPCCR_EL3", "GPTBR_EL3", "TTBR0_EL3"],
            },
            "armv8-m": {
                "architecture": "ARMv8-M (TrustZone for Cortex-M)",
                "manual": "DDI0553",
                "model": (
                    "TrustZone for microcontrollers:\n"
                    "- SAU (Security Attribution Unit): configurable memory security regions.\n"
                    "- IDAU (Implementation Defined Attribution Unit): fixed security map.\n"
                    "- Three region types: Secure, Non-secure, Non-secure Callable (NSC).\n"
                    "- SG (Secure Gateway) instruction at NSC entry points.\n"
                    "- Hardware stacking on Secure/NS transitions (preserves Secure state).\n"
                    "- BXNS/BLXNS instructions for Non-secure function calls.\n"
                    "- No ELs, no hypervisor — simpler two-state model for embedded."
                ),
                "key_registers": ["SAU_CTRL", "SAU_RNR", "SAU_RBAR", "SAU_RLAR", "NSACR"],
            },
        },
    },
    "interrupt_handling": {
        "topic": "Interrupt Controller Architecture",
        "versions": {
            "armv7-a": {
                "architecture": "ARMv7-A (GICv2)",
                "manual": "IHI0048 (GICv2)",
                "model": (
                    "GICv2 — memory-mapped interrupt controller:\n"
                    "- Distributor (GICD): manages SPIs, routing, priority.\n"
                    "- CPU Interface (GICC): per-core, memory-mapped at fixed offsets.\n"
                    "- 8-bit priority, configurable preemption via Binary Point Register.\n"
                    "- Max 1020 interrupt IDs (SGI 0-15, PPI 16-31, SPI 32-1019).\n"
                    "- No LPI support, no affinity routing.\n"
                    "- Security Extensions: Group 0 (Secure/FIQ) and Group 1 (Non-secure/IRQ)."
                ),
                "key_registers": ["GICD_CTLR", "GICC_IAR", "GICC_EOIR", "GICC_PMR"],
            },
            "armv8-a_aarch64": {
                "architecture": "ARMv8-A (GICv3/v4)",
                "manual": "IHI0069 (GICv3/v4)",
                "model": (
                    "GICv3/v4 — system register-based CPU interface:\n"
                    "- Distributor (GICD), Redistributor (GICR), CPU Interface (ICC via system regs).\n"
                    "- Affinity routing: interrupts routed by Aff3.Aff2.Aff1.Aff0.\n"
                    "- LPI support: message-based interrupts for PCIe MSI/MSI-X (IDs 8192+).\n"
                    "- ITS (Interrupt Translation Service): DeviceID+EventID -> LPI mapping.\n"
                    "- Three security groups: Group 0 (Secure), Group 1 Secure, Group 1 Non-secure.\n"
                    "- GICv4: direct virtual interrupt injection for VMs (vLPIs, no hypervisor trap)."
                ),
                "key_registers": ["ICC_IAR1_EL1", "ICC_EOIR1_EL1", "ICC_PMR_EL1", "ICC_SRE_EL1"],
            },
            "armv7-m": {
                "architecture": "ARMv7-M / ARMv8-M (NVIC)",
                "manual": "DDI0403 / DDI0553",
                "model": (
                    "NVIC — tightly coupled to core:\n"
                    "- Up to 496 external interrupts + 16 system exceptions.\n"
                    "- Hardware auto-stacking (R0-R3, R12, LR, PC, xPSR) on entry.\n"
                    "- Tail-chaining: <6 cycles between back-to-back ISRs.\n"
                    "- Late arrival: higher-priority interrupt steals pending exception's slot.\n"
                    "- Lazy FP stacking: defers FP context save (Cortex-M4F/M33).\n"
                    "- Priority grouping: configurable preemption/sub-priority split.\n"
                    "- Vector table contains handler addresses (not instructions).\n"
                    "- Deterministic interrupt latency (12-cycle minimum for Cortex-M3/M4)."
                ),
                "key_registers": ["NVIC_ISER", "NVIC_ICER", "NVIC_IPR", "VTOR", "AIRCR"],
            },
        },
    },
}


@mcp.tool()
def compare_manual_sections(topic: str) -> str:
    """Compare how a topic is covered across different ARM architecture versions.

    Shows side-by-side how exception handling, memory management, SIMD,
    security, or interrupts differ between ARMv7-A, ARMv8-A (AArch64),
    ARMv7-M, and ARMv9-A.

    Args:
        topic: The topic to compare. One of: "exception_handling",
               "memory_management", "simd_extensions", "security_model",
               "interrupt_handling". Case-insensitive.
    """
    key = topic.lower().strip().replace(" ", "_").replace("-", "_")

    # Try alias matching
    aliases = {
        "exceptions": "exception_handling",
        "exception": "exception_handling",
        "memory": "memory_management",
        "mmu": "memory_management",
        "mpu": "memory_management",
        "paging": "memory_management",
        "simd": "simd_extensions",
        "neon": "simd_extensions",
        "sve": "simd_extensions",
        "vector": "simd_extensions",
        "security": "security_model",
        "trustzone": "security_model",
        "rme": "security_model",
        "interrupts": "interrupt_handling",
        "gic": "interrupt_handling",
        "nvic": "interrupt_handling",
    }
    key = aliases.get(key, key)

    comparison = CROSS_VERSION_COMPARISONS.get(key)
    if comparison is None:
        available = sorted(CROSS_VERSION_COMPARISONS.keys())
        return (
            f"No comparison data for topic '{topic}'.\n\n"
            f"Available topics: {', '.join(available)}\n\n"
            "Also accepts aliases: exceptions, memory, mmu, simd, neon, "
            "sve, security, trustzone, interrupts, gic, nvic."
        )

    lines = [f"# Cross-Version Comparison: {comparison['topic']}"]
    lines.append(f"Comparing {len(comparison['versions'])} architecture versions\n")

    for version_key, version_data in comparison["versions"].items():
        lines.append(f"## {version_data['architecture']}")
        lines.append(f"Manual: {version_data['manual']}")
        lines.append(f"\n{version_data['model']}")
        lines.append(f"\n**Key registers:** {', '.join(version_data['key_registers'])}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 6: list_arm_documents — browse the documentation catalog
# ---------------------------------------------------------------------------

@mcp.tool()
def list_arm_documents(doc_scope: str | None = None, architecture: str | None = None) -> str:
    """List all ARM documentation entries in the reference database.

    Browse available ARM manuals, technical reference manuals, programming
    guides, optimization guides, and security specifications. Optionally
    filter by scope or architecture.

    Args:
        doc_scope: Optional filter. One of: "architecture", "processor_trm",
                   "programming_guide", "security", "firmware", "optimization".
                   If omitted, lists all documents.
        architecture: Optional architecture filter (e.g. "aarch64", "aarch32",
                      "armv8-m", "armv7-m"). If omitted, lists all architectures.
    """
    scope_filter = doc_scope.lower().strip() if doc_scope else None
    arch_filter = architecture.lower().strip() if architecture else None

    matches: list[dict] = []
    for doc in ARM_DOCS_DB:
        if scope_filter and doc.get("scope", "") != scope_filter:
            continue
        if arch_filter and doc.get("architecture", "") != arch_filter:
            continue
        matches.append(doc)

    if not matches:
        filters = []
        if scope_filter:
            filters.append(f"scope='{scope_filter}'")
        if arch_filter:
            filters.append(f"architecture='{arch_filter}'")
        all_scopes = sorted(set(d.get("scope", "") for d in ARM_DOCS_DB))
        all_archs = sorted(set(d.get("architecture", "") for d in ARM_DOCS_DB))
        return (
            f"No documents match filters: {', '.join(filters)}.\n\n"
            f"Available scopes: {', '.join(all_scopes)}\n"
            f"Available architectures: {', '.join(all_archs)}"
        )

    lines = ["# ARM Documentation Catalog"]
    if scope_filter or arch_filter:
        parts = []
        if scope_filter:
            parts.append(f"Scope: {scope_filter}")
        if arch_filter:
            parts.append(f"Architecture: {arch_filter}")
        lines.append(f"Filters: {', '.join(parts)}")
    lines.append(f"Found {len(matches)} document(s)\n")

    for doc in matches:
        doc_id = doc.get("doc_id", "N/A")
        title = doc.get("title", "Untitled")
        scope = doc.get("scope", "N/A")
        arch = doc.get("architecture", "N/A")
        sections = doc.get("sections", [])
        lines.append(f"## [{doc_id}] {title}")
        lines.append(f"Architecture: {arch}  |  Scope: {scope}  |  Sections: {len(sections)}")
        for sec in sections:
            section_id = sec.get("section", "")
            topic = sec.get("topic", "")
            lines.append(f"  - {section_id}: {topic}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 7: explain_instruction_encoding — instruction set encoding reference
# ---------------------------------------------------------------------------

INSTRUCTION_ENCODING_DB: dict[str, dict] = {
    "a64": {
        "name": "A64 (AArch64)",
        "architecture": "AArch64 (ARMv8-A / ARMv9-A)",
        "word_size": "32-bit fixed-width",
        "description": (
            "A64 is the AArch64 instruction set. All instructions are exactly "
            "32 bits (4 bytes) wide, simplifying fetch and decode. Instructions "
            "are always little-endian in memory regardless of data endianness."
        ),
        "encoding_groups": [
            {"bits": "[28:25]", "group": "Data Processing (Immediate)", "pattern": "100x",
             "examples": "ADD, SUB, MOV, MOVK, MOVZ, ADRP, AND, ORR, EOR"},
            {"bits": "[28:25]", "group": "Branches, Exception, System", "pattern": "101x",
             "examples": "B, BL, BR, BLR, RET, SVC, HVC, SMC, MSR, MRS, NOP, ISB, DSB"},
            {"bits": "[28:25]", "group": "Loads and Stores", "pattern": "x1x0",
             "examples": "LDR, STR, LDP, STP, LDXR, STXR, LDAR, STLR, PRFM"},
            {"bits": "[28:25]", "group": "Data Processing (Register)", "pattern": "x101",
             "examples": "ADD, SUB, AND, ORR, shift, extend, MADD, MSUB, SDIV, UDIV"},
            {"bits": "[28:25]", "group": "Data Processing (SIMD & FP)", "pattern": "x111",
             "examples": "FADD, FMUL, FMLA, FCVT, SCVTF, UCVTF, LD1, ST1"},
            {"bits": "[28:25]", "group": "SVE Instructions", "pattern": "0010",
             "examples": "LD1W, ST1W, ADD, MUL, FMLA, WHILELT, PTRUE, PTEST"},
        ],
        "key_fields": [
            {"field": "sf [31]", "meaning": "Size flag: 0=32-bit (W registers), 1=64-bit (X registers)"},
            {"field": "op [30]", "meaning": "Operation variant (instruction-specific)"},
            {"field": "Rd [4:0]", "meaning": "Destination register (0-30, 31=SP or ZR depending on context)"},
            {"field": "Rn [9:5]", "meaning": "First source register"},
            {"field": "Rm [20:16]", "meaning": "Second source register (register operand instructions)"},
            {"field": "imm12 [21:10]", "meaning": "12-bit immediate (add/sub immediate)"},
            {"field": "imm19 [23:5]", "meaning": "19-bit signed offset (conditional branch, LDR literal)"},
            {"field": "imm26 [25:0]", "meaning": "26-bit signed offset (B, BL unconditional branch)"},
        ],
        "notes": [
            "No condition codes on most instructions (unlike AArch32). Use conditional branches or CSEL/CSINC instead.",
            "PC is not a general-purpose register — cannot be directly used as Rd/Rn/Rm.",
            "Register 31 is context-dependent: SP for stack operations, ZR (zero register) for data processing.",
            "Unallocated encodings trigger Undefined Instruction exceptions.",
        ],
    },
    "t32": {
        "name": "T32 (Thumb-2)",
        "architecture": "AArch32 (ARMv6T2+, ARMv7, ARMv8-A AArch32)",
        "word_size": "16-bit and 32-bit mixed",
        "description": (
            "T32 (Thumb-2) is a variable-length instruction set mixing 16-bit "
            "and 32-bit instructions. 16-bit instructions provide high code density "
            "for common operations. 32-bit Thumb instructions (starting with 0b111x_1) "
            "extend the ISA to near-A32 capability while maintaining density."
        ),
        "encoding_groups": [
            {"bits": "[15:11]", "group": "16-bit Shift/Add/Sub/Move", "pattern": "000xx / 001xx",
             "examples": "LSL, LSR, ASR, ADD, SUB, MOV, CMP (3-bit register encodings)"},
            {"bits": "[15:11]", "group": "16-bit Data Processing", "pattern": "01000",
             "examples": "AND, EOR, LSL, LSR, ASR, ADC, SBC, ROR, TST, NEG, CMP, CMN, ORR, MUL, BIC, MVN"},
            {"bits": "[15:11]", "group": "16-bit Load/Store", "pattern": "0101x / 011xx / 100xx",
             "examples": "LDR, STR, LDRB, STRB, LDRH, STRH (5-bit immediate offset)"},
            {"bits": "[15:11]", "group": "16-bit Branch", "pattern": "1101x",
             "examples": "B<cond> (8-bit signed offset, conditional), SVC"},
            {"bits": "[15:11]", "group": "32-bit Prefix", "pattern": "111xx",
             "examples": "(First halfword of 32-bit instructions)"},
        ],
        "key_fields": [
            {"field": "hw1 [15:11]", "meaning": "First halfword bits determine 16-bit vs 32-bit and instruction group"},
            {"field": "Rd [2:0] / [11:8]", "meaning": "Destination register (3-bit for 16-bit, 4-bit for 32-bit)"},
            {"field": "Rn [5:3] / [19:16]", "meaning": "First source register"},
            {"field": "IT [15:0]", "meaning": "IT (If-Then) block: up to 4 conditional instructions without branches"},
        ],
        "notes": [
            "16-bit instructions can only access R0-R7 (low registers). Use 32-bit forms for R8-R15.",
            "IT blocks allow conditional execution of up to 4 instructions: IT{T|E}{T|E}{T|E} <cond>.",
            "32-bit instructions are identified by first halfword having bits [15:11] = 0b111xx (except 0b11100).",
            "Code must be 2-byte aligned (halfword aligned). 32-bit instructions can be unaligned to 4-byte boundary.",
            "BL is always 32-bit. BLX to ARM state requires 4-byte alignment at target.",
        ],
    },
    "a32": {
        "name": "A32 (ARM)",
        "architecture": "AArch32 (ARMv4 through ARMv8-A AArch32)",
        "word_size": "32-bit fixed-width",
        "description": (
            "A32 is the original ARM instruction set. All instructions are "
            "32 bits wide and must be 4-byte aligned. Nearly every instruction "
            "can be conditionally executed using a 4-bit condition code in bits [31:28]."
        ),
        "encoding_groups": [
            {"bits": "[27:25]", "group": "Data Processing (Immediate)", "pattern": "001",
             "examples": "MOV, ADD, SUB, AND, ORR, EOR, CMP, TST with rotated immediate"},
            {"bits": "[27:25]", "group": "Data Processing (Register)", "pattern": "000",
             "examples": "MOV, ADD, SUB with register operand and barrel shifter"},
            {"bits": "[27:25]", "group": "Load/Store Word/Byte", "pattern": "01x",
             "examples": "LDR, STR, LDRB, STRB with immediate or register offset"},
            {"bits": "[27:25]", "group": "Load/Store Multiple", "pattern": "100",
             "examples": "LDM, STM, PUSH, POP"},
            {"bits": "[27:25]", "group": "Branch", "pattern": "101",
             "examples": "B, BL (24-bit signed offset, +/- 32MB range)"},
            {"bits": "[27:25]", "group": "Coprocessor / SWI", "pattern": "11x",
             "examples": "CDP, MCR, MRC, LDC, STC, SVC (SWI)"},
        ],
        "key_fields": [
            {"field": "cond [31:28]", "meaning": "Condition code (0x0=EQ, 0xE=AL, 0xF=unconditional/special). Nearly all instructions are conditional."},
            {"field": "Rd [15:12]", "meaning": "Destination register (R0-R15). R15=PC allows branch-by-write."},
            {"field": "Rn [19:16]", "meaning": "First source register"},
            {"field": "Rm [3:0]", "meaning": "Second source register"},
            {"field": "shift [11:4]", "meaning": "Barrel shifter: type (LSL/LSR/ASR/ROR) + amount (immediate or Rs)"},
            {"field": "imm8 [7:0]", "meaning": "8-bit immediate value, right-rotated by 2*rotate_imm[11:8]"},
        ],
        "notes": [
            "Every instruction can be conditional — this is unique to A32 among mainstream ISAs.",
            "Barrel shifter on second operand provides shift+operate in one instruction (e.g., ADD R0, R1, R2, LSL #3).",
            "Writing to R15 (PC) causes a branch — used for function returns (MOV PC, LR) in non-Thumb code.",
            "Immediate values use 8-bit + 4-bit rotation encoding, giving non-contiguous representable values.",
            "S suffix (e.g., ADDS, MOVS) sets condition flags; without S, flags are not affected.",
        ],
    },
}


@mcp.tool()
def explain_instruction_encoding(encoding_format: str) -> str:
    """Explain an ARM instruction set encoding format in detail.

    Returns the word size, encoding group table, key fields, and important
    notes for A64 (AArch64), T32 (Thumb-2), or A32 (ARM) instruction sets.

    Args:
        encoding_format: The instruction set format. One of:
            "a64" / "aarch64" -- AArch64 A64 instruction set
            "t32" / "thumb" / "thumb2" -- Thumb-2 variable-length instruction set
            "a32" / "arm" / "aarch32" -- Classic 32-bit ARM instruction set
            "overview" / "list" -- Summary comparison of all three formats
    """
    key = encoding_format.lower().strip().replace("-", "").replace("_", "")

    # Aliases
    aliases = {
        "aarch64": "a64",
        "thumb": "t32",
        "thumb2": "t32",
        "arm": "a32",
        "aarch32": "a32",
    }
    key = aliases.get(key, key)

    # Overview mode
    if key in ("overview", "list", "all"):
        lines = ["# ARM Instruction Set Encoding Formats\n"]
        lines.append(f"{'Format':<10} {'Word Size':<30} {'Architecture'}")
        lines.append("-" * 75)
        for fmt_key in ("a64", "t32", "a32"):
            fmt = INSTRUCTION_ENCODING_DB[fmt_key]
            lines.append(f"{fmt['name']:<10} {fmt['word_size']:<30} {fmt['architecture']}")
        lines.append("")
        lines.append("Use `explain_instruction_encoding(format)` with one of: a64, t32, a32")
        return "\n".join(lines)

    fmt = INSTRUCTION_ENCODING_DB.get(key)
    if fmt is None:
        available = list(INSTRUCTION_ENCODING_DB.keys())
        return (
            f"Error: Unknown instruction encoding format '{encoding_format}'.\n"
            f"Available: {', '.join(available)}, overview\n"
            "Also accepts aliases: aarch64, thumb, thumb2, arm, aarch32."
        )

    lines = [f"# {fmt['name']} Instruction Encoding"]
    lines.append(f"Architecture: {fmt['architecture']}")
    lines.append(f"Word size: {fmt['word_size']}\n")
    lines.append(fmt["description"])
    lines.append("")

    # Encoding groups table
    lines.append("## Encoding Groups\n")
    lines.append(f"{'Bits':<14} {'Pattern':<10} {'Group':<35} Examples")
    lines.append("-" * 100)
    for grp in fmt["encoding_groups"]:
        lines.append(
            f"{grp['bits']:<14} {grp['pattern']:<10} {grp['group']:<35} {grp['examples']}"
        )
    lines.append("")

    # Key fields
    lines.append("## Key Instruction Fields\n")
    for field in fmt["key_fields"]:
        lines.append(f"  **{field['field']}**: {field['meaning']}")
    lines.append("")

    # Notes
    lines.append("## Important Notes\n")
    for i, note in enumerate(fmt["notes"], 1):
        lines.append(f"  {i}. {note}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run()


if __name__ == "__main__":
    main()
