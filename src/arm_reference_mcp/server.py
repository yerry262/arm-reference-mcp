"""ARM Register Reference MCP Server.

Provides twenty-three tools:
  - lookup_register:               Get detailed info on a specific ARM register.
  - list_registers:                Browse registers by architecture and category.
  - decode_instruction:            Decode a 32-bit hex value into AArch32 instruction fields.
  - explain_condition_code:        Explain an ARM condition code suffix in detail.
  - explain_calling_convention:    AAPCS calling convention reference.
  - search_registers:              Keyword search across all register data.
  - decode_register_value:         Decode a hex value against a register's bit fields.
  - explain_exception_levels:      ARM Exception Levels (EL0-EL3) and AArch32 processor modes.
  - explain_security_model:        ARM security model (TrustZone, RME, CCA).
  - lookup_core:                   ARM core/IP reference card (Cortex-A/R/M/X, Neoverse).
  - compare_cores:                 Side-by-side comparison of two ARM cores.
  - explain_extension:             ARM architecture extension reference (SVE, MTE, PAC, etc.).
  - compare_architecture_versions: ARM architecture version features and comparison.
  - explain_page_table_format:     AArch64 page table translation schemes.
  - explain_memory_attributes:     Memory attributes (cacheability, shareability, permissions).
  - show_assembly_pattern:         Annotated ARM assembly for common patterns.
  - explain_barrier:               Memory barriers and synchronization instructions.
  - explain_neon_intrinsic:        NEON/ASIMD intrinsic reference (instruction, types, latency).
  - explain_sme_tile:              SME tile operations, ZA storage, streaming SVE mode.
  - suggest_optimization:          ARM-specific optimization suggestions for code patterns.
  - lookup_system_register:        Full AArch64 system register reference.
  - explain_performance_counter:   ARM PMU performance counter event reference.
  - translate_intrinsic:           Translate between x86 SSE/AVX and ARM NEON/SVE intrinsics.
"""

from mcp.server.fastmcp import FastMCP

from arm_reference_mcp.data import REGISTERS

mcp = FastMCP(
    "ARM Register Reference",
    instructions="Quick-reference MCP for ARM register info across AArch32 and AArch64.",
)


def _find_register(name: str, architecture: str | None) -> list[dict]:
    """Case-insensitive register lookup supporting names and aliases."""
    name_upper = name.upper()
    results = []
    for reg in REGISTERS:
        names = [reg["name"].upper()] + [a.upper() for a in reg["aliases"]]
        if name_upper in names:
            if architecture is None or reg["architecture"] == architecture:
                results.append(reg)
    return results


def _format_register(reg: dict) -> str:
    """Format a single register entry as readable text."""
    lines = []
    aliases = f" (aliases: {', '.join(reg['aliases'])})" if reg["aliases"] else ""
    lines.append(f"## {reg['name']}{aliases}")
    lines.append(f"Architecture: {reg['architecture']}  |  {reg['bit_width']}-bit  |  Category: {reg['category']}")
    lines.append(f"\n{reg['description']}")

    if reg["fields"]:
        lines.append("\n### Bit Fields")
        for f in reg["fields"]:
            lines.append(f"  [{f['bits']:>5}]  {f['name']:<8}  {f['description']}")

    if reg["usage_notes"]:
        lines.append(f"\n**Usage:** {reg['usage_notes']}")

    return "\n".join(lines)


@mcp.tool()
def lookup_register(name: str, architecture: str | None = None) -> str:
    """Look up an ARM register by name or alias.

    Args:
        name: Register name (e.g. "X0", "CPSR", "SP", "LR", "W5").
              Case-insensitive. Supports aliases like FP, IP, LR, WZR.
        architecture: Optional filter — "aarch32" or "aarch64".
                      If omitted, returns matches from both architectures.
    """
    if architecture and architecture not in ("aarch32", "aarch64"):
        return "Error: architecture must be 'aarch32' or 'aarch64'."

    results = _find_register(name, architecture)

    if not results:
        return f"No register found matching '{name}'" + (
            f" in {architecture}." if architecture else "."
        ) + " Try list_registers to browse available registers."

    return "\n\n---\n\n".join(_format_register(r) for r in results)


@mcp.tool()
def list_registers(architecture: str, category: str | None = None) -> str:
    """List ARM registers for a given architecture.

    Args:
        architecture: "aarch32" or "aarch64".
        category: Optional filter — "general_purpose", "status", "system", or "floating_point".
                  If omitted, lists all registers for the architecture.
    """
    if architecture not in ("aarch32", "aarch64"):
        return "Error: architecture must be 'aarch32' or 'aarch64'."

    valid_categories = {"general_purpose", "status", "system", "floating_point"}
    if category and category not in valid_categories:
        return f"Error: category must be one of {', '.join(sorted(valid_categories))}."

    matches = [
        r for r in REGISTERS
        if r["architecture"] == architecture
        and (category is None or r["category"] == category)
    ]

    if not matches:
        return f"No registers found for {architecture}" + (
            f" in category '{category}'." if category else "."
        )

    header = f"# {architecture} Registers"
    if category:
        header += f" — {category}"
    header += f"\n\n{'Name':<12} {'Width':>5}  {'Aliases':<16} Description"
    header += "\n" + "-" * 72

    rows = []
    for r in matches:
        aliases = ", ".join(r["aliases"]) if r["aliases"] else "—"
        # Truncate description for the table view
        desc = r["description"]
        if len(desc) > 60:
            desc = desc[:57] + "..."
        rows.append(f"{r['name']:<12} {r['bit_width']:>3}-bit  {aliases:<16} {desc}")

    return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# Condition-code reference table (shared by both new tools)
# ---------------------------------------------------------------------------

_CONDITION_CODES: dict[int, dict] = {
    0x0: {
        "suffix": "EQ",
        "name": "Equal",
        "flags_tested": "Z",
        "flag_condition": "Z == 1",
        "opposite": "NE",
        "use_case": "Check if two values are equal (after CMP) or if a result is zero.",
    },
    0x1: {
        "suffix": "NE",
        "name": "Not Equal",
        "flags_tested": "Z",
        "flag_condition": "Z == 0",
        "opposite": "EQ",
        "use_case": "Check if two values differ or a result is non-zero.",
    },
    0x2: {
        "suffix": "CS/HS",
        "name": "Carry Set / Unsigned Higher or Same",
        "flags_tested": "C",
        "flag_condition": "C == 1",
        "opposite": "CC/LO",
        "use_case": "Unsigned >= comparison. Also used after additions to detect carry-out.",
    },
    0x3: {
        "suffix": "CC/LO",
        "name": "Carry Clear / Unsigned Lower",
        "flags_tested": "C",
        "flag_condition": "C == 0",
        "opposite": "CS/HS",
        "use_case": "Unsigned < comparison.",
    },
    0x4: {
        "suffix": "MI",
        "name": "Minus / Negative",
        "flags_tested": "N",
        "flag_condition": "N == 1",
        "opposite": "PL",
        "use_case": "Result is negative (bit 31 set).",
    },
    0x5: {
        "suffix": "PL",
        "name": "Plus / Positive or Zero",
        "flags_tested": "N",
        "flag_condition": "N == 0",
        "opposite": "MI",
        "use_case": "Result is positive or zero (bit 31 clear).",
    },
    0x6: {
        "suffix": "VS",
        "name": "Overflow Set",
        "flags_tested": "V",
        "flag_condition": "V == 1",
        "opposite": "VC",
        "use_case": "Signed arithmetic overflowed (result doesn't fit in 32 bits).",
    },
    0x7: {
        "suffix": "VC",
        "name": "Overflow Clear",
        "flags_tested": "V",
        "flag_condition": "V == 0",
        "opposite": "VS",
        "use_case": "Signed arithmetic did not overflow.",
    },
    0x8: {
        "suffix": "HI",
        "name": "Unsigned Higher",
        "flags_tested": "C, Z",
        "flag_condition": "C == 1 AND Z == 0",
        "opposite": "LS",
        "use_case": "Unsigned > comparison (strictly greater).",
    },
    0x9: {
        "suffix": "LS",
        "name": "Unsigned Lower or Same",
        "flags_tested": "C, Z",
        "flag_condition": "C == 0 OR Z == 1",
        "opposite": "HI",
        "use_case": "Unsigned <= comparison.",
    },
    0xA: {
        "suffix": "GE",
        "name": "Signed Greater Than or Equal",
        "flags_tested": "N, V",
        "flag_condition": "N == V",
        "opposite": "LT",
        "use_case": "Signed >= comparison.",
    },
    0xB: {
        "suffix": "LT",
        "name": "Signed Less Than",
        "flags_tested": "N, V",
        "flag_condition": "N != V",
        "opposite": "GE",
        "use_case": "Signed < comparison.",
    },
    0xC: {
        "suffix": "GT",
        "name": "Signed Greater Than",
        "flags_tested": "Z, N, V",
        "flag_condition": "Z == 0 AND N == V",
        "opposite": "LE",
        "use_case": "Signed > comparison (strictly greater).",
    },
    0xD: {
        "suffix": "LE",
        "name": "Signed Less Than or Equal",
        "flags_tested": "Z, N, V",
        "flag_condition": "Z == 1 OR N != V",
        "opposite": "GT",
        "use_case": "Signed <= comparison.",
    },
    0xE: {
        "suffix": "AL",
        "name": "Always (unconditional)",
        "flags_tested": "None",
        "flag_condition": "Always true",
        "opposite": "NV (never — deprecated)",
        "use_case": "Instruction always executes. The default when no suffix is written.",
    },
}

# Build a reverse lookup: suffix string -> entry (handles compound suffixes like "CS/HS")
_SUFFIX_TO_CC: dict[str, tuple[int, dict]] = {}
for _code, _entry in _CONDITION_CODES.items():
    for _part in _entry["suffix"].split("/"):
        _SUFFIX_TO_CC[_part.upper()] = (_code, _entry)
# Also store the full compound key
for _code, _entry in _CONDITION_CODES.items():
    _SUFFIX_TO_CC[_entry["suffix"].upper()] = (_code, _entry)

# Data-processing opcode names (bits [24:21])
_DP_OPCODES: dict[int, str] = {
    0x0: "AND",
    0x1: "EOR",
    0x2: "SUB",
    0x3: "RSB",
    0x4: "ADD",
    0x5: "ADC",
    0x6: "SBC",
    0x7: "RSC",
    0x8: "TST",
    0x9: "TEQ",
    0xA: "CMP",
    0xB: "CMN",
    0xC: "ORR",
    0xD: "MOV",
    0xE: "BIC",
    0xF: "MVN",
}

_REG_NAMES = {i: f"R{i}" for i in range(16)}
_REG_NAMES.update({13: "SP", 14: "LR", 15: "PC"})


def _bits(value: int, high: int, low: int) -> int:
    """Extract bits [high:low] (inclusive) from value."""
    mask = (1 << (high - low + 1)) - 1
    return (value >> low) & mask


def _format_field(bit_range: str, raw: int, meaning: str) -> str:
    """Format a single decoded field line."""
    return f"  [{bit_range:>5}]  0x{raw:X} ({raw})  —  {meaning}"


def _decode_data_processing(value: int, cond_name: str) -> list[str]:
    """Decode a data-processing instruction."""
    lines: list[str] = []
    lines.append("Instruction type: Data Processing (bits [27:26] = 00)")
    imm = _bits(value, 25, 25)
    opcode = _bits(value, 24, 21)
    s_bit = _bits(value, 20, 20)
    rn = _bits(value, 19, 16)
    rd = _bits(value, 15, 12)
    operand2 = _bits(value, 11, 0)

    op_name = _DP_OPCODES.get(opcode, "UNKNOWN")
    s_flag = "S" if s_bit else ""
    mnemonic = f"{op_name}{s_flag}{cond_name if cond_name != 'AL' else ''}"

    lines.append(f"Mnemonic: {mnemonic}")
    lines.append("")
    lines.append("### Field Breakdown")
    lines.append(_format_field("25", imm, f"I (Immediate) = {imm} ({'Immediate operand2' if imm else 'Register operand2'})"))
    lines.append(_format_field("24:21", opcode, f"Opcode = {op_name}"))
    lines.append(_format_field("20", s_bit, f"S (Set flags) = {s_bit} ({'Yes — updates CPSR' if s_bit else 'No'})"))
    lines.append(_format_field("19:16", rn, f"Rn (first operand register) = {_REG_NAMES.get(rn, f'R{rn}')}"))
    lines.append(_format_field("15:12", rd, f"Rd (destination register) = {_REG_NAMES.get(rd, f'R{rd}')}"))

    if imm:
        rotate = _bits(operand2, 11, 8)
        imm_val = _bits(operand2, 7, 0)
        rotated = (imm_val >> (rotate * 2)) | (imm_val << (32 - rotate * 2)) & 0xFFFFFFFF
        lines.append(_format_field("11:0", operand2, f"Operand2 (immediate): rotate={rotate}, imm8=0x{imm_val:02X} -> value = {rotated}"))
    else:
        shift_amount = _bits(operand2, 11, 7)
        shift_type = _bits(operand2, 6, 5)
        rm = _bits(operand2, 3, 0)
        shift_names = {0: "LSL", 1: "LSR", 2: "ASR", 3: "ROR"}
        lines.append(_format_field("11:0", operand2,
            f"Operand2 (register): {_REG_NAMES.get(rm, f'R{rm}')} "
            f"{shift_names.get(shift_type, '??')} #{shift_amount}"))

    return lines


def _decode_branch(value: int, cond_name: str) -> list[str]:
    """Decode a branch instruction."""
    lines: list[str] = []
    lines.append("Instruction type: Branch (bits [27:25] = 101)")
    l_bit = _bits(value, 24, 24)
    offset_raw = _bits(value, 23, 0)

    # Sign-extend 24-bit offset to 32 bits, then shift left 2
    if offset_raw & 0x800000:
        offset_signed = offset_raw - 0x1000000
    else:
        offset_signed = offset_raw
    byte_offset = offset_signed << 2
    # PC-relative: effective = PC + 8 + byte_offset (PC is 2 instructions ahead)
    effective_note = f"PC + 8 + ({byte_offset})"

    branch_type = "BL (Branch with Link)" if l_bit else "B (Branch)"
    mnemonic = f"{'BL' if l_bit else 'B'}{cond_name if cond_name != 'AL' else ''}"

    lines.append(f"Mnemonic: {mnemonic}")
    lines.append("")
    lines.append("### Field Breakdown")
    lines.append(_format_field("24", l_bit, f"L (Link) = {l_bit} ({branch_type})"))
    lines.append(_format_field("23:0", offset_raw, f"Signed offset = {offset_signed} (words)"))
    lines.append(f"  Byte offset: {byte_offset} (offset << 2)")
    lines.append(f"  Effective target: {effective_note}")
    if l_bit:
        lines.append("  Note: BL stores return address in LR (R14).")

    return lines


def _decode_load_store(value: int, cond_name: str) -> list[str]:
    """Decode a single load/store (word/byte) instruction."""
    lines: list[str] = []
    lines.append("Instruction type: Load/Store (bits [27:26] = 01)")
    imm = _bits(value, 25, 25)
    p_bit = _bits(value, 24, 24)
    u_bit = _bits(value, 23, 23)
    b_bit = _bits(value, 22, 22)
    w_bit = _bits(value, 21, 21)
    l_bit = _bits(value, 20, 20)
    rn = _bits(value, 19, 16)
    rd = _bits(value, 15, 12)
    offset = _bits(value, 11, 0)

    op = "LDR" if l_bit else "STR"
    byte_suffix = "B" if b_bit else ""
    mnemonic = f"{op}{byte_suffix}{cond_name if cond_name != 'AL' else ''}"

    sign = "+" if u_bit else "-"

    lines.append(f"Mnemonic: {mnemonic}")
    lines.append("")
    lines.append("### Field Breakdown")
    lines.append(_format_field("25", imm, f"I = {imm} ({'Register offset' if imm else 'Immediate offset'})"))
    lines.append(_format_field("24", p_bit, f"P (Pre/Post index) = {p_bit} ({'Pre-indexed' if p_bit else 'Post-indexed'})"))
    lines.append(_format_field("23", u_bit, f"U (Up/Down) = {u_bit} ({sign} offset)"))
    lines.append(_format_field("22", b_bit, f"B (Byte/Word) = {b_bit} ({'Byte' if b_bit else 'Word'})"))
    lines.append(_format_field("21", w_bit, f"W (Write-back) = {w_bit} ({'Write-back to Rn' if w_bit else 'No write-back'})"))
    lines.append(_format_field("20", l_bit, f"L (Load/Store) = {l_bit} ({'Load (LDR)' if l_bit else 'Store (STR)'})"))
    lines.append(_format_field("19:16", rn, f"Rn (base register) = {_REG_NAMES.get(rn, f'R{rn}')}"))
    lines.append(_format_field("15:12", rd, f"Rd (src/dest register) = {_REG_NAMES.get(rd, f'R{rd}')}"))

    if not imm:
        lines.append(_format_field("11:0", offset, f"Immediate offset = {sign}{offset} (0x{offset:03X})"))
    else:
        shift_amount = _bits(offset, 11, 7)
        shift_type = _bits(offset, 6, 5)
        rm = _bits(offset, 3, 0)
        shift_names = {0: "LSL", 1: "LSR", 2: "ASR", 3: "ROR"}
        lines.append(_format_field("11:0", offset,
            f"Register offset: {sign}{_REG_NAMES.get(rm, f'R{rm}')} "
            f"{shift_names.get(shift_type, '??')} #{shift_amount}"))

    return lines


@mcp.tool()
def decode_instruction(hex_value: str) -> str:
    """Decode a 32-bit ARM AArch32 instruction from its hex encoding.

    Breaks the instruction into fields (condition code, opcode, registers,
    offsets, etc.) and shows the meaning of each field.

    Args:
        hex_value: A 32-bit hex string, e.g. "0xE3A01005" or "E3A01005".
    """
    # Parse the hex value
    cleaned = hex_value.strip().lower()
    if cleaned.startswith("0x"):
        cleaned = cleaned[2:]
    try:
        value = int(cleaned, 16)
    except ValueError:
        return f"Error: '{hex_value}' is not a valid hex value."
    if value < 0 or value > 0xFFFFFFFF:
        return f"Error: Value 0x{value:X} is out of range for a 32-bit instruction."

    lines: list[str] = []
    lines.append(f"# ARM AArch32 Instruction Decode: 0x{value:08X}")
    lines.append(f"Binary: {value:032b}")
    lines.append("")

    # Condition code [31:28]
    cond = _bits(value, 31, 28)
    cond_entry = _CONDITION_CODES.get(cond)
    if cond_entry:
        cond_name = cond_entry["suffix"].split("/")[0]  # first name
        cond_desc = cond_entry["name"]
    else:
        cond_name = f"0x{cond:X}"
        cond_desc = "Reserved / Unconditional extension (0xF)"

    lines.append("### Condition Code")
    lines.append(_format_field("31:28", cond, f"Condition = {cond_name} ({cond_desc})"))
    lines.append("")

    # Determine instruction class from bits [27:25]
    type_bits_27_25 = _bits(value, 27, 25)
    type_bits_27_26 = _bits(value, 27, 26)

    if type_bits_27_26 == 0b00:
        lines.extend(_decode_data_processing(value, cond_name))
    elif type_bits_27_25 == 0b101:
        lines.extend(_decode_branch(value, cond_name))
    elif type_bits_27_26 == 0b01:
        lines.extend(_decode_load_store(value, cond_name))
    elif type_bits_27_25 == 0b100:
        lines.append("Instruction type: Block Data Transfer / Load-Store Multiple (bits [27:25] = 100)")
        lines.append("(Detailed decode for LDM/STM not yet implemented.)")
    elif type_bits_27_25 == 0b110:
        lines.append("Instruction type: Coprocessor Data Transfer (bits [27:25] = 110)")
        lines.append("(Detailed decode for coprocessor instructions not yet implemented.)")
    elif type_bits_27_25 == 0b111:
        if _bits(value, 24, 24):
            lines.append("Instruction type: Software Interrupt / SWI (bits [27:24] = 1111)")
            swi_num = _bits(value, 23, 0)
            lines.append(f"  SWI number: 0x{swi_num:06X} ({swi_num})")
        else:
            lines.append("Instruction type: Coprocessor operation (bits [27:25] = 111, bit 24 = 0)")
            lines.append("(Detailed decode not yet implemented.)")
    else:
        lines.append(f"Instruction type: Unknown / not decoded (bits [27:25] = {type_bits_27_25:03b})")

    return "\n".join(lines)


@mcp.tool()
def explain_condition_code(suffix: str) -> str:
    """Explain an ARM condition code suffix in detail.

    Returns the full name, which CPSR/NZCV flags are tested, the flag
    condition expression, the opposite condition, and a common use case.

    Args:
        suffix: The condition code mnemonic, e.g. "EQ", "NE", "GT", "CC",
                "VS", "CS/HS", "AL".  Case-insensitive.
    """
    key = suffix.strip().upper()
    result = _SUFFIX_TO_CC.get(key)

    if result is None:
        all_suffixes = sorted({e["suffix"] for e in _CONDITION_CODES.values()})
        return (
            f"Error: '{suffix}' is not a recognised ARM condition code.\n\n"
            f"Valid condition codes: {', '.join(all_suffixes)}"
        )

    code, entry = result

    lines: list[str] = []
    lines.append(f"# Condition Code: {entry['suffix']}  (0x{code:X} / {code:04b})")
    lines.append("")
    lines.append(f"**Full name:** {entry['name']}")
    lines.append(f"**Encoding:** bits [31:28] = 0b{code:04b} (0x{code:X})")
    lines.append(f"**Flags tested:** {entry['flags_tested']}")
    lines.append(f"**Flag condition:** {entry['flag_condition']}")
    lines.append(f"**Opposite condition:** {entry['opposite']}")
    lines.append(f"**Common use case:** {entry['use_case']}")

    # Show a mini example
    lines.append("")
    first_suffix = entry["suffix"].split("/")[0]
    lines.append("### Example")
    if first_suffix == "AL":
        lines.append(f"  ADD R0, R1, R2     ; Always executes (AL is the default)")
    else:
        lines.append(f"  CMP R0, R1         ; Compare R0 and R1, updating CPSR flags")
        lines.append(f"  B{first_suffix} label        ; Branch to 'label' if {entry['flag_condition']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5: explain_calling_convention — AAPCS reference
# ---------------------------------------------------------------------------

_AAPCS_DATA: dict[str, dict] = {
    "aarch32": {
        "standard": "AAPCS32 (ARM Architecture Procedure Call Standard for AArch32)",
        "spec_ref": "IHI0042 — Procedure Call Standard for the Arm Architecture",
        "argument_registers": {
            "registers": ["R0", "R1", "R2", "R3"],
            "count": 4,
            "notes": (
                "Integer/pointer arguments are passed in R0-R3 (in order). "
                "64-bit values (long long) consume two consecutive registers aligned to an even register "
                "(e.g., R0:R1 or R2:R3). Additional arguments beyond R3 are passed on the stack."
            ),
        },
        "return_registers": {
            "integer": ["R0", "R1"],
            "floating_point": ["S0", "D0"],
            "notes": (
                "32-bit return values use R0. 64-bit return values use R0 (low) and R1 (high). "
                "Structs >4 bytes are returned via a hidden pointer in R0. "
                "Float return: S0 (single), D0 (double) when the VFP calling convention is active."
            ),
        },
        "caller_saved": {
            "integer": ["R0", "R1", "R2", "R3", "R12 (IP)", "R14 (LR)"],
            "floating_point": ["S0-S15", "D0-D7"],
            "notes": (
                "May be freely modified by a called function. "
                "R12 (IP) is used by linker veneers. "
                "R14 (LR) is overwritten by BL/BLX."
            ),
        },
        "callee_saved": {
            "integer": ["R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11 (FP)"],
            "floating_point": ["S16-S31", "D8-D15"],
            "notes": (
                "Must be preserved across the call boundary. "
                "R9 may be reserved as a static base on some platforms. "
                "D8-D15 are callee-saved; D16-D31 are caller-saved (NEON only)."
            ),
        },
        "stack_alignment": {
            "at_call_boundary": "8 bytes",
            "internal": "4 bytes minimum",
            "notes": "SP must be 8-byte aligned at every public function interface.",
        },
        "frame_pointer": {
            "register": "R11 (FP) in ARM state; R7 (FP) in Thumb state",
            "notes": (
                "Frame pointer is optional. R11 in ARM state, R7 in Thumb/Thumb-2. "
                "Frame record: {previous FP, LR} pushed at function entry."
            ),
        },
        "special_notes": [
            "R15 (PC): Writing to PC causes a branch. MOV PC, LR returns from subroutine.",
            "R13 (SP): Must not be used as a GP register. Banked across processor modes.",
            "Thumb interworking: BX/BLX use bit 0 to select ARM (0) vs Thumb (1) state.",
            "VFP/NEON: Hard-float ABI passes float/double args in S0-S15/D0-D7 instead of R0-R3.",
            "HFA: Up to 4 float/double members passed in consecutive S/D registers in VFP ABI.",
        ],
    },
    "aarch64": {
        "standard": "AAPCS64 (ARM Architecture Procedure Call Standard for AArch64)",
        "spec_ref": "IHI0055 — Procedure Call Standard for the Arm 64-bit Architecture",
        "argument_registers": {
            "registers": ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"],
            "count": 8,
            "notes": (
                "Integer/pointer arguments in X0-X7. 32-bit values use W-form (W0-W7). "
                "Arguments beyond X7 are passed on the stack, 8-byte aligned. "
                "FP/SIMD arguments use V0-V7."
            ),
        },
        "return_registers": {
            "integer": ["X0", "X1"],
            "floating_point": ["V0", "V1", "V2", "V3"],
            "notes": (
                "Scalar: X0 (64-bit) or W0 (32-bit). 128-bit integers: X0 (low) + X1 (high). "
                "FP scalars: S0/D0/Q0. Structs too large for registers are returned via pointer in X8. "
                "HFA/HVA of up to 4 members return in V0-V3."
            ),
        },
        "caller_saved": {
            "integer": ["X0-X17", "X18 (platform-dependent)"],
            "floating_point": ["V0-V7 (argument/result)", "V16-V31 (scratch)"],
            "notes": (
                "X0-X15: temporaries. X16/IP0, X17/IP1: linker veneer scratch. "
                "X18: platform register (TEB on Windows, reserved on macOS). "
                "X30 (LR): overwritten by BL/BLR — save before nested calls."
            ),
        },
        "callee_saved": {
            "integer": ["X19-X28", "X29 (FP)", "X30 (LR) when saved"],
            "floating_point": ["V8-V15 (low 64 bits only — D8-D15)"],
            "notes": (
                "X19-X28 are GP callee-saved. X29 (FP) must be set up when frame records are used. "
                "V8-V15: only bottom 64 bits (D8-D15) are preserved; upper halves are caller-saved."
            ),
        },
        "stack_alignment": {
            "at_call_boundary": "16 bytes",
            "internal": "16 bytes (SP must always be 16-byte aligned for memory access)",
            "notes": (
                "SP alignment faults if misaligned (SCTLR_EL1.SA=1). "
                "Stack frames are allocated in multiples of 16 bytes."
            ),
        },
        "frame_pointer": {
            "register": "X29 (FP)",
            "notes": (
                "Frame record: STP X29, X30, [SP, #-16]! at entry. "
                "X29 points to saved {FP, LR} pair. Linked list enables stack unwinding."
            ),
        },
        "special_notes": [
            "X8: Indirect result register — caller passes struct return address here.",
            "X16/IP0, X17/IP1: Linker veneer scratch — may be clobbered by any BL/BLR through a veneer.",
            "X18: Platform register — avoid in portable code (Windows TEB, macOS reserved).",
            "XZR/WZR: Zero register — reads return 0, writes are discarded.",
            "NEON/FP: V0-V7 for arguments, V0-V3 for returns. Overlapping views: Bn/Hn/Sn/Dn/Qn.",
            "HFA/HVA: Structs of 2-4 identical FP/SIMD members passed in consecutive V-registers.",
            "Variadic: Named args follow normal rules; variadic overflow goes to the stack.",
            "PAC (ARMv8.3+): Return addresses may be signed with PACIASP and verified with AUTIASP.",
        ],
    },
}


def _format_calling_convention(arch: str, data: dict) -> str:
    """Format an AAPCS data dict into readable text."""
    lines: list[str] = []
    lines.append(f"# Calling Convention: {arch.upper()}")
    lines.append(f"## {data['standard']}")
    lines.append(f"Specification: {data['spec_ref']}")
    lines.append("")

    arg = data["argument_registers"]
    lines.append("## Argument Registers")
    lines.append(f"Registers ({arg['count']}):  {', '.join(arg['registers'])}")
    lines.append(f"Notes: {arg['notes']}")
    lines.append("")

    ret = data["return_registers"]
    lines.append("## Return Value Registers")
    lines.append(f"Integer/pointer:    {', '.join(ret['integer'])}")
    lines.append(f"Floating-point:     {', '.join(ret['floating_point'])}")
    lines.append(f"Notes: {ret['notes']}")
    lines.append("")

    cs = data["caller_saved"]
    lines.append("## Caller-Saved (Volatile)")
    lines.append(f"Integer:         {', '.join(cs['integer'])}")
    lines.append(f"Floating-point:  {', '.join(cs['floating_point'])}")
    lines.append(f"Notes: {cs['notes']}")
    lines.append("")

    ce = data["callee_saved"]
    lines.append("## Callee-Saved (Non-Volatile)")
    lines.append(f"Integer:         {', '.join(ce['integer'])}")
    lines.append(f"Floating-point:  {', '.join(ce['floating_point'])}")
    lines.append(f"Notes: {ce['notes']}")
    lines.append("")

    sa = data["stack_alignment"]
    lines.append("## Stack Alignment")
    lines.append(f"At call boundary:  {sa['at_call_boundary']}")
    lines.append(f"Internal:          {sa['internal']}")
    lines.append(f"Notes: {sa['notes']}")
    lines.append("")

    fp = data["frame_pointer"]
    lines.append("## Frame Pointer")
    lines.append(f"Register: {fp['register']}")
    lines.append(f"Notes: {fp['notes']}")
    lines.append("")

    lines.append("## Special Notes")
    for i, note in enumerate(data["special_notes"], 1):
        lines.append(f"  {i}. {note}")

    return "\n".join(lines)


@mcp.tool()
def explain_calling_convention(architecture: str) -> str:
    """Explain the AAPCS calling convention for a given ARM architecture.

    Returns a comprehensive reference covering argument registers, return value
    registers, caller-saved vs callee-saved registers, stack alignment, frame
    pointer convention, and special notes.

    Args:
        architecture: "aarch32" for AAPCS32 or "aarch64" for AAPCS64.
    """
    arch = architecture.strip().lower()
    if arch not in _AAPCS_DATA:
        return "Error: architecture must be 'aarch32' or 'aarch64'."
    return _format_calling_convention(arch, _AAPCS_DATA[arch])


# ---------------------------------------------------------------------------
# Tool 6: search_registers — keyword search across all register data
# ---------------------------------------------------------------------------

@mcp.tool()
def search_registers(query: str, architecture: str | None = None) -> str:
    """Search ARM registers by keyword across names, aliases, descriptions, and usage notes.

    Args:
        query: Case-insensitive keyword to search for (e.g. "stack", "carry", "cache").
        architecture: Optional filter — "aarch32" or "aarch64".
                      If omitted, searches both architectures.
    """
    if architecture and architecture not in ("aarch32", "aarch64"):
        return "Error: architecture must be 'aarch32' or 'aarch64'."

    needle = query.lower()
    results: list[str] = []

    for reg in REGISTERS:
        if architecture and reg["architecture"] != architecture:
            continue

        matched_fields: list[str] = []

        if needle in reg["name"].lower():
            matched_fields.append(f"name: {reg['name']}")

        for alias in reg["aliases"]:
            if needle in alias.lower():
                matched_fields.append(f"alias: {alias}")

        if needle in reg["description"].lower():
            desc = reg["description"]
            idx = desc.lower().find(needle)
            start = max(0, idx - 40)
            end = min(len(desc), idx + len(query) + 40)
            snippet = desc[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(desc):
                snippet = snippet + "..."
            matched_fields.append(f'description: "{snippet}"')

        if reg["usage_notes"] and needle in reg["usage_notes"].lower():
            notes = reg["usage_notes"]
            idx = notes.lower().find(needle)
            start = max(0, idx - 40)
            end = min(len(notes), idx + len(query) + 40)
            snippet = notes[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(notes):
                snippet = snippet + "..."
            matched_fields.append(f'usage_notes: "{snippet}"')

        if matched_fields:
            aliases_str = f" ({', '.join(reg['aliases'])})" if reg["aliases"] else ""
            header = (
                f"**{reg['name']}{aliases_str}**  "
                f"[{reg['architecture']}  |  {reg['bit_width']}-bit  |  {reg['category']}]"
            )
            match_lines = "\n".join(f"  Matched in {m}" for m in matched_fields)
            results.append(f"{header}\n{match_lines}")

    if not results:
        arch_note = f" in {architecture}" if architecture else ""
        return f"No registers found matching '{query}'{arch_note}. Try a broader term."

    arch_note = f" in {architecture}" if architecture else " across all architectures"
    header_line = f"# Search results for '{query}'{arch_note}\n{len(results)} register(s) matched.\n"
    return header_line + "\n\n".join(results)


# ---------------------------------------------------------------------------
# Tool 7: decode_register_value — decode hex value against bit fields
# ---------------------------------------------------------------------------

def _parse_bits(bits_str: str) -> tuple[int, int]:
    """Parse a field bits string like '31', '4:0', or '63:48' into (high, low)."""
    bits_str = bits_str.strip()
    if ":" in bits_str:
        high_s, low_s = bits_str.split(":", 1)
        return int(high_s), int(low_s)
    single = int(bits_str)
    return single, single


@mcp.tool()
def decode_register_value(register_name: str, hex_value: str, architecture: str | None = None) -> str:
    """Decode a hex value against a register's defined bit fields.

    Shows the value of each named field extracted from the raw value.
    Works with registers that have bit fields defined (CPSR, SCTLR_EL1,
    FPSCR, DAIF, NZCV, CurrentEL, TTBR0_EL1, FPSR, FPCR, etc.).

    Args:
        register_name: Name or alias of a register, e.g. "CPSR", "SCTLR_EL1".
        hex_value: Hex string, e.g. "0x600001D3" or "600001D3".
        architecture: Optional filter — "aarch32" or "aarch64".
    """
    if architecture and architecture not in ("aarch32", "aarch64"):
        return "Error: architecture must be 'aarch32' or 'aarch64'."

    cleaned = hex_value.strip().lower()
    if cleaned.startswith("0x"):
        cleaned = cleaned[2:]
    try:
        value = int(cleaned, 16)
    except ValueError:
        return f"Error: '{hex_value}' is not a valid hexadecimal value."

    matches = _find_register(register_name, architecture)
    if not matches:
        return (
            f"No register found matching '{register_name}'"
            + (f" in {architecture}." if architecture else ".")
            + " Try lookup_register to check the exact name."
        )

    # Prefer registers that have fields defined
    matches_with_fields = [r for r in matches if r["fields"]]
    candidates = matches_with_fields if matches_with_fields else matches

    sections: list[str] = []
    for reg in candidates:
        bit_width = reg["bit_width"]
        max_value = (1 << bit_width) - 1

        lines: list[str] = []
        aliases_str = f" ({', '.join(reg['aliases'])})" if reg["aliases"] else ""
        lines.append(f"# {reg['name']}{aliases_str}  [{reg['architecture']}  |  {bit_width}-bit]")

        if value > max_value:
            lines.append(
                f"Warning: 0x{value:X} exceeds {bit_width}-bit range (max 0x{max_value:X})."
            )

        hex_digits = (bit_width + 3) // 4
        lines.append(f"Raw value : 0x{value:0{hex_digits}X}")
        binary_str = f"{value:0{bit_width}b}"
        grouped = " ".join(binary_str[i:i + 4] for i in range(0, len(binary_str), 4))
        lines.append(f"Binary    : {grouped}")
        lines.append("")

        if not reg["fields"]:
            lines.append(f"{reg['name']} has no named bit fields in this reference.")
            lines.append(f"Description: {reg['description']}")
            sections.append("\n".join(lines))
            continue

        lines.append("### Bit Field Decode")
        lines.append(f"  {'Field':<10} {'Bits':>7}  {'Hex':>6}  {'Dec':>5}  {'Bin':<12}  Description")
        lines.append("  " + "-" * 80)

        for field in reg["fields"]:
            high, low = _parse_bits(field["bits"])
            field_value = _bits(value, high, low)
            field_width = high - low + 1
            bin_str = f"{field_value:0{field_width}b}"
            bits_label = field["bits"] if ":" in field["bits"] else f"   {field['bits']}"
            lines.append(
                f"  {field['name']:<10} [{bits_label:>5}]  "
                f"0x{field_value:02X}  {field_value:>5}  {bin_str:<12}  {field['description']}"
            )

        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Exception Levels and Security States — data and tools
# ---------------------------------------------------------------------------

_EXCEPTION_LEVELS_AARCH64: dict[str, dict] = {
    "EL0": {
        "name": "EL0 — User / Application",
        "privilege": "Unprivileged (lowest)",
        "what_runs": [
            "User-space applications",
            "Unprivileged library code",
        ],
        "accessible_registers": (
            "General-purpose X0-X30, SP_EL0, PC, NZCV, FPCR, FPSR, TPIDR_EL0, "
            "TPIDRRO_EL0, CNTVCT_EL0, CNTFRQ_EL0 (read-only timer registers). "
            "No direct access to system control registers (SCTLR_EL1, etc.)."
        ),
        "stack_pointer": "SP_EL0 (always used at EL0)",
        "entry_mechanism": (
            "EL0 is entered via ERET from EL1 (or higher). The kernel sets "
            "SPSR_EL1.M[3:0] to 0b0000 (EL0t) before executing ERET."
        ),
        "exit_mechanism": (
            "Exceptions (SVC, page fault, IRQ, etc.) cause a synchronous or "
            "asynchronous trap to EL1 (or EL2 if HCR_EL2.TGE=1). "
            "ELR_EL1 captures the return address, SPSR_EL1 captures PSTATE."
        ),
    },
    "EL1": {
        "name": "EL1 — OS Kernel",
        "privilege": "Privileged (OS level)",
        "what_runs": [
            "Operating system kernels (Linux, Windows, etc.)",
            "Kernel modules and drivers",
            "Rich OS in TrustZone Secure world (Secure EL1): OP-TEE, Trusty",
        ],
        "accessible_registers": (
            "All EL0 registers plus: SCTLR_EL1, TTBR0_EL1, TTBR1_EL1, "
            "TCR_EL1, MAIR_EL1, ESR_EL1, FAR_EL1, VBAR_EL1, CONTEXTIDR_EL1, "
            "ELR_EL1, SPSR_EL1, SP_EL1, TPIDR_EL1, AMAIR_EL1, PAR_EL1, "
            "CNTKCTL_EL1, CSSELR_EL1. Cannot access EL2/EL3 registers "
            "(traps to higher EL)."
        ),
        "stack_pointer": (
            "SP_EL1 (selected when SPSel.SP=1) or SP_EL0 (when SPSel.SP=0). "
            "Kernel typically uses SP_EL1 for interrupt/exception stacks and "
            "SP_EL0 for per-task kernel stacks (Linux convention)."
        ),
        "key_registers": {
            "SPSR_EL1": (
                "Saved Process State Register — holds PSTATE of the context "
                "that took an exception to EL1. Restored by ERET."
            ),
            "ELR_EL1": (
                "Exception Link Register — holds the return address for "
                "exceptions taken to EL1. Used by ERET to resume."
            ),
            "VBAR_EL1": (
                "Vector Base Address Register — points to the EL1 exception "
                "vector table (Synchronous, IRQ, FIQ, SError, each with "
                "current EL SP_EL0/SP_ELx and lower EL AArch64/AArch32 entries)."
            ),
            "ESR_EL1": (
                "Exception Syndrome Register — encodes the cause (EC field) "
                "and details (ISS field) of the exception."
            ),
        },
        "entry_mechanism": (
            "Entered by exceptions from EL0 (SVC, IRQ, faults) or by "
            "ERET from EL2/EL3 with target EL=1."
        ),
        "exit_mechanism": (
            "Returns to EL0 via ERET. Takes exceptions to EL2 (HVC, "
            "second-stage faults if HCR_EL2.VM=1) or to EL3 (SMC)."
        ),
    },
    "EL2": {
        "name": "EL2 — Hypervisor",
        "privilege": "Hypervisor (higher than OS)",
        "what_runs": [
            "Hypervisors / Virtual Machine Monitors (KVM, Xen, Hyper-V)",
            "Secure EL2 (ARMv8.4+): Secure Partition Manager (Hafnium SPM)",
        ],
        "accessible_registers": (
            "All EL1-accessible registers (some with EL2 aliases) plus: "
            "HCR_EL2, VTCR_EL2, VTTBR_EL2, SCTLR_EL2, TTBR0_EL2, "
            "TCR_EL2, MAIR_EL2, ESR_EL2, FAR_EL2, HPFAR_EL2, "
            "ELR_EL2, SPSR_EL2, SP_EL2, VBAR_EL2, VMPIDR_EL2, "
            "VPIDR_EL2, CNTHCTL_EL2, ICH_*_EL2 (GIC virtualisation). "
            "Cannot access EL3 registers."
        ),
        "stack_pointer": (
            "SP_EL2 (when SPSel.SP=1 at EL2) or SP_EL0 (when SPSel.SP=0). "
            "Hypervisors typically use SP_EL2."
        ),
        "key_registers": {
            "HCR_EL2": (
                "Hypervisor Configuration Register — controls trapping, "
                "second-stage translation, and virtualisation features. "
                "Key bits: VM (stage-2 enable), SWIO, TGE (trap general "
                "exceptions to EL2), AMO/IMO/FMO (route async exceptions), "
                "RW (EL1 is AArch64), E2H (VHE — run host OS at EL2), "
                "TVM, TRVM, TSC (trap SMC), etc."
            ),
            "SPSR_EL2": "Saved PSTATE for exceptions taken to EL2.",
            "ELR_EL2": "Return address for exceptions taken to EL2.",
            "VTTBR_EL2": (
                "Stage-2 translation table base — VMID in upper bits, "
                "base address of guest physical->physical page tables."
            ),
        },
        "entry_mechanism": (
            "Entered by HVC from EL1, by traps configured in HCR_EL2, "
            "by second-stage translation faults, or by ERET from EL3 "
            "targeting EL2."
        ),
        "exit_mechanism": (
            "Returns to EL1/EL0 via ERET. Takes exceptions to EL3 via SMC."
        ),
    },
    "EL3": {
        "name": "EL3 — Secure Monitor",
        "privilege": "Highest privilege (firmware level)",
        "what_runs": [
            "ARM Trusted Firmware (TF-A / BL31)",
            "Secure Monitor (SM) / EL3 Runtime firmware",
            "SMC dispatcher — routes to Secure or Non-secure worlds",
        ],
        "accessible_registers": (
            "All system registers plus: SCR_EL3, SCTLR_EL3, TTBR0_EL3, "
            "TCR_EL3, MAIR_EL3, ESR_EL3, FAR_EL3, ELR_EL3, SPSR_EL3, "
            "SP_EL3, VBAR_EL3, MDCR_EL3, CPTR_EL3, ICC_SRE_EL3. "
            "Full access to all lower-EL registers."
        ),
        "stack_pointer": (
            "SP_EL3 (when SPSel.SP=1 at EL3) or SP_EL0 (when SPSel.SP=0). "
            "TF-A uses SP_EL3 during SMC handling."
        ),
        "key_registers": {
            "SCR_EL3": (
                "Secure Configuration Register — the master security-state "
                "switch. Key bits: NS (Non-secure: 0=Secure, 1=Non-secure), "
                "IRQ/FIQ/EA (route to EL3), RW (EL2 is AArch64), HCE (HVC "
                "enable), SMD (SMC disable), TWI/TWE, ST (Secure EL1 timer), "
                "NSE (ARMv9 RME: with NS, selects Root/Realm/Secure/NS world)."
            ),
            "SPSR_EL3": "Saved PSTATE for exceptions taken to EL3.",
            "ELR_EL3": "Return address for exceptions taken to EL3.",
        },
        "entry_mechanism": (
            "Entered by SMC instruction from EL1/EL2, by FIQ (when "
            "SCR_EL3.FIQ=1), or at system reset (boot starts at EL3)."
        ),
        "exit_mechanism": (
            "ERET from EL3 — returns to any lower EL. SCR_EL3.NS bit "
            "(and NSE for RME) determines which world the target runs in."
        ),
    },
}

_EXCEPTION_ENTRY_RETURN_AARCH64: dict[str, str] = {
    "exception_entry": (
        "When an exception is taken to ELx:\n"
        "  1. PSTATE is saved to SPSR_ELx (processor state at time of exception).\n"
        "  2. The return address is saved to ELR_ELx.\n"
        "     - For synchronous exceptions (SVC/HVC/SMC): address of the instruction.\n"
        "     - For asynchronous exceptions (IRQ/FIQ/SError): address of the first\n"
        "       instruction that was not executed (next instruction).\n"
        "  3. PSTATE.DAIF is updated (interrupts masked as configured).\n"
        "  4. PSTATE.M[3:0] is set to the target EL and selected SP.\n"
        "  5. If SCTLR_ELx.IESB=1, an implicit error synchronisation barrier occurs.\n"
        "  6. PC branches to VBAR_ELx + offset (offset depends on exception type,\n"
        "     source EL, and source execution state)."
    ),
    "exception_return": (
        "The ERET instruction returns from an exception:\n"
        "  1. PC is restored from ELR_ELx (resume address).\n"
        "  2. PSTATE is restored from SPSR_ELx (including EL, SP, DAIF, NZCV, etc.).\n"
        "  3. The PE resumes execution at the target EL and security state.\n"
        "  Note: ERET is only valid at EL1 or higher. Executing ERET at EL0 is UNDEFINED."
    ),
    "vector_table_layout": (
        "VBAR_ELx exception vector table offsets (each entry is 128 bytes / 32 instructions):\n"
        "  Offset  Source                       Exception type\n"
        "  0x000   Current EL, SP_EL0           Synchronous\n"
        "  0x080   Current EL, SP_EL0           IRQ\n"
        "  0x100   Current EL, SP_EL0           FIQ\n"
        "  0x180   Current EL, SP_EL0           SError\n"
        "  0x200   Current EL, SP_ELx           Synchronous\n"
        "  0x280   Current EL, SP_ELx           IRQ\n"
        "  0x300   Current EL, SP_ELx           FIQ\n"
        "  0x380   Current EL, SP_ELx           SError\n"
        "  0x400   Lower EL, AArch64            Synchronous\n"
        "  0x480   Lower EL, AArch64            IRQ\n"
        "  0x500   Lower EL, AArch64            FIQ\n"
        "  0x580   Lower EL, AArch64            SError\n"
        "  0x600   Lower EL, AArch32            Synchronous\n"
        "  0x680   Lower EL, AArch32            IRQ\n"
        "  0x700   Lower EL, AArch32            FIQ\n"
        "  0x780   Lower EL, AArch32            SError"
    ),
    "spsel_mechanism": (
        "SPSel (Stack Pointer Select):\n"
        "  PSTATE.SP (bit 0 of PSTATE.M) selects which SP to use:\n"
        "    SP = 0  ->  SP_EL0 is used (suffix 't' — thread mode, e.g. EL1t)\n"
        "    SP = 1  ->  SP_ELx is used (suffix 'h' — handler mode, e.g. EL1h)\n"
        "  Written via MSR SPSel, #imm or by ERET restoring SPSR.\n"
        "  At EL0, only SP_EL0 is available (SPSel is always 0).\n"
        "  Each EL has its own dedicated SP_ELx:\n"
        "    SP_EL0 — shared/used by EL0 and as alternate stack at higher ELs\n"
        "    SP_EL1 — dedicated EL1 stack pointer\n"
        "    SP_EL2 — dedicated EL2 stack pointer\n"
        "    SP_EL3 — dedicated EL3 stack pointer"
    ),
}

_SECURITY_STATES_AARCH64: list[dict] = [
    {
        "name": "Non-secure (NS)",
        "scr_bits": "SCR_EL3.NS=1, SCR_EL3.NSE=0",
        "els_available": "NS-EL0, NS-EL1, NS-EL2",
        "description": (
            "The normal world. Runs rich OS (Linux, Windows), hypervisors, "
            "and user applications. Cannot access Secure memory unless "
            "explicitly shared. Most software runs here."
        ),
    },
    {
        "name": "Secure (S)",
        "scr_bits": "SCR_EL3.NS=0, SCR_EL3.NSE=0",
        "els_available": "S-EL0, S-EL1, S-EL2 (ARMv8.4+)",
        "description": (
            "TrustZone Secure world. Runs trusted OS (OP-TEE, Trusty) at "
            "S-EL1, trusted applications at S-EL0, and optionally a Secure "
            "Partition Manager (Hafnium) at S-EL2. Has access to both Secure "
            "and Non-secure physical address spaces."
        ),
    },
    {
        "name": "Realm (R) — ARMv9 RME",
        "scr_bits": "SCR_EL3.NS=1, SCR_EL3.NSE=1",
        "els_available": "R-EL0, R-EL1, R-EL2",
        "description": (
            "Introduced by ARMv9 Realm Management Extension (RME). Realms "
            "are isolated execution environments protected from both the "
            "Normal world and the Secure world. Used by Confidential Compute "
            "Architecture (CCA) for confidential VMs. The Realm Management "
            "Monitor (RMM) runs at R-EL2."
        ),
    },
    {
        "name": "Root — ARMv9 RME",
        "scr_bits": "SCR_EL3.NS=0, SCR_EL3.NSE=1 (EL3 always Root)",
        "els_available": "EL3 only",
        "description": (
            "The Root world is where EL3 firmware (TF-A) executes. In RME, "
            "EL3 is always in Root state — it is the trusted entity that "
            "manages transitions between all four worlds (NS, Secure, Realm, "
            "Root) and programs the Granule Protection Table."
        ),
    },
]

_EXCEPTION_LEVELS_AARCH32: dict[str, dict] = {
    "modes": {
        "User (USR)": {
            "encoding": "0b10000 (0x10)",
            "privilege": "PL0 (unprivileged)",
            "banked_registers": "R13_usr (SP), R14_usr (LR)",
            "description": "Normal application execution mode. Cannot directly change mode.",
            "el_mapping": "Maps to EL0 in AArch64",
        },
        "FIQ": {
            "encoding": "0b10001 (0x11)",
            "privilege": "PL1",
            "banked_registers": "R8_fiq-R12_fiq, R13_fiq (SP), R14_fiq (LR), SPSR_fiq",
            "description": (
                "Fast Interrupt Request. Has extra banked registers (R8-R12) "
                "for fast handler entry without saving registers."
            ),
            "el_mapping": "Maps to EL1 in AArch64",
        },
        "IRQ": {
            "encoding": "0b10010 (0x12)",
            "privilege": "PL1",
            "banked_registers": "R13_irq (SP), R14_irq (LR), SPSR_irq",
            "description": "Normal Interrupt Request handler mode.",
            "el_mapping": "Maps to EL1 in AArch64",
        },
        "Supervisor (SVC)": {
            "encoding": "0b10011 (0x13)",
            "privilege": "PL1",
            "banked_registers": "R13_svc (SP), R14_svc (LR), SPSR_svc",
            "description": (
                "Entered on reset and SVC instruction (system calls). "
                "The primary kernel-mode for OS code."
            ),
            "el_mapping": "Maps to EL1 in AArch64",
        },
        "Abort (ABT)": {
            "encoding": "0b10111 (0x17)",
            "privilege": "PL1",
            "banked_registers": "R13_abt (SP), R14_abt (LR), SPSR_abt",
            "description": "Data Abort and Prefetch Abort exception handler mode.",
            "el_mapping": "Maps to EL1 in AArch64",
        },
        "Undefined (UND)": {
            "encoding": "0b11011 (0x1B)",
            "privilege": "PL1",
            "banked_registers": "R13_und (SP), R14_und (LR), SPSR_und",
            "description": "Undefined Instruction exception handler mode.",
            "el_mapping": "Maps to EL1 in AArch64",
        },
        "System (SYS)": {
            "encoding": "0b11111 (0x1F)",
            "privilege": "PL1",
            "banked_registers": "Shares R13_usr (SP), R14_usr (LR) — same as User mode",
            "description": (
                "Privileged mode that uses the User-mode register set. "
                "Used by OS for tasks requiring privilege but User-mode stack."
            ),
            "el_mapping": "Maps to EL1 in AArch64",
        },
        "Monitor (MON)": {
            "encoding": "0b10110 (0x16)",
            "privilege": "PL1 (Secure only — effectively highest privilege)",
            "banked_registers": "R13_mon (SP), R14_mon (LR), SPSR_mon",
            "description": (
                "TrustZone Secure Monitor. Entered via SMC instruction or "
                "secure exceptions. Manages world-switching between Secure "
                "and Non-secure states. Equivalent of EL3 in AArch64."
            ),
            "el_mapping": "Maps to EL3 in AArch64",
        },
        "Hyp (HYP)": {
            "encoding": "0b11010 (0x1A)",
            "privilege": "PL2 (Non-secure only)",
            "banked_registers": "R13_hyp (SP), ELR_hyp, SPSR_hyp (NOT R14_hyp — uses ELR_hyp instead)",
            "description": (
                "Hypervisor mode for hardware virtualisation. Has its own "
                "dedicated ELR_hyp for exception return (unlike other modes "
                "that use LR). Only exists in Non-secure state."
            ),
            "el_mapping": "Maps to EL2 in AArch64",
        },
    },
    "privilege_levels": {
        "PL0": {
            "modes": ["User (USR)"],
            "description": "Unprivileged — runs application code.",
            "aarch64_mapping": "EL0",
        },
        "PL1": {
            "modes": ["SVC", "IRQ", "FIQ", "Abort", "Undefined", "System", "Monitor"],
            "description": (
                "OS-privileged — all kernel-mode processor modes. "
                "Monitor mode is technically PL1 but has secure-only "
                "privileges analogous to EL3."
            ),
            "aarch64_mapping": "EL1 (Monitor -> EL3)",
        },
        "PL2": {
            "modes": ["Hyp"],
            "description": "Hypervisor — virtualisation support (Non-secure only).",
            "aarch64_mapping": "EL2",
        },
    },
    "el_mapping_summary": (
        "AArch32 to AArch64 mapping:\n"
        "  PL0  (User)                    <->  EL0\n"
        "  PL1  (SVC, IRQ, FIQ, ABT, UND, SYS) <->  EL1\n"
        "  PL2  (Hyp)                     <->  EL2\n"
        "  PL1-Secure (Monitor)           <->  EL3\n\n"
        "In AArch32, the different PL1 modes (SVC, IRQ, etc.) are all collapsed "
        "into EL1 in AArch64. The banked registers per mode are replaced by the "
        "ELR_EL1/SPSR_EL1 mechanism and dedicated SP_ELx stack pointers."
    ),
}


def _format_exception_levels_aarch64() -> str:
    """Format AArch64 exception level reference."""
    lines: list[str] = []
    lines.append("# AArch64 Exception Levels (EL0-EL3)")
    lines.append("")
    lines.append(
        "ARM AArch64 defines four Exception Levels (EL0-EL3), with EL3 being "
        "the most privileged. Higher ELs control and can trap operations from "
        "lower ELs. Each EL has its own stack pointer and exception state."
    )
    lines.append("")

    for el_key in ["EL0", "EL1", "EL2", "EL3"]:
        el = _EXCEPTION_LEVELS_AARCH64[el_key]
        lines.append(f"## {el['name']}")
        lines.append(f"Privilege: {el['privilege']}")
        lines.append("")
        lines.append("**What runs here:**")
        for item in el["what_runs"]:
            lines.append(f"  - {item}")
        lines.append("")
        lines.append(f"**Accessible system registers (summary):** {el['accessible_registers']}")
        lines.append(f"**Stack pointer:** {el['stack_pointer']}")
        lines.append("")

        if "key_registers" in el:
            lines.append("**Key registers at this EL:**")
            for reg_name, reg_desc in el["key_registers"].items():
                lines.append(f"  - {reg_name}: {reg_desc}")
            lines.append("")

        lines.append(f"**Exception entry:** {el['entry_mechanism']}")
        lines.append(f"**Exception exit:**  {el['exit_mechanism']}")
        lines.append("")

    # Exception entry/return mechanism
    lines.append("---")
    lines.append("## Exception Entry and Return Mechanism")
    lines.append("")
    lines.append(_EXCEPTION_ENTRY_RETURN_AARCH64["exception_entry"])
    lines.append("")
    lines.append(_EXCEPTION_ENTRY_RETURN_AARCH64["exception_return"])
    lines.append("")
    lines.append("### Vector Table Layout")
    lines.append(_EXCEPTION_ENTRY_RETURN_AARCH64["vector_table_layout"])
    lines.append("")

    # SPSel mechanism
    lines.append("---")
    lines.append("## Stack Pointer Selection (SPSel)")
    lines.append("")
    lines.append(_EXCEPTION_ENTRY_RETURN_AARCH64["spsel_mechanism"])
    lines.append("")

    # Security states
    lines.append("---")
    lines.append("## Security States")
    lines.append("")
    lines.append(
        "Each EL (except EL3) runs in a specific security state. EL3 controls "
        "which world lower ELs operate in via SCR_EL3."
    )
    lines.append("")
    for state in _SECURITY_STATES_AARCH64:
        lines.append(f"### {state['name']}")
        lines.append(f"  SCR_EL3 bits: {state['scr_bits']}")
        lines.append(f"  Available ELs: {state['els_available']}")
        lines.append(f"  {state['description']}")
        lines.append("")

    return "\n".join(lines)


def _format_exception_levels_aarch32() -> str:
    """Format AArch32 processor modes reference."""
    data = _EXCEPTION_LEVELS_AARCH32
    lines: list[str] = []
    lines.append("# AArch32 Processor Modes and Privilege Levels")
    lines.append("")
    lines.append(
        "AArch32 uses processor modes rather than numbered exception levels. "
        "Each mode has its own banked subset of registers (SP, LR, SPSR, and "
        "for FIQ mode, R8-R12). Modes map to privilege levels PL0-PL2."
    )
    lines.append("")

    lines.append("## Processor Modes")
    lines.append("")
    for mode_name, mode in data["modes"].items():
        lines.append(f"### {mode_name}")
        lines.append(f"  Encoding: {mode['encoding']}")
        lines.append(f"  Privilege: {mode['privilege']}")
        lines.append(f"  Banked registers: {mode['banked_registers']}")
        lines.append(f"  Description: {mode['description']}")
        lines.append(f"  AArch64 equivalent: {mode['el_mapping']}")
        lines.append("")

    lines.append("## Privilege Levels")
    lines.append("")
    for pl_name, pl in data["privilege_levels"].items():
        lines.append(f"### {pl_name}")
        lines.append(f"  Modes: {', '.join(pl['modes'])}")
        lines.append(f"  Description: {pl['description']}")
        lines.append(f"  AArch64 mapping: {pl['aarch64_mapping']}")
        lines.append("")

    lines.append("## AArch32 to AArch64 Mapping")
    lines.append("")
    lines.append(data["el_mapping_summary"])

    return "\n".join(lines)


@mcp.tool()
def explain_exception_levels(architecture: str = "aarch64") -> str:
    """Explain ARM Exception Levels, processor modes, and privilege hierarchy.

    For AArch64: covers EL0-EL3, what runs at each level, key system registers
    per EL, exception entry/return (ERET), vector tables, SPSel/SP_ELx stack
    pointer mechanism, and security states (NS, Secure, Realm, Root).

    For AArch32: covers processor modes (User, FIQ, IRQ, SVC, Abort, Undef,
    System, Monitor, Hyp), privilege levels PL0-PL2, banked registers per mode,
    and mapping to the AArch64 EL model.

    Args:
        architecture: "aarch64" (default) or "aarch32".
    """
    arch = architecture.strip().lower()
    if arch not in ("aarch32", "aarch64"):
        return "Error: architecture must be 'aarch32' or 'aarch64'."
    if arch == "aarch64":
        return _format_exception_levels_aarch64()
    return _format_exception_levels_aarch32()


# ---------------------------------------------------------------------------
# Security Model — data and tool
# ---------------------------------------------------------------------------

_SECURITY_MODEL_AARCH64: dict[str, object] = {
    "trustzone": {
        "overview": (
            "ARM TrustZone is a hardware security technology that creates two "
            "execution environments (worlds) isolated by hardware:\n"
            "  - Non-secure (Normal) world: runs rich OS, hypervisor, and apps.\n"
            "  - Secure world: runs trusted firmware and trusted OS.\n\n"
            "TrustZone extends across the entire SoC — bus fabric, peripherals, "
            "memory controllers, and interrupt controller all enforce the NS bit. "
            "A peripheral can be assigned as Secure-only, and Non-secure software "
            "cannot access it."
        ),
        "world_switching": (
            "World switching is mediated by EL3 firmware (Secure Monitor):\n"
            "  1. Non-secure world executes SMC (Secure Monitor Call) instruction.\n"
            "  2. Exception is taken to EL3 (SCR_EL3.NS is irrelevant at EL3 — "
            "     EL3 has access to everything).\n"
            "  3. EL3 firmware (TF-A BL31) saves NS-world context (registers, "
            "     ELR_EL3, SPSR_EL3).\n"
            "  4. EL3 sets SCR_EL3.NS=0 to switch to Secure world.\n"
            "  5. EL3 restores Secure-world context and issues ERET to S-EL1.\n"
            "  6. Return path is the reverse — Secure world issues SMC or returns "
            "     via a response mechanism; EL3 sets SCR_EL3.NS=1 and ERETs "
            "     back to NS-EL1/EL2."
        ),
        "scr_el3_ns": {
            "bit": "SCR_EL3.NS (bit 0)",
            "values": {
                "0": "Lower ELs are in Secure state",
                "1": "Lower ELs are in Non-secure state",
            },
            "notes": (
                "SCR_EL3.NS only affects EL2, EL1, and EL0. EL3 is always "
                "the most privileged and can access both worlds. The NS bit is "
                "propagated as a bus signal (AxPROT[1]) for all memory accesses "
                "from lower ELs."
            ),
        },
    },
    "secure_els": {
        "S-EL0": {
            "what_runs": "Trusted Applications (TAs) — e.g., DRM, key management, biometric processing.",
            "description": (
                "Unprivileged Secure execution. TAs run at S-EL0, managed by "
                "a trusted OS at S-EL1. Isolation between TAs is provided by "
                "S-EL1 page tables."
            ),
        },
        "S-EL1": {
            "what_runs": "Trusted OS (OP-TEE, Trusty, T-base) or Secure Partitions.",
            "description": (
                "The Secure-world kernel. Manages S-EL0 TAs, Secure memory, "
                "and Secure peripherals. In older TrustZone deployments, S-EL1 "
                "is the highest Secure-world EL below EL3."
            ),
        },
        "S-EL2": {
            "what_runs": "Secure Partition Manager (SPM) — typically Hafnium-based.",
            "description": (
                "Introduced in ARMv8.4-SecEL2. Provides a hypervisor-like layer "
                "in the Secure world. The SPM creates isolated Secure Partitions "
                "(SPs) at S-EL0/S-EL1, each with its own stage-2 address space. "
                "Enables FF-A (Firmware Framework for ARM) communication between "
                "SPs and the Normal world. S-EL2 is enabled by SCR_EL3.EEL2=1."
            ),
        },
    },
    "rme": {
        "overview": (
            "ARM Realm Management Extension (RME), part of ARMv9, extends the "
            "two-world TrustZone model to four worlds:\n"
            "  1. Non-secure (NS)  — normal world (rich OS, apps)\n"
            "  2. Secure (S)       — TrustZone secure world\n"
            "  3. Realm (R)        — confidential VMs, protected from all other worlds\n"
            "  4. Root             — EL3 firmware only\n\n"
            "RME adds SCR_EL3.NSE bit alongside SCR_EL3.NS to select among four worlds:\n"
            "  NSE=0, NS=0  ->  Secure\n"
            "  NSE=0, NS=1  ->  Non-secure\n"
            "  NSE=1, NS=0  ->  Root (EL3 only)\n"
            "  NSE=1, NS=1  ->  Realm"
        ),
        "gpt": {
            "name": "Granule Protection Table (GPT)",
            "description": (
                "The GPT is a hardware-enforced lookup table that assigns every "
                "physical memory granule (page) to one of the four worlds (NS, "
                "Secure, Realm, Root). It is programmed by EL3 firmware.\n\n"
                "GPT structure:\n"
                "  - Level 0 table: coarse-grained (e.g., 1GB or 512MB blocks)\n"
                "  - Level 1 table: fine-grained (e.g., 4KB per GPI entry)\n"
                "  - Each entry contains a Granule Protection Information (GPI) value:\n"
                "      0b0000 — No access (GPT fault)\n"
                "      0b0001 — Secure\n"
                "      0b0010 — Non-secure\n"
                "      0b0011 — Root\n"
                "      0b0100 — Realm\n"
                "      0b1111 — Any (accessible from all worlds)"
            ),
        },
        "gpc": {
            "name": "Granule Protection Checks (GPC)",
            "description": (
                "GPCs are performed by the MMU on every memory access. After "
                "stage-1 and stage-2 address translation, the physical address "
                "is checked against the GPT. If the current world does not match "
                "the granule's assigned world, a GPT fault (Granule Protection "
                "Fault) is generated. This prevents:\n"
                "  - Normal world accessing Realm or Secure memory\n"
                "  - Secure world accessing Realm memory\n"
                "  - Realm accessing Secure memory\n"
                "  Only Root (EL3) can reprogram the GPT."
            ),
        },
        "cca": {
            "name": "Confidential Compute Architecture (CCA)",
            "description": (
                "ARM CCA is the overall architecture built on RME for "
                "confidential computing:\n\n"
                "Components:\n"
                "  - RMM (Realm Management Monitor): runs at R-EL2, manages Realm VMs.\n"
                "    Provides attestation and memory protection for Realm guests.\n"
                "  - Realm VM: a confidential VM running at R-EL1/R-EL0, isolated from\n"
                "    the hypervisor (NS-EL2), the Secure world, and other Realms.\n"
                "  - Monitor (EL3/Root): TF-A firmware, manages GPT and world transitions.\n"
                "  - Normal-world hypervisor (NS-EL2): can manage Realm VM scheduling and\n"
                "    I/O, but cannot read/write Realm memory.\n\n"
                "Trust model:\n"
                "  - The Realm VM trusts only the RMM (R-EL2) and EL3 firmware.\n"
                "  - The hypervisor and normal OS are UNTRUSTED from the Realm's perspective.\n"
                "  - Hardware attestation proves Realm integrity to remote verifiers.\n"
                "  - Memory assigned to a Realm is protected by GPT — even DMA from\n"
                "    Normal-world devices is blocked."
            ),
        },
    },
}

_SECURITY_MODEL_AARCH32: dict[str, object] = {
    "trustzone": {
        "overview": (
            "TrustZone on AArch32 separates the processor into two worlds:\n"
            "  - Non-secure (Normal) world: PL0 (User) + PL1 (SVC, IRQ, etc.) + PL2 (Hyp)\n"
            "  - Secure world: Secure PL0 (User) + Secure PL1 (SVC, IRQ, etc.)\n"
            "  - Monitor mode: the gatekeeper between worlds\n\n"
            "The security state is indicated by the NS bit on the bus for all memory "
            "accesses. Peripherals and memory regions can be partitioned as Secure-only "
            "via the TrustZone Address Space Controller (TZASC) and TrustZone "
            "Protection Controller (TZPC)."
        ),
        "monitor_mode": (
            "Monitor mode (MON) is the AArch32 equivalent of EL3:\n"
            "  - Entered via SMC instruction or Secure exceptions (FIQ if configured).\n"
            "  - Has its own banked SP_mon, LR_mon, and SPSR_mon.\n"
            "  - Can access both Secure and Non-secure system registers.\n"
            "  - Responsible for saving/restoring world context on switch.\n"
            "  - Sets SCR.NS to control which world PL0/PL1 runs in."
        ),
        "scr_ns": {
            "register": "SCR (Secure Configuration Register, CP15 c1 c1 0)",
            "bit": "SCR.NS (bit 0)",
            "values": {
                "0": "Processor is in Secure state (PL0/PL1 access Secure resources)",
                "1": "Processor is in Non-secure state (PL0/PL1 access Non-secure resources)",
            },
            "other_bits": (
                "SCR also controls: IRQ/FIQ/EA routing to Monitor mode, "
                "HCE (HVC enable), SCD (SMC disable), nET (early termination), "
                "and SIF (Secure instruction fetch from NS memory)."
            ),
        },
        "world_switching": (
            "World-switch flow in AArch32:\n"
            "  1. Normal world executes SMC instruction.\n"
            "  2. Processor enters Monitor mode. CPSR is saved to SPSR_mon.\n"
            "     Return address is saved to LR_mon.\n"
            "  3. Monitor firmware saves Normal-world banked registers.\n"
            "  4. Monitor clears SCR.NS (sets to 0) to enter Secure state.\n"
            "  5. Monitor restores Secure-world banked registers.\n"
            "  6. Monitor executes MOVS PC, LR (or SUBS PC, LR, #0) to return "
            "     to Secure world PL1 (SVC mode).\n"
            "  7. Return path: Secure world issues SMC -> Monitor sets SCR.NS=1 "
            "     -> restores Normal-world context -> returns to Normal PL1."
        ),
    },
    "memory_partitioning": (
        "AArch32 TrustZone memory partitioning:\n"
        "  - TZASC (TrustZone Address Space Controller): configures DRAM regions "
        "    as Secure or Non-secure. Non-secure masters get a bus error or "
        "    zero-read on Secure region access.\n"
        "  - TZPC (TrustZone Protection Controller): controls peripheral access. "
        "    Each peripheral can be marked as Secure-only.\n"
        "  - TZMA (TrustZone Memory Adapter): protects on-chip SRAM regions.\n"
        "  - GIC (Generic Interrupt Controller): interrupts can be configured as "
        "    Group 0 (Secure/FIQ) or Group 1 (Non-secure/IRQ) to route to the "
        "    correct world."
    ),
}


def _format_security_model_aarch64() -> str:
    """Format AArch64 security model reference."""
    data = _SECURITY_MODEL_AARCH64
    tz = data["trustzone"]
    lines: list[str] = []

    lines.append("# AArch64 Security Model")
    lines.append("")

    # TrustZone
    lines.append("## TrustZone: Secure vs Non-secure Worlds")
    lines.append("")
    lines.append(tz["overview"])
    lines.append("")

    lines.append("### SCR_EL3.NS — World Selection Bit")
    scr = tz["scr_el3_ns"]
    lines.append(f"  Register/bit: {scr['bit']}")
    for val, meaning in scr["values"].items():
        lines.append(f"    {val} -> {meaning}")
    lines.append(f"  Notes: {scr['notes']}")
    lines.append("")

    lines.append("### World-Switching Mechanism")
    lines.append(tz["world_switching"])
    lines.append("")

    # Secure ELs
    lines.append("## Secure Exception Levels")
    lines.append("")
    sec_els = data["secure_els"]
    for el_name, el_info in sec_els.items():
        lines.append(f"### {el_name}")
        lines.append(f"  Runs: {el_info['what_runs']}")
        lines.append(f"  {el_info['description']}")
        lines.append("")

    # RME
    rme = data["rme"]
    lines.append("---")
    lines.append("## ARMv9 Realm Management Extension (RME)")
    lines.append("")
    lines.append(rme["overview"])
    lines.append("")

    lines.append(f"### {rme['gpt']['name']}")
    lines.append(rme["gpt"]["description"])
    lines.append("")

    lines.append(f"### {rme['gpc']['name']}")
    lines.append(rme["gpc"]["description"])
    lines.append("")

    lines.append(f"### {rme['cca']['name']}")
    lines.append(rme["cca"]["description"])

    return "\n".join(lines)


def _format_security_model_aarch32() -> str:
    """Format AArch32 security model reference."""
    data = _SECURITY_MODEL_AARCH32
    tz = data["trustzone"]
    lines: list[str] = []

    lines.append("# AArch32 Security Model (TrustZone)")
    lines.append("")

    lines.append("## Overview")
    lines.append(tz["overview"])
    lines.append("")

    lines.append("## Monitor Mode")
    lines.append(tz["monitor_mode"])
    lines.append("")

    lines.append("## SCR.NS — Security State Control")
    scr = tz["scr_ns"]
    lines.append(f"  Register: {scr['register']}")
    lines.append(f"  Bit: {scr['bit']}")
    for val, meaning in scr["values"].items():
        lines.append(f"    {val} -> {meaning}")
    lines.append(f"  Other key bits: {scr['other_bits']}")
    lines.append("")

    lines.append("## World-Switching Flow")
    lines.append(tz["world_switching"])
    lines.append("")

    lines.append("## Memory Partitioning")
    lines.append(data["memory_partitioning"])

    return "\n".join(lines)


@mcp.tool()
def explain_security_model(architecture: str) -> str:
    """Explain the ARM security model (TrustZone, RME, CCA).

    For AArch64: covers TrustZone Secure/Non-secure worlds, SCR_EL3.NS bit,
    world-switching mechanism, Secure EL0/EL1/EL2, and the ARMv9 Realm
    Management Extension (RME) including the four-world model (Root, Realm,
    Secure, Non-secure), Granule Protection Tables (GPT), Granule Protection
    Checks (GPC), and Confidential Compute Architecture (CCA).

    For AArch32: covers TrustZone basics with Monitor mode, SCR.NS bit,
    world-switching flow, and memory partitioning (TZASC, TZPC).

    Args:
        architecture: "aarch64" or "aarch32".
    """
    arch = architecture.strip().lower()
    if arch not in ("aarch32", "aarch64"):
        return "Error: architecture must be 'aarch32' or 'aarch64'."
    if arch == "aarch64":
        return _format_security_model_aarch64()
    return _format_security_model_aarch32()


# ---------------------------------------------------------------------------
# ARM Core / IP Reference Data
# ---------------------------------------------------------------------------

_ARM_CORES: dict[str, dict] = {
    # -----------------------------------------------------------------------
    # Cortex-A series
    # -----------------------------------------------------------------------
    "Cortex-A55": {
        "full_name": "ARM Cortex-A55",
        "series": "Cortex-A",
        "architecture": "ARMv8.2-A",
        "pipeline_depth": "8-stage in-order",
        "microarchitecture": "DynamIQ little core",
        "decode_width": "2-wide",
        "pipeline_type": "In-order",
        "key_features": [
            "In-order pipeline",
            "ARMv8.2-A with optional dot-product and FP16 extensions",
            "L1I 16-64 KB, L1D 16-64 KB (configurable)",
            "Optional shared L2 (64-256 KB) or DynamIQ Shared Unit L3",
            "Full NEON/FP support",
            "AArch32 and AArch64 support",
            "DynamIQ big.LITTLE pairing with Cortex-A75/A76/A77/A78",
        ],
        "target_market": "Mobile (efficiency core), IoT, embedded Linux",
        "notable_products": [
            "Qualcomm Snapdragon 845/855/865/888 (efficiency cluster)",
            "Samsung Exynos 9810/9820/990",
            "MediaTek Dimensity series (efficiency cores)",
            "HiSilicon Kirin 980/990",
        ],
        "year": 2017,
    },
    "Cortex-A57": {
        "full_name": "ARM Cortex-A57",
        "series": "Cortex-A",
        "architecture": "ARMv8.0-A",
        "pipeline_depth": "15+ stage out-of-order",
        "microarchitecture": "First-gen ARMv8 big core",
        "decode_width": "3-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "Out-of-order execution with 3-wide decode",
            "ARMv8.0-A AArch32 and AArch64",
            "L1I 48 KB, L1D 32 KB",
            "Shared L2 up to 2 MB",
            "Full NEON/FP, optional cryptographic extensions",
            "big.LITTLE pairing with Cortex-A53",
        ],
        "target_market": "Mobile (performance core), networking, server (early)",
        "notable_products": [
            "NVIDIA Tegra X1 (Nintendo Switch)",
            "Qualcomm Snapdragon 810/808",
            "Samsung Exynos 7420",
            "MediaTek Helio X10/X20",
        ],
        "year": 2012,
    },
    "Cortex-A72": {
        "full_name": "ARM Cortex-A72",
        "series": "Cortex-A",
        "architecture": "ARMv8.0-A",
        "pipeline_depth": "15+ stage out-of-order",
        "microarchitecture": "Second-gen ARMv8 big core (Austin)",
        "decode_width": "3-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "Out-of-order execution, improved over A57",
            "ARMv8.0-A AArch32 and AArch64",
            "L1I 48 KB, L1D 32 KB",
            "Shared L2 up to 2 MB",
            "20-30% perf improvement over A57 at same power",
            "Full NEON/FP, cryptographic extensions",
            "big.LITTLE pairing with Cortex-A53",
        ],
        "target_market": "Mobile, networking, automotive, SBC",
        "notable_products": [
            "Raspberry Pi 4 (Broadcom BCM2711)",
            "Qualcomm Snapdragon 650/652",
            "HiSilicon Kirin 950/955",
            "MediaTek Helio X25/X27",
        ],
        "year": 2015,
    },
    "Cortex-A73": {
        "full_name": "ARM Cortex-A73",
        "series": "Cortex-A",
        "architecture": "ARMv8.0-A",
        "pipeline_depth": "11-stage out-of-order",
        "microarchitecture": "Artemis — compact high-performance core",
        "decode_width": "2-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "Out-of-order execution with compact 2-wide decode",
            "ARMv8.0-A AArch32 and AArch64",
            "L1I 64 KB, L1D 32-64 KB",
            "Shared L2 up to 8 MB",
            "30% higher sustained performance than A72 (better power efficiency)",
            "Full NEON/FP, cryptographic extensions",
            "big.LITTLE pairing with Cortex-A53",
        ],
        "target_market": "Mobile (performance core)",
        "notable_products": [
            "HiSilicon Kirin 960/970",
            "MediaTek Helio X30",
            "Samsung Exynos 9610",
        ],
        "year": 2016,
    },
    "Cortex-A75": {
        "full_name": "ARM Cortex-A75",
        "series": "Cortex-A",
        "architecture": "ARMv8.2-A",
        "pipeline_depth": "13-stage out-of-order",
        "microarchitecture": "Prometheus — first DynamIQ big core",
        "decode_width": "3-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "Out-of-order execution, first DynamIQ big core",
            "ARMv8.2-A with optional FP16 and dot-product",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 512 KB, DynamIQ Shared Unit L3",
            "~22% IPC uplift over A73",
            "Full NEON/FP, cryptographic extensions",
            "DynamIQ pairing with Cortex-A55",
        ],
        "target_market": "Mobile (performance core), laptops",
        "notable_products": [
            "Qualcomm Snapdragon 845",
            "Samsung Exynos 9810",
            "HiSilicon Kirin 980",
        ],
        "year": 2017,
    },
    "Cortex-A76": {
        "full_name": "ARM Cortex-A76",
        "series": "Cortex-A",
        "architecture": "ARMv8.2-A",
        "pipeline_depth": "13-stage out-of-order",
        "microarchitecture": "Enyo — laptop-class performance",
        "decode_width": "4-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "Out-of-order execution with 4-wide decode",
            "ARMv8.2-A with FP16, dot-product",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 512 KB, shared L3 via DSU",
            "35% IPC uplift over A75, 40% power reduction",
            "Full NEON/FP, cryptographic extensions",
            "DynamIQ pairing with Cortex-A55",
        ],
        "target_market": "Mobile, laptops, Chromebooks",
        "notable_products": [
            "Qualcomm Snapdragon 855",
            "Samsung Exynos 980/990",
            "HiSilicon Kirin 990",
            "MediaTek Dimensity 1000",
        ],
        "year": 2018,
    },
    "Cortex-A77": {
        "full_name": "ARM Cortex-A77",
        "series": "Cortex-A",
        "architecture": "ARMv8.2-A",
        "pipeline_depth": "13-stage out-of-order",
        "microarchitecture": "Deimos — deeper backend, wider frontend",
        "decode_width": "4-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "Out-of-order execution, wider MOP cache vs A76",
            "ARMv8.2-A with FP16, dot-product",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 512 KB, shared L3 via DSU",
            "~20% IPC uplift over A76",
            "Improved branch prediction and prefetch",
            "DynamIQ pairing with Cortex-A55",
        ],
        "target_market": "Mobile (performance core)",
        "notable_products": [
            "Qualcomm Snapdragon 865",
            "Samsung Exynos 990",
            "MediaTek Dimensity 1000+",
        ],
        "year": 2019,
    },
    "Cortex-A78": {
        "full_name": "ARM Cortex-A78",
        "series": "Cortex-A",
        "architecture": "ARMv8.2-A",
        "pipeline_depth": "13-stage out-of-order",
        "microarchitecture": "Hercules",
        "decode_width": "4-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "Out-of-order execution, 4-wide decode",
            "ARMv8.2-A with FP16, dot-product, optional RAS",
            "L1I 32-64 KB, L1D 32-64 KB",
            "Private L2 up to 512 KB, shared L3 via DSU-110",
            "~20% sustained performance uplift over A77 at same power",
            "Optimized for 5 nm process",
            "DynamIQ pairing with Cortex-A55 or Cortex-A78C",
        ],
        "target_market": "Mobile (premium performance core)",
        "notable_products": [
            "Qualcomm Snapdragon 888",
            "Samsung Exynos 2100",
            "MediaTek Dimensity 1200",
        ],
        "year": 2020,
    },
    "Cortex-A78C": {
        "full_name": "ARM Cortex-A78C",
        "series": "Cortex-A",
        "architecture": "ARMv8.2-A",
        "pipeline_depth": "13-stage out-of-order",
        "microarchitecture": "Hercules-C (laptop/always-on variant)",
        "decode_width": "4-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "Based on Cortex-A78, optimized for sustained laptop workloads",
            "ARMv8.2-A with FP16, dot-product, RAS",
            "L1I 32-64 KB, L1D 32-64 KB",
            "Private L2 up to 512 KB, shared L3 via DSU",
            "Larger L3 cache support for laptop configs",
            "Up to 8 cores in cluster (vs 4 for A78)",
            "DynamIQ pairing with Cortex-A55",
        ],
        "target_market": "Laptops, always-connected PCs, Chromebooks",
        "notable_products": [
            "Qualcomm Snapdragon 8cx Gen 3 (efficiency cluster)",
            "Various laptop/Chromebook SoCs",
        ],
        "year": 2020,
    },
    "Cortex-A710": {
        "full_name": "ARM Cortex-A710",
        "series": "Cortex-A",
        "architecture": "ARMv9.0-A",
        "pipeline_depth": "13-stage out-of-order",
        "microarchitecture": "Matterhorn (first ARMv9 big core)",
        "decode_width": "4-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "First ARMv9 Cortex-A big core",
            "SVE2 support, NEON retained",
            "Memory tagging (MTE) support",
            "L1I 32-64 KB, L1D 32-64 KB",
            "Private L2 up to 512 KB, shared L3 via DSU-110",
            "~10% IPC uplift and 30% energy efficiency over A78",
            "AArch32 support retained (last A-series big core with it)",
            "DynamIQ pairing with Cortex-A510",
        ],
        "target_market": "Mobile (performance core)",
        "notable_products": [
            "Qualcomm Snapdragon 8 Gen 1",
            "Samsung Exynos 2200",
            "MediaTek Dimensity 9000",
        ],
        "year": 2021,
    },
    "Cortex-A715": {
        "full_name": "ARM Cortex-A715",
        "series": "Cortex-A",
        "architecture": "ARMv9.0-A",
        "pipeline_depth": "13-stage out-of-order",
        "microarchitecture": "Makalu",
        "decode_width": "4-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "ARMv9.0-A, AArch64-only (drops AArch32)",
            "SVE2, MTE support",
            "L1I 32-64 KB, L1D 32-64 KB",
            "Private L2 up to 512 KB, shared L3 via DSU-110",
            "~5% IPC uplift over A710, 20% energy efficiency improvement",
            "DynamIQ pairing with Cortex-A510",
        ],
        "target_market": "Mobile (performance core)",
        "notable_products": [
            "Qualcomm Snapdragon 8 Gen 2",
            "MediaTek Dimensity 9200",
        ],
        "year": 2022,
    },
    "Cortex-A720": {
        "full_name": "ARM Cortex-A720",
        "series": "Cortex-A",
        "architecture": "ARMv9.2-A",
        "pipeline_depth": "13-stage out-of-order",
        "microarchitecture": "Makalu-ELP (3rd gen ARMv9)",
        "decode_width": "4-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "ARMv9.2-A, AArch64-only",
            "SVE2, MTE, BRBE (Branch Record Buffer Extension)",
            "L1I 32-64 KB, L1D 32-64 KB",
            "Private L2 up to 512 KB, shared L3 via DSU-120",
            "~3-5% IPC uplift over A715, continued efficiency gains",
            "DynamIQ pairing with Cortex-A520",
        ],
        "target_market": "Mobile (performance core)",
        "notable_products": [
            "Qualcomm Snapdragon 8 Gen 3",
            "MediaTek Dimensity 9300",
        ],
        "year": 2023,
    },
    "Cortex-A725": {
        "full_name": "ARM Cortex-A725",
        "series": "Cortex-A",
        "architecture": "ARMv9.2-A",
        "pipeline_depth": "13-stage out-of-order",
        "microarchitecture": "4th gen ARMv9 big core",
        "decode_width": "5-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "ARMv9.2-A, AArch64-only",
            "SVE2, MTE, BRBE",
            "5-wide decode, wider than A720",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 512 KB, shared L3 via DSU-120",
            "IPC and energy efficiency improvements over A720",
            "DynamIQ pairing with Cortex-A520/A525",
        ],
        "target_market": "Mobile (performance core), laptops",
        "notable_products": [
            "Qualcomm Snapdragon 8 Elite (Snapdragon 8 Gen 4)",
            "MediaTek Dimensity 9400",
        ],
        "year": 2024,
    },
    # -----------------------------------------------------------------------
    # Cortex-X series
    # -----------------------------------------------------------------------
    "Cortex-X1": {
        "full_name": "ARM Cortex-X1",
        "series": "Cortex-X",
        "architecture": "ARMv8.2-A",
        "pipeline_depth": "13+ stage out-of-order",
        "microarchitecture": "Hera — first Cortex-X Custom program core",
        "decode_width": "5-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "5-wide decode (wider than A78)",
            "ARMv8.2-A with FP16, dot-product, RAS",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 1 MB (larger than A78)",
            "Shared L3 via DSU-110",
            "~30% peak performance uplift over A78",
            "Cortex-X Custom program: partner-configurable cache/cluster",
            "DynamIQ pairing with A78 and A55",
        ],
        "target_market": "Mobile (prime/peak performance core)",
        "notable_products": [
            "Qualcomm Snapdragon 888 (prime core)",
            "Samsung Exynos 2100 (prime core)",
        ],
        "year": 2020,
    },
    "Cortex-X2": {
        "full_name": "ARM Cortex-X2",
        "series": "Cortex-X",
        "architecture": "ARMv9.0-A",
        "pipeline_depth": "13+ stage out-of-order",
        "microarchitecture": "Matterhorn-X (first ARMv9 X-class)",
        "decode_width": "5-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "5-wide decode, first ARMv9 Cortex-X core",
            "SVE2, MTE support",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 1 MB",
            "Shared L3 via DSU-110",
            "~16% IPC uplift over X1",
            "DynamIQ pairing with A710 and A510",
        ],
        "target_market": "Mobile (prime/peak performance core)",
        "notable_products": [
            "Qualcomm Snapdragon 8 Gen 1 (prime core)",
            "Samsung Exynos 2200 (prime core)",
            "MediaTek Dimensity 9000 (prime core)",
        ],
        "year": 2021,
    },
    "Cortex-X3": {
        "full_name": "ARM Cortex-X3",
        "series": "Cortex-X",
        "architecture": "ARMv9.0-A",
        "pipeline_depth": "13+ stage out-of-order",
        "microarchitecture": "Makalu-X",
        "decode_width": "6-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "6-wide decode (widest ARM core at release)",
            "ARMv9.0-A, SVE2, MTE",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 1 MB",
            "Shared L3 via DSU-110 (up to 16 MB)",
            "~25% IPC uplift over X2",
            "DynamIQ pairing with A715 and A510",
        ],
        "target_market": "Mobile (prime/peak performance core)",
        "notable_products": [
            "Qualcomm Snapdragon 8 Gen 2 (prime core)",
            "MediaTek Dimensity 9200 (prime core)",
        ],
        "year": 2022,
    },
    "Cortex-X4": {
        "full_name": "ARM Cortex-X4",
        "series": "Cortex-X",
        "architecture": "ARMv9.2-A",
        "pipeline_depth": "13+ stage out-of-order",
        "microarchitecture": "3rd gen ARMv9 X-class",
        "decode_width": "6-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "6-wide decode, ARMv9.2-A",
            "SVE2, MTE, BRBE",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 2 MB",
            "Shared L3 via DSU-120 (up to 16 MB)",
            "~15% IPC uplift over X3",
            "DynamIQ pairing with A720 and A520",
        ],
        "target_market": "Mobile (prime/peak performance core)",
        "notable_products": [
            "Qualcomm Snapdragon 8 Gen 3 (prime core)",
            "MediaTek Dimensity 9300 (prime core)",
        ],
        "year": 2023,
    },
    "Cortex-X925": {
        "full_name": "ARM Cortex-X925",
        "series": "Cortex-X",
        "architecture": "ARMv9.2-A",
        "pipeline_depth": "13+ stage out-of-order",
        "microarchitecture": "4th gen ARMv9 X-class (widest ARM core)",
        "decode_width": "8-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "8-wide decode (widest ARM core to date)",
            "ARMv9.2-A, SVE2, MTE, BRBE",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 2 MB",
            "Shared L3 via DSU-120",
            "~36% IPC uplift over X4 in peak workloads",
            "DynamIQ pairing with A725 and A520/A525",
        ],
        "target_market": "Mobile (prime/peak performance core), laptops",
        "notable_products": [
            "Qualcomm Snapdragon 8 Elite (prime core)",
            "MediaTek Dimensity 9400 (prime core)",
        ],
        "year": 2024,
    },
    # -----------------------------------------------------------------------
    # Cortex-R series
    # -----------------------------------------------------------------------
    "Cortex-R5": {
        "full_name": "ARM Cortex-R5",
        "series": "Cortex-R",
        "architecture": "ARMv7-R",
        "pipeline_depth": "8-stage in-order, dual-issue",
        "microarchitecture": "Real-time profile, deterministic pipeline",
        "decode_width": "1-wide (dual-issue for some instruction pairs)",
        "pipeline_type": "In-order",
        "key_features": [
            "Deterministic real-time pipeline",
            "ARMv7-R with Thumb-2",
            "Tightly coupled memories (TCM) for low-latency access",
            "Optional ECC on caches and TCM",
            "Optional FPU (VFPv3-D16)",
            "Optional MPU (up to 16 regions)",
            "Lock-step mode for safety-critical applications",
        ],
        "target_market": "Automotive, industrial, storage controllers, real-time embedded",
        "notable_products": [
            "Texas Instruments Hercules TMS570 safety MCUs",
            "Xilinx Zynq UltraScale+ RPU subsystem",
            "Various automotive ECU SoCs",
        ],
        "year": 2010,
    },
    "Cortex-R52": {
        "full_name": "ARM Cortex-R52",
        "series": "Cortex-R",
        "architecture": "ARMv8-R (AArch32)",
        "pipeline_depth": "8-stage in-order, dual-issue",
        "microarchitecture": "First ARMv8-R core with hypervisor support",
        "decode_width": "2-wide (dual-issue)",
        "pipeline_type": "In-order",
        "key_features": [
            "ARMv8-R (AArch32 only) with EL2 hypervisor support",
            "Hardware-enforced software isolation via MPU + EL2",
            "Deterministic real-time pipeline",
            "TCM for low-latency access",
            "ECC on caches and TCM",
            "Optional FPU (VFPv5, single/double precision)",
            "Up to 4 cores per cluster, lock-step capable",
            "Designed for ASIL-D / SIL-3 functional safety",
        ],
        "target_market": "Automotive (ADAS, powertrain), industrial safety, medical",
        "notable_products": [
            "NXP S32G vehicle networking processors",
            "Texas Instruments J721E/J784S4 Jacinto processors",
            "Renesas R-Car H3/V3M safety island",
        ],
        "year": 2016,
    },
    "Cortex-R82": {
        "full_name": "ARM Cortex-R82",
        "series": "Cortex-R",
        "architecture": "ARMv8-R (AArch64)",
        "pipeline_depth": "8+ stage in-order",
        "microarchitecture": "First 64-bit real-time core, runs Linux",
        "decode_width": "2-wide",
        "pipeline_type": "In-order",
        "key_features": [
            "First ARMv8-R AArch64 core — can run Linux",
            "Full MMU support (optional) alongside MPU",
            "EL2 hypervisor support",
            "TCM plus optional cache hierarchy",
            "AArch64 and AArch32 support",
            "ECC on caches and TCM",
            "Up to 4 cores per cluster",
            "Rich OS + real-time on same core (Linux + RTOS)",
        ],
        "target_market": "Storage controllers, enterprise SSDs, smart NICs, automotive",
        "notable_products": [
            "Next-gen storage/SSD controller SoCs",
            "Smart NIC offload processors",
            "Automotive domain controllers",
        ],
        "year": 2020,
    },
    # -----------------------------------------------------------------------
    # Cortex-M series
    # -----------------------------------------------------------------------
    "Cortex-M0": {
        "full_name": "ARM Cortex-M0",
        "series": "Cortex-M",
        "architecture": "ARMv6-M",
        "pipeline_depth": "3-stage in-order",
        "microarchitecture": "Minimal gate-count MCU core (12K gates min)",
        "decode_width": "1-wide",
        "pipeline_type": "In-order",
        "key_features": [
            "Smallest ARM core, ~12,000 gates minimum",
            "ARMv6-M (Thumb / subset of Thumb-2)",
            "3-stage pipeline (fetch, decode, execute)",
            "No cache, no MPU (optional in some implementations)",
            "Deterministic interrupt latency (16 cycles)",
            "NVIC with 1-32 external interrupts",
            "Ultra-low power",
        ],
        "target_market": "IoT, ultra-low-power sensors, wearables, 8/16-bit MCU replacement",
        "notable_products": [
            "NXP LPC800 series",
            "STMicroelectronics STM32F0 series",
            "Nordic nRF51 series (Bluetooth LE)",
        ],
        "year": 2009,
    },
    "Cortex-M0+": {
        "full_name": "ARM Cortex-M0+",
        "series": "Cortex-M",
        "architecture": "ARMv6-M",
        "pipeline_depth": "2-stage in-order",
        "microarchitecture": "Optimized Cortex-M0 with 2-stage pipeline",
        "decode_width": "1-wide",
        "pipeline_type": "In-order",
        "key_features": [
            "2-stage pipeline (lower power than M0's 3-stage)",
            "ARMv6-M (Thumb / subset of Thumb-2)",
            "Optional MPU (8 regions)",
            "Micro Trace Buffer (MTB) for debug",
            "Single-cycle I/O port for fast GPIO",
            "15 cycles worst-case interrupt latency",
            "Even lower power than M0",
        ],
        "target_market": "IoT, ultra-low-power, sensor hubs, wearables",
        "notable_products": [
            "Microchip SAM D/L/C series",
            "NXP LPC80x, Kinetis KL series",
            "STMicroelectronics STM32L0 series",
            "Raspberry Pi RP2040 (dual Cortex-M0+)",
        ],
        "year": 2012,
    },
    "Cortex-M3": {
        "full_name": "ARM Cortex-M3",
        "series": "Cortex-M",
        "architecture": "ARMv7-M",
        "pipeline_depth": "3-stage in-order",
        "microarchitecture": "First Thumb-2-only MCU core with hardware divide",
        "decode_width": "1-wide",
        "pipeline_type": "In-order",
        "key_features": [
            "ARMv7-M with full Thumb-2 instruction set",
            "3-stage pipeline with branch speculation",
            "Hardware integer divide (SDIV, UDIV)",
            "Optional MPU (8 regions)",
            "NVIC with up to 240 interrupts, configurable priorities",
            "12 cycles worst-case interrupt latency",
            "Bit-banding for atomic bit-level access",
        ],
        "target_market": "General-purpose MCU, industrial, consumer electronics",
        "notable_products": [
            "STMicroelectronics STM32F1/F2 series",
            "NXP LPC1700 series",
            "Texas Instruments Stellaris/Tiva-C",
        ],
        "year": 2004,
    },
    "Cortex-M4": {
        "full_name": "ARM Cortex-M4",
        "series": "Cortex-M",
        "architecture": "ARMv7E-M",
        "pipeline_depth": "3-stage in-order",
        "microarchitecture": "M3 + DSP extensions + optional FPU",
        "decode_width": "1-wide",
        "pipeline_type": "In-order",
        "key_features": [
            "ARMv7E-M with DSP extensions (SIMD, saturating math)",
            "3-stage pipeline with branch speculation",
            "Optional single-precision FPU (FPv4-SP)",
            "Hardware integer divide",
            "Optional MPU (8 regions)",
            "NVIC with up to 240 interrupts",
            "12 cycles worst-case interrupt latency",
            "Single-cycle MAC, dual 16-bit MAC",
        ],
        "target_market": "IoT, motor control, audio processing, sensor fusion, industrial",
        "notable_products": [
            "STMicroelectronics STM32F4/G4 series",
            "NXP LPC4000, Kinetis K series",
            "Nordic nRF52 series (Bluetooth LE/5)",
            "Texas Instruments CC2640 (BLE)",
        ],
        "year": 2010,
    },
    "Cortex-M7": {
        "full_name": "ARM Cortex-M7",
        "series": "Cortex-M",
        "architecture": "ARMv7E-M",
        "pipeline_depth": "6-stage superscalar in-order",
        "microarchitecture": "Highest-performance Cortex-M, superscalar with caches",
        "decode_width": "2-wide (dual-issue for some instruction pairs)",
        "pipeline_type": "In-order (superscalar dual-issue)",
        "key_features": [
            "ARMv7E-M with DSP extensions",
            "6-stage superscalar pipeline, dual-issue",
            "Optional double-precision FPU (FPv5)",
            "L1 instruction cache (0-64 KB), L1 data cache (0-64 KB)",
            "Tightly coupled memory (TCM) for deterministic access",
            "Optional MPU (8 or 16 regions)",
            "Optional ECC on caches and TCM",
            "Branch prediction with BTB and BHT",
        ],
        "target_market": "High-performance MCU, motor control, industrial, automotive, audio",
        "notable_products": [
            "STMicroelectronics STM32F7/H7 series",
            "NXP i.MX RT 1050/1060/1170 crossover MCUs",
            "Microchip SAM V71/E70 series",
        ],
        "year": 2014,
    },
    "Cortex-M23": {
        "full_name": "ARM Cortex-M23",
        "series": "Cortex-M",
        "architecture": "ARMv8-M Baseline",
        "pipeline_depth": "2-stage in-order",
        "microarchitecture": "Secure IoT baseline core (TrustZone-M)",
        "decode_width": "1-wide",
        "pipeline_type": "In-order",
        "key_features": [
            "ARMv8-M Baseline (successor to M0+)",
            "ARM TrustZone for ARMv8-M (hardware security isolation)",
            "2-stage pipeline",
            "Optional MPU (up to 16 regions, per security state)",
            "Hardware integer divide",
            "Deterministic interrupt latency",
            "MVIC or NVIC with up to 240 interrupts",
            "Anti-tampering and secure boot support",
        ],
        "target_market": "Secure IoT, smart cards, secure sensors, constrained devices",
        "notable_products": [
            "Microchip SAM L10/L11 (TrustZone MCU)",
            "Nuvoton M2351 series",
        ],
        "year": 2016,
    },
    "Cortex-M33": {
        "full_name": "ARM Cortex-M33",
        "series": "Cortex-M",
        "architecture": "ARMv8-M Mainline",
        "pipeline_depth": "3-stage in-order",
        "microarchitecture": "Secure IoT mainline core (TrustZone-M + DSP)",
        "decode_width": "1-wide",
        "pipeline_type": "In-order",
        "key_features": [
            "ARMv8-M Mainline (successor to M4)",
            "ARM TrustZone for ARMv8-M",
            "3-stage pipeline",
            "Optional DSP extension (SIMD, saturating)",
            "Optional single-precision FPU (FPv5)",
            "Optional MPU (up to 16 regions, per security state)",
            "Coprocessor interface for custom accelerators",
            "12 cycles worst-case interrupt latency",
        ],
        "target_market": "Secure IoT, wearables, industrial sensors, smart home",
        "notable_products": [
            "STMicroelectronics STM32L5/U5 series",
            "NXP LPC5500 series",
            "Nordic nRF9160 (cellular IoT), nRF5340",
        ],
        "year": 2016,
    },
    "Cortex-M52": {
        "full_name": "ARM Cortex-M52",
        "series": "Cortex-M",
        "architecture": "ARMv8.1-M Mainline",
        "pipeline_depth": "4-stage in-order",
        "microarchitecture": "Helium-capable M-class core (compact ML/DSP)",
        "decode_width": "1-wide",
        "pipeline_type": "In-order",
        "key_features": [
            "ARMv8.1-M Mainline with MVE (Helium) support",
            "ARM TrustZone for ARMv8-M",
            "4-stage pipeline",
            "MVE (M-Profile Vector Extension / Helium) for ML and DSP",
            "Optional single-precision FPU (FPv5)",
            "Optional MPU (up to 16 regions, per security state)",
            "Compact alternative to M55 for area-constrained designs",
            "Coprocessor interface",
        ],
        "target_market": "IoT with ML/DSP, wearables, keyword spotting, sensor hubs",
        "notable_products": [
            "Announced 2023; early silicon in development",
        ],
        "year": 2023,
    },
    "Cortex-M55": {
        "full_name": "ARM Cortex-M55",
        "series": "Cortex-M",
        "architecture": "ARMv8.1-M Mainline",
        "pipeline_depth": "4-stage in-order",
        "microarchitecture": "First Helium (MVE) core — ML/DSP focused",
        "decode_width": "1-wide (with MVE beat-based execution)",
        "pipeline_type": "In-order",
        "key_features": [
            "ARMv8.1-M Mainline with MVE (Helium)",
            "ARM TrustZone for ARMv8-M",
            "4-stage pipeline with MVE beat-based execution",
            "Up to 15x ML performance uplift vs M4 (with Helium)",
            "Optional half/single/double-precision FPU",
            "Optional MPU (up to 16 regions, per security state)",
            "Pairs with Arm Ethos-U55/U65 NPU for ML acceleration",
            "PACBTI (Pointer Authentication and BTI) support",
        ],
        "target_market": "Endpoint AI/ML, IoT, keyword spotting, anomaly detection, wearables",
        "notable_products": [
            "Arm Corstone-300 reference design",
            "Various MCU vendor announcements (Alif Semiconductor Ensemble)",
        ],
        "year": 2020,
    },
    "Cortex-M85": {
        "full_name": "ARM Cortex-M85",
        "series": "Cortex-M",
        "architecture": "ARMv8.1-M Mainline",
        "pipeline_depth": "7-stage in-order",
        "microarchitecture": "Highest-performance M-class with Helium",
        "decode_width": "2-wide (dual-beat MVE)",
        "pipeline_type": "In-order",
        "key_features": [
            "ARMv8.1-M Mainline with MVE (Helium)",
            "ARM TrustZone for ARMv8-M",
            "7-stage pipeline, highest-performance M-class core",
            "Dual-beat MVE execution for higher throughput",
            "Optional half/single/double-precision FPU",
            "Optional MPU (up to 16 regions, per security state)",
            "L1 instruction cache (up to 64 KB), L1 data cache (up to 64 KB)",
            "PACBTI support, ECC on caches and TCM",
            "~30% higher scalar performance than M7",
        ],
        "target_market": "High-performance embedded, automotive, ML at the edge, DSP",
        "notable_products": [
            "Renesas RA8 series (RA8M1/RA8D1)",
            "Arm Corstone-310 reference design",
        ],
        "year": 2022,
    },
    # -----------------------------------------------------------------------
    # Neoverse series
    # -----------------------------------------------------------------------
    "Neoverse-N1": {
        "full_name": "ARM Neoverse N1",
        "series": "Neoverse-N",
        "architecture": "ARMv8.2-A",
        "pipeline_depth": "11-stage out-of-order",
        "microarchitecture": "Ares — first Neoverse generation",
        "decode_width": "4-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "Out-of-order execution, 4-wide decode",
            "ARMv8.2-A with FP16, dot-product, RAS",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 1 MB",
            "Scalable to 128 cores per chip (with CMN-600 mesh)",
            "Optimized for cloud/server throughput",
            "NEON, optional cryptographic extensions",
        ],
        "target_market": "Cloud, server, infrastructure, HPC entry",
        "notable_products": [
            "AWS Graviton2 (64 cores)",
            "Ampere Altra (80 cores)",
            "Huawei Kunpeng 920",
        ],
        "year": 2019,
    },
    "Neoverse-N2": {
        "full_name": "ARM Neoverse N2",
        "series": "Neoverse-N",
        "architecture": "ARMv9.0-A",
        "pipeline_depth": "11-stage out-of-order",
        "microarchitecture": "Perseus — first ARMv9 infrastructure core",
        "decode_width": "5-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "5-wide decode, first ARMv9 Neoverse core",
            "SVE2 with 128-bit or 256-bit vector lengths",
            "MTE (Memory Tagging Extension)",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 1 MB",
            "Scalable to 128+ cores (with CMN-700 mesh)",
            "~40% IPC uplift over N1 in server workloads",
            "BFLOAT16 support for inference acceleration",
        ],
        "target_market": "Cloud, server, 5G infrastructure, networking, HPC",
        "notable_products": [
            "AWS Graviton3 (64 cores)",
            "Ampere AmpereOne (up to 192 cores, N2-derived)",
            "NVIDIA Grace CPU (72 cores, N2-based)",
        ],
        "year": 2021,
    },
    "Neoverse-N3": {
        "full_name": "ARM Neoverse N3",
        "series": "Neoverse-N",
        "architecture": "ARMv9.2-A",
        "pipeline_depth": "13-stage out-of-order",
        "microarchitecture": "3rd gen Neoverse N-series",
        "decode_width": "6-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "6-wide decode, ARMv9.2-A",
            "SVE2, MTE, BRBE",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 2 MB",
            "Scalable with CMN-S3 mesh interconnect",
            "~20% IPC uplift over N2",
            "Confidential Computing Architecture (CCA) support",
            "Optimized for next-gen cloud and infrastructure",
        ],
        "target_market": "Cloud, server, 5G, HPC, confidential computing",
        "notable_products": [
            "Expected in AWS Graviton4, next-gen Ampere, and other server SoCs",
        ],
        "year": 2023,
    },
    "Neoverse-V1": {
        "full_name": "ARM Neoverse V1",
        "series": "Neoverse-V",
        "architecture": "ARMv8.4-A",
        "pipeline_depth": "13+ stage out-of-order",
        "microarchitecture": "Zeus — first Neoverse V-class (performance optimized)",
        "decode_width": "5-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "5-wide decode, performance-optimized server core",
            "ARMv8.4-A with SVE (256-bit vectors), NEON",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 1 MB",
            "Shared L3 via CMN-700 mesh",
            "~50% IPC uplift over N1 in HPC workloads",
            "BFLOAT16, Int8 dot-product for ML inference",
            "RAS, cryptographic extensions",
        ],
        "target_market": "HPC, technical computing, ML inference, high-end server",
        "notable_products": [
            "AWS Graviton3 (V1-derived architecture)",
            "NVIDIA Grace Hopper (HPC superchip partner CPU)",
            "Alibaba Yitian 710",
        ],
        "year": 2020,
    },
    "Neoverse-V2": {
        "full_name": "ARM Neoverse V2",
        "series": "Neoverse-V",
        "architecture": "ARMv9.0-A",
        "pipeline_depth": "13+ stage out-of-order",
        "microarchitecture": "Demeter — ARMv9 V-class",
        "decode_width": "5-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "5-wide decode, ARMv9.0-A",
            "SVE2 (128-bit fixed vector length), MTE",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 2 MB",
            "Shared L3 via CMN-700 mesh",
            "~20-30% uplift over V1 in server workloads",
            "BFLOAT16, Int8 for inference",
            "Confidential Compute support (Arm CCA)",
        ],
        "target_market": "HPC, cloud, ML training/inference, technical computing",
        "notable_products": [
            "NVIDIA Grace CPU Superchip (72 V2 cores)",
            "AWS Graviton4 (V2-derived)",
            "Microsoft Azure Cobalt 100",
        ],
        "year": 2022,
    },
    "Neoverse-V3": {
        "full_name": "ARM Neoverse V3",
        "series": "Neoverse-V",
        "architecture": "ARMv9.2-A",
        "pipeline_depth": "13+ stage out-of-order",
        "microarchitecture": "3rd gen Neoverse V-class (highest perf infrastructure core)",
        "decode_width": "8-wide",
        "pipeline_type": "Out-of-order",
        "key_features": [
            "8-wide decode (widest Neoverse core to date)",
            "ARMv9.2-A, SVE2, MTE, BRBE",
            "L1I 64 KB, L1D 64 KB",
            "Private L2 up to 2 MB",
            "Scalable with CMN-S3 mesh interconnect",
            "Major IPC uplift for HPC and AI workloads",
            "Confidential Computing Architecture (CCA)",
            "Designed for AI/ML training and inference at scale",
        ],
        "target_market": "HPC, hyperscale cloud, AI/ML training, technical computing",
        "notable_products": [
            "Expected in next-gen NVIDIA Grace, AWS, and Azure server platforms",
        ],
        "year": 2024,
    },
    "Neoverse-E1": {
        "full_name": "ARM Neoverse E1",
        "series": "Neoverse-E",
        "architecture": "ARMv8.2-A",
        "pipeline_depth": "8-stage in-order",
        "microarchitecture": "Helios — throughput-optimized infrastructure little core",
        "decode_width": "2-wide",
        "pipeline_type": "In-order",
        "key_features": [
            "In-order pipeline for throughput and efficiency",
            "ARMv8.2-A with FP16, dot-product",
            "Dual-threaded (simultaneous multithreading / SMT)",
            "L1I 32 KB, L1D 32 KB",
            "Private L2 up to 256 KB",
            "Optimized for data-plane throughput (networking, CDN)",
            "NEON, cryptographic extensions",
        ],
        "target_market": "5G infrastructure, networking, data-plane, CDN, edge",
        "notable_products": [
            "Marvell OCTEON networking processors",
            "5G baseband and RAN processing SoCs",
        ],
        "year": 2019,
    },
}

# Build a case-insensitive lookup map for core names
_CORE_LOOKUP: dict[str, dict] = {}
for _core_key, _core_data in _ARM_CORES.items():
    _CORE_LOOKUP[_core_key.upper()] = _core_data
    # Also register without hyphen variants
    _CORE_LOOKUP[_core_key.upper().replace("-", "")] = _core_data
    # Also register just the short name portion (e.g., "A78" for "Cortex-A78")
    if _core_key.startswith("Cortex-"):
        short = _core_key[len("Cortex-"):]
        _CORE_LOOKUP[short.upper()] = _core_data
    elif _core_key.startswith("Neoverse-"):
        short = _core_key[len("Neoverse-"):]
        _CORE_LOOKUP[short.upper()] = _core_data


def _normalize_core_name(name: str) -> str | None:
    """Normalize a core name input to its canonical key, or return None if not found."""
    cleaned = name.strip().upper().replace(" ", "-")
    # Direct match
    if cleaned in _CORE_LOOKUP:
        return cleaned
    # Try without hyphens
    no_hyphen = cleaned.replace("-", "")
    if no_hyphen in _CORE_LOOKUP:
        return no_hyphen
    # Try with common prefixes
    for prefix in ("CORTEX-", "NEOVERSE-", "CORTEX", "NEOVERSE", "ARM-", "ARM "):
        prefixed = prefix.upper() + cleaned
        normalized = prefixed.replace(" ", "-")
        if normalized in _CORE_LOOKUP:
            return normalized
        if normalized.replace("-", "") in _CORE_LOOKUP:
            return normalized.replace("-", "")
    return None


def _format_core(data: dict) -> str:
    """Format a core data dict into a readable reference card."""
    lines: list[str] = []
    lines.append(f"# {data['full_name']}")
    lines.append(f"Series: {data['series']}  |  Architecture: {data['architecture']}  |  Year: {data['year']}")
    lines.append("")
    lines.append("## Microarchitecture")
    lines.append(f"  Generation:    {data['microarchitecture']}")
    lines.append(f"  Pipeline:      {data['pipeline_depth']}")
    lines.append(f"  Decode width:  {data['decode_width']}")
    lines.append(f"  Pipeline type: {data['pipeline_type']}")
    lines.append("")
    lines.append("## Key Features")
    for feat in data["key_features"]:
        lines.append(f"  - {feat}")
    lines.append("")
    lines.append(f"## Target Market\n  {data['target_market']}")
    lines.append("")
    lines.append("## Notable Products / SoCs")
    for prod in data["notable_products"]:
        lines.append(f"  - {prod}")
    return "\n".join(lines)


@mcp.tool()
def lookup_core(core_name: str) -> str:
    """Look up an ARM core/IP by name and get a detailed reference card.

    Returns architecture version, pipeline details, decode width, key features,
    target market, and notable SoCs/products that use the core.

    Args:
        core_name: Name of the ARM core, e.g. "Cortex-A78", "Cortex-X4",
                   "Neoverse-N2", "Cortex-M55", "Cortex-R82".
                   Also accepts short forms like "A78", "X4", "N2", "M55".
                   Case-insensitive.
    """
    key = _normalize_core_name(core_name)
    if key is None:
        # Build list of available cores grouped by series
        series_groups: dict[str, list[str]] = {}
        for core_key, core_data in _ARM_CORES.items():
            series = core_data["series"]
            series_groups.setdefault(series, []).append(core_key)
        available = []
        for series in ["Cortex-A", "Cortex-X", "Cortex-R", "Cortex-M",
                       "Neoverse-N", "Neoverse-V", "Neoverse-E"]:
            if series in series_groups:
                available.append(f"  {series}: {', '.join(series_groups[series])}")
        return (
            f"No core found matching '{core_name}'.\n\n"
            "Available cores:\n" + "\n".join(available) + "\n\n"
            "Tip: You can use short names like 'A78', 'X4', 'M55', 'N2', etc."
        )

    return _format_core(_CORE_LOOKUP[key])


@mcp.tool()
def compare_cores(core_a: str, core_b: str) -> str:
    """Compare two ARM cores side by side.

    Shows a comparison table of architecture, pipeline type, decode width,
    key features, target market, and generation/year for both cores.

    Args:
        core_a: First core name (e.g. "Cortex-A78", "A78"). Case-insensitive.
        core_b: Second core name (e.g. "Cortex-X4", "X4"). Case-insensitive.
    """
    key_a = _normalize_core_name(core_a)
    key_b = _normalize_core_name(core_b)

    errors = []
    if key_a is None:
        errors.append(f"Core not found: '{core_a}'")
    if key_b is None:
        errors.append(f"Core not found: '{core_b}'")

    if errors:
        all_names = sorted(_ARM_CORES.keys())
        return (
            "\n".join(errors) + "\n\n"
            f"Available cores: {', '.join(all_names)}\n"
            "Tip: You can use short names like 'A78', 'X4', 'M55', 'N2', etc."
        )

    data_a = _CORE_LOOKUP[key_a]
    data_b = _CORE_LOOKUP[key_b]

    name_a = data_a["full_name"]
    name_b = data_b["full_name"]

    col_w = 40

    def _pad(text: str, width: int) -> str:
        if len(text) <= width:
            return text + " " * (width - len(text))
        return text[:width - 3] + "..."

    lines: list[str] = []
    lines.append(f"# Core Comparison: {name_a} vs {name_b}")
    lines.append("")

    # Table header
    lines.append(f"{'Attribute':<22}  {_pad(name_a, col_w)}  {_pad(name_b, col_w)}")
    lines.append("-" * (22 + 2 + col_w + 2 + col_w))

    # Comparison rows
    rows = [
        ("Series", data_a["series"], data_b["series"]),
        ("Architecture", data_a["architecture"], data_b["architecture"]),
        ("Year", str(data_a["year"]), str(data_b["year"])),
        ("Pipeline Type", data_a["pipeline_type"], data_b["pipeline_type"]),
        ("Pipeline Depth", data_a["pipeline_depth"], data_b["pipeline_depth"]),
        ("Decode Width", data_a["decode_width"], data_b["decode_width"]),
        ("Microarchitecture", data_a["microarchitecture"], data_b["microarchitecture"]),
        ("Target Market", data_a["target_market"], data_b["target_market"]),
    ]

    for label, val_a, val_b in rows:
        marker = "  " if val_a == val_b else "<>"
        lines.append(f"{label:<22}{marker}{_pad(val_a, col_w)}  {_pad(val_b, col_w)}")

    lines.append("")

    # Key features comparison
    lines.append("## Key Features Comparison")
    lines.append("")
    lines.append(f"### {name_a}")
    for feat in data_a["key_features"]:
        lines.append(f"  - {feat}")
    lines.append("")
    lines.append(f"### {name_b}")
    for feat in data_b["key_features"]:
        lines.append(f"  - {feat}")
    lines.append("")

    # Notable products comparison
    lines.append("## Notable Products / SoCs")
    lines.append("")
    lines.append(f"### {name_a}")
    for prod in data_a["notable_products"]:
        lines.append(f"  - {prod}")
    lines.append("")
    lines.append(f"### {name_b}")
    for prod in data_b["notable_products"]:
        lines.append(f"  - {prod}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: explain_page_table_format — AArch64 page table translation schemes
# ---------------------------------------------------------------------------

# Page table configuration data keyed by (granule_size, va_bits).
# Each entry describes the levels used, index bit ranges, block sizes,
# and which descriptor types are valid at each level.

_PAGE_TABLE_CONFIGS: dict[tuple[str, int], dict] = {
    # ---- 4KB granule ----
    ("4KB", 39): {
        "granule_bytes": 4096,
        "entries_per_table": 512,
        "index_bits": 9,
        "offset_bits": 12,
        "levels": {
            "L0": {
                "used": False,
                "index_bits_range": None,
                "va_range": None,
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": [],
            },
            "L1": {
                "used": True,
                "index_bits_range": "38:30",
                "va_range": (38, 30),
                "block_size": "1GB",
                "block_size_bytes": 1 << 30,
                "descriptors": ["block", "table"],
            },
            "L2": {
                "used": True,
                "index_bits_range": "29:21",
                "va_range": (29, 21),
                "block_size": "2MB",
                "block_size_bytes": 1 << 21,
                "descriptors": ["block", "table"],
            },
            "L3": {
                "used": True,
                "index_bits_range": "20:12",
                "va_range": (20, 12),
                "block_size": "4KB",
                "block_size_bytes": 4096,
                "descriptors": ["page"],
            },
        },
    },
    ("4KB", 48): {
        "granule_bytes": 4096,
        "entries_per_table": 512,
        "index_bits": 9,
        "offset_bits": 12,
        "levels": {
            "L0": {
                "used": True,
                "index_bits_range": "47:39",
                "va_range": (47, 39),
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": ["table"],
            },
            "L1": {
                "used": True,
                "index_bits_range": "38:30",
                "va_range": (38, 30),
                "block_size": "1GB",
                "block_size_bytes": 1 << 30,
                "descriptors": ["block", "table"],
            },
            "L2": {
                "used": True,
                "index_bits_range": "29:21",
                "va_range": (29, 21),
                "block_size": "2MB",
                "block_size_bytes": 1 << 21,
                "descriptors": ["block", "table"],
            },
            "L3": {
                "used": True,
                "index_bits_range": "20:12",
                "va_range": (20, 12),
                "block_size": "4KB",
                "block_size_bytes": 4096,
                "descriptors": ["page"],
            },
        },
    },
    ("4KB", 52): {
        "granule_bytes": 4096,
        "entries_per_table": 512,
        "index_bits": 9,
        "offset_bits": 12,
        "levels": {
            "L-1": {
                "used": True,
                "index_bits_range": "51:48",
                "va_range": (51, 48),
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": ["table"],
                "note": "Only 16 entries (4 index bits). Requires FEAT_LPA2.",
            },
            "L0": {
                "used": True,
                "index_bits_range": "47:39",
                "va_range": (47, 39),
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": ["table"],
            },
            "L1": {
                "used": True,
                "index_bits_range": "38:30",
                "va_range": (38, 30),
                "block_size": "1GB",
                "block_size_bytes": 1 << 30,
                "descriptors": ["block", "table"],
            },
            "L2": {
                "used": True,
                "index_bits_range": "29:21",
                "va_range": (29, 21),
                "block_size": "2MB",
                "block_size_bytes": 1 << 21,
                "descriptors": ["block", "table"],
            },
            "L3": {
                "used": True,
                "index_bits_range": "20:12",
                "va_range": (20, 12),
                "block_size": "4KB",
                "block_size_bytes": 4096,
                "descriptors": ["page"],
            },
        },
    },
    # ---- 16KB granule ----
    ("16KB", 39): {
        "granule_bytes": 16384,
        "entries_per_table": 2048,
        "index_bits": 11,
        "offset_bits": 14,
        "levels": {
            "L0": {
                "used": False,
                "index_bits_range": None,
                "va_range": None,
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": [],
            },
            "L1": {
                "used": True,
                "index_bits_range": "38:36",
                "va_range": (38, 36),
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": ["table"],
                "note": "Only 8 entries (3 index bits). No block descriptor at L1 for 16KB granule.",
            },
            "L2": {
                "used": True,
                "index_bits_range": "35:25",
                "va_range": (35, 25),
                "block_size": "32MB",
                "block_size_bytes": 32 << 20,
                "descriptors": ["block", "table"],
            },
            "L3": {
                "used": True,
                "index_bits_range": "24:14",
                "va_range": (24, 14),
                "block_size": "16KB",
                "block_size_bytes": 16384,
                "descriptors": ["page"],
            },
        },
    },
    ("16KB", 48): {
        "granule_bytes": 16384,
        "entries_per_table": 2048,
        "index_bits": 11,
        "offset_bits": 14,
        "levels": {
            "L0": {
                "used": True,
                "index_bits_range": "47",
                "va_range": (47, 47),
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": ["table"],
                "note": "Only 2 entries (1 index bit).",
            },
            "L1": {
                "used": True,
                "index_bits_range": "46:36",
                "va_range": (46, 36),
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": ["table"],
                "note": "No block descriptor at L1 for 16KB granule.",
            },
            "L2": {
                "used": True,
                "index_bits_range": "35:25",
                "va_range": (35, 25),
                "block_size": "32MB",
                "block_size_bytes": 32 << 20,
                "descriptors": ["block", "table"],
            },
            "L3": {
                "used": True,
                "index_bits_range": "24:14",
                "va_range": (24, 14),
                "block_size": "16KB",
                "block_size_bytes": 16384,
                "descriptors": ["page"],
            },
        },
    },
    ("16KB", 52): {
        "granule_bytes": 16384,
        "entries_per_table": 2048,
        "index_bits": 11,
        "offset_bits": 14,
        "levels": {
            "L0": {
                "used": True,
                "index_bits_range": "51:47",
                "va_range": (51, 47),
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": ["table"],
                "note": "Requires FEAT_LPA2. Up to 32 entries (5 index bits).",
            },
            "L1": {
                "used": True,
                "index_bits_range": "46:36",
                "va_range": (46, 36),
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": ["table"],
            },
            "L2": {
                "used": True,
                "index_bits_range": "35:25",
                "va_range": (35, 25),
                "block_size": "32MB",
                "block_size_bytes": 32 << 20,
                "descriptors": ["block", "table"],
            },
            "L3": {
                "used": True,
                "index_bits_range": "24:14",
                "va_range": (24, 14),
                "block_size": "16KB",
                "block_size_bytes": 16384,
                "descriptors": ["page"],
            },
        },
    },
    # ---- 64KB granule ----
    ("64KB", 39): {
        "granule_bytes": 65536,
        "entries_per_table": 8192,
        "index_bits": 13,
        "offset_bits": 16,
        "levels": {
            "L0": {
                "used": False,
                "index_bits_range": None,
                "va_range": None,
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": [],
            },
            "L1": {
                "used": False,
                "index_bits_range": None,
                "va_range": None,
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": [],
            },
            "L2": {
                "used": True,
                "index_bits_range": "38:29",
                "va_range": (38, 29),
                "block_size": "512MB",
                "block_size_bytes": 512 << 20,
                "descriptors": ["block", "table"],
                "note": "Only 1024 entries (10 index bits) for 39-bit VA.",
            },
            "L3": {
                "used": True,
                "index_bits_range": "28:16",
                "va_range": (28, 16),
                "block_size": "64KB",
                "block_size_bytes": 65536,
                "descriptors": ["page"],
            },
        },
    },
    ("64KB", 48): {
        "granule_bytes": 65536,
        "entries_per_table": 8192,
        "index_bits": 13,
        "offset_bits": 16,
        "levels": {
            "L0": {
                "used": False,
                "index_bits_range": None,
                "va_range": None,
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": [],
            },
            "L1": {
                "used": True,
                "index_bits_range": "47:42",
                "va_range": (47, 42),
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": ["table"],
                "note": "Only 64 entries (6 index bits). No block descriptor at L1 for 64KB granule.",
            },
            "L2": {
                "used": True,
                "index_bits_range": "41:29",
                "va_range": (41, 29),
                "block_size": "512MB",
                "block_size_bytes": 512 << 20,
                "descriptors": ["block", "table"],
            },
            "L3": {
                "used": True,
                "index_bits_range": "28:16",
                "va_range": (28, 16),
                "block_size": "64KB",
                "block_size_bytes": 65536,
                "descriptors": ["page"],
            },
        },
    },
    ("64KB", 52): {
        "granule_bytes": 65536,
        "entries_per_table": 8192,
        "index_bits": 13,
        "offset_bits": 16,
        "levels": {
            "L1": {
                "used": True,
                "index_bits_range": "51:42",
                "va_range": (51, 42),
                "block_size": None,
                "block_size_bytes": None,
                "descriptors": ["table"],
                "note": "Requires FEAT_LPA2. Up to 1024 entries (10 index bits).",
            },
            "L2": {
                "used": True,
                "index_bits_range": "41:29",
                "va_range": (41, 29),
                "block_size": "512MB",
                "block_size_bytes": 512 << 20,
                "descriptors": ["block", "table"],
            },
            "L3": {
                "used": True,
                "index_bits_range": "28:16",
                "va_range": (28, 16),
                "block_size": "64KB",
                "block_size_bytes": 65536,
                "descriptors": ["page"],
            },
        },
    },
}

# TCR_EL1 TG0/TG1 field encodings for each granule size.
_TCR_GRANULE_ENCODINGS: dict[str, dict[str, str]] = {
    "4KB":  {"TG0": "0b00", "TG1": "0b10"},
    "16KB": {"TG0": "0b10", "TG1": "0b01"},
    "64KB": {"TG0": "0b01", "TG1": "0b11"},
}

# IPS / PS field encodings in TCR_EL1 bits [34:32].
_IPS_ENCODINGS: list[dict] = [
    {"value": "0b000", "pa_bits": 32, "pa_size": "4GB"},
    {"value": "0b001", "pa_bits": 36, "pa_size": "64GB"},
    {"value": "0b010", "pa_bits": 40, "pa_size": "1TB"},
    {"value": "0b011", "pa_bits": 42, "pa_size": "4TB"},
    {"value": "0b100", "pa_bits": 44, "pa_size": "16TB"},
    {"value": "0b101", "pa_bits": 48, "pa_size": "256TB"},
    {"value": "0b110", "pa_bits": 52, "pa_size": "4PB (requires FEAT_LPA / FEAT_LPA2)"},
]


def _build_va_layout_diagram(config: dict, va_bits: int) -> str:
    """Build an ASCII diagram showing which VA bits index each translation level."""
    lines: list[str] = []

    # Collect active segments in descending bit order.
    segments: list[tuple[str, int, int]] = []
    for level_name, level_data in config["levels"].items():
        if level_data["used"] and level_data["va_range"] is not None:
            high, low = level_data["va_range"]
            segments.append((level_name, high, low))
    segments.sort(key=lambda s: s[1], reverse=True)

    # Add the page offset segment.
    offset_bits = config["offset_bits"]
    segments.append(("Offset", offset_bits - 1, 0))

    # Build the diagram with a box per segment.
    top_border = "  +"
    label_row = "  |"
    bits_row = "  |"
    bot_border = "  +"

    for seg_label, high, low in segments:
        bit_range = f"[{high}:{low}]" if high != low else f"[{high}]"
        width = high - low + 1
        cell_w = max(len(seg_label), len(bit_range), width) + 2
        top_border += "-" * cell_w + "+"
        label_row += f" {seg_label:^{cell_w - 2}} " + "|"
        bits_row += f" {bit_range:^{cell_w - 2}} " + "|"
        bot_border += "-" * cell_w + "+"

    lines.append(f"  {va_bits - 1}{'':>{len(top_border) - 6}}0")
    lines.append(top_border)
    lines.append(label_row)
    lines.append(bits_row)
    lines.append(bot_border)

    return "\n".join(lines)


def _format_page_table_config(granule: str, va_bits: int, config: dict) -> str:
    """Format the full page table format explanation."""
    lines: list[str] = []
    lines.append(f"# AArch64 Page Table Format: {granule} Granule, {va_bits}-bit VA")
    lines.append("")

    # Overview section
    lines.append("## Overview")
    lines.append(f"  Translation granule:       {granule} ({config['granule_bytes']} bytes)")
    lines.append(f"  Virtual address bits:      {va_bits}")
    va_range_size = 1 << va_bits
    if va_bits <= 48:
        va_tb = va_range_size >> 40
        lines.append(f"  VA range per TTBR:         {va_tb} TB (2^{va_bits} bytes)")
    else:
        va_pb = va_range_size >> 50
        lines.append(f"  VA range per TTBR:         {va_pb} PB (2^{va_bits} bytes)")
    lines.append(f"  Page table entry size:     8 bytes (64 bits, always)")
    lines.append(f"  Entries per full table:     {config['entries_per_table']}")
    lines.append(f"  Index bits per full level:  {config['index_bits']} (= log2({config['entries_per_table']}))")
    lines.append(f"  Page offset bits:          {config['offset_bits']} (= log2({config['granule_bytes']}))")
    lines.append("")

    # Count and list active levels
    active_levels = [(name, data) for name, data in config["levels"].items() if data["used"]]
    lines.append(f"  Active translation levels: {len(active_levels)} ({', '.join(n for n, _ in active_levels)})")
    lines.append("")

    # VA bit layout diagram
    lines.append("## Virtual Address Bit Layout")
    lines.append("")
    lines.append(_build_va_layout_diagram(config, va_bits))
    lines.append("")

    # Per-level detail
    lines.append("## Translation Level Details")
    lines.append("")
    for level_name, level_data in config["levels"].items():
        if not level_data["used"]:
            lines.append(f"### {level_name}: Not used in this configuration.")
            lines.append("")
            continue

        lines.append(f"### {level_name}")
        lines.append(f"  Index bit range:  VA[{level_data['index_bits_range']}]")
        if level_data["va_range"]:
            high, low = level_data["va_range"]
            num_bits = high - low + 1
            num_entries = 1 << num_bits
            table_size = num_entries * 8
            lines.append(f"  Number of index bits: {num_bits}")
            lines.append(f"  Entries in table:     {num_entries}")
            if table_size >= 1024:
                lines.append(f"  Table size:           {table_size} bytes ({table_size // 1024} KB)")
            else:
                lines.append(f"  Table size:           {table_size} bytes")

        # Descriptor types
        desc_types = level_data["descriptors"]
        lines.append(f"  Valid descriptor types:")
        for dt in desc_types:
            if dt == "table":
                lines.append(f"    - Table descriptor  (bits [1:0] = 0b11) -> points to next-level table")
            elif dt == "block":
                lines.append(f"    - Block descriptor  (bits [1:0] = 0b01) -> maps a contiguous {level_data['block_size']} region")
            elif dt == "page":
                lines.append(f"    - Page descriptor   (bits [1:0] = 0b11) -> maps a final {level_data['block_size']} page")
        if not desc_types:
            lines.append(f"    (none -- level not used)")

        if level_data.get("block_size"):
            lines.append(f"  Mapped region size:   {level_data['block_size']} ({level_data['block_size_bytes']} bytes)")

        if level_data.get("note"):
            lines.append(f"  Note: {level_data['note']}")

        lines.append("")

    # Descriptor format
    lines.append("## Page Table Entry (PTE) Descriptor Format -- 64 bits")
    lines.append("")
    lines.append("  All valid descriptors (block, table, page) share a common 64-bit format.")
    lines.append("  The type is determined by bits [1:0] and the current translation level.")
    lines.append("")
    lines.append("  ### Common Bit Fields (Block and Page Descriptors)")
    lines.append("")
    lines.append("  | Bits    | Field      | Description                                          |")
    lines.append("  |---------|------------|------------------------------------------------------|")
    lines.append("  | [63]    | PBHA/SW    | Page-Based Hardware Attributes or software use       |")
    lines.append("  | [54]    | UXN (XN)   | Unprivileged Execute-Never: 1=EL0 cannot execute     |")
    lines.append("  | [53]    | PXN        | Privileged Execute-Never: 1=EL1+ cannot execute      |")
    lines.append("  | [52]    | Contiguous | Contiguous hint: adjacent entries map contiguous block|")
    lines.append("  | [51]    | DBM        | Dirty Bit Modifier (FEAT_HAFDBS): HW dirty tracking  |")
    lines.append("  | [50]    | GP         | Guarded Page (FEAT_BTI): enables BTI checks          |")
    lines.append("  | [47:n]  | OA[47:n]   | Output Address: physical address of mapped region    |")
    lines.append("  | [11]    | nG         | Not Global: 1=ASID-tagged TLB entry (per-process)    |")
    lines.append("  | [10]    | AF         | Access Flag: must be 1 for valid access (or HW set)  |")
    lines.append("  | [9:8]   | SH[1:0]    | Shareability: 00=Non, 10=Outer, 11=Inner Shareable   |")
    lines.append("  | [7:6]   | AP[2:1]    | Access Permissions (read/write control)              |")
    lines.append("  | [5]     | NS         | Non-Secure: controls output address security state   |")
    lines.append("  | [4:2]   | AttrIndx   | Index into MAIR_EL1 (selects memory attributes)     |")
    lines.append("  | [1:0]   | Type       | 00/10=Invalid, 01=Block (L1/L2), 11=Table/Page (L3)  |")
    lines.append("")
    lines.append("  ### Table Descriptor Fields (non-leaf, L0-L2)")
    lines.append("")
    lines.append("  | Bits    | Field        | Description                                        |")
    lines.append("  |---------|--------------|----------------------------------------------------|")
    lines.append("  | [63]    | NSTable      | If 1, subsequent levels are forced Non-Secure       |")
    lines.append("  | [62:61] | APTable[1:0] | Restricts AP at subsequent levels                   |")
    lines.append("  | [60]    | UXNTable     | Forces UXN=1 at subsequent levels                   |")
    lines.append("  | [59]    | PXNTable     | Forces PXN=1 at subsequent levels                   |")
    lines.append("  | [47:12] | Next-level   | Physical address of next-level translation table    |")
    lines.append("  | [1:0]   | Type = 0b11  | Identifies this as a table descriptor               |")
    lines.append("")

    # Output address and PA size
    lines.append("## Output Address (OA) Range and Physical Address Size")
    lines.append("")
    lines.append("  The Output Address in a descriptor provides the physical address of the")
    lines.append("  mapped block/page (for block/page descriptors) or the next-level table")
    lines.append("  (for table descriptors). The number of valid OA bits depends on the")
    lines.append("  physical address size configured in TCR_EL1.IPS:")
    lines.append("")
    lines.append("  | IPS Value | PA Bits | Physical Address Space  |")
    lines.append("  |-----------|---------|-------------------------|")
    for ips in _IPS_ENCODINGS:
        lines.append(f"  | {ips['value']:<9} | {ips['pa_bits']:<7} | {ips['pa_size']:<23} |")
    lines.append("")
    lines.append("  IPS must be set to a value <= the implementation's ID_AA64MMFR0_EL1.PARange.")
    lines.append("  If IPS is set larger than what hardware supports, behavior is CONSTRAINED")
    lines.append("  UNPREDICTABLE.")
    lines.append("")
    if va_bits == 52:
        lines.append("  ### 52-bit OA (FEAT_LPA2)")
        lines.append("  For 52-bit physical addresses, OA bits [51:48] are encoded in")
        lines.append("  repurposed descriptor fields (bits [9:8] and [5:2] are reinterpreted).")
        lines.append("  This requires ARMv8.7-A / FEAT_LPA2 hardware support.")
        lines.append("")

    # TCR_EL1 configuration
    tg = _TCR_GRANULE_ENCODINGS[granule]
    t0sz = 64 - va_bits
    lines.append("## TCR_EL1 Configuration Fields")
    lines.append("")
    lines.append(f"  ### T0SZ / T1SZ = {t0sz}")
    lines.append(f"    Formula: VA size = 2^(64 - TxSZ) = 2^{va_bits}")
    lines.append(f"    T0SZ = {t0sz} -> {va_bits}-bit VA for TTBR0_EL1 (lower/user range)")
    lines.append(f"    T1SZ = {t0sz} -> {va_bits}-bit VA for TTBR1_EL1 (upper/kernel range)")
    lines.append(f"    T0SZ is at TCR_EL1[5:0], T1SZ is at TCR_EL1[21:16].")
    lines.append("")
    lines.append(f"  ### TG0 / TG1 -- Granule Size Selection")
    lines.append(f"    TG0 (TCR_EL1[15:14], controls TTBR0): {tg['TG0']} -> {granule}")
    lines.append(f"    TG1 (TCR_EL1[31:30], controls TTBR1): {tg['TG1']} -> {granule}")
    lines.append(f"    TG0 encodings: 00=4KB, 01=64KB, 10=16KB")
    lines.append(f"    TG1 encodings: 01=16KB, 10=4KB, 11=64KB")
    lines.append(f"    Note: TG0 and TG1 use different encodings for the same granule size.")
    lines.append("")
    lines.append(f"  ### IPS -- Intermediate Physical Address Size")
    lines.append(f"    TCR_EL1[34:32]. Controls the maximum Output Address width.")
    lines.append(f"    Must be <= ID_AA64MMFR0_EL1.PARange (hardware capability).")
    lines.append(f"    See the OA Range table above for the encoding values.")

    return "\n".join(lines)


@mcp.tool()
def explain_page_table_format(granule_size: str, va_bits: int = 48) -> str:
    """Explain the AArch64 page table translation scheme for a given granule and VA size.

    Shows number of translation levels, VA bit layout diagram, index bits and
    entry count per level, block/page/table descriptor types and sizes at each
    level, PTE field layout, Output Address range and its relationship to
    physical address size, and the TCR_EL1 fields (T0SZ/T1SZ, TG0/TG1, IPS)
    that control the configuration.

    Args:
        granule_size: Translation granule size -- "4KB", "16KB", or "64KB".
        va_bits: Virtual address width -- 39, 48, or 52. Default is 48.
    """
    gs = granule_size.strip().upper()
    if gs not in ("4KB", "16KB", "64KB"):
        return "Error: granule_size must be '4KB', '16KB', or '64KB'."

    if va_bits not in (39, 48, 52):
        return "Error: va_bits must be 39, 48, or 52."

    key = (gs, va_bits)
    config = _PAGE_TABLE_CONFIGS.get(key)
    if config is None:
        return (
            f"Error: The combination granule={gs} with va_bits={va_bits} "
            f"is not a standard AArch64 configuration."
        )

    return _format_page_table_config(gs, va_bits, config)


# ---------------------------------------------------------------------------
# Tool: explain_memory_attributes -- ARM memory attributes reference
# ---------------------------------------------------------------------------

_MEMORY_ATTR_TOPICS: dict[str, str] = {
    "overview": (
        "# AArch64 Memory Attributes Overview\n"
        "\n"
        "ARM AArch64 uses a rich set of memory attributes to control caching, ordering,\n"
        "access permissions, and shareability for each memory region. Attributes are\n"
        "specified per-page (or per-block) through the page table entry and the MAIR_EL1\n"
        "register.\n"
        "\n"
        "## Key Concepts\n"
        "\n"
        "1. **MAIR_EL1 Register**\n"
        "   - Memory Attribute Indirection Register at EL1.\n"
        "   - Contains 8 attribute slots (Attr0-Attr7), each 8 bits wide (total: 64 bits).\n"
        "   - Page table entries use a 3-bit AttrIndx field (PTE bits [4:2]) to select one\n"
        "     of these 8 slots.\n"
        "   - This indirection lets the OS define up to 8 distinct memory types and\n"
        "     reference them compactly from every page table entry.\n"
        "\n"
        "2. **Normal vs Device Memory**\n"
        "   - Normal memory: cacheable, allows speculative access and reordering by the PE.\n"
        "   - Device memory: non-cacheable, accesses may have side-effects, ordering\n"
        "     constraints limit speculative and out-of-order behavior.\n"
        "\n"
        "3. **Cacheability** (for Normal memory)\n"
        "   - Write-Back (WB): best performance; cache may hold dirty data.\n"
        "   - Write-Through (WT): writes update both cache and memory.\n"
        "   - Non-cacheable (NC): bypasses cache entirely.\n"
        "   - Each attribute is configured independently for Inner and Outer cache domains.\n"
        "\n"
        "4. **Device Memory Types**\n"
        "   - Device-nGnRnE, Device-nGnRE, Device-nGRE, Device-GRE\n"
        "   - Flags control: Gathering (G), Reordering (R), Early write acknowledgement (E).\n"
        "   - nGnRnE is the most restrictive (safest for MMIO); GRE is the most relaxed.\n"
        "\n"
        "5. **Shareability**\n"
        "   - Non-shareable, Inner Shareable, Outer Shareable, Full System.\n"
        "   - Controls the cache coherency domain -- which observers maintain a consistent\n"
        "     view of memory.\n"
        "   - Set via SH bits [9:8] in the page table entry.\n"
        "\n"
        "6. **Access Permissions**\n"
        "   - AP[2:1] bits control read/write at EL0 and EL1+.\n"
        "   - PXN (Privileged Execute-Never) and UXN (Unprivileged Execute-Never) control\n"
        "     instruction fetch permissions.\n"
        "   - DBM (Dirty Bit Modifier) enables hardware dirty page tracking.\n"
        "   - AF (Access Flag) enables hardware or software tracking of page usage.\n"
        "\n"
        "7. **Stage 1 + Stage 2 Combination**\n"
        "   - When virtualization is active, Stage 1 (guest OS) and Stage 2 (hypervisor)\n"
        "     attributes are combined. The most restrictive attribute wins for memory type,\n"
        "     and permissions are intersected.\n"
        "\n"
        "Use the 'topic' parameter to drill into specific areas:\n"
        '  "cacheability", "shareability", "access_permissions", "mair"\n'
        "Or omit it for this overview."
    ),
    "mair": (
        "# MAIR_EL1 -- Memory Attribute Indirection Register\n"
        "\n"
        "## Register Layout (64 bits)\n"
        "\n"
        "  | Bits    | Field | Description                  |\n"
        "  |---------|-------|------------------------------|\n"
        "  | [63:56] | Attr7 | Memory attribute encoding 7  |\n"
        "  | [55:48] | Attr6 | Memory attribute encoding 6  |\n"
        "  | [47:40] | Attr5 | Memory attribute encoding 5  |\n"
        "  | [39:32] | Attr4 | Memory attribute encoding 4  |\n"
        "  | [31:24] | Attr3 | Memory attribute encoding 3  |\n"
        "  | [23:16] | Attr2 | Memory attribute encoding 2  |\n"
        "  | [15:8]  | Attr1 | Memory attribute encoding 1  |\n"
        "  | [7:0]   | Attr0 | Memory attribute encoding 0  |\n"
        "\n"
        "## How AttrIndx Works\n"
        "\n"
        "  1. Each page table entry has a 3-bit AttrIndx field at PTE bits [4:2].\n"
        "  2. AttrIndx selects one of Attr0-Attr7 from MAIR_EL1.\n"
        "  3. The selected 8-bit Attr value encodes the memory type and cacheability.\n"
        "  4. The SH (Shareability) bits [9:8] in the PTE complement the MAIR attribute\n"
        "     to fully define the memory region behavior.\n"
        "\n"
        "## Attr Field Encoding (8-bit value)\n"
        "\n"
        "Each Attr[n] is split into two 4-bit halves:\n"
        "  Attr[7:4] = Outer attribute\n"
        "  Attr[3:0] = Inner attribute\n"
        "\n"
        "### Normal Memory (Attr[7:4] != 0b0000)\n"
        "\n"
        "  Each 4-bit nibble (inner or outer) is encoded as:\n"
        "\n"
        "  | Nibble value | Cache policy                          |\n"
        "  |--------------|---------------------------------------|\n"
        "  | 0b0100       | Non-cacheable                         |\n"
        "  | 0b0110       | Write-Through, Read-Allocate          |\n"
        "  | 0b0111       | Write-Through, Read+Write Allocate    |\n"
        "  | 0b1000       | Write-Back, no Allocate               |\n"
        "  | 0b1010       | Write-Back, Read-Allocate             |\n"
        "  | 0b1011       | Write-Back, Read+Write Allocate       |\n"
        "  | 0b1100       | Write-Back, no Allocate (alternate)   |\n"
        "  | 0b1110       | Write-Back, Write-Allocate only       |\n"
        "  | 0b1111       | Write-Back, Read+Write Allocate       |\n"
        "\n"
        "  Within the nibble (when bits [3:2] indicate a cache policy):\n"
        "    Bit 1: Write-Allocate (WA)\n"
        "    Bit 0: Read-Allocate (RA)\n"
        "\n"
        "### Device Memory (Attr[7:4] == 0b0000)\n"
        "\n"
        "  When the outer nibble is 0b0000, the full 8-bit value selects a Device type:\n"
        "\n"
        "  | Attr[7:0] | Memory Type    | Properties                                |\n"
        "  |-----------|----------------|-------------------------------------------|\n"
        "  | 0x00      | Device-nGnRnE  | No Gathering, no Reordering, no Early ack |\n"
        "  | 0x04      | Device-nGnRE   | No Gathering, no Reordering, Early ack    |\n"
        "  | 0x08      | Device-nGRE    | No Gathering, Reordering, Early ack       |\n"
        "  | 0x0C      | Device-GRE     | Gathering, Reordering, Early ack          |\n"
        "\n"
        "## Common MAIR_EL1 Configurations\n"
        "\n"
        "  A typical Linux kernel (arm64) configuration:\n"
        "\n"
        "  MAIR_EL1 = 0x000000000044FF00  (example, varies by kernel version)\n"
        "\n"
        "  | Index | Attr value | Memory type                            |\n"
        "  |-------|------------|----------------------------------------|\n"
        "  | 0     | 0x00       | Device-nGnRnE (strongly-ordered MMIO)  |\n"
        "  | 1     | 0xFF       | Normal, Write-Back RW-Allocate (I+O)   |\n"
        "  | 2     | 0x44       | Normal, Non-cacheable (I+O)            |\n"
        "  | 3     | 0x00       | (unused or Device-nGnRnE)              |\n"
        "  | 4     | 0x00       | (unused)                               |\n"
        "\n"
        "  More complete example (some kernels):\n"
        "    Attr0 = 0x00 -> Device-nGnRnE\n"
        "    Attr1 = 0x04 -> Device-nGnRE (MMIO with early write ack)\n"
        "    Attr2 = 0x44 -> Normal Non-cacheable (DMA buffers)\n"
        "    Attr3 = 0xBB -> Normal Write-Back Read-Allocate\n"
        "    Attr4 = 0xFF -> Normal Write-Back RW-Allocate (default for RAM)\n"
        "\n"
        "## Relationship to Page Table Entries\n"
        "\n"
        "  PTE bits [4:2] = AttrIndx -> selects Attr<n> from MAIR_EL1\n"
        "    -> determines memory type (Normal/Device) and cache policy\n"
        "  PTE bits [9:8] = SH -> determines shareability domain\n"
        "  Together, AttrIndx + SH fully characterize memory behavior for the page."
    ),
    "cacheability": (
        "# ARM Memory Cacheability\n"
        "\n"
        "## Normal Memory Cache Policies\n"
        "\n"
        "Normal memory supports three caching policies, configured independently for\n"
        "the Inner cache domain and the Outer cache domain.\n"
        "\n"
        "### Write-Back (WB)\n"
        "  - Writes update only the cache; dirty lines are written to memory on eviction.\n"
        "  - Best performance for most workloads.\n"
        "  - Allocation policies (set independently per domain):\n"
        "    - Read-Allocate (RA): allocate a cache line on a read miss.\n"
        "    - Write-Allocate (WA): allocate a cache line on a write miss.\n"
        "    - Read+Write Allocate: allocate on both read and write misses (most common).\n"
        "    - No Allocate: do not allocate on any miss (data may still hit if cached).\n"
        "  - MAIR nibble encodings:\n"
        "    0b1011 = WB, Read+Write Allocate (e.g., 0xFF for both inner and outer)\n"
        "    0b1010 = WB, Read-Allocate only\n"
        "    0b1110 = WB, Write-Allocate only\n"
        "    0b1000 = WB, No Allocate\n"
        "\n"
        "### Write-Through (WT)\n"
        "  - Writes update both cache and memory simultaneously.\n"
        "  - Simpler coherency (no dirty lines) but lower write performance than WB.\n"
        "  - Useful for regions shared with non-coherent agents using software management.\n"
        "  - MAIR nibble encodings:\n"
        "    0b0110 = WT, Read-Allocate\n"
        "    0b0111 = WT, Read+Write Allocate\n"
        "    0b0100 = WT, No Allocate (effectively same as Non-cacheable)\n"
        "\n"
        "### Non-cacheable (NC)\n"
        "  - All accesses go directly to memory with no cache involvement.\n"
        "  - Used for DMA buffers, shared memory with non-coherent devices.\n"
        "  - MAIR encoding: 0b0100 for both inner and outer = Attr value 0x44.\n"
        "\n"
        "## Inner vs Outer Cache Domains\n"
        "\n"
        "  - **Inner cache domain**: typically L1 and L2 caches (PE-local caches).\n"
        "  - **Outer cache domain**: typically L3 or system-level caches.\n"
        "  - The boundary between inner and outer is IMPLEMENTATION DEFINED.\n"
        "    Check the SoC Technical Reference Manual for the exact split.\n"
        "  - Each domain can have an independent cache policy.\n"
        "  - Example: Inner=WB + Outer=WT for a region where L3 is shared with a\n"
        "    non-coherent DMA engine.\n"
        "\n"
        "## Device Memory Types (Always Non-cacheable)\n"
        "\n"
        "Device memory is never cached. It is used for memory-mapped I/O (MMIO)\n"
        "regions where accesses have side-effects.\n"
        "\n"
        "  | Type           | Gathering | Reordering | Early Ack | MAIR  | Use Case                    |\n"
        "  |----------------|-----------|------------|-----------|-------|-----------------------------|\n"
        "  | Device-nGnRnE  | No        | No         | No        | 0x00  | Strictest: status registers |\n"
        "  | Device-nGnRE   | No        | No         | Yes       | 0x04  | General MMIO registers      |\n"
        "  | Device-nGRE    | No        | Yes        | Yes       | 0x08  | PCIe config, DMA descs      |\n"
        "  | Device-GRE     | Yes       | Yes        | Yes       | 0x0C  | Framebuffers, bulk MMIO     |\n"
        "\n"
        "### What G, R, E Mean in Detail\n"
        "\n"
        "  **Gathering (G / nG)**\n"
        "    G (Gathering permitted):\n"
        "      Multiple accesses to the same or adjacent locations can be merged\n"
        "      into a single wider bus transaction. Example: two adjacent byte\n"
        "      writes can be gathered into a single halfword write.\n"
        "    nG (No Gathering):\n"
        "      Each individual access must appear as a separate transaction on the\n"
        "      bus. Required when each access triggers a distinct hardware side-\n"
        "      effect (e.g., pushing bytes to a FIFO register).\n"
        "\n"
        "  **Reordering (R / nR)**\n"
        "    R (Reordering permitted):\n"
        "      Accesses to the same device region can be reordered with respect to\n"
        "      each other. The interconnect or PE may deliver writes or reads in a\n"
        "      different order than program order.\n"
        "    nR (No Reordering):\n"
        "      Accesses to the device are delivered strictly in program order.\n"
        "      Required for devices where register access ordering matters\n"
        "      (e.g., write a command register, then read a status register).\n"
        "\n"
        "  **Early Write Acknowledgement (E / nE)**\n"
        "    E (Early acknowledgement):\n"
        "      A write can be acknowledged by an intermediate buffer (e.g., a\n"
        "      write buffer in the interconnect) before reaching the endpoint\n"
        "      device. This improves write performance.\n"
        "    nE (No Early acknowledgement):\n"
        "      A write is only acknowledged after it has reached the final\n"
        "      target device. Required when software must know the write has\n"
        "      actually taken effect before continuing (e.g., writing an\n"
        "      interrupt-clear register, then reading the status to confirm).\n"
        "\n"
        "### Choosing the Right Device Type\n"
        "\n"
        "  - MMIO registers (general purpose):         Device-nGnRnE (safest default)\n"
        "  - MMIO where early ack is acceptable:       Device-nGnRE\n"
        "  - PCIe configuration / DMA descriptors:     Device-nGRE\n"
        "  - Framebuffers (non-cacheable, bulk write):  Device-GRE\n"
        "\n"
        "## Cache Maintenance Operations\n"
        "\n"
        "  Key cache maintenance instructions:\n"
        "\n"
        "  | Instruction | Full Name                                  | Purpose                                  |\n"
        "  |-------------|--------------------------------------------|------------------------------------------|\n"
        "  | DC CIVAC    | Clean+Invalidate by VA to Point of Coherency| Flush dirty data to memory, remove line  |\n"
        "  | DC CVAC     | Clean by VA to Point of Coherency          | Write dirty data to memory, keep line    |\n"
        "  | DC IVAC     | Invalidate by VA to Point of Coherency     | Discard cache line (data lost if dirty)  |\n"
        "  | DC CVAU     | Clean by VA to Point of Unification        | For I-cache / D-cache coherency          |\n"
        "  | IC IALLU    | Invalidate All I-caches to PoU             | Flush instruction cache                  |\n"
        "  | DSB         | Data Synchronization Barrier               | Ensure all prior memory ops complete     |\n"
        "  | ISB         | Instruction Synchronization Barrier        | Flush pipeline, refetch instructions     |\n"
        "\n"
        "  When changing memory attributes of a live mapping:\n"
        "    1. Change the PTE attributes.\n"
        "    2. DSB ISH (ensure PTE store is visible).\n"
        "    3. TLBI for the affected VA(s) (invalidate stale TLB entries).\n"
        "    4. DSB ISH (ensure TLBI completes).\n"
        "    5. DC CIVAC on affected cache lines (clean+invalidate cached data).\n"
        "    6. DSB ISH.\n"
        "    7. ISB (if instruction fetch may be affected)."
    ),
    "shareability": (
        "# ARM Shareability Domains\n"
        "\n"
        "## Overview\n"
        "\n"
        "Shareability determines which observers (PEs, DMA controllers, GPUs, etc.)\n"
        "maintain hardware cache coherency for a given memory region. It is configured\n"
        "via the SH bits [9:8] in each page table entry.\n"
        "\n"
        "## SH Field Encodings\n"
        "\n"
        "  | SH[1:0] | Domain            | Description                                   |\n"
        "  |---------|-------------------|-----------------------------------------------|\n"
        "  | 0b00    | Non-shareable     | Only the local PE sees coherent data. Other   |\n"
        "  |         |                   | PEs or bus masters must use explicit cache     |\n"
        "  |         |                   | maintenance to see updates.                   |\n"
        "  | 0b01    | (Reserved)        | Unpredictable -- do not use.                  |\n"
        "  | 0b10    | Outer Shareable   | All agents in the outer shareability domain   |\n"
        "  |         |                   | maintain coherency (all PEs + coherent DMA).  |\n"
        "  | 0b11    | Inner Shareable   | All agents in the inner shareability domain   |\n"
        "  |         |                   | maintain coherency (typically all PEs).       |\n"
        "\n"
        "## Inner Shareable vs Outer Shareable\n"
        "\n"
        "  **Inner Shareable (ISH)**\n"
        "    - Typically covers all PEs that share the same inner cache domain.\n"
        "    - On most SoCs with a single cluster, ISH = all CPUs.\n"
        "    - On multi-cluster SoCs (e.g., big.LITTLE), ISH may cover one\n"
        "      cluster or all clusters, depending on the interconnect.\n"
        "    - This is the standard setting for SMP kernel data and page tables.\n"
        "\n"
        "  **Outer Shareable (OSH)**\n"
        "    - Typically covers all PEs plus coherent bus masters (DMA controllers,\n"
        "      GPUs, NPUs with coherent cache ports).\n"
        "    - OSH is a superset of ISH.\n"
        "    - Use for DMA buffers when the DMA engine is I/O-coherent.\n"
        "\n"
        "  The exact boundary is IMPLEMENTATION DEFINED and depends on the SoC\n"
        "  interconnect (CoreLink CCI, CCN, CMN, DSU cluster, etc.).\n"
        "\n"
        "## Full System Shareability (ARMv8.4+)\n"
        "\n"
        "  ARMv8.4-A introduces Full System shareability:\n"
        "    - All agents in the entire system see a coherent memory view.\n"
        "    - Relevant for systems with multiple coherent interconnect domains.\n"
        "    - FEAT_S2FWB and related features interact with Full System shareability.\n"
        "\n"
        "## Practical Guidelines\n"
        "\n"
        "  | Use Case                              | Recommended SH | Why                            |\n"
        "  |---------------------------------------|----------------|--------------------------------|\n"
        "  | Thread stacks, per-CPU data           | Inner Shareable| Coherent across all CPUs       |\n"
        "  | Shared kernel data structures         | Inner Shareable| SMP coherency                  |\n"
        "  | Page tables (TTBR0/TTBR1)            | Inner Shareable| HW table walker must see them  |\n"
        "  | DMA buffers (I/O-coherent DMA)        | Outer Shareable| DMA agent in outer domain      |\n"
        "  | DMA buffers (non-coherent DMA)        | Non-shareable  | + explicit DC CIVAC/IVAC ops   |\n"
        "  | Device MMIO                           | (ignored)      | Device memory ignores SH bits  |\n"
        "  | IPI / cross-CPU mailbox memory        | Inner Shareable| Cross-CPU signaling            |\n"
        "\n"
        "## Interaction with Cacheability\n"
        "\n"
        "  Shareability works together with the cacheability attributes from MAIR:\n"
        "\n"
        "  - WB + Inner Shareable: the standard SMP configuration. Hardware maintains\n"
        "    coherency automatically between all PEs in the inner domain.\n"
        "  - NC + Inner Shareable: accesses bypass caches but the interconnect still\n"
        "    ensures proper ordering with cacheable accesses (important for barriers).\n"
        "  - NC + Non-shareable: the cheapest coherency model. No hardware help;\n"
        "    software must manually flush/invalidate caches for sharing.\n"
        "  - WB + Non-shareable: each PE caches independently. Dangerous for shared\n"
        "    data -- only use for truly PE-private data.\n"
        "\n"
        "## Shareability for Device Memory\n"
        "\n"
        "  For Device memory types (Device-nGnRnE, Device-nGnRE, etc.), the SH bits\n"
        "  in the page table entry are architecturally ignored. Device memory is\n"
        "  implicitly treated as Outer Shareable regardless of the SH setting.\n"
        "\n"
        "## Barriers and Shareability Domains\n"
        "\n"
        "  Barrier instructions can be scoped to a shareability domain:\n"
        "\n"
        "  | Barrier     | Scope           | Effect                                      |\n"
        "  |-------------|-----------------|---------------------------------------------|\n"
        "  | DSB ISH     | Inner Shareable | Sync all memory ops in inner domain          |\n"
        "  | DSB OSH     | Outer Shareable | Sync all memory ops in outer domain          |\n"
        "  | DSB SY      | Full System     | Sync all memory ops system-wide              |\n"
        "  | DMB ISH     | Inner Shareable | Order memory ops (no completion guarantee)   |\n"
        "  | DMB OSH     | Outer Shareable | Order memory ops in outer domain             |\n"
        "  | DMB SY      | Full System     | Order all memory ops system-wide             |\n"
        "  | TLBI ...IS  | Inner Shareable | Broadcast TLB invalidate to inner domain     |\n"
        "  | TLBI ...OS  | Outer Shareable | Broadcast TLB invalidate to outer (v8.4+)    |\n"
        "\n"
        "  Common pattern (e.g., page table update):\n"
        "    STR  new_pte, [pte_addr]     ; write new PTE\n"
        "    DSB  ISH                     ; ensure PTE write is visible\n"
        "    TLBI VAE1IS, Xt             ; invalidate old TLB entry, broadcast ISH\n"
        "    DSB  ISH                     ; wait for TLBI to complete\n"
        "    ISB                          ; synchronize context"
    ),
    "access_permissions": (
        "# ARM Access Permissions\n"
        "\n"
        "## Page Table Entry Permission Bits\n"
        "\n"
        "Access control in AArch64 page table entries restricts read, write, and\n"
        "execute access at different exception levels (EL0 vs EL1+).\n"
        "\n"
        "### AP[2:1] -- Access Permission Bits (PTE [7:6])\n"
        "\n"
        "  | AP[2:1] | EL1+ (kernel) | EL0 (user)  | Description                     |\n"
        "  |---------|---------------|-------------|---------------------------------|\n"
        "  | 0b00    | Read/Write    | No access   | Kernel-only read/write          |\n"
        "  | 0b01    | Read/Write    | Read/Write  | Full access at all ELs          |\n"
        "  | 0b10    | Read-only     | No access   | Kernel read-only                |\n"
        "  | 0b11    | Read-only     | Read-only   | Read-only at all ELs            |\n"
        "\n"
        "  AP[2] is PTE bit [7]; AP[1] is PTE bit [6].\n"
        "  Note: There is no write-only encoding. Write access always implies read.\n"
        "\n"
        "### PAN -- Privileged Access Never (ARMv8.1+, FEAT_PAN)\n"
        "\n"
        "  When PSTATE.PAN = 1, EL1 (the kernel) cannot use normal loads/stores to\n"
        "  access any memory that is accessible at EL0. This prevents the kernel\n"
        "  from accidentally reading or writing user-space memory.\n"
        "\n"
        "  - Kernel must use LDTR/STTR (unprivileged load/store) for intentional\n"
        "    user-space memory access.\n"
        "  - PAN is enabled at boot by setting SCTLR_EL1.SPAN = 0 (PAN auto-set\n"
        "    on exception entry to EL1).\n"
        "  - Very important for security: prevents ret2user and similar attacks.\n"
        "\n"
        "### Execute Permission Bits\n"
        "\n"
        "  | PTE Bit | Name | Full Name                     | Effect when set to 1        |\n"
        "  |---------|------|-------------------------------|-----------------------------|"
        "\n"
        "  | [54]    | UXN  | Unprivileged Execute-Never    | EL0 cannot execute code     |\n"
        "  |         |      | (also called XN)              | from this page              |\n"
        "  | [53]    | PXN  | Privileged Execute-Never      | EL1+ cannot execute code    |\n"
        "  |         |      |                               | from this page              |\n"
        "\n"
        "  Common permission patterns:\n"
        "\n"
        "  | Page type       | AP[2:1] | UXN | PXN | Effect                          |\n"
        "  |-----------------|---------|-----|-----|---------------------------------|\n"
        "  | User code       | 0b10    | 0   | 1   | EL0 exec+read, kernel read-only |\n"
        "  | Kernel code     | 0b10    | 1   | 0   | Kernel exec+read, no user access|\n"
        "  | User data (RW)  | 0b01    | 1   | 1   | Both can R/W, nobody can exec   |\n"
        "  | Kernel data (RW)| 0b00    | 1   | 1   | Kernel R/W, no user, no exec    |\n"
        "  | Read-only data  | 0b11    | 1   | 1   | All read-only, no exec          |\n"
        "\n"
        "  W^X (Write XOR Execute) policy: if a page is writable, both UXN and PXN\n"
        "  must be set to 1. This is enforced by modern kernels for security.\n"
        "\n"
        "### Dirty Bit Modifier (DBM) -- PTE bit [51]\n"
        "\n"
        "  Part of FEAT_HAFDBS (Hardware Access Flag and Dirty Bit State):\n"
        "\n"
        "  How hardware dirty tracking works:\n"
        "    1. OS marks a clean page as: AP = read-only (AP[2]=1) + DBM = 1.\n"
        "    2. On first write to the page, the PE (in hardware) atomically clears\n"
        "       AP[2] to 0, making the page writable. This marks the page dirty.\n"
        "    3. The OS can later scan pages where AP[2] = 0 to find dirty pages\n"
        "       that need to be written back to disk / swap.\n"
        "    4. To re-clean a page: set AP[2] = 1 again (and TLBI + DSB).\n"
        "\n"
        "  Requires FEAT_HAFDBS support: check ID_AA64MMFR1_EL1.HAFDBS field.\n"
        "  Without FEAT_HAFDBS, the OS must handle dirty tracking in software\n"
        "  via permission faults.\n"
        "\n"
        "### Access Flag (AF) -- PTE bit [10]\n"
        "\n"
        "  - AF = 0: page has not been accessed. A first access causes:\n"
        "    - With FEAT_HAFDBS: hardware atomically sets AF = 1 (no fault).\n"
        "    - Without FEAT_HAFDBS: Access Flag fault; OS must set AF = 1 in\n"
        "      the fault handler and return.\n"
        "  - AF = 1: page has been accessed (no fault on subsequent accesses).\n"
        "  - Used by the OS for page reclamation (LRU approximation): periodically\n"
        "    clear AF bits and see which pages get accessed.\n"
        "\n"
        "### Not-Global (nG) -- PTE bit [11]\n"
        "\n"
        "  - nG = 0 (Global): the TLB entry is not tagged with an ASID. It matches\n"
        "    regardless of the current ASID. Used for kernel-space mappings.\n"
        "  - nG = 1 (Not Global): the TLB entry is tagged with the current ASID\n"
        "    (Address Space Identifier). Used for user-space mappings so that\n"
        "    different processes have isolated address spaces without requiring\n"
        "    a full TLB flush on context switch.\n"
        "  - ASID is specified in TTBR0_EL1 (or TTBR1_EL1 if TTBCR.A1 selects it).\n"
        "\n"
        "## Stage 1 vs Stage 2 Translation Attributes\n"
        "\n"
        "When a hypervisor is active, every memory access goes through two\n"
        "translation stages. Each stage independently specifies attributes.\n"
        "\n"
        "### Stage 1 (controlled by guest OS at EL1)\n"
        "  - Uses MAIR_EL1 and AttrIndx in PTE to select memory type.\n"
        "  - Uses AP[2:1], UXN, PXN for access control.\n"
        "  - Uses SH[1:0] for shareability.\n"
        "  - Translation base: TTBR0_EL1 (user) / TTBR1_EL1 (kernel).\n"
        "\n"
        "### Stage 2 (controlled by hypervisor at EL2)\n"
        "  - Does NOT use MAIR. Memory type is encoded directly in the descriptor.\n"
        "  - Stage 2 memory type encoding (descriptor bits [5:2] = MemAttr[3:0]):\n"
        "\n"
        "    | S2 MemAttr[3:0] | Memory Type             |\n"
        "    |-----------------|-------------------------|\n"
        "    | 0b0000          | Device-nGnRnE           |\n"
        "    | 0b0001          | Device-nGnRE            |\n"
        "    | 0b0010          | Device-nGRE             |\n"
        "    | 0b0011          | Device-GRE              |\n"
        "    | 0b0101          | Normal Non-cacheable    |\n"
        "    | 0b1010          | Normal Write-Through    |\n"
        "    | 0b1111          | Normal Write-Back       |\n"
        "\n"
        "  - Stage 2 access permissions (S2AP[1:0]):\n"
        "\n"
        "    | S2AP[1:0] | Access       |\n"
        "    |-----------|--------------|\n"
        "    | 0b00      | No access    |\n"
        "    | 0b01      | Read-only    |\n"
        "    | 0b10      | Write-only   |\n"
        "    | 0b11      | Read/Write   |\n"
        "\n"
        "  - Execute control: XN[1:0] bits control execute permission for EL1 and EL0\n"
        "    independently in Stage 2.\n"
        "  - Translation base: VTTBR_EL2, managed by the hypervisor.\n"
        "\n"
        "### How Stage 1 + Stage 2 Attributes Combine\n"
        "\n"
        "  When both stages are active, the final effective attributes are:\n"
        "\n"
        "  1. **Memory Type**: the more restrictive type wins.\n"
        "     - Device overrides Normal.\n"
        "     - Non-cacheable overrides Write-Through overrides Write-Back.\n"
        "     - If S1 = WB and S2 = WT, effective = WT.\n"
        "     - If S1 = Normal and S2 = Device, effective = Device.\n"
        "\n"
        "  2. **Shareability**: the more shareable domain wins.\n"
        "     - Outer Shareable > Inner Shareable > Non-shareable.\n"
        "     - If S1 = ISH and S2 = OSH, effective = OSH.\n"
        "\n"
        "  3. **Access Permissions**: intersection (logical AND) of allowed accesses.\n"
        "     - S1 = RW and S2 = RO -> effective = RO.\n"
        "     - S1 = RO and S2 = RW -> effective = RO.\n"
        "     - S1 = RW and S2 = No access -> GPF (Guest Permission Fault).\n"
        "\n"
        "  4. **Execute Permissions**: both stages must permit execution.\n"
        "     - If either stage marks XN, execution is denied.\n"
        "\n"
        "  With FEAT_S2FWB (Stage 2 Forced Write-Back, ARMv8.4+):\n"
        "    - The hypervisor can override Stage 1 memory type entirely from Stage 2.\n"
        "    - Stage 2 MemAttr encoding is reinterpreted to directly specify the\n"
        "      final memory type, ignoring Stage 1's MAIR-derived attributes.\n"
        "    - Simplifies hypervisor: S2 can force WB cacheable regardless of what\n"
        "      the guest configured in Stage 1.\n"
        "    - Enabled by HCR_EL2.FWB = 1."
    ),
}


def _format_memory_attributes(topic: str | None) -> str:
    """Return the formatted memory attributes documentation for the given topic."""
    if topic is None:
        return _MEMORY_ATTR_TOPICS["overview"]

    topic_lower = topic.strip().lower()
    if topic_lower in _MEMORY_ATTR_TOPICS:
        return _MEMORY_ATTR_TOPICS[topic_lower]

    valid = ", ".join(f'"{k}"' for k in _MEMORY_ATTR_TOPICS if k != "overview")
    return (
        f"Error: Unknown topic '{topic}'.\n\n"
        f"Valid topics: {valid}\n"
        f"Or omit the topic parameter for an overview of all memory attribute concepts."
    )


@mcp.tool()
def explain_memory_attributes(topic: str | None = None) -> str:
    """Explain AArch64 memory attributes: cacheability, shareability, and permissions.

    Returns detailed reference material on how ARM memory attributes work,
    covering the MAIR_EL1 register and AttrIndx, cacheability policies
    (Write-Back, Write-Through, Non-cacheable), device memory types
    (Device-nGnRnE/nGnRE/nGRE/GRE and what G, R, E mean), shareability
    domains (Non-shareable, Inner Shareable, Outer Shareable), access
    permissions (AP bits, PXN, UXN, DBM, AF, nG), and Stage 1 vs Stage 2
    attribute combination rules.

    Args:
        topic: Optional focus area. One of:
               "cacheability"       -- cache policies, device memory types, cache maintenance ops
               "shareability"       -- Non-shareable, Inner/Outer Shareable, barriers
               "access_permissions" -- AP bits, PXN, UXN, DBM, AF, nG, Stage 1 vs Stage 2
               "mair"              -- MAIR_EL1 register encoding and common configurations
               None (default)      -- high-level overview of all memory attribute concepts
    """
    return _format_memory_attributes(topic)


# ---------------------------------------------------------------------------
# Tool 12: explain_extension — ARM architecture extension reference
# ---------------------------------------------------------------------------

_EXTENSIONS: dict[str, dict] = {
    "SVE": {
        "full_name": "Scalable Vector Extension",
        "acronym": "SVE",
        "introduced": "ARMv8.2-A (optional)",
        "purpose": (
            "Provides a vector-length-agnostic SIMD programming model for HPC and "
            "scientific workloads. Unlike fixed-width NEON (128-bit), SVE supports "
            "vector lengths from 128 to 2048 bits in 128-bit increments, chosen by "
            "the hardware implementation. Code compiled for SVE runs on any SVE-capable "
            "core regardless of the physical vector length, enabling forward "
            "compatibility without recompilation."
        ),
        "problem_solved": (
            "NEON's fixed 128-bit vectors limit throughput on workloads that can "
            "exploit wider data parallelism. Porting code to new SIMD widths "
            "traditionally required rewriting or recompiling. SVE eliminates this "
            "by making vector length a runtime property."
        ),
        "key_registers": [
            "Z0-Z31: Scalable vector registers (VL bits wide, 128-2048)",
            "P0-P15: Predicate registers (VL/8 bits) for per-lane masking",
            "FFR: First Fault Register (for speculative memory access)",
            "ZCR_EL1/2/3: SVE Control Register (controls effective vector length)",
        ],
        "key_instructions": [
            "LD1B/LD1H/LD1W/LD1D: Contiguous predicated loads",
            "ST1B/ST1H/ST1W/ST1D: Contiguous predicated stores",
            "WHILELT/WHILELE: Generate predicate for loop tails",
            "FADD/FMUL/FMLA (Z variant): Scalable FP arithmetic",
            "GATHER/SCATTER loads/stores for indirect access",
            "COMPACT/SPLICE: Predicate-driven element packing",
        ],
        "detection": {
            "register": "ID_AA64PFR0_EL1",
            "field": "SVE (bits [35:32])",
            "values": "0b0001 = SVE implemented",
            "linux": "HWCAP_SVE in getauxval(AT_HWCAP)",
        },
        "use_cases": [
            "HPC: Dense linear algebra, FFT, weather/climate simulation",
            "ML inference: Vectorized activation functions, quantized dot products",
            "Bioinformatics: Sequence alignment, genome assembly",
            "App devs benefit via autovectorizing compilers (GCC -msve-vector-bits=scalable)",
        ],
        "audience": ["HPC developers", "compiler engineers", "ML/AI engineers"],
    },
    "SVE2": {
        "full_name": "Scalable Vector Extension version 2",
        "acronym": "SVE2",
        "introduced": "ARMv9.0-A (mandatory in ARMv9)",
        "purpose": (
            "Extends SVE with instructions that cover the full NEON/AdvSIMD "
            "functionality in a scalable form. SVE2 makes SVE a complete "
            "replacement for NEON, adding fixed-point arithmetic, polynomial "
            "operations, cryptographic primitives, and complex-number instructions "
            "that SVE v1 lacked."
        ),
        "problem_solved": (
            "SVE v1 was designed primarily for HPC floating-point. Many NEON "
            "use cases (media codecs, crypto, DSP) had no SVE equivalent. SVE2 "
            "fills these gaps so ARMv9 cores can retire NEON reliance entirely."
        ),
        "key_registers": [
            "Same as SVE: Z0-Z31, P0-P15, FFR, ZCR_ELx",
        ],
        "key_instructions": [
            "HISTCNT/HISTSEG: Histogram operations",
            "SM4E/SM4EKEY: SM4 cryptographic cipher",
            "RAX1: SHA-3 rotate-and-XOR",
            "SQDMULH/SQRDMULH: Saturating fixed-point multiply-high",
            "CADD/CMLA: Complex number addition/multiply-accumulate",
            "NBSL/BSL1N/BSL2N: Extended bitwise select operations",
            "MATCH/NMATCH: Character matching for string processing",
            "BDEP/BEXT/BGRP: Bit permutation (deposit, extract, group)",
        ],
        "detection": {
            "register": "ID_AA64ZFR0_EL1",
            "field": "SVEver (bits [3:0])",
            "values": "0b0001 = SVE2 implemented",
            "linux": "HWCAP2_SVE2 in getauxval(AT_HWCAP2)",
        },
        "use_cases": [
            "Media codecs: VP9/AV1 decode, HEVC encode with scalable vectors",
            "Cryptography: AES-GCM, SM4, SHA-3 acceleration",
            "DSP: Audio resampling, filtering with saturating arithmetic",
            "String/text processing: MATCH for search, HISTCNT for frequency analysis",
        ],
        "audience": ["app developers", "media/codec engineers", "security engineers"],
    },
    "SME": {
        "full_name": "Scalable Matrix Extension",
        "acronym": "SME",
        "introduced": "ARMv9.2-A (optional)",
        "purpose": (
            "Adds architectural support for matrix operations using a new "
            "two-dimensional tile storage (ZA array). SME provides outer-product "
            "and accumulate instructions that directly implement matrix multiply "
            "kernels, with streaming SVE mode for efficient data movement "
            "between vectors and tiles."
        ),
        "problem_solved": (
            "Matrix multiplication is fundamental to ML training/inference and "
            "scientific computing. Before SME, matrix ops were built from "
            "1D vector instructions, requiring complex tiling and register "
            "management. SME provides native 2D tile operations."
        ),
        "key_registers": [
            "ZA: Scalable 2D tile array (SVL x SVL bits), partitioned into ZA0-ZAn tiles",
            "ZA tile slices: Horizontal/vertical slices addressable as vectors",
            "SMCR_EL1/2/3: Streaming Mode Control Register (sets SVL)",
            "SVCR: Streaming Vector Control Register (PSTATE.SM and PSTATE.ZA bits)",
        ],
        "key_instructions": [
            "SMSTART/SMSTOP: Enter/exit streaming SVE mode or enable ZA",
            "FMOPA/FMOPS: FP outer product and accumulate/subtract into ZA tiles",
            "SMOPA/SMOPS: Signed integer outer product and accumulate",
            "UMOPA/UMOPS: Unsigned integer outer product and accumulate",
            "MOVA: Move vector to/from ZA tile slice",
            "ADDHA/ADDVA: Add vector to horizontal/vertical ZA slices",
            "ZERO {ZA}: Zero the entire ZA array",
            "LDR/STR (ZA): Save/restore ZA state for context switching",
        ],
        "detection": {
            "register": "ID_AA64PFR1_EL1",
            "field": "SME (bits [27:24])",
            "values": "0b0001 = SME implemented; 0b0010 = SME2",
            "linux": "HWCAP2_SME in getauxval(AT_HWCAP2)",
        },
        "use_cases": [
            "ML: GEMM kernels for training and inference (FP16/BF16/INT8 matmul)",
            "Scientific computing: Dense linear algebra, eigenvalue decomposition",
            "Signal processing: 2D convolution, correlation matrices",
            "Compiler/library devs: BLAS implementation, oneDNN backend",
        ],
        "audience": ["ML/AI engineers", "HPC developers", "library/runtime developers"],
    },
    "MTE": {
        "full_name": "Memory Tagging Extension",
        "acronym": "MTE",
        "introduced": "ARMv8.5-A (optional)",
        "purpose": (
            "Provides hardware-assisted memory safety by associating a 4-bit "
            "tag with every aligned 16-byte granule of physical memory and a "
            "corresponding tag in the top bits of pointers. On every memory "
            "access the CPU checks that the pointer tag matches the memory tag, "
            "trapping mismatches."
        ),
        "problem_solved": (
            "Use-after-free, buffer overflows, and other spatial/temporal memory "
            "safety bugs are the dominant source of security vulnerabilities in "
            "C/C++ code. Software-only sanitizers (ASan) are too slow for "
            "production. MTE provides low-overhead hardware detection."
        ),
        "key_registers": [
            "TFSR_EL1/EL2: Tag Fault Status Register (records async tag check faults)",
            "TFSRE0_EL1: Tag Fault Status Register for EL0",
            "GCR_EL1: Tag Generation Control Register (controls random tag exclusion)",
            "RGSR_EL1: Random Tag Generation Seed Register",
            "TCR_EL1.TBI0/1: Top Byte Ignore (must be enabled for MTE)",
            "SCTLR_EL1.TCF0/TCF: Tag Check Fault mode (sync/async/asymmetric)",
        ],
        "key_instructions": [
            "IRG: Insert Random tag into a pointer (Xd = Xn with random tag)",
            "ADDG/SUBG: Add/subtract with tag generation",
            "STG/STZG: Store Allocation Tag / Store Zero and Tag",
            "ST2G/STZ2G: Store tags for 32-byte granules",
            "LDG: Load Allocation Tag from memory into pointer",
            "CMPP: Compare pointer tags (subtracts and sets flags)",
        ],
        "detection": {
            "register": "ID_AA64PFR1_EL1",
            "field": "MTE (bits [11:8])",
            "values": "0b0001 = MTE1 (EL0 tag check); 0b0010 = MTE2 (full); 0b0011 = MTE3 (asymmetric)",
            "linux": "HWCAP2_MTE in getauxval(AT_HWCAP2); prctl(PR_SET_TAGGED_ADDR_CTRL)",
        },
        "use_cases": [
            "Security: Detect heap use-after-free and buffer overflows at near-zero overhead",
            "OS kernels: Tag kernel heap allocations (CONFIG_ARM64_MTE in Linux)",
            "Android: Enabled in production on Pixel 8+ for malloc hardening",
            "Debugging: Deterministic mode as a faster alternative to ASan in testing",
        ],
        "audience": ["security engineers", "OS/kernel developers", "app developers (C/C++)"],
    },
    "PAC": {
        "full_name": "Pointer Authentication Codes",
        "acronym": "PAC",
        "introduced": "ARMv8.3-A (optional)",
        "purpose": (
            "Uses cryptographic signing of pointer values (return addresses, "
            "function pointers, data pointers) to detect corruption caused by "
            "control-flow hijacking attacks. A keyed MAC is inserted into the "
            "unused upper bits of a 64-bit pointer and verified before use."
        ),
        "problem_solved": (
            "Return-Oriented Programming (ROP) and Jump-Oriented Programming (JOP) "
            "attacks overwrite return addresses or function pointers to chain "
            "existing code gadgets. PAC makes forging valid pointers computationally "
            "infeasible without the secret key."
        ),
        "key_registers": [
            "APIAKeyLo_EL1 / APIAKeyHi_EL1: Instruction A key (signs return addresses)",
            "APIBKeyLo_EL1 / APIBKeyHi_EL1: Instruction B key (alternative signing)",
            "APDAKeyLo_EL1 / APDAKeyHi_EL1: Data A key (signs data pointers)",
            "APDBKeyLo_EL1 / APDBKeyHi_EL1: Data B key (alternative data signing)",
            "APGAKeyLo_EL1 / APGAKeyHi_EL1: Generic authentication key",
        ],
        "key_instructions": [
            "PACIA/PACIB/PACDA/PACDB: Compute and insert PAC into pointer",
            "PACIASP/PACIBSP: Sign LR with SP as context (function prologue)",
            "AUTIA/AUTIB/AUTDA/AUTDB: Authenticate and strip PAC (function epilogue)",
            "AUTIASP/AUTIBSP: Authenticate LR with SP context",
            "XPACI/XPACD: Strip PAC without checking (for unwinders)",
            "PACGA: Generic authentication (computes a PAC in a GP register)",
            "RETAA/RETAB: Combined authenticate-and-return (AUT + RET)",
            "BRAA/BRAB/BLRAA/BLRAB: Authenticated branch/branch-with-link",
        ],
        "detection": {
            "register": "ID_AA64ISAR1_EL1",
            "field": "APA (bits [7:4]) or API (bits [11:8])",
            "values": "APA: QARMA PAC; API: IMP-DEF PAC algorithm. Non-zero = implemented.",
            "linux": "HWCAP_PACA / HWCAP_PACG in getauxval(AT_HWCAP)",
        },
        "use_cases": [
            "Security: ROP/JOP mitigation for all compiled code",
            "OS kernels: Linux CONFIG_ARM64_PTR_AUTH; signing kernel return addresses",
            "Compilers: GCC/Clang -mbranch-protection=pac-ret+leaf",
            "Apple: Mandatory on all Apple Silicon (M1+), enforced by iOS/macOS kernel",
        ],
        "audience": ["security engineers", "OS/kernel developers", "toolchain developers"],
    },
    "BTI": {
        "full_name": "Branch Target Identification",
        "acronym": "BTI",
        "introduced": "ARMv8.5-A (optional)",
        "purpose": (
            "Restricts the locations where indirect branches can land by marking "
            "valid branch targets with dedicated BTI instructions. Any indirect "
            "branch that lands on an instruction that is not a valid BTI landing "
            "pad generates a Branch Target Exception."
        ),
        "problem_solved": (
            "Jump-Oriented Programming (JOP) and Call-Oriented Programming (COP) "
            "attacks use indirect branches to jump to arbitrary code gadgets. "
            "BTI constrains the set of valid landing sites, drastically reducing "
            "the number of usable gadgets."
        ),
        "key_registers": [
            "SCTLR_EL1.BT0/BT1: Enable BTI enforcement at EL0/EL1",
            "SCTLR_EL2.BT: Enable BTI enforcement at EL2",
            "PSTATE.BTYPE: Tracks the branch type (00=none, 01=BR-type, 10=BLR-type, 11=jump-table)",
            "GPCCR_EL3 / GPCCR_EL2: Guard page control for BTI pages",
        ],
        "key_instructions": [
            "BTI (no operand): Valid landing pad for any indirect branch",
            "BTI C: Valid for BLR (call) type indirect branches",
            "BTI J: Valid for BR (jump) type indirect branches",
            "BTI JC: Valid for both BR and BLR type indirect branches",
            "Note: PACIASP/PACIBSP also act as BTI C landing pads",
        ],
        "detection": {
            "register": "ID_AA64PFR1_EL1",
            "field": "BT (bits [3:0])",
            "values": "0b0001 = BTI implemented",
            "linux": "HWCAP2_BTI in getauxval(AT_HWCAP2); GNU_PROPERTY_AARCH64_FEATURE_1_BTI in ELF note",
        },
        "use_cases": [
            "Security: Forward-edge CFI for all indirect branches",
            "OS/linker: ld --force-bti marks all pages as BTI-guarded",
            "Compilers: GCC/Clang -mbranch-protection=bti or standard+pac-ret",
            "Combines with PAC for full CFI: PAC protects backward edge (returns), BTI protects forward edge",
        ],
        "audience": ["security engineers", "toolchain developers", "OS/kernel developers"],
    },
    "RME": {
        "full_name": "Realm Management Extension",
        "acronym": "RME",
        "introduced": "ARMv9.2-A (optional, part of ARM CCA)",
        "purpose": (
            "Introduces a fourth security world -- the Realm world -- alongside "
            "Secure, Non-secure, and Root. RME enables ARM Confidential Compute "
            "Architecture (CCA), allowing workloads to run in isolated Realms "
            "that are protected from the hypervisor, host OS, and even secure "
            "firmware by hardware-enforced memory encryption and access controls."
        ),
        "problem_solved": (
            "Traditional TEEs (TrustZone) require trusting the secure world firmware "
            "and the hypervisor. Cloud tenants must trust the cloud provider's entire "
            "software stack. RME removes this trust requirement by cryptographically "
            "isolating Realm workloads from all other software, including privileged code."
        ),
        "key_registers": [
            "GPCCR_EL3: Granule Protection Check Control Register (Root world)",
            "GPTBR_EL3: Granule Protection Table Base Register",
            "GPT entries: Per-granule security state (NS/S/Realm/Root)",
            "MECID registers: Memory Encryption Context ID for realm isolation",
            "MFAR_EL3: Monitor Fault Address Register (for GPF faults)",
        ],
        "key_instructions": [
            "TLBI RPAOS/RPALOS: TLB invalidate by Realm PA",
            "SMC (to RMM): Realm Management Monitor calls",
            "RSI (Realm Services Interface): Realm-to-RMM calls for attestation",
            "GPT walk: Hardware granule protection table lookups on every access",
        ],
        "detection": {
            "register": "ID_AA64PFR0_EL1",
            "field": "RME (bits [55:52])",
            "values": "0b0001 = RME implemented",
            "linux": "Not directly exposed to userspace; firmware/hypervisor feature",
        },
        "use_cases": [
            "Confidential computing: Run VMs in Realms isolated from the hypervisor",
            "Cloud: Protect tenant workloads from cloud provider's software stack",
            "Attestation: Hardware-rooted attestation of Realm identity and integrity",
            "Edge/IoT: Isolate sensitive workloads without full TrustZone complexity",
        ],
        "audience": ["cloud/platform architects", "hypervisor developers", "firmware engineers"],
    },
    "GCS": {
        "full_name": "Guarded Control Stack",
        "acronym": "GCS",
        "introduced": "ARMv9.4-A (optional, FEAT_GCS)",
        "purpose": (
            "Provides a hardware-enforced shadow call stack that stores return "
            "addresses separately from the normal stack. On function return, the "
            "CPU checks that the return address on the normal stack matches the "
            "one on the guarded control stack. The GCS is not writable by normal "
            "store instructions, preventing tampering."
        ),
        "problem_solved": (
            "Return-Oriented Programming (ROP) attacks overwrite return addresses "
            "on the stack. While PAC signs return addresses, GCS provides an "
            "orthogonal defense: a separate hardware-protected stack that software "
            "cannot corrupt. Together with PAC, it provides defense-in-depth."
        ),
        "key_registers": [
            "GCSPR_EL0/EL1/EL2/EL3: GCS Stack Pointer for each exception level",
            "GCSCR_EL1/EL2/EL3: GCS Control Register (enable, push/pop policy)",
            "GCSCRE0_EL1: GCS Control for EL0 (userspace GCS policy)",
        ],
        "key_instructions": [
            "GCSPUSHM: Push a value onto the GCS (privileged or explicit)",
            "GCSPOPM: Pop a value from the GCS",
            "GCSSS1/GCSSS2: GCS switch stack (for context switch / signal handling)",
            "GCSSTTR: GCS store with tag (for kernel manipulation of user GCS)",
            "RET: Implicitly checks GCS entry against LR",
        ],
        "detection": {
            "register": "ID_AA64PFR1_EL1",
            "field": "GCS (bits [47:44])",
            "values": "0b0001 = GCS implemented",
            "linux": "HWCAP2_GCS; prctl(PR_SET_SHADOW_STACK_STATUS) to enable",
        },
        "use_cases": [
            "Security: Hardware shadow stack prevents ROP even without PAC",
            "Defense-in-depth: Layered with PAC for both cryptographic and structural protection",
            "OS: Linux and Windows support for hardware-enforced call stack integrity",
            "Transparent: User code needs no changes; enabled per-process by the OS",
        ],
        "audience": ["OS/kernel developers", "security engineers", "platform architects"],
    },
    "FEAT_THE": {
        "full_name": "Translation Hardening Extension",
        "acronym": "THE (FEAT_THE)",
        "introduced": "ARMv9.4-A (optional)",
        "purpose": (
            "Hardens the translation table walk mechanism against fault injection "
            "and hardware attacks. Adds instructions for atomic, unprivileged "
            "translation table entry (TTE) reads that are guaranteed to be "
            "consistent and tamper-evident, even under concurrent modification "
            "or physical fault injection."
        ),
        "problem_solved": (
            "Translation table entries can be targets for rowhammer or other "
            "physical fault injection attacks that flip bits to gain privilege "
            "escalation. THE adds hardware checks during table walks to detect "
            "and mitigate such corruptions."
        ),
        "key_registers": [
            "TCR_EL1 extensions: Additional control fields for translation hardening",
            "TCR2_EL1: Extended Translation Control Register (FEAT_TCR2, related)",
            "HCRX_EL2: Hypervisor Configuration Register extensions for THE controls",
        ],
        "key_instructions": [
            "RCWSWP/RCWSWPA/RCWSWPAL: Read-Check-Write Swap (atomic TTE update with check)",
            "RCWCLR/RCWSET: Read-Check-Write Clear/Set for TTE fields",
            "RCWCAS/RCWCASA/RCWCASAL: Read-Check-Write Compare-and-Swap",
            "These operate on 128-bit translation table descriptors atomically",
        ],
        "detection": {
            "register": "ID_AA64MMFR1_EL1",
            "field": "THE (bits [51:48])",
            "values": "0b0001 = FEAT_THE implemented",
            "linux": "Kernel-internal; not directly exposed to userspace",
        },
        "use_cases": [
            "Security: Protection against rowhammer attacks on page tables",
            "Hypervisors: Safe concurrent modification of stage-2 translation tables",
            "OS kernels: Hardened page table management resistant to fault injection",
            "Firmware: Secure boot chains with tamper-evident memory mappings",
        ],
        "audience": ["kernel developers", "hypervisor developers", "security researchers"],
    },
    "FEAT_NV2": {
        "full_name": "Enhanced Nested Virtualization",
        "acronym": "NV2 (FEAT_NV2)",
        "introduced": "ARMv8.4-A (optional, builds on FEAT_NV from ARMv8.3-A)",
        "purpose": (
            "Improves the performance of nested virtualization by allowing a guest "
            "hypervisor's EL2 register accesses to be redirected to memory rather "
            "than trapping to the host hypervisor. The hardware automatically reads "
            "and writes a VNCR (Virtual Nested Control Register) page in memory, "
            "eliminating thousands of traps during guest hypervisor operation."
        ),
        "problem_solved": (
            "Running a hypervisor inside a VM (nested virtualization) causes frequent "
            "traps on every EL2 register access, creating severe performance overhead. "
            "FEAT_NV2 reduces this by mapping most EL2 system register accesses to "
            "a memory-backed page, avoiding exits to the host hypervisor."
        ),
        "key_registers": [
            "HCR_EL2.NV: Enable nested virtualization (FEAT_NV)",
            "HCR_EL2.NV1: Enhanced nested virt (redefine EL1 registers at vEL2)",
            "HCR_EL2.NV2: Enable VNCR-based register redirection (FEAT_NV2)",
            "VNCR_EL2: Virtual Nested Control Register base address",
            "VNCR page: Memory page containing virtual EL2 register values",
        ],
        "key_instructions": [
            "MSR/MRS to EL2 registers: Redirected to VNCR page instead of trapping",
            "ERET at vEL2: Uses VNCR-page values for ELR_EL2, SPSR_EL2, etc.",
            "AT S1E2x: Address translation at virtual EL2 (redirected through VNCR)",
            "TLBI: TLB operations at vEL2 (may still trap for some operations)",
        ],
        "detection": {
            "register": "ID_AA64MMFR2_EL1",
            "field": "NV (bits [27:24])",
            "values": "0b0001 = FEAT_NV; 0b0010 = FEAT_NV2",
            "linux": "KVM uses NV2 internally when available; not exposed to userspace",
        },
        "use_cases": [
            "Cloud: Efficient nested virtualization for running hypervisors in VMs",
            "Development: Test hypervisor code inside a VM without massive overhead",
            "Security: Nested compartmentalization for defense-in-depth",
            "KVM on ARM: Significant performance improvement for nested guests",
        ],
        "audience": ["hypervisor developers", "cloud platform engineers", "kernel developers"],
    },
    "TME": {
        "full_name": "Transactional Memory Extension",
        "acronym": "TME",
        "introduced": "ARMv9.0-A (optional, FEAT_TME)",
        "purpose": (
            "Provides hardware transactional memory support, allowing a group of "
            "memory accesses to execute atomically and in isolation. If a conflict "
            "is detected (another core accesses the same cache line), the transaction "
            "aborts and all modifications are rolled back to the state before TSTART."
        ),
        "problem_solved": (
            "Fine-grained locking is difficult to get right, and coarse-grained locks "
            "limit parallelism. Transactional memory allows optimistic concurrent "
            "execution: multiple threads proceed without locks, and only abort/retry "
            "if an actual conflict occurs."
        ),
        "key_registers": [
            "TSTATE: Transaction state (nesting depth, stored in NZCV on abort)",
            "No dedicated architectural registers; uses existing GPRs and memory",
            "Transaction checkpoint: Microarchitectural snapshot of register state",
        ],
        "key_instructions": [
            "TSTART Xd: Begin transaction; Xd receives status (0 = success, nonzero = retry hint on abort)",
            "TCOMMIT: Commit transaction (make all writes visible atomically)",
            "TCANCEL #imm: Explicitly cancel transaction with reason code",
            "TTEST Xd: Test if currently inside a transaction (Xd = nesting depth)",
        ],
        "detection": {
            "register": "ID_AA64ISAR0_EL1",
            "field": "TME (bits [27:24])",
            "values": "0b0001 = TME implemented",
            "linux": "HWCAP2_TME in getauxval(AT_HWCAP2) (if kernel support present)",
        },
        "use_cases": [
            "Databases: Lock-free concurrent data structure updates",
            "Runtime systems: Optimistic concurrency for managed languages (JVM, .NET)",
            "OS kernels: Speculative lock elision for high-contention paths",
            "HPC: Parallel graph algorithms with speculative edge updates",
        ],
        "audience": ["systems programmers", "database developers", "runtime/VM developers"],
    },
    "DIT": {
        "full_name": "Data Independent Timing",
        "acronym": "DIT",
        "introduced": "ARMv8.4-A (optional, FEAT_DIT)",
        "purpose": (
            "When the DIT bit is set in PSTATE, the processor guarantees that the "
            "execution time of certain data-processing instructions depends only on "
            "the instruction type and operand register sizes, not on the actual data "
            "values. This eliminates timing side channels in cryptographic code."
        ),
        "problem_solved": (
            "Timing side-channel attacks extract secret keys by measuring how long "
            "operations take on different data (e.g., variable-time multiplies, "
            "data-dependent branch prediction). DIT forces constant-time execution "
            "for covered instructions."
        ),
        "key_registers": [
            "PSTATE.DIT: Data Independent Timing bit (set via MSR DAIFSet or MSR DIT, #1)",
            "DIT (bit [24] of PSTATE): When 1, timing-invariant execution is active",
        ],
        "key_instructions": [
            "MSR DIT, #1: Enable data-independent timing",
            "MSR DIT, #0: Disable data-independent timing",
            "MRS Xd, DIT: Read current DIT state",
            "Covered instructions include: ADD, SUB, AND, ORR, EOR, MUL, MADD, MSUB, AES*, SHA*, etc.",
        ],
        "detection": {
            "register": "ID_AA64PFR0_EL1",
            "field": "DIT (bits [51:48])",
            "values": "0b0001 = DIT implemented",
            "linux": "HWCAP_DIT in getauxval(AT_HWCAP)",
        },
        "use_cases": [
            "Cryptography: Constant-time AES, RSA, ECC implementations",
            "Security libraries: OpenSSL, BoringSSL, libsodium use DIT when available",
            "Side-channel hardening: Protect password comparison, HMAC verification",
            "Government/compliance: Meet FIPS 140-3 side-channel resistance requirements",
        ],
        "audience": ["cryptography engineers", "security engineers", "library developers"],
    },
    "MPAM": {
        "full_name": "Memory Partitioning and Monitoring",
        "acronym": "MPAM",
        "introduced": "ARMv8.4-A (optional, FEAT_MPAM)",
        "purpose": (
            "Enables software to partition shared resources (caches, memory bandwidth, "
            "interconnect) among different workloads or VMs. MPAM assigns a Partition "
            "ID (PARTID) to each context, and resource controllers use this to enforce "
            "allocation limits and monitor usage per partition."
        ),
        "problem_solved": (
            "In multi-tenant cloud and mixed-criticality systems, noisy neighbors "
            "can monopolize shared cache and memory bandwidth, causing unpredictable "
            "latency for other workloads. MPAM provides hardware-enforced QoS isolation."
        ),
        "key_registers": [
            "MPAMIDR_EL1: MPAM ID Register (number of PARTIDs, PMGs supported)",
            "MPAM0_EL1: Default PARTID/PMG for EL0",
            "MPAM1_EL1: PARTID/PMG for EL1 (kernel)",
            "MPAM2_EL2: PARTID/PMG for hypervisor-assigned virtual partitions",
            "MPAMHCR_EL2: MPAM Hypervisor Control Register",
            "MSC (Memory System Component) registers: Per-resource allocation/monitoring",
        ],
        "key_instructions": [
            "MSR MPAM0_EL1, Xn: Set EL0 partition ID",
            "MSR MPAM1_EL1, Xn: Set EL1 partition ID",
            "No dedicated instructions; PARTID flows with every memory transaction",
            "Resource controllers implement CPOR (cache portion), MBWMAX (bandwidth limit), etc.",
        ],
        "detection": {
            "register": "ID_AA64PFR0_EL1",
            "field": "MPAM (bits [43:40])",
            "values": "0b0001 = MPAM implemented",
            "linux": "resctrl filesystem interface (similar to Intel RDT/CAT)",
        },
        "use_cases": [
            "Cloud: Isolate LLC allocation per VM to prevent cache thrashing",
            "Real-time: Guarantee minimum cache/bandwidth for latency-critical tasks",
            "Mixed-criticality: ADAS + infotainment on same SoC with resource guarantees",
            "Monitoring: Track per-workload cache occupancy and bandwidth consumption",
        ],
        "audience": ["cloud platform engineers", "hypervisor developers", "RTOS developers"],
    },
    "RAS": {
        "full_name": "Reliability, Availability, and Serviceability Extension",
        "acronym": "RAS",
        "introduced": "ARMv8.2-A (optional; RASv1), ARMv8.4-A (RASv1p1)",
        "purpose": (
            "Standardizes error detection, reporting, and recovery for hardware "
            "errors (ECC faults, cache parity errors, bus errors). RAS defines "
            "a uniform error record format, error synchronization barriers, and "
            "a mechanism for software to query, inject, and handle errors at each "
            "exception level."
        ),
        "problem_solved": (
            "Server-grade systems require standardized error handling for silent "
            "data corruption, memory ECC errors, and interconnect faults. Before RAS, "
            "error reporting was implementation-defined and inconsistent across cores, "
            "making OS error handling fragile and non-portable."
        ),
        "key_registers": [
            "ERRIDR_EL1: Error Record ID Register (number of error records)",
            "ERRSELR_EL1: Error Record Select Register (choose active record)",
            "ERXSTATUS_EL1: Selected error record Status Register",
            "ERXADDR_EL1: Selected error record Address Register",
            "ERXCTLR_EL1: Selected error record Control Register",
            "ERXMISC0/1_EL1: Implementation-defined error syndrome info",
            "DISR_EL1: Deferred Interrupt Status Register (SError info)",
        ],
        "key_instructions": [
            "ESB: Error Synchronization Barrier (synchronize pending SErrors)",
            "MSR/MRS to ERX* registers: Access error records",
            "SCTLR_EL1.IESB: Implicit ESB on exception entry",
            "No error injection instructions (done via implementation-defined MSC regs)",
        ],
        "detection": {
            "register": "ID_AA64PFR0_EL1",
            "field": "RAS (bits [31:28])",
            "values": "0b0001 = RASv1; 0b0010 = RASv1p1 (with RAS error recovery)",
            "linux": "Kernel GHES/CPER error handling; /sys/devices/system/edac/",
        },
        "use_cases": [
            "Servers: ECC error logging and correctable error counting",
            "Cloud: Live migration away from failing hardware before data corruption",
            "OS kernels: Standardized SError and abort handling across ARM implementations",
            "Compliance: Required for server-grade ARM SystemReady certification",
        ],
        "audience": ["firmware engineers", "kernel developers", "server platform architects"],
    },
    "SPE": {
        "full_name": "Statistical Profiling Extension",
        "acronym": "SPE",
        "introduced": "ARMv8.2-A (optional, FEAT_SPE)",
        "purpose": (
            "Hardware-assisted statistical sampling that captures detailed "
            "microarchitectural events (cache misses, TLB misses, branch "
            "mispredicts, latencies) for sampled instructions and writes them "
            "to a memory buffer as structured records. Unlike PMU counters, "
            "SPE provides per-instruction attribution."
        ),
        "problem_solved": (
            "Traditional PMU counters tell you how many cache misses occurred but "
            "not which instructions caused them. Software-based profiling (perf + PMI) "
            "has skid and cannot capture microarchitectural detail. SPE provides "
            "precise, low-overhead, per-instruction profiling."
        ),
        "key_registers": [
            "PMSCR_EL1/EL2: Profiling Sample Control Register",
            "PMSICR_EL1: Sampling Interval Counter Register",
            "PMSFCR_EL1: Sample Filter Control Register (filter by event type)",
            "PMSEVFR_EL1: Sample Event Filter Register",
            "PMBPTR_EL1: Profiling Buffer Write Pointer",
            "PMBLIMITR_EL1: Profiling Buffer Limit Register",
            "PMSIRR_EL1: Sampling Interval Reload Register",
        ],
        "key_instructions": [
            "No user-facing instructions; SPE is configured via system registers",
            "PSB CSYNC: Profiling Synchronization Barrier (flush sample buffer)",
            "Records are written to memory automatically by hardware",
        ],
        "detection": {
            "register": "ID_AA64DFR0_EL1",
            "field": "PMSVer (bits [35:32])",
            "values": "0b0001 = SPEv1; 0b0010 = SPEv1p1; 0b0011 = SPEv1p2",
            "linux": "perf record -e arm_spe// ; CONFIG_ARM_SPE_PMU",
        },
        "use_cases": [
            "Performance analysis: Identify exact instructions causing cache misses",
            "Compiler optimization: Profile-guided optimization with precise data",
            "Data center: Continuous low-overhead production profiling",
            "Memory analysis: Track load/store latency distributions per-instruction",
        ],
        "audience": ["performance engineers", "compiler developers", "platform architects"],
    },
    "AMU": {
        "full_name": "Activity Monitors Extension",
        "acronym": "AMU",
        "introduced": "ARMv8.4-A (optional, FEAT_AMUv1); ARMv8.6-A (AMUv1p1)",
        "purpose": (
            "Provides always-on, non-interruptible hardware counters that track "
            "CPU activity metrics such as core cycles, constant-frequency cycles, "
            "instruction retirement, and memory stalls. Unlike PMU counters, AMU "
            "counters cannot be reprogrammed or reset by software, providing "
            "trustworthy utilization data."
        ),
        "problem_solved": (
            "OS schedulers and power management need reliable CPU utilization metrics "
            "that are not affected by context switches or PMU reprogramming. AMU "
            "provides non-resettable, always-counting activity monitors suitable for "
            "DVFS (Dynamic Voltage and Frequency Scaling) decisions."
        ),
        "key_registers": [
            "AMCNTENSET0_EL0/AMCNTENCLR0_EL0: Counter enable set/clear (group 0)",
            "AMCNTENSET1_EL0/AMCNTENCLR1_EL0: Counter enable set/clear (group 1)",
            "AMEVCNTR0<n>_EL0: Activity monitor event counter (group 0)",
            "AMEVCNTR1<n>_EL0: Activity monitor event counter (group 1, aux)",
            "AMEVTYPER1<n>_EL0: Event type for auxiliary group 1 counters",
            "AMCR_EL0: Activity Monitor Control Register",
        ],
        "key_instructions": [
            "MRS Xd, AMEVCNTR0<n>_EL0: Read activity monitor counter",
            "No write access to group 0 counters (read-only, always counting)",
            "Group 0 architected: cycles, const-freq cycles, retired instructions, stall cycles",
        ],
        "detection": {
            "register": "ID_AA64PFR0_EL1",
            "field": "AMU (bits [47:44])",
            "values": "0b0001 = AMUv1; 0b0010 = AMUv1p1",
            "linux": "arch_topology driver reads AMU for frequency invariance; /sys/devices/system/cpu/cpufreq/",
        },
        "use_cases": [
            "DVFS: Feed actual IPC and stall data to frequency governors",
            "EAS (Energy Aware Scheduling): Task placement based on real utilization",
            "Capacity planning: Long-running activity data unaffected by profiling overhead",
            "Thermal management: Track sustained activity for throttling decisions",
        ],
        "audience": ["kernel developers", "power management engineers", "scheduler developers"],
    },
    "BRBE": {
        "full_name": "Branch Record Buffer Extension",
        "acronym": "BRBE",
        "introduced": "ARMv9.2-A (optional, FEAT_BRBE)",
        "purpose": (
            "Maintains a hardware-managed circular buffer of recent branch records "
            "(source address, target address, branch type, cycle count). Captures "
            "the last N branches executed, providing a lightweight call-path trace "
            "without the overhead of full instruction tracing."
        ),
        "problem_solved": (
            "Understanding call paths and branch behavior is critical for "
            "performance analysis, but full tracing (CoreSight ETM) has high "
            "bandwidth overhead. Intel's LBR (Last Branch Record) provides this "
            "on x86; BRBE is ARM's equivalent, enabling perf branch-stack sampling."
        ),
        "key_registers": [
            "BRBCR_EL1/EL2: Branch Record Buffer Control Register",
            "BRBFCR_EL1: Branch Record Buffer Filter Control Register",
            "BRBTS_EL1: Branch Record Buffer Timestamp",
            "BRBINF<n>_EL1: Branch Record n Information (type, EL, mispredict, etc.)",
            "BRBSRC<n>_EL1: Branch Record n Source Address",
            "BRBTGT<n>_EL1: Branch Record n Target Address",
        ],
        "key_instructions": [
            "MRS Xd, BRBINF<n>_EL1: Read branch record metadata",
            "MRS Xd, BRBSRC<n>_EL1: Read branch source address",
            "MRS Xd, BRBTGT<n>_EL1: Read branch target address",
            "BRB IALL: Invalidate all branch records",
            "BRB INJ: Inject a branch record (for virtualization)",
        ],
        "detection": {
            "register": "ID_AA64DFR0_EL1",
            "field": "BRBE (bits [55:52])",
            "values": "0b0001 = BRBEv1; 0b0010 = BRBEv1p1",
            "linux": "perf record --branch-filter any; CONFIG_ARM64_BRBE",
        },
        "use_cases": [
            "Performance: Call-graph profiling via branch stack sampling",
            "Debugging: Reconstruct execution path leading to a crash or exception",
            "Security: Detect control-flow anomalies (unexpected branch targets)",
            "AutoFDO: Automated feedback-directed optimization using branch profiles",
        ],
        "audience": ["performance engineers", "compiler developers", "security researchers"],
    },
}

# Alias mapping for flexible lookup
_EXTENSION_ALIASES: dict[str, str] = {
    "POINTER AUTHENTICATION": "PAC",
    "POINTER AUTH": "PAC",
    "BRANCH TARGET IDENTIFICATION": "BTI",
    "BRANCH TARGET ID": "BTI",
    "MEMORY TAGGING": "MTE",
    "MEMORY TAGGING EXTENSION": "MTE",
    "SCALABLE VECTOR EXTENSION": "SVE",
    "SCALABLE VECTOR EXTENSION 2": "SVE2",
    "SCALABLE MATRIX EXTENSION": "SME",
    "REALM MANAGEMENT": "RME",
    "REALM MANAGEMENT EXTENSION": "RME",
    "GUARDED CONTROL STACK": "GCS",
    "SHADOW STACK": "GCS",
    "TRANSLATION HARDENING": "FEAT_THE",
    "TRANSLATION HARDENING EXTENSION": "FEAT_THE",
    "THE": "FEAT_THE",
    "NESTED VIRTUALIZATION": "FEAT_NV2",
    "ENHANCED NESTED VIRTUALIZATION": "FEAT_NV2",
    "NV2": "FEAT_NV2",
    "FEAT_NV": "FEAT_NV2",
    "TRANSACTIONAL MEMORY": "TME",
    "TRANSACTIONAL MEMORY EXTENSION": "TME",
    "DATA INDEPENDENT TIMING": "DIT",
    "MEMORY PARTITIONING": "MPAM",
    "MEMORY PARTITIONING AND MONITORING": "MPAM",
    "RELIABILITY AVAILABILITY SERVICEABILITY": "RAS",
    "STATISTICAL PROFILING": "SPE",
    "STATISTICAL PROFILING EXTENSION": "SPE",
    "ACTIVITY MONITORS": "AMU",
    "ACTIVITY MONITORS EXTENSION": "AMU",
    "BRANCH RECORD BUFFER": "BRBE",
    "BRANCH RECORD BUFFER EXTENSION": "BRBE",
}


def _format_extension(ext: dict) -> str:
    """Format an extension data dict into readable text."""
    lines: list[str] = []
    lines.append(f"# {ext['full_name']} ({ext['acronym']})")
    lines.append("")
    lines.append(f"**Introduced in:** {ext['introduced']}")
    lines.append("")
    lines.append("## Purpose")
    lines.append(ext["purpose"])
    lines.append("")
    lines.append("## Problem Solved")
    lines.append(ext["problem_solved"])
    lines.append("")

    lines.append("## Key Registers")
    for reg in ext["key_registers"]:
        lines.append(f"  - {reg}")
    lines.append("")

    lines.append("## Key Instructions")
    for instr in ext["key_instructions"]:
        lines.append(f"  - {instr}")
    lines.append("")

    det = ext["detection"]
    lines.append("## Detection / Feature Identification")
    lines.append(f"  Register: {det['register']}")
    lines.append(f"  Field:    {det['field']}")
    lines.append(f"  Values:   {det['values']}")
    lines.append(f"  Linux:    {det['linux']}")
    lines.append("")

    lines.append("## Practical Use Cases")
    for uc in ext["use_cases"]:
        lines.append(f"  - {uc}")
    lines.append("")

    lines.append("## Target Audience")
    lines.append(f"  {', '.join(ext['audience'])}")

    return "\n".join(lines)


@mcp.tool()
def explain_extension(extension_name: str) -> str:
    """Explain an ARM architecture extension in detail.

    Returns the full name, which architecture version introduced it, purpose,
    key registers and instructions, detection method, and practical use cases.

    Args:
        extension_name: Extension name or acronym, e.g. "SVE", "MTE", "PAC",
                        "BTI", "TME", "RME", "SME", "GCS", "FEAT_THE",
                        "FEAT_NV2", "DIT", "MPAM", "RAS", "SPE", "AMU", "BRBE".
                        Case-insensitive. Also accepts full names like
                        "Pointer Authentication" or "Memory Tagging".
    """
    key = extension_name.strip().upper()

    # Direct lookup
    if key in _EXTENSIONS:
        return _format_extension(_EXTENSIONS[key])

    # Try alias lookup
    if key in _EXTENSION_ALIASES:
        return _format_extension(_EXTENSIONS[_EXTENSION_ALIASES[key]])

    # Fuzzy: check if input is a substring of any extension name or full name
    partial_matches: list[str] = []
    for ext_key, ext_data in _EXTENSIONS.items():
        if key in ext_key or key in ext_data["full_name"].upper():
            partial_matches.append(f"  - {ext_key}: {ext_data['full_name']}")

    all_names = sorted(_EXTENSIONS.keys())
    if partial_matches:
        return (
            f"No exact match for '{extension_name}'. Possible matches:\n"
            + "\n".join(partial_matches)
            + f"\n\nAll supported extensions: {', '.join(all_names)}"
        )

    return (
        f"Error: '{extension_name}' is not a recognised ARM extension.\n\n"
        f"Supported extensions: {', '.join(all_names)}\n\n"
        f"You can also use full names like 'Pointer Authentication', "
        f"'Memory Tagging', 'Scalable Vector Extension', etc."
    )


# ---------------------------------------------------------------------------
# Tool 13: compare_architecture_versions — ARM version feature comparison
# ---------------------------------------------------------------------------

_ARCH_VERSIONS: dict[str, dict] = {
    "armv8.0": {
        "name": "ARMv8.0-A",
        "year": 2011,
        "mandatory_features": [
            "AArch64 execution state with A64 instruction set",
            "AArch32 backward compatibility (A32 + T32)",
            "NEON / Advanced SIMD (128-bit, mandatory in AArch64)",
            "VFPv4 floating-point",
            "Cryptographic extensions (AES, SHA-1, SHA-256) -- optional in early revisions",
            "EL0-EL3 exception levels",
            "Two-stage address translation (stage 1 + stage 2)",
            "48-bit virtual address space (256 TB)",
            "40-bit physical address space (default, up to 48-bit)",
            "PMSA / VMSA memory management",
            "GICv3 interrupt controller interface",
            "Load-Acquire / Store-Release atomics (LDADD, CAS, SWP -- v8.1 extends these)",
        ],
        "optional_features": [
            "FEAT_CRC32: CRC32 instructions",
            "Cryptographic Extension (AES + SHA)",
        ],
        "notable_changes": (
            "The foundational ARMv8-A architecture. Introduced 64-bit processing "
            "to ARM with the AArch64 state while maintaining full AArch32 backward "
            "compatibility. Defined the four exception levels (EL0-EL3), two-stage "
            "translation, and the A64 instruction set."
        ),
        "example_cores": "Cortex-A53, Cortex-A57, Cortex-A72, Apple A7 (Cyclone), X-Gene 1",
    },
    "armv8.1": {
        "name": "ARMv8.1-A",
        "year": 2014,
        "mandatory_features": [
            "LSE (Large System Extensions): Atomic read-modify-write instructions (LDADD, STADD, CAS, SWP, etc.)",
            "PAN (Privileged Access Never): Prevent kernel from accessing user memory without explicit override",
            "VHE (Virtualization Host Extensions): Run host kernel at EL2 efficiently",
            "VMID16: 16-bit VMID (up from 8-bit) for more VMs",
            "PMUv3p1: Extended PMU with 32-bit counters",
            "LOR (Limited Ordering Regions): Define memory ordering constraints",
        ],
        "optional_features": [
            "FEAT_HAFDBS: Hardware-managed Access Flag and Dirty Bit",
            "FEAT_HPD: Hierarchical Permission Disables (table descriptors)",
            "FEAT_RDMA: Rounding Double Multiply-Add (SQRDMLAH/SQRDMLSH for NEON)",
        ],
        "notable_changes": (
            "Major enhancements for server and virtualization. LSE atomics eliminated "
            "the need for LL/SC loops in multi-core systems, significantly improving "
            "lock performance. VHE allows Type-1 hypervisors like KVM to run directly "
            "at EL2 without performance penalties."
        ),
        "example_cores": "Cortex-A76 (also v8.2), ThunderX2, Neoverse N1 (also v8.2)",
    },
    "armv8.2": {
        "name": "ARMv8.2-A",
        "year": 2016,
        "mandatory_features": [
            "FEAT_ASMv8p2: ARMv8.2 mandatory AT (address translation) enhancements",
            "FEAT_DPB: DC CVAP -- data cache clean to Point of Persistence",
            "FEAT_IESB: Implicit Error Synchronization Barrier on exception entry",
            "FEAT_LPA: Large Physical Address (52-bit PA when FEAT_LPA is combined with translation)",
            "FEAT_LSMAOC: Load/Store Multiple Atomicity and Ordering (clarified)",
            "FEAT_PCSRv8p2: PC Sample-based profiling enhancements",
            "FEAT_UAO: User Access Override (kernel uses LDTR/STTR semantics with standard instructions)",
            "FEAT_TTL: Translation Table Level hint for TLBI operations",
        ],
        "optional_features": [
            "FEAT_SVE: Scalable Vector Extension (128-2048 bit vectors)",
            "FEAT_FP16: Half-precision (FP16) floating-point data processing",
            "FEAT_DotProd: UDOT/SDOT instructions for INT8 dot products (ML inference)",
            "FEAT_FHM: FP16 to FP32 multiply-accumulate (FMLAL)",
            "FEAT_RAS: RAS Extension v1 (error records, ESB)",
            "FEAT_SPE: Statistical Profiling Extension",
            "FEAT_SHA512: SHA-512 / SHA-3 instructions",
            "FEAT_SHA3: SHA-3 instructions",
            "FEAT_SM3: SM3 hash instructions",
            "FEAT_SM4: SM4 encryption instructions",
        ],
        "notable_changes": (
            "A pivotal release for HPC and ML. SVE introduced scalable vectors, FP16 "
            "enabled half-precision compute for ML, and DotProd provided 4x throughput "
            "improvement for INT8 inference. Also added 52-bit PA support for large "
            "memory servers."
        ),
        "example_cores": "Cortex-A76, Cortex-A77, Cortex-A55, Neoverse N1, Fujitsu A64FX (SVE)",
    },
    "armv8.3": {
        "name": "ARMv8.3-A",
        "year": 2016,
        "mandatory_features": [
            "FEAT_FCMA: Floating-point Complex Multiply-Add (FCMLA, FCADD)",
            "FEAT_JSCVT: JavaScript FP conversion (FJCVTZS -- for JS JIT compilers)",
            "FEAT_LRCPC: Load-Acquire RCpc (LDAPR -- weaker acquire for C/C++ memory model)",
            "Larger-than-VA range pointer authentication (PAC uses upper pointer bits)",
        ],
        "optional_features": [
            "FEAT_PAuth: Pointer Authentication (PAC) -- PACIA, AUTIA, PACGA, RETAA, etc.",
            "FEAT_NV: Nested Virtualization support (vEL2 state)",
            "FEAT_CCIDX: Extended cache ID (CCSIDR2_EL1 for large caches)",
            "FEAT_SPEv1p1: Statistical Profiling Extension v1.1",
        ],
        "notable_changes": (
            "Introduced Pointer Authentication, one of the most significant security "
            "features in ARMv8. LRCPC improved C/C++ atomic performance by providing "
            "a weaker-than-acquire load matching the C++ memory_order_consume semantics. "
            "Complex-number FP instructions benefit DSP and scientific code."
        ),
        "example_cores": "Cortex-A76, Cortex-A77, Neoverse N1, Apple A12 (Vortex/Tempest)",
    },
    "armv8.4": {
        "name": "ARMv8.4-A",
        "year": 2017,
        "mandatory_features": [
            "FEAT_LRCPC2: LDAPUR/STLUR -- RCpc acquire/release with immediate offset",
            "FEAT_FlagM: Condition flag manipulation (CFINV, RMIF, SETF8, SETF16)",
            "FEAT_TLBIOS: TLB invalidate by address in Outer Shareable domain",
            "FEAT_TLBIRANGE: TLB invalidate by range of addresses",
            "FEAT_SEL2: Secure EL2 (hypervisor in Secure world)",
            "FEAT_IDST: ID register traps for nested virtualization",
        ],
        "optional_features": [
            "FEAT_DIT: Data Independent Timing (constant-time crypto)",
            "FEAT_NV2: Enhanced Nested Virtualization (VNCR register page)",
            "FEAT_MPAM: Memory Partitioning and Monitoring",
            "FEAT_AMUv1: Activity Monitors Extension",
            "FEAT_TTRem: Translation Table Remainder bit for > 48-bit VA",
            "FEAT_S2FWB: Stage 2 Forced Write-Back (simplify cache management under hypervisor)",
            "FEAT_TTST: Small Translation Table support",
        ],
        "notable_changes": (
            "Strengthened virtualization with Secure EL2, TLB range invalidation "
            "(major performance win for VM migration), and enhanced nested virt (NV2). "
            "DIT enabled constant-time crypto. MPAM brought cache/BW partitioning "
            "for server QoS."
        ),
        "example_cores": "Cortex-A78, Cortex-X1, Neoverse V1, Apple A14 (Firestorm/Icestorm)",
    },
    "armv8.5": {
        "name": "ARMv8.5-A",
        "year": 2018,
        "mandatory_features": [
            "FEAT_FLAGM2: Additional flag manipulation (AXFLAG, XAFLAG for FP comparison)",
            "FEAT_FRINTTS: Floating-point round to integer (FRINT32Z, FRINT32X, FRINT64Z, FRINT64X)",
            "FEAT_SB: Speculation Barrier instruction (SB -- Spectre mitigation)",
            "FEAT_SPECRES: Speculation restriction (CFPRCTX, DVPRCTX, CPPRCTX -- predict restrict)",
            "FEAT_SSBS2: Speculative Store Bypass Safe (PSTATE.SSBS for Spectre v4)",
        ],
        "optional_features": [
            "FEAT_BTI: Branch Target Identification (forward-edge CFI)",
            "FEAT_MTE: Memory Tagging Extension (4-bit tag per 16-byte granule)",
            "FEAT_MTE2: Full MTE with synchronous/async checking modes",
            "FEAT_RNG: True random number generator (RNDR, RNDRRS instructions)",
            "FEAT_DF2: Additional debug features",
            "FEAT_E0PD: Preventing EL0 access to certain EL1 regions",
        ],
        "notable_changes": (
            "Focused heavily on security. MTE brought hardware memory safety, BTI "
            "completed the CFI story (forward-edge, complementing PAC's backward-edge). "
            "Speculation barriers and SSBS addressed Spectre-class side channels. "
            "FRINTTS helped JavaScript JIT compilers."
        ),
        "example_cores": "Cortex-A78, Cortex-X1, Cortex-X2, Neoverse V1, Apple A15",
    },
    "armv8.6": {
        "name": "ARMv8.6-A",
        "year": 2019,
        "mandatory_features": [
            "FEAT_BF16: BFloat16 data processing (BFCVT, BFDOT, BFMLAL, BFMMLA for NEON)",
            "FEAT_I8MM: INT8 Matrix Multiply (SMMLA, UMMLA, USMMLA -- 2x4 * 4x2 matmul)",
            "FEAT_ECV: Enhanced Counter Virtualization (virtual counter for VMs)",
            "FEAT_FGT: Fine-Grained Traps (per-register trap control for hypervisors)",
            "FEAT_TWED: Trap WFE Delay (configurable WFE timeout for hypervisors)",
            "FEAT_AMUv1p1: Activity Monitors v1.1 (virtual AMU offsets for VMs)",
        ],
        "optional_features": [
            "FEAT_MTPMU: Prohibit EL0 access to PMU counters",
            "FEAT_MTE3: MTE with asymmetric tag check fault handling",
        ],
        "notable_changes": (
            "Major ML/AI focus. BF16 and INT8 matrix multiply instructions dramatically "
            "accelerated inference on NEON. ECV and FGT significantly improved "
            "virtualization performance by reducing trap overhead. This version "
            "represents the baseline for ARMv9.1-A."
        ),
        "example_cores": "Cortex-A710, Cortex-X2, Neoverse N2, Neoverse V2, Apple M2",
    },
    "armv8.7": {
        "name": "ARMv8.7-A",
        "year": 2020,
        "mandatory_features": [
            "FEAT_WFxT: WFE and WFI with Timeout (WFET, WFIT -- bounded polling)",
            "FEAT_HCX: Extended Hypervisor Configuration (HCRX_EL2 register)",
            "FEAT_PAN3: PAN enhanced (SCTLR_EL1.EPAN for execute-never on EL0 accessible pages)",
            "FEAT_XS: XS attribute (non-cacheable with reduced ordering for accelerators)",
        ],
        "optional_features": [
            "FEAT_LPA2: 52-bit VA + 52-bit PA with 4KB/16KB granules (not just 64KB)",
            "FEAT_LS64: Single-copy atomic 64-byte loads/stores (LD64B, ST64B, ST64BV0)",
            "FEAT_RPRES: Increased floating-point reciprocal precision (12-bit vs 8-bit)",
            "FEAT_AFP: Alternate Floating-Point (FPCR.AH, FIZ, NEP for ML-friendly FP behavior)",
            "FEAT_SPEv1p2: Statistical Profiling Extension v1.2",
            "FEAT_PMUv3p7: PMU v3.7 extensions",
        ],
        "notable_changes": (
            "LS64 enables 512-bit atomic transfers for accelerator interfaces (PCIe, CXL). "
            "LPA2 extended 52-bit addressing to 4KB granule systems (previously only 64KB). "
            "WFxT with timeout improves latency in polling loops. AFP provides "
            "ML-friendly FP behavior (flush-to-zero on input, no exceptions)."
        ),
        "example_cores": "Cortex-A715, Cortex-X3, Neoverse V2",
    },
    "armv8.8": {
        "name": "ARMv8.8-A",
        "year": 2021,
        "mandatory_features": [
            "FEAT_NMI: Non-Maskable Interrupts (PSTATE.ALLINT, SCTLR_ELx.NMI)",
            "FEAT_TIDCP1: Trap IMPLEMENTATION DEFINED functionality (prevent EL0 access to imp-def sysregs)",
            "FEAT_CMOW: Control for cache maintenance (permission model for user-space cache ops)",
            "FEAT_HBC: Hinted Conditional Branches (BC.cond -- branch prediction hint)",
            "FEAT_MOPS: Memory Copy/Set operations (CPYFP/CPYFM/CPYFE, SETP/SETM/SETE)",
        ],
        "optional_features": [
            "FEAT_PMUv3p8: PMU v3.8 with instruction-counting extensions",
            "FEAT_SPEv1p3: Statistical Profiling Extension v1.3",
        ],
        "notable_changes": (
            "MOPS provides hardware-accelerated memcpy/memset/memmove using a three-instruction "
            "sequence that is restartable and handles overlaps correctly. NMI support brings "
            "ARM in line with x86 for critical interrupt handling. HBC enables software "
            "to hint branch direction to the predictor."
        ),
        "example_cores": "Cortex-A720, Cortex-X4, Neoverse V3 (Poseidon)",
    },
    "armv8.9": {
        "name": "ARMv8.9-A",
        "year": 2022,
        "mandatory_features": [
            "FEAT_CLRBHB: Clear Branch History (speculative execution mitigation)",
            "FEAT_CSSC: Common Short Sequence Compression (ABS, CNT, CTZ, SMAX, SMIN, UMAX, UMIN, etc.)",
            "FEAT_PRFMSLC: Prefetch hints for SLC (System-Level Cache)",
            "FEAT_SPECRES2: Enhanced speculation restriction",
            "FEAT_RASv2: RAS Extension v2 (additional error record fields)",
        ],
        "optional_features": [
            "FEAT_MTE4: MTE enhancements (canonical tag checking, tag store reliability)",
            "FEAT_THE: Translation Hardening Extension (RCW* instructions)",
            "FEAT_GCS: Guarded Control Stack",
            "FEAT_DoubleFault2: Double Fault handling at higher ELs",
            "FEAT_PFar: Physical Fault Address Register improvements",
            "FEAT_SEBEP: Synchronous-Exception-Based Event Profiling",
            "FEAT_PMUv3p9: PMU v3.9",
            "FEAT_SPE_FDS: Statistical Profiling with forced data source sampling",
        ],
        "notable_changes": (
            "CSSC added scalar utility instructions ARM previously lacked (count leading/trailing "
            "zeros, absolute value, min/max in GP registers). GCS brought hardware shadow stacks. "
            "THE hardened page table walks against physical attacks. RASv2 improved server-grade "
            "error handling."
        ),
        "example_cores": "Cortex-A725, Cortex-X925, Neoverse N3 (announced)",
    },
    "armv9.0": {
        "name": "ARMv9.0-A",
        "year": 2021,
        "mandatory_features": [
            "Everything in ARMv8.5-A (ARMv9.0 is based on ARMv8.5)",
            "FEAT_SVE2: SVE2 is mandatory (scalable vectors for all workloads)",
            "FEAT_ETE: Embedded Trace Extension (successor to ETM for trace)",
            "FEAT_TRBE: Trace Buffer Extension (self-hosted trace to memory buffer)",
            "FEAT_RME: Realm Management Extension (hardware isolation for confidential compute)",
        ],
        "optional_features": [
            "FEAT_TME: Transactional Memory Extension",
            "FEAT_MTE / FEAT_MTE2: Memory Tagging Extension",
            "FEAT_BTI: Branch Target Identification",
            "FEAT_SME: Scalable Matrix Extension (introduced at v9.2)",
        ],
        "notable_changes": (
            "ARMv9 is the first major version change since ARMv8 in 2011. It is "
            "based on ARMv8.5-A but makes SVE2 mandatory, adds RME for confidential "
            "computing (ARM CCA), and introduces new trace infrastructure (ETE/TRBE). "
            "Every ARMv9.x feature set includes the corresponding ARMv8.(x+5) features."
        ),
        "example_cores": "Cortex-A710, Cortex-X2, Neoverse N2 (ARMv9.0 baseline)",
    },
    "armv9.1": {
        "name": "ARMv9.1-A",
        "year": 2021,
        "mandatory_features": [
            "Everything in ARMv9.0-A + ARMv8.6-A features",
            "FEAT_BF16: BFloat16 data processing (mandatory)",
            "FEAT_I8MM: INT8 matrix multiply (mandatory)",
            "FEAT_ECV: Enhanced Counter Virtualization (mandatory)",
            "FEAT_FGT: Fine-Grained Traps (mandatory)",
        ],
        "optional_features": [
            "FEAT_MTE3: MTE with asymmetric checking",
            "FEAT_MTPMU: Monitor trap for PMU",
        ],
        "notable_changes": (
            "Incorporates all ARMv8.6-A features into the ARMv9 line. BF16 and I8MM "
            "become mandatory, ensuring all ARMv9.1+ cores have ML-optimized instructions. "
            "Fine-grained traps improve hypervisor performance."
        ),
        "example_cores": "Cortex-A710, Cortex-X2, Neoverse N2 (overlap with v9.0 cores)",
    },
    "armv9.2": {
        "name": "ARMv9.2-A",
        "year": 2022,
        "mandatory_features": [
            "Everything in ARMv9.1-A + ARMv8.7-A features",
            "FEAT_WFxT: WFE/WFI with Timeout",
            "FEAT_HCX: Extended Hypervisor Configuration",
            "FEAT_PAN3: Enhanced Privileged Access Never",
        ],
        "optional_features": [
            "FEAT_SME: Scalable Matrix Extension (outer product, streaming SVE, ZA tiles)",
            "FEAT_BRBE: Branch Record Buffer Extension",
            "FEAT_RME: Realm Management Extension (if not already implemented)",
            "FEAT_LPA2: 52-bit addressing with 4KB/16KB granules",
            "FEAT_LS64: 64-byte atomic loads/stores",
        ],
        "notable_changes": (
            "The first version where SME becomes available as an optional feature, "
            "bringing native matrix operations to ARM. BRBE adds Intel-LBR-style "
            "branch recording for profiling. Corresponds to ARMv8.7 feature base."
        ),
        "example_cores": "Cortex-A715, Cortex-X3, Neoverse V2",
    },
    "armv9.3": {
        "name": "ARMv9.3-A",
        "year": 2022,
        "mandatory_features": [
            "Everything in ARMv9.2-A + ARMv8.8-A features",
            "FEAT_NMI: Non-Maskable Interrupts",
            "FEAT_MOPS: Memory Copy/Set operations (memcpy/memset acceleration)",
            "FEAT_HBC: Hinted Conditional Branches",
            "FEAT_TIDCP1: Trap IMPLEMENTATION DEFINED functionality",
        ],
        "optional_features": [
            "FEAT_SME2: Scalable Matrix Extension v2 (multi-vector, lookup table, range prefetch)",
            "FEAT_SVE2p1: SVE2.1 (predicate-as-counter, expanded BFloat16 ops)",
        ],
        "notable_changes": (
            "MOPS hardware-accelerated memory operations become mandatory. NMI support "
            "enables better real-time interrupt handling. SME2 adds multi-vector "
            "processing and lookup table instructions for ML inference optimization. "
            "Corresponds to ARMv8.8 feature base."
        ),
        "example_cores": "Cortex-A720, Cortex-X4, Neoverse V3",
    },
    "armv9.4": {
        "name": "ARMv9.4-A",
        "year": 2023,
        "mandatory_features": [
            "Everything in ARMv9.3-A + ARMv8.9-A features",
            "FEAT_CSSC: Common Short Sequence Compression (ABS, CTZ, CNT, SMAX, etc.)",
            "FEAT_CLRBHB: Clear Branch History",
            "FEAT_RASv2: RAS Extension v2",
            "FEAT_SPECRES2: Enhanced speculation restriction",
        ],
        "optional_features": [
            "FEAT_GCS: Guarded Control Stack (hardware shadow stack)",
            "FEAT_THE: Translation Hardening Extension",
            "FEAT_MTE4: MTE enhancements",
            "FEAT_SME2p1: Scalable Matrix Extension 2.1",
            "FEAT_SVE2p1: SVE2.1",
            "FEAT_DoubleFault2: Double Fault handling improvements",
            "FEAT_SEBEP: Synchronous-Exception-Based Event Profiling",
        ],
        "notable_changes": (
            "GCS brings hardware shadow stacks for ROP mitigation. THE hardens page "
            "tables against rowhammer. CSSC fills long-standing gaps in the scalar "
            "instruction set (CLZ/CTZ on GP registers, min/max without branching). "
            "Corresponds to ARMv8.9 feature base."
        ),
        "example_cores": "Cortex-A725, Cortex-X925, Neoverse N3 (announced)",
    },
    "armv9.5": {
        "name": "ARMv9.5-A",
        "year": 2024,
        "mandatory_features": [
            "Everything in ARMv9.4-A",
            "FEAT_CPA: Checked Pointer Arithmetic (pointer overflow checks in hardware)",
            "FEAT_FAMINMAX: FP Absolute Min/Max instructions",
            "FEAT_LUT: Lookup Table instructions (LUTI2, LUTI4 for vector quantization)",
            "FEAT_CMPBR: Compare-and-Branch instructions (direct compare + branch fusion)",
        ],
        "optional_features": [
            "FEAT_SME2p2: SME 2.2 (FP8 matrix operations)",
            "FEAT_SVE2p2: SVE 2.2 (expanded FP8 support, coprocessor interface)",
            "FEAT_MEC: Memory Encryption Contexts (for RME-based confidential compute)",
            "FEAT_PAuth_LR: PAC signing that includes the return address location",
            "FEAT_TLBIW: TLB Invalidation by Walking (range-based invalidation)",
            "FEAT_F8F16MM/F8F32MM: FP8 to FP16/FP32 matrix multiply",
        ],
        "notable_changes": (
            "FP8 (8-bit floating-point) support via SME2p2/SVE2p2 for next-generation "
            "ML inference with minimal precision loss. CPA adds hardware-checked pointer "
            "arithmetic for spatial memory safety. CMPBR fuses compare and branch into "
            "a single instruction, improving code density. LUT instructions accelerate "
            "quantization and table-driven algorithms."
        ),
        "example_cores": "Cortex-A (next-gen, 2025+), Neoverse (next-gen, 2025+) -- announced",
    },
}

# Build aliases for version lookup (normalize input)
_VERSION_ALIASES: dict[str, str] = {}
for _vkey in _ARCH_VERSIONS:
    _VERSION_ALIASES[_vkey] = _vkey
    _VERSION_ALIASES[_vkey.replace("arm", "")] = _vkey
    _VERSION_ALIASES[_vkey.replace("armv", "")] = _vkey
    _VERSION_ALIASES[_ARCH_VERSIONS[_vkey]["name"].lower()] = _vkey
    _VERSION_ALIASES[_vkey + "-a"] = _vkey


def _normalize_version(version: str) -> str | None:
    """Normalize a version string to a key in _ARCH_VERSIONS, or return None."""
    v = version.strip().lower().rstrip("-a").rstrip("-")
    if v in _VERSION_ALIASES:
        return _VERSION_ALIASES[v]
    if not v.startswith("arm"):
        candidate = "arm" + v
        if candidate in _VERSION_ALIASES:
            return _VERSION_ALIASES[candidate]
        candidate = "armv" + v
        if candidate in _VERSION_ALIASES:
            return _VERSION_ALIASES[candidate]
    return None


def _format_version(key: str, data: dict) -> str:
    """Format a single architecture version into readable text."""
    lines: list[str] = []
    lines.append(f"# {data['name']} ({data['year']})")
    lines.append("")

    lines.append("## Mandatory Features")
    for feat in data["mandatory_features"]:
        lines.append(f"  - {feat}")
    lines.append("")

    lines.append("## Optional Features")
    if data["optional_features"]:
        for feat in data["optional_features"]:
            lines.append(f"  - {feat}")
    else:
        lines.append("  (None specific to this version)")
    lines.append("")

    lines.append("## Notable Changes")
    lines.append(data["notable_changes"])
    lines.append("")

    lines.append("## Example Cores")
    lines.append(f"  {data['example_cores']}")

    return "\n".join(lines)


def _format_version_comparison(key1: str, data1: dict, key2: str, data2: dict) -> str:
    """Format a side-by-side comparison of two architecture versions."""
    lines: list[str] = []
    lines.append(f"# Architecture Comparison: {data1['name']} vs {data2['name']}")
    lines.append("")

    all_keys = list(_ARCH_VERSIONS.keys())
    idx1 = all_keys.index(key1)
    idx2 = all_keys.index(key2)
    if idx1 > idx2:
        key1, key2 = key2, key1
        data1, data2 = data2, data1
        idx1, idx2 = idx2, idx1

    lines.append(f"## {data1['name']} ({data1['year']})")
    lines.append(f"Example cores: {data1['example_cores']}")
    lines.append("")
    lines.append(f"## {data2['name']} ({data2['year']})")
    lines.append(f"Example cores: {data2['example_cores']}")
    lines.append("")

    lines.append(f"## Features Added from {data1['name']} to {data2['name']}")
    lines.append("")

    for idx in range(idx1 + 1, idx2 + 1):
        step_key = all_keys[idx]
        step_data = _ARCH_VERSIONS[step_key]
        lines.append(f"### {step_data['name']} ({step_data['year']})")
        lines.append("")

        lines.append("**New Mandatory Features:**")
        for feat in step_data["mandatory_features"]:
            if feat.startswith("Everything in"):
                continue
            lines.append(f"  + {feat}")

        if step_data["optional_features"]:
            lines.append("")
            lines.append("**New Optional Features:**")
            for feat in step_data["optional_features"]:
                lines.append(f"  ~ {feat}")

        lines.append("")
        lines.append(f"**Summary:** {step_data['notable_changes']}")
        lines.append("")

    total_mandatory = 0
    total_optional = 0
    for idx in range(idx1 + 1, idx2 + 1):
        step_data = _ARCH_VERSIONS[all_keys[idx]]
        total_mandatory += len([f for f in step_data["mandatory_features"] if not f.startswith("Everything in")])
        total_optional += len(step_data["optional_features"])

    lines.append("---")
    lines.append(f"**Total new mandatory features:** {total_mandatory}")
    lines.append(f"**Total new optional features:** {total_optional}")
    lines.append(f"**Versions spanned:** {idx2 - idx1}")

    return "\n".join(lines)


@mcp.tool()
def compare_architecture_versions(version: str, compare_to: str | None = None) -> str:
    """List features for an ARM architecture version, or compare two versions.

    If only one version is provided, returns all mandatory and optional features
    introduced in that version, notable changes, and example cores.

    If two versions are provided, shows a side-by-side diff of all features
    added between the two versions.

    Covers ARMv8.0-A through ARMv8.9-A and ARMv9.0-A through ARMv9.5-A.

    Args:
        version: Architecture version, e.g. "armv8.0", "armv8.2", "armv9.0",
                 "v8.5", "8.3", "ARMv9.4-A". Case-insensitive, flexible format.
        compare_to: Optional second version to compare against.
                    If provided, shows features added between the two versions.
    """
    all_versions = sorted(_ARCH_VERSIONS.keys())
    version_display = ", ".join(_ARCH_VERSIONS[k]["name"] for k in all_versions)

    key1 = _normalize_version(version)
    if key1 is None:
        return (
            f"Error: '{version}' is not a recognised ARM architecture version.\n\n"
            f"Supported versions: {version_display}\n\n"
            f"Examples: 'armv8.0', 'v9.2', '8.5', 'ARMv9.4-A'"
        )

    if compare_to is None:
        return _format_version(key1, _ARCH_VERSIONS[key1])

    key2 = _normalize_version(compare_to)
    if key2 is None:
        return (
            f"Error: '{compare_to}' is not a recognised ARM architecture version.\n\n"
            f"Supported versions: {version_display}\n\n"
            f"Examples: 'armv8.0', 'v9.2', '8.5', 'ARMv9.4-A'"
        )

    if key1 == key2:
        return (
            f"Both versions resolve to {_ARCH_VERSIONS[key1]['name']}. "
            f"Showing single-version details instead.\n\n"
            + _format_version(key1, _ARCH_VERSIONS[key1])
        )

    return _format_version_comparison(key1, _ARCH_VERSIONS[key1], key2, _ARCH_VERSIONS[key2])


# ---------------------------------------------------------------------------
# Tool: show_assembly_pattern -- annotated ARM assembly for common patterns
# ---------------------------------------------------------------------------

_ASSEMBLY_PATTERNS: dict[str, dict[str, str]] = {
    "function_prologue": {
        "aarch64": (
            "// === AArch64 Function Prologue ===\n"
            "// Standard AAPCS64-compliant function entry sequence.\n"
            "\n"
            "    STP  X29, X30, [SP, #-96]!  // Pre-index: decrement SP by 96 bytes (frame size),\n"
            "                                 //   then store Frame Pointer (X29) and Link Register (X30)\n"
            "                                 //   at the new SP. This creates the frame record.\n"
            "    MOV  X29, SP                 // Set Frame Pointer to current Stack Pointer.\n"
            "                                 //   X29 now points to the saved {FP, LR} pair.\n"
            "    STP  X19, X20, [SP, #16]    // Save callee-saved registers X19-X20 at SP+16.\n"
            "    STP  X21, X22, [SP, #32]    // Save callee-saved registers X21-X22 at SP+32.\n"
            "    STP  X23, X24, [SP, #48]    // Save callee-saved registers X23-X24 at SP+48.\n"
            "    STP  X25, X26, [SP, #64]    // Save callee-saved registers X25-X26 at SP+64.\n"
            "    STP  X27, X28, [SP, #80]    // Save callee-saved registers X27-X28 at SP+80.\n"
            "\n"
            "// At this point:\n"
            "//   - SP is 16-byte aligned (required by AAPCS64)\n"
            "//   - X29 (FP) points to saved frame record {old_FP, old_LR}\n"
            "//   - All callee-saved registers are preserved\n"
            "//   - X0-X7 contain function arguments (available for use)\n"
            "//\n"
            "// For PAC-enabled code (ARMv8.3+), add at the very start:\n"
            "//     PACIASP                   // Sign LR with SP as context (Spectre/ROP mitigation)"
        ),
        "aarch32": (
            "// === AArch32 Function Prologue ===\n"
            "// Standard AAPCS32-compliant function entry sequence (ARM state).\n"
            "\n"
            "    PUSH {R4-R11, LR}           // Save callee-saved registers R4-R11 and Link Register\n"
            "                                 //   onto the stack. PUSH is alias for STMDB SP!, {reglist}.\n"
            "                                 //   SP is decremented by 4*N before storing.\n"
            "    SUB  SP, SP, #local_size    // Allocate space for local variables on the stack.\n"
            "                                 //   Replace 'local_size' with actual byte count\n"
            "                                 //   (must keep SP 8-byte aligned at public interfaces).\n"
            "    MOV  R11, SP                // (Optional) Set Frame Pointer (R11 in ARM state,\n"
            "                                 //   R7 in Thumb state) to current SP for stack unwinding.\n"
            "\n"
            "// At this point:\n"
            "//   - R0-R3 contain the first 4 arguments (available for use)\n"
            "//   - R4-R11 are free to use as local registers (saved on stack)\n"
            "//   - LR is saved so nested calls are safe\n"
            "//   - SP points to bottom of local variable area"
        ),
    },
    "function_epilogue": {
        "aarch64": (
            "// === AArch64 Function Epilogue ===\n"
            "// Standard AAPCS64-compliant function exit sequence.\n"
            "\n"
            "    LDP  X19, X20, [SP, #16]    // Restore callee-saved registers X19-X20.\n"
            "    LDP  X21, X22, [SP, #32]    // Restore callee-saved registers X21-X22.\n"
            "    LDP  X23, X24, [SP, #48]    // Restore callee-saved registers X23-X24.\n"
            "    LDP  X25, X26, [SP, #64]    // Restore callee-saved registers X25-X26.\n"
            "    LDP  X27, X28, [SP, #80]    // Restore callee-saved registers X27-X28.\n"
            "    LDP  X29, X30, [SP], #96    // Post-index: load saved FP and LR from SP,\n"
            "                                 //   then increment SP by 96 to deallocate the frame.\n"
            "    RET                          // Branch to address in X30 (LR). Functionally BR X30\n"
            "                                 //   but hints the branch predictor this is a return.\n"
            "\n"
            "// For PAC-enabled code (ARMv8.3+), replace RET with:\n"
            "//     AUTIASP                   // Authenticate LR using SP as context\n"
            "//     RET                       // Return (faults if PAC check failed)\n"
            "// Or use the combined:\n"
            "//     RETAA                     // Authenticate-and-return in one instruction"
        ),
        "aarch32": (
            "// === AArch32 Function Epilogue ===\n"
            "// Standard AAPCS32-compliant function exit sequence (ARM state).\n"
            "\n"
            "    ADD  SP, SP, #local_size    // Deallocate local variable space\n"
            "                                 //   (reverse of the SUB in the prologue).\n"
            "    POP  {R4-R11, PC}           // Restore callee-saved registers R4-R11 and load\n"
            "                                 //   saved LR directly into PC, which effects a return.\n"
            "                                 //   POP is alias for LDMIA SP!, {reglist}.\n"
            "\n"
            "// Alternative (classic style):\n"
            "//     POP  {R4-R11, LR}        // Restore regs and LR separately\n"
            "//     BX   LR                  // Branch-and-exchange to LR (ARM/Thumb interworking)"
        ),
    },
    "atomic_add": {
        "aarch64": (
            "// === AArch64 Atomic Add (using exclusive load/store) ===\n"
            "// Atomically increments the 64-bit value at [X0] by X1.\n"
            "// Old value returned in X2. Uses ARMv8.0 LDXR/STXR loop.\n"
            "\n"
            "1:  LDXR  X2, [X0]              // Load-Exclusive: read [X0] into X2.\n"
            "                                 //   Sets the local/global exclusive monitor.\n"
            "    ADD   X3, X2, X1            // Compute new_value = old_value + increment.\n"
            "    STXR  W4, X3, [X0]          // Store-Exclusive: attempt to write X3 to [X0].\n"
            "                                 //   W4 = 0 on success, 1 if monitor was cleared.\n"
            "    CBNZ  W4, 1b                // If store failed, retry (standard LL/SC loop).\n"
            "\n"
            "// NOTE: No memory ordering by itself.\n"
            "// For acquire: LDAXR. For release: STLXR. For seq_cst: both.\n"
            "//\n"
            "// ARMv8.1 LSE alternative (single instruction, no loop):\n"
            "//     LDADD  X1, X2, [X0]      // Atomic add: [X0] += X1, old value -> X2\n"
            "//     LDADDA / LDADDAL          // Acquire / Acquire-Release variants"
        ),
        "aarch32": (
            "// === AArch32 Atomic Add (using exclusive load/store) ===\n"
            "// Atomically increments the 32-bit value at [R0] by R1. Old value in R2.\n"
            "\n"
            "1:  LDREX  R2, [R0]             // Load-Exclusive: read [R0] into R2.\n"
            "    ADD    R3, R2, R1           // Compute new_value = old + increment.\n"
            "    STREX  R4, R3, [R0]         // Store-Exclusive: attempt write. R4=0 success, 1 fail.\n"
            "    CMP    R4, #0               // Check if store succeeded.\n"
            "    BNE    1b                   // If failed, retry.\n"
            "\n"
            "// For ordering: add DMB ISH before/after as needed.\n"
            "// For 64-bit atomics: use LDREXD/STREXD."
        ),
    },
    "atomic_cas": {
        "aarch64": (
            "// === AArch64 Compare-And-Swap ===\n"
            "// If [X0] == X1 (expected), store X2 (desired). Old value in X3.\n"
            "\n"
            "1:  LDAXR  X3, [X0]             // Load-Exclusive with Acquire: read [X0].\n"
            "    CMP    X3, X1               // Compare current with expected.\n"
            "    B.NE   2f                   // If not equal, CAS fails.\n"
            "    STLXR  W4, X2, [X0]         // Store-Exclusive with Release: try write desired.\n"
            "                                 //   W4 = 0 success, 1 failure.\n"
            "    CBNZ   W4, 1b               // If store failed, retry.\n"
            "2:                               // X3 = old value. Success if X3 == X1.\n"
            "\n"
            "// ARMv8.1 LSE alternative:\n"
            "//     MOV  X3, X1              // CAS: X3 = expected (in) and old (out)\n"
            "//     CAS  X3, X2, [X0]       // Atomic CAS\n"
            "//     CASA / CASAL             // Acquire / Acquire-Release variants"
        ),
        "aarch32": (
            "// === AArch32 Compare-And-Swap ===\n"
            "// If [R0] == R1, store R2. Old value in R3.\n"
            "\n"
            "1:  LDREX  R3, [R0]             // Load-Exclusive.\n"
            "    CMP    R3, R1               // Compare with expected.\n"
            "    BNE    2f                   // Not equal -> CAS fails.\n"
            "    STREX  R4, R2, [R0]         // Store-Exclusive: attempt write.\n"
            "    CMP    R4, #0\n"
            "    BNE    1b                   // Store failed -> retry.\n"
            "2:                               // R3 = old value.\n"
            "\n"
            "// For ordering, wrap with DMB ISH before and after."
        ),
    },
    "spinlock_acquire": {
        "aarch64": (
            "// === AArch64 Spinlock Acquire ===\n"
            "// X0 = lock address (0=unlocked, 1=locked). Acquire semantics.\n"
            "\n"
            "    MOV   W2, #1                // W2 = 1 (locked value).\n"
            "    SEVL                        // Send Event Locally: first WFE won't stall.\n"
            "1:  WFE                          // Wait For Event: low-power spin wait.\n"
            "    LDAXR  W3, [X0]             // Load-Exclusive with Acquire: read lock.\n"
            "                                 //   Acquire ensures we see all writes from\n"
            "                                 //   the previous lock holder.\n"
            "    CBNZ   W3, 1b               // If locked, spin.\n"
            "    STXR   W3, W2, [X0]         // Store-Exclusive: try to set lock=1.\n"
            "    CBNZ   W3, 1b               // If store failed, retry.\n"
            "\n"
            "// Critical section begins. All subsequent accesses ordered after LDAXR.\n"
            "//\n"
            "// ARMv8.1 LSE alternative:\n"
            "// 1:  SWPA W2, W3, [X0]        // Atomic swap with Acquire\n"
            "//     CBNZ W3, 1b              // If old!=0, lock was held"
        ),
        "aarch32": (
            "// === AArch32 Spinlock Acquire ===\n"
            "// R0 = lock address (0=unlocked, 1=locked).\n"
            "\n"
            "    MOV   R2, #1\n"
            "1:  LDREX  R3, [R0]             // Load-Exclusive: read lock.\n"
            "    CMP    R3, #0               // Free?\n"
            "    BNE    1b                   // If locked, spin (add WFE for power saving).\n"
            "    STREX  R3, R2, [R0]         // Try to set lock=1.\n"
            "    CMP    R3, #0\n"
            "    BNE    1b                   // Store failed -> retry.\n"
            "    DMB    ISH                  // Acquire barrier: order subsequent accesses\n"
            "                                 //   after lock acquisition."
        ),
    },
    "spinlock_release": {
        "aarch64": (
            "// === AArch64 Spinlock Release ===\n"
            "// X0 = lock address.\n"
            "\n"
            "    STLR  WZR, [X0]             // Store-Release zero (unlock).\n"
            "                                 //   Release ordering: all prior critical section\n"
            "                                 //   accesses visible before lock appears free.\n"
            "                                 //   Also wakes WFE spinners.\n"
            "\n"
            "// Alternative without STLR:\n"
            "//     DMB  ISH                 // Release barrier\n"
            "//     STR  WZR, [X0]           // Plain store: unlock\n"
            "//     SEV                      // Wake WFE spinners"
        ),
        "aarch32": (
            "// === AArch32 Spinlock Release ===\n"
            "// R0 = lock address.\n"
            "\n"
            "    MOV   R1, #0\n"
            "    DMB   ISH                   // Release barrier: all critical section\n"
            "                                 //   accesses complete before unlock.\n"
            "    STR   R1, [R0]              // Plain store: unlock.\n"
            "    SEV                          // Wake WFE spinners.\n"
            "\n"
            "// AArch32 has no STLR, so DMB+STR is the canonical pattern."
        ),
    },
    "context_switch": {
        "aarch64": (
            "// === AArch64 Simplified Context Switch ===\n"
            "// X0 = current task save area, X1 = next task save area.\n"
            "// Saves/restores callee-saved registers only (C convention).\n"
            "\n"
            "    // Save current task\n"
            "    MOV   X10, SP\n"
            "    STP   X19, X20, [X0, #0]\n"
            "    STP   X21, X22, [X0, #16]\n"
            "    STP   X23, X24, [X0, #32]\n"
            "    STP   X25, X26, [X0, #48]\n"
            "    STP   X27, X28, [X0, #64]\n"
            "    STP   X29, X30, [X0, #80]  // Save FP and LR.\n"
            "    STR   X10, [X0, #96]       // Save SP.\n"
            "\n"
            "    // Restore next task\n"
            "    LDP   X19, X20, [X1, #0]\n"
            "    LDP   X21, X22, [X1, #16]\n"
            "    LDP   X23, X24, [X1, #32]\n"
            "    LDP   X25, X26, [X1, #48]\n"
            "    LDP   X27, X28, [X1, #64]\n"
            "    LDP   X29, X30, [X1, #80]  // Restore FP and LR.\n"
            "    LDR   X10, [X1, #96]\n"
            "    MOV   SP, X10               // Restore SP.\n"
            "\n"
            "    RET                          // Resume next task via X30 (LR).\n"
            "\n"
            "// Caller-saved X0-X18 not saved (clobbered by C convention).\n"
            "// Real kernels also handle: FPSIMD, TPIDR_EL0, TTBR0_EL1."
        ),
        "aarch32": (
            "// === AArch32 Simplified Context Switch ===\n"
            "// R0 = current save area, R1 = next save area.\n"
            "\n"
            "    STMIA  R0, {R4-R11, SP, LR}  // Save callee-saved + SP + LR.\n"
            "    LDMIA  R1, {R4-R11, SP, LR}  // Restore from next task.\n"
            "    BX     LR                     // Resume next task."
        ),
    },
    "syscall": {
        "aarch64": (
            "// === AArch64 System Call (from userspace) ===\n"
            "// Linux: X8 = syscall number, X0-X5 = args, return in X0.\n"
            "// Example: write(1, msg, 5) = syscall #64\n"
            "\n"
            "    MOV   X0, #1                // fd = stdout\n"
            "    LDR   X1, =message          // buf = string address\n"
            "    MOV   X2, #5                // count = 5\n"
            "    MOV   X8, #64               // __NR_write = 64\n"
            "    SVC   #0                    // Supervisor Call -> EL1.\n"
            "                                 //   PSTATE saved to SPSR_EL1\n"
            "                                 //   Return address saved to ELR_EL1\n"
            "                                 //   PC -> VBAR_EL1 + 0x400 (EL0 Sync)\n"
            "    // X0 = return value after ERET\n"
            "\n"
            "// Kernel side: saves pt_regs, dispatches via X8, ERET back to EL0."
        ),
        "aarch32": (
            "// === AArch32 System Call (from userspace) ===\n"
            "// Linux EABI: R7 = syscall number, R0-R5 = args, return in R0.\n"
            "// Example: write(1, msg, 5) = syscall #4\n"
            "\n"
            "    MOV   R0, #1                // fd = stdout\n"
            "    LDR   R1, =message          // buf\n"
            "    MOV   R2, #5                // count\n"
            "    MOV   R7, #4                // __NR_write = 4 (EABI)\n"
            "    SVC   #0                    // -> SVC mode, vector 0x08.\n"
            "                                 //   CPSR -> SPSR_SVC, PC -> LR_SVC\n"
            "    // R0 = result after return"
        ),
    },
    "tlb_invalidate": {
        "aarch64": (
            "// === AArch64 TLB Invalidation Sequence ===\n"
            "// Required after modifying page tables. Order: TLBI -> DSB -> ISB.\n"
            "\n"
            "    TLBI  VMALLE1IS             // Invalidate all TLB entries at EL1, Inner Shareable.\n"
            "                                 //   Broadcasts to all PEs in IS domain.\n"
            "                                 //   Variants: VMALLE1 (local), VAE1IS (by VA),\n"
            "                                 //   VALE1IS (last-level), VMALLS12E1IS (stage 1+2)\n"
            "\n"
            "    DSB   ISH                   // Wait for TLBI to COMPLETE on all IS PEs.\n"
            "\n"
            "    ISB                          // Flush pipeline: next fetch uses new translations.\n"
            "\n"
            "// Single-address invalidation:\n"
            "//     LSR   X1, X0, #12        // Form TLBI operand (VA >> 12)\n"
            "//     TLBI  VAE1IS, X1\n"
            "//     DSB   ISH\n"
            "//     ISB\n"
            "\n"
            "// ORDERING IS MANDATORY: TLBI -> DSB -> ISB.\n"
            "//   TLBI without DSB: other PEs may not see the invalidation.\n"
            "//   DSB without ISB: pipeline may hold stale translations."
        ),
        "aarch32": (
            "// === AArch32 TLB Invalidation Sequence ===\n"
            "// Uses CP15 coprocessor. Privileged mode required.\n"
            "\n"
            "    MCR   p15, 0, R0, c8, c3, 0  // TLBIALLIS: Invalidate all, IS.\n"
            "    DSB   ISH                     // Wait for completion.\n"
            "    ISB                            // Flush pipeline.\n"
            "\n"
            "// By MVA: MCR p15, 0, R0, c8, c3, 1  (TLBIMVAIS)"
        ),
    },
    "cache_clean": {
        "aarch64": (
            "// === AArch64 Data Cache Clean by Virtual Address ===\n"
            "// Writes back dirty cache line at VA in X0 to memory.\n"
            "// Essential before DMA: device reads from main memory.\n"
            "\n"
            "    DC    CVAC, X0              // Clean by VA to Point of Coherency.\n"
            "                                 //   PoC = where all agents see same data.\n"
            "                                 //   Writes back dirty line; keeps it in cache.\n"
            "    DSB   ISH                   // Ensure clean completes before proceeding.\n"
            "\n"
            "// Other DC operations:\n"
            "//   DC CIVAC, X0  -- Clean AND Invalidate (flush + discard)\n"
            "//   DC IVAC, X0   -- Invalidate only (DANGEROUS: loses dirty data)\n"
            "//   DC CVAU, X0   -- Clean to PoU (for I/D coherency, JIT)\n"
            "\n"
            "// Range clean: loop with CTR_EL0.DminLine for cache line size."
        ),
        "aarch32": (
            "// === AArch32 Data Cache Clean by Virtual Address ===\n"
            "// R0 = virtual address to clean.\n"
            "\n"
            "    MCR   p15, 0, R0, c7, c10, 1  // DCCMVAC: Clean by MVA to PoC.\n"
            "    DSB   ISH                      // Ensure clean completes.\n"
            "\n"
            "// Also: c7,c14,1 = Clean+Invalidate; c7,c6,1 = Invalidate only;\n"
            "//        c7,c11,1 = Clean to PoU"
        ),
    },
    "enable_mmu": {
        "aarch64": (
            "// === AArch64 Basic MMU Enable Sequence ===\n"
            "// X0 = page table base (TTBR0). Running at EL1, MMU off.\n"
            "// Code MUST be in identity-mapped region (VA == PA).\n"
            "\n"
            "    // Step 1: MAIR_EL1 (memory attribute types)\n"
            "    MOV   X1, #0xFF             // Attr0 = Normal WB-RWA\n"
            "    ORR   X1, X1, #(0x04 << 8) // Attr1 = Device-nGnRE\n"
            "    MSR   MAIR_EL1, X1\n"
            "\n"
            "    // Step 2: TCR_EL1 (translation control)\n"
            "    MOV   X1, #0x19             // T0SZ=25 (512GB VA)\n"
            "    ORR   X1, X1, #(0x1 << 8)  // IRGN0=WB-WA\n"
            "    ORR   X1, X1, #(0x1 << 10) // ORGN0=WB-WA\n"
            "    ORR   X1, X1, #(0x3 << 12) // SH0=Inner Shareable\n"
            "    ORR   X1, X1, #(0x1 << 14) // TG0=64KB granule\n"
            "    MSR   TCR_EL1, X1\n"
            "\n"
            "    // Step 3: TTBR0_EL1\n"
            "    MSR   TTBR0_EL1, X0\n"
            "\n"
            "    // Step 4: Barrier before enable\n"
            "    ISB                          // System reg writes visible to MMU.\n"
            "\n"
            "    // Step 5: Enable MMU\n"
            "    MRS   X1, SCTLR_EL1\n"
            "    ORR   X1, X1, #0x1         // M bit: MMU on\n"
            "    ORR   X1, X1, #(0x1 << 2)  // C bit: D-cache on\n"
            "    ORR   X1, X1, #(0x1 << 12) // I bit: I-cache on\n"
            "    MSR   SCTLR_EL1, X1\n"
            "\n"
            "    ISB                          // CRITICAL: flush pipeline for MMU."
        ),
        "aarch32": (
            "// === AArch32 Basic MMU Enable Sequence ===\n"
            "// R0 = L1 page table physical address. PL1 mode.\n"
            "\n"
            "    ORR   R0, R0, #0x6A        // TTBR0 cacheability bits\n"
            "    MCR   p15, 0, R0, c2, c0, 0  // Write TTBR0\n"
            "    MOV   R1, #0x3\n"
            "    MCR   p15, 0, R1, c3, c0, 0  // DACR: Domain 0 = Manager\n"
            "    MOV   R1, #0\n"
            "    MCR   p15, 0, R1, c8, c7, 0  // TLBIALL\n"
            "    MCR   p15, 0, R1, c7, c5, 0  // ICIALLU\n"
            "    DSB\n"
            "    ISB\n"
            "    MRC   p15, 0, R1, c1, c0, 0  // Read SCTLR\n"
            "    ORR   R1, R1, #0x1            // M bit: MMU on\n"
            "    ORR   R1, R1, #(1 << 2)      // C bit: D-cache\n"
            "    ORR   R1, R1, #(1 << 12)     // I bit: I-cache\n"
            "    MCR   p15, 0, R1, c1, c0, 0  // Write SCTLR\n"
            "    ISB                            // Flush pipeline"
        ),
    },
    "exception_vector": {
        "aarch64": (
            "// === AArch64 Exception Vector Table ===\n"
            "// VBAR_EL1 points here. Must be 2KB aligned.\n"
            "// 4 groups x 4 entries = 16 entries, each 0x80 (128) bytes.\n"
            "//\n"
            "// +0x000 Current EL, SP_EL0: Sync/IRQ/FIQ/SError\n"
            "// +0x200 Current EL, SP_ELx: Sync/IRQ/FIQ/SError\n"
            "// +0x400 Lower EL, AArch64:  Sync/IRQ/FIQ/SError\n"
            "// +0x600 Lower EL, AArch32:  Sync/IRQ/FIQ/SError\n"
            "\n"
            "    .balign 0x800\n"
            "vector_table:\n"
            "    // ... groups 1-2 ...\n"
            "\n"
            "    .balign 0x80                // +0x400: EL0 AArch64 Sync\n"
            "el0_sync:\n"
            "    STP   X0, X1, [SP, #-16]!  // Quick-save on kernel stack.\n"
            "    MRS   X0, ESR_EL1           // Exception Syndrome: EC[31:26] = cause\n"
            "                                 //   0x15=SVC, 0x24=Data Abort, 0x20=Inst Abort\n"
            "    MRS   X1, ELR_EL1           // Return address.\n"
            "    B     full_exception_handler\n"
            "\n"
            "// Setup: ADR X0, vector_table / MSR VBAR_EL1, X0 / ISB"
        ),
        "aarch32": (
            "// === AArch32 Exception Vector Table ===\n"
            "// At 0x00000000 (or 0xFFFF0000 with SCTLR.V=1).\n"
            "// Each entry is one 4-byte branch.\n"
            "//\n"
            "// 0x00 Reset (SVC)    0x04 Undef (UND)\n"
            "// 0x08 SVC (SVC)      0x0C Prefetch Abort (ABT)\n"
            "// 0x10 Data Abort (ABT) 0x14 Reserved\n"
            "// 0x18 IRQ (IRQ)      0x1C FIQ (FIQ)\n"
            "\n"
            "    .balign 32\n"
            "vectors:\n"
            "    B     reset_handler          // 0x00\n"
            "    B     undef_handler          // 0x04\n"
            "    B     svc_handler            // 0x08\n"
            "    B     prefetch_abort_handler // 0x0C\n"
            "    B     data_abort_handler     // 0x10\n"
            "    B     .                      // 0x14 Reserved\n"
            "    B     irq_handler            // 0x18\n"
            "    B     fiq_handler            // 0x1C (banked R8-R12)"
        ),
    },
}

_PATTERN_NAMES_HELP = (
    "function_prologue, function_epilogue, atomic_add, atomic_cas, "
    "spinlock_acquire, spinlock_release, context_switch, syscall, "
    "tlb_invalidate, cache_clean, enable_mmu, exception_vector"
)


@mcp.tool()
def show_assembly_pattern(pattern_name: str, architecture: str = "aarch64") -> str:
    """Show annotated ARM assembly code for a common programming pattern.

    Returns line-by-line commented assembly showing how to implement standard
    ARM patterns such as function prologues, atomic operations, spinlocks,
    system calls, TLB/cache maintenance, MMU enable, and exception vectors.

    Args:
        pattern_name: The pattern to show. One of:
            function_prologue, function_epilogue, atomic_add, atomic_cas,
            spinlock_acquire, spinlock_release, context_switch, syscall,
            tlb_invalidate, cache_clean, enable_mmu, exception_vector.
        architecture: "aarch32" or "aarch64" (default: "aarch64").
    """
    arch = architecture.strip().lower()
    if arch not in ("aarch32", "aarch64"):
        return "Error: architecture must be 'aarch32' or 'aarch64'."

    key = pattern_name.strip().lower()

    if key not in _ASSEMBLY_PATTERNS:
        return (
            f"Error: unknown pattern '{pattern_name}'.\n\n"
            f"Available patterns: {_PATTERN_NAMES_HELP}\n\n"
            "Use architecture='aarch64' (default) or 'aarch32' for the ARM/Thumb variant."
        )

    pattern_data = _ASSEMBLY_PATTERNS[key]

    if arch not in pattern_data:
        available = ", ".join(sorted(pattern_data.keys()))
        return (
            f"Pattern '{key}' is not available for {arch}.\n"
            f"Available architectures for this pattern: {available}."
        )

    code = pattern_data[arch]

    lines: list[str] = []
    lines.append(f"# Assembly Pattern: {key} ({arch})")
    lines.append("")
    lines.append("```asm")
    lines.append(code.rstrip())
    lines.append("```")

    if arch == "aarch64" and "aarch32" in pattern_data:
        lines.append("")
        lines.append("_Tip: AArch32 variant also available -- call with architecture='aarch32'._")
    elif arch == "aarch32" and "aarch64" in pattern_data:
        lines.append("")
        lines.append("_Tip: AArch64 variant also available -- call with architecture='aarch64'._")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: explain_barrier -- ARM barrier and synchronization reference
# ---------------------------------------------------------------------------

_BARRIER_DATA: dict[str, dict] = {
    "DMB": {
        "full_name": "Data Memory Barrier",
        "encoding": "DMB <option>  (AArch64/AArch32)",
        "description": (
            "DMB ensures that all explicit memory accesses that appear in program order "
            "before the DMB are observed (by other PEs in the specified shareability domain) "
            "before any explicit memory accesses that appear after the DMB.\n\n"
            "CRITICAL: DMB only orders memory accesses relative to each other. It does NOT:\n"
            "  - Wait for memory accesses to complete (use DSB for that)\n"
            "  - Flush the pipeline (use ISB for that)\n"
            "  - Order instruction fetches or cache/TLB maintenance (use DSB+ISB)\n\n"
            "DMB is lighter-weight than DSB. It guarantees ordering but the barrier itself "
            "may complete before prior accesses reach memory."
        ),
        "when_to_use": [
            "Between a flag/lock write and subsequent data reads.",
            "Implementing acquire/release when LDAR/STLR are not available.",
            "Producer-consumer: write data, DMB, write flag; read flag, DMB, read data.",
            "Ordering stores observed by DMA engines or other bus masters.",
            "Memory-mapped I/O: ordering a register write before a status read.",
        ],
        "domain_options": True,
        "pipeline_effect": "None -- DMB does not flush or stall the pipeline.",
    },
    "DSB": {
        "full_name": "Data Synchronization Barrier",
        "encoding": "DSB <option>  (AArch64/AArch32)",
        "description": (
            "DSB ensures that all explicit memory accesses, cache maintenance operations, "
            "and TLB maintenance operations before the DSB have COMPLETED before any "
            "instruction after the DSB executes.\n\n"
            "DSB is strictly stronger than DMB. Where DMB says 'observed in order', "
            "DSB says 'have finished'. No instruction after DSB executes until the "
            "barrier completes. More expensive but necessary for completion guarantees."
        ),
        "when_to_use": [
            "After TLBI: ensures invalidation is visible on all PEs.",
            "After cache maintenance (DC/IC): ensures operation reached PoC/PoU.",
            "Before ISB: when you need both completion and pipeline flush.",
            "Before WFI/WFE: ensures pending accesses complete before low-power state.",
            "After device register writes that must take effect before proceeding.",
            "Before SEV: ensures operations that SEV signals about have completed.",
        ],
        "domain_options": True,
        "pipeline_effect": "Stalls ALL subsequent instructions until barrier completes.",
    },
    "ISB": {
        "full_name": "Instruction Synchronization Barrier",
        "encoding": "ISB  (no domain options; always SY)",
        "description": (
            "ISB flushes the pipeline and ensures all subsequent instructions are fetched "
            "and decoded ANEW. Context-altering operations before ISB (system register "
            "writes, cache/TLB maintenance) are visible to subsequent instructions.\n\n"
            "ISB does NOT order memory accesses (use DMB/DSB). ISB orders the effects "
            "of system configuration changes on instruction execution.\n\n"
            "Key effects:\n"
            "  - Flushes instruction pipeline (prefetched/decoded instructions discarded)\n"
            "  - Subsequent instructions see new system register state\n"
            "  - Next instruction refetched using current translation and permissions"
        ),
        "when_to_use": [
            "After writing system registers (SCTLR, TCR, TTBR, VBAR, etc.).",
            "After TLBI + DSB: ensures next fetch uses updated TLB.",
            "After enabling/disabling MMU.",
            "After modifying VBAR.",
            "After IC IALLU: ensures new instructions fetched from memory.",
            "Self-modifying code / JIT: after D-cache clean + I-cache invalidate.",
            "After CPTR/CPACR changes enabling/disabling FP/SIMD.",
        ],
        "domain_options": False,
        "pipeline_effect": "Full pipeline flush. All subsequent instructions re-fetched.",
    },
    "LDAR": {
        "full_name": "Load-Acquire Register",
        "encoding": "LDAR Wt/Xt, [Xn|SP]  (AArch64; AArch32 uses LDAEX)",
        "description": (
            "LDAR performs a load with Acquire semantics: the loaded value is observed "
            "before ANY subsequent memory read or write in program order.\n\n"
            "Acquire = 'nothing after me can be reordered before me'.\n\n"
            "LDAR prevents: subsequent loads/stores from being reordered before it, "
            "and itself from being satisfied from a store buffer.\n\n"
            "Variants: LDARB (Byte), LDARH (Halfword), LDAXR (Acquire Exclusive), "
            "LDAXP (Acquire Exclusive Pair for 128-bit atomics)"
        ),
        "when_to_use": [
            "Reading a lock/flag (spinlock acquire).",
            "Load side of CAS: LDAXR = acquire + exclusive.",
            "Consumer side of producer-consumer.",
            "Reading sequence counters (seqlock).",
            "Any load needing 'subsequent accesses see memory from this point'.",
        ],
        "domain_options": False,
        "pipeline_effect": "No pipeline flush. May stall subsequent memory ops.",
    },
    "STLR": {
        "full_name": "Store-Release Register",
        "encoding": "STLR Wt/Xt, [Xn|SP]  (AArch64; AArch32 uses DMB+STR)",
        "description": (
            "STLR performs a store with Release semantics: all prior memory accesses "
            "are observed before the STLR store becomes visible.\n\n"
            "Release = 'nothing before me can be reordered after me'.\n\n"
            "LDAR + STLR = acquire-release model:\n"
            "  Thread A: write data; STLR flag=1  (release)\n"
            "  Thread B: LDAR flag; read data      (acquire)\n"
            "  B sees A's data.\n\n"
            "Variants: STLRB, STLRH, STLXR (Release Exclusive), STLXP"
        ),
        "when_to_use": [
            "Releasing a spinlock.",
            "Store side of CAS: STLXR = release + exclusive.",
            "Producer: write data, STLR flag.",
            "Publishing a pointer: write struct, STLR pointer.",
            "Any store needing 'all prior ops visible before this store'.",
        ],
        "domain_options": False,
        "pipeline_effect": "No pipeline flush. May delay until prior ops complete.",
    },
    "LDAPR": {
        "full_name": "Load-Acquire RCpc Register (ARMv8.3+)",
        "encoding": "LDAPR Wt/Xt, [Xn|SP]  (AArch64, FEAT_LRCPC)",
        "description": (
            "LDAPR provides RCpc (Release-Consistent processor-consistent) acquire, "
            "weaker than LDAR's sequential-consistency acquire.\n\n"
            "Key difference: STLR followed by LDAPR to a DIFFERENT address can be "
            "reordered. LDAR would prevent this.\n\n"
            "LDAPR + STLR = C++ memory_order_acquire/release.\n"
            "LDAR + STLR = C++ memory_order_seq_cst.\n\n"
            "LDAPR is cheaper on some implementations. "
            "Variants: LDAPRB (Byte), LDAPRH (Halfword)"
        ),
        "when_to_use": [
            "C++ memory_order_acquire without seq_cst overhead.",
            "Lock-free structures needing only acquire-release.",
            "Reading published pointers (release-acquire sufficient).",
            "Performance-sensitive code where LDAR is a bottleneck.",
        ],
        "domain_options": False,
        "pipeline_effect": "Lighter than LDAR on some microarchitectures.",
    },
    "SB": {
        "full_name": "Speculation Barrier (ARMv8.5+, FEAT_SB)",
        "encoding": "SB  (no operands)",
        "description": (
            "SB prevents speculative execution of any instructions after the SB until "
            "the SB is known to be on the architecturally executed path.\n\n"
            "Unlike ISB (unconditional pipeline flush), SB specifically targets "
            "speculation. No instruction after SB executes speculatively until all "
            "prior branches resolve. Blocks Spectre-v1 attacks.\n\n"
            "Lighter-weight than ISB: no refetching, only prevents speculation."
        ),
        "when_to_use": [
            "Spectre-v1 mitigation: after bounds-checking conditional branch.",
            "After a branch gating sensitive data access.",
            "Replacing CSDB patterns when FEAT_SB available.",
            "Kernel: after checking user-supplied indices.",
        ],
        "domain_options": False,
        "pipeline_effect": "Prevents speculative execution. Does NOT flush pipeline.",
    },
    "CSDB": {
        "full_name": "Consumption of Speculative Data Barrier",
        "encoding": "CSDB  (hint-space, all ARMv8)",
        "description": (
            "CSDB ensures conditional-select/compare results before the CSDB cannot "
            "speculatively use data from speculative memory reads.\n\n"
            "Primary Spectre-v1 mitigation on ARM. Pattern:\n"
            "  1. CSEL to zero index if out-of-bounds\n"
            "  2. CSDB\n"
            "  3. Memory access with (now safe) index\n\n"
            "After CSDB, conditionally selected values are architecturally correct, "
            "not speculatively forwarded. Does NOT prevent all speculation."
        ),
        "when_to_use": [
            "Spectre-v1: CMP+CSEL+CSDB+LDR to mask out-of-bounds index.",
            "Example: CMP x0, x1 / CSEL x0, x0, xzr, LT / CSDB / LDR x2, [x3, x0]",
            "When SB (FEAT_SB) is not available (CSDB works on all ARMv8).",
            "Kernel array accesses gated by user-controlled indices.",
        ],
        "domain_options": False,
        "pipeline_effect": "No pipeline flush. Only prevents speculative use of conditional data.",
    },
    "SSBB": {
        "full_name": "Speculative Store Bypass Barrier",
        "encoding": "SSBB  (alias for DSB #0; all ARMv8)",
        "description": (
            "SSBB prevents speculative loads after SSBB from bypassing stores before "
            "SSBB when accessing the same virtual address.\n\n"
            "Addresses Spectre-v4: CPU speculatively loads a value before a prior store "
            "to the same address commits, reading stale data.\n\n"
            "After SSBB, loads see the most recent store to the same VA, even "
            "speculatively. Scope is local to the PE."
        ),
        "when_to_use": [
            "Spectre-v4: store then load at same address could leak data.",
            "Sandboxed/JIT code: prevent reading stale overwritten data.",
            "Between security-relevant store and subsequent load at same address.",
            "More targeted than full DSB.",
        ],
        "domain_options": False,
        "pipeline_effect": "Minimal. Only prevents store bypass for same-VA pairs.",
    },
    "PSSBB": {
        "full_name": "Physical Speculative Store Bypass Barrier",
        "encoding": "PSSBB  (alias for DSB #4; all ARMv8)",
        "description": (
            "PSSBB is like SSBB but for physical addresses. Prevents speculative loads "
            "from bypassing stores when they map to the same PHYSICAL address, even with "
            "different virtual addresses.\n\n"
            "Needed because VA aliasing (two VAs -> same PA) can bypass SSBB. "
            "PSSBB is strictly stronger than SSBB."
        ),
        "when_to_use": [
            "VA aliasing possible (shared memory at different VAs).",
            "Hypervisor/kernel with guest VA aliasing host PA.",
            "Spectre-v4 when SSBB insufficient due to VA aliasing.",
            "Shared memory IPC with same PA at different VAs.",
        ],
        "domain_options": False,
        "pipeline_effect": "Similar to SSBB. Minimal overhead.",
    },
}

_DMB_DSB_DOMAIN_OPTIONS: dict[str, str] = {
    "SY": "Full System (default). Strongest and most common.",
    "ST": "Full System, Store only. Orders store-store only.",
    "LD": "Full System, Load only (ARMv8.1+). Orders load-load and load-store.",
    "ISH": "Inner Shareable. All cache-coherent PEs. Most common for SMP.",
    "ISHST": "Inner Shareable, Store only.",
    "ISHLD": "Inner Shareable, Load only (ARMv8.1+).",
    "NSH": "Non-shareable. Local PE only.",
    "NSHST": "Non-shareable, Store only.",
    "NSHLD": "Non-shareable, Load only (ARMv8.1+).",
    "OSH": "Outer Shareable. Full system including GPUs/DMA. Stronger than ISH.",
    "OSHST": "Outer Shareable, Store only.",
    "OSHLD": "Outer Shareable, Load only (ARMv8.1+).",
}


def _format_barrier(barrier_type: str, data: dict) -> str:
    """Format a single barrier entry as readable text."""
    lines: list[str] = []
    lines.append(f"# {data['full_name']} ({barrier_type})")
    lines.append(f"Encoding: {data['encoding']}")
    lines.append("")
    lines.append("## What It Does")
    lines.append(data["description"])
    lines.append("")
    lines.append(f"**Pipeline effect:** {data['pipeline_effect']}")
    lines.append("")
    lines.append("## When To Use It")
    for i, use in enumerate(data["when_to_use"], 1):
        lines.append(f"  {i}. {use}")
    if data.get("domain_options"):
        lines.append("")
        lines.append("## Domain / Shareability Options")
        lines.append(f"Syntax: {barrier_type} <option>  (e.g., {barrier_type} ISH)")
        lines.append("")
        for opt, desc in _DMB_DSB_DOMAIN_OPTIONS.items():
            lines.append(f"  **{opt}**: {desc}")
    return "\n".join(lines)


def _format_barrier_overview() -> str:
    """Format the overview/summary table of all barriers."""
    lines: list[str] = []
    lines.append("# ARM Barrier Instructions -- Overview")
    lines.append("")
    lines.append("## Quick Reference Table")
    lines.append("")
    lines.append(f"{'Instruction':<10} {'Full Name':<48} Key Use Case")
    lines.append("-" * 110)
    for instr, name, use in [
        ("DMB", "Data Memory Barrier", "Order memory accesses (not completion). Lightweight."),
        ("DSB", "Data Synchronization Barrier", "Wait for ops to COMPLETE. Before ISB."),
        ("ISB", "Instruction Synchronization Barrier", "Flush pipeline. After sysreg changes."),
        ("LDAR", "Load-Acquire Register", "Load with acquire semantics."),
        ("STLR", "Store-Release Register", "Store with release semantics."),
        ("LDAPR", "Load-Acquire RCpc Register", "Weaker acquire (RCpc). ARMv8.3+."),
        ("SB", "Speculation Barrier", "Block all speculation. ARMv8.5+."),
        ("CSDB", "Consumption of Speculative Data Barrier", "Spectre-v1: resolve conditional data."),
        ("SSBB", "Speculative Store Bypass Barrier", "Spectre-v4: same VA store bypass."),
        ("PSSBB", "Physical Speculative Store Bypass Barrier", "Spectre-v4: same PA store bypass."),
    ]:
        lines.append(f"  {instr:<8} {name:<48} {use}")
    lines.append("")
    lines.append("## DMB vs DSB vs ISB")
    lines.append("")
    lines.append("### DMB: orders memory accesses. No completion wait, no pipeline flush. Cheapest.")
    lines.append("### DSB: waits for ALL prior memory/cache/TLB ops to COMPLETE. Stalls all instructions.")
    lines.append("### ISB: flushes pipeline. Refetches instructions. After system register changes.")
    lines.append("")
    lines.append("### Combined: TLBI -> DSB ISH -> ISB (mandatory order for TLB invalidation)")
    lines.append("")
    lines.append("## Acquire-Release vs Explicit Barriers")
    lines.append("")
    lines.append("  LDAR + STLR = sequential consistency (C++ seq_cst)")
    lines.append("  LDAPR + STLR = acquire-release (C++ acq_rel) -- weaker, cheaper")
    lines.append("  DMB = explicit barrier affecting ALL accesses (needed on AArch32)")
    lines.append("")
    lines.append("## Speculation Barriers")
    lines.append("")
    lines.append("  Spectre-v1: SB (ARMv8.5+) or CSDB+CSEL (all ARMv8)")
    lines.append("  Spectre-v4: SSBB (same VA) or PSSBB (same PA, handles aliasing)")
    lines.append("")
    lines.append("## Domain Hierarchy (DMB/DSB)")
    lines.append("")
    lines.append("  OSH > ISH > NSH  (Outer > Inner > Non-shareable)")
    lines.append("  SY > LD > ST     (All > Load-only > Store-only)")
    lines.append("  Typical: ISH for SMP, OSH for DMA/GPU, NSH for local device regs")
    return "\n".join(lines)


@mcp.tool()
def explain_barrier(barrier_type: str) -> str:
    """Explain an ARM barrier or synchronization instruction in detail.

    Covers what the barrier does, when to use it, domain options (for DMB/DSB),
    the differences between barrier types, acquire/release semantics, and
    speculation barriers for Spectre mitigation.

    Args:
        barrier_type: One of:
            "DMB" -- Data Memory Barrier (ordering only)
            "DSB" -- Data Synchronization Barrier (completion)
            "ISB" -- Instruction Synchronization Barrier (pipeline flush)
            "LDAR" -- Load-Acquire Register
            "STLR" -- Store-Release Register
            "LDAPR" -- Load-Acquire RCpc Register (ARMv8.3+)
            "SB" -- Speculation Barrier (ARMv8.5+)
            "CSDB" -- Consumption of Speculative Data Barrier
            "SSBB" -- Speculative Store Bypass Barrier
            "PSSBB" -- Physical Speculative Store Bypass Barrier
            "overview" -- Summary table of ALL barriers with comparison
    """
    key = barrier_type.strip().upper()

    if key == "OVERVIEW":
        return _format_barrier_overview()

    if key not in _BARRIER_DATA:
        valid = ", ".join(sorted(_BARRIER_DATA.keys()))
        return (
            f"Error: '{barrier_type}' is not a recognised barrier type.\n\n"
            f"Valid barrier types: {valid}\n"
            f"Use 'overview' for a summary comparison of all barriers."
        )

    return _format_barrier(key, _BARRIER_DATA[key])


# ---- NEON / Advanced SIMD Intrinsic Data ----

_NEON_INTRINSICS: dict[str, dict] = {
    # --- Arithmetic (float) ---
    "vaddq_f32": {
        "signature": "float32x4_t vaddq_f32(float32x4_t a, float32x4_t b)",
        "instruction": "FADD Vd.4S, Vn.4S, Vm.4S",
        "description": "Add two 128-bit vectors of four 32-bit floats element-wise.",
        "data_types": ["float32x4_t"],
        "category": "Arithmetic",
        "architecture": "AArch32/AArch64",
        "latency": "2-4 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = vld1q_f32(ptr_a);\n'
            'float32x4_t b = vld1q_f32(ptr_b);\n'
            'float32x4_t sum = vaddq_f32(a, b);\n'
            'vst1q_f32(out, sum);'
        ),
    },
    "vsubq_f32": {
        "signature": "float32x4_t vsubq_f32(float32x4_t a, float32x4_t b)",
        "instruction": "FSUB Vd.4S, Vn.4S, Vm.4S",
        "description": "Subtract two 128-bit vectors of four 32-bit floats element-wise (a - b).",
        "data_types": ["float32x4_t"],
        "category": "Arithmetic",
        "architecture": "AArch32/AArch64",
        "latency": "2-4 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = vld1q_f32(ptr_a);\n'
            'float32x4_t b = vld1q_f32(ptr_b);\n'
            'float32x4_t diff = vsubq_f32(a, b);\n'
            'vst1q_f32(out, diff);'
        ),
    },
    "vmulq_f32": {
        "signature": "float32x4_t vmulq_f32(float32x4_t a, float32x4_t b)",
        "instruction": "FMUL Vd.4S, Vn.4S, Vm.4S",
        "description": "Multiply two 128-bit vectors of four 32-bit floats element-wise.",
        "data_types": ["float32x4_t"],
        "category": "Arithmetic",
        "architecture": "AArch32/AArch64",
        "latency": "3-5 cycles (Cortex-A78 class)",
        "throughput": "1-2 per cycle",
        "example": (
            'float32x4_t a = vld1q_f32(ptr_a);\n'
            'float32x4_t b = vld1q_f32(ptr_b);\n'
            'float32x4_t prod = vmulq_f32(a, b);\n'
            'vst1q_f32(out, prod);'
        ),
    },
    "vmlaq_f32": {
        "signature": "float32x4_t vmlaq_f32(float32x4_t a, float32x4_t b, float32x4_t c)",
        "instruction": "FMLA Vd.4S, Vn.4S, Vm.4S",
        "description": (
            "Fused multiply-accumulate: a + b * c. Computes the product of b and c, "
            "adds a, and returns the result with a single rounding step."
        ),
        "data_types": ["float32x4_t"],
        "category": "Arithmetic",
        "architecture": "AArch32/AArch64",
        "latency": "4-6 cycles (Cortex-A78 class)",
        "throughput": "1-2 per cycle",
        "example": (
            'float32x4_t acc = vld1q_f32(ptr_acc);\n'
            'float32x4_t b = vld1q_f32(ptr_b);\n'
            'float32x4_t c = vld1q_f32(ptr_c);\n'
            'acc = vmlaq_f32(acc, b, c);  // acc += b * c\n'
            'vst1q_f32(out, acc);'
        ),
    },
    "vfmaq_f32": {
        "signature": "float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c)",
        "instruction": "FMLA Vd.4S, Vn.4S, Vm.4S",
        "description": (
            "Fused multiply-accumulate: a + b * c. This is the preferred FMA intrinsic "
            "on AArch64 (ARMv8.0+). Unlike vmlaq_f32, this intrinsic is guaranteed to "
            "emit a true fused multiply-add instruction with a single rounding step, "
            "providing both better accuracy and better performance. Critical for AI/ML "
            "inference kernels, GEMM, and convolution inner loops."
        ),
        "data_types": ["float32x4_t"],
        "category": "Arithmetic",
        "architecture": "AArch64",
        "latency": "4 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            '// Dot-product accumulation pattern\n'
            'float32x4_t acc = vdupq_n_f32(0.0f);\n'
            'for (int i = 0; i < N; i += 4) {\n'
            '    float32x4_t a = vld1q_f32(&src_a[i]);\n'
            '    float32x4_t b = vld1q_f32(&src_b[i]);\n'
            '    acc = vfmaq_f32(acc, a, b);  // acc += a * b\n'
            '}\n'
            'float32_t result = vaddvq_f32(acc);  // horizontal sum'
        ),
    },
    "vfmsq_f32": {
        "signature": "float32x4_t vfmsq_f32(float32x4_t a, float32x4_t b, float32x4_t c)",
        "instruction": "FMLS Vd.4S, Vn.4S, Vm.4S",
        "description": (
            "Fused multiply-subtract: a - b * c. Computes the product of b and c, "
            "subtracts from a, with a single rounding step. Useful for residual "
            "computations and iterative refinement."
        ),
        "data_types": ["float32x4_t"],
        "category": "Arithmetic",
        "architecture": "AArch64",
        "latency": "4 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t acc = vld1q_f32(ptr_acc);\n'
            'float32x4_t b = vld1q_f32(ptr_b);\n'
            'float32x4_t c = vld1q_f32(ptr_c);\n'
            'acc = vfmsq_f32(acc, b, c);  // acc -= b * c'
        ),
    },
    "vfmaq_laneq_f32": {
        "signature": "float32x4_t vfmaq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v, const int lane)",
        "instruction": "FMLA Vd.4S, Vn.4S, Vm.S[lane]",
        "description": (
            "Fused multiply-accumulate by lane: a + b * v[lane]. Multiplies each element "
            "of b by a single element (lane) of v, and adds to a. Essential for GEMM "
            "kernels where one operand is broadcast from a specific element."
        ),
        "data_types": ["float32x4_t"],
        "category": "Arithmetic",
        "architecture": "AArch64",
        "latency": "4 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            '// Matrix multiply inner loop: C[row] += A[row] * B[k][col]\n'
            'float32x4_t b_row = vld1q_f32(&B[k*N + j]);\n'
            'c0 = vfmaq_laneq_f32(c0, b_row, a_col, 0);\n'
            'c1 = vfmaq_laneq_f32(c1, b_row, a_col, 1);\n'
            'c2 = vfmaq_laneq_f32(c2, b_row, a_col, 2);\n'
            'c3 = vfmaq_laneq_f32(c3, b_row, a_col, 3);'
        ),
    },
    "vabdq_f32": {
        "signature": "float32x4_t vabdq_f32(float32x4_t a, float32x4_t b)",
        "instruction": "FABD Vd.4S, Vn.4S, Vm.4S",
        "description": "Compute the absolute difference |a - b| for each element of two float32 vectors.",
        "data_types": ["float32x4_t"],
        "category": "Arithmetic",
        "architecture": "AArch32/AArch64",
        "latency": "2-4 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = vld1q_f32(ptr_a);\n'
            'float32x4_t b = vld1q_f32(ptr_b);\n'
            'float32x4_t diff = vabdq_f32(a, b);  // |a - b|\n'
            'vst1q_f32(out, diff);'
        ),
    },
    "vnegq_f32": {
        "signature": "float32x4_t vnegq_f32(float32x4_t a)",
        "instruction": "FNEG Vd.4S, Vn.4S",
        "description": "Negate each element of a 128-bit vector of four 32-bit floats.",
        "data_types": ["float32x4_t"],
        "category": "Arithmetic",
        "architecture": "AArch32/AArch64",
        "latency": "2 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = vld1q_f32(ptr_a);\n'
            'float32x4_t neg = vnegq_f32(a);  // -a\n'
            'vst1q_f32(out, neg);'
        ),
    },
    "vabsq_f32": {
        "signature": "float32x4_t vabsq_f32(float32x4_t a)",
        "instruction": "FABS Vd.4S, Vn.4S",
        "description": "Compute the absolute value of each element of a 128-bit vector of four 32-bit floats.",
        "data_types": ["float32x4_t"],
        "category": "Arithmetic",
        "architecture": "AArch32/AArch64",
        "latency": "2 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = vld1q_f32(ptr_a);\n'
            'float32x4_t abs_a = vabsq_f32(a);  // |a|\n'
            'vst1q_f32(out, abs_a);'
        ),
    },
    # --- Integer Arithmetic ---
    "vaddq_s32": {
        "signature": "int32x4_t vaddq_s32(int32x4_t a, int32x4_t b)",
        "instruction": "ADD Vd.4S, Vn.4S, Vm.4S",
        "description": "Add two 128-bit vectors of four signed 32-bit integers element-wise.",
        "data_types": ["int32x4_t"],
        "category": "Integer Arithmetic",
        "architecture": "AArch32/AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'int32x4_t a = vld1q_s32(ptr_a);\n'
            'int32x4_t b = vld1q_s32(ptr_b);\n'
            'int32x4_t sum = vaddq_s32(a, b);\n'
            'vst1q_s32(out, sum);'
        ),
    },
    "vmulq_s32": {
        "signature": "int32x4_t vmulq_s32(int32x4_t a, int32x4_t b)",
        "instruction": "MUL Vd.4S, Vn.4S, Vm.4S",
        "description": "Multiply two 128-bit vectors of four signed 32-bit integers element-wise (low 32 bits of result).",
        "data_types": ["int32x4_t"],
        "category": "Integer Arithmetic",
        "architecture": "AArch32/AArch64",
        "latency": "3-5 cycles (Cortex-A78 class)",
        "throughput": "1 per cycle",
        "example": (
            'int32x4_t a = vld1q_s32(ptr_a);\n'
            'int32x4_t b = vld1q_s32(ptr_b);\n'
            'int32x4_t prod = vmulq_s32(a, b);\n'
            'vst1q_s32(out, prod);'
        ),
    },
    "vqaddq_s16": {
        "signature": "int16x8_t vqaddq_s16(int16x8_t a, int16x8_t b)",
        "instruction": "SQADD Vd.8H, Vn.8H, Vm.8H",
        "description": (
            "Saturating add of two 128-bit vectors of eight signed 16-bit integers. "
            "Results are clamped to [INT16_MIN, INT16_MAX] instead of wrapping on overflow."
        ),
        "data_types": ["int16x8_t"],
        "category": "Integer Arithmetic",
        "architecture": "AArch32/AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'int16x8_t a = vld1q_s16(ptr_a);\n'
            'int16x8_t b = vld1q_s16(ptr_b);\n'
            'int16x8_t sum = vqaddq_s16(a, b);  // saturating add\n'
            'vst1q_s16(out, sum);'
        ),
    },
    "vqdmulhq_s16": {
        "signature": "int16x8_t vqdmulhq_s16(int16x8_t a, int16x8_t b)",
        "instruction": "SQDMULH Vd.8H, Vn.8H, Vm.8H",
        "description": (
            "Signed saturating doubling multiply returning high half. Multiplies each "
            "element pair, doubles the result, and returns the high 16 bits with saturation. "
            "Useful for fixed-point arithmetic."
        ),
        "data_types": ["int16x8_t"],
        "category": "Integer Arithmetic",
        "architecture": "AArch32/AArch64",
        "latency": "3-5 cycles (Cortex-A78 class)",
        "throughput": "1 per cycle",
        "example": (
            'int16x8_t a = vld1q_s16(ptr_a);\n'
            'int16x8_t b = vld1q_s16(ptr_b);\n'
            'int16x8_t result = vqdmulhq_s16(a, b);  // high half of 2*a*b\n'
            'vst1q_s16(out, result);'
        ),
    },
    # --- Load/Store ---
    "vld1q_f32": {
        "signature": "float32x4_t vld1q_f32(float32_t const *ptr)",
        "instruction": "LD1 {Vt.4S}, [Xn]",
        "description": (
            "Load four contiguous 32-bit floats from memory into a 128-bit NEON register. "
            "This is the most common NEON load intrinsic."
        ),
        "data_types": ["float32x4_t", "float32_t"],
        "category": "Load/Store",
        "architecture": "AArch32/AArch64",
        "latency": "4-5 cycles (L1 hit, Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};\n'
            'float32x4_t v = vld1q_f32(data);'
        ),
    },
    "vst1q_f32": {
        "signature": "void vst1q_f32(float32_t *ptr, float32x4_t val)",
        "instruction": "ST1 {Vt.4S}, [Xn]",
        "description": "Store a 128-bit vector of four 32-bit floats to four contiguous memory locations.",
        "data_types": ["float32x4_t", "float32_t"],
        "category": "Load/Store",
        "architecture": "AArch32/AArch64",
        "latency": "4-5 cycles (Cortex-A78 class)",
        "throughput": "1-2 per cycle",
        "example": (
            'float out[4];\n'
            'float32x4_t v = vmulq_f32(a, b);\n'
            'vst1q_f32(out, v);'
        ),
    },
    "vld1q_u8": {
        "signature": "uint8x16_t vld1q_u8(uint8_t const *ptr)",
        "instruction": "LD1 {Vt.16B}, [Xn]",
        "description": "Load sixteen contiguous unsigned 8-bit integers from memory into a 128-bit NEON register.",
        "data_types": ["uint8x16_t", "uint8_t"],
        "category": "Load/Store",
        "architecture": "AArch32/AArch64",
        "latency": "4-5 cycles (L1 hit, Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'uint8_t pixels[16];\n'
            'uint8x16_t v = vld1q_u8(pixels);'
        ),
    },
    "vld2q_f32": {
        "signature": "float32x4x2_t vld2q_f32(float32_t const *ptr)",
        "instruction": "LD2 {Vt.4S, Vt2.4S}, [Xn]",
        "description": (
            "Interleaved load: load eight consecutive floats from memory and de-interleave "
            "into two registers. Element 0,2,4,6 go to the first register; 1,3,5,7 go to "
            "the second. Useful for loading interleaved x/y coordinate pairs."
        ),
        "data_types": ["float32x4x2_t", "float32_t"],
        "category": "Load/Store",
        "architecture": "AArch32/AArch64",
        "latency": "6-8 cycles (L1 hit, Cortex-A78 class)",
        "throughput": "1 per cycle",
        "example": (
            '// Data: x0,y0, x1,y1, x2,y2, x3,y3\n'
            'float32x4x2_t xy = vld2q_f32(interleaved_ptr);\n'
            'float32x4_t x = xy.val[0];  // x0,x1,x2,x3\n'
            'float32x4_t y = xy.val[1];  // y0,y1,y2,y3'
        ),
    },
    "vld1q_dup_f32": {
        "signature": "float32x4_t vld1q_dup_f32(float32_t const *ptr)",
        "instruction": "LD1R {Vt.4S}, [Xn]",
        "description": (
            "Broadcast load: load a single 32-bit float from memory and replicate it "
            "into all four lanes of a 128-bit vector."
        ),
        "data_types": ["float32x4_t", "float32_t"],
        "category": "Load/Store",
        "architecture": "AArch32/AArch64",
        "latency": "4-5 cycles (L1 hit, Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float scale = 2.5f;\n'
            'float32x4_t v_scale = vld1q_dup_f32(&scale);  // {2.5, 2.5, 2.5, 2.5}\n'
            'float32x4_t result = vmulq_f32(data, v_scale);'
        ),
    },
    # --- Conversion ---
    "vcvtq_f32_s32": {
        "signature": "float32x4_t vcvtq_f32_s32(int32x4_t a)",
        "instruction": "SCVTF Vd.4S, Vn.4S",
        "description": "Convert a vector of four signed 32-bit integers to four 32-bit floats.",
        "data_types": ["float32x4_t", "int32x4_t"],
        "category": "Conversion",
        "architecture": "AArch32/AArch64",
        "latency": "3-4 cycles (Cortex-A78 class)",
        "throughput": "1-2 per cycle",
        "example": (
            'int32x4_t ints = vld1q_s32(int_data);\n'
            'float32x4_t floats = vcvtq_f32_s32(ints);\n'
            'vst1q_f32(float_out, floats);'
        ),
    },
    "vcvtq_s32_f32": {
        "signature": "int32x4_t vcvtq_s32_f32(float32x4_t a)",
        "instruction": "FCVTZS Vd.4S, Vn.4S",
        "description": "Convert a vector of four 32-bit floats to four signed 32-bit integers (round toward zero).",
        "data_types": ["int32x4_t", "float32x4_t"],
        "category": "Conversion",
        "architecture": "AArch32/AArch64",
        "latency": "3-4 cycles (Cortex-A78 class)",
        "throughput": "1-2 per cycle",
        "example": (
            'float32x4_t floats = vld1q_f32(float_data);\n'
            'int32x4_t ints = vcvtq_s32_f32(floats);\n'
            'vst1q_s32(int_out, ints);'
        ),
    },
    "vcvtq_f16_f32": {
        "signature": "float16x4_t vcvtq_f16_f32(float32x4_t a)",
        "instruction": "FCVTN Vd.4H, Vn.4S",
        "description": (
            "Narrow conversion: convert four 32-bit floats to four 16-bit floats (half precision). "
            "Useful for reducing memory bandwidth in ML inference."
        ),
        "data_types": ["float16x4_t", "float32x4_t"],
        "category": "Conversion",
        "architecture": "AArch32/AArch64",
        "latency": "3-4 cycles (Cortex-A78 class)",
        "throughput": "1-2 per cycle",
        "example": (
            'float32x4_t f32 = vld1q_f32(data);\n'
            'float16x4_t f16 = vcvtq_f16_f32(f32);\n'
            'vst1_f16(out, f16);'
        ),
    },
    "vmovl_s16": {
        "signature": "int32x4_t vmovl_s16(int16x4_t a)",
        "instruction": "SSHLL Vd.4S, Vn.4H, #0  (alias: SXTL)",
        "description": (
            "Widen (sign-extend): convert four signed 16-bit integers in the lower 64 bits "
            "to four signed 32-bit integers. Equivalent to SXTL (sign-extend long)."
        ),
        "data_types": ["int32x4_t", "int16x4_t"],
        "category": "Conversion",
        "architecture": "AArch32/AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'int16x4_t narrow = vld1_s16(data);  // 4 x int16\n'
            'int32x4_t wide = vmovl_s16(narrow);  // 4 x int32'
        ),
    },
    # --- Comparison ---
    "vceqq_f32": {
        "signature": "uint32x4_t vceqq_f32(float32x4_t a, float32x4_t b)",
        "instruction": "FCMEQ Vd.4S, Vn.4S, Vm.4S",
        "description": (
            "Compare equal: for each lane, set all bits to 1 (0xFFFFFFFF) if a == b, "
            "else all bits to 0. Result is a mask suitable for bitwise select."
        ),
        "data_types": ["uint32x4_t", "float32x4_t"],
        "category": "Comparison",
        "architecture": "AArch32/AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = vld1q_f32(ptr_a);\n'
            'float32x4_t b = vld1q_f32(ptr_b);\n'
            'uint32x4_t mask = vceqq_f32(a, b);  // lane-wise a == b\n'
            '// Use with vbslq to select elements'
        ),
    },
    "vcgtq_f32": {
        "signature": "uint32x4_t vcgtq_f32(float32x4_t a, float32x4_t b)",
        "instruction": "FCMGT Vd.4S, Vn.4S, Vm.4S",
        "description": (
            "Compare greater-than: for each lane, set all bits to 1 if a > b, else 0."
        ),
        "data_types": ["uint32x4_t", "float32x4_t"],
        "category": "Comparison",
        "architecture": "AArch32/AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = vld1q_f32(ptr_a);\n'
            'float32x4_t b = vld1q_f32(ptr_b);\n'
            'uint32x4_t mask = vcgtq_f32(a, b);  // lane-wise a > b'
        ),
    },
    "vcltq_s32": {
        "signature": "uint32x4_t vcltq_s32(int32x4_t a, int32x4_t b)",
        "instruction": "CMGT Vd.4S, Vm.4S, Vn.4S  (reversed operands)",
        "description": (
            "Compare less-than for signed 32-bit integers: for each lane, set all bits to 1 "
            "if a < b, else 0. Implemented by reversing operands of CMGT."
        ),
        "data_types": ["uint32x4_t", "int32x4_t"],
        "category": "Comparison",
        "architecture": "AArch32/AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'int32x4_t a = vld1q_s32(ptr_a);\n'
            'int32x4_t b = vld1q_s32(ptr_b);\n'
            'uint32x4_t mask = vcltq_s32(a, b);  // lane-wise a < b'
        ),
    },
    "vmaxq_f32": {
        "signature": "float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b)",
        "instruction": "FMAX Vd.4S, Vn.4S, Vm.4S",
        "description": "Element-wise maximum of two float32 vectors. Returns the larger value in each lane.",
        "data_types": ["float32x4_t"],
        "category": "Comparison",
        "architecture": "AArch32/AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = vld1q_f32(ptr_a);\n'
            'float32x4_t b = vld1q_f32(ptr_b);\n'
            'float32x4_t m = vmaxq_f32(a, b);  // max(a, b) per lane'
        ),
    },
    "vminq_f32": {
        "signature": "float32x4_t vminq_f32(float32x4_t a, float32x4_t b)",
        "instruction": "FMIN Vd.4S, Vn.4S, Vm.4S",
        "description": "Element-wise minimum of two float32 vectors. Returns the smaller value in each lane.",
        "data_types": ["float32x4_t"],
        "category": "Comparison",
        "architecture": "AArch32/AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = vld1q_f32(ptr_a);\n'
            'float32x4_t b = vld1q_f32(ptr_b);\n'
            'float32x4_t m = vminq_f32(a, b);  // min(a, b) per lane'
        ),
    },
    # --- Bitwise ---
    "vandq_u32": {
        "signature": "uint32x4_t vandq_u32(uint32x4_t a, uint32x4_t b)",
        "instruction": "AND Vd.16B, Vn.16B, Vm.16B",
        "description": "Bitwise AND of two 128-bit vectors.",
        "data_types": ["uint32x4_t"],
        "category": "Bitwise",
        "architecture": "AArch32/AArch64",
        "latency": "1-2 cycles (Cortex-A78 class)",
        "throughput": "2-4 per cycle",
        "example": (
            'uint32x4_t a = vld1q_u32(ptr_a);\n'
            'uint32x4_t mask = vdupq_n_u32(0x7FFFFFFF);\n'
            'uint32x4_t result = vandq_u32(a, mask);  // clear sign bit'
        ),
    },
    "vorrq_u32": {
        "signature": "uint32x4_t vorrq_u32(uint32x4_t a, uint32x4_t b)",
        "instruction": "ORR Vd.16B, Vn.16B, Vm.16B",
        "description": "Bitwise OR of two 128-bit vectors.",
        "data_types": ["uint32x4_t"],
        "category": "Bitwise",
        "architecture": "AArch32/AArch64",
        "latency": "1-2 cycles (Cortex-A78 class)",
        "throughput": "2-4 per cycle",
        "example": (
            'uint32x4_t a = vld1q_u32(ptr_a);\n'
            'uint32x4_t b = vld1q_u32(ptr_b);\n'
            'uint32x4_t result = vorrq_u32(a, b);  // a | b'
        ),
    },
    "veorq_u32": {
        "signature": "uint32x4_t veorq_u32(uint32x4_t a, uint32x4_t b)",
        "instruction": "EOR Vd.16B, Vn.16B, Vm.16B",
        "description": "Bitwise exclusive-OR (XOR) of two 128-bit vectors.",
        "data_types": ["uint32x4_t"],
        "category": "Bitwise",
        "architecture": "AArch32/AArch64",
        "latency": "1-2 cycles (Cortex-A78 class)",
        "throughput": "2-4 per cycle",
        "example": (
            'uint32x4_t a = vld1q_u32(ptr_a);\n'
            'uint32x4_t b = vld1q_u32(ptr_b);\n'
            'uint32x4_t result = veorq_u32(a, b);  // a ^ b'
        ),
    },
    "vmvnq_u32": {
        "signature": "uint32x4_t vmvnq_u32(uint32x4_t a)",
        "instruction": "MVN Vd.16B, Vn.16B  (alias: NOT)",
        "description": "Bitwise NOT (ones' complement) of a 128-bit vector.",
        "data_types": ["uint32x4_t"],
        "category": "Bitwise",
        "architecture": "AArch32/AArch64",
        "latency": "1-2 cycles (Cortex-A78 class)",
        "throughput": "2-4 per cycle",
        "example": (
            'uint32x4_t a = vld1q_u32(ptr_a);\n'
            'uint32x4_t inv = vmvnq_u32(a);  // ~a'
        ),
    },
    "vbslq_u32": {
        "signature": "uint32x4_t vbslq_u32(uint32x4_t mask, uint32x4_t a, uint32x4_t b)",
        "instruction": "BSL Vd.16B, Vn.16B, Vm.16B",
        "description": (
            "Bitwise select: for each bit, if mask bit is 1 select from a, else select from b. "
            "Equivalent to (mask & a) | (~mask & b). Very useful for branchless conditional operations."
        ),
        "data_types": ["uint32x4_t"],
        "category": "Bitwise",
        "architecture": "AArch32/AArch64",
        "latency": "1-2 cycles (Cortex-A78 class)",
        "throughput": "2-4 per cycle",
        "example": (
            'uint32x4_t mask = vcgtq_f32(a, b);  // compare\n'
            'uint32x4_t result = vbslq_u32(mask,\n'
            '    vreinterpretq_u32_f32(a),\n'
            '    vreinterpretq_u32_f32(b));  // select a where a > b'
        ),
    },
    # --- Shuffle/Permute ---
    "vzip1q_f32": {
        "signature": "float32x4_t vzip1q_f32(float32x4_t a, float32x4_t b)",
        "instruction": "ZIP1 Vd.4S, Vn.4S, Vm.4S",
        "description": (
            "Interleave the lower halves of two vectors. "
            "Result: {a[0], b[0], a[1], b[1]}."
        ),
        "data_types": ["float32x4_t"],
        "category": "Shuffle/Permute",
        "architecture": "AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = {1, 2, 3, 4};\n'
            'float32x4_t b = {5, 6, 7, 8};\n'
            'float32x4_t z = vzip1q_f32(a, b);  // {1, 5, 2, 6}'
        ),
    },
    "vuzp1q_f32": {
        "signature": "float32x4_t vuzp1q_f32(float32x4_t a, float32x4_t b)",
        "instruction": "UZP1 Vd.4S, Vn.4S, Vm.4S",
        "description": (
            "De-interleave (unzip) even elements from two vectors. "
            "Result: {a[0], a[2], b[0], b[2]}."
        ),
        "data_types": ["float32x4_t"],
        "category": "Shuffle/Permute",
        "architecture": "AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = {1, 2, 3, 4};\n'
            'float32x4_t b = {5, 6, 7, 8};\n'
            'float32x4_t u = vuzp1q_f32(a, b);  // {1, 3, 5, 7}'
        ),
    },
    "vrev64q_f32": {
        "signature": "float32x4_t vrev64q_f32(float32x4_t a)",
        "instruction": "REV64 Vd.4S, Vn.4S",
        "description": (
            "Reverse elements within each 64-bit doubleword. Swaps pairs of 32-bit floats: "
            "{a[0], a[1], a[2], a[3]} -> {a[1], a[0], a[3], a[2]}."
        ),
        "data_types": ["float32x4_t"],
        "category": "Shuffle/Permute",
        "architecture": "AArch32/AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = {1, 2, 3, 4};\n'
            'float32x4_t r = vrev64q_f32(a);  // {2, 1, 4, 3}'
        ),
    },
    "vtrnq_f32": {
        "signature": "float32x4x2_t vtrnq_f32(float32x4_t a, float32x4_t b)",
        "instruction": "TRN1/TRN2 Vd.4S, Vn.4S, Vm.4S",
        "description": (
            "Transpose elements from two vectors. Returns two vectors: "
            "TRN1 = {a[0], b[0], a[2], b[2]}, TRN2 = {a[1], b[1], a[3], b[3]}. "
            "Useful for matrix transpose operations."
        ),
        "data_types": ["float32x4x2_t", "float32x4_t"],
        "category": "Shuffle/Permute",
        "architecture": "AArch32/AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t row0 = {1, 2, 3, 4};\n'
            'float32x4_t row1 = {5, 6, 7, 8};\n'
            'float32x4x2_t t = vtrnq_f32(row0, row1);\n'
            '// t.val[0] = {1, 5, 3, 7}\n'
            '// t.val[1] = {2, 6, 4, 8}'
        ),
    },
    "vextq_f32": {
        "signature": "float32x4_t vextq_f32(float32x4_t a, float32x4_t b, const int n)",
        "instruction": "EXT Vd.16B, Vn.16B, Vm.16B, #imm",
        "description": (
            "Extract a vector from a pair of vectors. Concatenates a and b and extracts "
            "4 elements starting at index n. Useful for sliding window operations."
        ),
        "data_types": ["float32x4_t"],
        "category": "Shuffle/Permute",
        "architecture": "AArch32/AArch64",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "2 per cycle",
        "example": (
            'float32x4_t a = {1, 2, 3, 4};\n'
            'float32x4_t b = {5, 6, 7, 8};\n'
            'float32x4_t e = vextq_f32(a, b, 1);  // {2, 3, 4, 5}'
        ),
    },
    # --- Reduction (AArch64 only) ---
    "vaddvq_f32": {
        "signature": "float32_t vaddvq_f32(float32x4_t a)",
        "instruction": "FADDP (repeated) -- pairwise add to scalar",
        "description": (
            "Horizontal (across-vector) add: sum all four float32 lanes and return a scalar. "
            "AArch64 only. Equivalent to a[0] + a[1] + a[2] + a[3]."
        ),
        "data_types": ["float32_t", "float32x4_t"],
        "category": "Reduction",
        "architecture": "AArch64",
        "latency": "4-6 cycles (Cortex-A78 class)",
        "throughput": "1 per cycle",
        "example": (
            'float32x4_t v = vld1q_f32(data);\n'
            'float sum = vaddvq_f32(v);  // sum of all 4 lanes'
        ),
    },
    "vmaxvq_f32": {
        "signature": "float32_t vmaxvq_f32(float32x4_t a)",
        "instruction": "FMAXP (repeated) -- pairwise max to scalar",
        "description": (
            "Horizontal (across-vector) maximum: return the maximum value across all four "
            "float32 lanes. AArch64 only."
        ),
        "data_types": ["float32_t", "float32x4_t"],
        "category": "Reduction",
        "architecture": "AArch64",
        "latency": "4-6 cycles (Cortex-A78 class)",
        "throughput": "1 per cycle",
        "example": (
            'float32x4_t v = vld1q_f32(data);\n'
            'float mx = vmaxvq_f32(v);  // max of all 4 lanes'
        ),
    },
    # --- Dot Product (ARMv8.2+) ---
    "vdotq_s32": {
        "signature": "int32x4_t vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b)",
        "instruction": "SDOT Vd.4S, Vn.16B, Vm.16B",
        "description": (
            "Signed integer dot product. For each 32-bit lane, multiply four corresponding "
            "8-bit elements from a and b, sum the four products, and accumulate into acc. "
            "Four independent dot products in parallel. Key intrinsic for ML/AI int8 inference."
        ),
        "data_types": ["int32x4_t", "int8x16_t"],
        "category": "Dot Product",
        "architecture": "AArch64 (ARMv8.2+, FEAT_DotProd)",
        "latency": "3-4 cycles (Cortex-A78 class)",
        "throughput": "1-2 per cycle",
        "example": (
            'int32x4_t acc = vdupq_n_s32(0);\n'
            'int8x16_t a = vld1q_s8(weights);\n'
            'int8x16_t b = vld1q_s8(activations);\n'
            'acc = vdotq_s32(acc, a, b);  // 4 parallel dot products'
        ),
    },
    # --- Crypto ---
    "vaeseq_u8": {
        "signature": "uint8x16_t vaeseq_u8(uint8x16_t data, uint8x16_t key)",
        "instruction": "AESE Vd.16B, Vn.16B",
        "description": (
            "AES single round encryption: performs SubBytes and ShiftRows stages of a single "
            "AES round. XORs the data with the round key first. Must be followed by AESMC "
            "(MixColumns) for a complete round except the final round."
        ),
        "data_types": ["uint8x16_t"],
        "category": "Crypto",
        "architecture": "AArch64 (FEAT_AES)",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "1-2 per cycle",
        "example": (
            'uint8x16_t state = vld1q_u8(plaintext);\n'
            'uint8x16_t rk = vld1q_u8(round_key);\n'
            'state = vaeseq_u8(state, rk);  // SubBytes + ShiftRows\n'
            'state = vaesmcq_u8(state);      // MixColumns'
        ),
    },
    "vaesdq_u8": {
        "signature": "uint8x16_t vaesdq_u8(uint8x16_t data, uint8x16_t key)",
        "instruction": "AESD Vd.16B, Vn.16B",
        "description": (
            "AES single round decryption: performs InvSubBytes and InvShiftRows stages of a "
            "single AES decryption round. XORs the data with the round key first. Must be "
            "followed by AESIMC (InvMixColumns) for a complete round except the final round."
        ),
        "data_types": ["uint8x16_t"],
        "category": "Crypto",
        "architecture": "AArch64 (FEAT_AES)",
        "latency": "2-3 cycles (Cortex-A78 class)",
        "throughput": "1-2 per cycle",
        "example": (
            'uint8x16_t state = vld1q_u8(ciphertext);\n'
            'uint8x16_t rk = vld1q_u8(round_key);\n'
            'state = vaesdq_u8(state, rk);  // InvSubBytes + InvShiftRows\n'
            'state = vaesimcq_u8(state);     // InvMixColumns'
        ),
    },
}


def _format_neon_intrinsic(name: str, data: dict) -> str:
    """Format a single NEON intrinsic entry as readable text."""
    lines: list[str] = []
    lines.append(f"# NEON Intrinsic: {name}")
    lines.append("")
    lines.append(f"**Signature:** `{data['signature']}`")
    lines.append(f"**Instruction:** {data['instruction']}")
    lines.append(f"**Category:** {data['category']}")
    lines.append(f"**Architecture:** {data['architecture']}")
    lines.append("")
    lines.append("## Description")
    lines.append(data["description"])
    lines.append("")
    lines.append("## Performance (Cortex-A78 class)")
    lines.append(f"- Latency: {data['latency']}")
    lines.append(f"- Throughput: {data['throughput']}")
    lines.append("")
    lines.append("## Data Types")
    lines.append(", ".join(data["data_types"]))
    lines.append("")
    lines.append("## Example")
    lines.append("```c")
    lines.append(data["example"])
    lines.append("```")
    return "\n".join(lines)


def _format_neon_overview() -> str:
    """Format an overview table of all NEON intrinsics grouped by category."""
    # Group intrinsics by category
    categories: dict[str, list[str]] = {}
    for name, data in _NEON_INTRINSICS.items():
        cat = data["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(name)

    lines: list[str] = []
    lines.append("# NEON / Advanced SIMD Intrinsics -- Overview")
    lines.append("")
    lines.append(f"Total intrinsics: {len(_NEON_INTRINSICS)}")
    lines.append("")

    for cat in categories:
        lines.append(f"## {cat}")
        lines.append("")
        lines.append(f"{'Intrinsic':<22} {'Instruction':<38} Architecture")
        lines.append("-" * 90)
        for name in sorted(categories[cat]):
            d = _NEON_INTRINSICS[name]
            lines.append(f"  {name:<20} {d['instruction']:<38} {d['architecture']}")
        lines.append("")

    lines.append("Use explain_neon_intrinsic(<name>) for full details on any intrinsic.")
    return "\n".join(lines)


@mcp.tool()
def explain_neon_intrinsic(intrinsic_name: str) -> str:
    """Look up an ARM NEON / Advanced SIMD intrinsic by name.

    Returns the full intrinsic signature, equivalent assembly instruction,
    supported data types, typical latency/throughput info, a C usage example,
    and which architecture it is available on.

    Args:
        intrinsic_name: The NEON intrinsic name (e.g. "vaddq_f32", "vld1q_u8").
                        Case-insensitive.
                        Use "list" or "overview" to see all available intrinsics grouped by category.
    """
    key = intrinsic_name.strip().lower()

    if key in ("list", "overview"):
        return _format_neon_overview()

    if key in _NEON_INTRINSICS:
        return _format_neon_intrinsic(key, _NEON_INTRINSICS[key])

    # Not found -- try to suggest intrinsics from a similar category
    # Heuristic: extract a prefix pattern to guess category
    suggestions: list[str] = []

    # Try prefix-based matching (e.g. "vadd" matches "vaddq_f32", "vaddq_s32", "vaddvq_f32")
    prefix = key.split("_")[0] if "_" in key else key[:4]
    for name in _NEON_INTRINSICS:
        if name.startswith(prefix):
            suggestions.append(name)

    # If no prefix matches, suggest all intrinsic names
    if not suggestions:
        suggestions = sorted(_NEON_INTRINSICS.keys())

    suggestion_text = ", ".join(sorted(suggestions)[:10])
    return (
        f"Error: NEON intrinsic '{intrinsic_name}' not found.\n\n"
        f"Did you mean one of: {suggestion_text}\n\n"
        f"Use explain_neon_intrinsic('list') to see all available intrinsics."
    )

# ---------------------------------------------------------------------------
# Tool 18: explain_sme_tile  --  ARM SME (Scalable Matrix Extension) topics
# ---------------------------------------------------------------------------

_SME_TOPICS: dict[str, dict] = {
    "overview": {
        "title": "SME Overview",
        "description": (
            "ARM Scalable Matrix Extension (SME) is an architecture extension introduced "
            "in ARMv9.2-A that adds native matrix operations to AArch64.  SME builds on "
            "the Scalable Vector Extension (SVE/SVE2) foundation and introduces three "
            "major architectural concepts:\n\n"
            "1. **ZA tile storage** -- a new square, two-dimensional register array whose "
            "dimensions are tied to the SVE vector length (VL).  If VL is 128 bits "
            "(16 bytes), then ZA is 16x16 bytes = 256 bytes.  At VL=512 it is 64x64 "
            "bytes = 4 KiB.\n\n"
            "2. **Streaming SVE mode** -- a new processor mode (PSTATE.SM=1) that gives "
            "access to ZA and a streaming-capable subset of SVE, but disables classic "
            "NEON (AdvSIMD) and non-streaming SVE instructions.\n\n"
            "3. **Outer-product accumulation** -- dedicated instructions (FMOPA, FMOPS, "
            "SMOPA, UMOPA, ...) that compute the outer product of two SVE vectors and "
            "accumulate the result directly into a ZA tile, enabling very high "
            "throughput for matrix-multiply workloads.\n\n"
            "Comparison:\n"
            "  - NEON: fixed 128-bit SIMD, no matrix concept.\n"
            "  - SVE/SVE2: scalable 128-2048 bit vectors, 1-D operations.\n"
            "  - SME: scalable 2-D tile storage + outer-product instructions.\n\n"
            "SME is designed for GEMM-heavy workloads (machine learning inference/training, "
            "scientific computing) where the dominant operation is matrix multiplication."
        ),
        "key_points": [
            "SME introduced in ARMv9.2-A; requires SVE2 as a prerequisite",
            "ZA is a square 2-D register file: SVL x SVL bytes (SVL = streaming VL in bytes)",
            "Streaming SVE mode (PSTATE.SM) gates access to ZA and streaming-SVE instructions",
            "Outer-product instructions (FMOPA, etc.) accumulate directly into ZA tiles",
            "SME does NOT replace SVE -- it complements it for 2-D workloads",
            "NEON/AdvSIMD is unavailable while in streaming mode",
        ],
        "instructions": ["SMSTART", "SMSTOP", "FMOPA", "FMOPS", "ZERO"],
        "code_example": (
            "// Enable streaming SVE mode + ZA\n"
            "SMSTART          // sets PSTATE.SM=1, PSTATE.ZA=1\n\n"
            "// Zero the ZA tile\n"
            "ZERO {ZA}        // clear all of ZA\n\n"
            "// Outer product: ZA0.S += Z0.S outer Z1.S\n"
            "FMOPA ZA0.S, P0/M, P1/M, Z0.S, Z1.S\n\n"
            "// Store a horizontal slice of ZA\n"
            "ST1W {ZA0H.S[W12, 0]}, P0, [X0]\n\n"
            "// Disable streaming mode + ZA\n"
            "SMSTOP           // sets PSTATE.SM=0, PSTATE.ZA=0"
        ),
        "related_topics": ["za_storage", "streaming_mode", "outer_product"],
    },
    "za_storage": {
        "title": "SME: ZA Tile Storage",
        "description": (
            "ZA is a two-dimensional register array introduced by SME.  Its size is "
            "tied to the Streaming SVE Vector Length (SVL):\n\n"
            "  ZA size = SVL_bytes x SVL_bytes\n\n"
            "For example, if SVL = 128 bits = 16 bytes, ZA is 16x16 = 256 bytes.  "
            "At SVL = 512 bits = 64 bytes, ZA is 64x64 = 4096 bytes.\n\n"
            "ZA is subdivided into *tiles* whose number and size depend on the element "
            "width used:\n\n"
            "  Element width | Tiles        | Each tile\n"
            "  8-bit         | ZA0-ZA15     | (SVL/8)  x (SVL/8)  bytes\n"
            "  16-bit        | ZA0-ZA7      | (SVL/16) x (SVL/16) halfwords\n"
            "  32-bit        | ZA0-ZA3      | (SVL/32) x (SVL/32) words\n"
            "  64-bit        | ZA0-ZA1      | (SVL/64) x (SVL/64) doublewords\n"
            "  128-bit       | ZA0          | (SVL/128)x(SVL/128) quadwords\n\n"
            "All tiles overlap -- they are views into the same ZA storage.  Writing "
            "ZA0.B affects bits that are also visible through ZA0.S, ZA1.S, etc.\n\n"
            "ZA is accessible only when PSTATE.ZA=1 (set by SMSTART ZA or SMSTART SM).  "
            "When PSTATE.ZA transitions from 0 to 1, all of ZA is zeroed.  When "
            "PSTATE.ZA transitions from 1 to 0, the contents are architecturally lost "
            "(the OS may lazily save them)."
        ),
        "key_points": [
            "ZA size = SVL_bytes x SVL_bytes (square, tied to streaming VL)",
            "8-bit element view: ZA0-ZA15 (16 tiles)",
            "16-bit element view: ZA0-ZA7 (8 tiles)",
            "32-bit element view: ZA0-ZA3 (4 tiles)",
            "64-bit element view: ZA0-ZA1 (2 tiles)",
            "128-bit element view: ZA0 (1 tile)",
            "All tiles alias the same underlying storage",
            "ZA is zeroed on activation (PSTATE.ZA 0->1) and lost on deactivation (1->0)",
            "SMSTART ZA enables ZA without entering streaming mode; SMSTART SM enables both",
        ],
        "instructions": ["SMSTART", "SMSTOP", "ZERO", "RDSVL"],
        "code_example": (
            "// Query the streaming VL in bytes\n"
            "RDSVL X0, #1     // X0 = SVL in bytes\n\n"
            "// Enable ZA only (not streaming mode)\n"
            "SMSTART ZA       // PSTATE.ZA=1, ZA zeroed\n\n"
            "// Zero specific tiles\n"
            "ZERO {ZA0.S, ZA1.S}  // zero 32-bit tiles 0 and 1\n\n"
            "// Disable ZA\n"
            "SMSTOP ZA        // PSTATE.ZA=0, contents lost"
        ),
        "related_topics": ["overview", "streaming_mode", "tile_load_store"],
    },
    "streaming_mode": {
        "title": "SME: Streaming SVE Mode",
        "description": (
            "Streaming SVE mode is a distinct processor execution mode controlled by "
            "PSTATE.SM.  When SM=1, the processor operates in streaming mode:\n\n"
            "**What changes in streaming mode:**\n"
            "  - Classic NEON (AdvSIMD) instructions are UNDEFINED and will trap.\n"
            "  - Only the *streaming-compatible* subset of SVE/SVE2 is available.\n"
            "  - The vector length may differ from non-streaming SVE VL; it is the "
            "Streaming SVE Vector Length (SVL), queried with RDSVL.\n"
            "  - ZA tile storage is accessible (if PSTATE.ZA=1).\n"
            "  - SME outer-product instructions (FMOPA, FMOPS, etc.) are available.\n\n"
            "**What stays the same:**\n"
            "  - All integer/FP scalar instructions work normally.\n"
            "  - Memory ordering, exceptions, and privilege levels are unchanged.\n\n"
            "SMSTART and SMSTOP control streaming mode:\n"
            "  SMSTART       -- sets both PSTATE.SM=1 and PSTATE.ZA=1\n"
            "  SMSTART SM    -- sets only PSTATE.SM=1\n"
            "  SMSTART ZA    -- sets only PSTATE.ZA=1 (no streaming mode)\n"
            "  SMSTOP        -- clears both PSTATE.SM and PSTATE.ZA\n"
            "  SMSTOP SM     -- clears only PSTATE.SM\n"
            "  SMSTOP ZA     -- clears only PSTATE.ZA\n\n"
            "On entry to streaming mode, the contents of the SVE Z/P/FFR registers "
            "are zeroed.  On exit, they are zeroed again.  This means you cannot "
            "pass data between streaming and non-streaming mode via vector registers "
            "without explicit save/restore."
        ),
        "key_points": [
            "PSTATE.SM=1 activates streaming SVE mode",
            "NEON/AdvSIMD instructions become UNDEFINED in streaming mode",
            "Only streaming-compatible SVE instructions are available",
            "SVE vector length may differ: non-streaming VL vs streaming SVL",
            "Z/P/FFR registers are zeroed on SM transitions (0->1 and 1->0)",
            "SMSTART sets SM+ZA; SMSTART SM sets only SM; SMSTART ZA sets only ZA",
            "SMSTOP is the inverse; clears SM, ZA, or both",
        ],
        "instructions": ["SMSTART", "SMSTOP", "RDSVL", "MSR", "MRS"],
        "code_example": (
            "// Enter streaming mode (enables SM + ZA)\n"
            "SMSTART           // PSTATE.SM=1, PSTATE.ZA=1\n\n"
            "// Read streaming vector length\n"
            "RDSVL X0, #1      // X0 = SVL in bytes\n"
            "CNTW X1           // X1 = number of 32-bit elements per vector\n\n"
            "// ... perform streaming SVE + SME operations ...\n\n"
            "// Exit streaming mode only (keep ZA alive)\n"
            "SMSTOP SM         // PSTATE.SM=0, Z/P/FFR zeroed\n\n"
            "// Later, release ZA\n"
            "SMSTOP ZA         // PSTATE.ZA=0, ZA contents lost"
        ),
        "related_topics": ["overview", "za_storage", "programming_model"],
    },
    "outer_product": {
        "title": "SME: Outer Product Operations",
        "description": (
            "The core computational primitive in SME is the outer-product-and-accumulate "
            "instruction.  Given two SVE vector registers, the instruction computes their "
            "outer product and adds (or subtracts) the result into a ZA tile.\n\n"
            "For FMOPA ZA0.S, P0/M, P1/M, Z0.S, Z1.S:\n"
            "  - For each active row i (where P0[i]=1) and active column j (where P1[j]=1):\n"
            "    ZA0.S[i][j] += Z0.S[i] * Z1.S[j]\n\n"
            "This single instruction performs (SVL/32)^2 FP32 multiply-accumulates in one "
            "step -- for SVL=512, that is 16x16 = 256 FMACs.\n\n"
            "**Comparison with traditional approach:**\n"
            "  Traditional FMLA: processes one row at a time, VL elements per instruction.\n"
            "  FMOPA: processes an entire (SVL/element)^2 tile in one instruction.\n"
            "  For a K-iteration GEMM inner loop, FMOPA needs K instructions vs K*N/VL "
            "for FMLA.\n\n"
            "**Available variants:**\n"
            "  FMOPA / FMOPS  -- FP32, FP64 outer product add/subtract\n"
            "  BFMOPA / BFMOPS -- BF16 inputs, FP32 accumulation (2x throughput)\n"
            "  SMOPA / SMOPS   -- signed int8 inputs, int32 accumulation\n"
            "  UMOPA / UMOPS   -- unsigned int8 inputs, int32 accumulation\n"
            "  USMOPA / USMOPS -- unsigned x signed int8 -> int32"
        ),
        "key_points": [
            "FMOPA computes outer product of two SVE vectors and accumulates into ZA tile",
            "One instruction performs (SVL/element_size)^2 multiply-accumulates",
            "Dual predicate masks: P0 controls active rows, P1 controls active columns",
            "BF16 variants double throughput by packing 2 BF16 elements per 32-bit lane",
            "INT8 variants quadruple throughput: 4 int8 values per 32-bit lane",
            "FMOPS subtracts the outer product (useful for negation patterns)",
            "Far more efficient than FMLA loop for GEMM-style computation",
        ],
        "instructions": ["FMOPA", "FMOPS", "BFMOPA", "BFMOPS", "SMOPA", "SMOPS", "UMOPA", "UMOPS"],
        "code_example": (
            "// FP32 matrix multiply: C[MxN] += A[MxK] * B[KxN]\n"
            "// Assume SVL=512 => 16 FP32 elements per vector\n"
            "// Each FMOPA produces a 16x16 tile update\n\n"
            "SMSTART               // enter streaming mode + enable ZA\n"
            "ZERO {ZA0.S}          // clear accumulator tile\n\n"
            "// Inner loop over K dimension\n"
            "loop:\n"
            "  LD1W {Z0.S}, P0/Z, [X_A]   // load column of A (M elements)\n"
            "  LD1W {Z1.S}, P1/Z, [X_B]   // load row of B (N elements)\n"
            "  FMOPA ZA0.S, P0/M, P1/M, Z0.S, Z1.S  // ZA0 += Z0 outer Z1\n"
            "  ADD X_A, X_A, X_STRIDE_A\n"
            "  ADD X_B, X_B, X_STRIDE_B\n"
            "  SUBS X_K, X_K, #1\n"
            "  B.NE loop\n\n"
            "// Store result tile\n"
            "MOV W12, #0\n"
            "store_loop:\n"
            "  ST1W {ZA0H.S[W12, 0]}, P0, [X_C]\n"
            "  ADD X_C, X_C, X_STRIDE_C\n"
            "  ADD W12, W12, #1\n"
            "  CMP W12, X_ROWS\n"
            "  B.LT store_loop\n\n"
            "SMSTOP                // exit streaming mode + release ZA"
        ),
        "related_topics": ["mopa_fmopa", "za_storage", "tile_load_store"],
    },
    "tile_load_store": {
        "title": "SME: Tile Load and Store Operations",
        "description": (
            "SME provides instructions to load and store ZA tile data from/to memory.  "
            "There are two levels of access:\n\n"
            "**Full ZA save/restore (context switch):**\n"
            "  LDR ZA[Wv, #imm], [Xn]   -- load one horizontal vector-slice of ZA\n"
            "  STR ZA[Wv, #imm], [Xn]    -- store one horizontal vector-slice of ZA\n"
            "  These are used for context switching.  The loop variable Wv ranges from "
            "0 to SVL_bytes-1, storing the entire ZA array one row at a time.\n\n"
            "**Tile slice access (computational):**\n"
            "  LD1B/LD1H/LD1W/LD1D {ZAn[HV].T[Wv, #imm]}, Pg/Z, [Xn, Xm]\n"
            "  ST1B/ST1H/ST1W/ST1D {ZAn[HV].T[Wv, #imm]}, Pg, [Xn, Xm]\n"
            "  H = horizontal slice, V = vertical slice.\n"
            "  These access a single row (H) or column (V) of a specific tile.\n\n"
            "**Horizontal vs Vertical slices:**\n"
            "  - ZA0H.S[W12, 0] = row W12 of tile ZA0 (32-bit elements)\n"
            "  - ZA0V.S[W12, 0] = column W12 of tile ZA0 (32-bit elements)\n"
            "  The H/V suffix selects the slice direction.  Both are predicated.\n\n"
            "The slice index uses a W-register (W12-W15) plus an immediate offset.  "
            "The hardware computes (Wv + imm) MOD (SVL/element_bytes) to select the "
            "row or column."
        ),
        "key_points": [
            "LDR/STR ZA -- load/store full ZA rows for context switch (one slice per instruction)",
            "LD1B/ST1B through LD1D/ST1D -- typed tile slice access for computation",
            "Horizontal slice (H): accesses a row of the tile",
            "Vertical slice (V): accesses a column of the tile",
            "Slice index = (Wv + imm) MOD (SVL / element_bytes)",
            "Only W12-W15 can be used as slice index registers",
            "Full ZA context save requires SVL_bytes iterations of STR",
        ],
        "instructions": ["LDR", "STR", "LD1B", "LD1H", "LD1W", "LD1D", "ST1B", "ST1H", "ST1W", "ST1D"],
        "code_example": (
            "// Save entire ZA to memory (context switch)\n"
            "RDSVL X1, #1          // X1 = SVL in bytes = number of rows\n"
            "MOV W12, #0\n"
            "save_loop:\n"
            "  STR ZA[W12, 0], [X0]  // store row W12\n"
            "  ADD X0, X0, X1        // advance by SVL bytes\n"
            "  ADD W12, W12, #1\n"
            "  CMP W12, W1\n"
            "  B.LT save_loop\n\n"
            "// Load a horizontal slice of tile ZA0 (32-bit)\n"
            "MOV W12, #3\n"
            "LD1W {ZA0H.S[W12, 0]}, P0/Z, [X2, X3, LSL #2]\n\n"
            "// Store a vertical slice of tile ZA1 (16-bit)\n"
            "MOV W13, #0\n"
            "ST1H {ZA1V.H[W13, 0]}, P1, [X4, X5, LSL #1]"
        ),
        "related_topics": ["za_storage", "context_switching", "outer_product"],
    },
    "mopa_fmopa": {
        "title": "SME: FMOPA Instruction Detail",
        "description": (
            "FMOPA (Floating-point Matrix Outer Product and Accumulate) is the "
            "central computational instruction in SME.\n\n"
            "**Syntax:**\n"
            "  FMOPA ZAda.S, Pn/M, Pm/M, Zn.S, Zm.S\n\n"
            "**Operand breakdown:**\n"
            "  ZAda.S -- destination/accumulator tile (e.g., ZA0.S for 32-bit)\n"
            "    'd' selects which tile (0-3 for .S, 0-1 for .D)\n"
            "  Pn/M   -- row predicate (merging).  Inactive rows are not modified.\n"
            "  Pm/M   -- column predicate (merging).  Inactive columns are not modified.\n"
            "  Zn.S   -- first source vector (provides row values)\n"
            "  Zm.S   -- second source vector (provides column values)\n\n"
            "**Operation (pseudocode):**\n"
            "  for row = 0 to (SVL/32)-1:\n"
            "    if Pn[row] == 1:\n"
            "      for col = 0 to (SVL/32)-1:\n"
            "        if Pm[col] == 1:\n"
            "          ZAda.S[row][col] += Zn.S[row] * Zm.S[col]\n\n"
            "**Predication:**\n"
            "  Both predicates use *merging* semantics (/M): elements corresponding "
            "to inactive predicate lanes are left UNCHANGED in the accumulator.  "
            "This allows partial tile updates for edge cases (e.g., last tile block "
            "in a matrix that does not fill the full SVL width).\n\n"
            "**Variants:**\n"
            "  FMOPA ZAda.D -- FP64 outer product (ZA0.D or ZA1.D)\n"
            "  FMOPA ZAda.S, ..., Zn.H, Zm.H -- widening FP16->FP32\n"
            "  BFMOPA ZAda.S, ..., Zn.H, Zm.H -- BF16->FP32 widening\n"
            "  FMOPS -- same but SUBTRACTS the outer product from the tile"
        ),
        "key_points": [
            "FMOPA ZAda.S, Pn/M, Pm/M, Zn.S, Zm.S -- full syntax",
            "ZAda: destination tile and accumulator (read-modify-write)",
            "Pn/M: row predicate with merging (inactive rows unchanged)",
            "Pm/M: column predicate with merging (inactive cols unchanged)",
            "Zn: provides row values, Zm: provides column values",
            "For .S tiles (FP32): ZA0-ZA3 available, each (SVL/32)x(SVL/32)",
            "For .D tiles (FP64): ZA0-ZA1 available, each (SVL/64)x(SVL/64)",
            "One FMOPA does (SVL/element_bits)^2 FP multiply-accumulate operations",
        ],
        "instructions": ["FMOPA", "FMOPS", "BFMOPA", "BFMOPS"],
        "code_example": (
            "// Detailed FMOPA example: 4x4 FP32 outer product (SVL=128)\n"
            "// Z0.S = [a0, a1, a2, a3]  (row vector)\n"
            "// Z1.S = [b0, b1, b2, b3]  (col vector)\n"
            "//\n"
            "// Result in ZA0.S:\n"
            "//   ZA0.S[0][0] += a0*b0  ZA0.S[0][1] += a0*b1  ...\n"
            "//   ZA0.S[1][0] += a1*b0  ZA0.S[1][1] += a1*b1  ...\n"
            "//   ZA0.S[2][0] += a2*b0  ZA0.S[2][1] += a2*b1  ...\n"
            "//   ZA0.S[3][0] += a3*b0  ZA0.S[3][1] += a3*b1  ...\n\n"
            "PTRUE P0.S            // all rows active\n"
            "PTRUE P1.S            // all columns active\n"
            "FMOPA ZA0.S, P0/M, P1/M, Z0.S, Z1.S\n\n"
            "// Partial update (only first 2 rows, first 3 columns):\n"
            "WHILELT P0.S, XZR, #2  // P0 = {1,1,0,0}\n"
            "WHILELT P1.S, XZR, #3  // P1 = {1,1,1,0}\n"
            "FMOPA ZA0.S, P0/M, P1/M, Z2.S, Z3.S"
        ),
        "related_topics": ["outer_product", "za_storage", "streaming_mode"],
    },
    "sme2": {
        "title": "SME2 Extensions",
        "description": (
            "SME2, introduced in ARMv9.4-A, significantly extends the Scalable Matrix "
            "Extension with multi-vector operations, structured memory access, and a "
            "new lookup table register.\n\n"
            "**Multi-vector outer products:**\n"
            "  SME2 adds FMOPA variants that take 2-register or 4-register groups as "
            "inputs.  For example, FMOPA ZA.S, {Z0.S-Z1.S}, {Z2.S-Z3.S} performs "
            "two outer products and accumulates them into the same tile, doubling "
            "throughput without additional instructions.\n\n"
            "**ZT0 lookup table register:**\n"
            "  SME2 introduces ZT0, a 512-bit (64-byte) register used for lookup-table "
            "operations.  The LUTI2 and LUTI4 instructions perform 2-bit or 4-bit "
            "indexed lookups into ZT0, useful for weight dequantization in ML inference "
            "(e.g., expanding 4-bit quantized weights to FP16/BF16).\n\n"
            "**Structured load/store:**\n"
            "  LD1B/ST1B (and other typed variants) gain multi-register forms:\n"
            "  LD1W {Z0.S, Z4.S, Z8.S, Z12.S}, PNn/Z, [Xn, #0, MUL VL]\n"
            "  These load/store 2 or 4 registers with a single instruction using the "
            "new predicate-as-counter (PNn) registers.\n\n"
            "**MOVT instruction:**\n"
            "  MOVT copies data between ZT0 and general-purpose/vector registers, "
            "enabling ZT0 to be initialized with lookup table contents."
        ),
        "key_points": [
            "SME2 introduced in ARMv9.4-A as an extension to SME",
            "Multi-vector FMOPA: 2- and 4-register group inputs double/quadruple outer product throughput",
            "ZT0: new 512-bit lookup table register for weight dequantization",
            "LUTI2/LUTI4: 2-bit and 4-bit indexed lookups into ZT0",
            "Structured multi-register loads/stores with predicate-as-counter (PN) registers",
            "MOVT: move data to/from ZT0 register",
            "Designed for ML inference with quantized weights (INT4/INT2 -> FP16/BF16)",
        ],
        "instructions": ["FMOPA", "LUTI2", "LUTI4", "MOVT", "LD1B", "ST1B"],
        "code_example": (
            "// SME2: Multi-vector outer product\n"
            "FMOPA ZA.S, {Z0.S-Z1.S}, {Z2.S-Z3.S}\n"
            "// Equivalent to:\n"
            "//   FMOPA ZA.S, ..., Z0.S, Z2.S\n"
            "//   FMOPA ZA.S, ..., Z1.S, Z3.S\n\n"
            "// SME2: Lookup table dequantization\n"
            "// Load 64-byte lookup table into ZT0\n"
            "LDR ZT0, [X0]         // X0 -> 64-byte LUT\n\n"
            "// 4-bit lookup: expand quantized weights\n"
            "LUTI4 {Z0.B}, ZT0, Z16[0]   // Z0 = ZT0[Z16 indices]\n\n"
            "// SME2: Structured multi-register load\n"
            "LD1W {Z0.S, Z4.S}, PN8/Z, [X1, #0, MUL VL]"
        ),
        "related_topics": ["overview", "outer_product", "mopa_fmopa"],
    },
    "context_switching": {
        "title": "SME: Context Switching and OS Support",
        "description": (
            "Because ZA is large stateful storage, the OS kernel must manage it during "
            "context switches, signal delivery, and process migration.\n\n"
            "**ZA state size:**\n"
            "  ZA contains SVL_bytes x SVL_bytes of data.  At SVL=512, this is 4 KiB; "
            "at SVL=2048, it would be 64 KiB.  Use RDSVL to determine the size:\n"
            "    RDSVL X0, #1  -> X0 = SVL in bytes\n"
            "    size = X0 * X0  (total ZA bytes)\n\n"
            "**Save/restore procedure:**\n"
            "  // Save\n"
            "  RDSVL X1, #1\n"
            "  MOV W12, #0\n"
            "  loop: STR ZA[W12, 0], [X_buf]; ADD X_buf, X_buf, X1; ...\n\n"
            "  // Restore\n"
            "  SMSTART ZA          // enables ZA, zeros it\n"
            "  MOV W12, #0\n"
            "  loop: LDR ZA[W12, 0], [X_buf]; ADD X_buf, X_buf, X1; ...\n\n"
            "**Lazy save optimization:**\n"
            "  Linux implements lazy ZA save: when a task is preempted while ZA is active, "
            "the kernel does NOT immediately save ZA.  Instead, it sets PSTATE.ZA=0 for "
            "the incoming task.  If the original task is rescheduled before any other task "
            "uses ZA, no save/restore was needed.  Only when another task executes SMSTART "
            "ZA does the kernel trap and save the first task's ZA.\n\n"
            "**Key OS responsibilities:**\n"
            "  1. Track per-thread PSTATE.SM and PSTATE.ZA state.\n"
            "  2. Allocate SVL^2-byte ZA save buffer (or defer until needed).\n"
            "  3. Save/restore ZA on context switch if active.\n"
            "  4. Handle signal delivery: save ZA before signal handler, restore after.\n"
            "  5. Support ptrace access to ZA for debuggers."
        ),
        "key_points": [
            "ZA size = SVL^2 bytes; can be large (up to 64 KiB at SVL=2048)",
            "Use RDSVL to determine SVL at runtime for buffer allocation",
            "Save: loop STR ZA[W12, 0] for each row; Restore: SMSTART ZA + loop LDR",
            "Linux uses lazy save: defer ZA save until another task needs ZA",
            "Kernel must track PSTATE.SM and PSTATE.ZA per thread",
            "Signal delivery must save/restore ZA around signal handlers",
            "ZA save buffer must be RDSVL-based (not compile-time fixed) for portability",
        ],
        "instructions": ["SMSTART", "SMSTOP", "STR", "LDR", "RDSVL"],
        "code_example": (
            "// Linux kernel: save ZA state for context switch\n"
            "// (simplified pseudocode in assembly)\n\n"
            "mrs   X0, SVCR         // read streaming VCR\n"
            "tbz   X0, #1, skip     // bit 1 = PSTATE.ZA; skip if not active\n\n"
            "// ZA is active -- save it\n"
            "rdsvl X1, #1           // X1 = SVL bytes\n"
            "mul   X2, X1, X1       // X2 = total ZA size\n"
            "// Allocate X2 bytes for save area (or use preallocated buffer)\n\n"
            "mov   W12, #0\n"
            "save_loop:\n"
            "  str  ZA[W12, 0], [X_buf]\n"
            "  add  X_buf, X_buf, X1\n"
            "  add  W12, W12, #1\n"
            "  cmp  W12, W1\n"
            "  b.lt save_loop\n\n"
            "smstop za              // release ZA (PSTATE.ZA=0)\n"
            "skip:"
        ),
        "related_topics": ["za_storage", "streaming_mode", "programming_model"],
    },
    "programming_model": {
        "title": "SME: Programming Model and Function Attributes",
        "description": (
            "SME introduces new function type attributes for the AAPCS64 (ARM Procedure "
            "Call Standard) to handle streaming mode and ZA state across function calls.\n\n"
            "**Function type attributes:**\n\n"
            "  __arm_streaming\n"
            "    The function requires streaming SVE mode.  The caller must enter "
            "streaming mode (SMSTART SM) before calling.  On entry, PSTATE.SM=1 is "
            "guaranteed.  The compiler may insert SMSTART/SMSTOP at call boundaries.\n\n"
            "  __arm_streaming_compatible\n"
            "    The function works in BOTH streaming and non-streaming mode.  It only "
            "uses instructions that are valid in both modes.  The compiler does not "
            "insert any mode switches.\n\n"
            "  __arm_za_shared\n"
            "    The function uses ZA and expects it to be live on entry/exit.  The "
            "caller is responsible for having PSTATE.ZA=1.  The function may read and "
            "modify ZA.\n\n"
            "  __arm_new_za\n"
            "    The function creates a new ZA context.  On entry, the compiler inserts "
            "SMSTART ZA (creating fresh, zeroed ZA).  On return, it inserts SMSTOP ZA.  "
            "This is for functions that use ZA internally but do not share it.\n\n"
            "**Example C function signatures:**\n"
            "  void sgemm_tile(float *A, float *B, float *C, int M, int N, int K)\n"
            "    __arm_streaming __arm_za_shared;\n\n"
            "  void helper(int x) __arm_streaming_compatible;\n\n"
            "  void top_level_matmul(float *A, float *B, float *C)\n"
            "    __arm_new_za;\n\n"
            "**Key rules:**\n"
            "  - A non-streaming function calling an __arm_streaming function must have "
            "the compiler insert SMSTART SM before and SMSTOP SM after the call.\n"
            "  - __arm_za_shared functions must not be called without ZA enabled.\n"
            "  - __arm_new_za functions manage their own ZA lifetime."
        ),
        "key_points": [
            "__arm_streaming: function requires streaming SVE mode (PSTATE.SM=1)",
            "__arm_streaming_compatible: function works in both streaming and non-streaming mode",
            "__arm_za_shared: function reads/writes shared ZA state (caller enables ZA)",
            "__arm_new_za: function creates and destroys its own ZA context",
            "Compiler inserts SMSTART/SMSTOP at call boundaries for mode mismatches",
            "Z/P/FFR registers are zeroed on streaming mode transitions",
            "AAPCS64 defines callee-saved rules for ZA and streaming state",
        ],
        "instructions": ["SMSTART", "SMSTOP"],
        "code_example": (
            "// C function using SME with attributes\n\n"
            "#include <arm_sme.h>\n\n"
            "// This function runs in streaming mode and uses shared ZA\n"
            "void sme_gemm_kernel(const float *A, const float *B,\n"
            "                     int K, int tile_row, int tile_col)\n"
            "    __arm_streaming __arm_za_shared\n"
            "{\n"
            "    svfloat32_t a_vec, b_vec;\n"
            "    svbool_t p0 = svptrue_b32();\n\n"
            "    for (int k = 0; k < K; k++) {\n"
            "        a_vec = svld1_f32(p0, &A[k * tile_row]);\n"
            "        b_vec = svld1_f32(p0, &B[k * tile_col]);\n"
            "        svmopa_za32_f32_m(0, p0, p0, a_vec, b_vec);\n"
            "    }\n"
            "}\n\n"
            "// Top-level function that owns ZA\n"
            "void matmul(float *A, float *B, float *C, int M, int N, int K)\n"
            "    __arm_new_za\n"
            "{\n"
            "    svzero_za();  // zero all ZA tiles\n"
            "    sme_gemm_kernel(A, B, K, M, N);\n"
            "    // ... store ZA tiles to C ...\n"
            "}"
        ),
        "related_topics": ["streaming_mode", "context_switching", "detection"],
    },
    "detection": {
        "title": "SME: Runtime Feature Detection",
        "description": (
            "Before using SME instructions, software must verify that the CPU supports "
            "SME.  Detection differs by privilege level.\n\n"
            "**EL1+ (kernel / bare-metal):**\n"
            "  Read ID_AA64SMFR0_EL1 (SME Feature Register 0):\n"
            "    Bits [63:60] (FA64)   -- if non-zero, full A64 instruction set in "
            "streaming mode\n"
            "    Bits [55:52] (SMEver) -- 0=SME, 1=SME2, 2=SME2p1\n"
            "    Bits [51:48] (I16I64) -- int16 x int64 outer product support\n"
            "    Bits [43:40] (F64F64) -- FP64 outer product support\n"
            "    Bits [35:32] (I16I32) -- int16 x int32 outer product support\n"
            "    Bits [3:0]   (F32F32) -- FP32 outer product support (1 = supported)\n\n"
            "  Also check ID_AA64PFR1_EL1 bits [27:24] (SME field):\n"
            "    0b0001 = SME implemented\n"
            "    0b0010 = SME2 implemented\n\n"
            "**EL0 (userspace Linux):**\n"
            "  Use HWCAP2 flags via getauxval(AT_HWCAP2):\n"
            "    HWCAP2_SME        (1 << 23) -- SME supported\n"
            "    HWCAP2_SME_I16I64 (1 << 24)\n"
            "    HWCAP2_SME_F64F64 (1 << 25)\n"
            "    HWCAP2_SME_I8I32  (1 << 26)\n"
            "    HWCAP2_SME_F16F32 (1 << 27)\n"
            "    HWCAP2_SME_B16F32 (1 << 28)\n"
            "    HWCAP2_SME_F32F32 (1 << 29)\n"
            "    HWCAP2_SME_FA64   (1 << 30)\n"
            "    HWCAP2_SME2       (1UL << 37) -- SME2 supported\n\n"
            "  Alternative: read /proc/cpuinfo 'Features' line for 'sme', 'sme2'.\n\n"
            "**CPACR_EL1 / CPTR_EL2 access control:**\n"
            "  The kernel must enable SME access by setting CPACR_EL1.SMEN=0b11 and "
            "(if EL2 is present) CPTR_EL2 appropriately.  Without this, SMSTART will "
            "trap."
        ),
        "key_points": [
            "ID_AA64SMFR0_EL1: per-feature SME capability register (EL1+)",
            "ID_AA64PFR1_EL1 bits [27:24]: SME/SME2 implementation level",
            "Linux userspace: getauxval(AT_HWCAP2) with HWCAP2_SME flag (1<<23)",
            "HWCAP2_SME2 (1<<37) indicates SME2 support",
            "/proc/cpuinfo Features line shows 'sme' and 'sme2' tokens",
            "CPACR_EL1.SMEN must be 0b11 to allow EL0 SME access",
            "Without kernel enablement, SMSTART traps to EL1",
        ],
        "instructions": ["MRS", "SMSTART", "RDSVL"],
        "code_example": (
            "// C: Runtime detection on Linux\n"
            "#include <sys/auxv.h>\n"
            "#include <stdio.h>\n\n"
            "#ifndef HWCAP2_SME\n"
            "#define HWCAP2_SME  (1UL << 23)\n"
            "#endif\n"
            "#ifndef HWCAP2_SME2\n"
            "#define HWCAP2_SME2 (1UL << 37)\n"
            "#endif\n\n"
            "int detect_sme(void) {\n"
            "    unsigned long hwcap2 = getauxval(AT_HWCAP2);\n"
            "    if (hwcap2 & HWCAP2_SME2) {\n"
            "        printf(\"SME2 supported\\n\");\n"
            "        return 2;\n"
            "    } else if (hwcap2 & HWCAP2_SME) {\n"
            "        printf(\"SME supported\\n\");\n"
            "        return 1;\n"
            "    }\n"
            "    printf(\"SME not supported\\n\");\n"
            "    return 0;\n"
            "}\n\n"
            "// Assembly: kernel-level detection\n"
            "// MRS X0, ID_AA64PFR1_EL1\n"
            "// UBFX X0, X0, #24, #4  // extract SME field\n"
            "// CBZ X0, no_sme        // 0 = not implemented"
        ),
        "related_topics": ["overview", "programming_model", "streaming_mode"],
    },
}


def _format_sme_topic(topic_key: str, topic: dict) -> str:
    """Format an SME topic entry into a readable string."""
    lines: list[str] = []

    lines.append(f"# SME: {topic['title']}")
    lines.append("")
    lines.append("## Description")
    lines.append(topic["description"])
    lines.append("")

    lines.append("## Key Points")
    for pt in topic["key_points"]:
        lines.append(f"- {pt}")
    lines.append("")

    lines.append("## Related Instructions")
    lines.append(", ".join(topic["instructions"]))
    lines.append("")

    lines.append("## Example")
    lines.append("```")
    lines.append(topic["code_example"])
    lines.append("```")
    lines.append("")

    lines.append("## Related Topics")
    for rt in topic["related_topics"]:
        if rt in _SME_TOPICS:
            lines.append(f'Use `explain_sme_tile("{rt}")` for {_SME_TOPICS[rt]["title"]}.')

    return "\n".join(lines)


@mcp.tool()
def explain_sme_tile(operation: str) -> str:
    """Explain ARM SME (Scalable Matrix Extension) tile operations, ZA storage,
    streaming SVE mode, and related concepts.

    SME adds a 2-D tile register (ZA) and outer-product instructions for
    efficient matrix multiplication on AArch64.

    Args:
        operation: One of:
            "overview" -- What SME is, SME vs SVE vs NEON, the ZA tile architecture
            "za_storage" -- ZA array structure, tile sizes, SMSTART/SMSTOP lifecycle
            "streaming_mode" -- SMSTART/SMSTOP, what changes in streaming mode, PSTATE.SM
            "outer_product" -- FMOPA/FMOPS, outer product into ZA, int8/bf16/fp32 variants
            "tile_load_store" -- LDR/STR ZA, LD1B/ST1B slice access, H vs V slices
            "mopa_fmopa" -- Detailed FMOPA.S instruction breakdown with operands
            "sme2" -- SME2 extensions: multi-vector, ZT0, LUTI2/LUTI4, MOVT
            "context_switching" -- OS responsibilities, lazy save, ZA state management
            "programming_model" -- __arm_streaming, __arm_za_shared, function attributes
            "detection" -- ID_AA64SMFR0_EL1, HWCAP2_SME, runtime feature detection
    """
    key = operation.strip().lower()

    if key not in _SME_TOPICS:
        valid = ", ".join(sorted(_SME_TOPICS.keys()))
        return (
            f"Error: '{operation}' is not a recognised SME topic.\n\n"
            f"Valid topics: {valid}\n"
            f'Use `explain_sme_tile("overview")` for a general introduction.'
        )

    return _format_sme_topic(key, _SME_TOPICS[key])

# ---------------------------------------------------------------------------
# Tool 18: suggest_optimization — ARM-specific code optimization patterns
# ---------------------------------------------------------------------------

_OPTIMIZATION_PATTERNS: dict[str, dict] = {
    "matrix_multiply": {
        "description": (
            "Dense matrix multiplication (C = A * B). A fundamental kernel in linear algebra, "
            "neural-network inference, and scientific computing. ARM cores have wide SIMD datapaths "
            "that can be exploited with tiling, fused multiply-add, and specialized dot-product / "
            "matrix instructions."
        ),
        "baseline_approach": (
            "A naive triple-nested loop (i, j, k) with scalar FP multiply and add. The compiler "
            "may auto-vectorize the inner loop but typically cannot tile or reorder across loops, "
            "leading to poor cache utilization and low SIMD throughput."
        ),
        "optimizations": [
            {
                "technique": "NEON 128-bit tile-based multiply",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "Tile the matrix into 4x4 (FP32) or 8x8 (FP16) blocks that fit in NEON "
                    "registers. Use FMLA (fused multiply-add) to accumulate partial results "
                    "in register tiles, then store back. This maximises register reuse and "
                    "minimises memory traffic."
                ),
                "code_snippet": (
                    "// 4x4 FP32 tile using NEON intrinsics\n"
                    "float32x4_t c0 = vdupq_n_f32(0);\n"
                    "for (int k = 0; k < K; k++) {\n"
                    "    float32x4_t a0 = vld1q_f32(&A[i*K + k]);\n"
                    "    float32x4_t b0 = vld1q_f32(&B[k*N + j]);\n"
                    "    c0 = vfmaq_f32(c0, a0, b0);\n"
                    "}\n"
                    "vst1q_f32(&C[i*N + j], c0);"
                ),
                "speedup": "4-8x over scalar (FP32), up to 16x with FP16",
            },
            {
                "technique": "SVE scalable vectorization",
                "architecture": "AArch64 (ARMv8.2+ with SVE)",
                "description": (
                    "Use SVE predicated loops that adapt to the hardware vector length "
                    "(128-2048 bits). Write once, run optimally across Neoverse V1 (256-bit), "
                    "V2 (128-bit SVE2), and A64FX (512-bit)."
                ),
                "code_snippet": (
                    "svfloat32_t acc = svdup_f32(0);\n"
                    "svbool_t pg = svwhilelt_b32(k, K);\n"
                    "while (svptest_first(svptrue_b32(), pg)) {\n"
                    "    svfloat32_t a = svld1(pg, &A[i*K + k]);\n"
                    "    svfloat32_t b = svld1(pg, &B[k*N + j]);\n"
                    "    acc = svmla_m(pg, acc, a, b);\n"
                    "    k += svcntw();\n"
                    "    pg = svwhilelt_b32(k, K);\n"
                    "}"
                ),
                "speedup": "Scales linearly with vector length; 2x over NEON on 256-bit SVE",
            },
            {
                "technique": "SME outer-product accumulation",
                "architecture": "AArch64 (ARMv9.2+ with SME)",
                "description": (
                    "Scalable Matrix Extension (SME) provides FMOPA (outer product and accumulate) "
                    "that writes directly into ZA tile registers. A single instruction computes a "
                    "rank-1 update of the output tile, dramatically reducing loop overhead."
                ),
                "code_snippet": (
                    "// SME outer-product: ZA += a * b^T\n"
                    "FMOPA ZA0.S, P0/M, P1/M, Z0.S, Z1.S"
                ),
                "speedup": "10-20x over scalar; designed for GEMM workloads",
            },
            {
                "technique": "SDOT/UDOT for int8 quantized multiply",
                "architecture": "AArch64 (ARMv8.2+ with DotProd)",
                "description": (
                    "For int8 quantized inference, SDOT/UDOT compute a 4-element dot product "
                    "per lane, accumulating into int32. This gives 4x the throughput of int8 "
                    "multiply-accumulate sequences."
                ),
                "code_snippet": (
                    "int32x4_t acc = vdupq_n_s32(0);\n"
                    "int8x16_t a = vld1q_s8(&A[i]);\n"
                    "int8x16_t b = vld1q_s8(&B[j]);\n"
                    "acc = vdotq_s32(acc, a, b);  // 16 int8 MACs -> 4 int32 results"
                ),
                "speedup": "4x over SMULL+SADALP sequence; 16x over scalar int8",
            },
        ],
        "compiler_flags": [
            "-O2",
            "-march=armv8.2-a+dotprod",
            "-march=armv9-a+sme  (for SME)",
            "-ffast-math  (enables FMA contraction)",
        ],
        "target_core_notes": {
            "cortex-a78": "2 NEON pipes, dual-issue FMLA with 4-cycle latency. Tile to 4x4 FP32 for best throughput.",
            "neoverse-v2": "4 NEON pipes, 128-bit SVE2, 2x256-bit SVE on Neoverse V1. Use SVE for portability.",
            "cortex-a55": "Single NEON pipe, in-order core. Keep tiles small (4x1), avoid complex scheduling.",
        },
    },
    "memcpy": {
        "description": (
            "Memory copy operation — moving a block of bytes from source to destination. "
            "One of the most frequently called functions; even small improvements have "
            "system-wide impact. ARM provides wide load/store pairs and cache management "
            "instructions for optimal throughput."
        ),
        "baseline_approach": (
            "A byte-by-byte copy loop. Even with -O2, the compiler may only emit LDR/STR "
            "of 8-byte (64-bit) chunks, leaving half the memory bandwidth unused on cores "
            "with 128-bit or wider datapaths."
        ),
        "optimizations": [
            {
                "technique": "LDP/STP 128-bit pairs",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "Use LDP/STP (Load Pair / Store Pair) of 64-bit registers to move 16 bytes "
                    "per instruction, or LDP/STP of Q-registers to move 32 bytes. Unroll 2-4x "
                    "to saturate the load/store unit."
                ),
                "code_snippet": (
                    "// Copy 64 bytes per iteration using Q-register pairs\n"
                    "LDP  Q0, Q1, [X1], #32\n"
                    "LDP  Q2, Q3, [X1], #32\n"
                    "STP  Q0, Q1, [X0], #32\n"
                    "STP  Q2, Q3, [X0], #32"
                ),
                "speedup": "4-8x over byte-by-byte; 2x over 64-bit LDR/STR",
            },
            {
                "technique": "NEON load/store with post-increment",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "NEON VLD1/VST1 with post-increment addressing moves 16 bytes and updates "
                    "the pointer in a single instruction, reducing loop overhead."
                ),
                "code_snippet": (
                    "uint8x16_t data;\n"
                    "for (size_t i = 0; i < len; i += 16) {\n"
                    "    data = vld1q_u8(src + i);\n"
                    "    vst1q_u8(dst + i, data);\n"
                    "}"
                ),
                "speedup": "4x over byte copy; good for small-to-medium buffers",
            },
            {
                "technique": "DC ZVA for zero-fill destination",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "When copying into a fresh allocation that will be fully overwritten, use "
                    "DC ZVA (Data Cache Zero by VA) to zero the cache line without fetching from "
                    "memory, then overwrite with source data. Eliminates read-for-ownership traffic."
                ),
                "code_snippet": (
                    "// Zero cache line (typically 64 bytes) before writing\n"
                    "DC ZVA, X0   // zero cache line at X0, no memory read needed\n"
                    "STP  Q0, Q1, [X0]\n"
                    "STP  Q2, Q3, [X0, #32]"
                ),
                "speedup": "Up to 2x for large copies by eliminating read-for-ownership",
            },
            {
                "technique": "Software prefetch with PRFM",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "Insert PRFM PLDL1STRM instructions 2-3 cache lines ahead of the current "
                    "read position to hide memory latency. Use PSTL1STRM for store prefetch."
                ),
                "code_snippet": (
                    "PRFM PLDL1STRM, [X1, #256]  // prefetch source 4 lines ahead\n"
                    "PRFM PSTL1STRM, [X0, #256]  // prefetch dest for writing\n"
                    "LDP  Q0, Q1, [X1], #32\n"
                    "STP  Q0, Q1, [X0], #32"
                ),
                "speedup": "10-30% improvement on large copies (> L2 cache size)",
            },
        ],
        "compiler_flags": [
            "-O2",
            "-march=armv8-a",
            "-moutline-atomics  (for glibc memcpy selection)",
        ],
        "target_core_notes": {
            "cortex-a78": "128-bit load/store, 2x LDP/STP per cycle. Unroll 4x for peak bandwidth.",
            "neoverse-v2": "256-bit load/store paths, 2x LDP Q per cycle. DC ZVA is 64 bytes.",
            "cortex-a55": "64-bit load/store unit. LDP of X registers is optimal; Q-reg pairs may serialize.",
        },
    },
    "memset": {
        "description": (
            "Memory fill operation — writing a constant byte value across a block of memory. "
            "The zero case (memset to 0) is especially common and can exploit ARM's cache "
            "zeroing instructions for maximum throughput."
        ),
        "baseline_approach": (
            "A byte-by-byte store loop. The compiler may widen to 64-bit stores with STR, "
            "but does not exploit cache-line zeroing or NEON fill patterns."
        ),
        "optimizations": [
            {
                "technique": "STP of zero register pairs",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "Use STP XZR, XZR to store 16 zero bytes per instruction. The zero "
                    "register is free (no data dependency), so the core can issue these "
                    "at full store bandwidth."
                ),
                "code_snippet": (
                    "// Zero 64 bytes per iteration\n"
                    "STP  XZR, XZR, [X0], #16\n"
                    "STP  XZR, XZR, [X0], #16\n"
                    "STP  XZR, XZR, [X0], #16\n"
                    "STP  XZR, XZR, [X0], #16"
                ),
                "speedup": "8x over byte-by-byte STR",
            },
            {
                "technique": "DC ZVA for page-aligned zeroing",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "DC ZVA (Data Cache Zero by VA) zeroes an entire cache line (typically "
                    "64 bytes) without reading from memory first. For large page-aligned "
                    "regions this is the fastest zeroing method."
                ),
                "code_snippet": (
                    "// Zero entire pages using DC ZVA\n"
                    "// Read DCZID_EL0 for block size (usually 64 bytes)\n"
                    "loop:\n"
                    "  DC ZVA, X0\n"
                    "  ADD  X0, X0, #64\n"
                    "  CMP  X0, X1\n"
                    "  B.LT loop"
                ),
                "speedup": "2-4x over STP XZR pairs for large (>4KB) regions",
            },
            {
                "technique": "NEON fill for non-zero values",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "For non-zero fill values, broadcast the byte into a NEON Q register with "
                    "VDUP and use STP Q pairs to write 32 bytes per iteration."
                ),
                "code_snippet": (
                    "uint8x16_t fill = vdupq_n_u8(value);\n"
                    "for (size_t i = 0; i < len; i += 32) {\n"
                    "    vst1q_u8(dst + i, fill);\n"
                    "    vst1q_u8(dst + i + 16, fill);\n"
                    "}"
                ),
                "speedup": "4-8x over byte stores; works for any fill value",
            },
        ],
        "compiler_flags": [
            "-O2",
            "-march=armv8-a",
        ],
        "target_core_notes": {
            "cortex-a78": "DC ZVA block is 64 bytes. Two STP ports; unroll 4x for peak throughput.",
            "neoverse-v2": "DC ZVA block is 64 bytes. Can issue 2x128-bit stores per cycle.",
            "cortex-a55": "DC ZVA may be slower on in-order cores; benchmark before using on small buffers.",
        },
    },
    "dot_product": {
        "description": (
            "Dot product of two vectors: sum(a[i] * b[i]). Fundamental to DSP, ML inference, "
            "and linear algebra. ARM offers widening multiply-accumulate, dedicated dot-product "
            "instructions, and scalable vector reductions."
        ),
        "baseline_approach": (
            "A scalar loop with FP multiply and add. Without -ffast-math, the compiler cannot "
            "reorder the accumulation, limiting SIMD opportunities. Typically achieves 1 FLOP "
            "per cycle on a modern ARM core."
        ),
        "optimizations": [
            {
                "technique": "NEON FMLA accumulation",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "Load 4 FP32 values from each vector, multiply-accumulate into a NEON "
                    "accumulator, then reduce with FADDP at the end. Unroll 2-4x to hide "
                    "FMLA latency (typically 4 cycles)."
                ),
                "code_snippet": (
                    "float32x4_t acc0 = vdupq_n_f32(0);\n"
                    "float32x4_t acc1 = vdupq_n_f32(0);\n"
                    "for (int i = 0; i < N; i += 8) {\n"
                    "    acc0 = vfmaq_f32(acc0, vld1q_f32(&a[i]), vld1q_f32(&b[i]));\n"
                    "    acc1 = vfmaq_f32(acc1, vld1q_f32(&a[i+4]), vld1q_f32(&b[i+4]));\n"
                    "}\n"
                    "float32x4_t sum = vaddq_f32(acc0, acc1);\n"
                    "float result = vaddvq_f32(sum);  // horizontal add"
                ),
                "speedup": "4-8x over scalar (FP32), depends on unroll factor",
            },
            {
                "technique": "SDOT/UDOT for integer vectors",
                "architecture": "AArch64 (ARMv8.2+ with DotProd)",
                "description": (
                    "SDOT/UDOT compute 4-element dot products of int8 values, accumulating "
                    "into int32 lanes. Processes 16 int8 multiplies per instruction."
                ),
                "code_snippet": (
                    "int32x4_t acc = vdupq_n_s32(0);\n"
                    "for (int i = 0; i < N; i += 16) {\n"
                    "    int8x16_t va = vld1q_s8(&a[i]);\n"
                    "    int8x16_t vb = vld1q_s8(&b[i]);\n"
                    "    acc = vdotq_s32(acc, va, vb);\n"
                    "}\n"
                    "int32_t result = vaddvq_s32(acc);"
                ),
                "speedup": "16x over scalar int8 multiply-add",
            },
            {
                "technique": "SVE predicated dot product",
                "architecture": "AArch64 (ARMv8.2+ with SVE)",
                "description": (
                    "SVE predicated loops handle arbitrary vector lengths and automatically "
                    "mask the final iteration. Combine with FMLA for FP or SDOT for integer."
                ),
                "code_snippet": (
                    "svfloat32_t acc = svdup_f32(0);\n"
                    "for (int i = 0; i < N; i += svcntw()) {\n"
                    "    svbool_t pg = svwhilelt_b32(i, N);\n"
                    "    svfloat32_t va = svld1(pg, &a[i]);\n"
                    "    svfloat32_t vb = svld1(pg, &b[i]);\n"
                    "    acc = svmla_m(pg, acc, va, vb);\n"
                    "}\n"
                    "float result = svaddv(svptrue_b32(), acc);"
                ),
                "speedup": "Scales with SVE vector length; portable across implementations",
            },
            {
                "technique": "SME outer product for batched dot products",
                "architecture": "AArch64 (ARMv9.2+ with SME)",
                "description": (
                    "When computing multiple dot products (e.g. matrix-vector), SME FMOPA "
                    "accumulates an outer product tile. Each instruction does VL*VL FP MACs."
                ),
                "code_snippet": (
                    "FMOPA ZA0.S, P0/M, P1/M, Z0.S, Z1.S  // ZA += Z0 outer Z1"
                ),
                "speedup": "VL^2 FLOPs per instruction; ideal for batched workloads",
            },
        ],
        "compiler_flags": [
            "-O2",
            "-ffast-math  (allows reordering of FP accumulation)",
            "-march=armv8.2-a+dotprod",
            "-march=armv8.2-a+sve  (for SVE path)",
        ],
        "target_core_notes": {
            "cortex-a78": "2 FMLA pipes, 4-cycle latency. Unroll 4x accumulators for full throughput. SDOT supported.",
            "neoverse-v2": "4 FP pipes, 128-bit SVE2. SDOT throughput is 2 per cycle. Use SVE for loop vectorization.",
            "cortex-a55": "1 FMLA pipe, 4-cycle latency. 2 accumulators sufficient. SDOT available on ARMv8.2 variants.",
        },
    },
    "sort": {
        "description": (
            "Sorting arrays of integers or floats. Comparison-based sorts are branch-heavy; "
            "ARM provides branchless conditional operations and SIMD compare-and-swap networks "
            "for small-array optimization."
        ),
        "baseline_approach": (
            "Standard qsort or std::sort with scalar comparisons and conditional branches. "
            "Branch mispredictions on random data cause ~15 cycle penalties per mispredicted "
            "branch on modern ARM cores."
        ),
        "optimizations": [
            {
                "technique": "NEON compare-and-swap for small arrays",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "For small arrays (4-16 elements), use NEON CMGT/BSL to build a sorting "
                    "network entirely in SIMD registers. No branches needed — the compare-and-swap "
                    "is done with bitwise select."
                ),
                "code_snippet": (
                    "// Branchless min/max of two NEON vectors\n"
                    "uint32x4_t mask = vcgtq_u32(a, b);\n"
                    "uint32x4_t lo = vbslq_u32(mask, b, a);  // min\n"
                    "uint32x4_t hi = vbslq_u32(mask, a, b);  // max"
                ),
                "speedup": "2-4x for sorting 4-16 elements vs scalar sort",
            },
            {
                "technique": "CSEL for branchless min/max",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "Replace branch-based comparisons with CMP+CSEL (conditional select). "
                    "Eliminates branch misprediction entirely for comparison-heavy inner loops."
                ),
                "code_snippet": (
                    "CMP  W0, W1\n"
                    "CSEL W2, W0, W1, LT  // W2 = min(W0, W1)\n"
                    "CSEL W3, W1, W0, LT  // W3 = max(W0, W1)"
                ),
                "speedup": "Eliminates ~15 cycle branch misprediction penalty per compare",
            },
            {
                "technique": "Prefetch for large-array sorting",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "During partitioning (quicksort) or merging (mergesort), prefetch the "
                    "next cache line of each input stream with PRFM to hide memory latency."
                ),
                "code_snippet": (
                    "PRFM PLDL1KEEP, [X0, #128]  // prefetch left partition\n"
                    "PRFM PLDL1KEEP, [X1, #128]  // prefetch right partition"
                ),
                "speedup": "10-20% for arrays larger than L2 cache",
            },
        ],
        "compiler_flags": [
            "-O2",
            "-march=armv8-a",
        ],
        "target_core_notes": {
            "cortex-a78": "Good branch predictor (TAGE-like), but random data still causes ~10% misprediction. Use CSEL.",
            "neoverse-v2": "Deep OoO buffer tolerates some misprediction. NEON sorting networks help on small partitions.",
            "cortex-a55": "In-order core; branch misprediction very costly (~10 cycles). CSEL is essential.",
        },
    },
    "string_search": {
        "description": (
            "Searching for a substring or byte pattern within a larger string/buffer. "
            "ARM NEON can compare 16 bytes in parallel, and SVE can compare even wider "
            "vectors with first-fault loads for safe overreads."
        ),
        "baseline_approach": (
            "Byte-by-byte comparison (naive strstr) or compiler-generated LDRB+CMP loop. "
            "Processes 1 byte per cycle at best, far below the memory bandwidth available."
        ),
        "optimizations": [
            {
                "technique": "NEON bytewise compare (16 bytes at a time)",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "Load 16 bytes, broadcast the search byte with VDUP, compare with VCEQ, "
                    "then extract a bitmask with VSHRN+UMOV to find match positions. Processes "
                    "16 bytes per iteration."
                ),
                "code_snippet": (
                    "uint8x16_t needle = vdupq_n_u8(search_byte);\n"
                    "for (size_t i = 0; i < len; i += 16) {\n"
                    "    uint8x16_t hay = vld1q_u8(&str[i]);\n"
                    "    uint8x16_t cmp = vceqq_u8(hay, needle);\n"
                    "    // Extract match positions from cmp\n"
                    "    uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(cmp), 4)), 0);\n"
                    "    if (mask) return i + __builtin_ctzll(mask) / 4;\n"
                    "}"
                ),
                "speedup": "8-16x over byte-by-byte search",
            },
            {
                "technique": "CRC32 for rolling hash (Rabin-Karp)",
                "architecture": "AArch64 (ARMv8.1+)",
                "description": (
                    "Use the CRC32 instruction to compute a rolling hash of the search window. "
                    "CRC32 has 3-cycle latency on most ARM cores and produces excellent hash "
                    "distribution for string matching."
                ),
                "code_snippet": (
                    "uint32_t hash = 0;\n"
                    "for (int i = 0; i < pat_len; i++)\n"
                    "    hash = __crc32b(hash, str[i]);\n"
                    "// Roll the hash for each position\n"
                    "// CRC32B X0, W0, W1"
                ),
                "speedup": "Efficient for multi-pattern search; O(n) average",
            },
            {
                "technique": "SVE match/search instructions",
                "architecture": "AArch64 (ARMv8.2+ with SVE2)",
                "description": (
                    "SVE2 provides MATCH instruction that compares each byte in a vector against "
                    "a set of search characters. Combined with first-fault loads (LDFF1) for "
                    "safe buffer-end handling without length checks."
                ),
                "code_snippet": (
                    "svuint8_t needle = svdup_u8(search_byte);\n"
                    "svbool_t pg = svptrue_b8();\n"
                    "svuint8_t data = svldff1(pg, &str[i]);\n"
                    "svbool_t match = svcmpeq(pg, data, needle);\n"
                    "if (svptest_any(pg, match))\n"
                    "    return i + svclastb(match, -1, svindex_u8(0, 1));"
                ),
                "speedup": "Scales with SVE vector length; safe end-of-buffer handling",
            },
        ],
        "compiler_flags": [
            "-O2",
            "-march=armv8.1-a+crc",
            "-march=armv8.2-a+sve2  (for SVE2 match)",
        ],
        "target_core_notes": {
            "cortex-a78": "NEON 16-byte compare is 1 cycle. CRC32 is 3 cycles. Good for memchr-style searches.",
            "neoverse-v2": "SVE2 MATCH instruction available. 128-bit SVE but efficient predication.",
            "cortex-a55": "NEON compare works well; avoid complex extraction — use simpler UMAXV != 0 check first.",
        },
    },
    "crc32": {
        "description": (
            "CRC-32 and CRC-32C checksum computation. Used in networking (Ethernet, iSCSI), "
            "storage (ext4, Btrfs), and data integrity checks. ARM provides dedicated CRC "
            "instructions and polynomial multiply for fast CRC."
        ),
        "baseline_approach": (
            "Table-driven CRC computation processing 1 byte per iteration. Requires a 256-entry "
            "lookup table and has a loop-carried dependency on the CRC accumulator, limiting "
            "throughput to ~1 byte per 3-4 cycles."
        ),
        "optimizations": [
            {
                "technique": "CRC32 instruction (hardware)",
                "architecture": "AArch64 (ARMv8.1+ with CRC)",
                "description": (
                    "ARMv8.1 added CRC32B/CRC32H/CRC32W/CRC32X instructions that compute "
                    "CRC-32 on 1/2/4/8 bytes per instruction. CRC32CX processes 8 bytes in "
                    "~3 cycles."
                ),
                "code_snippet": (
                    "uint32_t crc = ~0;\n"
                    "while (len >= 8) {\n"
                    "    crc = __crc32d(crc, *(uint64_t*)buf);\n"
                    "    buf += 8; len -= 8;\n"
                    "}\n"
                    "while (len--) crc = __crc32b(crc, *buf++);"
                ),
                "speedup": "8x over table-lookup per byte; ~2.7 bytes/cycle",
            },
            {
                "technique": "PMULL for CRC-32C (polynomial multiply)",
                "architecture": "AArch64 (ARMv8.0+ with Crypto)",
                "description": (
                    "Use PMULL (polynomial multiply long) to process 16 bytes at a time with "
                    "carryless multiplication. Requires precomputed folding constants specific "
                    "to the CRC polynomial."
                ),
                "code_snippet": (
                    "// Fold 128 bits using PMULL\n"
                    "poly128_t fold = vmull_p64(vget_low_p64(data), k1_k2);\n"
                    "// XOR with next 128-bit block\n"
                    "data = veorq_u8(vreinterpretq_u8_p128(fold), next_block);"
                ),
                "speedup": "4-8x over CRC32 instruction; up to 16 bytes/cycle",
            },
            {
                "technique": "3-way interleaved pipeline",
                "architecture": "AArch64 (ARMv8.1+ with CRC)",
                "description": (
                    "Split the buffer into 3 segments and compute CRC32 on each segment "
                    "independently, then combine with polynomial reduction. This breaks the "
                    "loop-carried dependency, allowing 3 CRC32 instructions in flight."
                ),
                "code_snippet": (
                    "// Process 3 segments in parallel\n"
                    "uint32_t crc0 = ~0, crc1 = 0, crc2 = 0;\n"
                    "size_t seg = len / 3;\n"
                    "for (size_t i = 0; i < seg; i += 8) {\n"
                    "    crc0 = __crc32d(crc0, *(uint64_t*)(buf + i));\n"
                    "    crc1 = __crc32d(crc1, *(uint64_t*)(buf + seg + i));\n"
                    "    crc2 = __crc32d(crc2, *(uint64_t*)(buf + 2*seg + i));\n"
                    "}\n"
                    "// Combine with polynomial multiply reduction"
                ),
                "speedup": "~3x over single-stream CRC32; approaches memory bandwidth limit",
            },
        ],
        "compiler_flags": [
            "-O2",
            "-march=armv8.1-a+crc",
            "-march=armv8-a+crc+crypto  (for PMULL path)",
        ],
        "target_core_notes": {
            "cortex-a78": "CRC32 is 3-cycle latency, 1/cycle throughput. 3-way interleave achieves ~8 bytes/cycle.",
            "neoverse-v2": "CRC32 is 3-cycle latency. PMULL is 2 cycles. Use PMULL path for large buffers.",
            "cortex-a55": "CRC32 is 3-cycle latency. Single-stream CRC32X is usually sufficient for small buffers.",
        },
    },
    "aes_encrypt": {
        "description": (
            "AES block cipher encryption (AES-128/192/256). ARM Crypto Extensions provide "
            "dedicated instructions that fuse SubBytes+ShiftRows (AESE) and MixColumns (AESMC) "
            "for high-throughput encryption."
        ),
        "baseline_approach": (
            "Software AES using lookup tables (T-tables). Vulnerable to cache-timing side "
            "channels. Processes one block (~10 rounds) in ~200 cycles on a modern ARM core."
        ),
        "optimizations": [
            {
                "technique": "AESE+AESMC fused pair",
                "architecture": "AArch64 (ARMv8.0+ with Crypto)",
                "description": (
                    "AESE performs SubBytes+ShiftRows+AddRoundKey; AESMC performs MixColumns. "
                    "On most ARM cores, an AESE immediately followed by AESMC can fuse or "
                    "pipeline efficiently, reducing the per-round cost."
                ),
                "code_snippet": (
                    "// One AES round\n"
                    "uint8x16_t state = vaeseq_u8(state, round_key);  // AESE\n"
                    "state = vaesmcq_u8(state);                        // AESMC\n"
                    "// Repeat for each round (10/12/14 for AES-128/192/256)"
                ),
                "speedup": "10-20x over T-table software AES; also eliminates timing side channels",
            },
            {
                "technique": "Pipeline 4 blocks in parallel (CTR/ECB mode)",
                "architecture": "AArch64 (ARMv8.0+ with Crypto)",
                "description": (
                    "In CTR or ECB mode, blocks are independent. Process 4 blocks simultaneously "
                    "to hide the AESE+AESMC pipeline latency. Each block uses a separate set "
                    "of NEON registers."
                ),
                "code_snippet": (
                    "// 4-block parallel AES-128 in CTR mode\n"
                    "for (int r = 0; r < 9; r++) {\n"
                    "    s0 = vaesmcq_u8(vaeseq_u8(s0, rk[r]));\n"
                    "    s1 = vaesmcq_u8(vaeseq_u8(s1, rk[r]));\n"
                    "    s2 = vaesmcq_u8(vaeseq_u8(s2, rk[r]));\n"
                    "    s3 = vaesmcq_u8(vaeseq_u8(s3, rk[r]));\n"
                    "}\n"
                    "// Final round (no AESMC)\n"
                    "s0 = vaeseq_u8(s0, rk[9]) ^ rk[10];\n"
                    "// ... same for s1, s2, s3"
                ),
                "speedup": "2-4x over single-block pipeline; approaches 1 byte/cycle",
            },
            {
                "technique": "ARMv8 Crypto Extensions full pipeline",
                "architecture": "AArch64 (ARMv8.0+ with Crypto)",
                "description": (
                    "Combine AESE/AESMC with AESD/AESIMC for decrypt. Use interleaved "
                    "encrypt/MAC operations for AES-GCM (AESE + PMULL for GHASH). The crypto "
                    "extensions are constant-time, preventing cache-timing attacks."
                ),
                "code_snippet": (
                    "// AES-GCM: interleave AES-CTR and GHASH\n"
                    "state = vaesmcq_u8(vaeseq_u8(state, rk));\n"
                    "ghash = vmull_p64(ghash_lo, H_lo);  // PMULL for GHASH\n"
                    "// Pipeline keeps both AES and PMULL units busy"
                ),
                "speedup": "AES-GCM at near AES-CTR speed; constant-time security guarantee",
            },
        ],
        "compiler_flags": [
            "-O2",
            "-march=armv8-a+crypto",
        ],
        "target_core_notes": {
            "cortex-a78": "AESE+AESMC fuse to 3-cycle latency. 4-block interleave achieves ~1 cycle/byte for AES-128.",
            "neoverse-v2": "2 AES pipes, AESE+AESMC 2-cycle fused latency. Best throughput with 4-8 block interleave.",
            "cortex-a55": "AESE is 3-cycle latency. 2-block interleave sufficient. Still 10x faster than software AES.",
        },
    },
    "sha256": {
        "description": (
            "SHA-256 cryptographic hash computation. ARM provides dedicated SHA-256 "
            "instructions that compute hash rounds directly in hardware, matching the "
            "algorithm's structure."
        ),
        "baseline_approach": (
            "Software SHA-256 with 64 rounds of scalar operations per block. Each round "
            "requires multiple shifts, adds, and logical operations. ~40 cycles/byte on "
            "a fast ARM core."
        ),
        "optimizations": [
            {
                "technique": "SHA256H/SHA256H2/SHA256SU0/SHA256SU1 instruction sequence",
                "architecture": "AArch64 (ARMv8.0+ with Crypto/SHA2)",
                "description": (
                    "Four dedicated instructions implement the SHA-256 algorithm:\n"
                    "  SHA256H: hash update (rounds) for first half of state\n"
                    "  SHA256H2: hash update for second half of state\n"
                    "  SHA256SU0: schedule update part 1\n"
                    "  SHA256SU1: schedule update part 2\n"
                    "Together they compute 4 SHA-256 rounds per iteration."
                ),
                "code_snippet": (
                    "// 4 SHA-256 rounds per iteration\n"
                    "uint32x4_t msg = vld1q_u32(&block[i]);\n"
                    "uint32x4_t tmp = vaddq_u32(msg, vld1q_u32(&K[i]));\n"
                    "uint32x4_t state0_saved = state0;\n"
                    "state0 = vsha256hq_u32(state0, state1, tmp);\n"
                    "state1 = vsha256h2q_u32(state1, state0_saved, tmp);\n"
                    "msg0 = vsha256su0q_u32(msg0, msg1);\n"
                    "msg0 = vsha256su1q_u32(msg0, msg2, msg3);"
                ),
                "speedup": "3-5x over software SHA-256; ~8-12 cycles/byte",
            },
            {
                "technique": "Hardware acceleration with multi-block pipelining",
                "architecture": "AArch64 (ARMv8.0+ with Crypto/SHA2)",
                "description": (
                    "For long messages, pipeline the schedule update of the next block "
                    "with the hash computation of the current block. SHA256SU0/SU1 can "
                    "execute in parallel with SHA256H/H2 on cores with 2 crypto pipes."
                ),
                "code_snippet": (
                    "// Interleave current block hash with next block schedule\n"
                    "state0 = vsha256hq_u32(state0, state1, tmp_curr);\n"
                    "next_msg0 = vsha256su0q_u32(next_msg0, next_msg1);  // parallel\n"
                    "state1 = vsha256h2q_u32(state1, state0_saved, tmp_curr);\n"
                    "next_msg0 = vsha256su1q_u32(next_msg0, next_msg2, next_msg3);"
                ),
                "speedup": "Additional 20-50% over single-block on dual-pipe cores",
            },
        ],
        "compiler_flags": [
            "-O2",
            "-march=armv8-a+sha2",
            "-march=armv8-a+crypto  (includes SHA2)",
        ],
        "target_core_notes": {
            "cortex-a78": "SHA256H is 4-cycle latency. Schedule and hash overlap well. ~8 cycles/byte.",
            "neoverse-v2": "2 SHA pipes, SHA256H 2-cycle throughput. ~4 cycles/byte with interleaving.",
            "cortex-a55": "SHA256H is 4-cycle latency, single pipe. Still 3x faster than software. ~12 cycles/byte.",
        },
    },
    "linked_list_traversal": {
        "description": (
            "Traversing a linked list — following pointer chains through memory. This is "
            "inherently latency-bound due to pointer-chasing. ARM provides prefetch hints "
            "and pointer authentication to optimize and secure traversals."
        ),
        "baseline_approach": (
            "A simple loop: node = node->next. Each load depends on the previous load's "
            "result, creating a serial dependency chain. On modern ARM cores with ~4-cycle "
            "L1 hit latency, this limits throughput to 1 node per 4 cycles."
        ),
        "optimizations": [
            {
                "technique": "PRFM for prefetch next node",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "Prefetch the next-next node while processing the current node. If the "
                    "node structure contains a 'next' pointer, issue PRFM on node->next->next "
                    "to overlap the memory access with computation."
                ),
                "code_snippet": (
                    "while (node) {\n"
                    "    if (node->next)\n"
                    "        __builtin_prefetch(node->next->next, 0, 3);\n"
                    "    process(node);\n"
                    "    node = node->next;\n"
                    "}\n"
                    "// ARM assembly: PRFM PLDL1KEEP, [X0, #NEXT_OFFSET]"
                ),
                "speedup": "20-50% if nodes are not in L1 cache; minimal effect if cache-hot",
            },
            {
                "technique": "Pointer authentication for security (PAC)",
                "architecture": "AArch64 (ARMv8.3+ with PAuth)",
                "description": (
                    "Use pointer authentication to protect next pointers from corruption. "
                    "PACIA/AUTIA sign and verify pointers with minimal overhead (~1 cycle). "
                    "Prevents pointer-hijacking attacks on linked structures."
                ),
                "code_snippet": (
                    "// Sign pointer when inserting\n"
                    "node->next = __builtin_arm_pacia(new_next, context);\n"
                    "// Authenticate pointer when traversing\n"
                    "struct node *next = __builtin_arm_autia(node->next, context);\n"
                    "// PACIA X0, X1  /  AUTIA X0, X1"
                ),
                "speedup": "Negligible overhead (~1 cycle); significant security improvement",
            },
            {
                "technique": "Avoid branch mispredicts with CBNZ",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "Use CBNZ (Compare and Branch if Not Zero) for the end-of-list check. "
                    "CBNZ fuses the comparison and branch into a single operation, and the "
                    "branch predictor can specialize on the pattern (long runs of taken)."
                ),
                "code_snippet": (
                    "loop:\n"
                    "  LDR  X0, [X0, #NEXT_OFFSET]  // node = node->next\n"
                    "  // ... process node ...\n"
                    "  CBNZ X0, loop                 // continue if not null"
                ),
                "speedup": "Saves 1 instruction vs CMP+B.NE; better branch prediction hint",
            },
        ],
        "compiler_flags": [
            "-O2",
            "-march=armv8.3-a  (for PAC)",
            "-mbranch-protection=standard  (enables PAC for return addresses)",
        ],
        "target_core_notes": {
            "cortex-a78": "4-cycle L1 load latency, good prefetcher. Software prefetch helps if stride is irregular.",
            "neoverse-v2": "4-cycle L1 latency, aggressive hardware prefetcher. Software PRFM less needed.",
            "cortex-a55": "4-cycle L1 latency, limited prefetcher. Software PRFM very beneficial for pointer chasing.",
        },
    },
    "atomic_counter": {
        "description": (
            "Atomic read-modify-write operations on shared counters. ARM provides both "
            "traditional exclusive-access loops (LDXR/STXR) and Large System Extensions (LSE) "
            "atomics that execute atomically in the cache/interconnect."
        ),
        "baseline_approach": (
            "A LDXR/STXR loop: load-exclusive, modify, store-exclusive, retry if store fails. "
            "Under contention, the STXR fails frequently, causing retries and wasted work. "
            "Each retry costs ~10-20 cycles."
        ),
        "optimizations": [
            {
                "technique": "LDADD (ARMv8.1 LSE atomics)",
                "architecture": "AArch64 (ARMv8.1+ with LSE)",
                "description": (
                    "LDADD performs an atomic load-add in a single instruction. The "
                    "interconnect/cache handles the read-modify-write atomically, eliminating "
                    "retry loops. LDADDA/LDADDAL add acquire/release semantics."
                ),
                "code_snippet": (
                    "// C11 atomics — compiler emits LDADD with -march=armv8.1-a\n"
                    "atomic_fetch_add(&counter, 1);\n"
                    "\n"
                    "// Direct intrinsic\n"
                    "LDADD  W0, W1, [X2]   // W1 = *X2; *X2 += W0 (atomically)\n"
                    "LDADDAL W0, W1, [X2]  // with acquire-release ordering"
                ),
                "speedup": "2-10x under contention vs LDXR/STXR loop; no retry overhead",
            },
            {
                "technique": "CASP for double-width CAS",
                "architecture": "AArch64 (ARMv8.1+ with LSE)",
                "description": (
                    "CASP (Compare-And-Swap Pair) atomically operates on 128 bits (two 64-bit "
                    "registers). Useful for lock-free data structures that need to update a "
                    "pointer and a counter atomically."
                ),
                "code_snippet": (
                    "// 128-bit CAS: atomically update {pointer, counter}\n"
                    "CASPAL X0, X1, X2, X3, [X4]\n"
                    "// Compares [X4] with {X0,X1}; if equal, stores {X2,X3}"
                ),
                "speedup": "Enables lock-free 128-bit updates; replaces LDXP/STXP loops",
            },
            {
                "technique": "SWP (atomic swap)",
                "architecture": "AArch64 (ARMv8.1+ with LSE)",
                "description": (
                    "SWP atomically swaps a register with memory. Useful for spinlock "
                    "implementations and queue operations where you need to exchange values."
                ),
                "code_snippet": (
                    "// Atomic swap: X0 = old *X1; *X1 = X0\n"
                    "SWP   X0, X0, [X1]    // relaxed\n"
                    "SWPAL X0, X0, [X1]    // acquire-release\n"
                    "\n"
                    "// Spinlock acquire\n"
                    "MOV  W0, #1\n"
                    "SWPA W0, W0, [X1]  // acquire semantics\n"
                    "CBNZ W0, spin      // retry if was locked"
                ),
                "speedup": "Simpler spinlocks; avoids LDXR/STXR retry loop for exchanges",
            },
        ],
        "compiler_flags": [
            "-O2",
            "-march=armv8.1-a  (enables LSE atomics)",
            "-moutline-atomics  (runtime LSE detection on Linux)",
        ],
        "target_core_notes": {
            "cortex-a78": "LSE atomics in cache controller. LDADD is 1 instruction, ~4 cycles uncontended.",
            "neoverse-v2": "LSE2 support (naturally-aligned atomics). LDADD/SWP near-atomic at L1. CASP for 128-bit.",
            "cortex-a55": "LSE supported on ARMv8.2 variants. LDADD still cheaper than LDXR/STXR under contention.",
        },
    },
    "simd_reduction": {
        "description": (
            "Reducing a SIMD vector to a single scalar value (sum, max, min). AArch64 "
            "provides efficient across-lane operations that avoid the shuffle-heavy "
            "reduction patterns needed on other architectures."
        ),
        "baseline_approach": (
            "After SIMD computation, extract each lane and reduce in scalar code. On a "
            "4-lane FP32 vector, this means 3 scalar FADD instructions in sequence — "
            "slow and defeating the purpose of SIMD."
        ),
        "optimizations": [
            {
                "technique": "FADDP pairwise reduction",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "FADDP (Floating-point Add Pairwise) adds adjacent pairs, halving the "
                    "number of elements each step. Two FADDP instructions reduce a 4-lane "
                    "FP32 vector to a scalar."
                ),
                "code_snippet": (
                    "float32x4_t v = ...;  // {a, b, c, d}\n"
                    "float32x2_t p = vadd_f32(vget_high_f32(v), vget_low_f32(v));  // {a+c, b+d}\n"
                    "float32x2_t s = vpadd_f32(p, p);  // {a+b+c+d, ...}\n"
                    "float result = vget_lane_f32(s, 0);\n"
                    "\n"
                    "// Or use vaddvq_f32 (AArch64 only):\n"
                    "float result = vaddvq_f32(v);  // single instruction"
                ),
                "speedup": "3-4x over scalar extraction and add",
            },
            {
                "technique": "ADDV/FMAXV/FMINV across-lane operations",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "AArch64 provides single-instruction across-lane reductions: ADDV (integer "
                    "sum), FMAXV (FP max), FMINV (FP min), SMAXV/SMINV (signed max/min), "
                    "UMAXV/UMINV (unsigned max/min). These are unique to AArch64 — AArch32 "
                    "NEON does not have them."
                ),
                "code_snippet": (
                    "// Integer sum reduction\n"
                    "int32x4_t v = ...;\n"
                    "int32_t sum = vaddvq_s32(v);  // ADDV: sum all 4 lanes\n"
                    "\n"
                    "// FP max reduction\n"
                    "float32x4_t fv = ...;\n"
                    "float max_val = vmaxvq_f32(fv);  // FMAXV: max of 4 lanes\n"
                    "\n"
                    "// Unsigned min reduction\n"
                    "uint8x16_t bv = ...;\n"
                    "uint8_t min_byte = vminvq_u8(bv);  // UMINV: min of 16 bytes"
                ),
                "speedup": "Single instruction vs 3-4 instruction pairwise tree; lower latency",
            },
            {
                "technique": "Tree reduction for multi-register accumulation",
                "architecture": "AArch64 (ARMv8.0+)",
                "description": (
                    "When using multiple accumulator registers (for latency hiding), combine "
                    "them in a tree pattern: add pairs of accumulators, then reduce the final "
                    "vector. This minimizes dependency chains."
                ),
                "code_snippet": (
                    "// 4 accumulators -> 1 scalar\n"
                    "float32x4_t acc0, acc1, acc2, acc3;\n"
                    "// ... loop fills all 4 accumulators ...\n"
                    "float32x4_t sum01 = vaddq_f32(acc0, acc1);\n"
                    "float32x4_t sum23 = vaddq_f32(acc2, acc3);\n"
                    "float32x4_t total = vaddq_f32(sum01, sum23);\n"
                    "float result = vaddvq_f32(total);"
                ),
                "speedup": "Reduces 16 FP32 values to scalar in ~6 instructions (log2 tree)",
            },
        ],
        "compiler_flags": [
            "-O2",
            "-march=armv8-a",
            "-ffast-math  (allows FP reassociation for tree reduction)",
        ],
        "target_core_notes": {
            "cortex-a78": "FADDP is 2-cycle latency. ADDV/FMAXV are 3-4 cycles. Use pairwise for lowest latency.",
            "neoverse-v2": "ADDV is 4 cycles for 128-bit. SVE FADDV provides scalable reduction across any VL.",
            "cortex-a55": "ADDV is 4 cycles. Single NEON pipe — tree reduction has same latency as sequential.",
        },
    },
}


def _format_optimization_overview() -> str:
    """Return a table of all optimization patterns."""
    lines = []
    lines.append("# ARM Code Optimization Patterns")
    lines.append("")
    lines.append("| # | Pattern | Description |")
    lines.append("|---|---------|-------------|")
    for idx, (name, data) in enumerate(_OPTIMIZATION_PATTERNS.items(), 1):
        short_desc = data["description"][:80].rstrip()
        if len(data["description"]) > 80:
            short_desc += "..."
        lines.append(f"| {idx} | `{name}` | {short_desc} |")
    lines.append("")
    lines.append(f"Total patterns: {len(_OPTIMIZATION_PATTERNS)}")
    lines.append("")
    lines.append("Use `suggest_optimization(\"<pattern_name>\")` for detailed optimization guide.")
    lines.append("Use `suggest_optimization(\"<pattern_name>\", target_core=\"cortex-a78\")` for core-specific advice.")
    return "\n".join(lines)


def _format_optimization(name: str, data: dict, target_core: str | None) -> str:
    """Format a single optimization pattern into a readable guide."""
    lines = []
    title = name.replace("_", " ").title()
    lines.append(f"# ARM Optimization Guide: {title}")
    lines.append("")

    # Pattern Description
    lines.append("## Pattern Description")
    lines.append(data["description"])
    lines.append("")

    # Baseline
    lines.append("## Baseline (Unoptimized)")
    lines.append(data["baseline_approach"])
    lines.append("")

    # Optimization Techniques
    lines.append("## Optimization Techniques")
    lines.append("")
    for idx, opt in enumerate(data["optimizations"], 1):
        lines.append(f"### {idx}. {opt['technique']} ({opt['architecture']})")
        lines.append(opt["description"])
        lines.append("")
        lines.append("```c")
        lines.append(opt["code_snippet"])
        lines.append("```")
        lines.append("")
        lines.append(f"Expected speedup: {opt['speedup']}")
        lines.append("")

    # Compiler Flags
    lines.append("## Recommended Compiler Flags")
    for flag in data["compiler_flags"]:
        lines.append(f"- `{flag}`")
    lines.append("")

    # Core-Specific Notes
    core_notes = data.get("target_core_notes", {})
    if core_notes:
        lines.append("## Core-Specific Notes")
        lines.append("")
        if target_core:
            # Normalize the target core name for lookup
            core_key = target_core.strip().lower()
            found = False
            for core_name, note in core_notes.items():
                if core_name.lower() == core_key:
                    display_name = core_name.replace("-", "-").title()
                    # Ensure proper capitalization like Cortex-A78
                    display_name = core_name
                    for prefix in ["cortex-", "neoverse-"]:
                        if core_name.lower().startswith(prefix):
                            display_name = prefix.title().rstrip("-") + "-" + core_name[len(prefix):].upper()
                            break
                    lines.append(f"### {display_name} (selected)")
                    lines.append(note)
                    lines.append("")
                    found = True
                    break
            if not found:
                lines.append(f"Note: Core '{target_core}' not found in specific notes for this pattern.")
                lines.append("Showing all available core notes:")
                lines.append("")
                for core_name, note in core_notes.items():
                    lines.append(f"### {core_name}")
                    lines.append(note)
                    lines.append("")
        else:
            for core_name, note in core_notes.items():
                lines.append(f"### {core_name}")
                lines.append(note)
                lines.append("")

    return "\n".join(lines)


@mcp.tool()
def suggest_optimization(code_pattern: str, target_core: str | None = None) -> str:
    """Suggest ARM-specific optimizations for a given code pattern.

    Given a common code pattern name, returns detailed ARM optimization suggestions
    including NEON vectorization, SVE scalable loops, and architecture-specific features.

    Args:
        code_pattern: The code pattern to optimize. Use "list" or "overview" to see
            all available patterns. Valid patterns include:
            "matrix_multiply" -- Dense matrix multiplication with NEON/SVE/SME
            "memcpy" -- Memory copy with LDP/STP pairs, NEON, DC ZVA, prefetch
            "memset" -- Memory fill with zero-register pairs, DC ZVA, NEON fill
            "dot_product" -- Vector dot product with FMLA, SDOT/UDOT, SVE, SME
            "sort" -- Sorting with NEON compare-and-swap, CSEL, prefetch
            "string_search" -- String search with NEON compare, CRC32 hash, SVE match
            "crc32" -- CRC-32 with hardware CRC, PMULL, interleaved pipeline
            "aes_encrypt" -- AES with AESE+AESMC, parallel blocks, Crypto Extensions
            "sha256" -- SHA-256 with dedicated SHA instructions
            "linked_list_traversal" -- Pointer chasing with prefetch, PAC, CBNZ
            "atomic_counter" -- Atomics with LSE (LDADD, CASP, SWP)
            "simd_reduction" -- SIMD reduction with FADDP, ADDV/FMAXV, tree reduction
        target_core: Optional ARM core name (e.g. "cortex-a78", "neoverse-v2",
            "cortex-a55"). If provided, highlights core-specific optimization notes.
    """
    key = code_pattern.strip().lower()

    if key in ("list", "overview"):
        return _format_optimization_overview()

    if key not in _OPTIMIZATION_PATTERNS:
        valid = ", ".join(sorted(_OPTIMIZATION_PATTERNS.keys()))
        return (
            f"Error: '{code_pattern}' is not a recognized optimization pattern.\n\n"
            f"Valid patterns: {valid}\n\n"
            f"Use suggest_optimization(\"list\") for an overview of all patterns."
        )

    return _format_optimization(key, _OPTIMIZATION_PATTERNS[key], target_core)

# ---------------------------------------------------------------------------
# Tool 18: lookup_system_register — AArch64 system register reference
# ---------------------------------------------------------------------------

_SYSTEM_REGISTERS: dict[str, dict] = {
    # ── Memory Management ──────────────────────────────────────────────────
    "SCTLR_EL1": {
        "full_name": "System Control Register",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C1_C0_0",
        "width": 64,
        "category": "memory_management",
        "description": (
            "Controls architectural features including MMU enable, data/instruction "
            "caches, alignment checking, endianness, and various memory-system behaviours "
            "at EL1/EL0."
        ),
        "key_fields": [
            {"bits": "[0]", "name": "M", "description": "MMU enable. 0=disabled, 1=enabled."},
            {"bits": "[2]", "name": "C", "description": "Data cache enable."},
            {"bits": "[12]", "name": "I", "description": "Instruction cache enable."},
            {"bits": "[1]", "name": "A", "description": "Alignment check enable."},
            {"bits": "[25]", "name": "EE", "description": "Exception endianness. 0=little-endian, 1=big-endian."},
        ],
        "usage_example": "MRS X0, SCTLR_EL1    // Read system control register\nMSR SCTLR_EL1, X0    // Write system control register",
    },
    "SCTLR_EL2": {
        "full_name": "System Control Register",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C1_C0_0",
        "width": 64,
        "category": "memory_management",
        "description": "Controls architectural features at EL2 (hypervisor). Same layout as SCTLR_EL1 but governs EL2 behaviour.",
        "key_fields": [
            {"bits": "[0]", "name": "M", "description": "MMU enable for EL2."},
            {"bits": "[2]", "name": "C", "description": "Data cache enable for EL2."},
        ],
        "usage_example": "MRS X0, SCTLR_EL2\nMSR SCTLR_EL2, X0",
    },
    "SCTLR_EL3": {
        "full_name": "System Control Register",
        "el": "EL3",
        "access": "RW",
        "encoding": "S3_6_C1_C0_0",
        "width": 64,
        "category": "memory_management",
        "description": "Controls architectural features at EL3 (secure monitor).",
        "key_fields": [
            {"bits": "[0]", "name": "M", "description": "MMU enable for EL3."},
            {"bits": "[2]", "name": "C", "description": "Data cache enable for EL3."},
        ],
        "usage_example": "MRS X0, SCTLR_EL3\nMSR SCTLR_EL3, X0",
    },
    "TCR_EL1": {
        "full_name": "Translation Control Register",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C2_C0_2",
        "width": 64,
        "category": "memory_management",
        "description": (
            "Controls translation table walks for EL0/EL1 address translation. "
            "Configures granule size, region sizes (T0SZ/T1SZ), cacheability and "
            "shareability of page-table walks, and intermediate physical address size."
        ),
        "key_fields": [
            {"bits": "[5:0]", "name": "T0SZ", "description": "Size offset for TTBR0_EL1 region (2^(64-T0SZ) VA range)."},
            {"bits": "[21:16]", "name": "T1SZ", "description": "Size offset for TTBR1_EL1 region."},
            {"bits": "[15:14]", "name": "TG0", "description": "Granule size for TTBR0 (00=4KB, 01=64KB, 10=16KB)."},
            {"bits": "[31:30]", "name": "TG1", "description": "Granule size for TTBR1 (01=16KB, 10=4KB, 11=64KB)."},
            {"bits": "[34:32]", "name": "IPS", "description": "Intermediate Physical Address Size (000=32b, 101=48b, 110=52b)."},
        ],
        "usage_example": "MRS X0, TCR_EL1\nMSR TCR_EL1, X0",
    },
    "TCR_EL2": {
        "full_name": "Translation Control Register",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C2_C0_2",
        "width": 64,
        "category": "memory_management",
        "description": "Controls translation table walks for EL2 address translation.",
        "key_fields": [
            {"bits": "[5:0]", "name": "T0SZ", "description": "Size offset for TTBR0_EL2 region."},
            {"bits": "[15:14]", "name": "TG0", "description": "Granule size for TTBR0_EL2."},
        ],
        "usage_example": "MRS X0, TCR_EL2\nMSR TCR_EL2, X0",
    },
    "TTBR0_EL1": {
        "full_name": "Translation Table Base Register 0",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C2_C0_0",
        "width": 64,
        "category": "memory_management",
        "description": "Holds the base address of the translation table for the lower VA range (typically user space).",
        "key_fields": [
            {"bits": "[47:1]", "name": "BADDR", "description": "Translation table base address."},
            {"bits": "[63:48]", "name": "ASID", "description": "Address Space Identifier (when TCR_EL1.A1=0)."},
        ],
        "usage_example": "MRS X0, TTBR0_EL1\nMSR TTBR0_EL1, X0",
    },
    "TTBR1_EL1": {
        "full_name": "Translation Table Base Register 1",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C2_C0_1",
        "width": 64,
        "category": "memory_management",
        "description": "Holds the base address of the translation table for the upper VA range (typically kernel space).",
        "key_fields": [
            {"bits": "[47:1]", "name": "BADDR", "description": "Translation table base address."},
            {"bits": "[63:48]", "name": "ASID", "description": "Address Space Identifier (when TCR_EL1.A1=1)."},
        ],
        "usage_example": "MRS X0, TTBR1_EL1\nMSR TTBR1_EL1, X0",
    },
    "TTBR0_EL2": {
        "full_name": "Translation Table Base Register 0",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C2_C0_0",
        "width": 64,
        "category": "memory_management",
        "description": "Holds the base address of the translation table for EL2.",
        "key_fields": [
            {"bits": "[47:1]", "name": "BADDR", "description": "Translation table base address."},
        ],
        "usage_example": "MRS X0, TTBR0_EL2\nMSR TTBR0_EL2, X0",
    },
    "MAIR_EL1": {
        "full_name": "Memory Attribute Indirection Register",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C10_C2_0",
        "width": 64,
        "category": "memory_management",
        "description": "Defines memory attribute encodings (Device, Normal, Cacheable) referenced by page table AttrIndx field.",
        "key_fields": [
            {"bits": "[7:0]", "name": "Attr0", "description": "Memory attributes for AttrIndx=0."},
            {"bits": "[15:8]", "name": "Attr1", "description": "Memory attributes for AttrIndx=1."},
        ],
        "usage_example": "MRS X0, MAIR_EL1\nMSR MAIR_EL1, X0",
    },
    "MAIR_EL2": {
        "full_name": "Memory Attribute Indirection Register",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C10_C2_0",
        "width": 64,
        "category": "memory_management",
        "description": "Defines memory attribute encodings for EL2 stage 1 translations.",
        "key_fields": [
            {"bits": "[7:0]", "name": "Attr0", "description": "Memory attributes for AttrIndx=0."},
        ],
        "usage_example": "MRS X0, MAIR_EL2\nMSR MAIR_EL2, X0",
    },
    "ESR_EL1": {
        "full_name": "Exception Syndrome Register",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C5_C2_0",
        "width": 64,
        "category": "memory_management",
        "description": "Holds syndrome information for an exception taken to EL1, including the exception class and instruction-specific syndrome.",
        "key_fields": [
            {"bits": "[31:26]", "name": "EC", "description": "Exception Class. Indicates reason for exception (e.g. 0x15=SVC, 0x20=Inst Abort lower, 0x24=Data Abort lower)."},
            {"bits": "[25]", "name": "IL", "description": "Instruction Length. 0=16-bit, 1=32-bit."},
            {"bits": "[24:0]", "name": "ISS", "description": "Instruction Specific Syndrome. Encoding depends on EC value."},
            {"bits": "[24]", "name": "ISV", "description": "Instruction Syndrome Valid (for data aborts). When 1, ISS[23:14] hold access details."},
            {"bits": "[5:0]", "name": "DFSC/IFSC", "description": "Data/Instruction Fault Status Code (within ISS). Indicates fault type (translation, access flag, permission)."},
        ],
        "usage_example": "MRS X0, ESR_EL1    // Read exception syndrome after taking exception",
    },
    "ESR_EL2": {
        "full_name": "Exception Syndrome Register",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C5_C2_0",
        "width": 64,
        "category": "memory_management",
        "description": "Holds syndrome information for an exception taken to EL2.",
        "key_fields": [
            {"bits": "[31:26]", "name": "EC", "description": "Exception Class."},
            {"bits": "[24:0]", "name": "ISS", "description": "Instruction Specific Syndrome."},
        ],
        "usage_example": "MRS X0, ESR_EL2",
    },
    "ESR_EL3": {
        "full_name": "Exception Syndrome Register",
        "el": "EL3",
        "access": "RW",
        "encoding": "S3_6_C5_C2_0",
        "width": 64,
        "category": "memory_management",
        "description": "Holds syndrome information for an exception taken to EL3.",
        "key_fields": [
            {"bits": "[31:26]", "name": "EC", "description": "Exception Class."},
            {"bits": "[24:0]", "name": "ISS", "description": "Instruction Specific Syndrome."},
        ],
        "usage_example": "MRS X0, ESR_EL3",
    },
    "FAR_EL1": {
        "full_name": "Fault Address Register",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C6_C0_0",
        "width": 64,
        "category": "memory_management",
        "description": "Holds the faulting virtual address for synchronous instruction/data aborts taken to EL1.",
        "key_fields": [
            {"bits": "[63:0]", "name": "VA", "description": "Faulting virtual address."},
        ],
        "usage_example": "MRS X0, FAR_EL1    // Read faulting address after data/instruction abort",
    },
    "FAR_EL2": {
        "full_name": "Fault Address Register",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C6_C0_0",
        "width": 64,
        "category": "memory_management",
        "description": "Holds the faulting virtual address for synchronous aborts taken to EL2.",
        "key_fields": [
            {"bits": "[63:0]", "name": "VA", "description": "Faulting virtual address."},
        ],
        "usage_example": "MRS X0, FAR_EL2",
    },
    # ── Exception Handling ─────────────────────────────────────────────────
    "VBAR_EL1": {
        "full_name": "Vector Base Address Register",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C12_C0_0",
        "width": 64,
        "category": "exception",
        "description": "Holds the exception vector base address for EL1. Vectors are at offsets 0x000-0x780 from this address.",
        "key_fields": [
            {"bits": "[63:11]", "name": "VBA", "description": "Vector base address. Bits [10:0] are RES0 (2KB aligned)."},
        ],
        "usage_example": "LDR X0, =vectors\nMSR VBAR_EL1, X0    // Set exception vector table",
    },
    "VBAR_EL2": {
        "full_name": "Vector Base Address Register",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C12_C0_0",
        "width": 64,
        "category": "exception",
        "description": "Holds the exception vector base address for EL2.",
        "key_fields": [
            {"bits": "[63:11]", "name": "VBA", "description": "Vector base address (2KB aligned)."},
        ],
        "usage_example": "MSR VBAR_EL2, X0",
    },
    "VBAR_EL3": {
        "full_name": "Vector Base Address Register",
        "el": "EL3",
        "access": "RW",
        "encoding": "S3_6_C12_C0_0",
        "width": 64,
        "category": "exception",
        "description": "Holds the exception vector base address for EL3.",
        "key_fields": [
            {"bits": "[63:11]", "name": "VBA", "description": "Vector base address (2KB aligned)."},
        ],
        "usage_example": "MSR VBAR_EL3, X0",
    },
    "ELR_EL1": {
        "full_name": "Exception Link Register",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C4_C0_1",
        "width": 64,
        "category": "exception",
        "description": "Holds the return address for an exception taken to EL1. ERET uses this as the target PC.",
        "key_fields": [
            {"bits": "[63:0]", "name": "ReturnAddress", "description": "Exception return address."},
        ],
        "usage_example": "MRS X0, ELR_EL1\nMSR ELR_EL1, X0",
    },
    "ELR_EL2": {
        "full_name": "Exception Link Register",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C4_C0_1",
        "width": 64,
        "category": "exception",
        "description": "Holds the return address for an exception taken to EL2.",
        "key_fields": [
            {"bits": "[63:0]", "name": "ReturnAddress", "description": "Exception return address."},
        ],
        "usage_example": "MRS X0, ELR_EL2\nMSR ELR_EL2, X0",
    },
    "ELR_EL3": {
        "full_name": "Exception Link Register",
        "el": "EL3",
        "access": "RW",
        "encoding": "S3_6_C4_C0_1",
        "width": 64,
        "category": "exception",
        "description": "Holds the return address for an exception taken to EL3.",
        "key_fields": [
            {"bits": "[63:0]", "name": "ReturnAddress", "description": "Exception return address."},
        ],
        "usage_example": "MRS X0, ELR_EL3\nMSR ELR_EL3, X0",
    },
    "SPSR_EL1": {
        "full_name": "Saved Program Status Register",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C4_C0_0",
        "width": 64,
        "category": "exception",
        "description": "Holds the saved PSTATE when an exception is taken to EL1. Restored on ERET.",
        "key_fields": [
            {"bits": "[3:0]", "name": "M", "description": "Mode field. AArch64 EL and SP selection."},
            {"bits": "[9]", "name": "D", "description": "Debug exception mask."},
            {"bits": "[8]", "name": "A", "description": "SError interrupt mask."},
            {"bits": "[7]", "name": "I", "description": "IRQ mask."},
            {"bits": "[6]", "name": "F", "description": "FIQ mask."},
        ],
        "usage_example": "MRS X0, SPSR_EL1\nMSR SPSR_EL1, X0",
    },
    "SPSR_EL2": {
        "full_name": "Saved Program Status Register",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C4_C0_0",
        "width": 64,
        "category": "exception",
        "description": "Holds the saved PSTATE when an exception is taken to EL2.",
        "key_fields": [
            {"bits": "[3:0]", "name": "M", "description": "Mode field."},
            {"bits": "[7:6]", "name": "I,F", "description": "IRQ and FIQ masks."},
        ],
        "usage_example": "MRS X0, SPSR_EL2\nMSR SPSR_EL2, X0",
    },
    "SPSR_EL3": {
        "full_name": "Saved Program Status Register",
        "el": "EL3",
        "access": "RW",
        "encoding": "S3_6_C4_C0_0",
        "width": 64,
        "category": "exception",
        "description": "Holds the saved PSTATE when an exception is taken to EL3.",
        "key_fields": [
            {"bits": "[3:0]", "name": "M", "description": "Mode field."},
            {"bits": "[7:6]", "name": "I,F", "description": "IRQ and FIQ masks."},
        ],
        "usage_example": "MRS X0, SPSR_EL3\nMSR SPSR_EL3, X0",
    },
    "SP_EL0": {
        "full_name": "Stack Pointer (EL0)",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_0_C4_C1_0",
        "width": 64,
        "category": "exception",
        "description": "The stack pointer used when SPSel selects SP_EL0 (shared user-space stack pointer).",
        "key_fields": [
            {"bits": "[63:0]", "name": "SP", "description": "Stack pointer value."},
        ],
        "usage_example": "MRS X0, SP_EL0\nMSR SP_EL0, X0",
    },
    "SP_EL1": {
        "full_name": "Stack Pointer (EL1)",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_4_C4_C1_0",
        "width": 64,
        "category": "exception",
        "description": "Dedicated stack pointer for EL1.",
        "key_fields": [
            {"bits": "[63:0]", "name": "SP", "description": "Stack pointer value."},
        ],
        "usage_example": "MRS X0, SP_EL1\nMSR SP_EL1, X0",
    },
    "SP_EL2": {
        "full_name": "Stack Pointer (EL2)",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_6_C4_C1_0",
        "width": 64,
        "category": "exception",
        "description": "Dedicated stack pointer for EL2.",
        "key_fields": [
            {"bits": "[63:0]", "name": "SP", "description": "Stack pointer value."},
        ],
        "usage_example": "MRS X0, SP_EL2\nMSR SP_EL2, X0",
    },
    # ── ID Registers (read-only) ──────────────────────────────────────────
    "ID_AA64PFR0_EL1": {
        "full_name": "AArch64 Processor Feature Register 0",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C4_0",
        "width": 64,
        "category": "id",
        "description": "Provides information about implemented PE features including exception level support, floating-point, and advanced SIMD.",
        "key_fields": [
            {"bits": "[3:0]", "name": "EL0", "description": "EL0 handling. 0x1=AArch64 only, 0x2=AArch64+AArch32."},
            {"bits": "[7:4]", "name": "EL1", "description": "EL1 handling."},
            {"bits": "[11:8]", "name": "EL2", "description": "EL2 handling."},
            {"bits": "[15:12]", "name": "EL3", "description": "EL3 handling."},
            {"bits": "[19:16]", "name": "FP", "description": "Floating-point support. 0x0=implemented, 0xF=not implemented."},
        ],
        "usage_example": "MRS X0, ID_AA64PFR0_EL1    // Read-only ID register",
    },
    "ID_AA64PFR1_EL1": {
        "full_name": "AArch64 Processor Feature Register 1",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C4_1",
        "width": 64,
        "category": "id",
        "description": "Provides additional PE feature information including Branch Target Identification (BTI), Speculative Store Bypass Safe (SSBS), and MTE.",
        "key_fields": [
            {"bits": "[3:0]", "name": "BT", "description": "Branch Target Identification. 0x0=not impl, 0x1=impl."},
            {"bits": "[7:4]", "name": "SSBS", "description": "Speculative Store Bypass Safe."},
            {"bits": "[11:8]", "name": "MTE", "description": "Memory Tagging Extension support."},
        ],
        "usage_example": "MRS X0, ID_AA64PFR1_EL1",
    },
    "ID_AA64ISAR0_EL1": {
        "full_name": "AArch64 Instruction Set Attribute Register 0",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C6_0",
        "width": 64,
        "category": "id",
        "description": "Indicates support for cryptographic, CRC32, atomic, and other instruction set features.",
        "key_fields": [
            {"bits": "[7:4]", "name": "AES", "description": "AES instruction support."},
            {"bits": "[15:12]", "name": "SHA1", "description": "SHA1 instruction support."},
            {"bits": "[23:20]", "name": "Atomic", "description": "Atomic instruction support (LSE). 0x2=implemented."},
        ],
        "usage_example": "MRS X0, ID_AA64ISAR0_EL1",
    },
    "ID_AA64ISAR1_EL1": {
        "full_name": "AArch64 Instruction Set Attribute Register 1",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C6_1",
        "width": 64,
        "category": "id",
        "description": "Indicates support for additional instruction set features including DPB, JSCVT, FCMA, LRCPC, and PAC.",
        "key_fields": [
            {"bits": "[3:0]", "name": "DPB", "description": "Data Persistence (DC CVAP) support."},
            {"bits": "[11:8]", "name": "LRCPC", "description": "Load-Acquire RCpc instructions."},
        ],
        "usage_example": "MRS X0, ID_AA64ISAR1_EL1",
    },
    "ID_AA64ISAR2_EL1": {
        "full_name": "AArch64 Instruction Set Attribute Register 2",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C6_2",
        "width": 64,
        "category": "id",
        "description": "Indicates support for further instruction set extensions introduced in ARMv8.7+.",
        "key_fields": [
            {"bits": "[3:0]", "name": "WFxT", "description": "WFE/WFI with timeout support."},
        ],
        "usage_example": "MRS X0, ID_AA64ISAR2_EL1",
    },
    "ID_AA64MMFR0_EL1": {
        "full_name": "AArch64 Memory Model Feature Register 0",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C7_0",
        "width": 64,
        "category": "id",
        "description": "Provides information about the implemented memory model and memory management features including PA size and granule support.",
        "key_fields": [
            {"bits": "[3:0]", "name": "PARange", "description": "Physical Address range. 0x0=32b, 0x5=48b, 0x6=52b."},
            {"bits": "[31:28]", "name": "TGran4", "description": "4KB granule support. 0x0=supported."},
            {"bits": "[27:24]", "name": "TGran64", "description": "64KB granule support."},
            {"bits": "[23:20]", "name": "TGran16", "description": "16KB granule support."},
        ],
        "usage_example": "MRS X0, ID_AA64MMFR0_EL1",
    },
    "ID_AA64MMFR1_EL1": {
        "full_name": "AArch64 Memory Model Feature Register 1",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C7_1",
        "width": 64,
        "category": "id",
        "description": "Provides additional memory model feature information including HPDS, VH, PAN, and LO support.",
        "key_fields": [
            {"bits": "[3:0]", "name": "HAFDBS", "description": "Hardware updates to Access/Dirty flags."},
            {"bits": "[11:8]", "name": "VH", "description": "Virtualization Host Extensions."},
        ],
        "usage_example": "MRS X0, ID_AA64MMFR1_EL1",
    },
    "ID_AA64MMFR2_EL1": {
        "full_name": "AArch64 Memory Model Feature Register 2",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C7_2",
        "width": 64,
        "category": "id",
        "description": "Provides further memory model feature information including CNP, UAO, IESB, and VARANGE.",
        "key_fields": [
            {"bits": "[3:0]", "name": "CnP", "description": "Common not Private translations."},
            {"bits": "[19:16]", "name": "VARANGE", "description": "VA Range. 0x1=52-bit VA supported."},
        ],
        "usage_example": "MRS X0, ID_AA64MMFR2_EL1",
    },
    "ID_AA64DFR0_EL1": {
        "full_name": "AArch64 Debug Feature Register 0",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C5_0",
        "width": 64,
        "category": "id",
        "description": "Provides top-level information about the debug system including number of breakpoints and watchpoints.",
        "key_fields": [
            {"bits": "[3:0]", "name": "DebugVer", "description": "Debug architecture version."},
            {"bits": "[15:12]", "name": "BRPs", "description": "Number of breakpoints minus 1."},
            {"bits": "[23:20]", "name": "WRPs", "description": "Number of watchpoints minus 1."},
        ],
        "usage_example": "MRS X0, ID_AA64DFR0_EL1",
    },
    "ID_AA64SMFR0_EL1": {
        "full_name": "SME Feature ID Register 0",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C4_5",
        "width": 64,
        "category": "id",
        "description": "Provides information about the implemented Scalable Matrix Extension (SME) features.",
        "key_fields": [
            {"bits": "[63]", "name": "FA64", "description": "Full A64 instruction set in Streaming SVE mode."},
            {"bits": "[55:52]", "name": "I16I64", "description": "16-bit integer outer product to 64-bit."},
        ],
        "usage_example": "MRS X0, ID_AA64SMFR0_EL1",
    },
    "ID_AA64ZFR0_EL1": {
        "full_name": "SVE Feature ID Register 0",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C4_4",
        "width": 64,
        "category": "id",
        "description": "Provides information about the implemented Scalable Vector Extension (SVE) features.",
        "key_fields": [
            {"bits": "[3:0]", "name": "SVEver", "description": "SVE version. 0x0=SVE, 0x1=SVE2."},
            {"bits": "[7:4]", "name": "AES", "description": "SVE AES instructions."},
        ],
        "usage_example": "MRS X0, ID_AA64ZFR0_EL1",
    },
    "MIDR_EL1": {
        "full_name": "Main ID Register",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C0_0",
        "width": 64,
        "category": "id",
        "description": "Identifies the PE implementation: implementer, architecture, variant, part number, and revision.",
        "key_fields": [
            {"bits": "[31:24]", "name": "Implementer", "description": "Implementer code. 0x41=Arm, 0x51=Qualcomm, 0x43=Cavium."},
            {"bits": "[23:20]", "name": "Variant", "description": "Major revision (variant) number."},
            {"bits": "[19:16]", "name": "Architecture", "description": "Architecture code. 0xF=defined by ID registers."},
            {"bits": "[15:4]", "name": "PartNum", "description": "Primary part number. Identifies core type."},
            {"bits": "[3:0]", "name": "Revision", "description": "Minor revision (patch) number."},
        ],
        "usage_example": "MRS X0, MIDR_EL1    // Identify the processor",
    },
    "MPIDR_EL1": {
        "full_name": "Multiprocessor Affinity Register",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C0_5",
        "width": 64,
        "category": "id",
        "description": "Provides an additional PE identification mechanism for scheduling purposes in multiprocessor systems.",
        "key_fields": [
            {"bits": "[7:0]", "name": "Aff0", "description": "Affinity level 0 (typically core ID)."},
            {"bits": "[15:8]", "name": "Aff1", "description": "Affinity level 1 (typically cluster ID)."},
            {"bits": "[23:16]", "name": "Aff2", "description": "Affinity level 2."},
            {"bits": "[39:32]", "name": "Aff3", "description": "Affinity level 3."},
            {"bits": "[30]", "name": "U", "description": "Uniprocessor. 0=part of multiprocessor, 1=uniprocessor."},
        ],
        "usage_example": "MRS X0, MPIDR_EL1    // Read CPU affinity",
    },
    "REVIDR_EL1": {
        "full_name": "Revision ID Register",
        "el": "EL1",
        "access": "RO",
        "encoding": "S3_0_C0_C0_6",
        "width": 64,
        "category": "id",
        "description": "Provides implementation-specific revision information that supplements MIDR_EL1.",
        "key_fields": [
            {"bits": "[63:0]", "name": "IMPLEMENTATION DEFINED", "description": "Implementation-specific revision information."},
        ],
        "usage_example": "MRS X0, REVIDR_EL1",
    },
    # ── Performance / Debug ───────────────────────────────────────────────
    "PMCR_EL0": {
        "full_name": "Performance Monitors Control Register",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C9_C12_0",
        "width": 64,
        "category": "perf_debug",
        "description": "Controls the performance monitor counters, including enable, reset, clock divider, and export.",
        "key_fields": [
            {"bits": "[0]", "name": "E", "description": "Enable. All counters are enabled when set."},
            {"bits": "[1]", "name": "P", "description": "Event counter reset. Resets all event counters to zero."},
            {"bits": "[2]", "name": "C", "description": "Cycle counter reset. Resets PMCCNTR_EL0."},
            {"bits": "[3]", "name": "D", "description": "Clock divider. 0=every cycle, 1=every 64th cycle."},
            {"bits": "[15:11]", "name": "N", "description": "Number of event counters implemented (read-only)."},
        ],
        "usage_example": "MRS X0, PMCR_EL0\nORR X0, X0, #1      // Enable\nMSR PMCR_EL0, X0",
    },
    "PMCNTENSET_EL0": {
        "full_name": "Performance Monitors Count Enable Set Register",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C9_C12_1",
        "width": 64,
        "category": "perf_debug",
        "description": "Enables individual event counters and the cycle counter. Write 1 to enable; reads return current enable state.",
        "key_fields": [
            {"bits": "[31]", "name": "C", "description": "Cycle counter enable."},
            {"bits": "[30:0]", "name": "P", "description": "Event counter enable bits (one per counter)."},
        ],
        "usage_example": "MOV X0, #(1 << 31)   // Enable cycle counter\nMSR PMCNTENSET_EL0, X0",
    },
    "PMCCNTR_EL0": {
        "full_name": "Performance Monitors Cycle Count Register",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C9_C13_0",
        "width": 64,
        "category": "perf_debug",
        "description": "Counts processor cycles. Controlled by PMCR_EL0.E and PMCNTENSET_EL0.C.",
        "key_fields": [
            {"bits": "[63:0]", "name": "CCNT", "description": "Cycle count value."},
        ],
        "usage_example": "MRS X0, PMCCNTR_EL0    // Read cycle counter",
    },
    "PMEVCNTR0_EL0": {
        "full_name": "Performance Monitors Event Count Register 0",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C14_C8_0",
        "width": 64,
        "category": "perf_debug",
        "description": "Event counter 0. Counts events selected by PMEVTYPER0_EL0.",
        "key_fields": [
            {"bits": "[63:0]", "name": "EVCNT", "description": "Event count value."},
        ],
        "usage_example": "MRS X0, PMEVCNTR0_EL0",
    },
    "PMEVCNTR1_EL0": {
        "full_name": "Performance Monitors Event Count Register 1",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C14_C8_1",
        "width": 64,
        "category": "perf_debug",
        "description": "Event counter 1.",
        "key_fields": [
            {"bits": "[63:0]", "name": "EVCNT", "description": "Event count value."},
        ],
        "usage_example": "MRS X0, PMEVCNTR1_EL0",
    },
    "PMEVCNTR2_EL0": {
        "full_name": "Performance Monitors Event Count Register 2",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C14_C8_2",
        "width": 64,
        "category": "perf_debug",
        "description": "Event counter 2.",
        "key_fields": [
            {"bits": "[63:0]", "name": "EVCNT", "description": "Event count value."},
        ],
        "usage_example": "MRS X0, PMEVCNTR2_EL0",
    },
    "PMEVCNTR3_EL0": {
        "full_name": "Performance Monitors Event Count Register 3",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C14_C8_3",
        "width": 64,
        "category": "perf_debug",
        "description": "Event counter 3.",
        "key_fields": [
            {"bits": "[63:0]", "name": "EVCNT", "description": "Event count value."},
        ],
        "usage_example": "MRS X0, PMEVCNTR3_EL0",
    },
    "PMEVCNTR4_EL0": {
        "full_name": "Performance Monitors Event Count Register 4",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C14_C8_4",
        "width": 64,
        "category": "perf_debug",
        "description": "Event counter 4.",
        "key_fields": [
            {"bits": "[63:0]", "name": "EVCNT", "description": "Event count value."},
        ],
        "usage_example": "MRS X0, PMEVCNTR4_EL0",
    },
    "PMEVCNTR5_EL0": {
        "full_name": "Performance Monitors Event Count Register 5",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C14_C8_5",
        "width": 64,
        "category": "perf_debug",
        "description": "Event counter 5.",
        "key_fields": [
            {"bits": "[63:0]", "name": "EVCNT", "description": "Event count value."},
        ],
        "usage_example": "MRS X0, PMEVCNTR5_EL0",
    },
    "PMSELR_EL0": {
        "full_name": "Performance Monitors Event Counter Selection Register",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C9_C12_5",
        "width": 64,
        "category": "perf_debug",
        "description": "Selects the current event counter for access via PMXEVCNTR_EL0 and PMXEVTYPER_EL0.",
        "key_fields": [
            {"bits": "[4:0]", "name": "SEL", "description": "Counter select. Selects which event counter is accessed."},
        ],
        "usage_example": "MOV X0, #0\nMSR PMSELR_EL0, X0    // Select counter 0",
    },
    "PMEVTYPER0_EL0": {
        "full_name": "Performance Monitors Event Type Selection Register 0",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C14_C12_0",
        "width": 64,
        "category": "perf_debug",
        "description": "Selects which event is counted by PMEVCNTR0_EL0 and filters by EL.",
        "key_fields": [
            {"bits": "[15:0]", "name": "evtCount", "description": "Event number to count (architecture-defined or IMPLEMENTATION DEFINED)."},
            {"bits": "[26]", "name": "U", "description": "EL0 filtering. 1=don't count at EL0."},
            {"bits": "[27]", "name": "NSK", "description": "Non-secure EL1 filtering."},
        ],
        "usage_example": "MOV X0, #0x11    // CPU_CYCLES event\nMSR PMEVTYPER0_EL0, X0",
    },
    "MDSCR_EL1": {
        "full_name": "Monitor Debug System Control Register",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C0_C2_2",
        "width": 64,
        "category": "perf_debug",
        "description": "Main debug control register. Enables software debug events, single-step, and breakpoint/watchpoint control.",
        "key_fields": [
            {"bits": "[0]", "name": "SS", "description": "Software step. When 1, enables single-step exceptions."},
            {"bits": "[13]", "name": "MDE", "description": "Monitor debug events enable."},
            {"bits": "[15]", "name": "KDE", "description": "Kernel debug events enable (EL1)."},
        ],
        "usage_example": "MRS X0, MDSCR_EL1\nORR X0, X0, #(1 << 13)   // Enable debug events\nMSR MDSCR_EL1, X0",
    },
    "DBGBVR0_EL1": {
        "full_name": "Debug Breakpoint Value Register 0",
        "el": "EL1",
        "access": "RW",
        "encoding": "S2_0_C0_C0_4",
        "width": 64,
        "category": "perf_debug",
        "description": "Holds the address or address range for hardware breakpoint 0 (first of up to 16).",
        "key_fields": [
            {"bits": "[63:2]", "name": "VA", "description": "Virtual address for breakpoint match."},
        ],
        "usage_example": "MSR DBGBVR0_EL1, X0    // Set breakpoint 0 address",
    },
    "DBGWVR0_EL1": {
        "full_name": "Debug Watchpoint Value Register 0",
        "el": "EL1",
        "access": "RW",
        "encoding": "S2_0_C0_C0_6",
        "width": 64,
        "category": "perf_debug",
        "description": "Holds the address or address range for hardware watchpoint 0.",
        "key_fields": [
            {"bits": "[63:2]", "name": "VA", "description": "Virtual address for watchpoint match."},
        ],
        "usage_example": "MSR DBGWVR0_EL1, X0    // Set watchpoint 0 address",
    },
    # ── Timer ─────────────────────────────────────────────────────────────
    "CNTPCT_EL0": {
        "full_name": "Counter-timer Physical Count",
        "el": "EL0",
        "access": "RO",
        "encoding": "S3_3_C14_C0_1",
        "width": 64,
        "category": "timer",
        "description": "Holds the 64-bit physical count value from the system counter (no virtual offset applied).",
        "key_fields": [
            {"bits": "[63:0]", "name": "PhysicalCount", "description": "Current physical count value."},
        ],
        "usage_example": "MRS X0, CNTPCT_EL0    // Read physical timer count",
    },
    "CNTVCT_EL0": {
        "full_name": "Counter-timer Virtual Count",
        "el": "EL0",
        "access": "RO",
        "encoding": "S3_3_C14_C0_2",
        "width": 64,
        "category": "timer",
        "description": "Holds the 64-bit virtual count value (physical count minus CNTVOFF_EL2 offset).",
        "key_fields": [
            {"bits": "[63:0]", "name": "VirtualCount", "description": "Current virtual count value."},
        ],
        "usage_example": "MRS X0, CNTVCT_EL0    // Read virtual timer count",
    },
    "CNTP_TVAL_EL0": {
        "full_name": "Counter-timer Physical Timer TimerValue Register",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C14_C2_0",
        "width": 64,
        "category": "timer",
        "description": "Holds the timer value for the EL1 physical timer. Counts down from the written value.",
        "key_fields": [
            {"bits": "[31:0]", "name": "TimerValue", "description": "Timer value (signed 32-bit). Timer fires when reaches 0 or negative."},
        ],
        "usage_example": "MOV X0, #1000000\nMSR CNTP_TVAL_EL0, X0    // Set timer for 1M ticks",
    },
    "CNTP_CTL_EL0": {
        "full_name": "Counter-timer Physical Timer Control Register",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C14_C2_1",
        "width": 64,
        "category": "timer",
        "description": "Controls the EL1 physical timer: enable, output mask, and status.",
        "key_fields": [
            {"bits": "[0]", "name": "ENABLE", "description": "Timer enable. 1=enabled."},
            {"bits": "[1]", "name": "IMASK", "description": "Interrupt mask. 0=not masked, 1=masked."},
            {"bits": "[2]", "name": "ISTATUS", "description": "Timer condition met (read-only)."},
        ],
        "usage_example": "MOV X0, #1\nMSR CNTP_CTL_EL0, X0    // Enable physical timer",
    },
    "CNTFRQ_EL0": {
        "full_name": "Counter-timer Frequency Register",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C14_C0_0",
        "width": 64,
        "category": "timer",
        "description": "Holds the clock frequency of the system counter in Hz. Typically written by firmware at boot.",
        "key_fields": [
            {"bits": "[31:0]", "name": "Freq", "description": "Clock frequency in Hz (commonly 100MHz or 19.2MHz)."},
        ],
        "usage_example": "MRS X0, CNTFRQ_EL0    // Read timer frequency",
    },
    # ── Security / Virtualization ─────────────────────────────────────────
    "SCR_EL3": {
        "full_name": "Secure Configuration Register",
        "el": "EL3",
        "access": "RW",
        "encoding": "S3_6_C1_C1_0",
        "width": 64,
        "category": "security",
        "description": (
            "Defines the security state configuration. Controls Non-secure bit (NS), "
            "Hypervisor Call enable (HCE), Secure Monitor Call disable (SMD), and routing "
            "of exceptions to EL3."
        ),
        "key_fields": [
            {"bits": "[0]", "name": "NS", "description": "Non-secure bit. 0=Secure state, 1=Non-secure state."},
            {"bits": "[8]", "name": "HCE", "description": "Hypervisor Call enable. 1=HVC instruction is enabled."},
            {"bits": "[7]", "name": "SMD", "description": "Secure Monitor Call disable. 1=SMC is UNDEFINED at EL1 and above."},
            {"bits": "[3]", "name": "EA", "description": "External Abort and SError routing to EL3."},
            {"bits": "[2]", "name": "FIQ", "description": "Physical FIQ routing to EL3."},
            {"bits": "[1]", "name": "IRQ", "description": "Physical IRQ routing to EL3."},
        ],
        "usage_example": "MRS X0, SCR_EL3\nORR X0, X0, #1       // Set NS bit (Non-secure)\nMSR SCR_EL3, X0",
    },
    "HCR_EL2": {
        "full_name": "Hypervisor Configuration Register",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C1_C1_0",
        "width": 64,
        "category": "security",
        "description": (
            "Provides configuration controls for virtualization including second-stage "
            "translation, trapping of various operations to EL2, and virtual exception injection."
        ),
        "key_fields": [
            {"bits": "[0]", "name": "VM", "description": "Virtualization enable. Enables stage 2 translation."},
            {"bits": "[3]", "name": "FMO", "description": "Physical FIQ routing. 1=route to EL2 (virtual FIQ for guest)."},
            {"bits": "[4]", "name": "IMO", "description": "Physical IRQ routing. 1=route to EL2 (virtual IRQ for guest)."},
            {"bits": "[5]", "name": "AMO", "description": "Physical SError routing. 1=route to EL2."},
            {"bits": "[31]", "name": "RW", "description": "Execution state for EL1. 0=AArch32, 1=AArch64."},
            {"bits": "[34]", "name": "E2H", "description": "EL2 Host enable. VHE (Virtualization Host Extensions)."},
        ],
        "usage_example": "MRS X0, HCR_EL2\nORR X0, X0, #1       // Enable stage 2 translation\nMSR HCR_EL2, X0",
    },
    "CPTR_EL2": {
        "full_name": "Architectural Feature Trap Register (EL2)",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C1_C1_2",
        "width": 64,
        "category": "security",
        "description": "Controls trapping to EL2 of accesses to CPACR, trace, floating-point, and Advanced SIMD functionality.",
        "key_fields": [
            {"bits": "[10]", "name": "TFP", "description": "Trap FP/SIMD. 1=accesses from lower ELs trap to EL2."},
            {"bits": "[20]", "name": "TTA", "description": "Trap Trace Access."},
        ],
        "usage_example": "MRS X0, CPTR_EL2\nMSR CPTR_EL2, X0",
    },
    "CPTR_EL3": {
        "full_name": "Architectural Feature Trap Register (EL3)",
        "el": "EL3",
        "access": "RW",
        "encoding": "S3_6_C1_C1_2",
        "width": 64,
        "category": "security",
        "description": "Controls trapping to EL3 of accesses to CPACR, trace, floating-point, Advanced SIMD, and SVE functionality.",
        "key_fields": [
            {"bits": "[10]", "name": "TFP", "description": "Trap FP/SIMD. 1=accesses trap to EL3."},
            {"bits": "[20]", "name": "TTA", "description": "Trap Trace Access."},
        ],
        "usage_example": "MRS X0, CPTR_EL3\nMSR CPTR_EL3, X0",
    },
    "VTTBR_EL2": {
        "full_name": "Virtualization Translation Table Base Register",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C2_C1_0",
        "width": 64,
        "category": "security",
        "description": "Holds the base address of the stage 2 translation table for VMs, along with the VMID.",
        "key_fields": [
            {"bits": "[47:1]", "name": "BADDR", "description": "Stage 2 translation table base address."},
            {"bits": "[63:48]", "name": "VMID", "description": "Virtual Machine Identifier."},
        ],
        "usage_example": "MSR VTTBR_EL2, X0    // Set VM page table base",
    },
    "VMPIDR_EL2": {
        "full_name": "Virtualization Multiprocessor ID Register",
        "el": "EL2",
        "access": "RW",
        "encoding": "S3_4_C0_C0_5",
        "width": 64,
        "category": "security",
        "description": "Holds the value of the Virtualization Multiprocessor ID, visible to EL1 reads of MPIDR_EL1 when running in a VM.",
        "key_fields": [
            {"bits": "[7:0]", "name": "Aff0", "description": "Virtual affinity level 0."},
            {"bits": "[15:8]", "name": "Aff1", "description": "Virtual affinity level 1."},
        ],
        "usage_example": "MSR VMPIDR_EL2, X0",
    },
    "CONTEXTIDR_EL1": {
        "full_name": "Context ID Register",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C13_C0_1",
        "width": 64,
        "category": "security",
        "description": "Holds a process ID value for the OS. Used by trace and debug to identify the current process.",
        "key_fields": [
            {"bits": "[31:0]", "name": "PROCID", "description": "Process ID (OS sets this, typically to PID or TID)."},
        ],
        "usage_example": "MRS X0, CONTEXTIDR_EL1\nMSR CONTEXTIDR_EL1, X0",
    },
    "TPIDR_EL0": {
        "full_name": "EL0 Read/Write Software Thread ID Register",
        "el": "EL0",
        "access": "RW",
        "encoding": "S3_3_C13_C0_2",
        "width": 64,
        "category": "security",
        "description": "Software thread pointer register accessible from EL0. Used by libc/runtime for TLS (Thread-Local Storage) base pointer.",
        "key_fields": [
            {"bits": "[63:0]", "name": "ThreadID", "description": "Software thread identifier / TLS pointer."},
        ],
        "usage_example": "MRS X0, TPIDR_EL0    // Read user-space TLS pointer",
    },
    "TPIDR_EL1": {
        "full_name": "EL1 Software Thread ID Register",
        "el": "EL1",
        "access": "RW",
        "encoding": "S3_0_C13_C0_4",
        "width": 64,
        "category": "security",
        "description": "Software thread pointer register accessible only from EL1 and above. Used by the OS kernel for per-CPU or per-thread data.",
        "key_fields": [
            {"bits": "[63:0]", "name": "ThreadID", "description": "Kernel thread/CPU pointer."},
        ],
        "usage_example": "MRS X0, TPIDR_EL1\nMSR TPIDR_EL1, X0",
    },
    "TPIDRRO_EL0": {
        "full_name": "EL0 Read-Only Software Thread ID Register",
        "el": "EL0",
        "access": "RO",
        "encoding": "S3_3_C13_C0_3",
        "width": 64,
        "category": "security",
        "description": "Thread pointer register read-only from EL0 but writable from EL1. Kernel sets it; user space reads it.",
        "key_fields": [
            {"bits": "[63:0]", "name": "ThreadID", "description": "Read-only thread identifier for EL0."},
        ],
        "usage_example": "MRS X0, TPIDRRO_EL0   // Read kernel-set thread pointer from user space",
    },
}

_SYSREG_CATEGORY_LABELS: dict[str, str] = {
    "memory_management": "Memory Management",
    "exception": "Exception Handling",
    "id": "ID Registers (Read-Only)",
    "perf_debug": "Performance / Debug",
    "timer": "Timer",
    "security": "Security / Virtualization",
}


def _format_system_register(name: str, reg: dict) -> str:
    """Format a single system register entry as readable text."""
    access_text = "Read/Write" if reg["access"] == "RW" else "Read-Only"
    lines = [
        f"# System Register: {name}",
        "",
        f"**Full Name:** {reg['full_name']}",
        f"**Exception Level:** {reg['el']}",
        f"**Access:** {access_text}",
        f"**Encoding:** {reg['encoding']}",
        f"**Width:** {reg['width']}-bit",
        f"**Category:** {_SYSREG_CATEGORY_LABELS.get(reg['category'], reg['category'])}",
        "",
        "## Description",
        reg["description"],
    ]

    if reg.get("key_fields"):
        lines.append("")
        lines.append("## Key Fields")
        lines.append("| Bits | Name | Description |")
        lines.append("|------|------|-------------|")
        for f in reg["key_fields"]:
            lines.append(f"| {f['bits']} | {f['name']} | {f['description']} |")

    if reg.get("usage_example"):
        lines.append("")
        lines.append("## Access Example")
        lines.append("```asm")
        lines.append(reg["usage_example"])
        lines.append("```")

    return "\n".join(lines)


def _list_system_registers_by_el(el: str) -> str:
    """List all system registers for a given exception level."""
    el_upper = el.upper()
    matches = {
        name: reg for name, reg in _SYSTEM_REGISTERS.items() if reg["el"] == el_upper
    }
    if not matches:
        return f"Error: No system registers found for exception level '{el}'."

    lines = [f"# System Registers at {el_upper}", ""]
    # Group by category
    by_cat: dict[str, list[str]] = {}
    for name, reg in matches.items():
        cat = reg["category"]
        by_cat.setdefault(cat, []).append(name)

    for cat, label in _SYSREG_CATEGORY_LABELS.items():
        if cat in by_cat:
            lines.append(f"## {label}")
            for name in sorted(by_cat[cat]):
                reg = _SYSTEM_REGISTERS[name]
                acc = "RO" if reg["access"] == "RO" else "RW"
                lines.append(f"  {name:<25} {acc}  {reg['full_name']}")
            lines.append("")

    lines.append(f"Total: {len(matches)} registers at {el_upper}")
    return "\n".join(lines)


def _list_all_system_registers() -> str:
    """List all system registers grouped by category."""
    lines = ["# All AArch64 System Registers", ""]
    for cat, label in _SYSREG_CATEGORY_LABELS.items():
        regs_in_cat = {
            name: reg
            for name, reg in _SYSTEM_REGISTERS.items()
            if reg["category"] == cat
        }
        if regs_in_cat:
            lines.append(f"## {label}")
            for name in sorted(regs_in_cat):
                reg = regs_in_cat[name]
                acc = "RO" if reg["access"] == "RO" else "RW"
                lines.append(
                    f"  {name:<25} {reg['el']:<5} {acc}  {reg['full_name']}"
                )
            lines.append("")

    lines.append(f"Total: {len(_SYSTEM_REGISTERS)} system registers")
    return "\n".join(lines)


def _list_system_registers_by_category(category: str) -> str:
    """List system registers filtered by category key."""
    matches = {
        name: reg
        for name, reg in _SYSTEM_REGISTERS.items()
        if reg["category"] == category
    }
    if not matches:
        return f"Error: No system registers found for category '{category}'."

    label = _SYSREG_CATEGORY_LABELS.get(category, category)
    lines = [f"# System Registers — {label}", ""]
    for name in sorted(matches):
        reg = matches[name]
        acc = "RO" if reg["access"] == "RO" else "RW"
        lines.append(f"  {name:<25} {reg['el']:<5} {acc}  {reg['full_name']}")
    lines.append("")
    lines.append(f"Total: {len(matches)} registers")
    return "\n".join(lines)


@mcp.tool()
def lookup_system_register(register: str, el: str | None = None) -> str:
    """Look up an AArch64 system register by name.

    Returns detailed information about the register including its encoding,
    bit fields, access type, and usage examples. The system register space
    covers hundreds of MRS/MSR-accessible registers used for memory management,
    exception handling, performance monitoring, debug, timers, and security.

    Args:
        register: System register name (e.g. "SCTLR_EL1", "TCR_EL1", "HCR_EL2").
            Special queries:
            "list" -- List all registers (optionally filtered by el)
            "memory" or "mmu" -- Show all memory management registers
            "timer" -- Show all timer registers
            "id" -- Show all ID registers
            "perf" or "pmu" -- Show all performance/debug registers
        el: Optional exception level filter ("EL0", "EL1", "EL2", "EL3").
            Used with "list" to filter by EL; ignored for direct lookups.
    """
    key = register.strip().upper()

    # Special queries
    if key == "LIST":
        if el is not None:
            return _list_system_registers_by_el(el)
        return _list_all_system_registers()

    if key in ("MEMORY", "MMU"):
        return _list_system_registers_by_category("memory_management")

    if key == "TIMER":
        return _list_system_registers_by_category("timer")

    if key == "ID":
        return _list_system_registers_by_category("id")

    if key in ("PERF", "PMU"):
        return _list_system_registers_by_category("perf_debug")

    # Direct register lookup
    if key in _SYSTEM_REGISTERS:
        return _format_system_register(key, _SYSTEM_REGISTERS[key])

    # Not found
    return (
        f"Error: System register '{register}' not found.\n\n"
        f"Use lookup_system_register('list') to see all available system registers, "
        f"or try one of the special queries: 'memory', 'timer', 'id', 'perf'."
    )

# ---------------------------------------------------------------------------
# Tool 18: explain_performance_counter — ARM PMU event reference
# ---------------------------------------------------------------------------

_PMU_EVENTS: dict[str, dict] = {
    "SW_INCR": {
        "event_name": "SW_INCR",
        "event_number": "0x00",
        "category": "misc",
        "description": (
            "Counts software increments. The counter only increments when "
            "software writes to the PMSWINC_EL0 register. This allows software "
            "to count arbitrary events by manually triggering the increment."
        ),
        "when_to_use": (
            "Use for custom software event counting — e.g., counting function "
            "calls, loop iterations, or specific code-path hits from within "
            "the application or OS. Useful for correlating software events with "
            "hardware PMU data."
        ),
        "formula_tips": [
            "Custom metric = SW_INCR / CPU_CYCLES (rate of software events per cycle)",
        ],
        "linux_perf_name": "sw_incr",
        "ai_workload_tips": (
            "Instrument inference entry/exit points to count per-layer or "
            "per-operator invocations. Combine with cycle counters to get "
            "operator-level throughput."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x00  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L1I_CACHE_REFILL": {
        "event_name": "L1I_CACHE_REFILL",
        "event_number": "0x01",
        "category": "cache",
        "description": (
            "Counts each cache line refill into the L1 instruction cache, "
            "caused by an instruction fetch that misses in the L1I cache. "
            "Each refill represents a full cache line fetch from a higher "
            "level of the memory hierarchy."
        ),
        "when_to_use": (
            "Profile code with large instruction footprints (e.g., deeply "
            "inlined template code, JIT-compiled code). A high L1I refill "
            "rate indicates instruction working set exceeds L1I capacity."
        ),
        "formula_tips": [
            "L1I miss rate = L1I_CACHE_REFILL / L1I_CACHE",
            "MPKI (misses per 1K instructions) = L1I_CACHE_REFILL * 1000 / INST_RETIRED",
        ],
        "linux_perf_name": "l1i_cache_refill",
        "ai_workload_tips": (
            "Large ML frameworks can have huge code footprints. If L1I miss rate "
            "is high, consider profile-guided optimization (PGO) or reducing "
            "template instantiations. JIT compilers (TVM, XLA) should aim for "
            "compact generated code."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x01  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L1I_TLB_REFILL": {
        "event_name": "L1I_TLB_REFILL",
        "event_number": "0x02",
        "category": "tlb",
        "description": (
            "Counts each refill of the L1 instruction TLB caused by an "
            "instruction fetch that misses in the L1 ITLB. This triggers "
            "a page table walk or L2 TLB lookup."
        ),
        "when_to_use": (
            "Profile applications with large code footprints spread across "
            "many pages. High ITLB miss rates indicate the instruction working "
            "set spans too many pages."
        ),
        "formula_tips": [
            "ITLB miss rate = L1I_TLB_REFILL / L1I_TLB",
            "Consider huge pages if ITLB miss rate > 1%",
        ],
        "linux_perf_name": "l1i_tlb_refill",
        "ai_workload_tips": (
            "Large ML frameworks with many shared libraries can suffer ITLB "
            "pressure. Use huge pages for code sections or link statically to "
            "reduce code spread across pages."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x02  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L1D_CACHE_REFILL": {
        "event_name": "L1D_CACHE_REFILL",
        "event_number": "0x03",
        "category": "cache",
        "description": (
            "Counts each cache line refill into the L1 data cache caused by "
            "a data access that misses in the L1D cache. Each refill brings "
            "a full cache line (typically 64 bytes) from a higher level of "
            "the memory hierarchy."
        ),
        "when_to_use": (
            "Profile memory-intensive code. High refill rate indicates poor "
            "spatial locality or cache thrashing. This is one of the most "
            "important events for performance tuning."
        ),
        "formula_tips": [
            "L1D miss rate = L1D_CACHE_REFILL / L1D_CACHE",
            "Combine with MEM_ACCESS to understand memory hierarchy pressure",
            "MPKI = L1D_CACHE_REFILL * 1000 / INST_RETIRED",
        ],
        "linux_perf_name": "l1d_cache_refill",
        "ai_workload_tips": (
            "For ML inference, high L1D misses suggest tensor data layout is "
            "not cache-friendly. Consider tiling or NHWC to NCHW layout changes. "
            "For GEMM, ensure inner-loop data fits in L1D."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x03  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L1D_CACHE": {
        "event_name": "L1D_CACHE",
        "event_number": "0x04",
        "category": "cache",
        "description": (
            "Counts each access that looks up in the L1 data cache. This "
            "includes both hits and misses. The count includes load, store, "
            "and atomic operations."
        ),
        "when_to_use": (
            "Use as a denominator for L1D miss-rate calculations. High access "
            "count is normal for data-intensive code; the miss rate "
            "(L1D_CACHE_REFILL / L1D_CACHE) is what matters."
        ),
        "formula_tips": [
            "L1D miss rate = L1D_CACHE_REFILL / L1D_CACHE",
            "L1D access intensity = L1D_CACHE / INST_RETIRED",
        ],
        "linux_perf_name": "l1d_cache",
        "ai_workload_tips": (
            "High L1D access rate with low miss rate means data is cache-resident. "
            "This is the ideal state for compute-bound ML kernels."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x04  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L1D_TLB_REFILL": {
        "event_name": "L1D_TLB_REFILL",
        "event_number": "0x05",
        "category": "tlb",
        "description": (
            "Counts each refill of the L1 data TLB caused by a data access "
            "that misses in the L1 DTLB. Each miss triggers either a L2 TLB "
            "lookup or a full page table walk."
        ),
        "when_to_use": (
            "Profile applications with large data footprints or random access "
            "patterns across many pages. High DTLB refill rates add latency "
            "to every missing access."
        ),
        "formula_tips": [
            "DTLB miss rate = L1D_TLB_REFILL / L1D_TLB",
            "If DTLB miss rate > 1%, consider huge pages (2MB or 1GB)",
        ],
        "linux_perf_name": "l1d_tlb_refill",
        "ai_workload_tips": (
            "Large tensor allocations with random access patterns (e.g., "
            "embedding table lookups) cause DTLB pressure. Use huge pages "
            "(madvise MADV_HUGEPAGE) for tensor memory."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x05  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "INST_RETIRED": {
        "event_name": "INST_RETIRED",
        "event_number": "0x08",
        "category": "instruction",
        "description": (
            "Counts each instruction that is architecturally executed. "
            "This excludes speculated instructions that are later flushed. "
            "It is the definitive count of useful work done by the processor."
        ),
        "when_to_use": (
            "Fundamental event for IPC (Instructions Per Cycle) calculation. "
            "Use as a baseline denominator for MPKI and other per-instruction "
            "metrics."
        ),
        "formula_tips": [
            "IPC = INST_RETIRED / CPU_CYCLES",
            "MPKI = <miss_event> * 1000 / INST_RETIRED",
            "Speculation overhead = (OP_SPEC - OP_RETIRED) / OP_SPEC",
        ],
        "linux_perf_name": "inst_retired",
        "ai_workload_tips": (
            "IPC is a primary indicator of CPU efficiency. For ML inference, "
            "IPC > 2.0 on modern ARM cores is good. Low IPC suggests memory "
            "stalls or branch mispredictions."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x08  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "EXC_TAKEN": {
        "event_name": "EXC_TAKEN",
        "event_number": "0x09",
        "category": "misc",
        "description": (
            "Counts each exception taken, including IRQs, FIQs, SVC calls, "
            "data aborts, prefetch aborts, and undefined instruction exceptions. "
            "This counts the architectural event of entering an exception handler."
        ),
        "when_to_use": (
            "Monitor interrupt and exception frequency. High exception rates "
            "can indicate excessive system calls, timer interrupts, or error "
            "conditions degrading performance."
        ),
        "formula_tips": [
            "Exception rate = EXC_TAKEN / CPU_CYCLES",
            "Exception overhead = EXC_TAKEN * avg_handler_cycles / CPU_CYCLES",
        ],
        "linux_perf_name": "exc_taken",
        "ai_workload_tips": (
            "Frequent interrupts during ML inference add jitter and reduce "
            "throughput. Consider CPU isolation (isolcpus), interrupt affinity, "
            "and NOHZ_FULL to minimize exceptions on compute cores."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x09  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "EXC_RETURN": {
        "event_name": "EXC_RETURN",
        "event_number": "0x0A",
        "category": "misc",
        "description": (
            "Counts each exception return executed (ERET instruction). "
            "This represents the completion of exception handling and return "
            "to the interrupted context."
        ),
        "when_to_use": (
            "Use with EXC_TAKEN to verify exception handling is balanced. "
            "A difference between EXC_TAKEN and EXC_RETURN may indicate "
            "nested exceptions or tail-chained interrupts."
        ),
        "formula_tips": [
            "Nested exception rate = (EXC_TAKEN - EXC_RETURN) / EXC_TAKEN",
            "Time in exception handlers approx = (EXC_TAKEN - EXC_RETURN) * latency",
        ],
        "linux_perf_name": "exc_return",
        "ai_workload_tips": (
            "Monitor EXC_TAKEN vs EXC_RETURN balance during inference to "
            "detect interrupt storms or excessive kernel re-entry."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x0A  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "BR_MIS_PRED": {
        "event_name": "BR_MIS_PRED",
        "event_number": "0x10",
        "category": "branch",
        "description": (
            "Counts each branch that is mispredicted or not predicted by the "
            "branch prediction unit. This causes a pipeline flush and refetch "
            "from the correct target, wasting cycles proportional to the "
            "pipeline depth."
        ),
        "when_to_use": (
            "Profile code with unpredictable branches (data-dependent control "
            "flow, virtual dispatch, switch statements with many cases). "
            "Each mispredict costs 10-20+ cycles on modern cores."
        ),
        "formula_tips": [
            "Branch mispredict rate = BR_MIS_PRED / BR_PRED",
            "Mispredict penalty cycles (approx) = BR_MIS_PRED * pipeline_depth",
            "MPKI = BR_MIS_PRED * 1000 / INST_RETIRED",
        ],
        "linux_perf_name": "br_mis_pred",
        "ai_workload_tips": (
            "Neural network inference with conditional operators (dynamic shapes, "
            "sparse activations) can cause branch mispredictions. Consider branchless "
            "implementations (CSEL, predication) for hot paths."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x10  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "CPU_CYCLES": {
        "event_name": "CPU_CYCLES",
        "event_number": "0x11",
        "category": "misc",
        "description": (
            "Counts every CPU clock cycle. This is the fundamental time "
            "reference for all cycle-based metrics. It counts actual cycles, "
            "so frequency scaling (DVFS) affects the relationship between "
            "cycles and wall-clock time."
        ),
        "when_to_use": (
            "Essential baseline event. Used as denominator for almost all "
            "PMU metrics (IPC, miss rates, stall ratios). Always include "
            "CPU_CYCLES in any PMU measurement session."
        ),
        "formula_tips": [
            "IPC = INST_RETIRED / CPU_CYCLES",
            "CPI = CPU_CYCLES / INST_RETIRED",
            "Stall ratio = STALL_FRONTEND / CPU_CYCLES + STALL_BACKEND / CPU_CYCLES",
        ],
        "linux_perf_name": "cpu_cycles",
        "ai_workload_tips": (
            "Lock CPU frequency (performance governor) during benchmarking to "
            "ensure cycle counts are comparable across runs. Use cycles rather "
            "than wall time for micro-benchmarks."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x11  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "BR_PRED": {
        "event_name": "BR_PRED",
        "event_number": "0x12",
        "category": "branch",
        "description": (
            "Counts each predictable branch instruction executed, whether "
            "correctly predicted or mispredicted. This provides the total "
            "branch count for computing mispredict rates."
        ),
        "when_to_use": (
            "Use as the denominator for branch mispredict rate calculation. "
            "A high branch density (BR_PRED / INST_RETIRED) may indicate "
            "code that could benefit from if-conversion or vectorization."
        ),
        "formula_tips": [
            "Branch mispredict rate = BR_MIS_PRED / BR_PRED",
            "Branch density = BR_PRED / INST_RETIRED",
        ],
        "linux_perf_name": "br_pred",
        "ai_workload_tips": (
            "High branch density in ML kernels often indicates scalar control "
            "flow that could be vectorized with SVE/NEON predicated operations."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x12  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "MEM_ACCESS": {
        "event_name": "MEM_ACCESS",
        "event_number": "0x13",
        "category": "memory",
        "description": (
            "Counts each memory access (data read or write) issued by the "
            "processor. This includes L1D cache hits, L1D misses, and "
            "uncacheable accesses."
        ),
        "when_to_use": (
            "Profile overall memory access intensity. Combine with cache "
            "miss events to understand the memory hierarchy behavior of "
            "your workload."
        ),
        "formula_tips": [
            "Memory access rate = MEM_ACCESS / INST_RETIRED",
            "L1D hit rate = 1 - (L1D_CACHE_REFILL / MEM_ACCESS)",
            "Bytes accessed per cycle = MEM_ACCESS * cache_line_size / CPU_CYCLES",
        ],
        "linux_perf_name": "mem_access",
        "ai_workload_tips": (
            "For GEMM and convolution kernels, high MEM_ACCESS with low miss "
            "rate indicates good data reuse. For memory-bound ops (element-wise, "
            "batch norm), optimize for memory bandwidth."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x13  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L1I_CACHE": {
        "event_name": "L1I_CACHE",
        "event_number": "0x14",
        "category": "cache",
        "description": (
            "Counts each access that looks up in the L1 instruction cache. "
            "This includes both hits and misses. Provides the total "
            "instruction fetch count for L1I miss rate calculation."
        ),
        "when_to_use": (
            "Use as denominator for L1I miss rate. High L1I access count "
            "with low miss rate means instruction working set fits in cache."
        ),
        "formula_tips": [
            "L1I miss rate = L1I_CACHE_REFILL / L1I_CACHE",
        ],
        "linux_perf_name": "l1i_cache",
        "ai_workload_tips": (
            "ML frameworks with many operator implementations can have large "
            "instruction footprints. Monitor L1I miss rate to detect icache "
            "pressure from bloated binaries."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x14  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L1D_CACHE_WB": {
        "event_name": "L1D_CACHE_WB",
        "event_number": "0x15",
        "category": "cache",
        "description": (
            "Counts each write-back of a dirty cache line from the L1 data "
            "cache to the next level of the memory hierarchy. Write-backs "
            "occur when a dirty line is evicted from L1D."
        ),
        "when_to_use": (
            "Profile write-intensive code. High write-back rate combined with "
            "high L1D_CACHE_REFILL indicates cache thrashing with dirty data. "
            "This puts pressure on memory bandwidth."
        ),
        "formula_tips": [
            "Write-back ratio = L1D_CACHE_WB / L1D_CACHE",
            "Dirty eviction rate = L1D_CACHE_WB / L1D_CACHE_REFILL",
        ],
        "linux_perf_name": "l1d_cache_wb",
        "ai_workload_tips": (
            "Training workloads with large gradient updates generate many "
            "write-backs. Ensure write buffers are not saturated. Consider "
            "non-temporal stores for write-only output tensors."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x15  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L2D_CACHE": {
        "event_name": "L2D_CACHE",
        "event_number": "0x16",
        "category": "cache",
        "description": (
            "Counts each access that looks up in the L2 data cache (or "
            "unified L2 cache). Includes hits and misses from L1D refills, "
            "L1I refills, and hardware prefetches."
        ),
        "when_to_use": (
            "Profile L2 cache behavior. High L2 access rate with low L2 "
            "miss rate (L2D_CACHE_REFILL / L2D_CACHE) means data fits "
            "within L1+L2 capacity."
        ),
        "formula_tips": [
            "L2 miss rate = L2D_CACHE_REFILL / L2D_CACHE",
            "L2 MPKI = L2D_CACHE_REFILL * 1000 / INST_RETIRED",
        ],
        "linux_perf_name": "l2d_cache",
        "ai_workload_tips": (
            "For ML inference, tile sizes should be chosen so that working "
            "sets fit in L2 (typically 256KB-1MB per core). Monitor L2 miss "
            "rate to validate tiling strategy."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x16  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L2D_CACHE_REFILL": {
        "event_name": "L2D_CACHE_REFILL",
        "event_number": "0x17",
        "category": "cache",
        "description": (
            "Counts each cache line refill into the L2 data cache caused "
            "by an L2 miss. This data must be fetched from L3/LLC or main "
            "memory, incurring significant latency (50-200+ cycles)."
        ),
        "when_to_use": (
            "Profile workloads that exceed L2 capacity. L2 misses are "
            "expensive and often the dominant source of backend stalls "
            "on memory-bound workloads."
        ),
        "formula_tips": [
            "L2 miss rate = L2D_CACHE_REFILL / L2D_CACHE",
            "L2 MPKI = L2D_CACHE_REFILL * 1000 / INST_RETIRED",
            "Estimated bandwidth = L2D_CACHE_REFILL * 64 / time_seconds (bytes/sec)",
        ],
        "linux_perf_name": "l2d_cache_refill",
        "ai_workload_tips": (
            "L2 misses are the biggest performance limiter for many ML workloads. "
            "Use cache-blocking / tiling to keep working sets in L2. For "
            "Neoverse V2 (1MB L2), target tile sizes < 768KB."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x17  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L2D_CACHE_WB": {
        "event_name": "L2D_CACHE_WB",
        "event_number": "0x18",
        "category": "cache",
        "description": (
            "Counts each write-back of a dirty cache line from the L2 cache "
            "to the next level (L3/LLC or main memory). Each write-back "
            "consumes memory bandwidth."
        ),
        "when_to_use": (
            "Profile write-heavy workloads that generate L2 write-backs. "
            "High L2 write-backs combined with L2 refills indicate the "
            "working set significantly exceeds L2 capacity."
        ),
        "formula_tips": [
            "L2 write-back ratio = L2D_CACHE_WB / L2D_CACHE",
            "Total L2 bandwidth = (L2D_CACHE_REFILL + L2D_CACHE_WB) * 64 / time",
        ],
        "linux_perf_name": "l2d_cache_wb",
        "ai_workload_tips": (
            "Training workloads generate large numbers of L2 write-backs "
            "due to gradient computation. Monitor total L2 bandwidth to "
            "ensure memory subsystem is not saturated."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x18  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "BUS_ACCESS": {
        "event_name": "BUS_ACCESS",
        "event_number": "0x19",
        "category": "memory",
        "description": (
            "Counts each access to the external bus or interconnect (e.g., "
            "AMBA/AXI). This includes LLC misses that must be serviced from "
            "main memory or another core's cache (coherence traffic)."
        ),
        "when_to_use": (
            "Profile memory bandwidth utilization. High bus access rates "
            "may indicate memory bandwidth saturation, which limits "
            "scalability on multi-core workloads."
        ),
        "formula_tips": [
            "Bus bandwidth = BUS_ACCESS * cache_line_size / time_seconds",
            "Bus accesses per instruction = BUS_ACCESS / INST_RETIRED",
        ],
        "linux_perf_name": "bus_access",
        "ai_workload_tips": (
            "For multi-core ML inference, monitor BUS_ACCESS to detect "
            "memory bandwidth contention. Consider NUMA-aware allocation "
            "and per-core data partitioning."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x19  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "MEMORY_ERROR": {
        "event_name": "MEMORY_ERROR",
        "event_number": "0x1A",
        "category": "memory",
        "description": (
            "Counts local memory errors that have been corrected (ECC) or "
            "detected. This includes single-bit correctable errors (CE) in "
            "caches, TLBs, and RAM."
        ),
        "when_to_use": (
            "Monitor memory subsystem health. Non-zero counts may indicate "
            "hardware issues (aging DRAM, thermal problems). Track over time "
            "to detect degradation trends."
        ),
        "formula_tips": [
            "Error rate = MEMORY_ERROR / CPU_CYCLES",
            "Any non-zero count warrants investigation",
        ],
        "linux_perf_name": "memory_error",
        "ai_workload_tips": (
            "Memory errors can cause silent data corruption in ML workloads, "
            "leading to incorrect inference results or training divergence. "
            "Enable ECC and monitor this counter in production."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x1A  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "BUS_CYCLES": {
        "event_name": "BUS_CYCLES",
        "event_number": "0x1D",
        "category": "memory",
        "description": (
            "Counts bus clock cycles. The bus may run at a different "
            "frequency than the CPU core, so BUS_CYCLES and CPU_CYCLES "
            "may increment at different rates."
        ),
        "when_to_use": (
            "Measure bus utilization and determine the ratio of bus "
            "frequency to core frequency. Useful for understanding "
            "memory subsystem timing."
        ),
        "formula_tips": [
            "Bus/Core frequency ratio = BUS_CYCLES / CPU_CYCLES",
            "Bus utilization = BUS_ACCESS / BUS_CYCLES",
        ],
        "linux_perf_name": "bus_cycles",
        "ai_workload_tips": (
            "If bus utilization is high (close to 1.0), memory bandwidth "
            "is saturated. Consider reducing data movement or using "
            "compression for tensor data."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x1D  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "CHAIN": {
        "event_name": "CHAIN",
        "event_number": "0x1E",
        "category": "misc",
        "description": (
            "Used for chaining two consecutive PMU counters to form a "
            "single 64-bit counter. When counter N overflows, it increments "
            "counter N+1 if counter N+1 is programmed with the CHAIN event. "
            "This allows counting beyond the 32-bit limit of a single counter."
        ),
        "when_to_use": (
            "Use when you need to count events that exceed 2^32 (about 4 "
            "billion). Common for CPU_CYCLES or INST_RETIRED on long-running "
            "workloads without interrupt-based overflow handling."
        ),
        "formula_tips": [
            "64-bit count = (counter_N+1 << 32) | counter_N",
            "Consumes two PMU counters for one 64-bit measurement",
        ],
        "linux_perf_name": "chain",
        "ai_workload_tips": (
            "Long-running training jobs may overflow 32-bit counters. "
            "Linux perf handles this automatically via interrupt-on-overflow, "
            "but bare-metal profiling may need CHAIN."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x1E  // Select event (counter 1 = CHAIN)\nMSR PMEVTYPER1_EL0, #0x11  // Counter 0 = CPU_CYCLES\nMSR PMCNTENSET_EL0, #0x03  // Enable counters 0 and 1",
    },
    "L1D_CACHE_ALLOCATE": {
        "event_name": "L1D_CACHE_ALLOCATE",
        "event_number": "0x1F",
        "category": "cache",
        "description": (
            "Counts each allocation of a cache line into the L1 data cache. "
            "This is similar to L1D_CACHE_REFILL but may also count "
            "speculative allocations depending on the implementation."
        ),
        "when_to_use": (
            "Compare with L1D_CACHE_REFILL to detect speculative cache "
            "allocations. A large difference indicates the prefetcher or "
            "speculative execution is populating the cache."
        ),
        "formula_tips": [
            "Speculative allocation ratio = (L1D_CACHE_ALLOCATE - L1D_CACHE_REFILL) / L1D_CACHE_ALLOCATE",
        ],
        "linux_perf_name": "l1d_cache_allocate",
        "ai_workload_tips": (
            "If L1D_CACHE_ALLOCATE >> L1D_CACHE_REFILL, the hardware "
            "prefetcher is active. This is generally beneficial for ML "
            "workloads with sequential access patterns."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x1F  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "STALL_FRONTEND": {
        "event_name": "STALL_FRONTEND",
        "event_number": "0x23",
        "category": "stall",
        "description": (
            "Counts cycles where the frontend (fetch and decode stages) "
            "cannot deliver instructions to the backend. Causes include "
            "L1I cache misses, ITLB misses, branch mispredictions that "
            "flush the pipeline, and instruction buffer starvation."
        ),
        "when_to_use": (
            "Use in TopDown analysis to determine if the workload is "
            "frontend-bound. High STALL_FRONTEND / CPU_CYCLES indicates "
            "instruction delivery is the bottleneck."
        ),
        "formula_tips": [
            "Frontend Bound = STALL_FRONTEND / CPU_CYCLES",
            "If Frontend Bound > 20%, investigate L1I misses and branch prediction",
        ],
        "linux_perf_name": "stall_frontend",
        "ai_workload_tips": (
            "ML frameworks with large codebases and deep call stacks can be "
            "frontend-bound. Use PGO, LTO, and BOLT to optimize code layout "
            "and reduce frontend stalls."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x23  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "STALL_BACKEND": {
        "event_name": "STALL_BACKEND",
        "event_number": "0x24",
        "category": "stall",
        "description": (
            "Counts cycles where the backend (execution and memory stages) "
            "cannot accept instructions from the frontend. Causes include "
            "L1D/L2 cache misses, DTLB misses, long-latency operations "
            "(divides, memory barriers), and resource exhaustion (ROB full, "
            "reservation station full)."
        ),
        "when_to_use": (
            "Use in TopDown analysis to determine if the workload is "
            "backend-bound. High STALL_BACKEND / CPU_CYCLES indicates "
            "data access or execution resources are the bottleneck."
        ),
        "formula_tips": [
            "Backend Bound = STALL_BACKEND / CPU_CYCLES",
            "If Backend Bound > 30%, investigate cache misses and memory latency",
        ],
        "linux_perf_name": "stall_backend",
        "ai_workload_tips": (
            "Most ML workloads are backend-bound due to memory access patterns. "
            "Reduce backend stalls by improving data locality (tiling, packing), "
            "using prefetch hints, and optimizing memory layout."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x24  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L1D_TLB": {
        "event_name": "L1D_TLB",
        "event_number": "0x25",
        "category": "tlb",
        "description": (
            "Counts each access that looks up in the L1 data TLB. "
            "Includes both hits and misses. Provides the total data "
            "TLB access count for miss rate calculations."
        ),
        "when_to_use": (
            "Use as denominator for L1 DTLB miss rate calculation. "
            "Compare with L1D_TLB_REFILL to determine TLB pressure."
        ),
        "formula_tips": [
            "DTLB miss rate = L1D_TLB_REFILL / L1D_TLB",
            "TLB accesses per instruction = L1D_TLB / INST_RETIRED",
        ],
        "linux_perf_name": "l1d_tlb",
        "ai_workload_tips": (
            "High TLB access rate with high miss rate suggests data is "
            "spread across too many pages. Use huge pages for tensor "
            "allocations to reduce TLB pressure."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x25  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L1I_TLB": {
        "event_name": "L1I_TLB",
        "event_number": "0x26",
        "category": "tlb",
        "description": (
            "Counts each access that looks up in the L1 instruction TLB. "
            "Includes both hits and misses. Provides the total instruction "
            "TLB access count."
        ),
        "when_to_use": (
            "Use as denominator for L1 ITLB miss rate. High ITLB miss "
            "rate can cause frontend stalls."
        ),
        "formula_tips": [
            "ITLB miss rate = L1I_TLB_REFILL / L1I_TLB",
        ],
        "linux_perf_name": "l1i_tlb",
        "ai_workload_tips": (
            "ITLB misses are rare unless code is spread across many pages. "
            "Consider huge pages for code or use BOLT to compact hot code."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x26  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "BR_RETIRED": {
        "event_name": "BR_RETIRED",
        "event_number": "0x21",
        "category": "branch",
        "description": (
            "Counts each branch instruction that is architecturally executed "
            "(retired). This is the definitive count of branches that "
            "completed execution, excluding speculated branches."
        ),
        "when_to_use": (
            "Profile branch frequency in your code. High branch density "
            "may indicate opportunities for branchless optimizations or "
            "loop vectorization."
        ),
        "formula_tips": [
            "Branch density = BR_RETIRED / INST_RETIRED",
            "Mispredict rate (retired) = BR_MIS_PRED_RETIRED / BR_RETIRED",
        ],
        "linux_perf_name": "br_retired",
        "ai_workload_tips": (
            "Vectorized ML kernels should have low branch density. High "
            "branch density in hot loops suggests scalar fallback paths "
            "are being taken."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x21  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "BR_MIS_PRED_RETIRED": {
        "event_name": "BR_MIS_PRED_RETIRED",
        "event_number": "0x22",
        "category": "branch",
        "description": (
            "Counts each mispredicted branch instruction that is "
            "architecturally executed (retired). Unlike BR_MIS_PRED, this "
            "only counts branches that actually completed, not speculative "
            "mispredictions on wrong-path instructions."
        ),
        "when_to_use": (
            "More accurate than BR_MIS_PRED for measuring true mispredict "
            "cost. Use for precise branch analysis where speculative "
            "overcounting would be misleading."
        ),
        "formula_tips": [
            "Precise mispredict rate = BR_MIS_PRED_RETIRED / BR_RETIRED",
            "Speculation noise = BR_MIS_PRED - BR_MIS_PRED_RETIRED",
        ],
        "linux_perf_name": "br_mis_pred_retired",
        "ai_workload_tips": (
            "For production inference profiling, prefer this over BR_MIS_PRED "
            "for accurate attribution of misprediction overhead to specific "
            "code regions."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x22  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L2D_TLB_REFILL": {
        "event_name": "L2D_TLB_REFILL",
        "event_number": "0x2D",
        "category": "tlb",
        "description": (
            "Counts each refill of the L2 unified TLB caused by a data or "
            "instruction access that misses in both L1 TLB and L2 TLB. "
            "This triggers a full page table walk, which is very expensive "
            "(100+ cycles)."
        ),
        "when_to_use": (
            "Profile workloads with very large memory footprints. L2 TLB "
            "misses are extremely costly due to page table walks through "
            "multiple levels of translation tables."
        ),
        "formula_tips": [
            "L2 TLB miss rate = L2D_TLB_REFILL / L2D_TLB",
            "Page walk rate = L2D_TLB_REFILL / INST_RETIRED",
        ],
        "linux_perf_name": "l2d_tlb_refill",
        "ai_workload_tips": (
            "Large embedding tables and feature maps can cause L2 TLB misses. "
            "Use 2MB or 1GB huge pages (hugetlbfs or transparent huge pages) "
            "to drastically reduce page walks."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x2D  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "L2D_TLB": {
        "event_name": "L2D_TLB",
        "event_number": "0x2F",
        "category": "tlb",
        "description": (
            "Counts each access that looks up in the L2 unified TLB. "
            "This is the total L2 TLB access count including hits and "
            "misses from L1 TLB refills."
        ),
        "when_to_use": (
            "Use as denominator for L2 TLB miss rate. High L2 TLB access "
            "count indicates significant L1 TLB miss traffic."
        ),
        "formula_tips": [
            "L2 TLB miss rate = L2D_TLB_REFILL / L2D_TLB",
        ],
        "linux_perf_name": "l2d_tlb",
        "ai_workload_tips": (
            "If L2 TLB miss rate is high, huge pages are essential. For "
            "Graviton3/4 with 48-entry L1 DTLB and 1024-entry L2 TLB, "
            "2MB pages cover up to 2GB of data TLB-efficiently."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x2F  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "LL_CACHE_RD": {
        "event_name": "LL_CACHE_RD",
        "event_number": "0x36",
        "category": "cache",
        "description": (
            "Counts read accesses to the Last Level Cache (LLC / L3). "
            "These are accesses that missed in L2 and must be serviced "
            "by the shared LLC. LLC is typically shared across a cluster "
            "or all cores."
        ),
        "when_to_use": (
            "Profile workloads that exceed per-core L2 capacity. High LLC "
            "read rate indicates significant off-core memory traffic."
        ),
        "formula_tips": [
            "LLC miss rate = LL_CACHE_MISS_RD / LL_CACHE_RD",
            "LLC MPKI = LL_CACHE_RD * 1000 / INST_RETIRED",
        ],
        "linux_perf_name": "ll_cache_rd",
        "ai_workload_tips": (
            "LLC accesses are shared across cores. For multi-threaded ML "
            "inference, monitor per-core LLC access rates to detect cache "
            "contention between threads."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x36  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "LL_CACHE_MISS_RD": {
        "event_name": "LL_CACHE_MISS_RD",
        "event_number": "0x37",
        "category": "cache",
        "description": (
            "Counts read accesses that miss in the Last Level Cache (LLC). "
            "These must be serviced from main memory (DRAM), incurring "
            "the highest latency in the memory hierarchy (100-300+ ns)."
        ),
        "when_to_use": (
            "Profile workloads that are memory-bandwidth-bound. LLC read "
            "misses directly translate to DRAM traffic and are the most "
            "expensive cache events."
        ),
        "formula_tips": [
            "LLC miss rate = LL_CACHE_MISS_RD / LL_CACHE_RD",
            "DRAM bandwidth = LL_CACHE_MISS_RD * 64 / time_seconds",
            "LLC miss MPKI = LL_CACHE_MISS_RD * 1000 / INST_RETIRED",
        ],
        "linux_perf_name": "ll_cache_miss_rd",
        "ai_workload_tips": (
            "LLC misses are the dominant cost for large ML models. For "
            "transformer inference, KV-cache sizes often exceed LLC. "
            "Consider quantization (INT8/INT4) to reduce memory footprint "
            "and LLC miss rate."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x37  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "DTLB_WALK": {
        "event_name": "DTLB_WALK",
        "event_number": "0x34",
        "category": "tlb",
        "description": (
            "Counts each data-side page table walk. A walk is triggered "
            "when a data access misses all levels of the data TLB "
            "hierarchy. Each walk accesses multiple levels of translation "
            "tables in memory."
        ),
        "when_to_use": (
            "More precise than TLB refill events for measuring page walk "
            "overhead. Each walk may access 3-4 levels of page tables, "
            "each requiring a memory access."
        ),
        "formula_tips": [
            "Walk rate = DTLB_WALK / INST_RETIRED",
            "Walk cost (cycles) = DTLB_WALK * avg_walk_latency",
            "Compare with L1D_TLB_REFILL to understand TLB hierarchy",
        ],
        "linux_perf_name": "dtlb_walk",
        "ai_workload_tips": (
            "Page walks are very expensive (100+ cycles each). For ML "
            "workloads with large memory footprints, DTLB_WALK > 0.1% "
            "of instructions warrants huge page configuration."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x34  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "ITLB_WALK": {
        "event_name": "ITLB_WALK",
        "event_number": "0x35",
        "category": "tlb",
        "description": (
            "Counts each instruction-side page table walk. Triggered when "
            "an instruction fetch misses all levels of the instruction TLB "
            "hierarchy. Causes frontend stalls while the walk completes."
        ),
        "when_to_use": (
            "Profile applications with very large instruction footprints "
            "that cause ITLB misses requiring page walks."
        ),
        "formula_tips": [
            "ITLB walk rate = ITLB_WALK / INST_RETIRED",
            "Frontend stall contribution = ITLB_WALK * walk_latency / CPU_CYCLES",
        ],
        "linux_perf_name": "itlb_walk",
        "ai_workload_tips": (
            "Rare in typical ML inference. If observed, indicates extremely "
            "large code footprint — consider static linking, LTO, or BOLT "
            "for code layout optimization."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x35  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "OP_RETIRED": {
        "event_name": "OP_RETIRED",
        "event_number": "0x3A",
        "category": "instruction",
        "description": (
            "Counts micro-operations (uops) that are architecturally "
            "retired. A single ARM instruction may decode into multiple "
            "micro-ops. This counts the actual work completed by the "
            "processor at the micro-op level."
        ),
        "when_to_use": (
            "Use in TopDown methodology for the Retiring metric. Compare "
            "with OP_SPEC to measure speculation overhead."
        ),
        "formula_tips": [
            "Retiring (TopDown) = OP_RETIRED / OP_SPEC",
            "Uops per instruction = OP_RETIRED / INST_RETIRED",
            "Speculation waste = 1 - (OP_RETIRED / OP_SPEC)",
        ],
        "linux_perf_name": "op_retired",
        "ai_workload_tips": (
            "Well-optimized ML kernels should show high Retiring ratio "
            "(OP_RETIRED / OP_SPEC > 0.5). Low retiring ratio means the "
            "processor is spending cycles on speculated-then-flushed work."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x3A  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "OP_SPEC": {
        "event_name": "OP_SPEC",
        "event_number": "0x3B",
        "category": "instruction",
        "description": (
            "Counts micro-operations speculatively executed, including "
            "those that are later flushed due to branch misprediction "
            "or other speculation failures. This is the total pipeline "
            "throughput at the micro-op level."
        ),
        "when_to_use": (
            "Use in TopDown methodology as the total pipeline slots "
            "denominator. Compare with OP_RETIRED to measure useful vs "
            "wasted pipeline work."
        ),
        "formula_tips": [
            "Retiring = OP_RETIRED / OP_SPEC",
            "Bad Speculation = 1 - Frontend_Bound - Backend_Bound - Retiring",
            "Speculation overhead = (OP_SPEC - OP_RETIRED) / OP_SPEC",
        ],
        "linux_perf_name": "op_spec",
        "ai_workload_tips": (
            "OP_SPEC significantly exceeding OP_RETIRED indicates bad "
            "speculation. In ML workloads this is usually from data-dependent "
            "branches in activation functions or sparse operations."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x3B  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
    "STALL_SLOT": {
        "event_name": "STALL_SLOT",
        "event_number": "0x3F",
        "category": "stall",
        "description": (
            "Counts pipeline slot-cycles where no operation is issued. "
            "This is a more granular version of stall counting that "
            "accounts for the pipeline width (e.g., 4-wide or 8-wide "
            "issue). Used in the ARM TopDown methodology."
        ),
        "when_to_use": (
            "Use for detailed TopDown analysis. STALL_SLOT gives a more "
            "accurate picture than STALL_FRONTEND + STALL_BACKEND on "
            "wide-issue processors because it accounts for partial stalls."
        ),
        "formula_tips": [
            "Total slots = CPU_CYCLES * pipeline_width",
            "Stall ratio = STALL_SLOT / (CPU_CYCLES * pipeline_width)",
            "Useful for TopDown Level 1 decomposition",
        ],
        "linux_perf_name": "stall_slot",
        "ai_workload_tips": (
            "For detailed performance analysis of ML kernels on Neoverse "
            "cores, STALL_SLOT provides the highest-fidelity stall metric. "
            "Combine with STALL_FRONTEND and STALL_BACKEND to classify stalls."
        ),
        "register_setup": "MSR PMEVTYPER0_EL0, #0x3F  // Select event\nMSR PMCNTENSET_EL0, #0x01  // Enable counter 0",
    },
}

# Build a reverse lookup from event number (hex string) to event name
_PMU_EVENT_BY_NUMBER: dict[str, str] = {}
for _name, _data in _PMU_EVENTS.items():
    _PMU_EVENT_BY_NUMBER[_data["event_number"].upper()] = _name


def _format_pmu_event(event: dict) -> str:
    """Format a single PMU event entry for display."""
    lines: list[str] = []
    lines.append(f"# PMU Event: {event['event_name']} ({event['event_number']})")
    lines.append("")
    lines.append(f"**Category:** {event['category'].capitalize()}")
    lines.append(f"**Linux perf:** `perf stat -e {event['linux_perf_name']}`")
    lines.append("")
    lines.append("## What It Measures")
    lines.append(event["description"])
    lines.append("")
    lines.append("## When To Use")
    lines.append(event["when_to_use"])
    lines.append("")
    if event["formula_tips"]:
        lines.append("## Useful Formulas")
        for tip in event["formula_tips"]:
            lines.append(f"- {tip}")
        lines.append("")
    lines.append("## AI/ML Workload Tips")
    lines.append(event["ai_workload_tips"])
    lines.append("")
    lines.append("## PMU Setup (bare metal)")
    lines.append("```asm")
    lines.append(event["register_setup"])
    lines.append("```")
    return "\n".join(lines)


def _format_pmu_overview() -> str:
    """Format a table of all PMU events grouped by category."""
    # Group events by category
    categories: dict[str, list[dict]] = {}
    for event in _PMU_EVENTS.values():
        cat = event["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(event)

    lines: list[str] = []
    lines.append("# ARM PMU Events Overview")
    lines.append("")
    lines.append(f"Total events: {len(_PMU_EVENTS)}")
    lines.append("")

    category_order = ["cache", "tlb", "branch", "instruction", "memory", "stall", "misc"]
    for cat in category_order:
        if cat not in categories:
            continue
        events = sorted(categories[cat], key=lambda e: e["event_number"])
        lines.append(f"## {cat.capitalize()} Events ({len(events)})")
        lines.append("")
        lines.append(f"  {'Event':<28} {'Number':<8} Description")
        lines.append(f"  {'-----':<28} {'------':<8} -----------")
        for ev in events:
            desc = ev["description"]
            # Truncate description for table
            if len(desc) > 60:
                desc = desc[:57] + "..."
            lines.append(f"  {ev['event_name']:<28} {ev['event_number']:<8} {desc}")
        lines.append("")

    lines.append("## Usage")
    lines.append("")
    lines.append("Query individual events by name: explain_performance_counter('L1D_CACHE_REFILL')")
    lines.append("Query by hex number: explain_performance_counter('0x03')")
    lines.append("Query by category: explain_performance_counter('cache')")
    lines.append("TopDown methodology: explain_performance_counter('topdown')")
    return "\n".join(lines)


def _format_pmu_category(category: str) -> str:
    """Format all events in a given category."""
    events = [e for e in _PMU_EVENTS.values() if e["category"] == category]
    if not events:
        return f"Error: No events found in category '{category}'."

    events.sort(key=lambda e: e["event_number"])
    lines: list[str] = []
    lines.append(f"# PMU Events — {category.capitalize()} Category ({len(events)} events)")
    lines.append("")

    for ev in events:
        lines.append(f"## {ev['event_name']} ({ev['event_number']})")
        lines.append(f"**Linux perf:** `perf stat -e {ev['linux_perf_name']}`")
        lines.append("")
        lines.append(ev["description"])
        lines.append("")
        lines.append(f"**When to use:** {ev['when_to_use']}")
        lines.append("")
        if ev["formula_tips"]:
            for tip in ev["formula_tips"]:
                lines.append(f"- {tip}")
            lines.append("")

    return "\n".join(lines)


def _format_topdown_guide() -> str:
    """Format a TopDown methodology guide using ARM PMU events."""
    lines: list[str] = []
    lines.append("# ARM TopDown Performance Analysis Methodology")
    lines.append("")
    lines.append("The TopDown methodology classifies CPU pipeline slots into four")
    lines.append("categories to identify the primary performance bottleneck.")
    lines.append("")
    lines.append("## Level 1 Breakdown")
    lines.append("")
    lines.append("### Frontend Bound")
    lines.append("  Formula: Frontend Bound = STALL_FRONTEND / CPU_CYCLES")
    lines.append("  The frontend cannot deliver instructions fast enough.")
    lines.append("  Causes: L1I misses, ITLB misses, branch misprediction flushes.")
    lines.append("  Fix: PGO, LTO, BOLT, reduce code size, improve branch prediction.")
    lines.append("")
    lines.append("### Backend Bound")
    lines.append("  Formula: Backend Bound = STALL_BACKEND / CPU_CYCLES")
    lines.append("  The backend cannot consume instructions fast enough.")
    lines.append("  Causes: L1D/L2/LLC misses, DTLB misses, long-latency ops, resource exhaustion.")
    lines.append("  Fix: Improve data locality, cache tiling, prefetching, reduce memory traffic.")
    lines.append("")
    lines.append("### Retiring")
    lines.append("  Formula: Retiring = OP_RETIRED / OP_SPEC")
    lines.append("  Fraction of pipeline slots used for useful (retired) work.")
    lines.append("  High retiring (> 50%) is good — the CPU is doing useful work.")
    lines.append("  To improve further: vectorize (NEON/SVE), reduce instruction count.")
    lines.append("")
    lines.append("### Bad Speculation")
    lines.append("  Formula: Bad Speculation = 1 - Frontend Bound - Backend Bound - Retiring")
    lines.append("  Pipeline slots wasted on instructions that are eventually flushed.")
    lines.append("  Causes: branch mispredictions, memory ordering violations.")
    lines.append("  Fix: branchless code (CSEL), better branch hints, reduce unpredictable branches.")
    lines.append("")
    lines.append("## Suggested perf stat Command")
    lines.append("")
    lines.append("```bash")
    lines.append("perf stat -e cpu_cycles,stall_frontend,stall_backend,op_retired,op_spec,\\")
    lines.append("inst_retired,br_mis_pred,l1d_cache_refill,l1d_cache,\\")
    lines.append("l2d_cache_refill,l2d_cache,ll_cache_miss_rd \\")
    lines.append("-- ./your_workload")
    lines.append("```")
    lines.append("")
    lines.append("## Interpreting Results")
    lines.append("")
    lines.append("  1. Compute the four Level-1 categories above.")
    lines.append("  2. The largest category is your primary bottleneck.")
    lines.append("  3. Drill down:")
    lines.append("     - Frontend Bound -> check L1I_CACHE_REFILL, L1I_TLB_REFILL, ITLB_WALK")
    lines.append("     - Backend Bound  -> check L1D_CACHE_REFILL, L2D_CACHE_REFILL, LL_CACHE_MISS_RD, DTLB_WALK")
    lines.append("     - Bad Speculation -> check BR_MIS_PRED, BR_MIS_PRED_RETIRED")
    lines.append("     - Retiring (high) -> vectorize more, use wider SIMD (SVE vs NEON)")
    lines.append("")
    lines.append("## ARM-Specific Notes")
    lines.append("")
    lines.append("- ARM PMU has limited counters (typically 6 programmable + cycle counter).")
    lines.append("  You may need to multiplex events across multiple runs.")
    lines.append("- STALL_SLOT (0x3F) provides slot-level granularity on Neoverse cores.")
    lines.append("- Some events are micro-architecture specific (Cortex-A78 vs Neoverse V2).")
    lines.append("- Use `perf stat --topdown` if supported by your kernel and PMU driver.")
    return "\n".join(lines)


@mcp.tool()
def explain_performance_counter(event_name: str) -> str:
    """Look up an ARM PMU (Performance Monitoring Unit) event by name or event number.

    Explains what the event measures, when to use it, useful formulas,
    AI/ML workload tips, and how to configure the PMU registers.

    Args:
        event_name: One of:
            An event name like "L1D_CACHE_REFILL", "CPU_CYCLES", "STALL_BACKEND"
            A hex event number like "0x03", "0x11"
            "list" or "overview" -- Summary table of ALL events grouped by category
            A category name: "cache", "tlb", "branch", "instruction", "memory", "stall", "misc"
            "topdown" -- ARM TopDown performance analysis methodology guide
    """
    key = event_name.strip()

    # Check for "list" or "overview"
    if key.upper() in ("LIST", "OVERVIEW"):
        return _format_pmu_overview()

    # Check for "topdown"
    if key.upper() == "TOPDOWN":
        return _format_topdown_guide()

    # Check for hex event number (e.g., "0x03")
    if key.upper().startswith("0X"):
        hex_key = key.upper()
        # Normalize to standard format: 0x + uppercase hex digits
        # Handle variations like "0x3" -> "0x03", "0X03" -> "0X03"
        try:
            num = int(hex_key, 16)
            # Try both "0xNN" formats
            for fmt in [f"0x{num:02X}", f"0x{num:X}"]:
                if fmt.upper() in _PMU_EVENT_BY_NUMBER:
                    name = _PMU_EVENT_BY_NUMBER[fmt.upper()]
                    return _format_pmu_event(_PMU_EVENTS[name])
        except ValueError:
            pass
        return (
            f"Error: No PMU event found with number '{event_name}'.\n\n"
            f"Use 'list' to see all available events and their numbers."
        )

    # Check for category name
    categories = {"cache", "tlb", "branch", "instruction", "memory", "stall", "misc"}
    if key.lower() in categories:
        return _format_pmu_category(key.lower())

    # Look up by event name (case-insensitive)
    upper_key = key.upper()
    if upper_key in _PMU_EVENTS:
        return _format_pmu_event(_PMU_EVENTS[upper_key])

    # Not found
    valid_names = ", ".join(sorted(_PMU_EVENTS.keys()))
    return (
        f"Error: '{event_name}' is not a recognised PMU event.\n\n"
        f"Valid event names: {valid_names}\n\n"
        f"You can also query by:\n"
        f"  - Hex event number (e.g., '0x03')\n"
        f"  - Category ('cache', 'tlb', 'branch', 'instruction', 'memory', 'stall', 'misc')\n"
        f"  - 'list' or 'overview' for all events\n"
        f"  - 'topdown' for the TopDown methodology guide"
    )

# ---------------------------------------------------------------------------
# Tool 18: translate_intrinsic  —  Translate SIMD intrinsics between x86 and ARM
# ---------------------------------------------------------------------------

_INTRINSIC_TRANSLATIONS: dict[str, dict] = {
    # === SSE float (128-bit, 4xf32) ===
    "_mm_add_ps": {
        "x86_intrinsic": "_mm_add_ps",
        "x86_instruction": "ADDPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "vaddq_f32",
        "neon_instruction": "FADD Vd.4S, Vn.4S, Vm.4S",
        "sve_intrinsic": "svadd_f32_x(pg, a, b)",
        "sve_instruction": "FADD Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "4x float32 (128-bit)",
        "notes": "Direct 1:1 translation. Both operate on 4 packed floats.",
        "porting_tips": "Drop-in replacement. No semantic difference.",
        "gotchas": "",
    },
    "_mm_sub_ps": {
        "x86_intrinsic": "_mm_sub_ps",
        "x86_instruction": "SUBPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "vsubq_f32",
        "neon_instruction": "FSUB Vd.4S, Vn.4S, Vm.4S",
        "sve_intrinsic": "svsub_f32_x(pg, a, b)",
        "sve_instruction": "FSUB Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "4x float32 (128-bit)",
        "notes": "Direct 1:1 translation. Both operate on 4 packed floats.",
        "porting_tips": "Drop-in replacement. No semantic difference.",
        "gotchas": "",
    },
    "_mm_mul_ps": {
        "x86_intrinsic": "_mm_mul_ps",
        "x86_instruction": "MULPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "vmulq_f32",
        "neon_instruction": "FMUL Vd.4S, Vn.4S, Vm.4S",
        "sve_intrinsic": "svmul_f32_x(pg, a, b)",
        "sve_instruction": "FMUL Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "4x float32 (128-bit)",
        "notes": "Direct 1:1 translation. Both operate on 4 packed floats.",
        "porting_tips": "Drop-in replacement. No semantic difference.",
        "gotchas": "",
    },
    "_mm_div_ps": {
        "x86_intrinsic": "_mm_div_ps",
        "x86_instruction": "DIVPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "vdivq_f32",
        "neon_instruction": "FDIV Vd.4S, Vn.4S, Vm.4S",
        "sve_intrinsic": "svdiv_f32_x(pg, a, b)",
        "sve_instruction": "FDIV Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "4x float32 (128-bit)",
        "notes": "NEON FDIV is AArch64 only. On AArch32 NEON, use VRECPE+VRECPS for approximate reciprocal then multiply.",
        "porting_tips": "On AArch64 this is a drop-in replacement. On AArch32 you need a reciprocal approximation sequence.",
        "gotchas": "AArch32 NEON has no direct divide instruction. vdivq_f32 is only available on AArch64.",
    },
    "_mm_sqrt_ps": {
        "x86_intrinsic": "_mm_sqrt_ps",
        "x86_instruction": "SQRTPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "vsqrtq_f32",
        "neon_instruction": "FSQRT Vd.4S, Vn.4S",
        "sve_intrinsic": "svsqrt_f32_x(pg, a)",
        "sve_instruction": "FSQRT Zd.S, Pg/M, Zn.S",
        "data_type": "4x float32 (128-bit)",
        "notes": "NEON FSQRT is AArch64 only. On AArch32, use VRSQRTE+VRSQRTS for approximate inverse sqrt then reciprocal.",
        "porting_tips": "On AArch64 this is a drop-in replacement.",
        "gotchas": "AArch32 NEON has no direct sqrt instruction. vsqrtq_f32 is only available on AArch64.",
    },
    "_mm_min_ps": {
        "x86_intrinsic": "_mm_min_ps",
        "x86_instruction": "MINPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "vminq_f32",
        "neon_instruction": "FMIN Vd.4S, Vn.4S, Vm.4S",
        "sve_intrinsic": "svmin_f32_x(pg, a, b)",
        "sve_instruction": "FMIN Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "4x float32 (128-bit)",
        "notes": "Direct 1:1 translation. NaN handling may differ slightly between architectures.",
        "porting_tips": "Drop-in replacement, but verify NaN behaviour if your code relies on NaN propagation.",
        "gotchas": "SSE MINPS and NEON FMIN differ in NaN handling: SSE returns the second operand if either is NaN, NEON returns NaN.",
    },
    "_mm_max_ps": {
        "x86_intrinsic": "_mm_max_ps",
        "x86_instruction": "MAXPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "vmaxq_f32",
        "neon_instruction": "FMAX Vd.4S, Vn.4S, Vm.4S",
        "sve_intrinsic": "svmax_f32_x(pg, a, b)",
        "sve_instruction": "FMAX Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "4x float32 (128-bit)",
        "notes": "Direct 1:1 translation. NaN handling may differ slightly between architectures.",
        "porting_tips": "Drop-in replacement, but verify NaN behaviour if your code relies on NaN propagation.",
        "gotchas": "SSE MAXPS and NEON FMAX differ in NaN handling: SSE returns the second operand if either is NaN, NEON returns NaN.",
    },
    "_mm_set1_ps": {
        "x86_intrinsic": "_mm_set1_ps",
        "x86_instruction": "SHUFPS / MOVSS+SHUFPS (compiler-dependent)",
        "x86_extension": "SSE",
        "neon_intrinsic": "vdupq_n_f32",
        "neon_instruction": "DUP Vd.4S, Rn (or from scalar)",
        "sve_intrinsic": "svdup_f32(a)",
        "sve_instruction": "DUP Zd.S, Rn (or FDUP for immediate)",
        "data_type": "4x float32 (128-bit)",
        "notes": "Broadcast a single float to all lanes. Direct equivalent.",
        "porting_tips": "Drop-in replacement. vdupq_n_f32(val) broadcasts val to all 4 lanes.",
        "gotchas": "",
    },
    "_mm_load_ps": {
        "x86_intrinsic": "_mm_load_ps",
        "x86_instruction": "MOVAPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "vld1q_f32",
        "neon_instruction": "LD1 {Vt.4S}, [Xn]",
        "sve_intrinsic": "svld1_f32(pg, ptr)",
        "sve_instruction": "LD1W Zt.S, Pg/Z, [Xn]",
        "data_type": "4x float32 (128-bit)",
        "notes": "_mm_load_ps requires 16-byte alignment; vld1q_f32 does not require alignment (though aligned is faster).",
        "porting_tips": "Drop-in replacement. NEON loads are naturally unaligned, so alignment faults won't occur.",
        "gotchas": "If using _mm_loadu_ps (unaligned), vld1q_f32 handles both aligned and unaligned.",
    },
    "_mm_store_ps": {
        "x86_intrinsic": "_mm_store_ps",
        "x86_instruction": "MOVAPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "vst1q_f32",
        "neon_instruction": "ST1 {Vt.4S}, [Xn]",
        "sve_intrinsic": "svst1_f32(pg, ptr, data)",
        "sve_instruction": "ST1W Zt.S, Pg, [Xn]",
        "data_type": "4x float32 (128-bit)",
        "notes": "_mm_store_ps requires 16-byte alignment; vst1q_f32 does not.",
        "porting_tips": "Drop-in replacement. NEON stores are naturally unaligned.",
        "gotchas": "",
    },
    "_mm_setzero_ps": {
        "x86_intrinsic": "_mm_setzero_ps",
        "x86_instruction": "XORPS (self-XOR idiom)",
        "x86_extension": "SSE",
        "neon_intrinsic": "vdupq_n_f32(0.0f)",
        "neon_instruction": "MOVI Vd.4S, #0",
        "sve_intrinsic": "svdup_f32(0.0f)",
        "sve_instruction": "FDUP Zd.S, #0.0 (or MOVPRFX+EOR)",
        "data_type": "4x float32 (128-bit)",
        "notes": "Set all elements to zero.",
        "porting_tips": "Use vdupq_n_f32(0.0f). The compiler will typically emit MOVI with zero.",
        "gotchas": "",
    },
    "_mm_cmpeq_ps": {
        "x86_intrinsic": "_mm_cmpeq_ps",
        "x86_instruction": "CMPPS (imm=0)",
        "x86_extension": "SSE",
        "neon_intrinsic": "vceqq_f32",
        "neon_instruction": "FCMEQ Vd.4S, Vn.4S, Vm.4S",
        "sve_intrinsic": "svcmpeq_f32(pg, a, b)",
        "sve_instruction": "FCMEQ Pd.S, Pg/Z, Zn.S, Zm.S",
        "data_type": "4x float32 (128-bit)",
        "notes": "Compare for equality. SSE returns bitmask in float lanes, NEON returns bitmask in integer lanes, SVE returns a predicate.",
        "porting_tips": "Result types differ: SSE returns __m128 with all-ones/all-zeros float lanes, NEON returns uint32x4_t, SVE returns svbool_t.",
        "gotchas": "Return type differences may require adjustments in surrounding code. NEON result is uint32x4_t, not float32x4_t.",
    },
    "_mm_cmpgt_ps": {
        "x86_intrinsic": "_mm_cmpgt_ps",
        "x86_instruction": "CMPPS (imm=6, NLT)",
        "x86_extension": "SSE",
        "neon_intrinsic": "vcgtq_f32",
        "neon_instruction": "FCMGT Vd.4S, Vn.4S, Vm.4S",
        "sve_intrinsic": "svcmpgt_f32(pg, a, b)",
        "sve_instruction": "FCMGT Pd.S, Pg/Z, Zn.S, Zm.S",
        "data_type": "4x float32 (128-bit)",
        "notes": "Compare for greater-than. Same return-type differences as cmpeq.",
        "porting_tips": "Result types differ: SSE returns __m128, NEON returns uint32x4_t, SVE returns svbool_t.",
        "gotchas": "Return type differences may require adjustments in surrounding code.",
    },
    "_mm_and_ps": {
        "x86_intrinsic": "_mm_and_ps",
        "x86_instruction": "ANDPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "vandq_u32 (reinterpret)",
        "neon_instruction": "AND Vd.16B, Vn.16B, Vm.16B",
        "sve_intrinsic": "svand_u32_x(pg, a, b)",
        "sve_instruction": "AND Zd.D, Zn.D, Zm.D",
        "data_type": "4x float32 (128-bit), bitwise",
        "notes": "Bitwise AND. On NEON, you must reinterpret float32x4_t to uint32x4_t first: vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)).",
        "porting_tips": "Requires vreinterpretq_u32_f32 / vreinterpretq_f32_u32 casts around the operation on NEON.",
        "gotchas": "NEON bitwise ops work on integer types. You must cast float to uint32 and back.",
    },
    "_mm_or_ps": {
        "x86_intrinsic": "_mm_or_ps",
        "x86_instruction": "ORPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "vorrq_u32 (reinterpret)",
        "neon_instruction": "ORR Vd.16B, Vn.16B, Vm.16B",
        "sve_intrinsic": "svor_u32_x(pg, a, b)",
        "sve_instruction": "ORR Zd.D, Zn.D, Zm.D",
        "data_type": "4x float32 (128-bit), bitwise",
        "notes": "Bitwise OR. On NEON, you must reinterpret float32x4_t to uint32x4_t first.",
        "porting_tips": "Requires vreinterpretq_u32_f32 / vreinterpretq_f32_u32 casts around the operation on NEON.",
        "gotchas": "NEON bitwise ops work on integer types. You must cast float to uint32 and back.",
    },
    "_mm_xor_ps": {
        "x86_intrinsic": "_mm_xor_ps",
        "x86_instruction": "XORPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "veorq_u32 (reinterpret)",
        "neon_instruction": "EOR Vd.16B, Vn.16B, Vm.16B",
        "sve_intrinsic": "sveor_u32_x(pg, a, b)",
        "sve_instruction": "EOR Zd.D, Zn.D, Zm.D",
        "data_type": "4x float32 (128-bit), bitwise",
        "notes": "Bitwise XOR. On NEON, you must reinterpret float32x4_t to uint32x4_t first.",
        "porting_tips": "Requires vreinterpretq_u32_f32 / vreinterpretq_f32_u32 casts around the operation on NEON.",
        "gotchas": "NEON bitwise ops work on integer types. You must cast float to uint32 and back.",
    },
    "_mm_shuffle_ps": {
        "x86_intrinsic": "_mm_shuffle_ps",
        "x86_instruction": "SHUFPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "combination of vzip/vuzp/vext/vtbl",
        "neon_instruction": "TBL Vd.16B, {Vn.16B}, Vm.16B (general case)",
        "sve_intrinsic": "svtbl_f32(data, indices)",
        "sve_instruction": "TBL Zd.S, {Zn.S}, Zm.S",
        "data_type": "4x float32 (128-bit)",
        "notes": "No single NEON intrinsic matches the full generality of _mm_shuffle_ps. For specific shuffle patterns, simpler instructions (VEXT, VZIP, VUZP, VREV) may suffice. For the general case, use VTBL/TBL.",
        "porting_tips": "Analyze the shuffle mask (_MM_SHUFFLE(d,c,b,a)) and pick the simplest NEON sequence. Common patterns: reverse = vrev64q_f32, interleave = vzipq_f32, broadcast = vdupq_laneq_f32.",
        "gotchas": "There is no single NEON equivalent. The replacement depends on the specific shuffle mask. TBL is the general fallback but is slower than specialized permute instructions.",
        "workaround": (
            "// General _mm_shuffle_ps replacement using TBL (AArch64 NEON)\n"
            "// For _mm_shuffle_ps(a, b, _MM_SHUFFLE(d,c,b_idx,a_idx)):\n"
            "float32x4_t shuffle_example(float32x4_t a, float32x4_t b) {\n"
            "    // Example: _MM_SHUFFLE(3,2,1,0) from two sources\n"
            "    // Combine both vectors into a table\n"
            "    uint8x16_t tbl_a = vreinterpretq_u8_f32(a);\n"
            "    uint8x16_t tbl_b = vreinterpretq_u8_f32(b);\n"
            "    // Build index vector (byte indices)\n"
            "    // Each float32 lane is 4 bytes, so lane N starts at byte N*4\n"
            "    uint8x16_t idx = {0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15};\n"
            "    // For lanes from 'a': indices 0-15, for lanes from 'b': use vqtbl2\n"
            "    uint8x16x2_t tbl = {tbl_a, tbl_b};\n"
            "    uint8x16_t result = vqtbl2q_u8(tbl, idx);\n"
            "    return vreinterpretq_f32_u8(result);\n"
            "}"
        ),
    },
    "_mm_movemask_ps": {
        "x86_intrinsic": "_mm_movemask_ps",
        "x86_instruction": "MOVMSKPS",
        "x86_extension": "SSE",
        "neon_intrinsic": "(no direct equivalent)",
        "neon_instruction": "(multi-instruction sequence)",
        "sve_intrinsic": "svcompact_f32(pg, data) or svptest_any/svptest_first for predicate queries",
        "sve_instruction": "COMPACT Zd.S, Pg, Zn.S",
        "data_type": "4x float32 (128-bit) → 4-bit integer mask",
        "notes": "Extracts the sign bits of each float lane into an integer bitmask. No single NEON equivalent exists.",
        "porting_tips": "On NEON, use a shift-and-narrow sequence. On SVE, predicates replace movemask patterns entirely.",
        "gotchas": "This is one of the hardest SSE intrinsics to port. Often the surrounding algorithm should be restructured to use NEON comparison results (uint32x4_t masks) directly instead of scalar bitmasks.",
        "workaround": (
            "// _mm_movemask_ps replacement for AArch64 NEON\n"
            "// Extracts sign bits of 4 float32 lanes into a 4-bit integer\n"
            "static inline int vmovemask_f32(float32x4_t v) {\n"
            "    uint32x4_t mask = vreinterpretq_u32_f32(v);\n"
            "    // Shift sign bits to bit 0 of each lane\n"
            "    uint32x4_t shifted = vshrq_n_u32(mask, 31);\n"
            "    // Pack lanes: lane0*1 + lane1*2 + lane2*4 + lane3*8\n"
            "    static const uint32_t multipliers[] = {1, 2, 4, 8};\n"
            "    uint32x4_t mul = vld1q_u32(multipliers);\n"
            "    uint32x4_t bits = vmulq_u32(shifted, mul);\n"
            "    return (int)vaddvq_u32(bits);  // AArch64 horizontal add\n"
            "}\n"
            "\n"
            "// Alternative (fewer instructions):\n"
            "static inline int vmovemask_f32_v2(float32x4_t v) {\n"
            "    static const int32_t shift_arr[] = {0, 1, 2, 3};\n"
            "    uint32x4_t mask = vshrq_n_u32(vreinterpretq_u32_f32(v), 31);\n"
            "    int32x4_t shift = vld1q_s32(shift_arr);\n"
            "    uint32x4_t shifted = vshlq_u32(mask, shift);\n"
            "    return (int)vaddvq_u32(shifted);\n"
            "}"
        ),
    },
    # === SSE2 integer (128-bit, 4xi32 or 16xi8) ===
    "_mm_add_epi32": {
        "x86_intrinsic": "_mm_add_epi32",
        "x86_instruction": "PADDD",
        "x86_extension": "SSE2",
        "neon_intrinsic": "vaddq_s32",
        "neon_instruction": "ADD Vd.4S, Vn.4S, Vm.4S",
        "sve_intrinsic": "svadd_s32_x(pg, a, b)",
        "sve_instruction": "ADD Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "4x int32 (128-bit)",
        "notes": "Direct 1:1 translation. Both operate on 4 packed 32-bit integers.",
        "porting_tips": "Drop-in replacement. No semantic difference.",
        "gotchas": "",
    },
    "_mm_sub_epi32": {
        "x86_intrinsic": "_mm_sub_epi32",
        "x86_instruction": "PSUBD",
        "x86_extension": "SSE2",
        "neon_intrinsic": "vsubq_s32",
        "neon_instruction": "SUB Vd.4S, Vn.4S, Vm.4S",
        "sve_intrinsic": "svsub_s32_x(pg, a, b)",
        "sve_instruction": "SUB Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "4x int32 (128-bit)",
        "notes": "Direct 1:1 translation. Both operate on 4 packed 32-bit integers.",
        "porting_tips": "Drop-in replacement. No semantic difference.",
        "gotchas": "",
    },
    "_mm_mullo_epi32": {
        "x86_intrinsic": "_mm_mullo_epi32",
        "x86_instruction": "PMULLD",
        "x86_extension": "SSE4.1",
        "neon_intrinsic": "vmulq_s32",
        "neon_instruction": "MUL Vd.4S, Vn.4S, Vm.4S",
        "sve_intrinsic": "svmul_s32_x(pg, a, b)",
        "sve_instruction": "MUL Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "4x int32 (128-bit)",
        "notes": "Low-half 32-bit multiply. Direct 1:1 translation.",
        "porting_tips": "Drop-in replacement. Both return the low 32 bits of each 32x32 product.",
        "gotchas": "",
    },
    "_mm_and_si128": {
        "x86_intrinsic": "_mm_and_si128",
        "x86_instruction": "PAND",
        "x86_extension": "SSE2",
        "neon_intrinsic": "vandq_u32",
        "neon_instruction": "AND Vd.16B, Vn.16B, Vm.16B",
        "sve_intrinsic": "svand_u32_x(pg, a, b)",
        "sve_instruction": "AND Zd.D, Zn.D, Zm.D",
        "data_type": "128-bit integer, bitwise",
        "notes": "Bitwise AND on 128-bit integer vectors. Direct equivalent.",
        "porting_tips": "Drop-in replacement. Both operate on the full 128-bit register.",
        "gotchas": "",
    },
    "_mm_or_si128": {
        "x86_intrinsic": "_mm_or_si128",
        "x86_instruction": "POR",
        "x86_extension": "SSE2",
        "neon_intrinsic": "vorrq_u32",
        "neon_instruction": "ORR Vd.16B, Vn.16B, Vm.16B",
        "sve_intrinsic": "svor_u32_x(pg, a, b)",
        "sve_instruction": "ORR Zd.D, Zn.D, Zm.D",
        "data_type": "128-bit integer, bitwise",
        "notes": "Bitwise OR on 128-bit integer vectors. Direct equivalent.",
        "porting_tips": "Drop-in replacement.",
        "gotchas": "",
    },
    "_mm_slli_epi32": {
        "x86_intrinsic": "_mm_slli_epi32",
        "x86_instruction": "PSLLD",
        "x86_extension": "SSE2",
        "neon_intrinsic": "vshlq_n_s32",
        "neon_instruction": "SHL Vd.4S, Vn.4S, #imm",
        "sve_intrinsic": "svlsl_n_s32_x(pg, a, imm)",
        "sve_instruction": "LSL Zd.S, Pg/M, Zn.S, #imm",
        "data_type": "4x int32 (128-bit)",
        "notes": "Left shift each 32-bit lane by immediate. Direct equivalent.",
        "porting_tips": "Drop-in replacement. Shift amount must be an immediate (compile-time constant).",
        "gotchas": "NEON vshlq_n_s32 requires an immediate shift amount (0-31). For variable shifts, use vshlq_s32 with a shift vector.",
    },
    "_mm_srli_epi32": {
        "x86_intrinsic": "_mm_srli_epi32",
        "x86_instruction": "PSRLD",
        "x86_extension": "SSE2",
        "neon_intrinsic": "vshrq_n_u32",
        "neon_instruction": "USHR Vd.4S, Vn.4S, #imm",
        "sve_intrinsic": "svlsr_n_u32_x(pg, a, imm)",
        "sve_instruction": "LSR Zd.S, Pg/M, Zn.S, #imm",
        "data_type": "4x uint32 (128-bit)",
        "notes": "Logical (unsigned) right shift each 32-bit lane by immediate.",
        "porting_tips": "Drop-in replacement. Use unsigned type (uint32x4_t) to get logical (not arithmetic) shift.",
        "gotchas": "NEON vshrq_n_u32 shift amount must be 1-32 (not 0). For shift by 0, use the input directly.",
    },
    "_mm_cmpeq_epi32": {
        "x86_intrinsic": "_mm_cmpeq_epi32",
        "x86_instruction": "PCMPEQD",
        "x86_extension": "SSE2",
        "neon_intrinsic": "vceqq_s32",
        "neon_instruction": "CMEQ Vd.4S, Vn.4S, Vm.4S",
        "sve_intrinsic": "svcmpeq_s32(pg, a, b)",
        "sve_instruction": "CMPEQ Pd.S, Pg/Z, Zn.S, Zm.S",
        "data_type": "4x int32 (128-bit)",
        "notes": "Compare for equality. SSE and NEON both return all-ones / all-zeros per lane. SVE returns a predicate.",
        "porting_tips": "Drop-in replacement for NEON. SVE returns svbool_t which may require code restructuring.",
        "gotchas": "",
    },
    "_mm_set1_epi32": {
        "x86_intrinsic": "_mm_set1_epi32",
        "x86_instruction": "MOVD+PSHUFD (compiler-dependent)",
        "x86_extension": "SSE2",
        "neon_intrinsic": "vdupq_n_s32",
        "neon_instruction": "DUP Vd.4S, Wn",
        "sve_intrinsic": "svdup_s32(a)",
        "sve_instruction": "DUP Zd.S, Wn",
        "data_type": "4x int32 (128-bit)",
        "notes": "Broadcast a single int32 to all lanes. Direct equivalent.",
        "porting_tips": "Drop-in replacement.",
        "gotchas": "",
    },
    # === AVX (256-bit, 8xf32) ===
    "_mm256_add_ps": {
        "x86_intrinsic": "_mm256_add_ps",
        "x86_instruction": "VADDPS (ymm)",
        "x86_extension": "AVX",
        "neon_intrinsic": "2x vaddq_f32 (split into high/low)",
        "neon_instruction": "FADD Vd.4S, Vn.4S, Vm.4S (x2)",
        "sve_intrinsic": "svadd_f32_x(pg, a, b)",
        "sve_instruction": "FADD Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "8x float32 (256-bit)",
        "notes": "NEON is 128-bit, so 256-bit AVX ops require two NEON operations on high and low halves. SVE with VL>=256 can handle this in one instruction.",
        "porting_tips": "Split into two 128-bit operations: result_lo = vaddq_f32(a_lo, b_lo); result_hi = vaddq_f32(a_hi, b_hi);",
        "gotchas": "NEON requires splitting 256-bit data into two 128-bit halves. Data layout must be managed carefully.",
    },
    "_mm256_mul_ps": {
        "x86_intrinsic": "_mm256_mul_ps",
        "x86_instruction": "VMULPS (ymm)",
        "x86_extension": "AVX",
        "neon_intrinsic": "2x vmulq_f32 (split into high/low)",
        "neon_instruction": "FMUL Vd.4S, Vn.4S, Vm.4S (x2)",
        "sve_intrinsic": "svmul_f32_x(pg, a, b)",
        "sve_instruction": "FMUL Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "8x float32 (256-bit)",
        "notes": "NEON is 128-bit, so split into two 128-bit multiply operations. SVE with VL>=256 handles this natively.",
        "porting_tips": "Split into two 128-bit operations: result_lo = vmulq_f32(a_lo, b_lo); result_hi = vmulq_f32(a_hi, b_hi);",
        "gotchas": "NEON requires splitting 256-bit data into two 128-bit halves.",
    },
    "_mm256_load_ps": {
        "x86_intrinsic": "_mm256_load_ps",
        "x86_instruction": "VMOVAPS (ymm)",
        "x86_extension": "AVX",
        "neon_intrinsic": "2x vld1q_f32 (sequential loads)",
        "neon_instruction": "LD1 {Vt.4S}, [Xn] (x2, with offset)",
        "sve_intrinsic": "svld1_f32(pg, ptr)",
        "sve_instruction": "LD1W Zt.S, Pg/Z, [Xn]",
        "data_type": "8x float32 (256-bit)",
        "notes": "Load 256 bits (8 floats). NEON needs two 128-bit loads. SVE with VL>=256 does one load.",
        "porting_tips": "float32x4_t lo = vld1q_f32(ptr); float32x4_t hi = vld1q_f32(ptr + 4);",
        "gotchas": "Ensure pointer arithmetic accounts for the split: second load at ptr+4 (not ptr+16 bytes, since pointer is float*).",
    },
    "_mm256_store_ps": {
        "x86_intrinsic": "_mm256_store_ps",
        "x86_instruction": "VMOVAPS (ymm)",
        "x86_extension": "AVX",
        "neon_intrinsic": "2x vst1q_f32 (sequential stores)",
        "neon_instruction": "ST1 {Vt.4S}, [Xn] (x2, with offset)",
        "sve_intrinsic": "svst1_f32(pg, ptr, data)",
        "sve_instruction": "ST1W Zt.S, Pg, [Xn]",
        "data_type": "8x float32 (256-bit)",
        "notes": "Store 256 bits (8 floats). NEON needs two 128-bit stores. SVE with VL>=256 does one store.",
        "porting_tips": "vst1q_f32(ptr, lo); vst1q_f32(ptr + 4, hi);",
        "gotchas": "",
    },
    "_mm256_set1_ps": {
        "x86_intrinsic": "_mm256_set1_ps",
        "x86_instruction": "VBROADCASTSS (ymm)",
        "x86_extension": "AVX",
        "neon_intrinsic": "2x vdupq_n_f32",
        "neon_instruction": "DUP Vd.4S, Rn (x2)",
        "sve_intrinsic": "svdup_f32(a)",
        "sve_instruction": "DUP Zd.S, Rn",
        "data_type": "8x float32 (256-bit)",
        "notes": "Broadcast scalar to all 8 lanes. On NEON, broadcast into two 128-bit registers.",
        "porting_tips": "float32x4_t v = vdupq_n_f32(val); — use the same register for both halves.",
        "gotchas": "",
    },
    "_mm256_fmadd_ps": {
        "x86_intrinsic": "_mm256_fmadd_ps",
        "x86_instruction": "VFMADD132PS / VFMADD213PS / VFMADD231PS (ymm)",
        "x86_extension": "FMA (AVX2)",
        "neon_intrinsic": "2x vfmaq_f32 (split into high/low)",
        "neon_instruction": "FMLA Vd.4S, Vn.4S, Vm.4S (x2)",
        "sve_intrinsic": "svmla_f32_x(pg, acc, a, b)",
        "sve_instruction": "FMLA Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "8x float32 (256-bit)",
        "notes": "Fused multiply-add: a*b+c. NEON vfmaq_f32 computes acc + a*b (note operand order difference). SVE svmla computes acc + a*b.",
        "porting_tips": "_mm256_fmadd_ps(a,b,c) = a*b+c. vfmaq_f32(c, a, b) = c + a*b. Same result, different argument order.",
        "gotchas": "Operand order differs! x86 FMA: fmadd(a,b,c) = a*b+c. NEON FMA: vfmaq_f32(acc,a,b) = acc+a*b. The accumulator is the FIRST argument in NEON.",
    },
    # === AVX-512 (512-bit, 16xf32) ===
    "_mm512_add_ps": {
        "x86_intrinsic": "_mm512_add_ps",
        "x86_instruction": "VADDPS (zmm)",
        "x86_extension": "AVX-512F",
        "neon_intrinsic": "4x vaddq_f32 (split into 4 quarters)",
        "neon_instruction": "FADD Vd.4S, Vn.4S, Vm.4S (x4)",
        "sve_intrinsic": "svadd_f32_x(pg, a, b)",
        "sve_instruction": "FADD Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "16x float32 (512-bit)",
        "notes": "NEON is 128-bit, so 512-bit AVX-512 ops require four NEON operations. SVE with VL>=512 can handle this in one instruction.",
        "porting_tips": "Split into four 128-bit operations on quarters. SVE is a much better target for AVX-512 code.",
        "gotchas": "Four-way split on NEON is cumbersome. Consider SVE or restructuring the algorithm for 128-bit vectors.",
    },
    "_mm512_mask_add_ps": {
        "x86_intrinsic": "_mm512_mask_add_ps",
        "x86_instruction": "VADDPS (zmm) {k}",
        "x86_extension": "AVX-512F",
        "neon_intrinsic": "(manual masking with vbslq_f32)",
        "neon_instruction": "BSL Vd.16B, Vn.16B, Vm.16B + FADD (multi-instruction)",
        "sve_intrinsic": "svadd_f32_m(pg, a, b)",
        "sve_instruction": "FADD Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "16x float32 (512-bit), masked",
        "notes": "Masked add: only lanes where mask bit is set are updated. SVE predication (_m suffix) is a natural fit. NEON requires manual masking with vbslq_f32.",
        "porting_tips": "SVE: use _m (merging) predication. NEON: compute add, then blend: vbslq_f32(mask, add_result, passthrough).",
        "gotchas": "NEON masking is more verbose. SVE predicated operations are the natural equivalent of AVX-512 masking.",
        "workaround": (
            "// _mm512_mask_add_ps replacement for NEON (per 128-bit quarter)\n"
            "// mask_add(src, k, a, b): result[i] = k[i] ? a[i]+b[i] : src[i]\n"
            "float32x4_t neon_mask_add(float32x4_t src, uint32x4_t mask,\n"
            "                          float32x4_t a, float32x4_t b) {\n"
            "    float32x4_t sum = vaddq_f32(a, b);\n"
            "    return vbslq_f32(mask, sum, src);  // select sum where mask, else src\n"
            "}"
        ),
    },
    "_mm512_fmadd_ps": {
        "x86_intrinsic": "_mm512_fmadd_ps",
        "x86_instruction": "VFMADD132PS / VFMADD213PS / VFMADD231PS (zmm)",
        "x86_extension": "AVX-512F",
        "neon_intrinsic": "4x vfmaq_f32 (split into 4 quarters)",
        "neon_instruction": "FMLA Vd.4S, Vn.4S, Vm.4S (x4)",
        "sve_intrinsic": "svmla_f32_x(pg, acc, a, b)",
        "sve_instruction": "FMLA Zd.S, Pg/M, Zn.S, Zm.S",
        "data_type": "16x float32 (512-bit)",
        "notes": "Fused multiply-add on 512 bits. Same operand order caveat as the 256-bit version.",
        "porting_tips": "Split into four 128-bit FMA operations on NEON. SVE with VL>=512 handles natively.",
        "gotchas": "Operand order differs! x86 FMA: fmadd(a,b,c) = a*b+c. NEON FMA: vfmaq_f32(acc,a,b) = acc+a*b.",
    },
}

# Reverse-lookup index: ARM intrinsic name → x86 intrinsic key
_ARM_TO_X86_INDEX: dict[str, str] = {}
for _x86_key, _entry in _INTRINSIC_TRANSLATIONS.items():
    # Index NEON intrinsic names (strip parenthetical notes and whitespace)
    _neon = _entry["neon_intrinsic"]
    # Handle entries like "2x vaddq_f32 (split into high/low)" or "vandq_u32 (reinterpret)"
    for _token in _neon.replace(",", " ").split():
        _clean = _token.strip("()")
        if _clean.startswith("v") and len(_clean) > 2 and _clean[1:2].isalpha():
            _ARM_TO_X86_INDEX[_clean.lower()] = _x86_key
    # Index SVE intrinsic names (take function name before parenthesis)
    _sve = _entry["sve_intrinsic"]
    _sve_name = _sve.split("(")[0].strip()
    if _sve_name.startswith("sv"):
        _ARM_TO_X86_INDEX[_sve_name.lower()] = _x86_key

# Clean up loop variables from module scope
del _x86_key, _entry, _neon, _token, _clean, _sve, _sve_name


def _format_intrinsic_translation(entry: dict, target_arch: str) -> str:
    """Format a single intrinsic translation entry."""
    x86 = entry["x86_intrinsic"]
    ext = entry["x86_extension"]

    if target_arch in ("neon", "sve"):
        neon_intr = entry["neon_intrinsic"]
        # Build the header
        header = f"# Intrinsic Translation: {x86} → {neon_intr}"
        if target_arch == "sve":
            sve_name = entry["sve_intrinsic"].split("(")[0].strip()
            header = f"# Intrinsic Translation: {x86} → {sve_name}"
    else:
        # x86 target (reverse direction)
        # Determine which ARM intrinsic was the source
        header = f"# Intrinsic Translation: (ARM) → {x86}"

    lines = [header, ""]

    # x86 section
    lines.append(f"## x86 ({ext})")
    lines.append(f"**Intrinsic:** `{x86}`")
    lines.append(f"**Instruction:** {entry['x86_instruction']}")
    lines.append("")

    # NEON section
    lines.append("## ARM NEON Equivalent")
    lines.append(f"**Intrinsic:** `{entry['neon_intrinsic']}`")
    lines.append(f"**Instruction:** {entry['neon_instruction']}")
    lines.append("")

    # SVE section
    lines.append("## ARM SVE Equivalent")
    lines.append(f"**Intrinsic:** `{entry['sve_intrinsic']}`")
    lines.append(f"**Instruction:** {entry['sve_instruction']}")
    lines.append("")

    # Data type
    lines.append("## Data Type")
    lines.append(entry["data_type"])
    lines.append("")

    # Porting notes
    lines.append("## Porting Notes")
    lines.append(entry["porting_tips"])
    lines.append("")

    # Gotchas
    lines.append("## Gotchas")
    gotchas = entry.get("gotchas", "")
    lines.append(gotchas if gotchas else "None — direct translation.")
    lines.append("")

    # Workaround (if present)
    workaround = entry.get("workaround", "")
    if workaround:
        lines.append("## Workaround Code")
        lines.append("```c")
        lines.append(workaround)
        lines.append("```")
        lines.append("")

    # Extra notes
    if entry.get("notes"):
        lines.append("## Additional Notes")
        lines.append(entry["notes"])

    return "\n".join(lines)


def _format_intrinsic_overview(target_arch: str) -> str:
    """Format an overview table of all intrinsic translations."""
    # Group by x86 extension
    groups: dict[str, list[dict]] = {}
    for entry in _INTRINSIC_TRANSLATIONS.values():
        ext = entry["x86_extension"]
        groups.setdefault(ext, []).append(entry)

    lines = ["# SIMD Intrinsic Translation Overview", ""]
    if target_arch in ("neon", "sve"):
        lines.append(f"x86 → ARM {target_arch.upper()} translation table")
    else:
        lines.append(f"ARM → x86 translation table")
    lines.append("")

    for ext in ("SSE", "SSE2", "SSE4.1", "AVX", "FMA (AVX2)", "AVX-512F"):
        if ext not in groups:
            continue
        entries = groups[ext]
        lines.append(f"## {ext}")
        lines.append("")
        if target_arch == "neon":
            lines.append(f"| x86 Intrinsic | NEON Equivalent | Data Type |")
            lines.append(f"|---|---|---|")
            for e in entries:
                lines.append(f"| `{e['x86_intrinsic']}` | `{e['neon_intrinsic']}` | {e['data_type']} |")
        elif target_arch == "sve":
            lines.append(f"| x86 Intrinsic | SVE Equivalent | Data Type |")
            lines.append(f"|---|---|---|")
            for e in entries:
                sve_name = e["sve_intrinsic"].split("(")[0].strip()
                lines.append(f"| `{e['x86_intrinsic']}` | `{sve_name}` | {e['data_type']} |")
        else:
            lines.append(f"| x86 Intrinsic | NEON Equivalent | SVE Equivalent |")
            lines.append(f"|---|---|---|")
            for e in entries:
                sve_name = e["sve_intrinsic"].split("(")[0].strip()
                lines.append(f"| `{e['x86_intrinsic']}` | `{e['neon_intrinsic']}` | `{sve_name}` |")
        lines.append("")

    lines.append(f"**Total translations:** {len(_INTRINSIC_TRANSLATIONS)}")
    return "\n".join(lines)


def _find_closest_intrinsics(name: str, limit: int = 5) -> list[str]:
    """Find intrinsic names that are close to the given name."""
    name_lower = name.lower()
    all_names = list(_INTRINSIC_TRANSLATIONS.keys()) + list(_ARM_TO_X86_INDEX.keys())
    # Simple substring matching for suggestions
    matches = []
    for candidate in all_names:
        if name_lower in candidate.lower() or candidate.lower() in name_lower:
            matches.append(candidate)
    # Also try prefix matching
    if not matches:
        for candidate in all_names:
            # Check if they share a common prefix of length >= 6
            c_lower = candidate.lower()
            prefix_len = 0
            for a, b in zip(name_lower, c_lower):
                if a == b:
                    prefix_len += 1
                else:
                    break
            if prefix_len >= 6:
                matches.append(candidate)
    # Deduplicate and limit
    seen = set()
    unique = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique[:limit]


@mcp.tool()
def translate_intrinsic(intrinsic: str, from_arch: str, to_arch: str) -> str:
    """Translate SIMD intrinsics between x86 (SSE/AVX) and ARM (NEON/SVE).

    Invaluable for porting code between x86 and ARM architectures. Covers
    SSE, SSE2, AVX, AVX-512 on the x86 side and NEON, SVE on the ARM side.

    Args:
        intrinsic: The intrinsic function name (e.g. "_mm_add_ps", "vaddq_f32"),
            or "list"/"overview" to see all available translations.
        from_arch: Source architecture. One of:
            "x86", "sse", "avx", "avx2", "avx512" — x86 SIMD
            "neon" — ARM NEON
            "sve" — ARM SVE
        to_arch: Target architecture. One of:
            "neon" — ARM NEON
            "sve" — ARM SVE
            "sse", "avx", "x86" — x86 SIMD
    """
    valid_from = {"x86", "sse", "avx", "avx2", "avx512", "neon", "sve"}
    valid_to = {"neon", "sve", "sse", "avx", "x86"}

    from_key = from_arch.strip().lower()
    to_key = to_arch.strip().lower()

    if from_key not in valid_from:
        return (
            f"Error: '{from_arch}' is not a valid source architecture.\n\n"
            f"Valid values: {', '.join(sorted(valid_from))}"
        )

    if to_key not in valid_to:
        return (
            f"Error: '{to_arch}' is not a valid target architecture.\n\n"
            f"Valid values: {', '.join(sorted(valid_to))}"
        )

    name = intrinsic.strip()
    name_lower = name.lower()

    # Handle overview / list request
    if name_lower in ("list", "overview"):
        return _format_intrinsic_overview(to_key)

    # Normalize from_arch to determine lookup direction
    x86_archs = {"x86", "sse", "avx", "avx2", "avx512"}
    arm_archs = {"neon", "sve"}

    if from_key in x86_archs:
        # x86 → ARM lookup
        entry = _INTRINSIC_TRANSLATIONS.get(name_lower)
        if entry is None:
            # Try case-insensitive search
            for key, val in _INTRINSIC_TRANSLATIONS.items():
                if key.lower() == name_lower:
                    entry = val
                    break

        if entry is None:
            suggestions = _find_closest_intrinsics(name)
            msg = f"Error: intrinsic '{name}' not found in translation database.\n\n"
            if suggestions:
                msg += "Did you mean one of these?\n"
                for s in suggestions:
                    msg += f"  - {s}\n"
            else:
                msg += "Use translate_intrinsic('list', ...) to see all available translations.\n"
            return msg

        return _format_intrinsic_translation(entry, to_key)

    elif from_key in arm_archs:
        # ARM → x86 lookup (reverse direction)
        x86_key = _ARM_TO_X86_INDEX.get(name_lower)
        if x86_key is None:
            # Try case-insensitive search in the index
            for arm_name, x86_k in _ARM_TO_X86_INDEX.items():
                if arm_name.lower() == name_lower:
                    x86_key = x86_k
                    break

        if x86_key is None:
            suggestions = _find_closest_intrinsics(name)
            msg = f"Error: ARM intrinsic '{name}' not found in translation database.\n\n"
            if suggestions:
                msg += "Did you mean one of these?\n"
                for s in suggestions:
                    msg += f"  - {s}\n"
            else:
                msg += "Use translate_intrinsic('list', ...) to see all available translations.\n"
            return msg

        entry = _INTRINSIC_TRANSLATIONS[x86_key]
        # For reverse direction, target is x86
        return _format_intrinsic_translation(entry, to_key)

    return f"Error: unsupported translation direction from '{from_arch}' to '{to_arch}'."

def main():
    mcp.run()


if __name__ == "__main__":
    main()
