"""ARM Register Reference MCP Server.

Provides seven tools:
  - lookup_register:          Get detailed info on a specific ARM register.
  - list_registers:           Browse registers by architecture and category.
  - decode_instruction:       Decode a 32-bit hex value into AArch32 instruction fields.
  - explain_condition_code:   Explain an ARM condition code suffix in detail.
  - explain_calling_convention: AAPCS calling convention reference.
  - search_registers:         Keyword search across all register data.
  - decode_register_value:    Decode a hex value against a register's bit fields.
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


def main():
    mcp.run()


if __name__ == "__main__":
    main()
