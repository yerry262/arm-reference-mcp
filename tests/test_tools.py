"""Smoke tests for all ARM Reference MCP tools.

Run with: python -m pytest tests/test_tools.py -v
Or simply: python tests/test_tools.py
"""

import sys
import os

# Ensure the src directory is on the path for editable installs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arm_reference_mcp.server import (
    lookup_register,
    list_registers,
    decode_instruction,
    explain_condition_code,
    explain_calling_convention,
    search_registers,
    decode_register_value,
    explain_exception_levels,
    explain_security_model,
    lookup_core,
    compare_cores,
    explain_page_table_format,
    explain_memory_attributes,
    explain_extension,
    compare_architecture_versions,
    show_assembly_pattern,
    explain_barrier,
)


# ---- Helpers ----

def assert_contains(result: str, *keywords: str):
    """Assert that result contains all keywords (case-insensitive)."""
    lower = result.lower()
    for kw in keywords:
        assert kw.lower() in lower, f"Expected '{kw}' in output, but not found.\nOutput (first 300 chars): {result[:300]}"


def assert_no_error(result: str):
    """Assert result doesn't start with 'Error:'."""
    assert not result.startswith("Error:"), f"Got error: {result}"


# ---- Tool 1: lookup_register ----

class TestLookupRegister:
    def test_cpsr(self):
        r = lookup_register("CPSR")
        assert_no_error(r)
        assert_contains(r, "CPSR", "aarch32", "32-bit", "Negative flag")

    def test_x0(self):
        r = lookup_register("X0")
        assert_no_error(r)
        assert_contains(r, "X0", "aarch64", "64-bit")

    def test_sp_both_architectures(self):
        r = lookup_register("SP")
        assert_no_error(r)
        assert_contains(r, "aarch32", "aarch64", "Stack Pointer")

    def test_sp_filtered(self):
        r = lookup_register("SP", architecture="aarch64")
        assert_no_error(r)
        assert "aarch32" not in r

    def test_alias_lr(self):
        r = lookup_register("LR", architecture="aarch64")
        assert_no_error(r)
        assert_contains(r, "X30", "Link Register")

    def test_alias_fp(self):
        r = lookup_register("FP", architecture="aarch32")
        assert_no_error(r)
        assert_contains(r, "R11")

    def test_case_insensitive(self):
        r = lookup_register("cpsr")
        assert_no_error(r)
        assert_contains(r, "CPSR")

    def test_not_found(self):
        r = lookup_register("NONEXISTENT")
        assert "No register found" in r

    def test_invalid_architecture(self):
        r = lookup_register("X0", architecture="mips")
        assert "Error" in r


# ---- Tool 2: list_registers ----

class TestListRegisters:
    def test_aarch64_all(self):
        r = list_registers("aarch64")
        assert_no_error(r)
        assert_contains(r, "X0", "SP", "NZCV")

    def test_aarch32_general_purpose(self):
        r = list_registers("aarch32", category="general_purpose")
        assert_no_error(r)
        assert_contains(r, "R0", "R15")

    def test_aarch64_status(self):
        r = list_registers("aarch64", category="status")
        assert_no_error(r)
        assert_contains(r, "NZCV", "DAIF")

    def test_invalid_arch(self):
        r = list_registers("mips")
        assert "Error" in r

    def test_invalid_category(self):
        r = list_registers("aarch64", category="fake")
        assert "Error" in r


# ---- Tool 3: decode_instruction ----

class TestDecodeInstruction:
    def test_mov_r1_5(self):
        # MOV R1, #5 = 0xE3A01005
        r = decode_instruction("0xE3A01005")
        assert_no_error(r)
        assert_contains(r, "MOV", "Data Processing", "AL")

    def test_branch(self):
        # B <offset> = 0xEA000000
        r = decode_instruction("0xEA000000")
        assert_no_error(r)
        assert_contains(r, "Branch")

    def test_ldr(self):
        # LDR R0, [R1] = 0xE5910000
        r = decode_instruction("0xE5910000")
        assert_no_error(r)
        assert_contains(r, "Load/Store", "LDR")

    def test_without_0x_prefix(self):
        r = decode_instruction("E3A01005")
        assert_no_error(r)
        assert_contains(r, "MOV")

    def test_invalid_hex(self):
        r = decode_instruction("ZZZZZZZZ")
        assert "Error" in r

    def test_conditional(self):
        # BEQ = 0x0A000000
        r = decode_instruction("0x0A000000")
        assert_no_error(r)
        assert_contains(r, "EQ", "Branch")


# ---- Tool 4: explain_condition_code ----

class TestExplainConditionCode:
    def test_eq(self):
        r = explain_condition_code("EQ")
        assert_no_error(r)
        assert_contains(r, "Equal", "Z == 1", "NE")

    def test_gt(self):
        r = explain_condition_code("GT")
        assert_no_error(r)
        assert_contains(r, "Greater Than", "LE")

    def test_al(self):
        r = explain_condition_code("AL")
        assert_no_error(r)
        assert_contains(r, "Always")

    def test_compound_cs_hs(self):
        r = explain_condition_code("CS")
        assert_no_error(r)
        assert_contains(r, "Carry Set")

    def test_case_insensitive(self):
        r = explain_condition_code("ne")
        assert_no_error(r)
        assert_contains(r, "Not Equal")

    def test_invalid(self):
        r = explain_condition_code("XX")
        assert "Error" in r


# ---- Tool 5: explain_calling_convention ----

class TestExplainCallingConvention:
    def test_aarch64(self):
        r = explain_calling_convention("aarch64")
        assert_no_error(r)
        assert_contains(r, "AAPCS64", "X0", "X7", "Callee-Saved", "16 bytes")

    def test_aarch32(self):
        r = explain_calling_convention("aarch32")
        assert_no_error(r)
        assert_contains(r, "AAPCS32", "R0", "R3", "8 bytes")

    def test_invalid(self):
        r = explain_calling_convention("x86")
        assert "Error" in r


# ---- Tool 6: search_registers ----

class TestSearchRegisters:
    def test_stack(self):
        r = search_registers("stack")
        assert_no_error(r)
        assert_contains(r, "SP", "Stack Pointer")

    def test_cache(self):
        r = search_registers("cache")
        assert_no_error(r)
        assert_contains(r, "SCTLR_EL1")

    def test_filtered_arch(self):
        r = search_registers("stack", architecture="aarch32")
        assert_no_error(r)
        assert "aarch64" not in r or "aarch32" in r  # should have aarch32 results

    def test_no_results(self):
        r = search_registers("quantum_entanglement_register")
        assert "No registers found" in r


# ---- Tool 7: decode_register_value ----

class TestDecodeRegisterValue:
    def test_cpsr(self):
        r = decode_register_value("CPSR", "0x600001D3")
        assert_no_error(r)
        assert_contains(r, "CPSR", "Bit Field Decode", "Negative", "Zero")

    def test_cpsr_svc_mode(self):
        r = decode_register_value("CPSR", "0x600001D3")
        # Mode bits [4:0] = 0x13 = SVC
        assert_contains(r, "10011")  # binary for mode 0x13

    def test_sctlr(self):
        r = decode_register_value("SCTLR_EL1", "0x30D00805")
        assert_no_error(r)
        assert_contains(r, "SCTLR_EL1", "MMU")

    def test_not_found(self):
        r = decode_register_value("FAKE_REG", "0x0")
        assert "No register found" in r

    def test_invalid_hex(self):
        r = decode_register_value("CPSR", "not_hex")
        assert "Error" in r


# ---- Tool 8: explain_exception_levels ----

class TestExplainExceptionLevels:
    def test_aarch64(self):
        r = explain_exception_levels("aarch64")
        assert_no_error(r)
        assert_contains(r, "EL0", "EL1", "EL2", "EL3", "ERET", "Vector Table")

    def test_aarch32(self):
        r = explain_exception_levels("aarch32")
        assert_no_error(r)
        assert_contains(r, "User", "SVC", "FIQ", "IRQ", "Monitor", "Hyp")

    def test_default_is_aarch64(self):
        r = explain_exception_levels()
        assert_contains(r, "EL0", "EL3")

    def test_invalid(self):
        r = explain_exception_levels("riscv")
        assert "Error" in r


# ---- Tool 9: explain_security_model ----

class TestExplainSecurityModel:
    def test_aarch64(self):
        r = explain_security_model("aarch64")
        assert_no_error(r)
        assert_contains(r, "TrustZone", "SCR_EL3", "Secure", "Non-secure")

    def test_aarch64_rme(self):
        r = explain_security_model("aarch64")
        assert_contains(r, "Realm", "RME", "Granule Protection")

    def test_aarch32(self):
        r = explain_security_model("aarch32")
        assert_no_error(r)
        assert_contains(r, "TrustZone", "Monitor", "SCR")

    def test_invalid(self):
        r = explain_security_model("x86")
        assert "Error" in r


# ---- Tool 10: lookup_core ----

class TestLookupCore:
    def test_cortex_a78(self):
        r = lookup_core("Cortex-A78")
        assert_no_error(r)
        assert_contains(r, "A78", "ARMv8")

    def test_short_name(self):
        r = lookup_core("A55")
        assert_no_error(r)
        assert_contains(r, "Cortex-A55")

    def test_neoverse(self):
        r = lookup_core("Neoverse-N2")
        assert_no_error(r)
        assert_contains(r, "N2")

    def test_cortex_m(self):
        r = lookup_core("M4")
        assert_no_error(r)
        assert_contains(r, "Cortex-M4")

    def test_not_found(self):
        r = lookup_core("FakeCore-Z99")
        assert "not found" in r.lower() or "error" in r.lower() or "available" in r.lower()


# ---- Tool 11: compare_cores ----

class TestCompareCores:
    def test_a55_vs_a78(self):
        r = compare_cores("A55", "A78")
        assert_no_error(r)
        assert_contains(r, "A55", "A78")

    def test_m0_vs_m7(self):
        r = compare_cores("M0", "M7")
        assert_no_error(r)
        assert_contains(r, "M0", "M7")

    def test_invalid_core(self):
        r = compare_cores("A55", "FakeCore")
        assert "not found" in r.lower() or "error" in r.lower()


# ---- Tool 12: explain_page_table_format ----

class TestExplainPageTableFormat:
    def test_4kb_48bit(self):
        r = explain_page_table_format("4KB", 48)
        assert_no_error(r)
        assert_contains(r, "4KB", "48", "L0", "L3")

    def test_64kb(self):
        r = explain_page_table_format("64KB")
        assert_no_error(r)
        assert_contains(r, "64KB")

    def test_16kb(self):
        r = explain_page_table_format("16KB", 48)
        assert_no_error(r)
        assert_contains(r, "16KB")

    def test_invalid_granule(self):
        r = explain_page_table_format("8KB")
        assert "Error" in r or "error" in r.lower()


# ---- Tool 13: explain_memory_attributes ----

class TestExplainMemoryAttributes:
    def test_overview(self):
        r = explain_memory_attributes()
        assert_no_error(r)
        assert_contains(r, "MAIR")

    def test_cacheability(self):
        r = explain_memory_attributes("cacheability")
        assert_no_error(r)
        assert_contains(r, "Write-Back")

    def test_shareability(self):
        r = explain_memory_attributes("shareability")
        assert_no_error(r)
        assert_contains(r, "Inner Shareable")

    def test_mair(self):
        r = explain_memory_attributes("mair")
        assert_no_error(r)
        assert_contains(r, "MAIR_EL1")

    def test_access_permissions(self):
        r = explain_memory_attributes("access_permissions")
        assert_no_error(r)
        assert_contains(r, "AP", "PXN")

    def test_invalid_topic(self):
        r = explain_memory_attributes("fake_topic")
        assert "Error" in r or "error" in r.lower()


# ---- Tool 14: explain_extension ----

class TestExplainExtension:
    def test_sve(self):
        r = explain_extension("SVE")
        assert_no_error(r)
        assert_contains(r, "Scalable Vector")

    def test_mte(self):
        r = explain_extension("MTE")
        assert_no_error(r)
        assert_contains(r, "Memory Tagging")

    def test_pac(self):
        r = explain_extension("PAC")
        assert_no_error(r)
        assert_contains(r, "Pointer Authentication")

    def test_bti(self):
        r = explain_extension("BTI")
        assert_no_error(r)
        assert_contains(r, "Branch Target")

    def test_case_insensitive(self):
        r = explain_extension("sve")
        assert_no_error(r)
        assert_contains(r, "Scalable Vector")

    def test_not_found(self):
        r = explain_extension("FAKE_EXT")
        assert "not found" in r.lower() or "error" in r.lower() or "available" in r.lower()


# ---- Tool 15: compare_architecture_versions ----

class TestCompareArchitectureVersions:
    def test_single_version(self):
        r = compare_architecture_versions("armv8.0")
        assert_no_error(r)
        assert_contains(r, "armv8.0", "feature")

    def test_compare_two(self):
        r = compare_architecture_versions("armv8.0", compare_to="armv9.0")
        assert_no_error(r)
        assert_contains(r, "armv8.0", "armv9.0")

    def test_invalid_version(self):
        r = compare_architecture_versions("armv99")
        assert "not found" in r.lower() or "error" in r.lower() or "available" in r.lower()


# ---- Tool 16: show_assembly_pattern ----

class TestShowAssemblyPattern:
    def test_function_prologue(self):
        r = show_assembly_pattern("function_prologue")
        assert_no_error(r)
        assert_contains(r, "prologue")

    def test_atomic_add(self):
        r = show_assembly_pattern("atomic_add")
        assert_no_error(r)

    def test_spinlock(self):
        r = show_assembly_pattern("spinlock_acquire")
        assert_no_error(r)

    def test_aarch32(self):
        r = show_assembly_pattern("function_prologue", architecture="aarch32")
        assert_no_error(r)

    def test_not_found(self):
        r = show_assembly_pattern("fake_pattern")
        assert "not found" in r.lower() or "error" in r.lower() or "available" in r.lower()


# ---- Tool 17: explain_barrier ----

class TestExplainBarrier:
    def test_dmb(self):
        r = explain_barrier("DMB")
        assert_no_error(r)
        assert_contains(r, "Data Memory Barrier")

    def test_dsb(self):
        r = explain_barrier("DSB")
        assert_no_error(r)
        assert_contains(r, "Data Synchronization")

    def test_isb(self):
        r = explain_barrier("ISB")
        assert_no_error(r)
        assert_contains(r, "Instruction Synchronization")

    def test_overview(self):
        r = explain_barrier("overview")
        assert_no_error(r)
        assert_contains(r, "DMB", "DSB", "ISB")

    def test_ldar(self):
        r = explain_barrier("LDAR")
        assert_no_error(r)

    def test_invalid(self):
        r = explain_barrier("FAKE_BARRIER")
        assert "not found" in r.lower() or "error" in r.lower() or "available" in r.lower()


# ---- Run all tests ----

def run_all():
    """Simple test runner — no pytest dependency needed."""
    import traceback

    test_classes = [
        TestLookupRegister,
        TestListRegisters,
        TestDecodeInstruction,
        TestExplainConditionCode,
        TestExplainCallingConvention,
        TestSearchRegisters,
        TestDecodeRegisterValue,
        TestExplainExceptionLevels,
        TestExplainSecurityModel,
        TestLookupCore,
        TestCompareCores,
        TestExplainPageTableFormat,
        TestExplainMemoryAttributes,
        TestExplainExtension,
        TestCompareArchitectureVersions,
        TestShowAssemblyPattern,
        TestExplainBarrier,
    ]

    total = 0
    passed = 0
    failed = 0
    failures: list[str] = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            total += 1
            test_id = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS  {test_id}")
            except Exception as e:
                failed += 1
                tb = traceback.format_exc()
                failures.append(f"  FAIL  {test_id}\n        {e}\n")
                print(f"  FAIL  {test_id}  —  {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failures:
        print(f"\nFailures:\n")
        for f in failures:
            print(f)
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
