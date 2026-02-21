"""Smoke tests for ARM Documentation RAG MCP tools.

Run with: python -m pytest tests/test_docs_rag.py -v
Or simply: python tests/test_docs_rag.py
"""

import sys
import os

# Ensure the src directory is on the path for editable installs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arm_reference_mcp.docs_rag_server import (
    search_arm_docs,
    explain_arm_concept,
    find_register_in_manual,
    get_errata,
    compare_manual_sections,
    list_arm_documents,
    explain_instruction_encoding,
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


# ---- Tool 1: search_arm_docs ----

class TestSearchArmDocs:
    def test_search_page_tables(self):
        r = search_arm_docs("page tables")
        assert_no_error(r)
        assert_contains(r, "page table", "DDI0487")

    def test_search_neon(self):
        r = search_arm_docs("NEON")
        assert_no_error(r)
        assert_contains(r, "neon", "simd")

    def test_search_trustzone(self):
        r = search_arm_docs("TrustZone")
        assert_no_error(r)
        assert_contains(r, "trustzone", "secure")

    def test_search_with_scope_filter(self):
        r = search_arm_docs("cache", doc_scope="programming_guide")
        assert_no_error(r)
        assert_contains(r, "cache", "programming_guide")

    def test_search_with_arch_filter(self):
        r = search_arm_docs("exception", architecture="aarch64")
        assert_no_error(r)
        assert_contains(r, "exception")
        # Should not include aarch32-only results
        assert "DDI0406" not in r

    def test_search_no_results(self):
        r = search_arm_docs("quantum_entanglement_protocol")
        assert "no documentation found" in r.lower()

    def test_search_invalid_scope(self):
        r = search_arm_docs("cache", doc_scope="invalid_scope")
        assert "Error" in r

    def test_search_sve(self):
        r = search_arm_docs("SVE predicate gather")
        assert_no_error(r)
        assert_contains(r, "sve")

    def test_search_gic(self):
        r = search_arm_docs("GIC distributor interrupt")
        assert_no_error(r)
        assert_contains(r, "gic", "distributor")

    def test_search_memory_ordering(self):
        r = search_arm_docs("memory ordering")
        assert_no_error(r)
        assert_contains(r, "memory", "ordering")


# ---- Tool 2: explain_arm_concept ----

class TestExplainArmConcept:
    def test_exception_levels(self):
        r = explain_arm_concept("exception_levels")
        assert_no_error(r)
        assert_contains(r, "EL0", "EL1", "EL2", "EL3", "privilege")

    def test_trustzone(self):
        r = explain_arm_concept("trustzone")
        assert_no_error(r)
        assert_contains(r, "secure", "non-secure", "SMC")

    def test_vmsa(self):
        r = explain_arm_concept("vmsa")
        assert_no_error(r)
        assert_contains(r, "translation", "page", "TTBR")

    def test_memory_ordering(self):
        r = explain_arm_concept("memory_ordering")
        assert_no_error(r)
        assert_contains(r, "DMB", "DSB", "ISB", "barrier")

    def test_cache_coherency(self):
        r = explain_arm_concept("cache_coherency")
        assert_no_error(r)
        assert_contains(r, "coherency", "MOESI", "PoC")

    def test_sve(self):
        r = explain_arm_concept("sve")
        assert_no_error(r)
        assert_contains(r, "scalable", "vector", "predicate", "Z0")

    def test_mte(self):
        r = explain_arm_concept("mte")
        assert_no_error(r)
        assert_contains(r, "memory tagging", "tag", "16 bytes")

    def test_pac(self):
        r = explain_arm_concept("pac")
        assert_no_error(r)
        assert_contains(r, "pointer authentication", "PACIA", "AUTIA")

    def test_gic(self):
        r = explain_arm_concept("gic")
        assert_no_error(r)
        assert_contains(r, "distributor", "redistributor", "SPI", "PPI", "SGI")

    def test_nvic(self):
        r = explain_arm_concept("nvic")
        assert_no_error(r)
        assert_contains(r, "NVIC", "tail-chaining", "cortex-m")

    def test_psci(self):
        r = explain_arm_concept("psci")
        assert_no_error(r)
        assert_contains(r, "CPU_ON", "CPU_OFF", "power")

    def test_alias_el(self):
        r = explain_arm_concept("el")
        assert_no_error(r)
        assert_contains(r, "exception level")

    def test_alias_barriers(self):
        r = explain_arm_concept("barriers")
        assert_no_error(r)
        assert_contains(r, "DMB", "DSB")

    def test_alias_page_tables(self):
        r = explain_arm_concept("page tables")
        assert_no_error(r)
        assert_contains(r, "vmsa", "translation")

    def test_not_found(self):
        r = explain_arm_concept("quantum_computing")
        assert "no concept found" in r.lower()


# ---- Tool 3: find_register_in_manual ----

class TestFindRegisterInManual:
    def test_sctlr_el1(self):
        r = find_register_in_manual("SCTLR_EL1")
        assert_no_error(r)
        assert_contains(r, "SCTLR_EL1", "DDI0487", "MMU")

    def test_tcr_el1(self):
        r = find_register_in_manual("TCR_EL1")
        assert_no_error(r)
        assert_contains(r, "TCR_EL1", "translation control", "granule")

    def test_vbar_el1(self):
        r = find_register_in_manual("VBAR_EL1")
        assert_no_error(r)
        assert_contains(r, "VBAR_EL1", "vector", "exception")

    def test_esr_el1(self):
        r = find_register_in_manual("ESR_EL1")
        assert_no_error(r)
        assert_contains(r, "syndrome", "exception")

    def test_cpsr(self):
        r = find_register_in_manual("CPSR")
        assert_no_error(r)
        assert_contains(r, "CPSR", "DDI0406", "aarch32")

    def test_vtor(self):
        r = find_register_in_manual("VTOR")
        assert_no_error(r)
        assert_contains(r, "VTOR", "vector table", "armv7-m")

    def test_case_insensitive(self):
        r = find_register_in_manual("sctlr_el1")
        assert_no_error(r)
        assert_contains(r, "SCTLR_EL1")

    def test_with_context_filter(self):
        r = find_register_in_manual("MAIR_EL1", context="memory")
        assert_no_error(r)
        assert_contains(r, "memory attribute")

    def test_not_found(self):
        r = find_register_in_manual("NONEXISTENT_REG")
        assert "not found" in r.lower()

    def test_partial_match_multiple(self):
        r = find_register_in_manual("SPSR")
        # Should find SPSR_EL1 and potentially others
        assert "SPSR" in r


# ---- Tool 4: get_errata ----

class TestGetErrata:
    def test_cortex_a53(self):
        r = get_errata("cortex-a53")
        assert_no_error(r)
        assert_contains(r, "843419", "835769", "cortex-a53")

    def test_cortex_a53_functional(self):
        r = get_errata("cortex-a53", category="functional")
        assert_no_error(r)
        assert_contains(r, "functional")
        assert "performance" not in r.lower().split("category")[0]

    def test_cortex_a76_security(self):
        r = get_errata("cortex-a76", category="security")
        assert_no_error(r)
        assert_contains(r, "spectre", "SSB")

    def test_cortex_a55(self):
        r = get_errata("cortex-a55")
        assert_no_error(r)
        assert_contains(r, "cortex-a55", "1530923")

    def test_neoverse_n1(self):
        r = get_errata("neoverse-n1")
        assert_no_error(r)
        assert_contains(r, "neoverse n1")

    def test_cortex_m33(self):
        r = get_errata("cortex-m33")
        assert_no_error(r)
        assert_contains(r, "cortex-m33", "lazy")

    def test_case_insensitive(self):
        r = get_errata("Cortex-A72")
        assert_no_error(r)
        assert_contains(r, "cortex-a72")

    def test_space_in_name(self):
        r = get_errata("cortex a53")
        assert_no_error(r)
        assert_contains(r, "cortex-a53")

    def test_not_found(self):
        r = get_errata("cortex-z99")
        assert "not found" in r.lower()

    def test_invalid_category(self):
        r = get_errata("cortex-a53", category="magical")
        assert "Error" in r

    def test_errata_has_workaround(self):
        r = get_errata("cortex-a53")
        assert_contains(r, "workaround")


# ---- Tool 5: compare_manual_sections ----

class TestCompareManualSections:
    def test_exception_handling(self):
        r = compare_manual_sections("exception_handling")
        assert_no_error(r)
        assert_contains(r, "ARMv7-A", "ARMv8-A", "ARMv7-M")

    def test_memory_management(self):
        r = compare_manual_sections("memory_management")
        assert_no_error(r)
        assert_contains(r, "translation", "MPU")

    def test_simd_extensions(self):
        r = compare_manual_sections("simd_extensions")
        assert_no_error(r)
        assert_contains(r, "NEON", "SVE2", "128-bit")

    def test_security_model(self):
        r = compare_manual_sections("security_model")
        assert_no_error(r)
        assert_contains(r, "trustzone", "RME", "SAU")

    def test_interrupt_handling(self):
        r = compare_manual_sections("interrupt_handling")
        assert_no_error(r)
        assert_contains(r, "GIC", "NVIC")

    def test_alias_exceptions(self):
        r = compare_manual_sections("exceptions")
        assert_no_error(r)
        assert_contains(r, "exception")

    def test_alias_neon(self):
        r = compare_manual_sections("neon")
        assert_no_error(r)
        assert_contains(r, "NEON")

    def test_alias_trustzone(self):
        r = compare_manual_sections("trustzone")
        assert_no_error(r)
        assert_contains(r, "trustzone")

    def test_not_found(self):
        r = compare_manual_sections("quantum_algorithms")
        assert "no comparison data" in r.lower()


# ---- Tool 6: list_arm_documents ----

class TestListArmDocuments:
    def test_list_all(self):
        r = list_arm_documents()
        assert_no_error(r)
        assert_contains(r, "ARM Documentation Catalog", "DDI0487")

    def test_filter_by_scope(self):
        r = list_arm_documents(doc_scope="architecture")
        assert_no_error(r)
        assert_contains(r, "architecture")

    def test_filter_by_architecture(self):
        r = list_arm_documents(architecture="aarch64")
        assert_no_error(r)
        assert_contains(r, "aarch64")

    def test_filter_both(self):
        r = list_arm_documents(doc_scope="programming_guide", architecture="aarch64")
        assert_no_error(r)
        assert_contains(r, "programming_guide", "aarch64")

    def test_no_matches(self):
        r = list_arm_documents(doc_scope="nonexistent_scope")
        assert "no documents" in r.lower()

    def test_sections_listed(self):
        r = list_arm_documents()
        assert_no_error(r)
        # Should list section topics
        assert_contains(r, "sections")


# ---- Tool 7: explain_instruction_encoding ----

class TestExplainInstructionEncoding:
    def test_a64(self):
        r = explain_instruction_encoding("a64")
        assert_no_error(r)
        assert_contains(r, "A64", "32-bit fixed", "encoding groups")

    def test_aarch64_alias(self):
        r = explain_instruction_encoding("aarch64")
        assert_no_error(r)
        assert_contains(r, "A64")

    def test_t32(self):
        r = explain_instruction_encoding("t32")
        assert_no_error(r)
        assert_contains(r, "T32", "Thumb-2", "16-bit", "32-bit")

    def test_thumb_alias(self):
        r = explain_instruction_encoding("thumb")
        assert_no_error(r)
        assert_contains(r, "T32")

    def test_a32(self):
        r = explain_instruction_encoding("a32")
        assert_no_error(r)
        assert_contains(r, "A32", "condition", "barrel shifter")

    def test_arm_alias(self):
        r = explain_instruction_encoding("arm")
        assert_no_error(r)
        assert_contains(r, "A32")

    def test_overview(self):
        r = explain_instruction_encoding("overview")
        assert_no_error(r)
        assert_contains(r, "A64", "T32", "A32")

    def test_not_found(self):
        r = explain_instruction_encoding("risc-v")
        assert "Error" in r


# ---- Self-contained runner ----

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestSearchArmDocs,
        TestExplainArmConcept,
        TestFindRegisterInManual,
        TestGetErrata,
        TestCompareManualSections,
        TestListArmDocuments,
        TestExplainInstructionEncoding,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            test_func = getattr(instance, method_name)
            label = f"{cls.__name__}.{method_name}"
            try:
                test_func()
                passed += 1
                print(f"  PASS  {label}")
            except Exception:
                failed += 1
                tb = traceback.format_exc()
                errors.append((label, tb))
                print(f"  FAIL  {label}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")

    if errors:
        print(f"\n{'='*60}")
        print("FAILURES:\n")
        for label, tb in errors:
            print(f"--- {label} ---")
            print(tb)

    sys.exit(1 if failed else 0)
