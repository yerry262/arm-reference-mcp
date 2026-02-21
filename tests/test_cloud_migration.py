"""Smoke tests for ARM Cloud Migration Advisor MCP tools.

Run with: python -m pytest tests/test_cloud_migration.py -v
Or simply: python tests/test_cloud_migration.py
"""

import sys
import os

# Ensure the src directory is on the path for editable installs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arm_reference_mcp.cloud_migration_server import (
    scan_x86_dependencies,
    suggest_arm_cloud_instance,
    check_docker_arm_support,
    generate_ci_matrix,
    estimate_migration_effort,
    generate_arm_dockerfile,
    compare_arm_vs_x86_perf,
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


# ---- Tool 1: scan_x86_dependencies ----

class TestScanX86Dependencies:
    def test_python_compatible(self):
        r = scan_x86_dependencies("python", "numpy,pandas,scipy")
        assert_no_error(r)
        assert_contains(r, "compatible", "numpy", "pandas", "scipy")

    def test_python_x86_only(self):
        r = scan_x86_dependencies("python", "intel-mkl,openvino")
        assert_no_error(r)
        assert_contains(r, "x86-only", "intel-mkl", "openvino", "alternative")

    def test_python_mixed(self):
        r = scan_x86_dependencies("python", "numpy,intel-mkl,psycopg2")
        assert_no_error(r)
        assert_contains(r, "compatible", "x86-only", "conditional")

    def test_nodejs(self):
        r = scan_x86_dependencies("nodejs", "sharp,esbuild,node-sass")
        assert_no_error(r)
        assert_contains(r, "sharp", "esbuild", "node-sass")

    def test_java(self):
        r = scan_x86_dependencies("java", "netty,snappy-java,rocksdbjni")
        assert_no_error(r)
        assert_contains(r, "compatible", "netty", "snappy-java")

    def test_invalid_language(self):
        r = scan_x86_dependencies("cobol", "some-dep")
        assert "Error" in r


# ---- Tool 2: suggest_arm_cloud_instance ----

class TestSuggestArmCloudInstance:
    def test_web_server(self):
        r = suggest_arm_cloud_instance("web_server")
        assert_no_error(r)
        assert_contains(r, "web_server", "graviton", "cobalt", "axion")

    def test_ml_inference(self):
        r = suggest_arm_cloud_instance("ml_inference")
        assert_no_error(r)
        assert_contains(r, "ml_inference")

    def test_specific_provider_aws(self):
        r = suggest_arm_cloud_instance("database", provider="aws")
        assert_no_error(r)
        assert_contains(r, "graviton", "database")
        # Should not mention other providers by name in the main section
        assert "cobalt" not in r.lower().split("cost savings")[0]

    def test_specific_provider_oracle(self):
        r = suggest_arm_cloud_instance("general", provider="oracle")
        assert_no_error(r)
        assert_contains(r, "ampere", "oracle")

    def test_hpc(self):
        r = suggest_arm_cloud_instance("hpc")
        assert_no_error(r)
        assert_contains(r, "hpc")

    def test_invalid_profile(self):
        r = suggest_arm_cloud_instance("quantum_computing")
        assert "Error" in r


# ---- Tool 3: check_docker_arm_support ----

class TestCheckDockerArmSupport:
    def test_ubuntu(self):
        r = check_docker_arm_support("ubuntu")
        assert_no_error(r)
        assert_contains(r, "supported", "multi-arch", "arm64")

    def test_node(self):
        r = check_docker_arm_support("node")
        assert_no_error(r)
        assert_contains(r, "supported", "arm64")

    def test_selenium_x86_only(self):
        r = check_docker_arm_support("selenium")
        assert_no_error(r)
        assert_contains(r, "not supported", "alternative")

    def test_postgres(self):
        r = check_docker_arm_support("postgres")
        assert_no_error(r)
        assert_contains(r, "supported", "arm64")

    def test_unknown_image(self):
        r = check_docker_arm_support("my-custom-private-image-xyz")
        assert_no_error(r)
        assert_contains(r, "unknown", "not in the compatibility database")


# ---- Tool 4: generate_ci_matrix ----

class TestGenerateCiMatrix:
    def test_github_actions(self):
        r = generate_ci_matrix("github_actions")
        assert_no_error(r)
        assert_contains(r, "github actions", "arm64", "multi-arch")

    def test_gitlab_ci(self):
        r = generate_ci_matrix("gitlab_ci")
        assert_no_error(r)
        assert_contains(r, "gitlab ci", "arm64")

    def test_circleci(self):
        r = generate_ci_matrix("circleci")
        assert_no_error(r)
        assert_contains(r, "circleci", "arm")

    def test_jenkins(self):
        r = generate_ci_matrix("jenkins")
        assert_no_error(r)
        assert_contains(r, "jenkins", "arm64")

    def test_with_language(self):
        r = generate_ci_matrix("github_actions", language="python")
        assert_no_error(r)
        assert_contains(r, "python", "pytest")

    def test_invalid_platform(self):
        r = generate_ci_matrix("bamboo")
        assert "Error" in r


# ---- Tool 5: estimate_migration_effort ----

class TestEstimateMigrationEffort:
    def test_python_web(self):
        r = estimate_migration_effort("python_web")
        assert_no_error(r)
        assert_contains(r, "python web", "low", "checklist", "rollback")

    def test_cpp_native(self):
        r = estimate_migration_effort("cpp_native")
        assert_no_error(r)
        assert_contains(r, "c++ native", "high", "simd", "neon")

    def test_go_microservice(self):
        r = estimate_migration_effort("go_microservice")
        assert_no_error(r)
        assert_contains(r, "go microservice", "low", "goarch")

    def test_rust_systems(self):
        r = estimate_migration_effort("rust_systems")
        assert_no_error(r)
        assert_contains(r, "rust systems", "medium")

    def test_invalid_profile(self):
        r = estimate_migration_effort("fortran_hpc")
        assert "Error" in r


# ---- Tool 6: generate_arm_dockerfile ----

class TestGenerateArmDockerfile:
    def test_python(self):
        r = generate_arm_dockerfile("python")
        assert_no_error(r)
        assert_contains(r, "python", "dockerfile", "multi-arch", "arm64")

    def test_nodejs(self):
        r = generate_arm_dockerfile("nodejs")
        assert_no_error(r)
        assert_contains(r, "node", "dockerfile", "arm64")

    def test_node_alias(self):
        r = generate_arm_dockerfile("node")
        assert_no_error(r)
        assert_contains(r, "node")

    def test_java(self):
        r = generate_arm_dockerfile("java")
        assert_no_error(r)
        assert_contains(r, "java", "temurin", "dockerfile")

    def test_go(self):
        r = generate_arm_dockerfile("go")
        assert_no_error(r)
        assert_contains(r, "go", "CGO_ENABLED", "dockerfile")

    def test_rust(self):
        r = generate_arm_dockerfile("rust")
        assert_no_error(r)
        assert_contains(r, "rust", "cargo", "dockerfile")

    def test_overview(self):
        r = generate_arm_dockerfile("overview")
        assert_no_error(r)
        assert_contains(r, "python", "node", "java", "go", "rust")

    def test_not_found(self):
        r = generate_arm_dockerfile("fortran")
        assert "Error" in r


# ---- Tool 7: compare_arm_vs_x86_perf ----

class TestCompareArmVsX86Perf:
    def test_web_server(self):
        r = compare_arm_vs_x86_perf("web_server")
        assert_no_error(r)
        assert_contains(r, "web server", "graviton", "requests/sec")

    def test_database(self):
        r = compare_arm_vs_x86_perf("database")
        assert_no_error(r)
        assert_contains(r, "database", "postgresql", "TPS")

    def test_ml_inference(self):
        r = compare_arm_vs_x86_perf("ml_inference")
        assert_no_error(r)
        assert_contains(r, "ml inference", "resnet", "inferences/sec")

    def test_ci_cd(self):
        r = compare_arm_vs_x86_perf("ci_cd")
        assert_no_error(r)
        assert_contains(r, "ci/cd", "build", "compile")

    def test_hpc(self):
        r = compare_arm_vs_x86_perf("hpc")
        assert_no_error(r)
        assert_contains(r, "hpc", "GFLOPS")

    def test_alias_web(self):
        r = compare_arm_vs_x86_perf("web")
        assert_no_error(r)
        assert_contains(r, "web server")

    def test_overview(self):
        r = compare_arm_vs_x86_perf("overview")
        assert_no_error(r)
        assert_contains(r, "web_server", "database", "ml_inference", "ci_cd", "hpc")

    def test_not_found(self):
        r = compare_arm_vs_x86_perf("quantum_computing")
        assert "Error" in r


# ---- Run all tests ----

def run_all():
    """Simple test runner -- no pytest dependency needed."""
    import traceback

    test_classes = [
        TestScanX86Dependencies,
        TestSuggestArmCloudInstance,
        TestCheckDockerArmSupport,
        TestGenerateCiMatrix,
        TestEstimateMigrationEffort,
        TestGenerateArmDockerfile,
        TestCompareArmVsX86Perf,
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
                print(f"  FAIL  {test_id}  --  {e}")

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
