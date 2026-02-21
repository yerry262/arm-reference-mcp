"""Smoke tests for ARM TinyML & Edge AI MCP tools.

Run with: python -m pytest tests/test_tinyml.py -v
Or simply: python tests/test_tinyml.py
"""

import sys
import os

# Ensure the src directory is on the path for editable installs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arm_reference_mcp.tinyml_server import (
    check_operator_support,
    suggest_quantization,
    estimate_inference,
    compare_tinyml_targets,
    generate_deployment_config,
    suggest_model_architecture,
    explain_tinyml_framework,
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


# ---- Tool 1: check_operator_support ----

class TestCheckOperatorSupport:
    def test_ethos_u55_supported(self):
        r = check_operator_support("ethos-u55", "conv2d,relu,softmax,reshape")
        assert_no_error(r)
        assert_contains(r, "supported", "conv2d", "relu", "softmax")

    def test_ethos_u55_unsupported(self):
        r = check_operator_support("ethos-u55", "conv2d,lstm,gelu,layer_norm")
        assert_no_error(r)
        assert_contains(r, "not supported", "lstm", "gelu")

    def test_ethos_u65_resize(self):
        r = check_operator_support("ethos-u65", "resize_bilinear,conv2d")
        assert_no_error(r)
        assert_contains(r, "supported", "resize_bilinear")

    def test_cortex_m_cmsis_nn(self):
        r = check_operator_support("cortex-m_cmsis_nn", "conv2d,lstm,relu,softmax")
        assert_no_error(r)
        assert_contains(r, "supported", "conv2d", "lstm")

    def test_cortex_a_armnn(self):
        r = check_operator_support("cortex-a_armnn", "conv2d,gelu,layer_norm,lstm")
        assert_no_error(r)
        assert_contains(r, "supported")

    def test_all_supported_score(self):
        r = check_operator_support("cortex-a_armnn", "conv2d,relu,softmax")
        assert_contains(r, "100%")

    def test_invalid_target(self):
        r = check_operator_support("quantum-chip", "conv2d")
        assert "Error" in r

    def test_empty_operators(self):
        r = check_operator_support("ethos-u55", "")
        assert "Error" in r

    def test_unknown_operator(self):
        r = check_operator_support("ethos-u55", "conv2d,quantum_entangle")
        assert_no_error(r)
        assert_contains(r, "unknown", "quantum_entangle")

    def test_compatibility_score(self):
        r = check_operator_support("ethos-u55", "conv2d,relu,lstm")
        assert_no_error(r)
        assert_contains(r, "compatibility score")


# ---- Tool 2: suggest_quantization ----

class TestSuggestQuantization:
    def test_image_classification_ethos_u55(self):
        r = suggest_quantization("image_classification", "ethos-u55")
        assert_no_error(r)
        assert_contains(r, "INT8", "quantization", "vela")

    def test_image_classification_cortex_m(self):
        r = suggest_quantization("image_classification", "cortex-m_cmsis_nn")
        assert_no_error(r)
        assert_contains(r, "QAT", "INT8", "CMSIS-NN")

    def test_image_classification_cortex_a(self):
        r = suggest_quantization("image_classification", "cortex-a_armnn")
        assert_no_error(r)
        assert_contains(r, "FP16")

    def test_object_detection_ethos_u55(self):
        r = suggest_quantization("object_detection", "ethos-u55")
        assert_no_error(r)
        assert_contains(r, "INT8", "NMS")

    def test_keyword_spotting_cortex_m(self):
        r = suggest_quantization("keyword_spotting", "cortex-m_cmsis_nn")
        assert_no_error(r)
        assert_contains(r, "INT8", "CMSIS")

    def test_anomaly_detection(self):
        r = suggest_quantization("anomaly_detection", "ethos-u55")
        assert_no_error(r)
        assert_contains(r, "INT8", "autoencoder")

    def test_nlp_embedding_ethos_u55(self):
        r = suggest_quantization("nlp_embedding", "ethos-u55")
        assert_no_error(r)
        assert_contains(r, "not recommended")

    def test_nlp_embedding_cortex_a(self):
        r = suggest_quantization("nlp_embedding", "cortex-a_armnn")
        assert_no_error(r)
        assert_contains(r, "FP16")

    def test_invalid_model_type(self):
        r = suggest_quantization("quantum_model", "ethos-u55")
        assert "Error" in r

    def test_invalid_target(self):
        r = suggest_quantization("image_classification", "fpga-custom")
        assert "Error" in r


# ---- Tool 3: estimate_inference ----

class TestEstimateInference:
    def test_mobilenetv2_ethos_u55(self):
        r = estimate_inference("mobilenetv2_1.0_224", "ethos-u55")
        assert_no_error(r)
        assert_contains(r, "mobilenetv2", "inference time", "ms")

    def test_mobilenetv2_cortex_a78(self):
        r = estimate_inference("mobilenetv2_1.0_224", "cortex-a78_armnn")
        assert_no_error(r)
        assert_contains(r, "mobilenetv2", "ms", "FPS")

    def test_ds_cnn_cortex_m4(self):
        r = estimate_inference("ds_cnn_s", "cortex-m4_cmsis_nn")
        assert_no_error(r)
        assert_contains(r, "ds-cnn", "keyword")

    def test_yolov8_ethos_u65(self):
        r = estimate_inference("yolov8n_320", "ethos-u65")
        assert_no_error(r)
        assert_contains(r, "yolov8", "ms")

    def test_autoencoder_cortex_m4(self):
        r = estimate_inference("autoencoder_small", "cortex-m4_cmsis_nn")
        assert_no_error(r)
        assert_contains(r, "autoencoder", "anomaly")

    def test_alias_mobilenet(self):
        r = estimate_inference("mobilenet", "ethos-u55")
        assert_no_error(r)
        assert_contains(r, "mobilenetv2")

    def test_alias_yolo(self):
        r = estimate_inference("yolo", "cortex-a78_armnn")
        assert_no_error(r)
        assert_contains(r, "yolov8")

    def test_alias_autoencoder(self):
        r = estimate_inference("autoencoder", "cortex-m4_cmsis_nn")
        assert_no_error(r)

    def test_not_found_model(self):
        r = estimate_inference("gpt4_model", "ethos-u55")
        assert "not found" in r.lower() or "no inference data" in r.lower()

    def test_not_found_target(self):
        r = estimate_inference("mobilenetv2_1.0_224", "fpga-custom")
        assert "not found" in r.lower() or "no inference data" in r.lower()

    def test_memory_requirements(self):
        r = estimate_inference("mobilenetv2_1.0_224", "ethos-u55")
        assert_contains(r, "SRAM", "flash")

    def test_small_model_cortex_m55(self):
        r = estimate_inference("mobilenetv2_0.35_96", "cortex-m55_cmsis_nn")
        assert_no_error(r)
        assert_contains(r, "0.35")


# ---- Tool 4: compare_tinyml_targets ----

class TestCompareTinymlTargets:
    def test_m4_vs_m55(self):
        r = compare_tinyml_targets("cortex-m4", "cortex-m55")
        assert_no_error(r)
        assert_contains(r, "cortex-m4", "cortex-m55", "helium")

    def test_ethos_u55_vs_u65(self):
        r = compare_tinyml_targets("ethos-u55", "ethos-u65")
        assert_no_error(r)
        assert_contains(r, "ethos-u55", "ethos-u65", "MAC")

    def test_m55_vs_a55(self):
        r = compare_tinyml_targets("cortex-m55", "cortex-a55")
        assert_no_error(r)
        assert_contains(r, "cortex-m55", "cortex-a55")

    def test_a55_vs_a78(self):
        r = compare_tinyml_targets("cortex-a55", "cortex-a78")
        assert_no_error(r)
        assert_contains(r, "cortex-a55", "cortex-a78")

    def test_m7_vs_m85(self):
        r = compare_tinyml_targets("cortex-m7", "cortex-m85")
        assert_no_error(r)
        assert_contains(r, "cortex-m7", "cortex-m85")

    def test_has_frameworks(self):
        r = compare_tinyml_targets("cortex-m55", "ethos-u55")
        assert_contains(r, "ML Frameworks")

    def test_has_best_for(self):
        r = compare_tinyml_targets("cortex-m4", "cortex-m7")
        assert_contains(r, "Best For")

    def test_has_example_devices(self):
        r = compare_tinyml_targets("cortex-m4", "cortex-a55")
        assert_contains(r, "Example Devices")

    def test_not_found(self):
        r = compare_tinyml_targets("cortex-z99", "cortex-m4")
        assert "not found" in r.lower()


# ---- Tool 5: generate_deployment_config ----

class TestGenerateDeploymentConfig:
    def test_tflite_micro(self):
        r = generate_deployment_config("tflite_micro_cmsis")
        assert_no_error(r)
        assert_contains(r, "TFLite", "CMSIS", "cmake", "cortex-m")

    def test_vela_ethos(self):
        r = generate_deployment_config("vela_ethos_u")
        assert_no_error(r)
        assert_contains(r, "vela", "ethos", "accelerator-config")

    def test_armnn(self):
        r = generate_deployment_config("armnn_cortex_a")
        assert_no_error(r)
        assert_contains(r, "arm nn", "cortex-a", "neon")

    def test_alias_tflite(self):
        r = generate_deployment_config("tflite")
        assert_no_error(r)
        assert_contains(r, "TFLite")

    def test_alias_cmsis(self):
        r = generate_deployment_config("cmsis")
        assert_no_error(r)
        assert_contains(r, "CMSIS")

    def test_alias_vela(self):
        r = generate_deployment_config("vela")
        assert_no_error(r)
        assert_contains(r, "vela")

    def test_alias_armnn(self):
        r = generate_deployment_config("armnn")
        assert_no_error(r)
        assert_contains(r, "arm nn")

    def test_alias_ethos(self):
        r = generate_deployment_config("ethos")
        assert_no_error(r)
        assert_contains(r, "ethos")

    def test_has_build_system(self):
        r = generate_deployment_config("tflite_micro_cmsis")
        assert_contains(r, "build system", "cmake")

    def test_has_model_conversion(self):
        r = generate_deployment_config("tflite_micro_cmsis")
        assert_contains(r, "model conversion")

    def test_has_runtime_code(self):
        r = generate_deployment_config("tflite_micro_cmsis")
        assert_contains(r, "runtime code")

    def test_not_found(self):
        r = generate_deployment_config("custom_framework_xyz")
        assert "no deployment config" in r.lower()


# ---- Tool 6: suggest_model_architecture ----

class TestSuggestModelArchitecture:
    def test_image_classification_small(self):
        r = suggest_model_architecture("image_classification", sram_kb=64, flash_kb=512)
        assert_no_error(r)
        assert_contains(r, "MobileNetV2 0.35")

    def test_image_classification_large(self):
        r = suggest_model_architecture("image_classification", sram_kb=1024, flash_kb=8192)
        assert_no_error(r)
        # Should return multiple fitting models
        assert_contains(r, "models that fit")

    def test_keyword_spotting(self):
        r = suggest_model_architecture("keyword_spotting", sram_kb=32, flash_kb=128)
        assert_no_error(r)
        assert_contains(r, "DS-CNN")

    def test_object_detection(self):
        r = suggest_model_architecture("object_detection", sram_kb=64, flash_kb=128)
        assert_no_error(r)
        assert_contains(r, "FOMO")

    def test_anomaly_detection(self):
        r = suggest_model_architecture("anomaly_detection", sram_kb=16, flash_kb=64)
        assert_no_error(r)
        assert_contains(r, "autoencoder")

    def test_too_small_hardware(self):
        r = suggest_model_architecture("image_classification", sram_kb=8, flash_kb=16)
        assert_no_error(r)
        assert_contains(r, "no models fit")

    def test_overview(self):
        r = suggest_model_architecture("overview", sram_kb=0, flash_kb=0)
        assert_no_error(r)
        assert_contains(r, "image_classification", "keyword_spotting", "anomaly_detection")

    def test_alias_kws(self):
        r = suggest_model_architecture("kws", sram_kb=64, flash_kb=512)
        assert_no_error(r)
        assert_contains(r, "DS-CNN")

    def test_not_found(self):
        r = suggest_model_architecture("quantum_task", sram_kb=256, flash_kb=1024)
        assert "Error" in r

    def test_headroom_shown(self):
        r = suggest_model_architecture("anomaly_detection", sram_kb=256, flash_kb=1024)
        assert_no_error(r)
        assert_contains(r, "headroom")


# ---- Tool 7: explain_tinyml_framework ----

class TestExplainTinymlFramework:
    def test_tflite_micro(self):
        r = explain_tinyml_framework("tflite_micro")
        assert_no_error(r)
        assert_contains(r, "TFLite", "Google", "arena", "FlatBuffer")

    def test_tflite_alias(self):
        r = explain_tinyml_framework("tflite")
        assert_no_error(r)
        assert_contains(r, "TFLite")

    def test_cmsis_nn(self):
        r = explain_tinyml_framework("cmsis_nn")
        assert_no_error(r)
        assert_contains(r, "CMSIS-NN", "Arm", "DSP", "Helium")

    def test_vela(self):
        r = explain_tinyml_framework("vela")
        assert_no_error(r)
        assert_contains(r, "Vela", "Ethos", "offline compiler")

    def test_armnn(self):
        r = explain_tinyml_framework("armnn")
        assert_no_error(r)
        assert_contains(r, "Arm NN", "NEON", "Compute Library")

    def test_edge_impulse(self):
        r = explain_tinyml_framework("edge_impulse")
        assert_no_error(r)
        assert_contains(r, "Edge Impulse", "EON", "FOMO")

    def test_alias_ei(self):
        r = explain_tinyml_framework("ei")
        assert_no_error(r)
        assert_contains(r, "Edge Impulse")

    def test_alias_ethos(self):
        r = explain_tinyml_framework("ethos")
        assert_no_error(r)
        assert_contains(r, "Vela")

    def test_overview(self):
        r = explain_tinyml_framework("overview")
        assert_no_error(r)
        assert_contains(r, "tflite_micro", "cmsis_nn", "vela", "armnn", "edge_impulse")

    def test_not_found(self):
        r = explain_tinyml_framework("custom_framework")
        assert "Error" in r


# ---- Self-contained runner ----

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestCheckOperatorSupport,
        TestSuggestQuantization,
        TestEstimateInference,
        TestCompareTinymlTargets,
        TestGenerateDeploymentConfig,
        TestSuggestModelArchitecture,
        TestExplainTinymlFramework,
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
