---
title: ARM TinyML & Edge AI MCP
---

# ARM TinyML & Edge AI MCP

Plan and optimize ML model deployment on ARM microcontrollers and edge accelerators. **7 tools** for operator compatibility analysis, quantization planning, inference estimation, hardware comparison, deployment configuration, model architecture selection, and framework reference.

**Server entry point:** `arm-tinyml-mcp`

---

## Tool Reference

| Tool | Description |
|------|-------------|
| `check_operator_support(target, operators)` | Check if ML operators are supported on a given ARM target. Targets: `ethos-u55`, `ethos-u65`, `cortex-m_cmsis_nn`, `cortex-a_armnn`. Reports support status, constraints, and alternatives for unsupported operators. |
| `suggest_quantization(model_type, target)` | Recommend a quantization strategy for a model type and ARM target. Model types: `image_classification`, `object_detection`, `keyword_spotting`, `anomaly_detection`, `nlp_embedding`. Returns precision, method, tools, accuracy impact, and tips. |
| `estimate_inference(model_name, target)` | Estimate inference latency and memory requirements. Models: `mobilenetv2_1.0_224`, `mobilenetv2_0.35_96`, `ds_cnn_s`, `yolov8n_320`, `autoencoder_small`. Returns estimated FPS, SRAM/Flash requirements, and deployment notes. |
| `compare_tinyml_targets(target_a, target_b)` | Side-by-side comparison of two hardware targets: architecture, clock speed, SIMD capabilities, memory, power, supported frameworks, best-use scenarios, and example devices. |
| `generate_deployment_config(framework)` | Generate build system, model conversion, and runtime code for an ARM edge AI framework. Frameworks: `tflite_micro_cmsis`/`tflite`/`cmsis`, `vela_ethos_u`/`vela`/`ethos`, `armnn_cortex_a`/`armnn`. |
| `suggest_model_architecture(task, sram_kb, flash_kb)` | Recommend ML models that fit within SRAM and Flash constraints. Tasks: `image_classification`, `keyword_spotting`, `object_detection`, `anomaly_detection`. Filters by hardware limits and shows headroom. |
| `explain_tinyml_framework(framework)` | Deep-dive on a TinyML framework: organization, targets, model format, language, quantization support, key features, getting started, and limitations. Frameworks: `tflite_micro`, `cmsis_nn`, `vela`, `armnn`, `edge_impulse`. |

---

## Examples

### Checking operator support

```
> check_operator_support("ethos-u55", "conv2d,relu,lstm,softmax,gelu")

# Operator Support: ethos-u55

## Supported (3/5)
  conv2d   -- Fully accelerated (INT8, INT16)
  relu     -- Fused with preceding op (zero overhead)
  softmax  -- Accelerated

## Not Supported (2/5)
  lstm     -- Must unroll into FC + Add + Sigmoid + Tanh
               (each sub-op is individually supported)
  gelu     -- Not supported. Replace with ReLU6 for similar behavior.

## Compatibility Score: 60%
```

### Quantization strategy

```
> suggest_quantization("image_classification", "ethos-u55")

# Quantization Strategy
  Model type: image_classification
  Target: ethos-u55
  Recommended: INT8 Post-Training Quantization (PTQ)

## Method
  1. Train model in FP32 as usual
  2. Use TFLite converter with representative dataset (100-500 images)
  3. Run Vela compiler to optimize for Ethos-U55

## Expected Impact
  Accuracy: <1% top-1 drop for MobileNetV2
  Size: 4x reduction (FP32 -> INT8)
  Speed: 10-50x vs FP32 on CPU

## Tips
  - Use per-channel quantization for Conv2D layers
  - Calibrate with data from deployment domain
  - Test edge cases (low light, unusual angles)
```

### Inference estimation

```
> estimate_inference("mobilenetv2_1.0_224", "cortex-m55_cmsis_nn")

# Inference Estimate
  Model: MobileNetV2 1.0 224x224 (INT8)
  Target: Cortex-M55 @ 400 MHz + Helium MVE + CMSIS-NN

## Performance
  Parameters: 3.4M
  MACs: 300M
  Latency: ~150 ms
  Throughput: ~6.7 FPS

## Memory Requirements
  SRAM: 512 KB minimum (activation tensors)
  Flash: 3400 KB (model weights)

## Notes
  Helium MVE provides ~4x speedup over scalar Cortex-M
  Consider MobileNetV2 0.35 96x96 for tighter memory budgets
```

### Comparing hardware targets

```
> compare_tinyml_targets("cortex-m55", "ethos-u55")

# Target Comparison: Cortex-M55 vs Ethos-U55

  Attribute        Cortex-M55              Ethos-U55
  ----------------------------------------------------------------
  Type             CPU (general purpose)   NPU (ML accelerator)
  Architecture     ARMv8.1-M              Dedicated ML pipeline
  Clock            Up to 400 MHz           Up to 500 MHz
  SIMD             Helium MVE (128-bit)    Fixed-function MAC array
  ML Perf          ~0.5 TOPS (INT8)        ~1.0 TOPS (INT8)
  Power            30-150 mW               5-50 mW
  Memory           256 KB - 4 MB SRAM      Shared with host CPU
  Frameworks       TFLite Micro, CMSIS-NN  Vela + TFLite Micro
  Best For         General compute + ML     Dedicated ML inference

## Recommendation
  Pair Cortex-M55 + Ethos-U55 for best results:
  M55 handles pre/post-processing, U55 runs the neural network.
  Example: Corstone-300 reference platform.
```

### Suggesting model architectures

```
> suggest_model_architecture("image_classification", sram_kb=256, flash_kb=512)

# Model Recommendations: image_classification
  Constraints: 256 KB SRAM, 512 KB Flash

## Recommended Models

  1. MobileNetV2 0.35 96x96
     Params: 0.4M | SRAM: ~200 KB | Flash: ~400 KB
     Headroom: SRAM 22%, Flash 20%
     Accuracy: ~60% top-1 (ImageNet)

  2. MCUNet 64x64
     Params: 0.7M | SRAM: ~180 KB | Flash: ~350 KB
     Headroom: SRAM 30%, Flash 32%
     Optimized for microcontroller deployment

## Filtered Out
  MobileNetV2 1.0 224 -- exceeds Flash (3400 KB > 512 KB)
  EfficientNet-Lite0 -- exceeds SRAM (800 KB > 256 KB)
```

### Deployment configuration

```
> generate_deployment_config("vela")

# Vela Deployment Configuration (Ethos-U)

## Vela Compiler
  pip install ethos-u-vela
  vela model.tflite \
    --accelerator-config ethos-u55-128 \
    --optimise Performance

## CMake Build System
  cmake_minimum_required(VERSION 3.21)
  project(ethos_u_app)
  set(CMAKE_SYSTEM_PROCESSOR cortex-m55)
  ...

## Runtime Code Pattern (C++)
  #include "tensorflow/lite/micro/micro_interpreter.h"
  #include "ethosu_driver.h"
  // Initialize Ethos-U driver
  // Create TFLite Micro interpreter
  // Run inference
```

### Framework deep-dive

```
> explain_tinyml_framework("cmsis_nn")

# CMSIS-NN
  Organization: ARM
  Targets: Cortex-M0/M3/M4/M7/M33/M55/M85
  Model Format: Integrated via TFLite Micro delegate
  Language: C
  Quantization: INT8, INT16

## Key Features
  - Hand-tuned assembly kernels for DSP and Helium MVE
  - Optimized Conv2D, DepthwiseConv, FC, Pooling, Softmax
  - 2-10x speedup over naive C implementations
  - Zero external dependencies

## Getting Started
  1. Include CMSIS-NN as a TFLite Micro delegate
  2. Build with CMSIS-DSP and CMSIS-Core
  3. Quantize model to INT8 using TFLite converter

## Limitations
  - INT8/INT16 only (no FP32 acceleration)
  - Operator coverage narrower than full TFLite
  - Manual memory arena sizing required
```

---

## Quick Setup

```bash
claude mcp add --transport stdio arm-tinyml -- \
  uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-tinyml-mcp
```

See the full [Installation Guide](../installation) for other editors and clients.
