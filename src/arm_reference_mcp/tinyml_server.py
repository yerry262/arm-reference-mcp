"""ARM TinyML & Edge AI Deployment MCP Server.

Provides seven tools for deploying ML models on ARM microcontrollers and edge devices:
  - check_operator_support:      Check ML operator compatibility on ARM accelerators.
  - suggest_quantization:        Suggest quantization strategy for a model and target.
  - estimate_inference:          Estimate inference time and memory for a model on ARM.
  - compare_tinyml_targets:      Compare ARM TinyML hardware targets side by side.
  - generate_deployment_config:  Generate deployment configuration for ARM edge AI.
  - suggest_model_architecture:  Suggest ML model architectures for given memory constraints.
  - explain_tinyml_framework:    Explain TinyML/edge AI frameworks in detail.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "ARM TinyML & Edge AI",
    instructions="Plan and optimize ML model deployment on ARM Cortex-M, Cortex-A, Ethos NPU, and edge AI accelerators.",
)


# ---------------------------------------------------------------------------
# Tool 1: check_operator_support — operator compatibility matrix
# ---------------------------------------------------------------------------

OPERATOR_SUPPORT: dict[str, dict[str, dict]] = {
    "ethos-u55": {
        "accelerator": "Arm Ethos-U55 (microNPU)",
        "architecture": "Dedicated NPU for Cortex-M",
        "operators": {
            "conv2d": {"supported": True, "notes": "Fully accelerated. Supports 1x1, 3x3, 5x5, 7x7 kernels. INT8/INT16 only. Depthwise supported.", "constraints": "Max 8 input channels per MAC unit. Kernel size <= 65x65."},
            "depthwise_conv2d": {"supported": True, "notes": "Fully accelerated. Efficient for MobileNet-style architectures.", "constraints": "Depth multiplier must be 1. INT8/INT16 only."},
            "fully_connected": {"supported": True, "notes": "Accelerated via matrix multiply. INT8/INT16.", "constraints": "Max output features limited by SRAM tile size."},
            "max_pool2d": {"supported": True, "notes": "Accelerated. Fused with preceding Conv2D when possible.", "constraints": "Pool size <= 256x256. Stride <= 3."},
            "average_pool2d": {"supported": True, "notes": "Accelerated. Can be fused with Conv2D.", "constraints": "Pool size <= 256x256."},
            "add": {"supported": True, "notes": "Element-wise add, accelerated. Used in residual connections.", "constraints": "Both inputs must have same shape and quantization."},
            "relu": {"supported": True, "notes": "Fused with Conv2D/FC as activation. Zero overhead when fused.", "constraints": "None."},
            "relu6": {"supported": True, "notes": "Fused with Conv2D/FC. Used in MobileNetV2.", "constraints": "None."},
            "sigmoid": {"supported": True, "notes": "Accelerated via LUT (lookup table).", "constraints": "LUT size is fixed. INT8 only."},
            "tanh": {"supported": True, "notes": "Accelerated via LUT.", "constraints": "INT8 only."},
            "softmax": {"supported": True, "notes": "Accelerated. Typically last layer.", "constraints": "Single axis only. INT8."},
            "reshape": {"supported": True, "notes": "Zero-cost operation (metadata only).", "constraints": "None."},
            "concatenate": {"supported": True, "notes": "Accelerated via DMA.", "constraints": "All inputs same data type."},
            "transpose_conv2d": {"supported": True, "notes": "Accelerated. Used in decoder/upsampling networks.", "constraints": "INT8 only. Stride <= 2."},
            "resize_bilinear": {"supported": False, "notes": "NOT supported on Ethos-U55. Falls back to CPU.", "constraints": "Use nearest-neighbor resize or transpose convolution instead."},
            "lstm": {"supported": False, "notes": "NOT natively supported. Must be unrolled into supported ops (FC, Add, Sigmoid, Tanh).", "constraints": "Use unrolled LSTM or replace with 1D Conv."},
            "gelu": {"supported": False, "notes": "NOT supported. Use ReLU or approximate GELU with supported ops.", "constraints": "Replace with ReLU6 for microNPU deployment."},
            "layer_norm": {"supported": False, "notes": "NOT supported natively. Falls back to CPU.", "constraints": "Use batch normalization (fused into Conv) instead."},
            "matmul": {"supported": True, "notes": "Mapped to fully_connected operation.", "constraints": "INT8/INT16. Static shapes only."},
        },
    },
    "ethos-u65": {
        "accelerator": "Arm Ethos-U65 (microNPU)",
        "architecture": "Dedicated NPU for Cortex-M/A",
        "operators": {
            "conv2d": {"supported": True, "notes": "Fully accelerated with higher throughput than U55. 256/512 MAC configurations. INT8/INT16.", "constraints": "Kernel size <= 65x65. Dilation supported."},
            "depthwise_conv2d": {"supported": True, "notes": "Accelerated. Supports dilated depthwise convolutions.", "constraints": "INT8/INT16."},
            "fully_connected": {"supported": True, "notes": "Accelerated with higher parallelism than U55.", "constraints": "INT8/INT16."},
            "max_pool2d": {"supported": True, "notes": "Accelerated.", "constraints": "Pool size <= 256x256."},
            "average_pool2d": {"supported": True, "notes": "Accelerated.", "constraints": "Pool size <= 256x256."},
            "add": {"supported": True, "notes": "Element-wise add accelerated.", "constraints": "Same shape inputs."},
            "relu": {"supported": True, "notes": "Fused with Conv/FC.", "constraints": "None."},
            "relu6": {"supported": True, "notes": "Fused activation.", "constraints": "None."},
            "sigmoid": {"supported": True, "notes": "LUT-based acceleration.", "constraints": "INT8."},
            "tanh": {"supported": True, "notes": "LUT-based.", "constraints": "INT8."},
            "softmax": {"supported": True, "notes": "Accelerated.", "constraints": "INT8."},
            "reshape": {"supported": True, "notes": "Zero-cost.", "constraints": "None."},
            "concatenate": {"supported": True, "notes": "DMA-accelerated.", "constraints": "Same data type."},
            "transpose_conv2d": {"supported": True, "notes": "Accelerated with stride <= 2.", "constraints": "INT8."},
            "resize_bilinear": {"supported": True, "notes": "Supported on U65 (unlike U55). Accelerated.", "constraints": "Scale factor <= 8x. INT8."},
            "lstm": {"supported": False, "notes": "NOT natively supported. Unroll to supported ops.", "constraints": "Use FC+Sigmoid+Tanh decomposition."},
            "gelu": {"supported": False, "notes": "NOT supported. Approximate with supported ops.", "constraints": "Use ReLU/Sigmoid approximation."},
            "layer_norm": {"supported": False, "notes": "NOT natively supported.", "constraints": "Use fused batch norm."},
            "matmul": {"supported": True, "notes": "Mapped to FC.", "constraints": "INT8/INT16."},
        },
    },
    "cortex-m_cmsis_nn": {
        "accelerator": "Cortex-M CPU with CMSIS-NN",
        "architecture": "ARMv7E-M / ARMv8.1-M (DSP + Helium MVE)",
        "operators": {
            "conv2d": {"supported": True, "notes": "Optimized kernel using DSP instructions (SMLAD) or Helium MVE vectorization. INT8/INT16.", "constraints": "Performance depends on SRAM for im2col buffer."},
            "depthwise_conv2d": {"supported": True, "notes": "Optimized CMSIS-NN kernel. ~3x faster than naive implementation.", "constraints": "INT8. Depth multiplier=1 for optimized path."},
            "fully_connected": {"supported": True, "notes": "Optimized with DSP/MVE. SMLAD for INT8 dot products.", "constraints": "INT8/INT16."},
            "max_pool2d": {"supported": True, "notes": "Optimized kernel.", "constraints": "None."},
            "average_pool2d": {"supported": True, "notes": "Optimized kernel.", "constraints": "None."},
            "add": {"supported": True, "notes": "Element-wise, uses requantization.", "constraints": "INT8/INT16."},
            "relu": {"supported": True, "notes": "Fused with Conv/FC output activation. VMAX instruction on MVE.", "constraints": "None."},
            "relu6": {"supported": True, "notes": "Clamped activation. VMIN+VMAX on MVE.", "constraints": "None."},
            "sigmoid": {"supported": True, "notes": "LUT-based implementation in CMSIS-NN.", "constraints": "INT8. LUT stored in flash."},
            "tanh": {"supported": True, "notes": "LUT-based.", "constraints": "INT8."},
            "softmax": {"supported": True, "notes": "Optimized fixed-point implementation.", "constraints": "INT8."},
            "reshape": {"supported": True, "notes": "Zero-cost (pointer manipulation).", "constraints": "None."},
            "concatenate": {"supported": True, "notes": "Memory copy based.", "constraints": "None."},
            "transpose_conv2d": {"supported": True, "notes": "Supported but slower than Conv2D. No dedicated optimization.", "constraints": "INT8."},
            "resize_bilinear": {"supported": True, "notes": "CPU implementation. Not optimized for DSP.", "constraints": "Performance-intensive for large tensors."},
            "lstm": {"supported": True, "notes": "CMSIS-NN provides optimized LSTM kernel with INT8/INT16 support.", "constraints": "Gate operations fused. Peephole connections optional."},
            "gelu": {"supported": True, "notes": "Approximated via LUT or polynomial. Not hardware-accelerated.", "constraints": "Higher latency than ReLU."},
            "layer_norm": {"supported": True, "notes": "CPU implementation with DSP optimization.", "constraints": "INT8/INT16."},
            "matmul": {"supported": True, "notes": "Uses CMSIS-NN fully_connected kernel.", "constraints": "INT8/INT16."},
        },
    },
    "cortex-a_armnn": {
        "accelerator": "Cortex-A CPU with Arm NN / ACL",
        "architecture": "ARMv8-A / ARMv9-A (NEON, SVE2)",
        "operators": {
            "conv2d": {"supported": True, "notes": "NEON-optimized Winograd (3x3), im2col+GEMM, or direct convolution. FP32/FP16/INT8. SVE2 acceleration on ARMv9.", "constraints": "Winograd best for 3x3. Large kernels use im2col+GEMM."},
            "depthwise_conv2d": {"supported": True, "notes": "NEON-optimized. FP32/FP16/INT8.", "constraints": "None."},
            "fully_connected": {"supported": True, "notes": "NEON GEMM. FP32/FP16/INT8/BF16.", "constraints": "None."},
            "max_pool2d": {"supported": True, "notes": "NEON vectorized.", "constraints": "None."},
            "average_pool2d": {"supported": True, "notes": "NEON vectorized.", "constraints": "None."},
            "add": {"supported": True, "notes": "NEON element-wise.", "constraints": "Broadcasting supported."},
            "relu": {"supported": True, "notes": "NEON VMAX with zero. Near-zero overhead.", "constraints": "None."},
            "relu6": {"supported": True, "notes": "NEON VMIN+VMAX.", "constraints": "None."},
            "sigmoid": {"supported": True, "notes": "NEON polynomial approximation or LUT.", "constraints": "FP32/FP16."},
            "tanh": {"supported": True, "notes": "NEON polynomial approximation.", "constraints": "FP32/FP16."},
            "softmax": {"supported": True, "notes": "NEON optimized with exp approximation.", "constraints": "FP32/FP16/INT8."},
            "reshape": {"supported": True, "notes": "Zero-cost.", "constraints": "None."},
            "concatenate": {"supported": True, "notes": "Optimized memory copy.", "constraints": "None."},
            "transpose_conv2d": {"supported": True, "notes": "Supported. FP32/FP16/INT8.", "constraints": "None."},
            "resize_bilinear": {"supported": True, "notes": "NEON optimized. FP32/FP16.", "constraints": "None."},
            "lstm": {"supported": True, "notes": "Full LSTM support. FP32/FP16. CuDNN-style fused kernel.", "constraints": "None."},
            "gelu": {"supported": True, "notes": "FP32/FP16 with NEON approximation.", "constraints": "None."},
            "layer_norm": {"supported": True, "notes": "Supported. FP32/FP16.", "constraints": "None."},
            "matmul": {"supported": True, "notes": "NEON GEMM. FP32/FP16/INT8/BF16.", "constraints": "None."},
        },
    },
}

_VALID_TARGETS = set(OPERATOR_SUPPORT.keys())


@mcp.tool()
def check_operator_support(target: str, operators: str) -> str:
    """Check ML operator compatibility on an ARM accelerator or runtime.

    Reports which operators are supported, their constraints, and
    alternatives for unsupported operators.

    Args:
        target: ARM target platform. One of: "ethos-u55", "ethos-u65",
                "cortex-m_cmsis_nn", "cortex-a_armnn". Case-insensitive.
        operators: Comma-separated list of operator names to check
                   (e.g. "conv2d,relu,lstm,softmax,gelu").
    """
    target_key = target.lower().strip().replace(" ", "_")
    if target_key not in _VALID_TARGETS:
        return f"Error: target must be one of {', '.join(sorted(_VALID_TARGETS))}."

    op_list = [o.strip().lower() for o in operators.split(",") if o.strip()]
    if not op_list:
        return "Error: operators must be a non-empty comma-separated list."

    target_data = OPERATOR_SUPPORT[target_key]
    supported = []
    unsupported = []
    unknown = []

    for op in op_list:
        entry = target_data["operators"].get(op)
        if entry is None:
            unknown.append(op)
        elif entry["supported"]:
            supported.append((op, entry))
        else:
            unsupported.append((op, entry))

    lines = [f"# Operator Support: {target_data['accelerator']}"]
    lines.append(f"Architecture: {target_data['architecture']}")
    lines.append(f"Checked {len(op_list)} operators\n")

    lines.append("## Summary")
    lines.append(f"- Supported: {len(supported)}")
    lines.append(f"- NOT supported: {len(unsupported)}")
    lines.append(f"- Unknown: {len(unknown)}")
    lines.append("")

    if supported:
        lines.append("## Supported Operators")
        for op, entry in supported:
            lines.append(f"### {op}")
            lines.append(f"  {entry['notes']}")
            if entry["constraints"] and entry["constraints"] != "None.":
                lines.append(f"  Constraints: {entry['constraints']}")
            lines.append("")

    if unsupported:
        lines.append("## NOT Supported (need alternatives)")
        for op, entry in unsupported:
            lines.append(f"### {op}")
            lines.append(f"  {entry['notes']}")
            lines.append(f"  Constraints: {entry['constraints']}")
            lines.append("")

    if unknown:
        lines.append("## Unknown Operators")
        lines.append("These operators are not in the compatibility matrix:")
        for op in unknown:
            lines.append(f"  - {op}")
        lines.append("")

    # Compatibility score
    total_known = len(supported) + len(unsupported)
    if total_known > 0:
        score = (len(supported) * 100) / total_known
    else:
        score = 0
    lines.append(f"## Compatibility Score: {score:.0f}%")
    if score == 100:
        lines.append("All checked operators are supported on this target.")
    elif score >= 75:
        lines.append("Most operators supported. Replace unsupported ops for full acceleration.")
    elif score >= 50:
        lines.append("Partial support. Significant fallback to CPU expected.")
    else:
        lines.append("Low compatibility. Consider a different target or major model changes.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2: suggest_quantization — quantization strategy advisor
# ---------------------------------------------------------------------------

QUANTIZATION_STRATEGIES: dict[str, dict[str, dict]] = {
    "image_classification": {
        "model_type": "Image Classification (MobileNet, EfficientNet, ResNet)",
        "strategies": {
            "ethos-u55": {
                "recommended": "INT8 Post-Training Quantization (PTQ)",
                "precision": "INT8 (weights and activations)",
                "method": (
                    "1. Train model in FP32 using standard training pipeline.\n"
                    "2. Use TensorFlow Lite converter with representative dataset (100-500 images).\n"
                    "3. Apply full integer quantization (--inference_type=INT8).\n"
                    "4. Use Vela compiler to optimize for Ethos-U55.\n\n"
                    "Accuracy impact: typically <1% top-1 accuracy drop for MobileNetV2.\n"
                    "Latency improvement: 5-10x vs FP32 on Cortex-M CPU."
                ),
                "tools": ["TensorFlow Lite converter", "Vela compiler (ethos-u-vela)"],
                "tips": "Use per-channel quantization for Conv2D weights. Calibrate with representative data from the deployment domain. Avoid quantizing the first and last layers if accuracy is critical.",
            },
            "ethos-u65": {
                "recommended": "INT8 PTQ or INT16 for sensitive layers",
                "precision": "INT8 primary, INT16 for sensitive layers",
                "method": (
                    "1. Start with full INT8 PTQ.\n"
                    "2. If accuracy drops >2%, use mixed INT8/INT16:\n"
                    "   - Keep most Conv2D layers as INT8.\n"
                    "   - Use INT16 for layers with high sensitivity (first conv, attention layers).\n"
                    "3. Vela compiler handles mixed precision automatically."
                ),
                "tools": ["TensorFlow Lite converter", "Vela compiler"],
                "tips": "U65 supports INT16 with minimal performance overhead vs U55. Use INT16 for layers where quantization error is >5% per-layer.",
            },
            "cortex-m_cmsis_nn": {
                "recommended": "INT8 Quantization-Aware Training (QAT)",
                "precision": "INT8 (symmetric for weights, asymmetric for activations)",
                "method": (
                    "1. Start with pre-trained FP32 model.\n"
                    "2. Apply QAT using TensorFlow Model Optimization Toolkit.\n"
                    "3. Fine-tune for 10-20% of original training epochs.\n"
                    "4. Convert to TFLite INT8 model.\n"
                    "5. Use CMSIS-NN backend for TFLite Micro.\n\n"
                    "Accuracy: QAT typically recovers accuracy to within 0.5% of FP32.\n"
                    "Memory: INT8 model is 4x smaller than FP32."
                ),
                "tools": ["TF Model Optimization Toolkit", "TFLite Micro", "CMSIS-NN"],
                "tips": "For Cortex-M with Helium MVE (M55/M85), INT8 dot-product instructions give 4x throughput vs non-MVE cores. Consider MobileNetV2 0.35x width for ultra-constrained targets (<256KB SRAM).",
            },
            "cortex-a_armnn": {
                "recommended": "FP16 or INT8 PTQ",
                "precision": "FP16 for quality, INT8 for speed",
                "method": (
                    "1. FP16: simply convert FP32 model to FP16 (no calibration needed).\n"
                    "   - 2x memory savings, ~1.5-2x speedup on NEON.\n"
                    "   - Negligible accuracy impact for most models.\n"
                    "2. INT8 PTQ: use calibration dataset for activation range estimation.\n"
                    "   - 4x memory savings, ~2-4x speedup.\n"
                    "   - May need QAT for <1% accuracy target."
                ),
                "tools": ["Arm NN SDK", "TFLite", "ONNX Runtime", "PyTorch quantization"],
                "tips": "Cortex-A with BF16 support (ARMv8.6+) allows BF16 inference with near-FP32 accuracy. SVE2 on ARMv9 provides additional vectorization for quantized ops.",
            },
        },
    },
    "object_detection": {
        "model_type": "Object Detection (SSD-MobileNet, YOLO, EfficientDet)",
        "strategies": {
            "ethos-u55": {
                "recommended": "INT8 PTQ with careful NMS handling",
                "precision": "INT8",
                "method": (
                    "1. Train detection model in FP32.\n"
                    "2. Apply INT8 PTQ with representative images.\n"
                    "3. NMS (Non-Max Suppression) must run on CPU (not accelerated on U55).\n"
                    "4. Keep detection head (box regression) in higher precision if mAP drops.\n"
                    "5. Compile backbone with Vela, NMS runs on Cortex-M CPU."
                ),
                "tools": ["TFLite converter", "Vela compiler", "CMSIS-NN (for NMS)"],
                "tips": "SSD-MobileNetV2 is the recommended architecture for Ethos-U55. YOLO variants may have unsupported ops (resize, concat patterns). Use <320x320 input resolution for real-time.",
            },
            "ethos-u65": {
                "recommended": "INT8 PTQ",
                "precision": "INT8",
                "method": (
                    "1. Similar to U55 but supports more ops (resize_bilinear).\n"
                    "2. Higher throughput allows larger models and input resolutions.\n"
                    "3. NMS still on CPU but U65 handles more of the post-processing."
                ),
                "tools": ["TFLite converter", "Vela compiler"],
                "tips": "U65 can handle EfficientDet-Lite0 at reasonable framerates. Consider 256 MAC config for real-time detection at 320x320.",
            },
            "cortex-m_cmsis_nn": {
                "recommended": "INT8 QAT",
                "precision": "INT8",
                "method": (
                    "1. Use QAT to maintain mAP within 1-2% of FP32.\n"
                    "2. Detection models are more sensitive to quantization than classifiers.\n"
                    "3. Box regression layers benefit most from QAT fine-tuning.\n"
                    "4. Deploy with TFLite Micro + CMSIS-NN."
                ),
                "tools": ["TF Model Optimization Toolkit", "TFLite Micro", "CMSIS-NN"],
                "tips": "Person detection at 96x96 is feasible on Cortex-M4 (256KB SRAM). Full SSD-MobileNet needs Cortex-M7/M55 with >=512KB SRAM.",
            },
            "cortex-a_armnn": {
                "recommended": "FP16 or INT8 depending on latency budget",
                "precision": "FP16 for quality, INT8 for real-time",
                "method": (
                    "1. FP16: good for single-shot detection with relaxed latency.\n"
                    "2. INT8: required for real-time multi-object detection.\n"
                    "3. Use Arm NN or ONNX Runtime for inference.\n"
                    "4. NMS can run efficiently on Cortex-A NEON."
                ),
                "tools": ["Arm NN", "ONNX Runtime", "TFLite"],
                "tips": "YOLOv8n runs real-time on Cortex-A78 at INT8. Use multi-threaded inference (4 cores) for best throughput.",
            },
        },
    },
    "keyword_spotting": {
        "model_type": "Keyword Spotting / Audio Classification (DS-CNN, DSCNN, MicroNet)",
        "strategies": {
            "ethos-u55": {
                "recommended": "INT8 PTQ",
                "precision": "INT8",
                "method": (
                    "1. Train DS-CNN model on MFCC features.\n"
                    "2. Apply INT8 PTQ — keyword spotting models are robust to quantization.\n"
                    "3. Typical accuracy drop: <0.5% for 12-keyword task.\n"
                    "4. Compile with Vela for Ethos-U55."
                ),
                "tools": ["TFLite converter", "Vela compiler"],
                "tips": "DS-CNN (Depthwise Separable CNN) is ideal for Ethos-U55 — all ops are fully accelerated. Model size typically <100KB. Inference <10ms on U55.",
            },
            "ethos-u65": {
                "recommended": "INT8 PTQ",
                "precision": "INT8",
                "method": "Same as U55 — keyword spotting models are small and well-supported.",
                "tools": ["TFLite converter", "Vela compiler"],
                "tips": "U65 is overkill for keyword spotting. Consider U55 for cost optimization.",
            },
            "cortex-m_cmsis_nn": {
                "recommended": "INT8 PTQ",
                "precision": "INT8",
                "method": (
                    "1. Standard INT8 PTQ with representative audio clips.\n"
                    "2. CMSIS-NN provides optimized kernels for DS-CNN layers.\n"
                    "3. Runs efficiently on Cortex-M4 (no Helium needed).\n"
                    "4. Total inference: ~20ms on Cortex-M4 @ 80MHz."
                ),
                "tools": ["TFLite Micro", "CMSIS-NN", "CMSIS-DSP (for MFCC)"],
                "tips": "Use CMSIS-DSP for MFCC feature extraction — optimized FFT/DCT with DSP instructions. Total pipeline (MFCC+inference) fits in <64KB SRAM on Cortex-M4.",
            },
            "cortex-a_armnn": {
                "recommended": "FP32 or INT8 — models are small enough",
                "precision": "FP32 (model is tiny, quantization not needed for speed)",
                "method": (
                    "1. FP32 inference is fast enough — model is <500KB.\n"
                    "2. Inference time: <1ms on any Cortex-A core.\n"
                    "3. Quantize only if running alongside other heavy workloads."
                ),
                "tools": ["TFLite", "ONNX Runtime"],
                "tips": "On Cortex-A, the bottleneck is audio capture and MFCC, not inference. Focus optimization on the audio pipeline.",
            },
        },
    },
    "anomaly_detection": {
        "model_type": "Anomaly Detection (Autoencoder, Isolation Forest, DCASE-style)",
        "strategies": {
            "ethos-u55": {
                "recommended": "INT8 PTQ",
                "precision": "INT8",
                "method": (
                    "1. Train autoencoder in FP32 on normal data.\n"
                    "2. Apply INT8 PTQ with representative normal samples.\n"
                    "3. Reconstruction error threshold needs recalibration after quantization.\n"
                    "4. Compile encoder+decoder with Vela."
                ),
                "tools": ["TFLite converter", "Vela compiler"],
                "tips": "Simple FC-based autoencoders work best on U55. Conv1D autoencoders for vibration data are also well-supported. Ensure the threshold for anomaly detection is recalibrated on the INT8 model.",
            },
            "ethos-u65": {
                "recommended": "INT8 PTQ",
                "precision": "INT8",
                "method": "Same as U55. U65 overkill for most anomaly detection models.",
                "tools": ["TFLite converter", "Vela compiler"],
                "tips": "Use U65 only if combining anomaly detection with other inference tasks.",
            },
            "cortex-m_cmsis_nn": {
                "recommended": "INT8 PTQ",
                "precision": "INT8",
                "method": (
                    "1. INT8 PTQ with normal operation samples.\n"
                    "2. Small FC autoencoder (e.g., 640->128->32->128->640) fits easily.\n"
                    "3. Total model size: <20KB. Inference: <5ms on Cortex-M4."
                ),
                "tools": ["TFLite Micro", "CMSIS-NN"],
                "tips": "For vibration-based anomaly detection (DCASE), use MFCC or mel-spectrogram features with CMSIS-DSP. Autoencoder reconstruction error is the anomaly score.",
            },
            "cortex-a_armnn": {
                "recommended": "FP32",
                "precision": "FP32 (models are small)",
                "method": "FP32 inference. Models are typically <1MB. Quantization unnecessary.",
                "tools": ["TFLite", "ONNX Runtime", "scikit-learn (for classical methods)"],
                "tips": "On Cortex-A, consider more sophisticated methods: Isolation Forest, one-class SVM. These run fast enough without acceleration.",
            },
        },
    },
    "nlp_embedding": {
        "model_type": "NLP / Text Embedding (DistilBERT, TinyBERT, MobileBERT)",
        "strategies": {
            "ethos-u55": {
                "recommended": "NOT recommended for Transformer models",
                "precision": "N/A",
                "method": (
                    "Transformer models use operations not well-supported on Ethos-U55:\n"
                    "- Layer normalization: falls back to CPU.\n"
                    "- GELU activation: not supported.\n"
                    "- Multi-head attention (MatMul + Softmax patterns): partially accelerated.\n\n"
                    "Consider using a simpler architecture (CNN-based text classifier) for U55."
                ),
                "tools": ["TFLite converter", "Vela compiler (partial acceleration)"],
                "tips": "Replace Transformer with 1D CNN + Global Average Pooling for text classification on Ethos-U55. If Transformer is required, use Ethos-U65 or Cortex-A.",
            },
            "ethos-u65": {
                "recommended": "INT8 PTQ with partial acceleration",
                "precision": "INT8",
                "method": (
                    "1. Distill from larger model (BERT -> TinyBERT/DistilBERT).\n"
                    "2. Apply INT8 PTQ. Attention MatMuls are accelerated.\n"
                    "3. LayerNorm and GELU fall back to CPU.\n"
                    "4. ~60-70% of computation runs on NPU."
                ),
                "tools": ["TFLite converter", "Vela compiler"],
                "tips": "TinyBERT (4-layer, 312-hidden) is the sweet spot for U65. Full DistilBERT is too large for most embedded memory budgets.",
            },
            "cortex-m_cmsis_nn": {
                "recommended": "INT8 QAT with model distillation",
                "precision": "INT8",
                "method": (
                    "1. Distill to very small model (2-layer, 128-hidden Transformer).\n"
                    "2. QAT to recover accuracy lost during distillation + quantization.\n"
                    "3. Deploy with TFLite Micro.\n"
                    "4. Requires >512KB SRAM even for tiny Transformers."
                ),
                "tools": ["TFLite Micro", "CMSIS-NN"],
                "tips": "Transformer on Cortex-M is extremely constrained. Consider CNN-based alternatives (TextCNN, CharCNN) which map better to CMSIS-NN kernels. Helium MVE (M55/M85) makes Transformers more feasible.",
            },
            "cortex-a_armnn": {
                "recommended": "FP16 or INT8 dynamic quantization",
                "precision": "FP16 for quality, INT8 for throughput",
                "method": (
                    "1. FP16: automatic conversion, minimal accuracy loss.\n"
                    "2. INT8 dynamic quantization: quantize weights, keep activations FP32.\n"
                    "3. INT8 static quantization: best performance, needs calibration.\n"
                    "4. Use ONNX Runtime or PyTorch for Transformer inference."
                ),
                "tools": ["ONNX Runtime", "PyTorch Mobile", "Arm NN"],
                "tips": "DistilBERT runs at ~50ms per inference on Cortex-A78 (INT8, 4 cores). MobileBERT is optimized for ARM with bottleneck attention. BF16 on ARMv8.6+ gives near-FP32 quality at FP16 speed.",
            },
        },
    },
}

_VALID_MODEL_TYPES = set(QUANTIZATION_STRATEGIES.keys())


@mcp.tool()
def suggest_quantization(model_type: str, target: str) -> str:
    """Suggest a quantization strategy for deploying an ML model on an ARM target.

    Returns the recommended quantization approach, precision, step-by-step method,
    tools to use, and practical tips.

    Args:
        model_type: Type of ML model. One of: "image_classification",
                    "object_detection", "keyword_spotting", "anomaly_detection",
                    "nlp_embedding". Case-insensitive.
        target: ARM target platform. One of: "ethos-u55", "ethos-u65",
                "cortex-m_cmsis_nn", "cortex-a_armnn". Case-insensitive.
    """
    model_key = model_type.lower().strip().replace(" ", "_").replace("-", "_")
    target_key = target.lower().strip().replace(" ", "_")

    if model_key not in _VALID_MODEL_TYPES:
        return f"Error: model_type must be one of {', '.join(sorted(_VALID_MODEL_TYPES))}."
    if target_key not in _VALID_TARGETS:
        return f"Error: target must be one of {', '.join(sorted(_VALID_TARGETS))}."

    model_data = QUANTIZATION_STRATEGIES[model_key]
    strategy = model_data["strategies"].get(target_key)
    if strategy is None:
        return f"Error: no quantization strategy found for {model_type} on {target}."

    lines = [f"# Quantization Strategy"]
    lines.append(f"Model: {model_data['model_type']}")
    lines.append(f"Target: {OPERATOR_SUPPORT[target_key]['accelerator']}")
    lines.append("")
    lines.append(f"## Recommended: {strategy['recommended']}")
    lines.append(f"Precision: {strategy['precision']}")
    lines.append(f"\n## Method")
    lines.append(strategy["method"])
    lines.append(f"\n## Tools")
    for tool in strategy["tools"]:
        lines.append(f"- {tool}")
    lines.append(f"\n## Tips")
    lines.append(strategy["tips"])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: estimate_inference — inference estimation
# ---------------------------------------------------------------------------

INFERENCE_ESTIMATES: dict[str, dict[str, dict]] = {
    "mobilenetv2_1.0_224": {
        "model": "MobileNetV2 1.0 224x224",
        "params": "3.4M",
        "macs": "300M MACs",
        "model_size_int8": "3.4 MB",
        "model_size_fp32": "13.6 MB",
        "targets": {
            "ethos-u55": {
                "inference_ms": 45.0,
                "clock_mhz": 500,
                "sram_kb": 512,
                "flash_kb": 3400,
                "notes": "128 MAC config at 500 MHz. Majority of ops on NPU. Depthwise convolutions fully accelerated.",
            },
            "ethos-u65": {
                "inference_ms": 12.0,
                "clock_mhz": 500,
                "sram_kb": 512,
                "flash_kb": 3400,
                "notes": "256 MAC config at 500 MHz. ~3.5x faster than U55 for this model.",
            },
            "cortex-m55_cmsis_nn": {
                "inference_ms": 150.0,
                "clock_mhz": 400,
                "sram_kb": 512,
                "flash_kb": 3400,
                "notes": "Cortex-M55 with Helium MVE at 400 MHz. INT8 with CMSIS-NN optimized kernels.",
            },
            "cortex-m7_cmsis_nn": {
                "inference_ms": 800.0,
                "clock_mhz": 480,
                "sram_kb": 512,
                "flash_kb": 3400,
                "notes": "Cortex-M7 at 480 MHz. INT8 with CMSIS-NN. Limited by DSP throughput (no MVE).",
            },
            "cortex-a55_armnn": {
                "inference_ms": 25.0,
                "clock_mhz": 2000,
                "sram_kb": 0,
                "flash_kb": 0,
                "notes": "Cortex-A55 single core at 2 GHz. INT8 with Arm NN. NEON optimized.",
            },
            "cortex-a78_armnn": {
                "inference_ms": 8.0,
                "clock_mhz": 3000,
                "sram_kb": 0,
                "flash_kb": 0,
                "notes": "Cortex-A78 single core at 3 GHz. INT8 with Arm NN. Big core advantage.",
            },
        },
    },
    "mobilenetv2_0.35_96": {
        "model": "MobileNetV2 0.35 96x96",
        "params": "0.4M",
        "macs": "7M MACs",
        "model_size_int8": "0.4 MB",
        "model_size_fp32": "1.6 MB",
        "targets": {
            "ethos-u55": {
                "inference_ms": 2.5,
                "clock_mhz": 500,
                "sram_kb": 128,
                "flash_kb": 400,
                "notes": "Ultra-fast on U55. Ideal for always-on visual wake word.",
            },
            "ethos-u65": {
                "inference_ms": 1.0,
                "clock_mhz": 500,
                "sram_kb": 128,
                "flash_kb": 400,
                "notes": "Sub-millisecond inference.",
            },
            "cortex-m55_cmsis_nn": {
                "inference_ms": 8.0,
                "clock_mhz": 400,
                "sram_kb": 128,
                "flash_kb": 400,
                "notes": "Very fast on M55 Helium. Good for always-on applications.",
            },
            "cortex-m4_cmsis_nn": {
                "inference_ms": 50.0,
                "clock_mhz": 80,
                "sram_kb": 128,
                "flash_kb": 400,
                "notes": "Cortex-M4 at 80 MHz with CMSIS-NN. Feasible for <1 FPS applications.",
            },
            "cortex-a55_armnn": {
                "inference_ms": 0.5,
                "clock_mhz": 2000,
                "sram_kb": 0,
                "flash_kb": 0,
                "notes": "Trivially fast on Cortex-A. Sub-millisecond.",
            },
        },
    },
    "ds_cnn_s": {
        "model": "DS-CNN Small (Keyword Spotting)",
        "params": "24K",
        "macs": "5.4M MACs",
        "model_size_int8": "24 KB",
        "model_size_fp32": "96 KB",
        "targets": {
            "ethos-u55": {
                "inference_ms": 0.5,
                "clock_mhz": 500,
                "sram_kb": 32,
                "flash_kb": 24,
                "notes": "Near-instant inference on U55. Ideal for keyword spotting.",
            },
            "cortex-m55_cmsis_nn": {
                "inference_ms": 3.0,
                "clock_mhz": 400,
                "sram_kb": 32,
                "flash_kb": 24,
                "notes": "Very fast on M55. Leaves CPU headroom for audio processing.",
            },
            "cortex-m4_cmsis_nn": {
                "inference_ms": 20.0,
                "clock_mhz": 80,
                "sram_kb": 32,
                "flash_kb": 24,
                "notes": "Cortex-M4 at 80 MHz. Well within real-time for keyword spotting (1s windows).",
            },
            "cortex-a55_armnn": {
                "inference_ms": 0.1,
                "clock_mhz": 2000,
                "sram_kb": 0,
                "flash_kb": 0,
                "notes": "Negligible latency on Cortex-A.",
            },
        },
    },
    "yolov8n_320": {
        "model": "YOLOv8 Nano 320x320",
        "params": "3.2M",
        "macs": "4.4G MACs",
        "model_size_int8": "3.2 MB",
        "model_size_fp32": "12.8 MB",
        "targets": {
            "ethos-u65": {
                "inference_ms": 80.0,
                "clock_mhz": 500,
                "sram_kb": 1024,
                "flash_kb": 3200,
                "notes": "256 MAC config. ~12 FPS. Some ops (SiLU, Concat) partially on CPU.",
            },
            "cortex-m55_cmsis_nn": {
                "inference_ms": 2000.0,
                "clock_mhz": 400,
                "sram_kb": 1024,
                "flash_kb": 3200,
                "notes": "Very slow on pure CPU. Not recommended without NPU.",
            },
            "cortex-a55_armnn": {
                "inference_ms": 85.0,
                "clock_mhz": 2000,
                "sram_kb": 0,
                "flash_kb": 0,
                "notes": "Cortex-A55 single core. INT8. ~12 FPS.",
            },
            "cortex-a78_armnn": {
                "inference_ms": 28.0,
                "clock_mhz": 3000,
                "sram_kb": 0,
                "flash_kb": 0,
                "notes": "Cortex-A78 single core. INT8. ~35 FPS. Multi-core: ~70+ FPS.",
            },
        },
    },
    "autoencoder_small": {
        "model": "FC Autoencoder (Anomaly Detection, 640-128-32-128-640)",
        "params": "120K",
        "macs": "0.24M MACs",
        "model_size_int8": "120 KB",
        "model_size_fp32": "480 KB",
        "targets": {
            "ethos-u55": {
                "inference_ms": 0.2,
                "clock_mhz": 500,
                "sram_kb": 16,
                "flash_kb": 120,
                "notes": "Near-instant on U55. FC layers fully accelerated.",
            },
            "cortex-m4_cmsis_nn": {
                "inference_ms": 3.0,
                "clock_mhz": 80,
                "sram_kb": 16,
                "flash_kb": 120,
                "notes": "Very fast even on low-end Cortex-M4. Ideal for vibration/audio anomaly detection.",
            },
            "cortex-m55_cmsis_nn": {
                "inference_ms": 0.5,
                "clock_mhz": 400,
                "sram_kb": 16,
                "flash_kb": 120,
                "notes": "Trivially fast on M55.",
            },
            "cortex-a55_armnn": {
                "inference_ms": 0.05,
                "clock_mhz": 2000,
                "sram_kb": 0,
                "flash_kb": 0,
                "notes": "Negligible latency.",
            },
        },
    },
}


@mcp.tool()
def estimate_inference(model_name: str, target: str) -> str:
    """Estimate inference time and memory requirements for a model on an ARM target.

    Returns estimated latency, memory requirements, and deployment notes.

    Args:
        model_name: Model to estimate. One of: "mobilenetv2_1.0_224",
                    "mobilenetv2_0.35_96", "ds_cnn_s", "yolov8n_320",
                    "autoencoder_small". Case-insensitive.
        target: ARM target platform (e.g. "ethos-u55", "cortex-m55_cmsis_nn",
                "cortex-a78_armnn"). Case-insensitive.
    """
    model_key = model_name.lower().strip().replace(" ", "_").replace("-", "_")

    # Try alias matching
    model_aliases = {
        "mobilenetv2": "mobilenetv2_1.0_224",
        "mobilenet": "mobilenetv2_1.0_224",
        "mobilenetv2_small": "mobilenetv2_0.35_96",
        "ds_cnn": "ds_cnn_s",
        "dscnn": "ds_cnn_s",
        "keyword_spotting": "ds_cnn_s",
        "yolov8": "yolov8n_320",
        "yolo": "yolov8n_320",
        "autoencoder": "autoencoder_small",
        "anomaly": "autoencoder_small",
    }
    model_key = model_aliases.get(model_key, model_key)

    model_data = INFERENCE_ESTIMATES.get(model_key)
    if model_data is None:
        available = sorted(INFERENCE_ESTIMATES.keys())
        return (
            f"No inference data for model '{model_name}'.\n\n"
            f"Available models: {', '.join(available)}\n"
            f"Also accepts aliases: mobilenet, ds_cnn, yolo, autoencoder."
        )

    target_key = target.lower().strip().replace(" ", "_")
    target_data = model_data["targets"].get(target_key)
    if target_data is None:
        available = sorted(model_data["targets"].keys())
        return (
            f"No inference data for '{model_name}' on target '{target}'.\n\n"
            f"Available targets for this model: {', '.join(available)}"
        )

    lines = [f"# Inference Estimate: {model_data['model']}"]
    lines.append(f"Parameters: {model_data['params']}  |  Compute: {model_data['macs']}")
    lines.append(f"Model size: {model_data['model_size_int8']} (INT8) / {model_data['model_size_fp32']} (FP32)")
    lines.append(f"\n## Target: {target_key}")
    lines.append(f"Clock: {target_data['clock_mhz']} MHz")
    lines.append(f"**Estimated inference time: {target_data['inference_ms']:.1f} ms**")

    fps = 1000.0 / target_data["inference_ms"] if target_data["inference_ms"] > 0 else 0
    lines.append(f"**Estimated throughput: {fps:.1f} FPS**")

    if target_data["sram_kb"] > 0:
        lines.append(f"\n## Memory Requirements")
        lines.append(f"- SRAM (working memory): {target_data['sram_kb']} KB minimum")
        lines.append(f"- Flash (model storage): {target_data['flash_kb']} KB")
    else:
        lines.append(f"\n## Memory")
        lines.append(f"- Model loaded into system RAM. No SRAM/Flash constraints.")

    lines.append(f"\n## Notes")
    lines.append(target_data["notes"])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: compare_tinyml_targets — hardware comparison
# ---------------------------------------------------------------------------

TINYML_TARGETS: dict[str, dict] = {
    "cortex-m4": {
        "name": "ARM Cortex-M4",
        "category": "microcontroller",
        "architecture": "ARMv7E-M",
        "clock_range": "48-240 MHz",
        "typical_clock": "80-168 MHz",
        "dsp_support": "Single-cycle 32-bit MAC, dual 16-bit MAC (SMLAD)",
        "simd_support": "None (DSP extensions only)",
        "ml_accelerator": "None",
        "typical_sram": "64-512 KB",
        "typical_flash": "256 KB - 2 MB",
        "fp_support": "Optional FPv4-SP (single-precision)",
        "power_profile": "10-100 mW (active), <10 uW (deep sleep)",
        "ml_frameworks": ["TFLite Micro + CMSIS-NN", "CMSIS-NN standalone", "Edge Impulse"],
        "best_for": "Keyword spotting, simple anomaly detection, sensor fusion. Models <500KB with <256KB SRAM.",
        "example_devices": "STM32F4, STM32L4, nRF52840, NXP LPC54xxx",
        "cost_range": "$1-5 per unit",
    },
    "cortex-m7": {
        "name": "ARM Cortex-M7",
        "category": "microcontroller",
        "architecture": "ARMv7E-M",
        "clock_range": "200-600 MHz",
        "typical_clock": "480 MHz",
        "dsp_support": "Dual-issue MAC, faster than M4",
        "simd_support": "None (DSP extensions only)",
        "ml_accelerator": "None",
        "typical_sram": "256 KB - 1 MB (+ external SDRAM option)",
        "typical_flash": "1-2 MB (+ external QSPI option)",
        "fp_support": "FPv5 double-precision",
        "power_profile": "50-300 mW (active)",
        "ml_frameworks": ["TFLite Micro + CMSIS-NN", "CMSIS-NN", "Edge Impulse", "STM32Cube.AI"],
        "best_for": "Image classification (small), object detection (tiny), audio classification. Models <2MB.",
        "example_devices": "STM32H7, NXP i.MX RT1060/1170, Renesas RA6M5",
        "cost_range": "$3-10 per unit",
    },
    "cortex-m55": {
        "name": "ARM Cortex-M55",
        "category": "microcontroller",
        "architecture": "ARMv8.1-M",
        "clock_range": "200-400 MHz",
        "typical_clock": "400 MHz",
        "dsp_support": "Full DSP + MVE (Helium)",
        "simd_support": "Helium MVE: 128-bit vector, INT8/INT16/INT32/FP16/FP32",
        "ml_accelerator": "None (Helium MVE provides 4-8x ML speedup vs M4)",
        "typical_sram": "256 KB - 4 MB",
        "typical_flash": "1-4 MB",
        "fp_support": "FPv5 + FP16 (via MVE)",
        "power_profile": "30-150 mW (active)",
        "ml_frameworks": ["TFLite Micro + CMSIS-NN (Helium-optimized)", "Edge Impulse", "Arm Virtual Hardware"],
        "best_for": "Image classification, keyword spotting, anomaly detection, small object detection. Best pure-CPU ML performance in Cortex-M.",
        "example_devices": "Arm Corstone-300 (reference), Alif Ensemble E7, Samsung Exynos W920",
        "cost_range": "$5-15 per unit",
    },
    "cortex-m85": {
        "name": "ARM Cortex-M85",
        "category": "microcontroller",
        "architecture": "ARMv8.1-M",
        "clock_range": "200-600 MHz",
        "typical_clock": "600 MHz",
        "dsp_support": "Full DSP + Helium MVE",
        "simd_support": "Helium MVE: 128-bit vector, INT8/INT16/INT32/FP16/FP32",
        "ml_accelerator": "None (highest-performance Cortex-M CPU for ML)",
        "typical_sram": "512 KB - 4 MB",
        "typical_flash": "2-8 MB",
        "fp_support": "FPv5 + FP16 (via MVE)",
        "power_profile": "50-200 mW (active)",
        "ml_frameworks": ["TFLite Micro + CMSIS-NN", "Edge Impulse", "Arm Virtual Hardware"],
        "best_for": "Largest ML models on Cortex-M. Small Transformers, complex CNNs. Bridge between M-class and A-class.",
        "example_devices": "Arm Corstone-310 (reference), Renesas RA8M1",
        "cost_range": "$8-20 per unit",
    },
    "ethos-u55": {
        "name": "Arm Ethos-U55 (microNPU)",
        "category": "npu",
        "architecture": "Dedicated microNPU (paired with Cortex-M55/M85)",
        "clock_range": "250-500 MHz",
        "typical_clock": "500 MHz",
        "dsp_support": "N/A (dedicated NPU)",
        "simd_support": "N/A (fixed-function ML accelerator)",
        "ml_accelerator": "32/64/128/256 MAC units. INT8/INT16. Up to 512 GOPS (128 MAC @ 500 MHz).",
        "typical_sram": "Shares with host CPU (256 KB - 4 MB)",
        "typical_flash": "Shares with host CPU",
        "fp_support": "N/A (INT8/INT16 only)",
        "power_profile": "5-50 mW (NPU active)",
        "ml_frameworks": ["TFLite Micro + Vela compiler", "Edge Impulse", "Arm Virtual Hardware"],
        "best_for": "Always-on ML: keyword spotting, visual wake word, gesture recognition, anomaly detection. Best performance/watt for INT8 CNNs.",
        "example_devices": "Arm Corstone-300 (M55+U55), NXP i.MX 93 (A55+U65)",
        "cost_range": "Included with SoC ($5-20 SoC cost)",
    },
    "ethos-u65": {
        "name": "Arm Ethos-U65 (microNPU)",
        "category": "npu",
        "architecture": "Dedicated microNPU (paired with Cortex-M55/M85 or Cortex-A)",
        "clock_range": "250-1000 MHz",
        "typical_clock": "500 MHz",
        "dsp_support": "N/A (dedicated NPU)",
        "simd_support": "N/A (fixed-function ML accelerator)",
        "ml_accelerator": "256/512 MAC units. INT8/INT16. Up to 1 TOPS (512 MAC @ 1 GHz).",
        "typical_sram": "Shared or dedicated (512 KB - 4 MB)",
        "typical_flash": "Shared",
        "fp_support": "N/A (INT8/INT16 only)",
        "power_profile": "50-200 mW (NPU active)",
        "ml_frameworks": ["TFLite Micro + Vela compiler", "Arm NN", "Edge Impulse"],
        "best_for": "Object detection, image classification, pose estimation, face detection. Higher throughput than U55 for larger models.",
        "example_devices": "Arm Corstone-310, NXP i.MX 93, Samsung Exynos",
        "cost_range": "Included with SoC ($10-30 SoC cost)",
    },
    "cortex-a55": {
        "name": "ARM Cortex-A55",
        "category": "application_processor",
        "architecture": "ARMv8.2-A",
        "clock_range": "1.0-2.0 GHz",
        "typical_clock": "1.8 GHz",
        "dsp_support": "Full AArch64 DSP",
        "simd_support": "NEON/ASIMD 128-bit, optional DotProd (INT8 dot product)",
        "ml_accelerator": "None (NEON provides ML acceleration)",
        "typical_sram": "N/A (uses system DRAM, 512 MB - 4 GB)",
        "typical_flash": "N/A (uses eMMC/UFS storage)",
        "fp_support": "FP16, FP32, FP64",
        "power_profile": "200-500 mW per core",
        "ml_frameworks": ["Arm NN", "TFLite", "ONNX Runtime", "PyTorch Mobile", "XNNPACK"],
        "best_for": "Edge AI gateway, smart camera, mobile inference. MobileNet, EfficientNet, small YOLO. Multi-model pipelines.",
        "example_devices": "Raspberry Pi 4/5, i.MX 8M, RK3568, MT8183",
        "cost_range": "$5-30 per SoC",
    },
    "cortex-a78": {
        "name": "ARM Cortex-A78",
        "category": "application_processor",
        "architecture": "ARMv8.2-A",
        "clock_range": "2.0-3.0 GHz",
        "typical_clock": "2.8 GHz",
        "dsp_support": "Full AArch64 DSP",
        "simd_support": "NEON/ASIMD 128-bit, DotProd, optional FP16",
        "ml_accelerator": "None (NEON, high clock speed)",
        "typical_sram": "N/A (uses system DRAM)",
        "typical_flash": "N/A",
        "fp_support": "FP16, FP32, FP64",
        "power_profile": "500 mW - 1.5 W per core",
        "ml_frameworks": ["Arm NN", "TFLite", "ONNX Runtime", "PyTorch Mobile"],
        "best_for": "High-performance edge AI. Real-time object detection, NLP inference, multi-model serving. Can handle DistilBERT, YOLOv8.",
        "example_devices": "Qualcomm Snapdragon 888, MediaTek Dimensity, Samsung Exynos 2100",
        "cost_range": "$15-50 per SoC (in mobile SoCs)",
    },
}


@mcp.tool()
def compare_tinyml_targets(target_a: str, target_b: str) -> str:
    """Compare two ARM TinyML hardware targets side by side.

    Shows architecture, performance, memory, power, supported frameworks,
    and best-use recommendations for each target.

    Args:
        target_a: First ARM target (e.g. "cortex-m55", "ethos-u55",
                  "cortex-a55"). Case-insensitive.
        target_b: Second ARM target to compare against.
    """
    key_a = target_a.lower().strip().replace(" ", "-")
    key_b = target_b.lower().strip().replace(" ", "-")

    # Try with underscores and hyphens
    data_a = TINYML_TARGETS.get(key_a) or TINYML_TARGETS.get(key_a.replace("-", "_"))
    data_b = TINYML_TARGETS.get(key_b) or TINYML_TARGETS.get(key_b.replace("-", "_"))

    if data_a is None:
        available = sorted(TINYML_TARGETS.keys())
        return f"Target '{target_a}' not found.\n\nAvailable targets: {', '.join(available)}"
    if data_b is None:
        available = sorted(TINYML_TARGETS.keys())
        return f"Target '{target_b}' not found.\n\nAvailable targets: {', '.join(available)}"

    fields = [
        ("Category", "category"),
        ("Architecture", "architecture"),
        ("Clock Range", "clock_range"),
        ("Typical Clock", "typical_clock"),
        ("DSP Support", "dsp_support"),
        ("SIMD Support", "simd_support"),
        ("ML Accelerator", "ml_accelerator"),
        ("SRAM", "typical_sram"),
        ("Flash", "typical_flash"),
        ("FP Support", "fp_support"),
        ("Power Profile", "power_profile"),
        ("Cost Range", "cost_range"),
    ]

    lines = [f"# Comparison: {data_a['name']} vs {data_b['name']}"]
    lines.append("")

    # Side-by-side table
    col_a = data_a["name"]
    col_b = data_b["name"]
    lines.append(f"{'Feature':<20} | {col_a:<35} | {col_b}")
    lines.append("-" * 20 + "-+-" + "-" * 35 + "-+-" + "-" * 35)

    for label, key in fields:
        val_a = str(data_a.get(key, "N/A"))
        val_b = str(data_b.get(key, "N/A"))
        # Truncate long values for table
        if len(val_a) > 35:
            val_a = val_a[:32] + "..."
        if len(val_b) > 35:
            val_b = val_b[:32] + "..."
        lines.append(f"{label:<20} | {val_a:<35} | {val_b}")

    lines.append("")

    # Frameworks
    lines.append(f"## ML Frameworks")
    lines.append(f"\n**{data_a['name']}:** {', '.join(data_a['ml_frameworks'])}")
    lines.append(f"**{data_b['name']}:** {', '.join(data_b['ml_frameworks'])}")

    # Best for
    lines.append(f"\n## Best For")
    lines.append(f"\n**{data_a['name']}:** {data_a['best_for']}")
    lines.append(f"**{data_b['name']}:** {data_b['best_for']}")

    # Example devices
    lines.append(f"\n## Example Devices")
    lines.append(f"\n**{data_a['name']}:** {data_a['example_devices']}")
    lines.append(f"**{data_b['name']}:** {data_b['example_devices']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5: generate_deployment_config — deployment configuration generator
# ---------------------------------------------------------------------------

DEPLOYMENT_CONFIGS: dict[str, dict[str, dict]] = {
    "tflite_micro_cmsis": {
        "name": "TensorFlow Lite Micro + CMSIS-NN",
        "targets": ["cortex-m4", "cortex-m7", "cortex-m55", "cortex-m85"],
        "config": {
            "build_system": "CMake",
            "cmake_template": (
                "cmake_minimum_required(VERSION 3.16)\n"
                "project(tinyml_app C CXX ASM)\n\n"
                "set(CMAKE_C_STANDARD 11)\n"
                "set(CMAKE_CXX_STANDARD 17)\n\n"
                "# Target-specific flags\n"
                "# Cortex-M4:  -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard\n"
                "# Cortex-M7:  -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard\n"
                "# Cortex-M55: -mcpu=cortex-m55 -mthumb -mfloat-abi=hard\n"
                "# Cortex-M85: -mcpu=cortex-m85 -mthumb -mfloat-abi=hard\n\n"
                "# TFLite Micro\n"
                "add_subdirectory(tensorflow/lite/micro)\n\n"
                "# CMSIS-NN acceleration\n"
                "target_compile_definitions(tflite-micro PRIVATE\n"
                "    CMSIS_NN\n"
                ")\n\n"
                "# Your application\n"
                "add_executable(app main.cpp model_data.cc)\n"
                "target_link_libraries(app tflite-micro)"
            ),
            "model_conversion": (
                "# Convert SavedModel/Keras to TFLite INT8\n"
                "import tensorflow as tf\n\n"
                "converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_dir')\n"
                "converter.optimizations = [tf.lite.Optimize.DEFAULT]\n"
                "converter.representative_dataset = representative_dataset_gen\n"
                "converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]\n"
                "converter.inference_input_type = tf.int8\n"
                "converter.inference_output_type = tf.int8\n"
                "tflite_model = converter.convert()\n\n"
                "with open('model.tflite', 'wb') as f:\n"
                "    f.write(tflite_model)"
            ),
            "runtime_code_pattern": (
                '#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"\n'
                '#include "tensorflow/lite/micro/micro_interpreter.h"\n'
                '#include "tensorflow/lite/schema/schema_generated.h"\n'
                '#include "model_data.h"\n\n'
                "// Tensor arena — size depends on model\n"
                "constexpr int kTensorArenaSize = 128 * 1024;  // 128 KB\n"
                "alignas(16) uint8_t tensor_arena[kTensorArenaSize];\n\n"
                "// Setup interpreter\n"
                "const tflite::Model* model = tflite::GetModel(g_model_data);\n"
                "tflite::MicroMutableOpResolver<10> resolver;\n"
                "resolver.AddConv2D();\n"
                "resolver.AddDepthwiseConv2D();\n"
                "resolver.AddReshape();\n"
                "resolver.AddSoftmax();\n"
                "// ... add operators your model uses\n\n"
                "tflite::MicroInterpreter interpreter(model, resolver,\n"
                "    tensor_arena, kTensorArenaSize);\n"
                "interpreter.AllocateTensors();\n\n"
                "// Run inference\n"
                "// Copy input data to interpreter.input(0)->data.int8\n"
                "interpreter.Invoke();\n"
                "// Read output from interpreter.output(0)->data.int8"
            ),
        },
    },
    "vela_ethos_u": {
        "name": "Vela Compiler for Ethos-U NPU",
        "targets": ["ethos-u55", "ethos-u65"],
        "config": {
            "build_system": "Vela CLI + CMake",
            "cmake_template": (
                "# Step 1: Compile TFLite model with Vela\n"
                "# vela model.tflite --accelerator-config=ethos-u55-128\n"
                "# Available configs: ethos-u55-32, ethos-u55-64, ethos-u55-128, ethos-u55-256\n"
                "#                    ethos-u65-256, ethos-u65-512\n\n"
                "# Step 2: Build application with Ethos-U driver\n"
                "cmake_minimum_required(VERSION 3.16)\n"
                "project(ethos_u_app C CXX ASM)\n\n"
                "set(CMAKE_C_STANDARD 11)\n"
                "set(CMAKE_CXX_STANDARD 17)\n\n"
                "# Host CPU flags (Cortex-M55 paired with Ethos-U55)\n"
                "set(CMAKE_C_FLAGS \"-mcpu=cortex-m55 -mthumb -mfloat-abi=hard\")\n\n"
                "# Ethos-U driver\n"
                "add_subdirectory(ethos-u-core-driver)\n\n"
                "# TFLite Micro with Ethos-U delegate\n"
                "add_subdirectory(tensorflow/lite/micro)\n"
                "target_compile_definitions(tflite-micro PRIVATE\n"
                "    ETHOS_U\n"
                "    CMSIS_NN\n"
                ")\n\n"
                "add_executable(app main.cpp model_vela.cc)\n"
                "target_link_libraries(app tflite-micro ethosu_core_driver)"
            ),
            "model_conversion": (
                "# Step 1: Convert to TFLite INT8 (same as CMSIS-NN path)\n"
                "# Step 2: Optimize with Vela compiler\n\n"
                "# Install Vela\n"
                "pip install ethos-u-vela\n\n"
                "# Compile for Ethos-U55 (128 MAC config)\n"
                "vela model.tflite \\\n"
                "    --accelerator-config=ethos-u55-128 \\\n"
                "    --system-config=Ethos_U55_High_End_Embedded \\\n"
                "    --memory-mode=Shared_Sram \\\n"
                "    --output-dir=output/\n\n"
                "# Output: output/model_vela.tflite\n"
                "# Vela report shows: NPU vs CPU operator split, estimated performance"
            ),
            "runtime_code_pattern": (
                "// Same TFLite Micro pattern as CMSIS-NN, but:\n"
                "// 1. Use the Vela-compiled model (model_vela.tflite)\n"
                "// 2. Include Ethos-U driver initialization\n\n"
                '#include "ethosu_driver.h"\n\n'
                "// Initialize Ethos-U NPU\n"
                "struct ethosu_driver ethosu_drv;\n"
                "ethosu_init(&ethosu_drv,\n"
                "    (void*)ETHOS_U_BASE_ADDRESS,  // NPU base address\n"
                "    NULL, 0, 1);                    // cache, size, secure\n\n"
                "// TFLite Micro interpreter automatically delegates\n"
                "// supported ops to the NPU via the Ethos-U custom op.\n"
                "// Unsupported ops fall back to CMSIS-NN on the CPU."
            ),
        },
    },
    "armnn_cortex_a": {
        "name": "Arm NN SDK for Cortex-A",
        "targets": ["cortex-a55", "cortex-a78"],
        "config": {
            "build_system": "CMake",
            "cmake_template": (
                "cmake_minimum_required(VERSION 3.16)\n"
                "project(armnn_app CXX)\n\n"
                "set(CMAKE_CXX_STANDARD 17)\n\n"
                "# Cross-compile for aarch64\n"
                "# set(CMAKE_TOOLCHAIN_FILE aarch64-linux-gnu.cmake)\n\n"
                "# Find Arm NN\n"
                "find_package(ArmNN REQUIRED)\n\n"
                "# Find ACL (Arm Compute Library)\n"
                "find_package(ArmComputeLibrary REQUIRED)\n\n"
                "add_executable(app main.cpp)\n"
                "target_link_libraries(app ArmNN::ArmNN)\n\n"
                "# For TFLite model loading:\n"
                "# target_link_libraries(app ArmNN::ArmNNTfLiteParser)\n"
                "# For ONNX model loading:\n"
                "# target_link_libraries(app ArmNN::ArmNNOnnxParser)"
            ),
            "model_conversion": (
                "# Option 1: Use TFLite model directly with Arm NN TfLite Parser\n"
                "# No conversion needed — Arm NN loads .tflite files natively.\n\n"
                "# Option 2: Use ONNX model with Arm NN ONNX Parser\n"
                "# Export from PyTorch:\n"
                "import torch\n"
                "model = ...  # your PyTorch model\n"
                "dummy_input = torch.randn(1, 3, 224, 224)\n"
                "torch.onnx.export(model, dummy_input, 'model.onnx',\n"
                "    opset_version=13, do_constant_folding=True)\n\n"
                "# Option 3: Quantize ONNX model for INT8 inference\n"
                "# Use onnxruntime.quantization or TF converter"
            ),
            "runtime_code_pattern": (
                '#include <armnn/ArmNN.hpp>\n'
                '#include <armnnTfLiteParser/ITfLiteParser.hpp>\n\n'
                "// Create runtime\n"
                "armnn::IRuntime::CreationOptions options;\n"
                "auto runtime = armnn::IRuntime::Create(options);\n\n"
                "// Parse TFLite model\n"
                "auto parser = armnnTfLiteParser::ITfLiteParser::Create();\n"
                'auto network = parser->CreateNetworkFromBinaryFile("model.tflite");\n\n'
                "// Optimize for target backend\n"
                "std::vector<armnn::BackendId> backends = {\n"
                '    armnn::Compute::CpuAcc,  // NEON-optimized\n'
                '    armnn::Compute::CpuRef   // Reference fallback\n'
                "};\n\n"
                "armnn::IOptimizedNetworkPtr optimizedNet =\n"
                "    armnn::Optimize(*network, backends, runtime->GetDeviceSpec());\n\n"
                "// Load and run\n"
                "armnn::NetworkId networkId;\n"
                "runtime->LoadNetwork(networkId, std::move(optimizedNet));\n"
                "// EnqueueWorkload with input/output tensors..."
            ),
        },
    },
}


@mcp.tool()
def generate_deployment_config(framework: str) -> str:
    """Generate deployment configuration for an ARM edge AI framework.

    Returns build system configuration, model conversion steps, and
    runtime code patterns for deploying ML models on ARM targets.

    Args:
        framework: Deployment framework. One of: "tflite_micro_cmsis",
                   "vela_ethos_u", "armnn_cortex_a". Case-insensitive.
                   Also accepts aliases: "tflite", "cmsis", "vela",
                   "ethos", "armnn", "arm_nn".
    """
    key = framework.lower().strip().replace(" ", "_").replace("-", "_")

    aliases = {
        "tflite": "tflite_micro_cmsis",
        "tflite_micro": "tflite_micro_cmsis",
        "cmsis": "tflite_micro_cmsis",
        "cmsis_nn": "tflite_micro_cmsis",
        "vela": "vela_ethos_u",
        "ethos": "vela_ethos_u",
        "ethos_u": "vela_ethos_u",
        "ethos_u55": "vela_ethos_u",
        "ethos_u65": "vela_ethos_u",
        "armnn": "armnn_cortex_a",
        "arm_nn": "armnn_cortex_a",
        "cortex_a": "armnn_cortex_a",
    }
    key = aliases.get(key, key)

    config_data = DEPLOYMENT_CONFIGS.get(key)
    if config_data is None:
        available = sorted(DEPLOYMENT_CONFIGS.keys())
        return (
            f"No deployment config for framework '{framework}'.\n\n"
            f"Available frameworks: {', '.join(available)}\n\n"
            "Also accepts aliases: tflite, cmsis, vela, ethos, armnn, arm_nn."
        )

    config = config_data["config"]

    lines = [f"# Deployment Configuration: {config_data['name']}"]
    lines.append(f"Target platforms: {', '.join(config_data['targets'])}")
    lines.append("")

    lines.append("## Build System Configuration")
    lines.append(f"Build system: {config['build_system']}")
    lines.append(f"\n```cmake\n{config['cmake_template']}\n```")

    lines.append("\n## Model Conversion")
    lines.append(f"\n```python\n{config['model_conversion']}\n```")

    lines.append("\n## Runtime Code Pattern")
    lines.append(f"\n```cpp\n{config['runtime_code_pattern']}\n```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 6: suggest_model_architecture — model selection given hardware constraints
# ---------------------------------------------------------------------------

MODEL_RECOMMENDATIONS: dict[str, list[dict]] = {
    "image_classification": [
        {
            "name": "MobileNetV2 0.35 96x96",
            "params": "0.4M", "macs": "12M", "model_size_int8": "0.4 MB",
            "sram_min_kb": 64, "flash_min_kb": 400,
            "accuracy": "~60% ImageNet top-1 (INT8)",
            "target_fps_m55": 25, "target_fps_u55": 100,
            "notes": "Smallest MobileNet variant. Good for visual wake word and simple 2-10 class problems.",
        },
        {
            "name": "MobileNetV2 0.5 128x128",
            "params": "0.7M", "macs": "32M", "model_size_int8": "0.7 MB",
            "sram_min_kb": 128, "flash_min_kb": 700,
            "accuracy": "~65% ImageNet top-1 (INT8)",
            "target_fps_m55": 12, "target_fps_u55": 60,
            "notes": "Good balance of accuracy and size for embedded vision.",
        },
        {
            "name": "MobileNetV2 1.0 224x224",
            "params": "3.4M", "macs": "300M", "model_size_int8": "3.4 MB",
            "sram_min_kb": 512, "flash_min_kb": 3400,
            "accuracy": "~71% ImageNet top-1 (INT8)",
            "target_fps_m55": 2, "target_fps_u55": 15,
            "notes": "Full MobileNetV2. Requires Cortex-M55 or Ethos-U. Too large for most Cortex-M4/M7.",
        },
        {
            "name": "EfficientNet-Lite0 224x224",
            "params": "4.7M", "macs": "390M", "model_size_int8": "4.7 MB",
            "sram_min_kb": 768, "flash_min_kb": 4700,
            "accuracy": "~75% ImageNet top-1 (INT8)",
            "target_fps_m55": 1, "target_fps_u55": 10,
            "notes": "Higher accuracy than MobileNetV2 but larger. Best for Cortex-A or Ethos-U65.",
        },
        {
            "name": "MCUNet (MCUNetV2)",
            "params": "0.7M", "macs": "80M", "model_size_int8": "0.7 MB",
            "sram_min_kb": 256, "flash_min_kb": 700,
            "accuracy": "~68% ImageNet top-1 (INT8)",
            "target_fps_m55": 8, "target_fps_u55": 35,
            "notes": "NAS-optimized for microcontrollers. Patch-based inference reduces peak SRAM.",
        },
    ],
    "keyword_spotting": [
        {
            "name": "DS-CNN-S (Small)",
            "params": "24K", "macs": "5.4M", "model_size_int8": "24 KB",
            "sram_min_kb": 16, "flash_min_kb": 32,
            "accuracy": "~93% (Google Speech Commands v2, 12 classes)",
            "target_fps_m55": 500, "target_fps_u55": 2000,
            "notes": "Tiny depthwise-separable CNN. Fits on Cortex-M0+. Always-on keyword detection.",
        },
        {
            "name": "DS-CNN-M (Medium)",
            "params": "80K", "macs": "19M", "model_size_int8": "80 KB",
            "sram_min_kb": 32, "flash_min_kb": 80,
            "accuracy": "~95% (Google Speech Commands v2, 12 classes)",
            "target_fps_m55": 200, "target_fps_u55": 800,
            "notes": "Good accuracy/size trade-off for keyword spotting.",
        },
        {
            "name": "DS-CNN-L (Large)",
            "params": "490K", "macs": "56M", "model_size_int8": "490 KB",
            "sram_min_kb": 64, "flash_min_kb": 500,
            "accuracy": "~96.5% (Google Speech Commands v2, 12 classes)",
            "target_fps_m55": 50, "target_fps_u55": 200,
            "notes": "Best accuracy for keyword spotting. Suitable for 35-class problems.",
        },
    ],
    "object_detection": [
        {
            "name": "YOLOv8n 160x160",
            "params": "3.2M", "macs": "130M", "model_size_int8": "3.2 MB",
            "sram_min_kb": 512, "flash_min_kb": 3200,
            "accuracy": "~25 mAP (COCO, INT8 160x160)",
            "target_fps_m55": 2, "target_fps_u55": 10,
            "notes": "Nano YOLO at reduced resolution. Basic detection on microcontrollers.",
        },
        {
            "name": "SSD-MobileNetV2 320x320",
            "params": "4.5M", "macs": "800M", "model_size_int8": "4.5 MB",
            "sram_min_kb": 1024, "flash_min_kb": 4500,
            "accuracy": "~22 mAP (COCO, INT8)",
            "target_fps_m55": 0.5, "target_fps_u55": 5,
            "notes": "Classic lightweight detector. Better suited for Cortex-A or Ethos-U65.",
        },
        {
            "name": "FOMO (Fast Objects, More Objects)",
            "params": "55K", "macs": "3M", "model_size_int8": "55 KB",
            "sram_min_kb": 64, "flash_min_kb": 64,
            "accuracy": "~80% F1 (custom datasets, centroid detection)",
            "target_fps_m55": 30, "target_fps_u55": 100,
            "notes": "Edge Impulse FOMO. Centroid-based detection, not bounding boxes. Very fast on Cortex-M.",
        },
    ],
    "anomaly_detection": [
        {
            "name": "Autoencoder (Small)",
            "params": "10K", "macs": "0.5M", "model_size_int8": "10 KB",
            "sram_min_kb": 8, "flash_min_kb": 16,
            "accuracy": "AUC ~0.85 (vibration/audio anomaly)",
            "target_fps_m55": 1000, "target_fps_u55": 5000,
            "notes": "3-layer dense autoencoder. Ideal for vibration, audio, sensor anomaly detection.",
        },
        {
            "name": "Autoencoder (Medium)",
            "params": "50K", "macs": "2.5M", "model_size_int8": "50 KB",
            "sram_min_kb": 16, "flash_min_kb": 64,
            "accuracy": "AUC ~0.92 (vibration/audio anomaly)",
            "target_fps_m55": 500, "target_fps_u55": 2000,
            "notes": "5-layer autoencoder with larger hidden dimensions. Better reconstruction quality.",
        },
        {
            "name": "1D-CNN Anomaly Detector",
            "params": "30K", "macs": "8M", "model_size_int8": "30 KB",
            "sram_min_kb": 32, "flash_min_kb": 32,
            "accuracy": "AUC ~0.90 (time-series anomaly)",
            "target_fps_m55": 200, "target_fps_u55": 800,
            "notes": "1D convolutional network for time-series. Good for predictive maintenance.",
        },
    ],
}


@mcp.tool()
def suggest_model_architecture(task: str, sram_kb: int, flash_kb: int) -> str:
    """Suggest ML model architectures that fit within hardware constraints.

    Given a task type and available SRAM/Flash, returns models that fit
    with estimated performance on Cortex-M55 and Ethos-U55.

    Args:
        task: ML task type. One of: "image_classification",
              "keyword_spotting", "object_detection", "anomaly_detection".
              Use "list" or "overview" to see available tasks.
        sram_kb: Available SRAM in kilobytes (e.g., 256, 512, 1024).
        flash_kb: Available Flash in kilobytes (e.g., 512, 2048, 4096).
    """
    key = task.lower().strip().replace(" ", "_").replace("-", "_")

    # Aliases
    aliases = {
        "classification": "image_classification",
        "vision": "image_classification",
        "image": "image_classification",
        "kws": "keyword_spotting",
        "speech": "keyword_spotting",
        "wake_word": "keyword_spotting",
        "wakeword": "keyword_spotting",
        "detection": "object_detection",
        "yolo": "object_detection",
        "anomaly": "anomaly_detection",
        "predictive_maintenance": "anomaly_detection",
    }
    key = aliases.get(key, key)

    if key in ("list", "overview", "all"):
        lines = ["# Available ML Task Types\n"]
        for tkey, models in MODEL_RECOMMENDATIONS.items():
            lines.append(f"  **{tkey}**: {len(models)} model(s)")
            smallest = min(models, key=lambda m: m["sram_min_kb"])
            lines.append(f"    Smallest model: {smallest['name']} ({smallest['sram_min_kb']} KB SRAM, {smallest['flash_min_kb']} KB Flash)")
        lines.append(f"\nUse `suggest_model_architecture(task, sram_kb, flash_kb)` to find models that fit.")
        return "\n".join(lines)

    models = MODEL_RECOMMENDATIONS.get(key)
    if models is None:
        available = list(MODEL_RECOMMENDATIONS.keys())
        return (
            f"Error: Unknown task type '{task}'.\n"
            f"Available: {', '.join(available)}, overview"
        )

    # Filter models that fit
    fitting = [m for m in models if m["sram_min_kb"] <= sram_kb and m["flash_min_kb"] <= flash_kb]
    too_large = [m for m in models if m not in fitting]

    lines = [f"# Model Recommendations: {key.replace('_', ' ').title()}"]
    lines.append(f"Hardware constraints: {sram_kb} KB SRAM, {flash_kb} KB Flash\n")

    if fitting:
        lines.append(f"## Models That Fit ({len(fitting)})\n")
        for m in fitting:
            lines.append(f"### {m['name']}")
            lines.append(f"  Parameters: {m['params']}  |  MACs: {m['macs']}  |  Model size (INT8): {m['model_size_int8']}")
            lines.append(f"  SRAM required: {m['sram_min_kb']} KB  |  Flash required: {m['flash_min_kb']} KB")
            lines.append(f"  Accuracy: {m['accuracy']}")
            lines.append(f"  Est. FPS on Cortex-M55: {m['target_fps_m55']}  |  Est. FPS on Ethos-U55: {m['target_fps_u55']}")
            lines.append(f"  Notes: {m['notes']}")
            sram_margin = ((sram_kb - m["sram_min_kb"]) / sram_kb) * 100
            flash_margin = ((flash_kb - m["flash_min_kb"]) / flash_kb) * 100
            lines.append(f"  Headroom: SRAM {sram_margin:.0f}%, Flash {flash_margin:.0f}%")
            lines.append("")
    else:
        lines.append("## No Models Fit Your Constraints\n")
        smallest = min(models, key=lambda m: m["sram_min_kb"])
        lines.append(f"Smallest available model ({smallest['name']}) requires:")
        lines.append(f"  SRAM: {smallest['sram_min_kb']} KB (you have {sram_kb} KB)")
        lines.append(f"  Flash: {smallest['flash_min_kb']} KB (you have {flash_kb} KB)")
        lines.append("")

    if too_large:
        lines.append(f"## Too Large for Your Hardware ({len(too_large)})\n")
        for m in too_large:
            reason = []
            if m["sram_min_kb"] > sram_kb:
                reason.append(f"needs {m['sram_min_kb']} KB SRAM")
            if m["flash_min_kb"] > flash_kb:
                reason.append(f"needs {m['flash_min_kb']} KB Flash")
            lines.append(f"  - {m['name']}: {', '.join(reason)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 7: explain_tinyml_framework — deep-dive on TinyML frameworks
# ---------------------------------------------------------------------------

TINYML_FRAMEWORKS: dict[str, dict] = {
    "tflite_micro": {
        "name": "TensorFlow Lite for Microcontrollers (TFLite Micro)",
        "organization": "Google / TensorFlow",
        "description": (
            "TFLite Micro is a lightweight inference engine designed to run "
            "ML models on microcontrollers with as little as 16 KB of memory. "
            "It interprets TFLite FlatBuffer models using a minimal C++ runtime "
            "with no dynamic memory allocation (arena-based allocator)."
        ),
        "targets": "Cortex-M0+, M3, M4, M7, M33, M55, Cortex-A (resource-constrained)",
        "model_format": ".tflite (FlatBuffer)",
        "language": "C++ (C++11), C API available",
        "quantization": "INT8, INT16, Float32 (not recommended for MCU)",
        "optimized_kernels": [
            "CMSIS-NN: ARM-optimized kernels using DSP/Helium instructions (2-8x speedup over reference)",
            "Ethos-U delegate: Offloads supported ops to Ethos-U55/U65 microNPU via Vela compiler",
            "XNNPack: Optimized kernels for Cortex-A (FP32 and quantized)",
        ],
        "key_features": [
            "No dynamic memory allocation — uses a pre-allocated tensor arena",
            "Minimal binary size: 20-50 KB for a basic inference setup",
            "Operator resolver: only link operators your model uses (reduces code size)",
            "Interpreter-based execution (not compiled/code-generated)",
            "FlatBuffer model parsing with zero-copy for weights",
        ],
        "getting_started": (
            "1. Train model in TensorFlow/Keras\n"
            "2. Convert to TFLite with INT8 quantization\n"
            "3. Include tflite-micro library in your firmware project\n"
            "4. Create interpreter, allocate tensors, copy input, invoke, read output\n"
            "5. For Cortex-M: link with CMSIS-NN for optimized kernels"
        ),
        "limitations": [
            "No training — inference only",
            "Limited operator support (~80 ops, no dynamic shapes)",
            "No multi-threading (single-threaded execution)",
            "Model must fit in Flash + working memory in SRAM arena",
        ],
    },
    "cmsis_nn": {
        "name": "CMSIS-NN",
        "organization": "Arm",
        "description": (
            "CMSIS-NN is a collection of optimized neural network kernels for "
            "ARM Cortex-M processors. It provides hand-tuned assembly and "
            "intrinsic-based implementations of common ML operators that "
            "leverage DSP instructions (SMLAD, SMLALD) on Cortex-M4/M7 and "
            "Helium MVE vector instructions on Cortex-M55."
        ),
        "targets": "Cortex-M4, M7, M33, M55 (DSP and/or Helium required for full speedup)",
        "model_format": "Not a runtime — provides kernel functions called by TFLite Micro or directly",
        "language": "C (CMSIS-standard)",
        "quantization": "INT8 (primary), INT16 (selected kernels)",
        "optimized_kernels": [
            "Conv2D: im2col + GEMM with DSP/MVE. 2-8x faster than naive C.",
            "Depthwise Conv2D: Optimized for depth_multiplier=1 (MobileNet pattern)",
            "Fully Connected: SMLAD-based dot product on M4/M7, MVE on M55",
            "Pooling: Max and Average pool with loop optimization",
            "Activation: ReLU, Sigmoid (LUT), Tanh (LUT), Softmax",
            "LSTM: Fused gate operations for recurrent networks",
        ],
        "key_features": [
            "Part of the CMSIS standard — widely supported across ARM toolchains",
            "Zero dependencies — pure C with ARM intrinsics",
            "Integrated as TFLite Micro's optimized backend for ARM",
            "Supports both DSP (Cortex-M4/M7) and Helium MVE (Cortex-M55) paths",
            "Im2col + GEMM approach for Conv2D — requires temporary SRAM buffer",
        ],
        "getting_started": (
            "1. Include CMSIS-NN source files in your project (or link via CMSIS-Pack)\n"
            "2. Use TFLite Micro with CMSIS-NN resolver for automatic integration\n"
            "3. Or call CMSIS-NN functions directly: arm_convolve_wrapper_s8(), etc.\n"
            "4. Ensure compiler flags enable DSP: -mcpu=cortex-m4 -mfpu=fpv4-sp-d16"
        ),
        "limitations": [
            "Not a standalone runtime — provides kernels, not model loading/scheduling",
            "Im2col buffer for Conv2D can consume significant SRAM",
            "Limited operator set compared to full TFLite",
            "INT8 only for most optimized paths (INT16 support is partial)",
        ],
    },
    "vela": {
        "name": "Vela Compiler (ethos-u-vela)",
        "organization": "Arm",
        "description": (
            "Vela is an offline compiler that optimizes TFLite models for "
            "Arm Ethos-U microNPUs (U55, U65). It analyzes the model graph, "
            "determines which operators can run on the NPU vs CPU, partitions "
            "the graph, and generates an optimized binary with NPU command "
            "streams and weight layouts."
        ),
        "targets": "Ethos-U55 (32/64/128/256 MAC), Ethos-U65 (256/512 MAC)",
        "model_format": "Input: .tflite (INT8 quantized), Output: .tflite (Vela-optimized)",
        "language": "Python (CLI tool)",
        "quantization": "Accepts INT8 pre-quantized models only",
        "optimized_kernels": [
            "Automatic graph partitioning: NPU-compatible ops on NPU, rest on CPU (CMSIS-NN)",
            "Weight compression: Huffman coding reduces model size by 20-40%",
            "Operator fusion: Conv+BN+ReLU fused into single NPU command",
            "Memory scheduling: Optimizes SRAM usage across layers to minimize peak footprint",
            "Performance estimation: Reports estimated cycles, NPU utilization, and memory usage",
        ],
        "key_features": [
            "Offline compilation — no runtime overhead",
            "Generates detailed performance reports (--timing option)",
            "Configurable for different Ethos-U MAC configurations",
            "Memory modes: Shared_Sram, Dedicated_Sram, Off_Chip_Flash",
            "Output is a standard .tflite file — runs with unmodified TFLite Micro + Ethos-U driver",
        ],
        "getting_started": (
            "1. pip install ethos-u-vela\n"
            "2. vela model.tflite --accelerator-config=ethos-u55-128\n"
            "3. Output: model_vela.tflite (optimized for NPU)\n"
            "4. Deploy with TFLite Micro + Ethos-U core driver\n"
            "5. Check vela output report for NPU/CPU op split"
        ),
        "limitations": [
            "Only works with Ethos-U NPUs (not for CPU-only deployment)",
            "Input must be fully INT8 quantized — no mixed precision",
            "Some operators not supported (see check_operator_support tool)",
            "Requires offline compilation — no JIT or dynamic model loading",
        ],
    },
    "armnn": {
        "name": "Arm NN",
        "organization": "Arm",
        "description": (
            "Arm NN is an inference engine for Cortex-A and Neoverse processors. "
            "It provides optimized backends using the Arm Compute Library (ACL) "
            "for NEON SIMD and GPU (Mali/Immortalis) acceleration. It supports "
            "TFLite, ONNX, and TF models."
        ),
        "targets": "Cortex-A (A53, A55, A72, A76, A78, X1-X4), Neoverse (N1, N2, V1, V2)",
        "model_format": "TFLite, ONNX, TensorFlow SavedModel",
        "language": "C++ (C++17), Python bindings (pyarmnn)",
        "quantization": "FP32, FP16, INT8 (per-tensor and per-channel), BF16 (selected backends)",
        "optimized_kernels": [
            "NEON backend: Optimized GEMM, Conv2D, Pooling using NEON SIMD intrinsics",
            "CL (OpenCL) backend: GPU acceleration on Mali/Immortalis GPUs",
            "Reference backend: Portable C++ (for validation and non-ARM platforms)",
            "Uses Arm Compute Library (ACL) for low-level optimized primitives",
        ],
        "key_features": [
            "Multi-backend: automatic selection of NEON, CL, or Reference per-operator",
            "Graph optimization: operator fusion, constant folding, layout conversion",
            "Dynamic model loading from file or memory buffer",
            "Supports multi-threaded inference with configurable thread count",
            "FP16 mode for 2x throughput on Cortex-A cores with FP16 hardware",
        ],
        "getting_started": (
            "1. Install Arm NN (build from source or use pre-built packages)\n"
            "2. Load model: INetworkPtr network = parser->CreateNetworkFromFile(model_path)\n"
            "3. Optimize: IOptimizedNetworkPtr optNet = Optimize(*network, {Compute::CpuAcc})\n"
            "4. Run: runtime->EnqueueWorkload(networkId, inputTensors, outputTensors)\n"
            "5. For Python: import pyarmnn, use similar API"
        ),
        "limitations": [
            "Larger binary size than TFLite Micro (not suitable for Cortex-M)",
            "Requires Linux OS (not bare-metal compatible)",
            "Build from source can be complex (depends on ACL, Flatbuffers, etc.)",
            "GPU backend requires OpenCL driver support",
        ],
    },
    "edge_impulse": {
        "name": "Edge Impulse",
        "organization": "Edge Impulse Inc.",
        "description": (
            "Edge Impulse is an end-to-end platform for developing embedded ML "
            "applications. It provides data collection, model training, "
            "quantization, and deployment in a single workflow. Generates "
            "optimized C++ libraries for deployment on ARM targets."
        ),
        "targets": "Cortex-M4, M7, M33, M55, Cortex-A (Linux), Ethos-U55/U65, Neoverse",
        "model_format": "Platform-specific C++ library (generated), TFLite, ONNX export",
        "language": "Web UI + CLI (edge-impulse-cli), Generated C++ for deployment",
        "quantization": "Automatic INT8 quantization during EON Compiler export, FP32 also available",
        "optimized_kernels": [
            "EON Compiler: Generates optimized, model-specific C++ code (no interpreter overhead)",
            "Uses CMSIS-NN under the hood for Cortex-M targets",
            "FOMO: Proprietary fast object detection architecture for tiny devices",
            "Automatic DSP pipeline for audio (MFCC, spectrograms)",
        ],
        "key_features": [
            "End-to-end workflow: data → model → deployment",
            "EON Compiler: Ahead-of-time compilation, 25-55% less RAM than TFLite Micro interpreter",
            "FOMO: Real-time centroid-based object detection at 30+ FPS on Cortex-M7",
            "Built-in DSP blocks for audio (MFCC), IMU (spectral features), and image processing",
            "One-click deployment to 100+ boards (Arduino, STM32, Nordic, Espressif, etc.)",
            "Free tier available for individuals and small projects",
        ],
        "getting_started": (
            "1. Create project at edgeimpulse.com\n"
            "2. Collect data via mobile phone, CLI, or API\n"
            "3. Design impulse: input block → processing block → learning block\n"
            "4. Train model in browser\n"
            "5. Deploy: Download C++ library or flash directly to device"
        ),
        "limitations": [
            "Cloud-dependent for training (cannot train locally in free tier)",
            "Generated code is somewhat opaque (harder to customize than raw TFLite)",
            "Advanced features require paid tiers",
            "Model architecture choices limited to platform-supported options",
        ],
    },
}


@mcp.tool()
def explain_tinyml_framework(framework: str) -> str:
    """Explain a TinyML / edge AI framework in detail.

    Returns description, supported targets, model format, optimized kernels,
    key features, getting started guide, and limitations.

    Args:
        framework: Framework name. One of: "tflite_micro", "cmsis_nn",
                   "vela", "armnn", "edge_impulse". Also accepts aliases
                   like "tflite", "cmsis", "ethos", "arm_nn", "ei".
                   Use "list" or "overview" to see all available frameworks.
    """
    key = framework.lower().strip().replace("-", "_").replace(" ", "_")

    # Aliases
    aliases = {
        "tflite": "tflite_micro",
        "tflm": "tflite_micro",
        "tensorflow_lite_micro": "tflite_micro",
        "tensorflow_lite": "tflite_micro",
        "cmsis": "cmsis_nn",
        "ethos": "vela",
        "ethos_u": "vela",
        "ethos_u_vela": "vela",
        "arm_nn": "armnn",
        "acl": "armnn",
        "arm_compute_library": "armnn",
        "ei": "edge_impulse",
        "edgeimpulse": "edge_impulse",
    }
    key = aliases.get(key, key)

    if key in ("list", "overview", "all"):
        lines = ["# TinyML & Edge AI Frameworks\n"]
        for fkey, fw in TINYML_FRAMEWORKS.items():
            lines.append(f"  **{fkey}**: {fw['name']} ({fw['organization']})")
            lines.append(f"    Targets: {fw['targets']}")
            lines.append("")
        lines.append("Use `explain_tinyml_framework(name)` for detailed info.")
        return "\n".join(lines)

    fw = TINYML_FRAMEWORKS.get(key)
    if fw is None:
        available = list(TINYML_FRAMEWORKS.keys())
        return (
            f"Error: Unknown framework '{framework}'.\n"
            f"Available: {', '.join(available)}, overview\n"
            "Also accepts aliases: tflite, cmsis, ethos, arm_nn, ei."
        )

    lines = [f"# {fw['name']}"]
    lines.append(f"Organization: {fw['organization']}")
    lines.append(f"Targets: {fw['targets']}")
    lines.append(f"Model format: {fw['model_format']}")
    lines.append(f"Language: {fw['language']}")
    lines.append(f"Quantization: {fw['quantization']}")
    lines.append(f"\n{fw['description']}\n")

    lines.append("## Optimized Kernels / Backends\n")
    for kernel in fw["optimized_kernels"]:
        lines.append(f"  - {kernel}")
    lines.append("")

    lines.append("## Key Features\n")
    for feature in fw["key_features"]:
        lines.append(f"  - {feature}")
    lines.append("")

    lines.append("## Getting Started\n")
    lines.append(fw["getting_started"])
    lines.append("")

    lines.append("## Limitations\n")
    for lim in fw["limitations"]:
        lines.append(f"  - {lim}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run()


if __name__ == "__main__":
    main()
