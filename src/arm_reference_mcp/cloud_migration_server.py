"""ARM Cloud Migration Advisor MCP Server.

Provides seven tools for planning and executing x86-to-ARM cloud migrations:
  - scan_x86_dependencies:      Analyze dependencies for ARM compatibility.
  - suggest_arm_cloud_instance: Map workload profiles to ARM cloud instances.
  - check_docker_arm_support:   Check Docker base images for arm64 support.
  - generate_ci_matrix:         Generate cross-architecture CI config snippets.
  - estimate_migration_effort:  Estimate migration complexity and get checklists.
  - generate_arm_dockerfile:    Generate multi-stage ARM-optimized Dockerfiles.
  - compare_arm_vs_x86_perf:   Compare ARM vs x86 performance benchmarks.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "ARM Cloud Migration Advisor",
    instructions="Analyze codebases, dependencies, and infrastructure for x86-to-ARM cloud migration readiness.",
)


# ---------------------------------------------------------------------------
# Tool 1: scan_x86_dependencies — inline data
# ---------------------------------------------------------------------------

X86_DEPENDENCY_DB: dict[str, dict[str, dict]] = {
    "python": {
        "numpy": {
            "status": "compatible",
            "notes": "Native ARM wheels available since 1.21. Uses OpenBLAS on ARM with good performance. Apple Silicon and Linux aarch64 fully supported.",
            "alternative": None,
        },
        "scipy": {
            "status": "compatible",
            "notes": "ARM wheels available since 1.7. Builds against OpenBLAS on ARM. Full functionality on aarch64.",
            "alternative": None,
        },
        "tensorflow": {
            "status": "compatible",
            "notes": "Official aarch64 wheels available since TF 2.10. AWS Graviton optimized builds exist via amazon-tensorflow. GPU support via NVIDIA Jetson.",
            "alternative": None,
        },
        "pytorch": {
            "status": "compatible",
            "notes": "Official aarch64 wheels since PyTorch 1.13. ACL (Arm Compute Library) backend available for optimized inference. Build from source for best performance.",
            "alternative": None,
        },
        "pandas": {
            "status": "compatible",
            "notes": "Fully supported on ARM. Pure Python + NumPy/C extensions compile cleanly on aarch64.",
            "alternative": None,
        },
        "scikit-learn": {
            "status": "compatible",
            "notes": "ARM wheels available. Uses OpenBLAS on ARM. Performance comparable to x86 with OpenBLAS backend.",
            "alternative": None,
        },
        "opencv-python": {
            "status": "compatible",
            "notes": "ARM wheels available via opencv-python-headless. NEON SIMD optimizations included. Build from source for full hardware acceleration.",
            "alternative": None,
        },
        "pillow": {
            "status": "compatible",
            "notes": "Fully supported on ARM. Pre-built wheels available for aarch64 Linux.",
            "alternative": None,
        },
        "cryptography": {
            "status": "compatible",
            "notes": "ARM wheels available. Uses OpenSSL which has ARM-optimized crypto (AES, SHA via hardware instructions). Excellent performance on Graviton.",
            "alternative": None,
        },
        "psycopg2": {
            "status": "conditional",
            "notes": "Requires libpq-dev to be available for ARM. Use psycopg2-binary for pre-built wheels on aarch64, or psycopg (v3) which has better ARM support.",
            "alternative": "psycopg2-binary or psycopg (v3)",
        },
        "mysqlclient": {
            "status": "conditional",
            "notes": "Requires MySQL/MariaDB client libraries compiled for ARM. Works when system libs are available. Consider PyMySQL as a pure-Python fallback.",
            "alternative": "PyMySQL (pure Python)",
        },
        "grpcio": {
            "status": "compatible",
            "notes": "ARM wheels available since grpcio 1.49. Builds cleanly from source on aarch64. Performance is excellent on Graviton3.",
            "alternative": None,
        },
        "pyarrow": {
            "status": "compatible",
            "notes": "ARM wheels available since 8.0. Apache Arrow has first-class aarch64 support with NEON SIMD optimizations.",
            "alternative": None,
        },
        "polars": {
            "status": "compatible",
            "notes": "ARM wheels available. Written in Rust with NEON SIMD support. Excellent performance on ARM.",
            "alternative": None,
        },
        "numba": {
            "status": "compatible",
            "notes": "ARM support via LLVM backend since 0.54. JIT compilation works on aarch64. No GPU (CUDA) support on ARM Linux servers.",
            "alternative": None,
        },
        "cupy": {
            "status": "x86_only",
            "notes": "Requires NVIDIA CUDA GPU. Not available on ARM cloud instances (Graviton, Cobalt, Axion). Only works on NVIDIA Jetson for ARM.",
            "alternative": "numpy (CPU), jax (CPU/TPU)",
        },
        "intel-mkl": {
            "status": "x86_only",
            "notes": "Intel Math Kernel Library is x86-only by design. Not available on any ARM platform.",
            "alternative": "OpenBLAS, Arm Performance Libraries (ArmPL), BLIS",
        },
        "openvino": {
            "status": "x86_only",
            "notes": "Intel OpenVINO toolkit is optimized for Intel hardware only. No ARM support.",
            "alternative": "ONNX Runtime (ARM-optimized), TensorFlow Lite, Arm NN",
        },
        "triton": {
            "status": "x86_only",
            "notes": "OpenAI Triton requires NVIDIA GPU with CUDA. Not available on ARM cloud instances.",
            "alternative": "ONNX Runtime, TVM for ARM inference",
        },
        "faiss-cpu": {
            "status": "conditional",
            "notes": "ARM support available since faiss 1.7.3 but requires building from source with NEON support. No pre-built aarch64 wheels on PyPI.",
            "alternative": "faiss-cpu (build from source), Annoy, HNSWlib",
        },
    },
    "nodejs": {
        "sharp": {
            "status": "compatible",
            "notes": "Based on libvips which has excellent ARM support. Pre-built ARM binaries included. NEON SIMD optimizations for image processing.",
            "alternative": None,
        },
        "bcrypt": {
            "status": "compatible",
            "notes": "Native addon with ARM support. Pre-built binaries for linux-arm64 included since v5.1.",
            "alternative": None,
        },
        "canvas": {
            "status": "conditional",
            "notes": "Requires system Cairo library compiled for ARM. Works when libcairo2-dev is installed. Pre-built binaries may not be available.",
            "alternative": "sharp (for image manipulation), skia-canvas",
        },
        "node-sass": {
            "status": "x86_only",
            "notes": "Deprecated. No ARM pre-built binaries. LibSass binding has build issues on ARM.",
            "alternative": "sass (Dart Sass, pure JS, fully cross-platform)",
        },
        "sqlite3": {
            "status": "compatible",
            "notes": "Native addon with pre-built ARM binaries via prebuild-install. Works out of the box on aarch64.",
            "alternative": None,
        },
        "better-sqlite3": {
            "status": "compatible",
            "notes": "Includes pre-built ARM binaries. Compiles cleanly from source on aarch64. Good performance.",
            "alternative": None,
        },
        "cpu-features": {
            "status": "compatible",
            "notes": "Google cpu_features library supports ARM. Detects NEON, SVE, AES, SHA, and other ARM features.",
            "alternative": None,
        },
        "farmhash": {
            "status": "conditional",
            "notes": "Google FarmHash has ARM support but pre-built binaries may not be available. Builds from source on aarch64.",
            "alternative": "xxhash (excellent ARM support)",
        },
        "leveldown": {
            "status": "compatible",
            "notes": "LevelDB has ARM support. Pre-built binaries available for linux-arm64 via prebuild.",
            "alternative": None,
        },
        "sodium-native": {
            "status": "compatible",
            "notes": "libsodium has optimized ARM assembly. Pre-built binaries available for aarch64.",
            "alternative": None,
        },
        "node-rdkafka": {
            "status": "conditional",
            "notes": "Requires librdkafka compiled for ARM. Works when system library is available. Build from source needed on some distros.",
            "alternative": "kafkajs (pure JavaScript, cross-platform)",
        },
        "node-gyp": {
            "status": "compatible",
            "notes": "Build tool, not a library. Works on ARM when Python and C++ toolchain are installed.",
            "alternative": None,
        },
        "esbuild": {
            "status": "compatible",
            "notes": "Written in Go with native ARM64 binaries. Excellent ARM support and performance.",
            "alternative": None,
        },
        "swc": {
            "status": "compatible",
            "notes": "Written in Rust with native ARM64 binaries. First-class aarch64 support.",
            "alternative": None,
        },
        "turbo": {
            "status": "compatible",
            "notes": "Turborepo has native ARM64 binaries. Written in Rust with full cross-platform support.",
            "alternative": None,
        },
    },
    "java": {
        "netty": {
            "status": "compatible",
            "notes": "Native transport (epoll) works on ARM Linux. netty-tcnative has ARM builds. BoringSSL ARM support included.",
            "alternative": None,
        },
        "jna": {
            "status": "compatible",
            "notes": "Java Native Access supports aarch64. JNA 5.x includes ARM64 shared libraries.",
            "alternative": None,
        },
        "javacv": {
            "status": "compatible",
            "notes": "Uses JavaCPP presets which include aarch64 builds for OpenCV, FFmpeg, etc. ARM builds available.",
            "alternative": None,
        },
        "nd4j": {
            "status": "conditional",
            "notes": "ND4J native backend requires platform-specific builds. OpenBLAS backend works on ARM. CUDA backend not available on ARM servers.",
            "alternative": "nd4j-native (with OpenBLAS on ARM)",
        },
        "deeplearning4j": {
            "status": "conditional",
            "notes": "DL4J CPU backend works on ARM via OpenBLAS. GPU (CUDA) backend not available on ARM cloud. Performance may vary.",
            "alternative": "ONNX Runtime Java (ARM-optimized)",
        },
        "snappy-java": {
            "status": "compatible",
            "notes": "Includes pre-built aarch64 JNI library. Snappy compression has ARM NEON optimizations.",
            "alternative": None,
        },
        "lz4-java": {
            "status": "compatible",
            "notes": "Includes aarch64 native library. LZ4 has ARM optimizations. Falls back to Java impl if native unavailable.",
            "alternative": None,
        },
        "zstd-jni": {
            "status": "compatible",
            "notes": "Includes pre-built aarch64 JNI libraries. Zstandard has NEON optimizations in core C library.",
            "alternative": None,
        },
        "sqlite-jdbc": {
            "status": "compatible",
            "notes": "Xerial SQLite JDBC includes aarch64-linux native library. Works out of the box on ARM.",
            "alternative": None,
        },
        "rocksdbjni": {
            "status": "compatible",
            "notes": "RocksDB includes aarch64 JNI builds. ARM NEON CRC32 optimizations available. Good Graviton performance.",
            "alternative": None,
        },
    },
    "cpp": {
        "intel-ipp": {
            "status": "x86_only",
            "notes": "Intel Integrated Performance Primitives. x86/x64 only. Heavily uses SSE/AVX intrinsics.",
            "alternative": "Arm Performance Libraries (ArmPL), Ne10, libyuv",
        },
        "intel-tbb": {
            "status": "conditional",
            "notes": "Intel TBB (oneTBB) has experimental ARM support since 2021.x. Threading primitives work but x86-specific atomics may need porting.",
            "alternative": "oneTBB (recent versions), taskflow, C++17 parallel algorithms",
        },
        "mkl": {
            "status": "x86_only",
            "notes": "Intel Math Kernel Library. x86/x64 only. Uses AVX-512 extensively.",
            "alternative": "OpenBLAS, Arm Performance Libraries (ArmPL), BLIS, FFTW",
        },
        "openblas": {
            "status": "compatible",
            "notes": "Excellent ARM support with NEON/SVE optimized kernels. First-class aarch64 platform. Performance competitive with MKL on Graviton3.",
            "alternative": None,
        },
        "cuda": {
            "status": "x86_only",
            "notes": "NVIDIA CUDA is not available on ARM cloud instances (Graviton, Cobalt, Axion). Only available on NVIDIA Jetson/Grace platforms.",
            "alternative": "OpenCL, Vulkan Compute, ARM Ethos NPU, CPU with NEON/SVE",
        },
        "tensorrt": {
            "status": "x86_only",
            "notes": "NVIDIA TensorRT requires CUDA GPU. Not available on standard ARM cloud instances.",
            "alternative": "ONNX Runtime (ARM-optimized), TensorFlow Lite, Arm NN SDK",
        },
        "onednn": {
            "status": "conditional",
            "notes": "oneAPI DNN Library has experimental AArch64 support via ACL (Arm Compute Library) backend since v2.6. Not all primitives are optimized.",
            "alternative": "Arm Compute Library (ACL), XNNPACK",
        },
        "pcl": {
            "status": "conditional",
            "notes": "Point Cloud Library builds on ARM but some SSE-optimized paths need NEON porting. Use -DPCL_ENABLE_SSE=OFF and enable NEON manually.",
            "alternative": "Open3D (better ARM support)",
        },
        "vtk": {
            "status": "compatible",
            "notes": "VTK builds on ARM Linux. CMake build system handles cross-platform well. Rendering requires ARM-compatible GPU drivers.",
            "alternative": None,
        },
        "opencv": {
            "status": "compatible",
            "notes": "Excellent ARM support with NEON SIMD optimizations. CMake detects ARM and enables optimizations automatically. SVE support in development.",
            "alternative": None,
        },
    },
    "rust": {
        "ring": {
            "status": "compatible",
            "notes": "Crypto library with hand-written ARM assembly. Excellent aarch64 support using hardware AES/SHA instructions. One of the best-performing crypto libs on ARM.",
            "alternative": None,
        },
        "rustls": {
            "status": "compatible",
            "notes": "Uses ring for crypto. Full ARM support inherited from ring. No issues on aarch64.",
            "alternative": None,
        },
        "tikv-jemallocator": {
            "status": "compatible",
            "notes": "jemalloc has excellent ARM support. Widely used in ARM production (e.g., Cloudflare Workers on ARM).",
            "alternative": None,
        },
        "lz4-sys": {
            "status": "compatible",
            "notes": "LZ4 C library builds on ARM. Has NEON-optimized paths. Works out of the box via cc crate.",
            "alternative": None,
        },
        "zstd-sys": {
            "status": "compatible",
            "notes": "Zstandard C library builds on ARM with NEON optimizations. No issues on aarch64.",
            "alternative": None,
        },
        "rocksdb": {
            "status": "compatible",
            "notes": "RocksDB Rust binding builds on ARM. Uses ARM CRC32 instructions for checksums. Good performance on Graviton.",
            "alternative": None,
        },
        "rdkafka": {
            "status": "conditional",
            "notes": "Requires librdkafka system library or builds from source via cmake. Works on ARM when build tools are available.",
            "alternative": "kafka-rust (pure Rust, but less feature-complete)",
        },
        "simd-json": {
            "status": "compatible",
            "notes": "Has NEON SIMD backend for ARM since v0.4. Performance excellent on aarch64 with 128-bit NEON path.",
            "alternative": None,
        },
    },
    "go": {
        "github.com/minio/sha256-simd": {
            "status": "compatible",
            "notes": "Has ARM SHA256 hardware acceleration support. Uses crypto extensions on ARMv8. Excellent performance on Graviton.",
            "alternative": None,
        },
        "github.com/klauspost/compress": {
            "status": "compatible",
            "notes": "Pure Go with optional assembly acceleration. ARM64 assembly paths available for S2/zstd/gzip. Top-tier ARM performance.",
            "alternative": None,
        },
        "github.com/dgraph-io/badger": {
            "status": "compatible",
            "notes": "Pure Go key-value store. Works on all platforms Go supports including ARM64. No native dependencies.",
            "alternative": None,
        },
        "github.com/cockroachdb/pebble": {
            "status": "compatible",
            "notes": "Pure Go LSM key-value store. Full ARM64 support. Used in CockroachDB on Graviton.",
            "alternative": None,
        },
        "github.com/mattn/go-sqlite3": {
            "status": "compatible",
            "notes": "CGo SQLite binding. Requires C compiler on ARM but builds cleanly. Uses ARM-optimized SQLite.",
            "alternative": "modernc.org/sqlite (pure Go, no CGo needed)",
        },
        "github.com/go-audio/audio": {
            "status": "compatible",
            "notes": "Pure Go audio library. No platform-specific code. Works on all Go-supported architectures.",
            "alternative": None,
        },
        "github.com/ClickHouse/clickhouse-go": {
            "status": "compatible",
            "notes": "Pure Go ClickHouse driver. No native dependencies. Works on ARM64 without issues.",
            "alternative": None,
        },
        "google.golang.org/grpc": {
            "status": "compatible",
            "notes": "Pure Go gRPC implementation. Full ARM64 support. Widely used in ARM cloud deployments.",
            "alternative": None,
        },
    },
}

_VALID_LANGUAGES = {"python", "nodejs", "java", "cpp", "rust", "go"}


@mcp.tool()
def scan_x86_dependencies(language: str, dependencies: str) -> str:
    """Analyze a list of dependencies for x86-only packages and ARM compatibility.

    Scans the provided dependency list and reports which packages are fully
    compatible with ARM (aarch64), which need conditional work, and which are
    x86-only with suggested ARM-compatible alternatives.

    Args:
        language: Programming language ecosystem. One of: "python", "nodejs",
                  "java", "cpp", "rust", "go".
        dependencies: Comma-separated list of package names to check
                      (e.g. "numpy,scipy,intel-mkl,tensorflow").
    """
    lang = language.lower().strip()
    if lang not in _VALID_LANGUAGES:
        return f"Error: language must be one of {', '.join(sorted(_VALID_LANGUAGES))}."

    dep_list = [d.strip() for d in dependencies.split(",") if d.strip()]
    if not dep_list:
        return "Error: dependencies must be a non-empty comma-separated list of package names."

    db = X86_DEPENDENCY_DB.get(lang, {})

    compatible = []
    conditional = []
    x86_only = []
    unknown = []

    for dep in dep_list:
        dep_lower = dep.lower()
        entry = db.get(dep_lower)
        if entry is None:
            # Try case-insensitive match
            for k, v in db.items():
                if k.lower() == dep_lower:
                    entry = v
                    dep = k
                    break
        if entry is None:
            unknown.append(dep)
        elif entry["status"] == "compatible":
            compatible.append((dep, entry))
        elif entry["status"] == "conditional":
            conditional.append((dep, entry))
        elif entry["status"] == "x86_only":
            x86_only.append((dep, entry))

    lines = []
    lines.append(f"# ARM Dependency Scan: {lang}")
    lines.append(f"Scanned {len(dep_list)} dependencies\n")

    # Summary counts
    lines.append("## Summary")
    lines.append(f"- Compatible: {len(compatible)}")
    lines.append(f"- Conditional (needs work): {len(conditional)}")
    lines.append(f"- x86-only (must replace): {len(x86_only)}")
    lines.append(f"- Unknown (not in database): {len(unknown)}")
    lines.append("")

    if compatible:
        lines.append("## Compatible (ARM-ready)")
        for dep, entry in compatible:
            lines.append(f"### {dep}")
            lines.append(f"  Status: COMPATIBLE")
            lines.append(f"  {entry['notes']}")
            lines.append("")

    if conditional:
        lines.append("## Conditional (needs attention)")
        for dep, entry in conditional:
            lines.append(f"### {dep}")
            lines.append(f"  Status: CONDITIONAL")
            lines.append(f"  {entry['notes']}")
            if entry.get("alternative"):
                lines.append(f"  Alternative: {entry['alternative']}")
            lines.append("")

    if x86_only:
        lines.append("## x86-Only (must replace)")
        for dep, entry in x86_only:
            lines.append(f"### {dep}")
            lines.append(f"  Status: X86-ONLY")
            lines.append(f"  {entry['notes']}")
            if entry.get("alternative"):
                lines.append(f"  ARM Alternative: {entry['alternative']}")
            lines.append("")

    if unknown:
        lines.append("## Unknown Packages")
        lines.append("These packages are not in the compatibility database. Manual verification recommended:")
        for dep in unknown:
            lines.append(f"  - {dep}")
        lines.append("")

    # Migration readiness score
    total_known = len(compatible) + len(conditional) + len(x86_only)
    if total_known > 0:
        score = (len(compatible) * 100 + len(conditional) * 50) / total_known
    else:
        score = 0
    lines.append("## Migration Readiness Score")
    if score >= 80:
        grade = "HIGH"
    elif score >= 50:
        grade = "MEDIUM"
    else:
        grade = "LOW"
    lines.append(f"Score: {score:.0f}/100 ({grade})")
    if x86_only:
        lines.append(f"Note: {len(x86_only)} package(s) must be replaced before migrating to ARM.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2: suggest_arm_cloud_instance — inline data
# ---------------------------------------------------------------------------

ARM_CLOUD_INSTANCES: dict[str, list[dict]] = {
    "aws": [
        {
            "name": "t4g.micro",
            "series": "Graviton2 (burstable)",
            "vcpus": 2,
            "memory_gb": 1,
            "network_gbps": 5,
            "storage": "EBS only",
            "price_per_hour_usd": 0.0084,
            "best_for": ["web_server", "ci_cd"],
            "notes": "Burstable instance ideal for light web workloads. Free-tier eligible. 20% baseline CPU.",
        },
        {
            "name": "t4g.medium",
            "series": "Graviton2 (burstable)",
            "vcpus": 2,
            "memory_gb": 4,
            "network_gbps": 5,
            "storage": "EBS only",
            "price_per_hour_usd": 0.0336,
            "best_for": ["web_server", "ci_cd", "general"],
            "notes": "Popular burstable instance for small web apps and dev environments.",
        },
        {
            "name": "t4g.xlarge",
            "series": "Graviton2 (burstable)",
            "vcpus": 4,
            "memory_gb": 16,
            "network_gbps": 5,
            "storage": "EBS only",
            "price_per_hour_usd": 0.1344,
            "best_for": ["web_server", "general", "ci_cd"],
            "notes": "Burstable instance with good memory for medium web workloads.",
        },
        {
            "name": "c7g.medium",
            "series": "Graviton3 (compute-optimized)",
            "vcpus": 1,
            "memory_gb": 2,
            "network_gbps": 12.5,
            "storage": "EBS only",
            "price_per_hour_usd": 0.0361,
            "best_for": ["web_server", "ci_cd"],
            "notes": "Compute-optimized Graviton3. 25% better compute performance vs Graviton2. DDR5 memory.",
        },
        {
            "name": "c7g.xlarge",
            "series": "Graviton3 (compute-optimized)",
            "vcpus": 4,
            "memory_gb": 8,
            "network_gbps": 12.5,
            "storage": "EBS only",
            "price_per_hour_usd": 0.1445,
            "best_for": ["web_server", "ci_cd", "general"],
            "notes": "Excellent for compute-bound web workloads, CI/CD builds, and batch processing.",
        },
        {
            "name": "c7g.4xlarge",
            "series": "Graviton3 (compute-optimized)",
            "vcpus": 16,
            "memory_gb": 32,
            "network_gbps": 12.5,
            "storage": "EBS only",
            "price_per_hour_usd": 0.5781,
            "best_for": ["hpc", "ci_cd", "ml_inference"],
            "notes": "High compute for HPC, ML inference, and large build workloads.",
        },
        {
            "name": "c7g.16xlarge",
            "series": "Graviton3 (compute-optimized)",
            "vcpus": 64,
            "memory_gb": 128,
            "network_gbps": 30,
            "storage": "EBS only",
            "price_per_hour_usd": 2.3124,
            "best_for": ["hpc", "ml_inference"],
            "notes": "Maximum compute-optimized Graviton3. SVE support for HPC/scientific workloads.",
        },
        {
            "name": "m7g.medium",
            "series": "Graviton3 (general-purpose)",
            "vcpus": 1,
            "memory_gb": 4,
            "network_gbps": 12.5,
            "storage": "EBS only",
            "price_per_hour_usd": 0.0408,
            "best_for": ["general", "web_server"],
            "notes": "Balanced compute/memory for general-purpose workloads. Good for application servers.",
        },
        {
            "name": "m7g.xlarge",
            "series": "Graviton3 (general-purpose)",
            "vcpus": 4,
            "memory_gb": 16,
            "network_gbps": 12.5,
            "storage": "EBS only",
            "price_per_hour_usd": 0.1632,
            "best_for": ["general", "web_server", "database"],
            "notes": "General-purpose with good memory ratio. Popular for mid-size applications and caching.",
        },
        {
            "name": "m7g.4xlarge",
            "series": "Graviton3 (general-purpose)",
            "vcpus": 16,
            "memory_gb": 64,
            "network_gbps": 12.5,
            "storage": "EBS only",
            "price_per_hour_usd": 0.6528,
            "best_for": ["general", "database", "ml_inference"],
            "notes": "Large general-purpose for databases and medium ML inference workloads.",
        },
        {
            "name": "r7g.xlarge",
            "series": "Graviton3 (memory-optimized)",
            "vcpus": 4,
            "memory_gb": 32,
            "network_gbps": 12.5,
            "storage": "EBS only",
            "price_per_hour_usd": 0.2128,
            "best_for": ["database", "general"],
            "notes": "Memory-optimized for in-memory databases, caching (Redis/Memcached), and analytics.",
        },
        {
            "name": "r7g.4xlarge",
            "series": "Graviton3 (memory-optimized)",
            "vcpus": 16,
            "memory_gb": 128,
            "network_gbps": 12.5,
            "storage": "EBS only",
            "price_per_hour_usd": 0.8512,
            "best_for": ["database", "ml_inference"],
            "notes": "High memory for large databases (PostgreSQL, MySQL) and memory-intensive ML inference.",
        },
        {
            "name": "r7g.16xlarge",
            "series": "Graviton3 (memory-optimized)",
            "vcpus": 64,
            "memory_gb": 512,
            "network_gbps": 30,
            "storage": "EBS only",
            "price_per_hour_usd": 3.4048,
            "best_for": ["database", "hpc"],
            "notes": "Maximum memory for very large in-memory databases and analytics.",
        },
        {
            "name": "im4gn.xlarge",
            "series": "Graviton2 (storage-optimized)",
            "vcpus": 4,
            "memory_gb": 16,
            "network_gbps": 25,
            "storage": "1x 937.5 GB NVMe SSD",
            "price_per_hour_usd": 0.2891,
            "best_for": ["database"],
            "notes": "Storage-optimized with NVMe. Ideal for I/O-intensive databases, Elasticsearch, and logging.",
        },
        {
            "name": "c7gn.xlarge",
            "series": "Graviton3 (network-optimized)",
            "vcpus": 4,
            "memory_gb": 8,
            "network_gbps": 30,
            "storage": "EBS only",
            "price_per_hour_usd": 0.1962,
            "best_for": ["web_server", "general"],
            "notes": "Network-optimized Graviton3. Up to 200 Gbps on largest size. Ideal for load balancers, proxies, and network appliances.",
        },
    ],
    "azure": [
        {
            "name": "Standard_D2ps_v6",
            "series": "Dpsv6 (Cobalt 100, general-purpose)",
            "vcpus": 2,
            "memory_gb": 8,
            "network_gbps": 12.5,
            "storage": "Remote storage only",
            "price_per_hour_usd": 0.077,
            "best_for": ["web_server", "general", "ci_cd"],
            "notes": "ARM-based Azure Cobalt 100 processor. General-purpose with balanced compute/memory ratio.",
        },
        {
            "name": "Standard_D4ps_v6",
            "series": "Dpsv6 (Cobalt 100, general-purpose)",
            "vcpus": 4,
            "memory_gb": 16,
            "network_gbps": 12.5,
            "storage": "Remote storage only",
            "price_per_hour_usd": 0.154,
            "best_for": ["web_server", "general", "ci_cd", "database"],
            "notes": "Good general-purpose choice for web apps and moderate database workloads on Cobalt 100.",
        },
        {
            "name": "Standard_D16ps_v6",
            "series": "Dpsv6 (Cobalt 100, general-purpose)",
            "vcpus": 16,
            "memory_gb": 64,
            "network_gbps": 12.5,
            "storage": "Remote storage only",
            "price_per_hour_usd": 0.616,
            "best_for": ["general", "database", "ml_inference"],
            "notes": "Larger Cobalt 100 general-purpose for databases and ML inference workloads.",
        },
        {
            "name": "Standard_D4pds_v6",
            "series": "Dpdsv6 (Cobalt 100, general + local SSD)",
            "vcpus": 4,
            "memory_gb": 16,
            "network_gbps": 12.5,
            "storage": "150 GB temp SSD",
            "price_per_hour_usd": 0.183,
            "best_for": ["database", "general"],
            "notes": "General-purpose with local temp SSD. Good for workloads needing fast scratch storage.",
        },
        {
            "name": "Standard_D2pls_v6",
            "series": "Dplsv6 (Cobalt 100, low-memory)",
            "vcpus": 2,
            "memory_gb": 4,
            "network_gbps": 12.5,
            "storage": "Remote storage only",
            "price_per_hour_usd": 0.062,
            "best_for": ["web_server", "ci_cd"],
            "notes": "Low-memory Cobalt 100 for compute-focused workloads. Cost-effective for scale-out web tiers.",
        },
        {
            "name": "Standard_E4ps_v6",
            "series": "Epsv6 (Cobalt 100, memory-optimized)",
            "vcpus": 4,
            "memory_gb": 32,
            "network_gbps": 12.5,
            "storage": "Remote storage only",
            "price_per_hour_usd": 0.202,
            "best_for": ["database", "ml_inference"],
            "notes": "Memory-optimized Cobalt 100 for in-memory databases and caching workloads.",
        },
        {
            "name": "Standard_E16ps_v6",
            "series": "Epsv6 (Cobalt 100, memory-optimized)",
            "vcpus": 16,
            "memory_gb": 128,
            "network_gbps": 12.5,
            "storage": "Remote storage only",
            "price_per_hour_usd": 0.808,
            "best_for": ["database", "hpc", "ml_inference"],
            "notes": "Large memory-optimized for significant database and analytics workloads.",
        },
        {
            "name": "Standard_E4pds_v6",
            "series": "Epdsv6 (Cobalt 100, memory + local SSD)",
            "vcpus": 4,
            "memory_gb": 32,
            "network_gbps": 12.5,
            "storage": "150 GB temp SSD",
            "price_per_hour_usd": 0.232,
            "best_for": ["database"],
            "notes": "Memory-optimized with local temp SSD. Ideal for database temp tables and caching with local spill.",
        },
    ],
    "gcp": [
        {
            "name": "c4a-standard-1",
            "series": "C4A (Axion, compute-optimized)",
            "vcpus": 1,
            "memory_gb": 4,
            "network_gbps": 10,
            "storage": "Persistent disk",
            "price_per_hour_usd": 0.0378,
            "best_for": ["web_server", "ci_cd"],
            "notes": "Google Axion (Arm Neoverse V2) compute-optimized. Best price-performance for scale-out workloads.",
        },
        {
            "name": "c4a-standard-4",
            "series": "C4A (Axion, compute-optimized)",
            "vcpus": 4,
            "memory_gb": 16,
            "network_gbps": 10,
            "storage": "Persistent disk",
            "price_per_hour_usd": 0.1512,
            "best_for": ["web_server", "ci_cd", "general"],
            "notes": "Solid mid-range Axion for web serving, CI/CD, and general-purpose workloads.",
        },
        {
            "name": "c4a-standard-16",
            "series": "C4A (Axion, compute-optimized)",
            "vcpus": 16,
            "memory_gb": 64,
            "network_gbps": 23,
            "storage": "Persistent disk",
            "price_per_hour_usd": 0.6048,
            "best_for": ["general", "hpc", "ml_inference"],
            "notes": "Large Axion instance for compute-heavy workloads. Good for batch processing and ML inference.",
        },
        {
            "name": "c4a-standard-48",
            "series": "C4A (Axion, compute-optimized)",
            "vcpus": 48,
            "memory_gb": 192,
            "network_gbps": 32,
            "storage": "Persistent disk",
            "price_per_hour_usd": 1.8144,
            "best_for": ["hpc", "ml_inference"],
            "notes": "High-performance Axion for HPC and large-scale inference workloads.",
        },
        {
            "name": "c4a-highmem-4",
            "series": "C4A (Axion, high-memory)",
            "vcpus": 4,
            "memory_gb": 32,
            "network_gbps": 10,
            "storage": "Persistent disk",
            "price_per_hour_usd": 0.2058,
            "best_for": ["database", "general"],
            "notes": "Axion high-memory for databases and memory-intensive applications.",
        },
        {
            "name": "c4a-highmem-16",
            "series": "C4A (Axion, high-memory)",
            "vcpus": 16,
            "memory_gb": 128,
            "network_gbps": 23,
            "storage": "Persistent disk",
            "price_per_hour_usd": 0.8232,
            "best_for": ["database", "ml_inference", "hpc"],
            "notes": "Large high-memory Axion for significant database and analytics workloads.",
        },
        {
            "name": "t2a-standard-1",
            "series": "T2A (Ampere Altra, general-purpose)",
            "vcpus": 1,
            "memory_gb": 4,
            "network_gbps": 10,
            "storage": "Persistent disk",
            "price_per_hour_usd": 0.0316,
            "best_for": ["web_server", "general", "ci_cd"],
            "notes": "Ampere Altra based. Cost-effective for scale-out web and microservice workloads.",
        },
        {
            "name": "t2a-standard-4",
            "series": "T2A (Ampere Altra, general-purpose)",
            "vcpus": 4,
            "memory_gb": 16,
            "network_gbps": 10,
            "storage": "Persistent disk",
            "price_per_hour_usd": 0.1264,
            "best_for": ["web_server", "general", "ci_cd"],
            "notes": "Budget-friendly ARM instance for moderate web and general workloads.",
        },
    ],
    "oracle": [
        {
            "name": "VM.Standard.A1.Flex (1 OCPU)",
            "series": "Ampere A1 (flexible)",
            "vcpus": 1,
            "memory_gb": 6,
            "network_gbps": 1,
            "storage": "Block volume",
            "price_per_hour_usd": 0.01,
            "best_for": ["web_server", "ci_cd", "general"],
            "notes": "Extremely cost-effective. Free tier includes 4 OCPUs + 24 GB. Flexible shape — choose 1-80 OCPUs and 1-512 GB RAM.",
        },
        {
            "name": "VM.Standard.A1.Flex (4 OCPU)",
            "series": "Ampere A1 (flexible)",
            "vcpus": 4,
            "memory_gb": 24,
            "network_gbps": 4,
            "storage": "Block volume",
            "price_per_hour_usd": 0.04,
            "best_for": ["web_server", "general", "ci_cd", "database"],
            "notes": "Popular configuration. Up to 4 OCPUs + 24 GB RAM in Oracle Cloud free tier (Always Free).",
        },
        {
            "name": "VM.Standard.A1.Flex (16 OCPU)",
            "series": "Ampere A1 (flexible)",
            "vcpus": 16,
            "memory_gb": 96,
            "network_gbps": 16,
            "storage": "Block volume",
            "price_per_hour_usd": 0.16,
            "best_for": ["database", "general", "ml_inference"],
            "notes": "Medium-large ARM instance. Excellent value at $0.01/OCPU/hour. Good for databases and mid-size workloads.",
        },
        {
            "name": "VM.Standard.A1.Flex (80 OCPU)",
            "series": "Ampere A1 (flexible)",
            "vcpus": 80,
            "memory_gb": 512,
            "network_gbps": 40,
            "storage": "Block volume",
            "price_per_hour_usd": 0.80,
            "best_for": ["hpc", "ml_inference", "database"],
            "notes": "Maximum VM configuration. 80 Ampere A1 cores with up to 512 GB RAM. Outstanding price-performance.",
        },
        {
            "name": "BM.Standard.A1.160",
            "series": "Ampere A1 (bare metal)",
            "vcpus": 160,
            "memory_gb": 1024,
            "network_gbps": 50,
            "storage": "Block volume + NVMe optional",
            "price_per_hour_usd": 1.60,
            "best_for": ["hpc", "database", "ml_inference"],
            "notes": "Bare metal Ampere A1. 160 cores, 1 TB RAM. Full isolation. Best for large-scale HPC and databases.",
        },
        {
            "name": "VM.Standard.A2.Flex (4 OCPU)",
            "series": "Ampere A2 (AmpereOne, flexible)",
            "vcpus": 4,
            "memory_gb": 24,
            "network_gbps": 4,
            "storage": "Block volume",
            "price_per_hour_usd": 0.048,
            "best_for": ["web_server", "general", "ci_cd"],
            "notes": "AmpereOne processor. Newer generation with improved single-thread performance vs A1.",
        },
    ],
}

_VALID_PROVIDERS = {"aws", "azure", "gcp", "oracle"}
_VALID_WORKLOADS = {"web_server", "database", "ml_inference", "ci_cd", "hpc", "general"}


@mcp.tool()
def suggest_arm_cloud_instance(workload_profile: str, provider: str | None = None) -> str:
    """Map a workload type to recommended ARM cloud instances across providers.

    Returns instance recommendations with specifications, pricing, and notes
    tailored to the specified workload profile.

    Args:
        workload_profile: Type of workload. One of: "web_server", "database",
                          "ml_inference", "ci_cd", "hpc", "general".
        provider: Optional cloud provider filter. One of: "aws", "azure",
                  "gcp", "oracle". If omitted, shows recommendations from
                  all providers.
    """
    profile = workload_profile.lower().strip()
    if profile not in _VALID_WORKLOADS:
        return f"Error: workload_profile must be one of {', '.join(sorted(_VALID_WORKLOADS))}."

    if provider is not None:
        prov = provider.lower().strip()
        if prov not in _VALID_PROVIDERS:
            return f"Error: provider must be one of {', '.join(sorted(_VALID_PROVIDERS))}."
        providers_to_check = [prov]
    else:
        providers_to_check = sorted(ARM_CLOUD_INSTANCES.keys())

    lines = []
    lines.append(f"# ARM Cloud Instance Recommendations: {profile}")
    if provider:
        lines.append(f"Provider: {provider.upper()}")
    lines.append("")

    _provider_names = {
        "aws": "Amazon Web Services (Graviton)",
        "azure": "Microsoft Azure (Cobalt)",
        "gcp": "Google Cloud (Axion / T2A)",
        "oracle": "Oracle Cloud (Ampere)",
    }

    total_matches = 0

    for prov in providers_to_check:
        instances = ARM_CLOUD_INSTANCES.get(prov, [])
        matches = [i for i in instances if profile in i["best_for"]]

        if not matches:
            continue

        total_matches += len(matches)
        lines.append(f"## {_provider_names.get(prov, prov)}")
        lines.append("")
        lines.append(f"{'Instance':<35} {'vCPUs':>5}  {'RAM (GB)':>8}  {'$/hr':>7}  {'Network':>10}  Storage")
        lines.append("-" * 100)

        # Sort by price
        matches.sort(key=lambda x: x["price_per_hour_usd"])

        for inst in matches:
            lines.append(
                f"{inst['name']:<35} {inst['vcpus']:>5}  {inst['memory_gb']:>8}  "
                f"${inst['price_per_hour_usd']:<6.4f}  {inst['network_gbps']:>7} Gbps  {inst['storage']}"
            )

        lines.append("")
        # Add detailed notes for top 3
        lines.append("### Details")
        for inst in matches[:3]:
            lines.append(f"**{inst['name']}** ({inst['series']})")
            lines.append(f"  {inst['notes']}")
            monthly = inst['price_per_hour_usd'] * 730
            lines.append(f"  Estimated monthly cost: ${monthly:.2f}")
            lines.append("")

    if total_matches == 0:
        return f"No ARM instances found for workload profile '{profile}'."

    # Cost comparison note
    lines.append("## Cost Savings Note")
    lines.append("ARM instances typically offer 20-40% cost savings compared to equivalent x86 instances,")
    lines.append("with comparable or better performance for most workloads. Graviton3 and Axion processors")
    lines.append("provide particularly strong performance for web serving, databases, and ML inference.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: check_docker_arm_support — inline data
# ---------------------------------------------------------------------------

DOCKER_ARM_COMPATIBILITY: dict[str, dict] = {
    "ubuntu": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Full arm64 support since 16.04. Excellent ARM ecosystem.",
        "arm64_alternative": None,
        "known_issues": "None. First-class arm64 support.",
    },
    "debian": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Full arm64 port maintained by Debian project.",
        "arm64_alternative": None,
        "known_issues": "None. arm64 is a release architecture.",
    },
    "alpine": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Very small footprint (~5 MB). Uses musl libc which works well on ARM.",
        "arm64_alternative": None,
        "known_issues": "Some packages in apk may not be available for aarch64. glibc-dependent software needs compatibility layer.",
    },
    "centos": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "CentOS Stream 9 supports arm64. CentOS 7/8 have limited arm64 images.",
        "arm64_alternative": "rockylinux or almalinux for RHEL-compatible ARM images",
        "known_issues": "CentOS 7 arm64 images are community-maintained. Use CentOS Stream 9 for best support.",
    },
    "fedora": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Fedora has strong ARM support with regular aarch64 releases.",
        "arm64_alternative": None,
        "known_issues": "None. aarch64 is a primary architecture.",
    },
    "amazonlinux": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Amazon Linux 2 and 2023 have native arm64 images. Optimized for Graviton processors.",
        "arm64_alternative": None,
        "known_issues": "None. Designed to work with Graviton.",
    },
    "node": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Node.js has native arm64 builds since v12. V8 engine highly optimized for ARM.",
        "arm64_alternative": None,
        "known_issues": "Native addons (node-gyp) need ARM compilation tools in the image. Add build-essential or equivalent.",
    },
    "python": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Python arm64 wheels ecosystem is mature. CPython compiles natively on ARM.",
        "arm64_alternative": None,
        "known_issues": "Some pip packages may lack pre-built arm64 wheels, requiring build tools for compilation.",
    },
    "golang": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Go has first-class arm64 support. Cross-compilation built into the toolchain.",
        "arm64_alternative": None,
        "known_issues": "CGo packages need ARM C toolchain. Pure Go programs work perfectly.",
    },
    "rust": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Rust has aarch64-unknown-linux-gnu as a Tier 1 target.",
        "arm64_alternative": None,
        "known_issues": "None. aarch64 is a Tier 1 supported platform.",
    },
    "ruby": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Ruby builds natively on ARM. Gem native extensions may need ARM compilation.",
        "arm64_alternative": None,
        "known_issues": "Some gems with native extensions may not have pre-built arm64 binaries.",
    },
    "php": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. PHP works well on ARM. Extensions compile natively.",
        "arm64_alternative": None,
        "known_issues": "Some PECL extensions may not be tested on ARM. ionCube loader historically x86-only.",
    },
    "openjdk": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. OpenJDK has excellent ARM support with AArch64 JIT. Eclipse Temurin also provides ARM builds.",
        "arm64_alternative": None,
        "known_issues": "None. AArch64 is a first-class JVM platform with full JIT support.",
    },
    "nginx": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. NGINX has native ARM builds with excellent performance.",
        "arm64_alternative": None,
        "known_issues": "None. Works out of the box on arm64.",
    },
    "httpd": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Apache HTTPD builds natively on ARM.",
        "arm64_alternative": None,
        "known_issues": "None. Standard configuration works on ARM.",
    },
    "redis": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Redis has ARM-optimized code paths. Excellent performance on Graviton.",
        "arm64_alternative": None,
        "known_issues": "None. Redis 6+ has ARM-specific optimizations.",
    },
    "postgres": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. PostgreSQL works excellently on ARM. Up to 30% better price-performance on Graviton3 vs x86.",
        "arm64_alternative": None,
        "known_issues": "Extensions needing compilation may require ARM build tools. PostGIS, pgvector work on ARM.",
    },
    "mysql": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. MySQL 8.0+ has native ARM builds with good performance.",
        "arm64_alternative": None,
        "known_issues": "Some storage engines or plugins may have limited ARM testing.",
    },
    "mariadb": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. MariaDB has first-class ARM support. ColumnStore engine available on ARM.",
        "arm64_alternative": None,
        "known_issues": "None. Full feature parity on ARM.",
    },
    "mongo": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. MongoDB 5.0+ has official arm64 builds. MongoDB 6.0/7.0 fully supported.",
        "arm64_alternative": None,
        "known_issues": "MongoDB versions before 4.4 do not support arm64. Ensure you use 5.0+.",
    },
    "elasticsearch": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official arm64 images since ES 7.12. Elasticsearch 8.x fully supports ARM with good performance.",
        "arm64_alternative": None,
        "known_issues": "Older versions (pre-7.12) lack arm64 images. Some ML features use x86-specific optimizations.",
    },
    "kibana": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official arm64 images available. Kibana is primarily Node.js, which runs well on ARM.",
        "arm64_alternative": None,
        "known_issues": "Ensure Kibana version matches Elasticsearch ARM version.",
    },
    "grafana": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image from grafana/grafana. Written in Go, excellent ARM support.",
        "arm64_alternative": None,
        "known_issues": "Some Grafana plugins with native dependencies may need ARM builds.",
    },
    "prometheus": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Written in Go with first-class ARM support. Widely deployed on Graviton.",
        "arm64_alternative": None,
        "known_issues": "None. Pure Go application.",
    },
    "jenkins": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official jenkins/jenkins image has arm64 support. Runs on OpenJDK which is ARM-ready.",
        "arm64_alternative": None,
        "known_issues": "Some Jenkins plugins with native dependencies may not work on ARM. Agent images need ARM variants.",
    },
    "gitlab-runner": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "GitLab Runner has official arm64 images. Docker executor works on ARM. Shell executor works natively.",
        "arm64_alternative": None,
        "known_issues": "Helper images must also be arm64. Docker-in-Docker works on arm64.",
    },
    "traefik": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Written in Go, works perfectly on ARM. Popular for ARM-based edge deployments.",
        "arm64_alternative": None,
        "known_issues": "None. Full feature parity on ARM.",
    },
    "caddy": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Written in Go with native ARM support.",
        "arm64_alternative": None,
        "known_issues": "None. Excellent ARM performance.",
    },
    "haproxy": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. HAProxy has ARM builds with good performance for load balancing.",
        "arm64_alternative": None,
        "known_issues": "None. Works well on ARM.",
    },
    "memcached": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Memcached is simple C code that builds cleanly on ARM.",
        "arm64_alternative": None,
        "known_issues": "None.",
    },
    "rabbitmq": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Erlang/OTP has excellent ARM support. RabbitMQ works perfectly on ARM.",
        "arm64_alternative": None,
        "known_issues": "None. Erlang BEAM VM has first-class ARM support.",
    },
    "kafka": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Confluent and apache/kafka images have arm64 support. Runs on JVM which is ARM-ready. Use KRaft mode for simpler ARM deployments.",
        "arm64_alternative": None,
        "known_issues": "Older Confluent Platform images (pre-7.0) may lack arm64. Use recent versions.",
    },
    "zookeeper": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. JVM-based, works on ARM via OpenJDK.",
        "arm64_alternative": None,
        "known_issues": "None. Standard JVM application.",
    },
    "consul": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image from hashicorp/consul. Written in Go, full ARM support.",
        "arm64_alternative": None,
        "known_issues": "None. Go binary, cross-platform.",
    },
    "vault": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image from hashicorp/vault. Written in Go, ARM support available.",
        "arm64_alternative": None,
        "known_issues": "HSM plugins may need ARM-compatible PKCS#11 libraries.",
    },
    "minio": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Written in Go with ARM SIMD optimizations for hashing. Excellent ARM performance.",
        "arm64_alternative": None,
        "known_issues": "None. MinIO is widely deployed on ARM (Graviton, Ampere).",
    },
    "localstack": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "LocalStack provides arm64 images since v1.0. Most AWS service emulations work on ARM.",
        "arm64_alternative": None,
        "known_issues": "Some Lambda runtimes in LocalStack may default to x86 emulation. Check specific service compatibility.",
    },
    "selenium": {
        "arm64_support": False,
        "multi_arch": False,
        "notes": "Selenium standalone images (selenium/standalone-chrome, etc.) are x86-only. Chrome/Chromium ARM builds are limited.",
        "arm64_alternative": "Use Playwright with ARM-compatible Chromium, or run Selenium Grid with ARM-native browser images.",
        "known_issues": "No official ARM images. Chromium ARM builds available separately but not in official Selenium images.",
    },
    "chrome": {
        "arm64_support": False,
        "multi_arch": False,
        "notes": "Google Chrome Docker images (e.g., browserless/chrome) are typically x86-only.",
        "arm64_alternative": "chromium (ARM builds available in Debian/Ubuntu repos), Playwright Chromium",
        "known_issues": "Chrome headless on ARM requires building Chromium from source or using distro packages.",
    },
    "firefox": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Firefox has ARM builds. Some Docker images include arm64 variants (e.g., selenium/standalone-firefox may not).",
        "arm64_alternative": None,
        "known_issues": "Not all Firefox Docker images are multi-arch. Check specific image for arm64 manifest.",
    },
    "dotnet/sdk": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official .NET SDK image supports arm64. .NET 6+ has first-class ARM support. .NET 8 further improves ARM performance.",
        "arm64_alternative": None,
        "known_issues": "Some NuGet packages with native dependencies may need ARM variants. P/Invoke calls need ARM-compatible libraries.",
    },
    "dotnet/aspnet": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official ASP.NET runtime image supports arm64. Kestrel performs well on ARM.",
        "arm64_alternative": None,
        "known_issues": "None for pure .NET applications. Check native dependencies.",
    },
    "maven": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Maven itself is pure Java. Runs on ARM-compatible OpenJDK.",
        "arm64_alternative": None,
        "known_issues": "None. Build tools are Java-based.",
    },
    "gradle": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Gradle runs on JVM, fully ARM-compatible.",
        "arm64_alternative": None,
        "known_issues": "None. JVM-based build tool.",
    },
    "terraform": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official hashicorp/terraform image has arm64 support. Written in Go, cross-platform.",
        "arm64_alternative": None,
        "known_issues": "Some Terraform providers may not have arm64 builds. Check provider compatibility.",
    },
    "ansible": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Ansible is pure Python. Docker images based on Python arm64 images work well.",
        "arm64_alternative": None,
        "known_issues": "None. Python-based, cross-platform.",
    },
    "busybox": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Official multi-arch image. Busybox has excellent ARM support (historically used in embedded ARM systems).",
        "arm64_alternative": None,
        "known_issues": "None.",
    },
    "scratch": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Empty base image. Platform-independent — works on any architecture. Binary inside must be compiled for arm64.",
        "arm64_alternative": None,
        "known_issues": "Ensure your application binary is compiled for aarch64 (e.g., GOARCH=arm64 or cross-compiled).",
    },
    "distroless": {
        "arm64_support": True,
        "multi_arch": True,
        "notes": "Google distroless images (gcr.io/distroless/) have arm64 variants. Minimal attack surface on ARM.",
        "arm64_alternative": None,
        "known_issues": "Debugging is harder due to no shell. Use debug variant for troubleshooting.",
    },
}


@mcp.tool()
def check_docker_arm_support(image_name: str) -> str:
    """Check whether a Docker base image supports arm64/aarch64 architecture.

    Returns information about ARM compatibility, multi-arch manifest support,
    known issues, and alternatives for images that lack ARM support.

    Args:
        image_name: Docker image name to check (e.g. "ubuntu", "node",
                    "postgres", "selenium"). Use the short/common name
                    without tags.
    """
    name = image_name.strip().lower()

    # Try exact match first, then partial match
    entry = DOCKER_ARM_COMPATIBILITY.get(name)
    if entry is None:
        # Try without common prefixes/suffixes
        for key in DOCKER_ARM_COMPATIBILITY:
            if key in name or name in key:
                entry = DOCKER_ARM_COMPATIBILITY[key]
                name = key
                break

    if entry is None:
        lines = []
        lines.append(f"# Docker ARM Support: {image_name}")
        lines.append("")
        lines.append("**Status: UNKNOWN**")
        lines.append("")
        lines.append(f"The image '{image_name}' is not in the compatibility database.")
        lines.append("")
        lines.append("### How to check manually")
        lines.append(f"1. Run: `docker manifest inspect {image_name}:latest`")
        lines.append("2. Look for `linux/arm64` or `linux/aarch64` in the platform list.")
        lines.append("3. Or use: `docker buildx imagetools inspect {image_name}:latest`")
        lines.append("")
        lines.append("### General guidance")
        lines.append("- Official Docker Hub images are increasingly multi-arch.")
        lines.append("- Language runtime images (python, node, go, rust, ruby) almost always support arm64.")
        lines.append("- Database images (postgres, mysql, redis, mongo) generally support arm64.")
        lines.append("- Images depending on x86-specific software (Chrome, Selenium) may not.")
        return "\n".join(lines)

    lines = []
    lines.append(f"# Docker ARM Support: {name}")
    lines.append("")

    if entry["arm64_support"]:
        lines.append("**Status: ARM64 SUPPORTED**")
    else:
        lines.append("**Status: ARM64 NOT SUPPORTED**")
    lines.append("")

    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| arm64 support | {'Yes' if entry['arm64_support'] else 'No'} |")
    lines.append(f"| Multi-arch manifest | {'Yes' if entry['multi_arch'] else 'No'} |")
    lines.append("")

    lines.append("### Details")
    lines.append(entry["notes"])
    lines.append("")

    if entry.get("arm64_alternative"):
        lines.append("### ARM64 Alternative")
        lines.append(entry["arm64_alternative"])
        lines.append("")

    lines.append("### Known Issues")
    lines.append(entry["known_issues"])
    lines.append("")

    if entry["arm64_support"] and entry["multi_arch"]:
        lines.append("### Usage")
        lines.append(f"Pull with automatic platform detection: `docker pull {name}`")
        lines.append(f"Force arm64: `docker pull --platform linux/arm64 {name}`")
    elif not entry["arm64_support"]:
        lines.append("### Migration Steps")
        lines.append(f"1. This image does not support arm64 natively.")
        if entry.get("arm64_alternative"):
            lines.append(f"2. Consider using: {entry['arm64_alternative']}")
        lines.append("3. You may need to build a custom image for arm64.")
        lines.append("4. Check if the upstream project now offers ARM builds.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: generate_ci_matrix — inline data
# ---------------------------------------------------------------------------

CI_TEMPLATES: dict[str, dict] = {
    "github_actions": {
        "base_template": """# GitHub Actions: Multi-architecture build and test
name: CI (Multi-Arch)

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            arch: x64
          - os: ubuntu-24.04-arm
            arch: arm64
      fail-fast: false

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Show architecture
        run: |
          uname -m
          echo "Running on ${{ matrix.arch }}"

      # Add your language-specific setup and test steps here

  docker-multi-arch:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build multi-arch image
        uses: docker/build-push-action@v6
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: false
          tags: myapp:test
""",
        "arm_runner_notes": """## GitHub Actions ARM Runner Notes

- **Native ARM runners**: Use `ubuntu-24.04-arm` for native aarch64 runners (GitHub-hosted, available since late 2024).
- **QEMU emulation**: For older setups, use `docker/setup-qemu-action` to emulate arm64 on x86 runners. Slower but works.
- **Self-hosted runners**: You can run self-hosted GitHub Actions runners on ARM instances (Graviton, Cobalt, Ampere).
- **Buildx**: Use `docker/setup-buildx-action` + `docker/build-push-action` for multi-arch Docker image builds.
- **Cost**: ARM runners may have different pricing. Check GitHub Actions pricing for ARM runner minutes.
- **Matrix strategy**: Use `strategy.matrix` to test on both x64 and arm64 in parallel.
""",
        "multi_arch_build": """# Multi-arch Docker build with GitHub Actions
# Add to your workflow after setting up Buildx and QEMU

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push multi-arch
        uses: docker/build-push-action@v6
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
""",
        "language_overrides": {
            "python": """      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: pytest --tb=short -v
""",
            "nodejs": """      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test
""",
            "java": """      - name: Set up JDK
        uses: actions/setup-java@v4
        with:
          distribution: 'temurin'
          java-version: '21'
          cache: 'maven'

      - name: Build and test
        run: mvn -B verify
""",
            "go": """      - name: Set up Go
        uses: actions/setup-go@v5
        with:
          go-version: '1.22'
          cache: true

      - name: Build
        run: go build ./...

      - name: Test
        run: go test -v ./...
""",
            "rust": """      - name: Set up Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: aarch64-unknown-linux-gnu

      - name: Build
        run: cargo build --verbose

      - name: Test
        run: cargo test --verbose
""",
            "cpp": """      - name: Install build tools
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential cmake

      - name: Configure
        run: cmake -B build -DCMAKE_BUILD_TYPE=Release

      - name: Build
        run: cmake --build build --parallel

      - name: Test
        run: ctest --test-dir build --output-on-failure
""",
        },
    },
    "gitlab_ci": {
        "base_template": """# GitLab CI: Multi-architecture build and test
stages:
  - test
  - build

variables:
  DOCKER_BUILDKIT: "1"

# Test on x86_64
test:x64:
  stage: test
  tags:
    - linux
    - amd64
  script:
    - echo "Testing on $(uname -m)"
    # Add your test commands here

# Test on arm64
test:arm64:
  stage: test
  tags:
    - linux
    - arm64
  script:
    - echo "Testing on $(uname -m)"
    # Add your test commands here

# Build multi-arch Docker image
build:multi-arch:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  tags:
    - linux
    - amd64
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker buildx create --use --name multiarch --driver docker-container
    - docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
  script:
    - >
      docker buildx build
      --platform linux/amd64,linux/arm64
      --tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
      --tag $CI_REGISTRY_IMAGE:latest
      --push
      .
""",
        "arm_runner_notes": """## GitLab CI ARM Runner Notes

- **Self-hosted runners**: Install GitLab Runner on ARM instances. Use shell or docker executor.
- **Runner tags**: Use tags like `arm64` and `amd64` to route jobs to the correct architecture.
- **SaaS runners**: GitLab SaaS offers Linux arm64 runners on Premium/Ultimate tiers.
- **Docker executor**: Ensure docker images in your CI jobs support arm64 (check with `check_docker_arm_support` tool).
- **QEMU**: For building multi-arch images on x86 runners, use QEMU via `multiarch/qemu-user-static`.
""",
        "multi_arch_build": """# Multi-arch build with Kaniko (alternative to Docker-in-Docker)
build:kaniko:
  stage: build
  image:
    name: gcr.io/kaniko-project/executor:v1.23.0
    entrypoint: [""]
  script:
    - >
      /kaniko/executor
      --context $CI_PROJECT_DIR
      --dockerfile $CI_PROJECT_DIR/Dockerfile
      --destination $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
      --customPlatform linux/arm64
""",
        "language_overrides": {
            "python": """test:python:
  stage: test
  image: python:3.12
  script:
    - pip install -r requirements.txt
    - pytest --tb=short -v
""",
            "nodejs": """test:nodejs:
  stage: test
  image: node:20
  script:
    - npm ci
    - npm test
""",
            "java": """test:java:
  stage: test
  image: maven:3.9-eclipse-temurin-21
  script:
    - mvn -B verify
  cache:
    paths:
      - .m2/repository
""",
            "go": """test:go:
  stage: test
  image: golang:1.22
  script:
    - go build ./...
    - go test -v ./...
""",
            "rust": """test:rust:
  stage: test
  image: rust:1.77
  script:
    - cargo build --verbose
    - cargo test --verbose
  cache:
    paths:
      - target/
""",
            "cpp": """test:cpp:
  stage: test
  image: gcc:14
  script:
    - cmake -B build -DCMAKE_BUILD_TYPE=Release
    - cmake --build build --parallel
    - ctest --test-dir build --output-on-failure
""",
        },
    },
    "circleci": {
        "base_template": """# CircleCI: Multi-architecture build and test
version: 2.1

orbs:
  docker: circleci/docker@2.6

jobs:
  test-x64:
    machine:
      image: ubuntu-2404:current
    resource_class: medium
    steps:
      - checkout
      - run:
          name: Show architecture
          command: uname -m
      # Add your test steps here

  test-arm64:
    machine:
      image: ubuntu-2404:current
    resource_class: arm.medium
    steps:
      - checkout
      - run:
          name: Show architecture
          command: uname -m
      # Add your test steps here

  build-multi-arch:
    machine:
      image: ubuntu-2404:current
    resource_class: medium
    steps:
      - checkout
      - run:
          name: Set up QEMU and Buildx
          command: |
            docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
            docker buildx create --use --name multiarch
      - run:
          name: Build multi-arch image
          command: |
            docker buildx build \\
              --platform linux/amd64,linux/arm64 \\
              --tag myapp:$CIRCLE_SHA1 \\
              .

workflows:
  build-test:
    jobs:
      - test-x64
      - test-arm64
      - build-multi-arch:
          requires:
            - test-x64
            - test-arm64
""",
        "arm_runner_notes": """## CircleCI ARM Runner Notes

- **Native ARM**: Use `resource_class: arm.medium` or `arm.large` for native ARM64 execution.
- **Machine executor**: ARM is available via the machine executor (not Docker executor).
- **Pricing**: ARM resource classes may have different per-minute pricing. Check CircleCI pricing page.
- **Docker images**: When using Docker executor on ARM, ensure images support arm64.
- **Self-hosted runners**: CircleCI runner can be installed on ARM instances for dedicated ARM capacity.
""",
        "multi_arch_build": """  # Multi-arch Docker build and push
  build-and-push:
    machine:
      image: ubuntu-2404:current
    resource_class: medium
    steps:
      - checkout
      - run:
          name: Login to registry
          command: echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
      - run:
          name: Build and push multi-arch
          command: |
            docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
            docker buildx create --use
            docker buildx build \\
              --platform linux/amd64,linux/arm64 \\
              --tag $DOCKER_USERNAME/myapp:$CIRCLE_SHA1 \\
              --tag $DOCKER_USERNAME/myapp:latest \\
              --push \\
              .
""",
        "language_overrides": {
            "python": """  test-python:
    machine:
      image: ubuntu-2404:current
    resource_class: arm.medium
    steps:
      - checkout
      - run:
          name: Install Python and test
          command: |
            sudo apt-get update && sudo apt-get install -y python3-pip python3-venv
            python3 -m venv venv
            . venv/bin/activate
            pip install -r requirements.txt
            pytest --tb=short -v
""",
            "nodejs": """  test-nodejs:
    machine:
      image: ubuntu-2404:current
    resource_class: arm.medium
    steps:
      - checkout
      - run:
          name: Install Node.js and test
          command: |
            curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
            sudo apt-get install -y nodejs
            npm ci
            npm test
""",
            "java": """  test-java:
    machine:
      image: ubuntu-2404:current
    resource_class: arm.medium
    steps:
      - checkout
      - run:
          name: Install JDK and test
          command: |
            sudo apt-get update && sudo apt-get install -y openjdk-21-jdk maven
            mvn -B verify
""",
            "go": """  test-go:
    machine:
      image: ubuntu-2404:current
    resource_class: arm.medium
    steps:
      - checkout
      - run:
          name: Install Go and test
          command: |
            wget https://go.dev/dl/go1.22.0.linux-arm64.tar.gz
            sudo tar -C /usr/local -xzf go1.22.0.linux-arm64.tar.gz
            export PATH=$PATH:/usr/local/go/bin
            go build ./...
            go test -v ./...
""",
            "rust": """  test-rust:
    machine:
      image: ubuntu-2404:current
    resource_class: arm.medium
    steps:
      - checkout
      - run:
          name: Install Rust and test
          command: |
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source $HOME/.cargo/env
            cargo build --verbose
            cargo test --verbose
""",
            "cpp": """  test-cpp:
    machine:
      image: ubuntu-2404:current
    resource_class: arm.medium
    steps:
      - checkout
      - run:
          name: Build and test
          command: |
            sudo apt-get update && sudo apt-get install -y build-essential cmake
            cmake -B build -DCMAKE_BUILD_TYPE=Release
            cmake --build build --parallel
            ctest --test-dir build --output-on-failure
""",
        },
    },
    "jenkins": {
        "base_template": """// Jenkins: Multi-architecture pipeline (Declarative)
pipeline {
    agent none

    stages {
        stage('Test') {
            parallel {
                stage('Test x64') {
                    agent { label 'linux && amd64' }
                    steps {
                        checkout scm
                        sh 'uname -m'
                        // Add your test steps here
                    }
                }
                stage('Test arm64') {
                    agent { label 'linux && arm64' }
                    steps {
                        checkout scm
                        sh 'uname -m'
                        // Add your test steps here
                    }
                }
            }
        }

        stage('Build Multi-Arch Docker Image') {
            agent { label 'linux && amd64 && docker' }
            steps {
                checkout scm
                sh '''
                    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
                    docker buildx create --use --name multiarch || true
                    docker buildx build \\
                        --platform linux/amd64,linux/arm64 \\
                        --tag myapp:${BUILD_NUMBER} \\
                        .
                '''
            }
        }
    }
}
""",
        "arm_runner_notes": """## Jenkins ARM Runner Notes

- **Agent labels**: Use labels like `arm64` and `amd64` to route builds to correct agents.
- **ARM agents**: Install Jenkins agents on ARM instances (Graviton, Cobalt, Ampere).
- **Docker plugin**: Ensure Docker plugin is configured with ARM-compatible images.
- **Kubernetes plugin**: For Jenkins on Kubernetes, add ARM node pools and use nodeSelector/tolerations.
- **JNLP agent**: The Jenkins JNLP agent (inbound-agent) has arm64 Docker images.
- **Pipeline syntax**: Use `parallel` stages to run x64 and arm64 tests simultaneously.
""",
        "multi_arch_build": """        // Multi-arch Docker build and push (add to pipeline stages)
        stage('Build and Push Multi-Arch') {
            agent { label 'linux && docker' }
            environment {
                REGISTRY_CREDENTIALS = credentials('docker-registry')
            }
            steps {
                sh '''
                    echo "$REGISTRY_CREDENTIALS_PSW" | docker login -u "$REGISTRY_CREDENTIALS_USR" --password-stdin
                    docker buildx create --use --name multiarch || true
                    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
                    docker buildx build \\
                        --platform linux/amd64,linux/arm64 \\
                        --tag registry.example.com/myapp:${BUILD_NUMBER} \\
                        --tag registry.example.com/myapp:latest \\
                        --push \\
                        .
                '''
            }
        }
""",
        "language_overrides": {
            "python": """                stage('Test Python arm64') {
                    agent { label 'linux && arm64' }
                    steps {
                        checkout scm
                        sh '''
                            python3 -m venv venv
                            . venv/bin/activate
                            pip install -r requirements.txt
                            pytest --tb=short -v
                        '''
                    }
                }
""",
            "nodejs": """                stage('Test Node.js arm64') {
                    agent { label 'linux && arm64' }
                    tools { nodejs 'Node-20' }
                    steps {
                        checkout scm
                        sh '''
                            npm ci
                            npm test
                        '''
                    }
                }
""",
            "java": """                stage('Test Java arm64') {
                    agent { label 'linux && arm64' }
                    tools { jdk 'JDK-21' }
                    steps {
                        checkout scm
                        sh 'mvn -B verify'
                    }
                }
""",
            "go": """                stage('Test Go arm64') {
                    agent { label 'linux && arm64' }
                    tools { go 'Go-1.22' }
                    steps {
                        checkout scm
                        sh '''
                            go build ./...
                            go test -v ./...
                        '''
                    }
                }
""",
            "rust": """                stage('Test Rust arm64') {
                    agent { label 'linux && arm64' }
                    steps {
                        checkout scm
                        sh '''
                            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                            . $HOME/.cargo/env
                            cargo build --verbose
                            cargo test --verbose
                        '''
                    }
                }
""",
            "cpp": """                stage('Test C++ arm64') {
                    agent { label 'linux && arm64' }
                    steps {
                        checkout scm
                        sh '''
                            cmake -B build -DCMAKE_BUILD_TYPE=Release
                            cmake --build build --parallel
                            ctest --test-dir build --output-on-failure
                        '''
                    }
                }
""",
        },
    },
}

_VALID_CI_PLATFORMS = {"github_actions", "gitlab_ci", "circleci", "jenkins"}


@mcp.tool()
def generate_ci_matrix(ci_platform: str, language: str | None = None) -> str:
    """Generate cross-architecture CI configuration snippets for building and
    testing on both x86 and ARM.

    Produces copy-paste-ready configuration for the specified CI platform,
    including multi-arch Docker build steps and optional language-specific
    setup.

    Args:
        ci_platform: CI/CD platform. One of: "github_actions", "gitlab_ci",
                     "circleci", "jenkins".
        language: Optional programming language for language-specific steps.
                  One of: "python", "nodejs", "java", "go", "rust", "cpp".
                  If omitted, generates a generic multi-arch template.
    """
    platform = ci_platform.lower().strip()
    if platform not in _VALID_CI_PLATFORMS:
        return f"Error: ci_platform must be one of {', '.join(sorted(_VALID_CI_PLATFORMS))}."

    valid_languages = {"python", "nodejs", "java", "go", "rust", "cpp"}
    if language is not None:
        lang = language.lower().strip()
        if lang not in valid_languages:
            return f"Error: language must be one of {', '.join(sorted(valid_languages))}."
    else:
        lang = None

    template = CI_TEMPLATES[platform]

    _platform_names = {
        "github_actions": "GitHub Actions",
        "gitlab_ci": "GitLab CI",
        "circleci": "CircleCI",
        "jenkins": "Jenkins",
    }

    lines = []
    lines.append(f"# Cross-Architecture CI: {_platform_names[platform]}")
    lines.append("")

    lines.append("## Base Template")
    lines.append("```yaml" if platform != "jenkins" else "```groovy")
    lines.append(template["base_template"].strip())
    lines.append("```")
    lines.append("")

    if lang and lang in template.get("language_overrides", {}):
        lines.append(f"## Language-Specific Steps ({lang})")
        lines.append("Add these steps to your pipeline:")
        lines.append("```yaml" if platform != "jenkins" else "```groovy")
        lines.append(template["language_overrides"][lang].strip())
        lines.append("```")
        lines.append("")

    lines.append("## Multi-Arch Docker Build")
    lines.append("```yaml" if platform != "jenkins" else "```groovy")
    lines.append(template["multi_arch_build"].strip())
    lines.append("```")
    lines.append("")

    lines.append(template["arm_runner_notes"].strip())

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5: estimate_migration_effort — inline data
# ---------------------------------------------------------------------------

MIGRATION_PROFILES: dict[str, dict] = {
    "python_web": {
        "complexity": "low",
        "estimated_phases": [
            {"phase": "Assessment", "duration": "1-2 days", "description": "Audit dependencies and Docker images for ARM compatibility."},
            {"phase": "Environment Setup", "duration": "1 day", "description": "Set up ARM CI runners and ARM development/staging environments."},
            {"phase": "Dependency Resolution", "duration": "1-3 days", "description": "Replace or rebuild any packages lacking arm64 wheels."},
            {"phase": "Testing", "duration": "2-3 days", "description": "Run full test suite on ARM. Profile performance and fix any issues."},
            {"phase": "Deployment", "duration": "1-2 days", "description": "Deploy to ARM instances behind load balancer. Canary rollout recommended."},
        ],
        "risk_areas": [
            "Python C extensions without arm64 wheels (e.g., older versions of psycopg2, mysqlclient).",
            "Docker base images using x86-specific packages or scripts.",
            "Native library dependencies (libxml2, libffi, etc.) that need ARM-compatible system packages.",
            "Machine learning libraries (if used) may need ARM-specific builds.",
            "Performance-sensitive numerical code relying on Intel MKL.",
        ],
        "checklist": [
            "Audit requirements.txt / pyproject.toml for ARM compatibility using scan_x86_dependencies.",
            "Check Docker base image for arm64 support using check_docker_arm_support.",
            "Verify all system-level dependencies (apt/yum packages) are available for arm64.",
            "Build Docker image for linux/arm64 and test locally or in CI.",
            "Run full test suite on ARM (unit, integration, end-to-end).",
            "Profile application performance on ARM vs. x86 (HTTP latency, throughput).",
            "Update CI/CD pipeline for multi-arch builds using generate_ci_matrix.",
            "Configure deployment tooling for ARM instance types.",
            "Plan canary or blue/green deployment strategy.",
            "Monitor for 48-72 hours post-deployment (latency, error rates, resource usage).",
        ],
        "common_blockers": [
            "Older pip packages without arm64 wheels. Fix: upgrade to latest versions or build from source.",
            "Docker images pinned to x86-specific base images. Fix: use multi-arch base images.",
            "Compiled dependencies requiring ARM build tools in Docker. Fix: add build-essential to Dockerfile.",
            "QEMU emulation performance during multi-arch Docker builds. Fix: use native ARM CI runners.",
        ],
        "quick_wins": [
            "Python web apps (Django, Flask, FastAPI) typically work on ARM with zero code changes.",
            "Use multi-arch Docker base images (python:3.12) for immediate arm64 support.",
            "Most popular Python packages (numpy, pandas, requests, etc.) have arm64 wheels.",
            "ARM instances offer 20-40% cost savings for typical web workloads.",
            "HTTP-bound workloads see minimal performance difference between x86 and ARM.",
        ],
        "testing_strategy": "Run existing test suite on ARM. Focus on: integration tests with databases and external services, "
            "load testing to compare throughput, and smoke tests for any C extension functionality. "
            "Use parallel CI to run tests on both architectures during the transition period.",
        "rollback_plan": "Keep x86 deployment running alongside ARM during migration. Use load balancer "
            "weighted routing (e.g., 90/10 split) to gradually shift traffic. If ARM deployment shows "
            "errors or degraded performance, shift traffic back to x86 instances immediately. "
            "Maintain x86 container images for at least 30 days post-migration.",
    },
    "java_enterprise": {
        "complexity": "low",
        "estimated_phases": [
            {"phase": "Assessment", "duration": "1-2 days", "description": "Verify JDK ARM support, audit JNI dependencies and native libraries."},
            {"phase": "Environment Setup", "duration": "1 day", "description": "Set up ARM JDK (Temurin/Corretto), configure CI for ARM."},
            {"phase": "Dependency Resolution", "duration": "1-3 days", "description": "Update JNI libraries and native Maven/Gradle dependencies for ARM."},
            {"phase": "Testing", "duration": "3-5 days", "description": "Full integration test suite on ARM. JVM performance profiling."},
            {"phase": "Deployment", "duration": "2-3 days", "description": "Deploy to ARM instances. Tune JVM flags for ARM (G1/ZGC performs well)."},
        ],
        "risk_areas": [
            "JNI (Java Native Interface) libraries compiled for x86 only.",
            "Native agents (APM, profilers) that need ARM builds (e.g., older Datadog, New Relic agents).",
            "JVM flags or GC tuning specific to x86 micro-architecture.",
            "JDBC drivers with native components (e.g., Oracle OCI driver).",
            "Application servers or middleware with x86-specific optimizations.",
        ],
        "checklist": [
            "Verify JDK distribution supports aarch64 (Eclipse Temurin, Amazon Corretto, Azul Zulu all do).",
            "Audit pom.xml/build.gradle for JNI dependencies (netty-tcnative, snappy, lz4, zstd, RocksDB).",
            "Check APM/monitoring agents for arm64 builds (Datadog, New Relic, Dynatrace).",
            "Build project on ARM and run full test suite.",
            "Profile JVM performance: startup time, throughput, GC pauses.",
            "Review JVM flags — remove x86-specific flags (e.g., -XX:UseAVX), add ARM-friendly flags.",
            "Test JDBC connectivity to databases from ARM instances.",
            "Update Docker base image to ARM-compatible JDK image (eclipse-temurin:21, amazoncorretto:21).",
            "Load test on ARM to establish performance baseline.",
            "Plan rolling deployment with health checks.",
        ],
        "common_blockers": [
            "JNI libraries without ARM builds. Fix: update to latest versions (most now include aarch64).",
            "x86-specific JVM flags causing startup failures. Fix: audit and remove architecture-specific flags.",
            "APM agents lacking ARM support. Fix: upgrade agent or use OpenTelemetry (universal).",
            "Oracle-specific JDBC drivers. Fix: use thin JDBC driver (pure Java) instead of OCI driver.",
        ],
        "quick_wins": [
            "JVM applications are highly portable — most Java code works on ARM with zero changes.",
            "OpenJDK AArch64 JIT is mature and performant (C2 compiler optimized for ARM since JDK 9).",
            "Amazon Corretto is optimized for Graviton with crypto acceleration.",
            "Spring Boot, Quarkus, and Micronaut all work on ARM without modification.",
            "JVM GC (G1, ZGC, Shenandoah) performs well on ARM with large core counts.",
        ],
        "testing_strategy": "Run complete test suite (unit + integration) on ARM JDK. Profile with JFR (Java Flight Recorder) "
            "on both architectures. Compare: startup time, warmup period, steady-state throughput, GC behavior. "
            "Run JMH benchmarks if available. Stress test with production-like load.",
        "rollback_plan": "Deploy ARM instances behind existing load balancer using blue/green strategy. "
            "Keep x86 instances running at full capacity initially. Gradually shift traffic using "
            "weighted routing. Monitor JVM metrics (heap, GC, response time). Rollback by shifting "
            "traffic back to x86 blue pool. Maintain x86 deployment for 2+ weeks.",
    },
    "cpp_native": {
        "complexity": "high",
        "estimated_phases": [
            {"phase": "Assessment", "duration": "3-5 days", "description": "Audit codebase for x86 intrinsics (SSE/AVX), inline assembly, and platform assumptions."},
            {"phase": "Toolchain Setup", "duration": "2-3 days", "description": "Set up ARM cross-compilation toolchain or native ARM build environment."},
            {"phase": "Code Porting", "duration": "5-15 days", "description": "Port x86 SIMD to NEON/SVE, fix endianness/alignment issues, update build system."},
            {"phase": "Testing", "duration": "5-7 days", "description": "Extensive testing: unit, integration, performance benchmarks, numerical correctness."},
            {"phase": "Optimization", "duration": "3-7 days", "description": "Profile on ARM, optimize hot paths with NEON/SVE, tune compiler flags."},
            {"phase": "Deployment", "duration": "2-3 days", "description": "Deploy and validate in production ARM environment."},
        ],
        "risk_areas": [
            "x86 SIMD intrinsics (SSE, SSE2, SSE4, AVX, AVX-512) need NEON/SVE equivalents.",
            "Inline x86 assembly must be rewritten for AArch64 assembly syntax.",
            "Assumptions about cache line size (ARM often 64B but can vary), memory alignment, and atomics.",
            "Third-party libraries compiled for x86 only (Intel IPP, MKL, CUDA).",
            "Compiler-specific x86 builtins (__builtin_ia32_*, _mm_* intrinsics).",
            "Endianness assumptions in serialization code (both ARM and x86 are little-endian, but verify).",
            "Platform-specific preprocessor guards (#ifdef __x86_64__) that exclude ARM code paths.",
            "Pointer size and struct alignment differences between compilers/platforms.",
        ],
        "checklist": [
            "Grep codebase for x86 intrinsics: _mm_, _mm256_, _mm512_, __m128, __m256, __m512.",
            "Grep for inline assembly: __asm__, asm volatile, .intel_syntax.",
            "Grep for platform checks: #ifdef __x86_64__, #if defined(__SSE__), ARCH_X86.",
            "Audit third-party dependencies for ARM support.",
            "Set up cross-compilation toolchain (aarch64-linux-gnu-gcc) or native ARM build.",
            "Update CMakeLists.txt / Makefile for ARM architecture detection and NEON/SVE flags.",
            "Port SSE/AVX intrinsics to NEON using sse2neon.h or manual translation.",
            "Replace Intel IPP/MKL calls with ARM-compatible alternatives (OpenBLAS, ArmPL).",
            "Fix any platform-dependent code (inline assembly, compiler builtins).",
            "Build and run unit tests on ARM — focus on numerical correctness.",
            "Run performance benchmarks and profile hot paths on ARM.",
            "Optimize critical sections with ARM-specific NEON/SVE intrinsics.",
            "Update CI for multi-arch builds.",
            "Validate binary compatibility and ABI correctness.",
        ],
        "common_blockers": [
            "Extensive SSE/AVX intrinsic usage. Fix: use sse2neon.h header for automatic translation, or rewrite with NEON intrinsics.",
            "Intel MKL/IPP dependency. Fix: replace with OpenBLAS, Arm Performance Libraries, or FFTW.",
            "CUDA dependency. Fix: evaluate OpenCL, Vulkan Compute, or restructure for CPU with NEON/SVE.",
            "Inline x86 assembly. Fix: rewrite in AArch64 assembly or C/C++ with intrinsics.",
            "Build system hard-coded for x86. Fix: add proper architecture detection in CMake/Meson.",
        ],
        "quick_wins": [
            "Use sse2neon.h for quick SSE-to-NEON translation (covers most SSE/SSE2 intrinsics).",
            "GCC/Clang -march=armv8-a+simd enables NEON auto-vectorization for many loops.",
            "OpenBLAS is a drop-in replacement for MKL with excellent ARM NEON/SVE kernels.",
            "CMake 3.20+ detects ARM and sets appropriate flags automatically.",
            "Many C/C++ projects 'just work' on ARM if they avoid x86-specific code — try building first.",
        ],
        "testing_strategy": "Build on ARM natively (preferred) or via cross-compilation. Run unit tests with focus on: "
            "numerical correctness (bit-exact comparison where possible, epsilon-based for floating point), "
            "SIMD code paths (test with various input sizes including non-aligned), memory alignment and "
            "atomics correctness. Run AddressSanitizer and UndefinedBehaviorSanitizer on ARM build. "
            "Benchmark against x86 baseline — ARM should be within 80-120% for most workloads.",
        "rollback_plan": "Maintain x86 build artifacts and deployment pipeline throughout migration. "
            "Deploy ARM builds to a separate environment initially. Run shadow traffic or A/B testing "
            "to validate correctness and performance. Keep x86 production deployment as primary until "
            "ARM is validated for at least 2 weeks. Binary rollback: redeploy x86 binaries.",
    },
    "nodejs_api": {
        "complexity": "low",
        "estimated_phases": [
            {"phase": "Assessment", "duration": "1 day", "description": "Check native addons (node-gyp packages) for ARM compatibility."},
            {"phase": "Environment Setup", "duration": "0.5 days", "description": "Set up ARM Node.js environment and CI runners."},
            {"phase": "Dependency Resolution", "duration": "1-2 days", "description": "Update or replace any native addons lacking ARM support."},
            {"phase": "Testing", "duration": "1-2 days", "description": "Run test suite on ARM. Check HTTP performance."},
            {"phase": "Deployment", "duration": "1 day", "description": "Deploy to ARM instances with canary rollout."},
        ],
        "risk_areas": [
            "Native Node.js addons (C++ via node-gyp) that lack pre-built ARM binaries.",
            "node-sass is deprecated and x86-only — must migrate to sass (Dart Sass).",
            "Canvas/image processing libraries requiring system-level ARM packages.",
            "Puppeteer/Playwright with Chromium — ARM Chromium builds may need special setup.",
            "Binary npm packages distributed as platform-specific pre-built binaries.",
        ],
        "checklist": [
            "Audit package.json for native addons (anything using node-gyp, prebuild, node-pre-gyp).",
            "Replace node-sass with sass (Dart Sass) if still in use.",
            "Check sharp, bcrypt, better-sqlite3 — these have ARM pre-built binaries.",
            "Build Docker image for linux/arm64 and verify all npm install steps work.",
            "Run full test suite on ARM.",
            "Load test API endpoints to compare x86 vs ARM performance.",
            "Update CI pipeline for multi-arch testing.",
            "Deploy to ARM instances behind load balancer.",
        ],
        "common_blockers": [
            "node-sass dependency. Fix: migrate to sass (pure JS Dart Sass implementation).",
            "Native addons without ARM prebuild. Fix: add build tools to Docker image for compilation.",
            "Puppeteer requiring Chrome. Fix: use Playwright with ARM Chromium, or run browser tests on x86.",
            "npm ci failing on ARM. Fix: delete package-lock.json and regenerate on ARM, or use --ignore-scripts and build manually.",
        ],
        "quick_wins": [
            "Pure JavaScript/TypeScript APIs work on ARM with zero changes.",
            "Node.js V8 engine is highly optimized for ARM (used in Android for years).",
            "Most popular npm packages (express, fastify, next.js, etc.) are pure JS and ARM-ready.",
            "ARM instances handle Node.js async I/O workloads very efficiently.",
            "esbuild and swc (popular build tools) have native ARM binaries.",
        ],
        "testing_strategy": "Run npm test on ARM. Focus integration tests on: native addon functionality, "
            "database connectivity, external API calls. Load test with autocannon or k6 to compare "
            "throughput and P99 latency between x86 and ARM. Test npm install in fresh ARM environment "
            "to catch any native compilation issues.",
        "rollback_plan": "Deploy ARM instances to existing load balancer pool. Use weighted routing to "
            "gradually shift traffic (10% -> 25% -> 50% -> 100%). Monitor error rates and response "
            "times. Rollback by removing ARM instances from the pool. Keep x86 auto-scaling group "
            "active during transition.",
    },
    "go_microservice": {
        "complexity": "low",
        "estimated_phases": [
            {"phase": "Assessment", "duration": "0.5 days", "description": "Check for CGo dependencies and platform-specific build tags."},
            {"phase": "Cross-Compilation", "duration": "0.5 days", "description": "Build for arm64 using GOARCH=arm64. Test cross-compiled binary."},
            {"phase": "Testing", "duration": "1-2 days", "description": "Run tests on ARM. Verify CGo dependencies if any."},
            {"phase": "Deployment", "duration": "1 day", "description": "Deploy arm64 binary to ARM instances."},
        ],
        "risk_areas": [
            "CGo dependencies requiring ARM C libraries (sqlite3, librdkafka, etc.).",
            "Build tags that exclude ARM (//go:build amd64 or +build amd64).",
            "SIMD-optimized Go assembly files (.s) written for x86.",
            "Third-party Go modules with x86-specific assembly stubs.",
        ],
        "checklist": [
            "Search for CGo usage: grep for 'import \"C\"' in .go files.",
            "Check for x86 build tags: grep for 'go:build amd64' or '+build amd64'.",
            "Check for x86 assembly: look for *_amd64.s files without *_arm64.s counterparts.",
            "Build: GOARCH=arm64 GOOS=linux go build ./...",
            "Run tests: GOARCH=arm64 go test ./... (on ARM or via QEMU).",
            "Build multi-arch Docker image with docker buildx.",
            "Deploy arm64 binary to ARM instance and run smoke tests.",
            "Load test to compare performance.",
        ],
        "common_blockers": [
            "CGo dependency without ARM support. Fix: use pure Go alternative (e.g., modernc.org/sqlite instead of go-sqlite3).",
            "x86-only assembly files. Fix: add arm64 assembly or use pure Go fallback.",
            "Docker image using x86-only base. Fix: use multi-arch base image.",
        ],
        "quick_wins": [
            "Go has first-class arm64 support. Most pure-Go programs compile and run on ARM with zero changes.",
            "Cross-compilation is built in: GOARCH=arm64 GOOS=linux go build",
            "Go standard library is fully ARM-compatible including crypto (hardware-accelerated on ARM).",
            "Static binaries from Go work perfectly with scratch or distroless Docker images on ARM.",
            "Go compiler generates efficient ARM64 code with good register allocation.",
        ],
        "testing_strategy": "Cross-compile with GOARCH=arm64 and run on ARM instance. Run go test ./... with "
            "race detector enabled (-race). Benchmark with go test -bench=. to compare x86 vs ARM. "
            "If using CGo, test native ARM build (not cross-compiled) to verify C library compatibility.",
        "rollback_plan": "Deploy arm64 binary alongside x86 binary using container orchestration (Kubernetes, ECS). "
            "Use service mesh or load balancer for traffic splitting. Rollback by scaling down ARM "
            "pods/tasks and scaling up x86. Maintain both binary artifacts in CI/CD pipeline.",
    },
    "rust_systems": {
        "complexity": "medium",
        "estimated_phases": [
            {"phase": "Assessment", "duration": "1-2 days", "description": "Audit Cargo.toml for -sys crates, platform-specific code, and SIMD usage."},
            {"phase": "Cross-Compilation", "duration": "1-2 days", "description": "Set up cross-compilation target (aarch64-unknown-linux-gnu) or native ARM build."},
            {"phase": "Dependency Resolution", "duration": "1-3 days", "description": "Resolve -sys crates needing ARM libraries. Update conditional compilation."},
            {"phase": "Testing", "duration": "2-3 days", "description": "Run test suite on ARM. Verify unsafe code and FFI correctness."},
            {"phase": "Optimization", "duration": "1-3 days", "description": "Profile on ARM, optimize SIMD-heavy code with NEON intrinsics."},
            {"phase": "Deployment", "duration": "1 day", "description": "Deploy arm64 binary to ARM instances."},
        ],
        "risk_areas": [
            "-sys crates that wrap C libraries needing ARM cross-compilation (lz4-sys, zstd-sys, openssl-sys).",
            "Explicit x86 SIMD intrinsics (std::arch::x86_64) in hot paths.",
            "cfg(target_arch = \"x86_64\") guards excluding ARM code paths.",
            "FFI (Foreign Function Interface) bindings to x86-only C/C++ libraries.",
            "Inline assembly using x86 instructions (asm! macro with x86 syntax).",
            "Build scripts (build.rs) with hard-coded x86 paths or flags.",
        ],
        "checklist": [
            "Add target: rustup target add aarch64-unknown-linux-gnu",
            "Audit Cargo.toml for -sys crates: lz4-sys, zstd-sys, openssl-sys, rdkafka, rocksdb.",
            "Search for x86 SIMD: grep for std::arch::x86_64, _mm_, __m128 in source.",
            "Search for platform cfg: grep for cfg(target_arch = \"x86_64\") in source.",
            "Check for inline assembly: grep for asm! macro usage.",
            "Cross-compile: cargo build --target aarch64-unknown-linux-gnu",
            "Run tests on ARM natively: cargo test",
            "Profile with perf or flamegraph on ARM.",
            "Build multi-arch Docker image.",
            "Deploy and load test on ARM.",
        ],
        "common_blockers": [
            "-sys crates failing to cross-compile. Fix: install ARM cross-compilation libraries (e.g., aarch64-linux-gnu-gcc).",
            "x86 SIMD intrinsics. Fix: use std::arch::aarch64 NEON intrinsics behind cfg, or use portable SIMD (std::simd).",
            "openssl-sys cross-compilation. Fix: use rustls (pure Rust TLS) instead of openssl-sys.",
            "Build scripts assuming x86. Fix: update build.rs to detect and handle ARM architecture.",
        ],
        "quick_wins": [
            "aarch64-unknown-linux-gnu is a Tier 1 Rust target — full support guaranteed.",
            "ring and rustls provide excellent ARM crypto performance without system OpenSSL.",
            "Most -sys crates support ARM when system libraries are available.",
            "Rust compiler generates good ARM64 code with automatic vectorization.",
            "Portable SIMD (std::simd, nightly) works across x86 and ARM.",
        ],
        "testing_strategy": "Cross-compile and run tests on ARM (native preferred over QEMU for accuracy). "
            "Run cargo test with address sanitizer on ARM. Test unsafe code blocks carefully — "
            "alignment and atomics may behave differently. Run cargo bench to compare x86 vs ARM "
            "performance. Verify FFI correctness with integration tests.",
        "rollback_plan": "Maintain x86 binary artifacts in CI/CD. Deploy ARM binaries to separate instance group. "
            "Use load balancer for traffic splitting. Rollback by switching traffic back to x86 instances. "
            "Keep both target compilations in CI for at least 1 month post-migration.",
    },
}

_VALID_PROFILES = {"python_web", "java_enterprise", "cpp_native", "nodejs_api", "go_microservice", "rust_systems"}


@mcp.tool()
def estimate_migration_effort(codebase_profile: str) -> str:
    """Estimate the effort and complexity of migrating a codebase from x86 to ARM.

    Returns a detailed migration assessment including complexity rating, phased
    timeline, risk areas, actionable checklist, common blockers and solutions,
    quick wins, testing strategy, and rollback plan.

    Args:
        codebase_profile: Type of codebase to assess. One of: "python_web",
                          "java_enterprise", "cpp_native", "nodejs_api",
                          "go_microservice", "rust_systems".
    """
    profile_key = codebase_profile.lower().strip()
    if profile_key not in _VALID_PROFILES:
        return f"Error: codebase_profile must be one of {', '.join(sorted(_VALID_PROFILES))}."

    profile = MIGRATION_PROFILES[profile_key]

    _profile_names = {
        "python_web": "Python Web Application",
        "java_enterprise": "Java Enterprise Application",
        "cpp_native": "C++ Native Application",
        "nodejs_api": "Node.js API Service",
        "go_microservice": "Go Microservice",
        "rust_systems": "Rust Systems Application",
    }

    _complexity_emoji = {
        "low": "LOW (straightforward)",
        "medium": "MEDIUM (moderate effort)",
        "high": "HIGH (significant effort)",
    }

    lines = []
    lines.append(f"# Migration Assessment: {_profile_names[profile_key]}")
    lines.append("")

    lines.append(f"## Complexity: {_complexity_emoji[profile['complexity']]}")
    lines.append("")

    # Estimated total duration
    total_min = 0
    total_max = 0
    for phase in profile["estimated_phases"]:
        dur = phase["duration"]
        # Parse "X-Y days" or "X days" or "X day"
        parts = dur.replace(" days", "").replace(" day", "").strip()
        if "-" in parts:
            lo, hi = parts.split("-")
            total_min += float(lo)
            total_max += float(hi)
        else:
            val = float(parts)
            total_min += val
            total_max += val
    lines.append(f"**Estimated total duration: {total_min:.0f}-{total_max:.0f} days**")
    lines.append("")

    lines.append("## Phased Timeline")
    lines.append("")
    lines.append(f"{'Phase':<25} {'Duration':<15} Description")
    lines.append("-" * 80)
    for phase in profile["estimated_phases"]:
        lines.append(f"{phase['phase']:<25} {phase['duration']:<15} {phase['description']}")
    lines.append("")

    lines.append("## Risk Areas")
    for i, risk in enumerate(profile["risk_areas"], 1):
        lines.append(f"  {i}. {risk}")
    lines.append("")

    lines.append("## Migration Checklist")
    for i, item in enumerate(profile["checklist"], 1):
        lines.append(f"  [ ] {i}. {item}")
    lines.append("")

    lines.append("## Common Blockers and Solutions")
    for blocker in profile["common_blockers"]:
        lines.append(f"  - {blocker}")
    lines.append("")

    lines.append("## Quick Wins")
    for win in profile["quick_wins"]:
        lines.append(f"  - {win}")
    lines.append("")

    lines.append("## Testing Strategy")
    lines.append(profile["testing_strategy"])
    lines.append("")

    lines.append("## Rollback Plan")
    lines.append(profile["rollback_plan"])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 6: generate_arm_dockerfile — multi-arch Dockerfile templates
# ---------------------------------------------------------------------------

DOCKERFILE_TEMPLATES: dict[str, dict] = {
    "python": {
        "language": "Python",
        "base_image": "python:3.12-slim",
        "dockerfile": (
            "# Multi-architecture Dockerfile for Python applications\n"
            "# Supports: linux/amd64, linux/arm64\n"
            "# Build: docker buildx build --platform linux/amd64,linux/arm64 -t myapp .\n"
            "\n"
            "FROM python:3.12-slim AS builder\n"
            "\n"
            "WORKDIR /app\n"
            "\n"
            "# Install build dependencies (needed for compiling C extensions on ARM)\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
            "    build-essential \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n"
            "\n"
            "COPY requirements.txt .\n"
            "RUN pip install --no-cache-dir --prefix=/install -r requirements.txt\n"
            "\n"
            "FROM python:3.12-slim\n"
            "\n"
            "WORKDIR /app\n"
            "\n"
            "COPY --from=builder /install /usr/local\n"
            "COPY . .\n"
            "\n"
            'EXPOSE 8000\n'
            'CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]\n'
        ),
        "notes": [
            "python:3.12-slim has official multi-arch support (amd64, arm64, arm/v7).",
            "build-essential is needed for C extensions (e.g., psycopg2, lxml) on ARM.",
            "Multi-stage build keeps final image small — builder stage installs packages, runtime stage copies them.",
            "Use --no-cache-dir with pip to reduce image size.",
            "For intel-mkl dependencies, switch to OpenBLAS: set OPENBLAS_NUM_THREADS=4 in ENV.",
        ],
    },
    "nodejs": {
        "language": "Node.js",
        "base_image": "node:20-slim",
        "dockerfile": (
            "# Multi-architecture Dockerfile for Node.js applications\n"
            "# Supports: linux/amd64, linux/arm64\n"
            "# Build: docker buildx build --platform linux/amd64,linux/arm64 -t myapp .\n"
            "\n"
            "FROM node:20-slim AS builder\n"
            "\n"
            "WORKDIR /app\n"
            "\n"
            "# Install build tools for native addons (sharp, bcrypt, etc.)\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
            "    python3 make g++ \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n"
            "\n"
            "COPY package*.json ./\n"
            "RUN npm ci --only=production\n"
            "\n"
            "FROM node:20-slim\n"
            "\n"
            "WORKDIR /app\n"
            "\n"
            "COPY --from=builder /app/node_modules ./node_modules\n"
            "COPY . .\n"
            "\n"
            'EXPOSE 3000\n'
            'CMD ["node", "server.js"]\n'
        ),
        "notes": [
            "node:20-slim has multi-arch support including arm64.",
            "Native addons (sharp, bcrypt, better-sqlite3) rebuild automatically for the target architecture.",
            "python3, make, g++ are needed for node-gyp to compile native modules on ARM.",
            "sharp uses pre-built ARM binaries (libvips) — no special handling needed.",
            "For esbuild: pre-built arm64 binaries available since v0.14.",
        ],
    },
    "java": {
        "language": "Java",
        "base_image": "eclipse-temurin:21-jre-jammy",
        "dockerfile": (
            "# Multi-architecture Dockerfile for Java applications\n"
            "# Supports: linux/amd64, linux/arm64\n"
            "# Build: docker buildx build --platform linux/amd64,linux/arm64 -t myapp .\n"
            "\n"
            "FROM eclipse-temurin:21-jdk-jammy AS builder\n"
            "\n"
            "WORKDIR /app\n"
            "\n"
            "COPY pom.xml .\n"
            "COPY src ./src\n"
            "\n"
            "# Use Maven wrapper or install maven\n"
            "COPY mvnw .mvn ./\n"
            "RUN chmod +x mvnw && ./mvnw package -DskipTests\n"
            "\n"
            "FROM eclipse-temurin:21-jre-jammy\n"
            "\n"
            "WORKDIR /app\n"
            "\n"
            "COPY --from=builder /app/target/*.jar app.jar\n"
            "\n"
            'EXPOSE 8080\n'
            'CMD ["java", "-jar", "app.jar"]\n'
        ),
        "notes": [
            "Eclipse Temurin (Adoptium) JDK/JRE has excellent arm64 support.",
            "Java bytecode is architecture-independent — no recompilation needed for most apps.",
            "JNI libraries (e.g., RocksDB, Netty native transport, snappy-java) need arm64 native builds.",
            "Graviton3 benefits from OpenJDK's AArch64-specific JIT optimizations (vectorized String ops, crypto intrinsics).",
            "Add -XX:+UseZGC or -XX:+UseShenandoahGC for optimal GC on ARM (large core counts).",
        ],
    },
    "go": {
        "language": "Go",
        "base_image": "golang:1.22-alpine",
        "dockerfile": (
            "# Multi-architecture Dockerfile for Go applications\n"
            "# Supports: linux/amd64, linux/arm64\n"
            "# Build: docker buildx build --platform linux/amd64,linux/arm64 -t myapp .\n"
            "\n"
            "FROM golang:1.22-alpine AS builder\n"
            "\n"
            "WORKDIR /app\n"
            "\n"
            "COPY go.mod go.sum ./\n"
            "RUN go mod download\n"
            "\n"
            "COPY . .\n"
            "RUN CGO_ENABLED=0 go build -ldflags='-s -w' -o /app/server .\n"
            "\n"
            "FROM alpine:3.19\n"
            "\n"
            "RUN apk --no-cache add ca-certificates\n"
            "\n"
            "COPY --from=builder /app/server /server\n"
            "\n"
            'EXPOSE 8080\n'
            'CMD ["/server"]\n'
        ),
        "notes": [
            "Go cross-compiles natively — GOARCH=arm64 produces arm64 binaries from any host.",
            "CGO_ENABLED=0 produces a fully static binary — no libc dependency, works with scratch/distroless.",
            "For CGo dependencies (e.g., sqlite3): use alpine which has musl libc for both amd64 and arm64.",
            "Go's runtime has ARM64-optimized assembly for crypto, math, and memory operations.",
            "Binary size with -ldflags='-s -w' is typically 5-15 MB — excellent for minimal containers.",
        ],
    },
    "rust": {
        "language": "Rust",
        "base_image": "rust:1.77-slim",
        "dockerfile": (
            "# Multi-architecture Dockerfile for Rust applications\n"
            "# Supports: linux/amd64, linux/arm64\n"
            "# Build: docker buildx build --platform linux/amd64,linux/arm64 -t myapp .\n"
            "\n"
            "FROM rust:1.77-slim AS builder\n"
            "\n"
            "WORKDIR /app\n"
            "\n"
            "# Cache dependencies\n"
            "COPY Cargo.toml Cargo.lock ./\n"
            "RUN mkdir src && echo 'fn main() {}' > src/main.rs \\\n"
            "    && cargo build --release && rm -rf src\n"
            "\n"
            "COPY . .\n"
            "RUN cargo build --release\n"
            "\n"
            "FROM debian:bookworm-slim\n"
            "\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
            "    ca-certificates \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n"
            "\n"
            "COPY --from=builder /app/target/release/myapp /usr/local/bin/myapp\n"
            "\n"
            'EXPOSE 8080\n'
            'CMD ["myapp"]\n'
        ),
        "notes": [
            "Rust cross-compiles to aarch64-unknown-linux-gnu with the appropriate target.",
            "Native compilation on ARM is preferred for optimal performance (auto-vectorization, target-cpu=native).",
            "Dependency caching trick (empty main.rs) speeds up rebuilds significantly.",
            "LLVM backend generates high-quality ARM64 code with NEON auto-vectorization.",
            "For musl static builds: use rust:alpine and target aarch64-unknown-linux-musl.",
        ],
    },
}


@mcp.tool()
def generate_arm_dockerfile(language: str) -> str:
    """Generate a multi-architecture Dockerfile template for ARM deployment.

    Returns a production-ready multi-stage Dockerfile that works on both
    x86 (amd64) and ARM (arm64) with buildx, plus ARM-specific notes.

    Args:
        language: Programming language / runtime. One of: "python", "nodejs",
                  "java", "go", "rust". Use "list" or "overview" to see all
                  available templates.
    """
    key = language.lower().strip().replace("-", "").replace("_", "").replace(".", "")

    # Aliases
    aliases = {
        "node": "nodejs",
        "javascript": "nodejs",
        "typescript": "nodejs",
        "ts": "nodejs",
        "js": "nodejs",
        "py": "python",
        "python3": "python",
        "golang": "go",
        "jvm": "java",
        "kotlin": "java",
        "spring": "java",
    }
    key = aliases.get(key, key)

    if key in ("list", "overview", "all"):
        lines = ["# Available Multi-Arch Dockerfile Templates\n"]
        lines.append(f"{'Language':<12} {'Base Image':<35} Notes")
        lines.append("-" * 80)
        for tkey, tmpl in DOCKERFILE_TEMPLATES.items():
            lines.append(f"{tmpl['language']:<12} {tmpl['base_image']:<35} {tmpl['notes'][0][:50]}...")
        lines.append("\nUse `generate_arm_dockerfile(language)` to get the full template.")
        return "\n".join(lines)

    tmpl = DOCKERFILE_TEMPLATES.get(key)
    if tmpl is None:
        available = list(DOCKERFILE_TEMPLATES.keys())
        return (
            f"Error: No Dockerfile template for '{language}'.\n"
            f"Available: {', '.join(available)}, overview"
        )

    lines = [f"# Multi-Arch Dockerfile: {tmpl['language']}"]
    lines.append(f"Base image: {tmpl['base_image']}")
    lines.append("")
    lines.append("## Dockerfile")
    lines.append(f"```dockerfile\n{tmpl['dockerfile']}```")
    lines.append("")
    lines.append("## Build Commands")
    lines.append("```bash")
    lines.append("# Build for both architectures")
    lines.append("docker buildx build --platform linux/amd64,linux/arm64 -t myapp .")
    lines.append("")
    lines.append("# Build for ARM only")
    lines.append("docker buildx build --platform linux/arm64 -t myapp:arm64 .")
    lines.append("")
    lines.append("# Build and push to registry")
    lines.append("docker buildx build --platform linux/amd64,linux/arm64 -t registry/myapp:latest --push .")
    lines.append("```")
    lines.append("")
    lines.append("## ARM-Specific Notes")
    for i, note in enumerate(tmpl["notes"], 1):
        lines.append(f"  {i}. {note}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 7: compare_arm_vs_x86_perf — benchmark comparison data
# ---------------------------------------------------------------------------

ARM_VS_X86_BENCHMARKS: dict[str, dict] = {
    "web_server": {
        "workload": "Web Server (HTTP request/response)",
        "description": "Nginx/Apache serving static content and proxying to app servers. Measured in requests/second and p99 latency.",
        "comparisons": [
            {"arm": "Graviton3 (c7g.xlarge)", "x86": "Intel Ice Lake (c6i.xlarge)", "metric": "Requests/sec (nginx)", "arm_value": "48,000", "x86_value": "41,000", "ratio": "1.17x", "notes": "ARM 17% higher throughput at 20% lower cost"},
            {"arm": "Graviton3 (c7g.xlarge)", "x86": "AMD EPYC (c6a.xlarge)", "metric": "Requests/sec (nginx)", "arm_value": "48,000", "x86_value": "44,000", "ratio": "1.09x", "notes": "ARM 9% higher throughput at ~25% lower cost"},
            {"arm": "Graviton3 (c7g.xlarge)", "x86": "Intel Ice Lake (c6i.xlarge)", "metric": "p99 Latency (ms)", "arm_value": "2.1", "x86_value": "2.4", "ratio": "0.88x", "notes": "ARM 12% lower tail latency"},
        ],
        "cost_savings": "20-40% lower cost per request on Graviton3 vs comparable x86 instances.",
        "key_insight": "Web workloads are typically I/O-bound. ARM's higher core counts and efficiency translate directly to better price-performance.",
    },
    "database": {
        "workload": "Database (OLTP / OLAP)",
        "description": "PostgreSQL, MySQL, Redis performance. Measured in transactions/sec (TPS) and queries/sec.",
        "comparisons": [
            {"arm": "Graviton3 (r7g.xlarge)", "x86": "Intel Ice Lake (r6i.xlarge)", "metric": "PostgreSQL TPS (pgbench)", "arm_value": "12,500", "x86_value": "10,800", "ratio": "1.16x", "notes": "ARM 16% higher TPS. Benefits from DDR5 on Graviton3"},
            {"arm": "Graviton3 (r7g.xlarge)", "x86": "Intel Ice Lake (r6i.xlarge)", "metric": "Redis GET ops/sec", "arm_value": "210,000", "x86_value": "195,000", "ratio": "1.08x", "notes": "ARM 8% higher throughput. Redis is mostly memory-bound"},
            {"arm": "Graviton3 (r7g.2xlarge)", "x86": "Intel Ice Lake (r6i.2xlarge)", "metric": "MySQL sysbench TPS", "arm_value": "18,200", "x86_value": "15,900", "ratio": "1.14x", "notes": "InnoDB benefits from ARM's memory bandwidth"},
        ],
        "cost_savings": "25-35% lower cost per transaction on Graviton3.",
        "key_insight": "Database workloads benefit from Graviton3's DDR5 memory bandwidth and larger L2 caches. PostgreSQL and MySQL are well-optimized for ARM.",
    },
    "ml_inference": {
        "workload": "ML Inference (CPU-based)",
        "description": "TensorFlow/PyTorch inference on CPU. Measured in inferences/second and latency.",
        "comparisons": [
            {"arm": "Graviton3 (c7g.4xlarge)", "x86": "Intel Ice Lake (c6i.4xlarge)", "metric": "ResNet-50 inferences/sec (INT8)", "arm_value": "320", "x86_value": "285", "ratio": "1.12x", "notes": "Graviton3 benefits from BF16 and int8 dot-product instructions"},
            {"arm": "Graviton3 (c7g.4xlarge)", "x86": "Intel Ice Lake (c6i.4xlarge)", "metric": "BERT-base latency (ms)", "arm_value": "45", "x86_value": "52", "ratio": "0.87x", "notes": "13% lower latency. Use amazon-tensorflow for best ARM perf"},
            {"arm": "Neoverse V2 (c8g.4xlarge)", "x86": "Intel Sapphire Rapids (c7i.4xlarge)", "metric": "ResNet-50 inferences/sec (BF16)", "arm_value": "480", "x86_value": "420", "ratio": "1.14x", "notes": "SVE2 + BF16 gives ARM edge in quantized inference"},
        ],
        "cost_savings": "30-45% lower cost per inference on Graviton3 vs comparable x86.",
        "key_insight": "ARM CPUs with BF16/INT8 dot-product instructions are competitive with x86 AVX-512 for inference. Use ARM-optimized frameworks (amazon-tensorflow, ACL-backed PyTorch).",
    },
    "ci_cd": {
        "workload": "CI/CD Build & Test",
        "description": "Compilation, test execution, Docker image builds. Measured in build time and cost per build minute.",
        "comparisons": [
            {"arm": "Graviton3 (c7g.2xlarge)", "x86": "Intel Ice Lake (c6i.2xlarge)", "metric": "C++ compile time (large project)", "arm_value": "142 sec", "x86_value": "158 sec", "ratio": "0.90x", "notes": "ARM 10% faster compilation"},
            {"arm": "Graviton3 (c7g.2xlarge)", "x86": "Intel Ice Lake (c6i.2xlarge)", "metric": "Go build time", "arm_value": "38 sec", "x86_value": "42 sec", "ratio": "0.90x", "notes": "Go compiler well-optimized for ARM"},
            {"arm": "Graviton3 (c7g.2xlarge)", "x86": "Intel Ice Lake (c6i.2xlarge)", "metric": "Java Maven build", "arm_value": "95 sec", "x86_value": "102 sec", "ratio": "0.93x", "notes": "JVM startup slightly slower, compilation comparable"},
            {"arm": "GitHub ARM runner", "x86": "GitHub x64 runner", "metric": "Cost per build minute", "arm_value": "$0.005", "x86_value": "$0.008", "ratio": "0.63x", "notes": "37% cheaper on ARM runners"},
        ],
        "cost_savings": "35-40% lower CI/CD costs with ARM runners (GitHub Actions, GitLab).",
        "key_insight": "ARM CI runners are significantly cheaper. Most build workloads are CPU-bound and perform comparably or faster on Graviton3.",
    },
    "hpc": {
        "workload": "HPC / Scientific Computing",
        "description": "Dense linear algebra, FFT, simulations. Measured in GFLOPS and time-to-solution.",
        "comparisons": [
            {"arm": "Graviton3 (hpc7g.16xlarge)", "x86": "Intel Ice Lake (hpc6i.32xlarge)", "metric": "DGEMM GFLOPS/core", "arm_value": "38.4", "x86_value": "35.2", "ratio": "1.09x", "notes": "Graviton3 SVE (256-bit) vs AVX-512. ArmPL BLAS."},
            {"arm": "Graviton3 (hpc7g.16xlarge)", "x86": "Intel Ice Lake (hpc6i.32xlarge)", "metric": "HPL GFLOPS (cluster)", "arm_value": "2,850", "x86_value": "3,100", "ratio": "0.92x", "notes": "x86 slightly ahead in peak FLOPS. ARM wins on price-performance."},
            {"arm": "Neoverse V2 (SVE 128-bit)", "x86": "AMD EPYC Genoa (AVX-512)", "metric": "WRF Weather Model", "arm_value": "1.05x", "x86_value": "1.00x (baseline)", "ratio": "1.05x", "notes": "ARM competitive with SVE auto-vectorization"},
        ],
        "cost_savings": "15-30% lower cost for HPC workloads on Graviton3. Higher savings for memory-bandwidth-bound codes.",
        "key_insight": "For peak FLOPS, x86 with AVX-512 still has an edge. But for price-performance and energy efficiency, ARM (especially with SVE/SVE2) is highly competitive. Use ArmPL or OpenBLAS for best BLAS/LAPACK performance.",
    },
}


@mcp.tool()
def compare_arm_vs_x86_perf(workload: str) -> str:
    """Compare ARM vs x86 performance and cost for a specific workload type.

    Returns benchmark data showing throughput, latency, and cost comparisons
    between ARM (Graviton, Neoverse) and x86 (Intel, AMD) instances.

    Args:
        workload: Type of workload. One of: "web_server", "database",
                  "ml_inference", "ci_cd", "hpc". Use "list" or "overview"
                  to see all available workloads.
    """
    key = workload.lower().strip().replace(" ", "_").replace("-", "_")

    # Aliases
    aliases = {
        "web": "web_server",
        "http": "web_server",
        "nginx": "web_server",
        "db": "database",
        "postgres": "database",
        "postgresql": "database",
        "mysql": "database",
        "redis": "database",
        "ml": "ml_inference",
        "inference": "ml_inference",
        "ai": "ml_inference",
        "ci": "ci_cd",
        "build": "ci_cd",
        "compile": "ci_cd",
        "scientific": "hpc",
        "compute": "hpc",
    }
    key = aliases.get(key, key)

    if key in ("list", "overview", "all"):
        lines = ["# ARM vs x86 Performance Comparison\n"]
        lines.append("Available workload profiles:\n")
        for wkey, wdata in ARM_VS_X86_BENCHMARKS.items():
            lines.append(f"  **{wkey}**: {wdata['workload']}")
            lines.append(f"    Cost savings: {wdata['cost_savings']}")
            lines.append("")
        lines.append("Use `compare_arm_vs_x86_perf(workload)` for detailed benchmarks.")
        return "\n".join(lines)

    benchmark = ARM_VS_X86_BENCHMARKS.get(key)
    if benchmark is None:
        available = list(ARM_VS_X86_BENCHMARKS.keys())
        return (
            f"Error: No benchmark data for workload '{workload}'.\n"
            f"Available: {', '.join(available)}, overview"
        )

    lines = [f"# ARM vs x86: {benchmark['workload']}"]
    lines.append(f"{benchmark['description']}\n")

    # Comparison table
    lines.append("## Benchmarks\n")
    for comp in benchmark["comparisons"]:
        lines.append(f"### {comp['metric']}")
        lines.append(f"  ARM: {comp['arm']}  →  **{comp['arm_value']}**")
        lines.append(f"  x86: {comp['x86']}  →  **{comp['x86_value']}**")
        lines.append(f"  Ratio (ARM/x86): **{comp['ratio']}**")
        lines.append(f"  Notes: {comp['notes']}")
        lines.append("")

    lines.append(f"## Cost Savings\n{benchmark['cost_savings']}")
    lines.append(f"\n## Key Insight\n{benchmark['key_insight']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run()


if __name__ == "__main__":
    main()
