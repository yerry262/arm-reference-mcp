---
title: ARM Cloud Migration Advisor MCP
---

# ARM Cloud Migration Advisor MCP

Plan and execute x86-to-ARM cloud migrations with confidence. **7 tools** covering dependency analysis, instance selection, Docker compatibility, CI/CD generation, migration planning, Dockerfile generation, and performance benchmarking.

**Server entry point:** `arm-cloud-migration-mcp`

---

## Tool Reference

| Tool | Description |
|------|-------------|
| `scan_x86_dependencies(language, dependencies)` | Analyze a comma-separated dependency list for ARM compatibility. Reports which packages have arm64 support, which are x86-only, and suggests alternatives. Languages: `python`, `nodejs`, `java`, `cpp`, `rust`, `go`. Returns a migration readiness score. |
| `suggest_arm_cloud_instance(workload_profile, provider?)` | Recommend ARM instances across AWS (Graviton), Azure (Cobalt), GCP (Axion), and Oracle (Ampere). Workloads: `web_server`, `database`, `ml_inference`, `ci_cd`, `hpc`, `general`. Includes vCPUs, RAM, pricing, and monthly cost estimates. |
| `check_docker_arm_support(image_name)` | Check if a Docker base image supports arm64/aarch64. Reports multi-arch manifest status, ARM-specific performance notes, known issues, and pull commands. |
| `generate_ci_matrix(ci_platform, language?)` | Generate cross-architecture CI config for building and testing on both x86 and ARM. Platforms: `github_actions`, `gitlab_ci`, `circleci`, `jenkins`. Includes multi-arch Docker build steps. |
| `estimate_migration_effort(codebase_profile)` | Assess migration complexity with a phased timeline, risk areas, checklist, common blockers and solutions, quick wins, testing strategy, and rollback plan. Profiles: `python_web`, `java_enterprise`, `cpp_native`, `nodejs_api`, `go_microservice`, `rust_systems`. |
| `generate_arm_dockerfile(language)` | Generate a multi-stage ARM-optimized Dockerfile. Languages: `python`, `nodejs`/`node`, `java`, `go`, `rust`. Includes ARM base images, platform-specific build flags, and deployment best practices. |
| `compare_arm_vs_x86_perf(workload)` | Real-world benchmark comparisons for ARM vs x86. Workloads: `web_server`, `database`, `ml_inference`, `ci_cd`, `hpc`. Shows specific instances, metrics, performance ratios, and cost savings. |

---

## Examples

### Scanning dependencies

```
> scan_x86_dependencies("python", "numpy,scipy,intel-mkl,tensorflow,pandas")

# x86 Dependency Scan: python (5 packages)

## Fully Compatible (4 packages)
  numpy    -- Native arm64 wheels available (uses OpenBLAS on ARM)
  scipy    -- Native arm64 wheels available
  tensorflow -- ARM-optimized builds available (tensorflow-aarch64)
  pandas   -- Native arm64 wheels available

## x86-Only (1 package)
  intel-mkl -- x86-only (Intel proprietary)
    Alternatives: OpenBLAS, Arm Performance Libraries (ArmPL), BLIS

## Summary
  Compatible: 4/5 (80%)
  Needs work: 1/5 (20%)
  Migration Readiness: 80/100 (HIGH)
```

### Recommending instances

```
> suggest_arm_cloud_instance("database", provider="aws")

# ARM Cloud Instances: database (AWS)

## Recommended Instances
  r7g.large    -- 2 vCPU, 16 GB RAM, $0.1008/hr (~$73/mo)
                  Graviton3, great for memory-intensive DB workloads
  r7g.xlarge   -- 4 vCPU, 32 GB RAM, $0.2016/hr (~$147/mo)
                  Mid-size production databases
  r7g.2xlarge  -- 8 vCPU, 64 GB RAM, $0.4032/hr (~$294/mo)
                  Large production workloads
  im4gn.large  -- 2 vCPU, 8 GB RAM + NVMe SSD, $0.1002/hr
                  I/O-intensive: Redis, Elasticsearch

## Cost Savings vs x86
  20-40% lower cost vs equivalent r6i (Intel) instances
```

### Checking Docker support

```
> check_docker_arm_support("postgres")

# Docker ARM Support: postgres

## Status: SUPPORTED
  Multi-arch manifest: Yes (linux/amd64, linux/arm64, linux/arm/v7)
  ARM base image: arm64v8/postgres

## Performance Notes
  Up to 30% better price-performance on Graviton3 vs x86

## Known Issues
  Extensions requiring native compilation may need ARM build tools

## Pull Commands
  docker pull postgres                              # auto-selects platform
  docker pull --platform linux/arm64 postgres       # force ARM
```

### Generating CI config

```
> generate_ci_matrix("github_actions", language="python")

# GitHub Actions CI Matrix (Python + ARM)

name: CI
on: [push, pull_request]
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, ubuntu-24.04-arm]
        python-version: ['3.12']
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[test]"
      - run: pytest

  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: docker/setup-qemu-action@v3
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v5
        with:
          platforms: linux/amd64,linux/arm64
          push: false
```

### Estimating migration effort

```
> estimate_migration_effort("python_web")

# Migration Assessment: python_web
Complexity: LOW
Timeline: 6-11 days

## Phase Breakdown
  1. Assessment (1-2 days)     -- Dependency audit, compatibility check
  2. Environment Setup (1 day) -- ARM CI runner, Docker multi-arch
  3. Dependencies (2-3 days)   -- Replace x86-only packages
  4. Testing (1-3 days)        -- Functional + performance validation
  5. Deployment (1-2 days)     -- Staged rollout with rollback plan

## Quick Wins
  - Most Python packages have arm64 wheels
  - Docker multi-arch builds handle most cases
  - 20-40% cost savings on Graviton instances
```

### ARM vs x86 benchmarks

```
> compare_arm_vs_x86_perf("web_server")

# ARM vs x86 Performance: web_server

## Benchmark: Nginx Static Throughput
  ARM:  c7g.xlarge (Graviton3)  -- 145,000 RPS
  x86:  c6i.xlarge (Ice Lake)   -- 116,000 RPS
  Ratio: 1.25x (ARM 25% faster)
  Cost: ARM $0.145/hr vs x86 $0.170/hr (15% cheaper)

## Benchmark: Node.js HTTP
  ARM:  c7g.xlarge              -- 52,000 RPS
  x86:  c6i.xlarge              -- 40,000 RPS
  Ratio: 1.30x (ARM 30% faster)

## Key Insight
  Web workloads see strong gains on Graviton3 due to
  higher memory bandwidth and efficient branch prediction.
  20-40% total cost savings when combining perf + pricing.
```

---

## Quick Setup

```bash
claude mcp add --transport stdio arm-cloud-migration -- \
  uvx --from "git+https://github.com/yerry262/arm-reference-mcp.git" arm-cloud-migration-mcp
```

See the full [Installation Guide](../installation) for other editors and clients.
