# Runtime Contracts

## Scope

Phase 0 defines shared boundaries used by MLLibrary, KairoMath, KairoGPU,
KairoONNX, KairoTransformers, and KairoAI. A backend may optimize an operation,
but it must not reinterpret shapes, layouts, ownership, action safety, or file
versions.

## Tensor Contract

`mllibrary::contracts::TensorDescriptor` records:

- non-empty, non-zero logical shape;
- optional physical strides in element units;
- data type and element byte width;
- row-major, NCHW, NHWC, OHWI, or explicit-strided layout;
- scalar, threaded, SIMD, Metal, Vulkan, CUDA, or WebGPU device;
- owned, shared-view, borrowed, memory-mapped, or device-owned storage.

Shape and byte products are checked for `size_t` overflow. Rank-four image and
filter layouts are rejected at other ranks. Explicit-strided layout requires
strides matching rank. Device-owned storage is rejected for CPU devices.

The descriptor does not allocate or transfer memory. The owner represented by
the associated runtime object remains responsible for lifetime and
synchronization.

## Error Contract

Contract violations throw `ContractError` with a stable `ContractErrorCode`.
Programmer-independent data errors therefore remain distinguishable from
allocation, backend, parser, and application failures. Hot Tensor kernels may
still use their existing specialized exceptions after descriptor validation.

## Model Capabilities

`ModelCapabilityRegistry` binds a stable model id to task, accepted dtypes,
devices, maximum rank/input bytes, and inference/training support. Registration
rejects incomplete and duplicate entries. Selection is deterministic because
the registry is ordered by model id before capability matching.

## Format Versions

- Dataset metadata contract: version 1.
- Legacy arena-model parameter checkpoint: version 1, parameters only.
- KairoAI action contract: version 1.
- Benchmark JSON report: version 1.

Optimizer and random-state persistence must introduce a compatible checkpoint
version rather than silently changing version 1.

## Transactional Actions

`Kairo.AI.DeviceAgent` binds each action to:

- risk and approval policy;
- observable state preconditions;
- expected effects and targets;
- reversibility mode and rollback adapter operation;
- pre-action snapshot requirement;
- bounded verification effects and timeout.

The contract is copied from registration into the proposal and replay record.
Models and application adapters cannot weaken it. Irreversible actions require
destructive classification and independent-device approval.

## Repository Dependencies

```text
KairoScheduler ─┐
KairoSIMD ──────┼─> KairoMath.Tensor
KairoGPU ───────┘          │
                           ├─> MLLibrary
KairoONNX ─────────────────┤
KairoTransformers ─────────┘

KairoMacPerception ─> KairoAI.DeviceAgent ─> application adapters
                                           └> iPad supervision
```

Infrastructure repositories do not depend on MLLibrary. MLLibrary composes
them through optional CMake targets so scalar builds remain available.

## Supported Matrix

| Surface | macOS arm64 | Other platforms |
| --- | --- | --- |
| Scalar C++ runtime | Verified | Contracted, CI pending |
| KairoScheduler CPU | Verified | Contracted, CI pending |
| KairoSIMD NEON | Verified | Scalar fallback |
| KairoSIMD AVX2/AVX-512 | Cross-compiled, Rosetta scalar fallback | Native x86 execution observation pending |
| KairoGPU Metal | Verified | Not available |
| Vulkan/CUDA/WebGPU | Contract only | Not implemented |
| KairoMath C++ modules | LLVM Clang verified | Compiler-dependent |
| KairoAI device contracts | LLVM Clang verified | Platform-neutral core |

## Measurement Contract

`MLLibraryBenchmark` writes a versioned JSON document containing:

- reproducible seed and arguments;
- compiler and platform;
- warmup and measured iteration counts;
- mean, p50, and p95 latency;
- operation throughput;
- peak resident memory;
- maximum absolute numerical error.

Tests execute a bounded matrix fixture and validate report generation. Benchmark
comparisons must use matching schema, arguments, compiler mode, and hardware.
