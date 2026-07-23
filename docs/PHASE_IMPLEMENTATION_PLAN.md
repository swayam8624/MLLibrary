# Kairo ML Phase Implementation Plan

This is the execution order for the native C++ ML stack. A phase is complete
only when its exit gate is demonstrated by tests and machine-readable evidence.

## Phase 0: Contracts And Measurement

**Delivered:** tensor layout/dtype/device/ownership contracts, typed device
action and rollback contracts, model capability registry, versioned formats,
reproducible seeds, benchmark JSON, dependency graph, and platform matrix.

**Exit state:** shared public contracts and phase APIs document input, output,
preconditions, failure, and ownership; machine-readable benchmarks and
reproducible manifests are exported. API review must preserve this contract for
new operations.

## Phase 1: Tensor And Training Foundation

**Delivered and verified:** dynamic tensors/views, broadcasting, Float32,
Float16, BFloat16, Int8/index dtypes, elementwise/reduction kernels,
matmul/batched matmul, convolution/pooling/normalization, reverse autodiff,
gradient lifecycle, layers/losses, SGD family/RMSProp/Adam/AdamW, schedules,
clipping, mixed precision, versioned full-state checkpoints, dataset
shuffle/split/batching/prefetch, metrics, and gradient checks.

**Exit evidence:** MLP and CNN end-to-end training, finite-difference agreement,
and bit-equivalent interrupted/resumed optimization all pass in KairoMath.

## Phase 1.5: CPU Performance

**Delivered:** dependency scheduling, task cancellation, deterministic parallel
reductions, scalar/NEON runtime dispatch, packed and batched matmul, scheduled
convolution, and scalar-parity integration.

**Exit state:** targeted AVX2/AVX-512 and NEON kernels retain scalar fallback;
fused bias-activation, LayerNorm, and optimizer kernels exist; ASan/TSan pass;
benchmark JSON and configurable runner-specific throughput floors are enforced.
Native x86 CI remains necessary to observe AVX dispatch rather than Rosetta's
scalar emulation.

## Phase 1.75: ONNX Import

**Delivered:** bounded protobuf import, malformed-wire validation, opset/domain
metadata, Float32/Int64 initializers, graph validation, native execution for the
declared MLP/CNN/indexing operator set, static inference for core compute ops,
constant folding, and dead-value elimination.

**Exit state:** static inference includes data-driven indexing and reshape;
serialized MLP, CNN, and transformer-attention models match pinned ONNX Runtime
outputs; malformed and unsupported models fail with stable diagnostics.

## Phase 1.9: Transformer Runtime

**Delivered:** embeddings, sinusoidal/RoPE positions, LayerNorm/RMSNorm, fused
QKV projection, stable causal attention, GQA/MQA, decoder model, incremental KV
cache, all requested sampling modes, tokenizer interface, byte tokenizer,
bounded archive/layer streaming, INT8/INT4, autodiff training, accumulation,
exact checkpoint resume, LoRA, corpus overfit, and benchmark JSON.

**Exit state:** grouped-query caches store only KV heads; per-layer activation
recomputation matches retained-graph gradients; complete bounded decoder
checkpoints reload under per-tensor budgets and reproduce greedy generation;
cached/full decoding agree; corpus overfit and benchmark evidence pass.

## Phase 2: Visual Computing

The Phase 1.5, 1.75, and 1.9 implementation gates are closed. Phase 2 can now
start from these tested contracts.

1. Stream training events and tensor probes over a versioned local protocol.
2. Use the iPad as an optional supervision/control surface while Mac rendering
   and application automation remain local.
3. Add image/video ingestion, augmentation, batching, and dataset inspection.
4. Add feature maps, saliency, Grad-CAM, confusion matrices, and graph/kernel
   profiling.
5. Keep every device action transactional: preview, approval, verification,
   undo, and immutable replay records.

The target is a local multimodal learning and device-interaction runtime, not a
large-language-model wrapper: compact task models perceive visual/audio state,
the deterministic planner maps observations to typed reversible actions, and
larger models remain optional bounded advisors.
