# Kairo ML Phase Implementation Plan

This is the execution order for the native C++ ML stack. A phase is complete
only when its exit gate is demonstrated by tests and machine-readable evidence.

## Phase 0: Contracts And Measurement

**Delivered:** tensor layout/dtype/device/ownership contracts, typed device
action and rollback contracts, model capability registry, versioned formats,
reproducible seeds, benchmark JSON, dependency graph, and platform matrix.

**Closure work:** finish teaching-style Input, Output, Preconditions, Failure,
and Ownership documentation on every public module operation. Enforce this in
API review rather than treating repository-level prose as sufficient.

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

**Next implementation order:**

1. Add x86 AVX2 kernels and optional AVX-512 kernels behind runtime CPUID.
2. Add fused bias-activation, normalization, and optimizer update kernels.
3. Persist benchmark baselines keyed by CPU/compiler/build arguments.
4. Fail CI on statistically meaningful regression, not one noisy sample.
5. Run AddressSanitizer and ThreadSanitizer configurations in supported CI.

**Exit gate:** all accelerated outputs match scalar references; model code is
backend-independent; ASan/TSan pass; benchmark regression checks are enforced.

## Phase 1.75: ONNX Import

**Delivered:** bounded protobuf import, malformed-wire validation, opset/domain
metadata, Float32/Int64 initializers, graph validation, native execution for the
declared MLP/CNN/indexing operator set, static inference for core compute ops,
constant folding, and dead-value elimination.

**Next implementation order:**

1. Extend data-driven shape inference for Reshape/Gather/Slice/Concat.
2. Add representative small-transformer graph fixtures.
3. Generate fixture outputs with a pinned ONNX Runtime tool environment.
4. Compare Kairo outputs by dtype-specific absolute/relative tolerances.
5. Expand typed initializers and opset-specific semantics.

**Exit gate:** MLP, CNN, and transformer imports match trusted outputs;
malformed and unsupported models fail with stable diagnostics.

## Phase 1.9: Transformer Runtime

**Delivered:** embeddings, sinusoidal/RoPE positions, LayerNorm/RMSNorm, fused
QKV projection, stable causal attention, GQA/MQA, decoder model, incremental KV
cache, all requested sampling modes, tokenizer interface, byte tokenizer,
bounded archive/layer streaming, INT8/INT4, autodiff training, accumulation,
exact checkpoint resume, LoRA, corpus overfit, and benchmark JSON.

**Next implementation order:**

1. Store only key/value heads in grouped-query caches.
2. Add activation-recomputation policies to the training graph.
3. Add a versioned external checkpoint name/shape mapping layer.
4. Load a compact imported checkpoint and prove deterministic generation.
5. Add larger fixed-hardware benchmark histories without making them universal.

**Exit gate:** imported-checkpoint generation is deterministic; cached and full
decoding agree; a small corpus overfits; throughput, memory, and numerical
equivalence are recorded.

## Phase 2: Visual Computing

Start only after the open Phase 1.5, 1.75, and 1.9 gates above are closed.

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
