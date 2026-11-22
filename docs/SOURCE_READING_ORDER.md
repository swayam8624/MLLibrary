# MLLibrary — guided reading order and repository teaching map

MLLibrary currently has **two distinct implementation routes**, and it is important not to conflate them. The running MNIST application uses traditional C++ source/header files and its own arena-backed matrix/autograd implementation. Optional integration with KairoMath compiles C++23 modules when `MLLIB_USE_KAIRO_MATH=ON`. The presence of declarations in `ModuleInterfaces/` does not mean those files replace the normal native application implementation in every configuration.

**Coverage status:** [See which functions are annotated and what is still pending](SOURCE_TEACHING_PROGRESS.md) · [Understand checkpoint weight-only semantics and reproducibility](CHECKPOINT_AND_REPRODUCIBILITY_LAB.md).

**Begin with:** [the complete MNIST training walkthrough](MNIST_TRAINING_FROM_FIRST_PRINCIPLES.md). It connects the source files by data ownership, API calls, computations, failure paths and testability.

## Stage 1 — build and program entry

1. [`CMakeLists.txt`](../CMakeLists.txt) — identify `MLLibraryCore`, `MLLibrary`, test and report targets. Follow the optional KairoMath/GPU configuration separately.
2. [`README.md`](../README.md) — distinguish present capabilities from the longer-term roadmap.
3. [`MLLibrary/main.cpp`](../MLLibrary/main.cpp) — trace initialization, dataset loading, architecture construction and training configuration. The visible classifier topology is `784 -> 16 -> 16 -> 10`, with a residual addition between its hidden layers.
4. [`MLLibrary/Core/base.h`](../MLLibrary/Core/base.h) — study numeric aliases, macros and platform contracts before reading memory code.

## Stage 2 — memory and randomness

5. [`MLLibrary/Core/arena.h`](../MLLibrary/Core/arena.h) and [`arena.cpp`](../MLLibrary/Core/arena.cpp) — virtual memory reserve/commit, alignment, bump allocation, scratch lifetime, invalidation and ownership.
6. [`MLLibrary/Core/prng.h`](../MLLibrary/Core/prng.h) and [`prng.cpp`](../MLLibrary/Core/prng.cpp) — seeded PCG-style state transition, integer sampling, float conversion and global-wrapper state. Distinguish deterministic repeatability from thread-safe use.

**Checkpoint:** explain who owns a `Matrix`'s data after `mat_create`, when its pointer becomes invalid, and why training reproducibility requires preserving the RNG state and data ordering.

## Stage 3 — numerical primitives and autograd

7. [`MLLibrary/Math/matrix.hpp`](../MLLibrary/Math/matrix.hpp) and [`matrix.cpp`](../MLLibrary/Math/matrix.cpp) — row-major storage, basic operations, multiplication, activations, softmax, cross-entropy, and derivative kernels. Identify shape validation at every API boundary.
8. [`MLLibrary/Model/model.hpp`](../MLLibrary/Model/model.hpp) and [`model.cpp`](../MLLibrary/Model/model.cpp) — `ModelContext`, `ModelVar`, node flags, graph edges, operation types, the forward program and reverse-mode gradient propagation. Know which nodes require gradients and which ones represent labels.
9. [`MLLibrary/Training/training.hpp`](../MLLibrary/Training/training.hpp) and [`training.cpp`](../MLLibrary/Training/training.cpp) — request validation, mini-batches, optimizer state, gradient clipping, metrics and return results.
10. [`MLLibrary/Training/checkpoint.cpp`](../MLLibrary/Training/checkpoint.cpp) — inspect serialization and restoration of model/optimizer/progress before relying on resumed training.
11. [`MLLibrary/Data/data.cpp`](../MLLibrary/Data/data.cpp) — binary dataset input and label conversion.

**Checkpoint:** trace one sample through `input -> matmul -> bias -> ReLU -> residual -> logits -> softmax -> cross-entropy`, then reverse through gradients to parameter updates. Explain why class labels are not trainable and which matrices require gradient storage.

## Stage 4 — ML experiments and supporting tools

12. [`MLLibrary/Data/dataset.cpp`](../MLLibrary/Data/dataset.cpp) and [`profiling.cpp`](../MLLibrary/Data/profiling.cpp) — dataset and measurement utility paths.
13. [`MLLibrary/Preprocessing/pipeline.cpp`](../MLLibrary/Preprocessing/pipeline.cpp), [`Classical/classical.cpp`](../MLLibrary/Classical/classical.cpp) — feature preparation and classical algorithms.
14. [`MLLibrary/Evaluation/metrics.cpp`](../MLLibrary/Evaluation/metrics.cpp), [`cross_validation.cpp`](../MLLibrary/Evaluation/cross_validation.cpp) — evaluation and split protocols.
15. [`MLLibrary/Experiment/experiment.cpp`](../MLLibrary/Experiment/experiment.cpp), [`Search/search.cpp`](../MLLibrary/Search/search.cpp), [`AutoML/automl.cpp`](../MLLibrary/AutoML/automl.cpp) — reproducibility, search and selection.
16. [`MLLibrary/Benchmark/benchmark.cpp`](../MLLibrary/Benchmark/benchmark.cpp), [`matrix_benchmark.cpp`](../MLLibrary/Benchmark/matrix_benchmark.cpp) — benchmark definitions, correctness reference and reporting.
17. [`MLLibrary/Visual/training_report.cpp`](../MLLibrary/Visual/training_report.cpp) and [`dataset_report.cpp`](../MLLibrary/Visual/dataset_report.cpp) — reproducible self-contained HTML reports and dataset inspection.

## Stage 5 — module and cross-repository integration

18. [`ModuleInterfaces/`](../ModuleInterfaces) — inspect interface declarations and compare them with the actual CMake target sources. Do not assume an interface-only declaration is the implementation used by the default app.
19. [`MLLibrary/TensorRuntime.cppm`](../MLLibrary/TensorRuntime.cppm) — optional KairoMath module-based execution boundary.
20. [`MLLibrary/TensorGPURuntime.cppm`](../MLLibrary/TensorGPURuntime.cppm) — optional GPU integration, only when enabled and supported.
21. [`Tests/tensor_integration.cpp`](../Tests/tensor_integration.cpp), [`Tests/tensor_gpu_integration.cpp`](../Tests/tensor_gpu_integration.cpp), and [`Tests/contracts_module.cpp`](../Tests/contracts_module.cpp) — check the exact optional module and GPU evidence.

**Cross-repository route:** [KairoMath's tensor execution lab](https://github.com/swayam8624/KairoMath/blob/master/docs/TENSOR_EXECUTION_END_TO_END_LAB.md) -> [KairoScheduler's concurrency lab](https://github.com/swayam8624/KairoScheduler/blob/main/docs/ADVANCED_CONCURRENCY_LAB.md) -> [KairoSIMD's dispatch lab](https://github.com/swayam8624/KairoSIMD/blob/main/docs/ADVANCED_SIMD_API_AND_DISPATCH_LAB.md). The classic MNIST route does **not** automatically use the SIMD/Scheduler implementation just because these repositories exist.

## Verification and annotation standards

- For each function, document **inputs, ownership, shape/range assumptions, side effects, failures and downstream callers**. Never replace behavioral explanations with a generic “this function does X” comment.
- Distinguish an invalid return value, a thrown exception, a debug assertion, and an unchecked precondition; these are not interchangeable.
- Trace memory lifetime and ownership from `MemArena::create` to `destroy`, including temporary buffers and scratch arenas.
- Record the compiler, commit, host, dataset provenance and exact tests before claiming a verified result. Source inspection and documentation do not prove production readiness or performance.
- Treat external datasets, borrowed snippets, dependencies and media under their own original licenses; check the repository's [license](../LICENSE) and provenance before redistribution.

**Documentation snapshot:** built from repository files reviewed on 2026-11-22. Source links and future additions need re-auditing as the repo evolves.
