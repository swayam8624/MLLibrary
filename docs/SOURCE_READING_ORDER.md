# MLLibrary: linked source reading order

This is the recommended **conceptual reading order**, not a requirement to compile in this sequence. Most of the current executable/library implementation uses classic C++ headers and sources. The files under `ModuleInterfaces/` express a module-oriented public API; consult [README](../README.md) and [CMakeLists](../CMakeLists.txt) for the active build targets before assuming that every module is linked into the demo.

## Foundations: own your data before training models

1. [README](../README.md) and [CMakeLists](../CMakeLists.txt) — understand intended scope, C++23 build options, optional external Kairo integrations and test setup.
2. [base.h](../MLLibrary/base.h) — basic types and common utilities.
3. [Core/arena.h](../MLLibrary/Core/arena.h) → [arena.cpp](../MLLibrary/Core/arena.cpp) — reserve/commit virtual memory and allocate temporary model/work buffers through an arena; `MemArena::Temp` expresses scoped rollback.
4. [Core/prng.h](../MLLibrary/Core/prng.h) → [prng.cpp](../MLLibrary/Core/prng.cpp) — deterministic pseudo-random state used for initialization or sampling.
5. [Math/matrix.hpp](../MLLibrary/Math/matrix.hpp) → [matrix.cpp](../MLLibrary/Math/matrix.cpp) — shape, storage, matrix multiplication, activation and loss kernels. Compare allocation, reuse and transpose flags.

## Train a differentiable model

6. [Model/model.hpp](../MLLibrary/Model/model.hpp) → [model.cpp](../MLLibrary/Model/model.cpp) — computation-graph nodes, forward execution and reverse-mode derivatives; model functions call matrix kernels.
7. [Data/data.hpp](../MLLibrary/Data/data.hpp) → [data.cpp](../MLLibrary/Data/data.cpp) — data ingestion and conversions for training examples.
8. [Training/training.hpp](../MLLibrary/Training/training.hpp) → [training.cpp](../MLLibrary/Training/training.cpp) — minibatches, gradient calculation and parameter updates.
9. [Training/checkpoint.hpp](../MLLibrary/Training/checkpoint.hpp) → [checkpoint.cpp](../MLLibrary/Training/checkpoint.cpp) — model/training state persistence.
10. [main.cpp](../MLLibrary/main.cpp) — compose arena + PRNG + matrices + model + training into the MNIST demonstration.

**Trace one training step:** training driver → model graph forward pass → matrix kernels → loss → backward graph → gradient matrices → optimizer update. Ownership and lifetimes flow through the arena allocator.

## Data science and experimentation

11. [Data/dataset.hpp](../MLLibrary/Data/dataset.hpp) → [dataset.cpp](../MLLibrary/Data/dataset.cpp), then [Data/profiling.hpp](../MLLibrary/Data/profiling.hpp) → [profiling.cpp](../MLLibrary/Data/profiling.cpp).
12. [Preprocessing/pipeline.hpp](../MLLibrary/Preprocessing/pipeline.hpp) → [pipeline.cpp](../MLLibrary/Preprocessing/pipeline.cpp).
13. [Classical/classical.hpp](../MLLibrary/Classical/classical.hpp) → [classical.cpp](../MLLibrary/Classical/classical.cpp).
14. [Evaluation/metrics.hpp](../MLLibrary/Evaluation/metrics.hpp) → [metrics.cpp](../MLLibrary/Evaluation/metrics.cpp), then [cross_validation.hpp](../MLLibrary/Evaluation/cross_validation.hpp) → [cross_validation.cpp](../MLLibrary/Evaluation/cross_validation.cpp).
15. [Experiment/experiment.hpp](../MLLibrary/Experiment/experiment.hpp) → [experiment.cpp](../MLLibrary/Experiment/experiment.cpp).
16. [Search/search.hpp](../MLLibrary/Search/search.hpp) → [search.cpp](../MLLibrary/Search/search.cpp).
17. [AutoML/automl.hpp](../MLLibrary/AutoML/automl.hpp) → [automl.cpp](../MLLibrary/AutoML/automl.cpp).

## Public contracts, module interfaces, GPU, and proof

18. [Contracts/runtime_contracts.hpp](../MLLibrary/Contracts/runtime_contracts.hpp) → [runtime_contracts.cpp](../MLLibrary/Contracts/runtime_contracts.cpp) — agreements between a model and its consumers.
19. [ModuleInterfaces/MLLibrary.cppm](../ModuleInterfaces/MLLibrary.cppm) — intended module import surface; follow each export into its named module interface, and compare against classic header-based targets.
20. [TensorRuntime.cppm](../MLLibrary/TensorRuntime.cppm) and [TensorGPURuntime.cppm](../MLLibrary/TensorGPURuntime.cppm) — tensor execution and optional GPU-oriented concerns.
21. [Benchmark/benchmark.cpp](../MLLibrary/Benchmark/benchmark.cpp), [matrix_benchmark.cpp](../MLLibrary/Benchmark/matrix_benchmark.cpp), and [Tests](../Tests) — validate allocation, numerical behaviour, autograd, training and higher-level workflows. Check test targets in CMake before executing.

## Exercises

1. Trace a matrix's allocation from `mat_create` through `MemArena::push` and identify its lifetime.
2. Explain why a graph node stores gradients separately from the current value.
3. Follow one MNIST sample from ingestion to loss, backward propagation, and parameter update.
4. Find where PRNG seeding makes two experiments reproducible and where data order can still change the result.
5. Compare classic headers to module exports and list which targets **actually compile** each interface.
6. Review checkpoint serialization for compatibility and whether optimizer state is restored.

**Scope:** This guide links tracked files; generated binaries, datasets, debug symbols and Xcode user configuration are not source study steps. A file appearing in the repository does not imply it is active in every build configuration.
