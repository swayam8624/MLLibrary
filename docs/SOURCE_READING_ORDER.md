# Teaching source map — MLLibrary

> A guided source-reading itinerary. It is not a test result or claim that every file is documented yet.

## How to use this guide

Start from scalar tensor/matrix invariants, then autodiff and optimizers, then data pipelines and model-selection experiments. Separate the existing classic-header runtime from proposed C++ modules.

Read each phase in order. Every link goes to a source file or directory in this repository. For a directory, inspect its public interface and then its implementation files. After each phase, identify the caller, the state it owns, the inputs it validates, and its observable outputs.

## 1. Architecture and building

1. [`README.md`](../README.md)
2. [`CMakeLists.txt`](../CMakeLists.txt)
3. [`MLLibrary/Core`](../MLLibrary/Core)
4. [`MLLibrary/Contracts`](../MLLibrary/Contracts)
5. [`MLLibrary/Math`](../MLLibrary/Math)

## 2. Differentiation and training

1. [`MLLibrary/Model`](../MLLibrary/Model)
2. [`MLLibrary/Training`](../MLLibrary/Training)
3. [`MLLibrary/Data`](../MLLibrary/Data)
4. [`ModuleInterfaces`](../ModuleInterfaces)

## 3. Evaluation and reproducibility

1. [`MLLibrary/Benchmark`](../MLLibrary/Benchmark)
2. [`Tests/core_tests.cpp`](../Tests/core_tests.cpp)
3. [`Tests/autograd_tests.cpp`](../Tests/autograd_tests.cpp)
4. [`Tests/training_tests.cpp`](../Tests/training_tests.cpp)
5. [`Tests/dataset_tests.cpp`](../Tests/dataset_tests.cpp)
6. [`Tests/evaluation_tests.cpp`](../Tests/evaluation_tests.cpp)
7. [`Tests/automl_tests.cpp`](../Tests/automl_tests.cpp)

## 4. Kairo runtime neighbors

1. [`docs`](../docs)
2. [`Tests/tensor_integration.cpp`](../Tests/tensor_integration.cpp)
3. [`Tests/tensor_gpu_integration.cpp`](../Tests/tensor_gpu_integration.cpp)

## End-to-end interaction map

Data loaders produce matrices/tensors; math operations feed a static differentiation graph; reverse-mode gradients reach optimizers and training loops; evaluation and search code compare candidates without leaking validation rows into training. KairoScheduler/SIMD/GPU/ONNX are separate libraries with bounded available operators.

## APIs and code-reading questions

For each function, distinguish **contract** (preconditions, postconditions, failure modes), **implementation** (state/algorithms), **collaborators** (imports, calls and consumers), **lifetime/ownership**, and **verification** (test or evidence that checks it). For C++ modules, `export module` names the public compiled interface and `import` expresses module dependencies; an implementation unit's `module ...;` contributes definitions without exporting them.

## Boundaries and honesty

This repository is a seed implementation. Do not describe planned full ONNX support, arbitrary GPU training, or large-transformer serving as completed.

## Documentation rules

- Teaching comments describe *why*, invariants, edge cases, API ownership and cross-file effects; avoid narrating obvious assignments.
- Preserve behavior during documentation-only edits; keep correctness changes in separate reviewed commits.
- Link function names and tests where possible; mark claims as source-present, CI-observed, host-verified or research-evidenced.
- License grants apply only to material for which the repository owner holds rights; third-party source, models, media and datasets keep their own notices.


## Complete source-file inventory

After the curated teaching path, follow this deterministic inventory of first-party source, scripts, shaders, and tests. Grouping is a suggested learning order, not an indication that every function is already annotated. Vendored dependencies, ROMs, binaries, data, media, and generated evidence are excluded.

### Entry points and build (2)

1. [`CMakeLists.txt`](../CMakeLists.txt)
2. [`MLLibrary/main.cpp`](../MLLibrary/main.cpp)

### Interfaces and modules (31)

1. [`MLLibrary/AutoML/automl.hpp`](../MLLibrary/AutoML/automl.hpp)
2. [`MLLibrary/base.h`](../MLLibrary/base.h)
3. [`MLLibrary/Benchmark/benchmark.hpp`](../MLLibrary/Benchmark/benchmark.hpp)
4. [`MLLibrary/Classical/classical.hpp`](../MLLibrary/Classical/classical.hpp)
5. [`MLLibrary/Contracts/runtime_contracts.hpp`](../MLLibrary/Contracts/runtime_contracts.hpp)
6. [`MLLibrary/Core/arena.h`](../MLLibrary/Core/arena.h)
7. [`MLLibrary/Core/prng.h`](../MLLibrary/Core/prng.h)
8. [`MLLibrary/Data/data.hpp`](../MLLibrary/Data/data.hpp)
9. [`MLLibrary/Data/dataset.hpp`](../MLLibrary/Data/dataset.hpp)
10. [`MLLibrary/Data/profiling.hpp`](../MLLibrary/Data/profiling.hpp)
11. [`MLLibrary/Evaluation/cross_validation.hpp`](../MLLibrary/Evaluation/cross_validation.hpp)
12. [`MLLibrary/Evaluation/metrics.hpp`](../MLLibrary/Evaluation/metrics.hpp)
13. [`MLLibrary/Experiment/experiment.hpp`](../MLLibrary/Experiment/experiment.hpp)
14. [`MLLibrary/Math/matrix.hpp`](../MLLibrary/Math/matrix.hpp)
15. [`MLLibrary/Model/model.hpp`](../MLLibrary/Model/model.hpp)
16. [`MLLibrary/Preprocessing/pipeline.hpp`](../MLLibrary/Preprocessing/pipeline.hpp)
17. [`MLLibrary/Search/search.hpp`](../MLLibrary/Search/search.hpp)
18. [`MLLibrary/TensorGPURuntime.cppm`](../MLLibrary/TensorGPURuntime.cppm)
19. [`MLLibrary/TensorRuntime.cppm`](../MLLibrary/TensorRuntime.cppm)
20. [`MLLibrary/Training/checkpoint.hpp`](../MLLibrary/Training/checkpoint.hpp)
21. [`MLLibrary/Training/training.hpp`](../MLLibrary/Training/training.hpp)
22. [`ModuleInterfaces/arena.cppm`](../ModuleInterfaces/arena.cppm)
23. [`ModuleInterfaces/base.cppm`](../ModuleInterfaces/base.cppm)
24. [`ModuleInterfaces/contracts_impl.cpp`](../ModuleInterfaces/contracts_impl.cpp)
25. [`ModuleInterfaces/contracts.cppm`](../ModuleInterfaces/contracts.cppm)
26. [`ModuleInterfaces/data.cppm`](../ModuleInterfaces/data.cppm)
27. [`ModuleInterfaces/matrix.cppm`](../ModuleInterfaces/matrix.cppm)
28. [`ModuleInterfaces/MLLibrary.cppm`](../ModuleInterfaces/MLLibrary.cppm)
29. [`ModuleInterfaces/model.cppm`](../ModuleInterfaces/model.cppm)
30. [`ModuleInterfaces/prng.cppm`](../ModuleInterfaces/prng.cppm)
31. [`ModuleInterfaces/training.cppm`](../ModuleInterfaces/training.cppm)

### Implementation and other source (23)

1. [`MLLibrary/AutoML/automl.cpp`](../MLLibrary/AutoML/automl.cpp)
2. [`MLLibrary/Benchmark/benchmark.cpp`](../MLLibrary/Benchmark/benchmark.cpp)
3. [`MLLibrary/Benchmark/matrix_benchmark.cpp`](../MLLibrary/Benchmark/matrix_benchmark.cpp)
4. [`MLLibrary/Classical/classical.cpp`](../MLLibrary/Classical/classical.cpp)
5. [`MLLibrary/Contracts/runtime_contracts.cpp`](../MLLibrary/Contracts/runtime_contracts.cpp)
6. [`MLLibrary/Core/arena.cpp`](../MLLibrary/Core/arena.cpp)
7. [`MLLibrary/Core/prng.cpp`](../MLLibrary/Core/prng.cpp)
8. [`MLLibrary/Data/data.cpp`](../MLLibrary/Data/data.cpp)
9. [`MLLibrary/Data/dataset.cpp`](../MLLibrary/Data/dataset.cpp)
10. [`MLLibrary/Data/profiling.cpp`](../MLLibrary/Data/profiling.cpp)
11. [`MLLibrary/Evaluation/cross_validation.cpp`](../MLLibrary/Evaluation/cross_validation.cpp)
12. [`MLLibrary/Evaluation/metrics.cpp`](../MLLibrary/Evaluation/metrics.cpp)
13. [`MLLibrary/Experiment/experiment.cpp`](../MLLibrary/Experiment/experiment.cpp)
14. [`MLLibrary/Math/matrix.cpp`](../MLLibrary/Math/matrix.cpp)
15. [`MLLibrary/mnist.py`](../MLLibrary/mnist.py)
16. [`MLLibrary/Model/model.cpp`](../MLLibrary/Model/model.cpp)
17. [`MLLibrary/Preprocessing/pipeline.cpp`](../MLLibrary/Preprocessing/pipeline.cpp)
18. [`MLLibrary/Search/search.cpp`](../MLLibrary/Search/search.cpp)
19. [`MLLibrary/Training/checkpoint.cpp`](../MLLibrary/Training/checkpoint.cpp)
20. [`MLLibrary/Training/training.cpp`](../MLLibrary/Training/training.cpp)
21. [`MLLibrary/Visual/dataset_inspector.cpp`](../MLLibrary/Visual/dataset_inspector.cpp)
22. [`MLLibrary/Visual/dataset_report.cpp`](../MLLibrary/Visual/dataset_report.cpp)
23. [`MLLibrary/Visual/training_report.cpp`](../MLLibrary/Visual/training_report.cpp)

### Verification and tests (14)

1. [`Tests/arena_tests.cpp`](../Tests/arena_tests.cpp)
2. [`Tests/autograd_tests.cpp`](../Tests/autograd_tests.cpp)
3. [`Tests/automl_tests.cpp`](../Tests/automl_tests.cpp)
4. [`Tests/contracts_module.cpp`](../Tests/contracts_module.cpp)
5. [`Tests/core_tests.cpp`](../Tests/core_tests.cpp)
6. [`Tests/dataset_tests.cpp`](../Tests/dataset_tests.cpp)
7. [`Tests/evaluation_tests.cpp`](../Tests/evaluation_tests.cpp)
8. [`Tests/experiment_tests.cpp`](../Tests/experiment_tests.cpp)
9. [`Tests/pipeline_tests.cpp`](../Tests/pipeline_tests.cpp)
10. [`Tests/profiling_tests.cpp`](../Tests/profiling_tests.cpp)
11. [`Tests/search_tests.cpp`](../Tests/search_tests.cpp)
12. [`Tests/tensor_gpu_integration.cpp`](../Tests/tensor_gpu_integration.cpp)
13. [`Tests/tensor_integration.cpp`](../Tests/tensor_integration.cpp)
14. [`Tests/training_tests.cpp`](../Tests/training_tests.cpp)

Snapshot totals: 70 first-party source/build/test files, 93 tracked files overall. Re-audit when files are renamed or added.
