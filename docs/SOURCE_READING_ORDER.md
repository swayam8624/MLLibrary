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
