# MLLibrary — linked learning order

Begin at allocator/PRNG and tensor/matrix representation, then graph autograd, losses and optimizers, dataset parsing, training, evaluation, search/AutoML and benchmarks. The standalone library owns CPU-first models and tests; separate KairoGPU/ONNX/Transformers packages extend bounded capabilities without implying framework parity.

## Public APIs and interactions

Trace data tensor shape/ownership -> matrix ops -> forward tape -> reverse-mode gradients -> optimizer update -> metrics. Protect against data leakage by fitting preprocessing only on training folds; measure accuracy, numerical gradients and performance independently.

## Read every tracked source in this order (69 files)

1. [`MLLibrary/Core/arena.cpp`](MLLibrary/Core/arena.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
2. [`MLLibrary/Core/prng.cpp`](MLLibrary/Core/prng.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
3. [`MLLibrary/Math/matrix.cpp`](MLLibrary/Math/matrix.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
4. [`MLLibrary/Model/model.cpp`](MLLibrary/Model/model.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
5. [`MLLibrary/Training/training.cpp`](MLLibrary/Training/training.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
6. [`MLLibrary/Data/dataset.cpp`](MLLibrary/Data/dataset.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
7. [`MLLibrary/Evaluation/metrics.cpp`](MLLibrary/Evaluation/metrics.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
8. [`MLLibrary/Evaluation/cross_validation.cpp`](MLLibrary/Evaluation/cross_validation.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
9. [`MLLibrary/Preprocessing/pipeline.cpp`](MLLibrary/Preprocessing/pipeline.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
10. [`MLLibrary/Search/search.cpp`](MLLibrary/Search/search.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
11. [`MLLibrary/AutoML/automl.cpp`](MLLibrary/AutoML/automl.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
12. [`Tests/autograd_tests.cpp`](Tests/autograd_tests.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
13. [`Tests/training_tests.cpp`](Tests/training_tests.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
14. [`MLLibrary/AutoML/automl.hpp`](MLLibrary/AutoML/automl.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
15. [`MLLibrary/base.h`](MLLibrary/base.h) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
16. [`MLLibrary/Benchmark/benchmark.cpp`](MLLibrary/Benchmark/benchmark.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
17. [`MLLibrary/Benchmark/benchmark.hpp`](MLLibrary/Benchmark/benchmark.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
18. [`MLLibrary/Benchmark/matrix_benchmark.cpp`](MLLibrary/Benchmark/matrix_benchmark.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
19. [`MLLibrary/Classical/classical.cpp`](MLLibrary/Classical/classical.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
20. [`MLLibrary/Classical/classical.hpp`](MLLibrary/Classical/classical.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
21. [`MLLibrary/Contracts/runtime_contracts.cpp`](MLLibrary/Contracts/runtime_contracts.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
22. [`MLLibrary/Contracts/runtime_contracts.hpp`](MLLibrary/Contracts/runtime_contracts.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
23. [`MLLibrary/Core/arena.h`](MLLibrary/Core/arena.h) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
24. [`MLLibrary/Core/prng.h`](MLLibrary/Core/prng.h) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
25. [`MLLibrary/Data/data.cpp`](MLLibrary/Data/data.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
26. [`MLLibrary/Data/data.hpp`](MLLibrary/Data/data.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
27. [`MLLibrary/Data/dataset.hpp`](MLLibrary/Data/dataset.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
28. [`MLLibrary/Data/profiling.cpp`](MLLibrary/Data/profiling.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
29. [`MLLibrary/Data/profiling.hpp`](MLLibrary/Data/profiling.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
30. [`MLLibrary/Evaluation/cross_validation.hpp`](MLLibrary/Evaluation/cross_validation.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
31. [`MLLibrary/Evaluation/metrics.hpp`](MLLibrary/Evaluation/metrics.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
32. [`MLLibrary/Experiment/experiment.cpp`](MLLibrary/Experiment/experiment.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
33. [`MLLibrary/Experiment/experiment.hpp`](MLLibrary/Experiment/experiment.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
34. [`MLLibrary/main.cpp`](MLLibrary/main.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
35. [`MLLibrary/Math/matrix.hpp`](MLLibrary/Math/matrix.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
36. [`MLLibrary/mnist.py`](MLLibrary/mnist.py) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
37. [`MLLibrary/Model/model.hpp`](MLLibrary/Model/model.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
38. [`MLLibrary/Preprocessing/pipeline.hpp`](MLLibrary/Preprocessing/pipeline.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
39. [`MLLibrary/Search/search.hpp`](MLLibrary/Search/search.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
40. [`MLLibrary/TensorGPURuntime.cppm`](MLLibrary/TensorGPURuntime.cppm) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
41. [`MLLibrary/TensorRuntime.cppm`](MLLibrary/TensorRuntime.cppm) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
42. [`MLLibrary/Training/checkpoint.cpp`](MLLibrary/Training/checkpoint.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
43. [`MLLibrary/Training/checkpoint.hpp`](MLLibrary/Training/checkpoint.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
44. [`MLLibrary/Training/training.hpp`](MLLibrary/Training/training.hpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
45. [`MLLibrary/Visual/dataset_inspector.cpp`](MLLibrary/Visual/dataset_inspector.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
46. [`MLLibrary/Visual/dataset_report.cpp`](MLLibrary/Visual/dataset_report.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
47. [`MLLibrary/Visual/training_report.cpp`](MLLibrary/Visual/training_report.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
48. [`ModuleInterfaces/arena.cppm`](ModuleInterfaces/arena.cppm) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
49. [`ModuleInterfaces/base.cppm`](ModuleInterfaces/base.cppm) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
50. [`ModuleInterfaces/contracts_impl.cpp`](ModuleInterfaces/contracts_impl.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
51. [`ModuleInterfaces/contracts.cppm`](ModuleInterfaces/contracts.cppm) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
52. [`ModuleInterfaces/data.cppm`](ModuleInterfaces/data.cppm) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
53. [`ModuleInterfaces/matrix.cppm`](ModuleInterfaces/matrix.cppm) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
54. [`ModuleInterfaces/MLLibrary.cppm`](ModuleInterfaces/MLLibrary.cppm) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
55. [`ModuleInterfaces/model.cppm`](ModuleInterfaces/model.cppm) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
56. [`ModuleInterfaces/prng.cppm`](ModuleInterfaces/prng.cppm) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
57. [`ModuleInterfaces/training.cppm`](ModuleInterfaces/training.cppm) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
58. [`Tests/arena_tests.cpp`](Tests/arena_tests.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
59. [`Tests/automl_tests.cpp`](Tests/automl_tests.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
60. [`Tests/contracts_module.cpp`](Tests/contracts_module.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
61. [`Tests/core_tests.cpp`](Tests/core_tests.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
62. [`Tests/dataset_tests.cpp`](Tests/dataset_tests.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
63. [`Tests/evaluation_tests.cpp`](Tests/evaluation_tests.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
64. [`Tests/experiment_tests.cpp`](Tests/experiment_tests.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
65. [`Tests/pipeline_tests.cpp`](Tests/pipeline_tests.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
66. [`Tests/profiling_tests.cpp`](Tests/profiling_tests.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
67. [`Tests/search_tests.cpp`](Tests/search_tests.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
68. [`Tests/tensor_gpu_integration.cpp`](Tests/tensor_gpu_integration.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.
69. [`Tests/tensor_integration.cpp`](Tests/tensor_integration.cpp) — inspect the public API, state ownership, input validation, cross-file calls, and its matching tests.

For further work, implement small tests for normal, invalid and boundary conditions instead of relying only on documentation. Original code is available under the [MIT license](LICENSE) for study and extension. Third-party libraries, frameworks, datasets and fixtures retain their rights. Per-function source-level comments and CI/host verification are separate passes.
