# Teaching coverage ledger — MLLibrary

This ledger is deliberately conservative: **a linked file is not automatically a fully explained file**. It separates completed targeted source annotations from files and tests that remain to be audited. Update this ledger when implementation changes; do not equate a comment or README statement with a successful test run.

## Guided route

The [source reading order](SOURCE_READING_ORDER.md) is the navigation entrypoint. The [MNIST training laboratory](MNIST_TRAINING_FROM_FIRST_PRINCIPLES.md) follows memory -> matrix kernels -> graph -> gradients -> optimizer -> evaluation. The [checkpoint and reproducibility lab](CHECKPOINT_AND_REPRODUCIBILITY_LAB.md) explains why weight-only snapshots are not exact training resumption.

## Annotated code in the latest teaching wave

| File | What the inline explanations cover | Coverage scope |
| --- | --- | --- |
| [`MLLibrary/main.cpp`](../MLLibrary/main.cpp) | program entrypoint, model architecture, source calls, arena ownership, training result | entrypoint overview |
| [`MLLibrary/Model/model.cpp`](../MLLibrary/Model/model.cpp) | graph allocation, node builders, shape guards, forward interpretation, dependency order, gradient flow | 16 functions annotated |
| [`MLLibrary/Training/training.cpp`](../MLLibrary/Training/training.cpp) | validation, local shuffle, batch gradient accumulation, global norm clipping, candidate update and commit, held-out metrics | 14 function/stage annotations |
| [`MLLibrary/Training/checkpoint.cpp`](../MLLibrary/Training/checkpoint.cpp) | KMLCP01 weight-only format, temporary file, buffered validation, load boundary | format and entrypoints |
| [`MLLibrary/Math/matrix.cpp`](../MLLibrary/Math/matrix.cpp) | arena-backed Float32 matrices, row-major contracts, math kernels, fused/unfused losses and reverse helpers | 20 function/stage annotations |
| [`MLLibrary/Core/arena.h`](../MLLibrary/Core/arena.h) | public arena, Temp RAII rewind and typed allocation helpers | 4 API sections |
| [`MLLibrary/Core/arena.cpp`](../MLLibrary/Core/arena.cpp) | reserve/commit, alignment, bump/rewind and conflict-aware thread-local scratch | 10 function/stage annotations |
| [`MLLibrary/Core/prng.cpp`](../MLLibrary/Core/prng.cpp) | seed/state, PCG-style integer/Float32 outputs, process-global wrappers | 7 functions/stages |

## New coverage: tabular data, evaluation and model selection

**Worked guide:** [From CSV rows through fold-safe preprocessing to AutoML](DATA_TO_AUTOML_LAB.md). It maps every layer to the actual C++23 source and includes a documented client example, leakage exercises and supported-model boundaries.

| File | Added function or stage comments | Teaching emphasis |
| --- | ---: | --- |
| [`Data/data.cpp`](../MLLibrary/Data/data.cpp) | 4 | raw Float32 loader, terminal sample rendering, one-hot labels and silent invalid-ID handling |
| [`Data/dataset.cpp`](../MLLibrary/Data/dataset.cpp) | 11 | CSV state machine, typed schema, errors, fingerprint and split ownership |
| [`Data/profiling.cpp`](../MLLibrary/Data/profiling.cpp) | 9 | schema heuristics, missingness, targets and reporting |
| [`Preprocessing/pipeline.cpp`](../MLLibrary/Preprocessing/pipeline.cpp) | 10 | train-only fitting, frozen transforms, categorical unknowns and saved state |
| [`Classical/classical.cpp`](../MLLibrary/Classical/classical.cpp) | 19 | scaling, PCA, KMeans, regression, KNN, Naive Bayes, trees and forests |
| [`Evaluation/metrics.cpp`](../MLLibrary/Evaluation/metrics.cpp) | 7 | classification/regression metrics, ROC-AUC, undefined cases |
| [`Evaluation/cross_validation.cpp`](../MLLibrary/Evaluation/cross_validation.cpp) | 12 | KFold, stratification, group/time splits, per-fold leakage barriers |
| [`Experiment/experiment.cpp`](../MLLibrary/Experiment/experiment.cpp) | 12 | estimator factory, feature conversion, fitting and prediction evidence |
| [`Search/search.cpp`](../MLLibrary/Search/search.cpp) | 14 | discrete candidates, trial budget, successive halving and ranking |
| [`AutoML/automl.cpp`](../MLLibrary/AutoML/automl.cpp) | 14 | task inference, family constraints, bounded planning and orchestration |

These are **112 targeted teaching comments/sections** across ten first-party implementation files. They explain the dominant contracts but do not certify every branch of those files. The code edits are comments only; build, CTest and sample execution have not been performed in this wave.

These comments are grounded in the file implementation and cross-file calls. They do **not** certify that every branch, error path, platform function, or test within these files is complete.

## Public-header and test-reading continuation

**Ten related public API headers annotated:** [`Data/data.hpp`](../MLLibrary/Data/data.hpp), [`Data/dataset.hpp`](../MLLibrary/Data/dataset.hpp), [`Data/profiling.hpp`](../MLLibrary/Data/profiling.hpp), [`Preprocessing/pipeline.hpp`](../MLLibrary/Preprocessing/pipeline.hpp), [`Classical/classical.hpp`](../MLLibrary/Classical/classical.hpp), [`Evaluation/metrics.hpp`](../MLLibrary/Evaluation/metrics.hpp), [`Evaluation/cross_validation.hpp`](../MLLibrary/Evaluation/cross_validation.hpp), [`Experiment/experiment.hpp`](../MLLibrary/Experiment/experiment.hpp), [`Search/search.hpp`](../MLLibrary/Search/search.hpp), [`AutoML/automl.hpp`](../MLLibrary/AutoML/automl.hpp). Coverage includes 50 new explanatory declarations/sections, including ownership, train-only fit and safe evaluation rules.

**Eleven first-party default test source files annotated:** `core_tests.cpp`, `arena_tests.cpp`, `training_tests.cpp`, `autograd_tests.cpp`, `dataset_tests.cpp`, `profiling_tests.cpp`, `pipeline_tests.cpp`, `experiment_tests.cpp`, `evaluation_tests.cpp`, `search_tests.cpp` and `automl_tests.cpp` under [`Tests/`](../Tests). The additions identify the intentions of **54 test/helper functions**, plus file-level evidence boundaries. See [the named CTest-target learning and evidence map](TEST_EVIDENCE_AND_LEARNING_MAP.md).

**Note:** comments and test names are not test-run evidence. Test targets were inspected in [`CMakeLists.txt`](../CMakeLists.txt) but this pass did not compile or execute them.

## Remaining targeted code study

**Source and headers:** [`Core/prng.h`](../MLLibrary/Core/prng.h), [`Math/matrix.hpp`](../MLLibrary/Math/matrix.hpp), [`Model/model.hpp`](../MLLibrary/Model/model.hpp), [`Training/training.hpp`](../MLLibrary/Training/training.hpp) and [`Training/checkpoint.hpp`](../MLLibrary/Training/checkpoint.hpp) need API-by-API contract review to align all declarations with their implementations. Platform-specific allocation adapters need explicit native error and decommit checks.

**Data and preprocessing:** targeted source-level annotations and the worked data-to-AutoML lab are now present. Still required: a declaration/implementation contract audit of their headers, expanded parser and held-out-vocabulary tests, and real CI/compiler verification.

**Classical ML and experiment design:** the primary implementation methods have new comments; remaining work is to connect their public headers and tests to exact error/metric contracts and to validate split leakage and optimizer/model-selection evidence on a real host.

**Visualization/benchmark:** [`MLLibrary/Visual/`](../MLLibrary/Visual), [`MLLibrary/Benchmark/`](../MLLibrary/Benchmark). Teach metrics lineage, numerical error, benchmark validity and report generation.

**Optional module integrations:** [`MLLibrary/TensorRuntime.cppm`](../MLLibrary/TensorRuntime.cppm), [`MLLibrary/TensorGPURuntime.cppm`](../MLLibrary/TensorGPURuntime.cppm), [`ModuleInterfaces/`](../ModuleInterfaces). Compare actual enabled CMake module targets with merely declared interfaces before claiming a working module path.

**Tests:** the eleven default scalar/ML test files now have targeted teaching comments and a [CTest evidence map](TEST_EVIDENCE_AND_LEARNING_MAP.md), but execution, sanitizer checks and coverage measurement remain to be performed on a known host and commit. Optional module/GPU tests still need a separate host-qualified review.

## Source-level issues to distinguish from fixes

- The current checkpoint format saves **weights only**, not optimizer state or the shuffled PRNG position. See the [checkpoint lab](CHECKPOINT_AND_REPRODUCIBILITY_LAB.md).
- `main.cpp` does not inspect `model_train`'s result before printing post-training predictions.
- An unsuccessful checkpoint write can skip `fclose` due to short-circuit boolean evaluation; requires a separate code fix and fault-injection test.
- A graph forward interpreter returning `void` does not surface underlying matrix-kernel boolean failures as structured exceptions. Validate shapes before execution.
- The classic MNIST training binary is not automatically KairoScheduler/KairoSIMD/Metal enabled.
- `Data/data.cpp::one_hot_encode` can silently skip unsupported class IDs; labels should be validated before conversion.
- `AutoMLRunner` selects models using cross-validation rather than a locked independent final test; source guide [DATA_TO_AUTOML_LAB.md](DATA_TO_AUTOML_LAB.md) documents that distinction.
- Schema inference and AutoML planning may inspect full provided-dataset distributional statistics; hold the final test partition outside the planning process.

## How to extend this ledger

For each upcoming source file: read its declarations and full body; identify owners and borrowed pointers; trace caller and callee files; document numerical and platform API assumptions; add function-level comments at nontrivial boundaries; design an observable test; verify that no behavior statements were changed; only then mark the new scope here. Never claim repository-wide completion from partial coverage.

**Evidence state:** comments and manuals committed; no compiler, CTest, sanitizer, GPU profiler, or benchmark execution has been performed as part of these documentation-only commits.
