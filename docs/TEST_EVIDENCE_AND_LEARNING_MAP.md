# Learning from tests: evidence map for the MLLibrary C++23 implementation

This guide links [the source reading order](SOURCE_READING_ORDER.md), [the MNIST/autograd walkthrough](MNIST_TRAINING_FROM_FIRST_PRINCIPLES.md), [the CSV-to-AutoML laboratory](DATA_TO_AUTOML_LAB.md), and [the checkpoint audit](CHECKPOINT_AND_REPRODUCIBILITY_LAB.md) to **actual** first-party test files and the targets registered in [CMakeLists.txt](../CMakeLists.txt).

**Evidence warning:** these test targets exist in source, and teaching comments have been added to their entrypoints/assertion helpers. This pass did **not** compile or execute them. An existing test case does not prove correctness on every supported compiler, host, dataset or failure path.

## 1. Three different kinds of evidence

**Source review:** reading a function's input and ownership contract establishes what the implementation attempts, not whether it always behaves correctly.

**Deterministic test:** running a pinned binary and checking exact expected outcomes on a known fixture can establish a specific behavior under those conditions.

**Benchmark/system evidence:** repeated measurements on identified hardware, compared against a valid baseline, are needed to claim throughput, stability or memory efficiency. Correctness and performance should not be conflated.

A good report records: Git commit SHA; compiler version; platform/architecture; CMake flags; test name; exit status; tests actually executed; sanitizer settings; dataset/fixture provenance; and any failed assertions or captured logs.

## 2. The test-to-source map

| CTest target | Source test file | Production implementation / concept |
| --- | --- | --- |
| `MLLibraryArenaTests` | [`Tests/arena_tests.cpp`](../Tests/arena_tests.cpp) | [`Core/arena.cpp`](../MLLibrary/Core/arena.cpp): alignment, zeroed typed allocations, invalid alignment |
| `MLLibraryCoreTests` | [`Tests/core_tests.cpp`](../Tests/core_tests.cpp) | [`Math/matrix.cpp`](../MLLibrary/Math/matrix.cpp), [`Model/model.cpp`](../MLLibrary/Model/model.cpp), optimizer/weight checkpoint and runtime contracts |
| `MLLibraryAutogradTests` | [`Tests/autograd_tests.cpp`](../Tests/autograd_tests.cpp) | reverse-mode gradients against central finite differences, residual fanout, fused/unfused losses |
| `MLLibraryTrainingTests` | [`Tests/training_tests.cpp`](../Tests/training_tests.cpp) | mini-batch accumulation, deterministic seeded training, invalid descriptors, non-finite update rejection |
| `MLLibraryDatasetTests` | [`Tests/dataset_tests.cpp`](../Tests/dataset_tests.cpp) | typed CSV ingestion/fingerprint/splits and exact-size raw Float32 loader |
| `MLLibraryProfilingTests` | [`Tests/profiling_tests.cpp`](../Tests/profiling_tests.cpp) | sampled schema inference, profiles, report files, missing-target error |
| `MLLibraryPipelineTests` | [`Tests/pipeline_tests.cpp`](../Tests/pipeline_tests.cpp) | fitted imputation/scaling/one-hot vocabulary, unknown categories, serialization and schema mismatch |
| `MLLibraryExperimentTests` | [`Tests/experiment_tests.cpp`](../Tests/experiment_tests.cpp) | estimator creation and fitting, logistic/linear/tree examples, incompatible task and report artifacts |
| `MLLibraryEvaluationTests` | [`Tests/evaluation_tests.cpp`](../Tests/evaluation_tests.cpp) | confusion matrix/metrics, fold generators, stratified/group/time-series evaluation |
| `MLLibrarySearchTests` | [`Tests/search_tests.cpp`](../Tests/search_tests.cpp) | candidate enumeration, reproducible random search, failed trials, successive halving and metric direction |
| `MLLibraryAutoMLTests` | [`Tests/automl_tests.cpp`](../Tests/automl_tests.cpp) | binary/multiclass/regression planner limits, numeric label inference, runner artifacts |

These test executables are built under CMake's test-enabled configuration. The optional KairoMath and GPU module integration tests have separate feature flags, compilers, native dependencies and host requirements; a default scalar test pass does **not** execute them automatically.

## 3. Run and inspect the existing tests

On a system with a suitable C++23 compiler, CMake and Ninja:

```bash
# Configure the default core/test project. Explicitly select another compiler
# here if the system default lacks the required C++23 toolchain support.
cmake -S . -B build -G Ninja -DBUILD_TESTING=ON

# Verify which targets CMake actually registered on THIS machine.
ctest --test-dir build -N

# Compile using the machine's supported build parallelism.
cmake --build build --parallel 4

# Run tests with failure diagnostics.
ctest --test-dir build --output-on-failure

# To concentrate on the current data/preprocessing/evaluation teaching pass:
ctest --test-dir build --output-on-failure \
  -R 'MLLibrary(Dataset|Profiling|Pipeline|Experiment|Evaluation|Search|AutoML)Tests'
```

The commands are suggestions for the user or a connected CI host; **they were not executed in this documentation pass**. If CMake configuration fails because of compiler/modules or platform settings, capture that failure before modifying the build or claiming test success.

## 4. Why the autograd tests matter

The original MNIST model stores an operation graph. Every differentiable node has a forward `val` and gradient `grad` matrix. In a reverse-mode pass, intermediate gradients are zeroed, the loss derivative is seeded, and contributions are propagated backward; parameter gradients may accumulate from repeated downstream use.

A finite-difference check approximates

```text
dL/dw ≈ [L(w + h) - L(w - h)] / (2h)
```

for one weight at a time. The [autograd tests](../Tests/autograd_tests.cpp) compare this numerical reference to the code-generated reverse derivative, including a *residual fanout*. If the implementation overwrites the parameter gradient from one branch instead of adding it, a two-path network can produce a plausible forward pass and a wrong training update.

These tests use a finite perturbation size and numerical tolerance. That is not a proof of differentiability at ReLU's zero kink or of bitwise identical gradients on all FP hardware.

## 5. Testing the train/validation boundary

The most important tabular regression test is not whether a model obtains a high score on easy synthetic data. It is whether **no information from a held-out row is used to construct the training representation**.

The [pipeline tests](../Tests/pipeline_tests.cpp) check learned feature order, imputation/scaling, categorical missing levels and error on unknown categories. [Evaluation tests](../Tests/evaluation_tests.cpp) check fold construction and group/time-series principles.

For stronger coverage, add a controlled validation-only outlier and check that the training-fitted mean and category vocabulary are unchanged. Build a group split whose group IDs repeat across rows and assert no group appears on both sides of any fold. For time series, test sorted and tied timestamps explicitly and make sure leakage-prone time features have a deliberate policy.

Test **what the actual implementation promises**: `Dataset::split` is a seeded ordinary split, not a substitute for a group- or time-aware fold generator.

## 6. Metric tests and class imbalance

The [evaluation test](../Tests/evaluation_tests.cpp) includes a small confusion matrix with known correct/incorrect labels and checks macro precision/recall/F1 and ROC-AUC. The implementation uses rows for *actual* classes and columns for *predicted* classes. That convention must be preserved when interpreting a heatmap or report.

Further tests worth adding include:
- Precision with no predicted positives, recall with no actual positives and extreme class imbalance.
- Binary ROC-AUC ties and a fold with only one class (AUC undefined, not zero).
- Non-finite probability scores and scores outside `[0,1]`.
- Constant regression targets, near-zero denominators for MAPE and negative R² cases.

A high accuracy value on a class-imbalanced dataset does not imply a useful classifier. Select the primary metric to match the task and study its failure cases.

## 7. Hyperparameter search and AutoML tests are selection tests

The [search tests](../Tests/search_tests.cpp) exercise grid enumeration, random-search reproducibility, graceful handling of failed candidates, promotion in successive halving and metric direction. The [AutoML tests](../Tests/automl_tests.cpp) check candidate planning constraints and end-to-end artifact creation.

None of those tests alone establishes that the **selected** CV winner will generalize to a new dataset. Repeatedly comparing many candidates on the same folds can bias the best observed score upward. Real-world performance claims need a locked untouched test set or an outer evaluation loop.

The planner currently makes bounded choices and discloses unsupported model families. The tests that check binary logistic skip, linear-only regression and candidate selection are therefore important *honesty contracts*, not signs of missing test quality.

## 8. Required next verification work

1. **Compile all eleven default test targets** and retain CTest logs with commit/host/toolchain.
2. Add hostile CSV, partial-overlap and malformed-checkpoint cases beyond the currently checked happy paths.
3. Add an explicit freeze-the-preprocessor data-leakage regression and nested/outer model-selection evaluation.
4. Add fault-injection for checkpoint write failure and verify file-handle closure, cleanup and no live-model mutation.
5. Run address/undefined-behavior sanitizers where supported, especially for arena lifetime and raw binary parsing.
6. Separately validate optional C++23 module integrations and Metal device execution on qualified hosts.

**This is a test-reading and evidence-design deliverable.** It does not claim that unexecuted tests passed, that a fixture score proves model quality, or that machine-specific performance was measured.
