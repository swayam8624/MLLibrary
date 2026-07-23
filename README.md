# MLLibrary

MLLibrary is a from-first-principles C++ machine-learning library. The current
repo is the seed implementation: a small neural-network runtime that trains a
fully connected MNIST classifier while the larger Kairo ML infrastructure is
being split into deliberate foundation packages.

The long-term goal is a production-grade C++ ML stack covering ML, deep
learning, and data workflows without depending on external ML/DL frameworks.

It currently includes:

- a custom arena allocator,
- a PCG-style pseudo-random number generator,
- row-major matrix operations,
- a tiny static autograd graph,
- ReLU, softmax, and cross-entropy,
- mini-batch SGD, momentum SGD, Nesterov, RMSProp, Adam, AdamW, and optional
  weight decay and global gradient clipping,
- dependency-free tabular `StandardScaler`, deterministic farthest-first
  `KMeans`, PCA, linear/logistic regression, KNN, and Gaussian naive Bayes,
- CSV training metrics for visual training dashboards.

The current Xcode target still builds through `.cpp` implementation files and
classic headers. C++20-style module interface files live in `ModuleInterfaces/`
as the intended public API shape for the next build-system step.

## Repository Layout

- `MLLibrary/Core/`: base types, memory arena, and PRNG.
- `MLLibrary/Math/`: matrix creation, elementwise ops, matrix multiply,
  activations, losses, and their gradient helpers.
- `MLLibrary/Model/`: graph nodes, graph construction, topological program
  generation, forward execution, and reverse-mode gradient propagation.
- `MLLibrary/Training/`: mini-batch training loop and evaluation.
- `MLLibrary/Data/`: raw matrix loading, MNIST terminal drawing, one-hot labels.
- `MLLibrary/Contracts/`: canonical Tensor, format, error, and model-capability
  contracts.
- `MLLibrary/Benchmark/`: reproducible machine-readable benchmark manifests
  and measurements.
- `ModuleInterfaces/`: `.cppm` module declarations mirroring the public API.

## Kairo Infrastructure Repos

These separate infrastructure projects now exist under the `Kairo` name and
have been pushed to GitHub:

- [KairoScheduler](https://github.com/swayam8624/KairoScheduler): CPU
  scheduling foundation with a thread pool and `ParallelFor` range splitter.
- [KairoSIMD](https://github.com/swayam8624/KairoSIMD): SIMD-kernel API with
  scalar fallback kernels for vector math, dot products, AXPY, and activation
  primitives.
- [KairoGPU](https://github.com/swayam8624/KairoGPU): GPU backend/device
  contract for future Metal, Vulkan, CUDA, or WebGPU implementations.
- [KairoONNX](https://github.com/swayam8624/KairoONNX): ONNX graph IR and
  bounded import surface for a deliberate operator-by-operator importer.
- [KairoTransformers](https://github.com/swayam8624/KairoTransformers):
  transformer configuration, token-batch, mask, and parameter-estimation
  foundation.
- [KairoAI](https://github.com/swayam8624/KairoAI): provider-neutral request,
  permission, and device-agent intent-routing boundary. It is where local
  multimodal observations become narrow, approval-gated application actions.
- [KairoMacPerception](https://github.com/swayam8624/KairoMacPerception):
  native macOS ScreenCaptureKit/Vision adapter with stable two-hand rectangle
  recognition and reversible, in-memory capture previews.

The packages build as independent C++23 module projects with CMake, Ninja, and
upstream LLVM Clang. The implemented vertical slices are deliberately bounded:
KairoScheduler has exception-safe task groups; KairoSIMD has Apple-Silicon NEON
paths; KairoONNX parses bounded graph metadata and Float32 initializers into
Kairo tensors; KairoTransformers executes a Tensor-backed decoder block; and
KairoGPU discovers Metal devices, manages shared buffers, and runs a validated
Float32 vector-add compute dispatch. General ONNX lowering, reusable GPU
pipelines, GPU matmul, and transformer training are still unfinished.

## Local Multimodal Device-Agent Direction

The library is also the learning/runtime foundation for a local-first device
agent: camera, microphone, and screen observations are interpreted by compact
specialized models, grounded against visible application state, and converted
only into registered, verified actions. The first implementation plan, exact
input/output contracts, reuse map, safety boundary, performance budget, and
phase gates are in [docs/LOCAL_MULTIMODAL_DEVICE_AGENT.md](docs/LOCAL_MULTIMODAL_DEVICE_AGENT.md).
The external code and platform API inventory, integration decisions, and
anti-duplication rules are in [docs/EXTERNAL_REUSE_AUDIT.md](docs/EXTERNAL_REUSE_AUDIT.md).

## Build

Use CMake for the normal non-module app build:

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

Generate a local visual report from the trainer's `training_metrics.csv`:

```sh
./build/MLLibraryTrainingReport training_metrics.csv training_report.html
```

The report is a self-contained HTML file with canvas charts for loss, accuracy,
and learning rate plus the final-run summary. It reads only the metrics CSV and
does not upload training data or require a web dependency.

Inspect a raw Float32 MNIST-compatible image/label pair:

```sh
./build/MLLibraryDatasetReport \
  MLLibrary/test_images.mat \
  MLLibrary/test_labels.mat \
  dataset_report.html \
  64
```

The dataset report validates the complete binary dataset, computes pixel
range/mean/standard deviation and class counts, and embeds a bounded,
deterministic sample grid. The generated HTML is self-contained and responsive;
its CTest fixture processes all 10,000 checked-in test samples.

Generate a reproducible scalar matrix benchmark:

```sh
./build/MLLibraryBenchmark \
  --output benchmark.json \
  --size 128 \
  --iterations 10 \
  --seed 5489
```

The versioned JSON records compiler/platform identity, arguments, latency
percentiles, throughput, peak resident memory, and maximum absolute error. The
canonical runtime, format, transactional-action, dependency, and supported
platform contracts are documented in
[docs/CONTRACTS.md](docs/CONTRACTS.md).

The CTest target exercises matrix multiplication, the softmax/cross-entropy
gradient path with a non-differentiable target tensor, and each supported
optimizer on a tiny trainable classifier. This guards the minimum invariant for
the current classifier: loss gradients must reach trainable logits even though
labels do not require gradients.

Use Xcode:

```sh
xcodebuild -project MLLibrary.xcodeproj -scheme MLLibrary -configuration Debug build
```

To also build and link the module-based KairoMath foundation, use upstream LLVM
Clang and enable `MLLIB_USE_KAIRO_MATH`:

```sh
cmake -S . -B build-kairo -G Ninja \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DMLLIB_USE_KAIRO_MATH=ON
cmake --build build-kairo
ctest --test-dir build-kairo --output-on-failure
```

That configuration includes a module-consuming integration test for
`Kairo.Foundation.Math.Tensor`, covering Tensor matrix multiplication and the
softmax/cross-entropy inference path. This confirms the C++ module graph is
used by MLLibrary's build rather than only linked transitively.

To include the Apple Metal backend and the GPU linear-forward equivalence test:

```sh
cmake -S . -B build-gpu -G Ninja \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DMLLIB_USE_KAIRO_MATH=ON \
  -DMLLIB_USE_KAIRO_GPU=ON
cmake --build build-gpu
ctest --test-dir build-gpu --output-on-failure
```

The target uses a file-system synchronized group. It currently compiles normal
C++ sources with `.cpp` files. The `.cppm` interfaces are outside the
synchronized target folder so they do not disturb the existing build.

In this environment, `xcrun clang++ --version` reports Apple clang 21.0.0. A
direct syntax check of `export module` syntax is not accepted by that command
line toolchain, and `-fmodules-ts` is unavailable. Treat `ModuleInterfaces/` as
the module migration contract until the Xcode target or compiler is configured
with working C++ module support.

## C++ Module Migration

The module interfaces are split by subsystem:

- `MLLibrary.Base`
- `MLLibrary.Arena`
- `MLLibrary.PRNG`
- `MLLibrary.Matrix`
- `MLLibrary.Model`
- `MLLibrary.Data`
- `MLLibrary.Training`
- `MLLibrary.Contracts`
- `MLLibrary` as the aggregate module

KairoMath now also includes `Kairo.Foundation.Math.Tensor`, a module-based
tensor foundation with row-major dynamic tensors, shape/stride views, elementwise
ops, reductions, rank-2 and batched matrix multiply, Float16/BFloat16/Int8/index
dtypes, normalization, activations, losses, one-hot encoding, gradient clipping,
and DynamicMatrix interop.
`Kairo.Foundation.Math.TensorAutograd` adds dynamic Float32 reverse-mode graphs
for dense operations, normalization, convolution, pooling, MSE, and softmax
cross-entropy. Its tests validate dense, convolution, LayerNorm, and RMSNorm
gradients through central finite differences and train both MLP and CNN models.
`TensorTraining` provides six stateful optimizers, schedules, clipping, dynamic
loss scaling, and atomic checkpoints containing parameters, optimizer moments,
step position, and RNG state. Exact interrupted/resumed AdamW training is
bit-for-bit equivalent to uninterrupted training. `TensorData` provides
deterministic split/shuffle/batching and bounded background prefetch.

When `MLLIB_USE_KAIRO_MATH=ON`, MLLibrary also builds the compiled
`MLLibrary.TensorRuntime` module. Its first production vertical slice is a
batch-linear classifier with Tensor-owned weights and bias, batched forward
logits, stable softmax/cross-entropy, analytic gradients, SGD updates,
prediction, and accuracy. The module test proves loss reduction on a batched
two-class dataset; it is the migration target for the legacy arena-backed graph.
With `MLLIB_USE_KAIRO_GPU=ON`, `MLLibrary.TensorGPURuntime` executes the dense
matrix product through Metal and applies classifier bias to the synchronized
host result. Its integration test compares every output against the scalar
Tensor path.

The safe migration sequence is:

1. Keep the existing `.cpp` files as implementation units.
2. Configure the build system to compile the files in `ModuleInterfaces/` as
   C++20 module interface units.
3. Change consumers from `#include "matrix.hpp"` style includes to
   `import MLLibrary;` or targeted imports such as `import MLLibrary.Matrix;`.
4. Once all consumers import modules, either remove the compatibility headers or
   turn them into thin wrappers for non-module builds.

## Development Phases

### Phase 1: CPU ML Foundation

Goal: make the library correct, testable, and useful on CPU before adding
backend complexity.

- Move training data, weights, activations, and optimizer state onto
  `Kairo.Foundation.Math.Tensor`.
- Add tensor-backed dense layers, activations, losses, and metrics.
- Add gradient checks and deterministic unit tests for matrix, tensor, graph,
  and optimizer code.
- Add batched training so kernels operate on real mini-batches instead of one
  sample at a time.
- Add optimizers: SGD, momentum SGD, Nesterov, RMSProp, Adam, AdamW, weight
  decay, learning-rate schedules, and gradient clipping.
- Add data tools: CSV/binary tensor loading, normalization, shuffling,
  train/validation/test split, batching, and metrics.
- Add model serialization: weights, optimizer state, and training checkpoints.

**Verified result:** the Phase 1 MLP/CNN, numerical-gradient, and exact-resume
exit gates pass in KairoMath. Float32 master training supports low-precision
storage/casting and dynamic loss scaling. Remaining breadth includes dropout,
embedding autograd, and additional dataset file formats; these no longer block
the declared Phase 1 exit gates.

### Phase 1.5: Performance Foundation

Goal: make CPU execution fast without changing public model code.

- Wire `KairoScheduler` into tensor kernels for parallel reductions, elementwise
  ops, matrix multiply, convolution, and dataloading.
- Wire `KairoSIMD` into tensor kernels through a scalar/SIMD dispatch layer.
- Add tiled matrix multiply, batched matrix multiply, cache-aware transpose
  buffers, and profiling hooks.
- Keep scalar fallback behavior available for correctness, debugging, and
  unsupported platforms.

**Implemented:** dependency-aware cancellable scheduling, deterministic
parallel reductions, runtime NEON detection, SIMD elementwise/reduction kernels,
packed matrix multiplication, batched matrix multiplication, and scheduled
NHWC convolution. The combined accelerated build runs eight scalar-parity and
training tests. AVX implementation, sanitizer matrices, and machine-enforced
performance regression baselines remain.

### Phase 1.75: ONNX And Interop

Goal: import useful inference models without pretending to support all ONNX.

- Build `KairoONNX` into a bounded importer.
- First operator set: `Gemm`, `MatMul`, `Add`, `Sub`, `Mul`, `Div`, `Relu`,
  `Softmax`, `Conv`, `MaxPool`, `Flatten`, `Reshape`, `Transpose`, and
  `LayerNormalization`.
- Convert ONNX weights into Kairo tensors.
- Lower imported graphs into the native runtime IR.
- Validate imported inference against known fixtures.

**Implemented:** bounded protobuf parsing for graph/opset/type/attribute
metadata, Float32 and Int64 initializers, topological validation, and native
execution for the declared MLP/CNN operator surface plus GELU, Sigmoid, Gather,
Slice, and Concat. MLP, CNN, pooling, and indexing fixtures pass. External
ONNX Runtime parity, constant folding, grouped convolution, and broader opset
semantics remain.

### Phase 1.9: Transformer Foundation

Goal: support transformer inference first, training second.

- Implement embeddings, positional encodings, causal masks, LayerNorm, GELU,
  residual streams, attention score scaling, and multi-head attention.
- Add transformer block and decoder-only model definitions.
- Add tokenizer-facing interfaces without coupling the core runtime to one
  tokenizer format.
- Add inference checkpoints before training checkpoints.
- Add transformer training only after tensor autodiff, scheduling, matmul,
  normalization, and checkpointing are stable.

**Implemented for inference:** token embeddings, RoPE, LayerNorm/RMSNorm,
multi-head causal attention, decoder blocks, a multi-layer decoder model,
per-layer KV caches, cached/full-logit equivalence, deterministic sampling,
INT8 dense weights, and byte-bounded layer streaming. Transformer autodiff
training, LoRA, grouped-query attention, INT4, and benchmark manifests remain.

### Phase 2: Visual Computing

Goal: make training and data visible, inspectable, and useful for computer
vision workflows.

- Build a visual training dashboard from CSV/live metrics: loss, accuracy,
  learning rate, gradients, activation histograms, and confusion matrix.
  The CSV report generator for loss, accuracy, and learning rate is implemented;
  live streaming, histogram, and confusion-matrix sources remain next.
- Add dataset viewers for images, labels, augmentations, and tensor batches.
- Add image data pipelines and computer-vision model layers.
- Add CNN visualizations: filters, feature maps, saliency, Grad-CAM-style
  explanations, and misclassification inspection.
- Add model graph visualization and kernel profiling views.

**Current phase result:** dependency-free HTML tools cover training curves and
dataset inspection. The training report plots loss, accuracy, and learning
rate. The dataset report validates raw Float32 MNIST data, summarizes class and
pixel statistics, and renders responsive image samples. Both are post-run
artifacts; live streaming, augmentation inspection, feature maps, saliency,
confusion matrices, and model/kernel graphs remain.

### Phase 3: GPU Backends

Goal: accelerate tensor kernels with explicit backend implementations.

- Implement `KairoGPU` backends deliberately, starting with one platform.
- Add GPU buffers, command lists, shader/kernel compilation, synchronization,
  readback, and profiling.
- Move tensor kernels behind backend dispatch: scalar, threaded CPU, SIMD CPU,
  and GPU.
- Keep CPU reference kernels as the correctness oracle.

**Current phase result:** `KairoGPU` has a real Apple Metal vertical slice:
device discovery, capability reporting, owned shared buffers, byte upload and
readback, cached compute pipelines/queues, Float32 vector addition, and a
16-by-16 threadgroup-tiled Float32 matmul. `KairoMath.TensorGPU` dispatches
contiguous Tensor add/matmul through that backend, and MLLibrary validates a GPU
linear-classifier forward pass against its CPU reference. Persistent
GPU-resident Tensor storage, generic resource binding, queued asynchronous
execution, reductions, convolution, and profiling remain completion work.

### Phase 4: Full ML Platform

Goal: turn the foundations into a complete ML, DL, and data platform.

- Classical ML: linear/logistic regression, k-means, PCA, decision trees,
  random forests, SVM-style learners where practical, naive Bayes, KNN, and
  evaluation metrics.
- Deep learning: MLPs, CNNs, RNNs, LSTMs/GRUs, autoencoders, transformers, and
  inference/training utilities.
- Data: tensor datasets, tabular preprocessing, feature normalization,
  categorical encoding, missing-value handling, pipelines, and experiment
  tracking.

The current standard C++ build includes a tested tabular baseline subset:
column standardization, deterministic KMeans, linear/logistic regression, KNN,
Gaussian naive Bayes, a deterministic CART-style classifier, and bootstrap
random forests with per-node feature subsampling. SVM-style learners and shared
evaluation suites remain.
- Deployment: model save/load, inference runtimes, quantization, benchmarking,
  and model import/export.

**Phase 4 entry criteria:** retain CPU reference tests while promoting
Tensor-backed dense, convolution, pooling, normalization, optimizer, dataset,
checkpoint, and evaluation paths to stable public contracts. Every added
algorithm needs deterministic fixtures, serialization expectations, and a
benchmarkable inference/training path before it is presented as complete.

## Final Capability Target

Without external ML/DL libraries, this repo can still grow into a useful
educational and engine-style ML foundation:

- Dense neural networks: MLPs, residual MLP blocks, classification, regression.
- Classic training: SGD, momentum, RMSProp, Adam, weight decay, learning-rate
  schedules, gradient clipping.
- CPU tensor basics: matrices, small tensors, broadcasting, reductions,
  normalization, im2col-style convolution.
- Common layers: linear, ReLU, sigmoid, tanh, GELU approximation, softmax,
  dropout, batch/layer normalization, embeddings.
- Small CNNs and RNNs: practical for MNIST/CIFAR-scale experiments on CPU, but
  not competitive with optimized BLAS/GPU frameworks.
- Inference runtimes: loading your own weights, running small models, quantized
  int8/f16-style kernels if implemented manually.
- Data utilities: CSV/raw binary readers, normalization, batching, shuffling,
  train/test splitting.

The practical limits are performance and ecosystem, not language capability.
Large transformers, production CNN training, GPU acceleration, automatic kernel
fusion, distributed training, mature serialization, and broad model import
support require either significant custom infrastructure or external libraries.

The final Kairo ML library should be able to:

- Train and run dense neural networks, CNNs, RNNs, and transformer-style models.
- Run classical ML algorithms for tabular and numerical data.
- Load, preprocess, batch, shuffle, split, normalize, and inspect datasets.
- Use a unified tensor runtime for model parameters, activations, gradients, and
  data batches.
- Train with SGD-family and Adam-family optimizers, schedules, regularization,
  clipping, checkpointing, and metrics.
- Import a bounded, tested subset of ONNX models for inference.
- Run on scalar CPU, threaded CPU, SIMD CPU, and eventually GPU backends.
- Visualize training, data, model graphs, kernels, gradients, activations, and
  computer-vision outputs.
- Save and load models, weights, optimizer state, and training runs.
- Serve as both a learning-quality implementation and a reusable engine-grade
  ML foundation.

## Current Constraints

- Matrix storage is row-major.
- The autograd graph is static after construction.
- Batch training currently loops one sample at a time and accumulates gradients.
- Only a small operation set exists: add, subtract, matrix multiply, ReLU,
  softmax, and cross-entropy.
- Dataset loading expects raw `float` matrix files beside the executable working
  directory.
- Training writes optional CSV metrics through `metrics_csv_path`; `main.cpp`
  currently emits `training_metrics.csv` for dashboard/plotting work.
