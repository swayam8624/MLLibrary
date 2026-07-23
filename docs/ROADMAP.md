# MLLibrary Roadmap

Goal: build a production-grade C++ ML library covering ML, DL, and data, with
visual computing as the second major phase.

## Phase 1: CPU Foundation

- CMake build for the current app.
- KairoMath tensor module as the shared numeric substrate.
- Compiled `MLLibrary.TensorRuntime` batch-linear classifier plus KairoMath
  Tensor autodiff for dense and convolutional training.
- Matrix/tensor tests and gradient checks.
- Regression coverage for the softmax/cross-entropy path where target labels do
  not require gradients; this is now implemented in `MLLibraryCoreTests`.
- Batched training instead of per-sample gradient accumulation.
- Optimizers: Tensor-native SGD, momentum, Nesterov, RMSProp, Adam, AdamW,
  schedules, clipping, mixed-precision loss scaling, and full-state checkpoint.
- Layers: linear, activations, dropout, normalization, convolution, pooling,
  embeddings.
- Data: CSV, binary tensors, normalization, split/shuffle/batching, metrics.
- Serialization: atomic model, optimizer, schedule-step, and RNG checkpoints
  with bit-exact interrupted/resumed training coverage.

## Phase 1.5: Performance Foundation

- Thread-pool scheduler and parallel tensor kernels.
- SIMD backend layer with scalar fallback.
- Cache-tiled matrix multiply and batched matrix multiply.
- Profiling hooks around kernels, dataloading, and graph execution.
- Runtime backend dispatch: scalar, threaded, SIMD, and eventually GPU.

Current: dependency DAG scheduling, cancellation, NEON kernels, deterministic
parallel reductions, packed/batched matmul, and convolution are integrated and
scalar-parity tested. AVX and sanitizer/performance CI gates remain.

## Phase 1.75: Interop

- ONNX import for a bounded operator set first: Gemm, MatMul, Add, Relu,
  Softmax, Conv, MaxPool, Flatten, Reshape, Transpose, LayerNormalization.
- Model graph IR that can represent imported graphs and native training graphs.
- Weight conversion into KairoMath tensors.

Current: protobuf graph/opset/type/attribute parsing and native MLP/CNN/indexing
execution are implemented. External-runtime parity and graph optimization
passes remain.

## Phase 1.9: Transformers

- Token/embedding pipeline.
- LayerNorm, GELU, attention masks, scaled dot-product attention.
- Multi-head attention, MLP blocks, residual streams.
- Inference first, training second.
- Mixed precision and checkpointing only after the core kernels are correct.

Current inference: embeddings, RoPE, decoder model, KV cache, deterministic
sampling, INT8 weights, and bounded layer streaming. Training/LoRA remain.

## Phase 2: Visual Computing

- Dataset and tensor visualizer.
- Training dashboard with loss, accuracy, gradients, activation histograms.
  A C++ CSV-to-HTML dashboard is implemented for loss, accuracy, learning rate,
  and final run summary; add live metrics and richer tensor probes next.
- Model graph viewer.
- Image/video data pipelines.
- CNN visual explanations: feature maps, filters, saliency, confusion matrix.
- GPU compute visualization and profiling.

## Phase 4: Classical Baselines

- Implemented: `StandardScaler`, deterministic farthest-first `KMeans`, PCA
  through covariance power iteration/deflation, full-batch linear/logistic
  regression, KNN, Gaussian naive Bayes, deterministic CART classification
  trees, and seeded bootstrap random forests with nonlinear/reproducibility
  tests.
- Next: SVM-style learners and shared evaluation metrics.

## Do Not Fake

These are separate infrastructure projects and should be built deliberately:

- GPU acceleration.
- ONNX import.
- Transformer training.
- Multi-threaded scheduling.
- SIMD-heavy kernels.

The first pass should create clean extension points and CPU-correct behavior;
performance backends can then replace kernels without changing public model code.
