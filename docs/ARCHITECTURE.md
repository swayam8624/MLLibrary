# MLLibrary Architecture

## Current Design

The codebase is organized as a compact C++ ML stack:

- `Core` owns low-level primitives: fixed-width aliases, arena allocation, and
  random numbers.
- `Math` owns the `Matrix` type and numeric kernels.
- `Model` owns a tiny computation graph and reverse-mode gradient execution.
- `Training` owns the mini-batch SGD loop and evaluation.
- `Data` owns raw matrix loading and MNIST-specific helpers.

The graph is built manually by composing `ModelVar` nodes. `model_compile`
walks backward from the output and cost nodes to produce topologically sorted
programs. Forward execution runs the program in order. Backward execution seeds
the cost gradient with `1.0f`, then walks the program in reverse and accumulates
input gradients.

## Optimization Notes

The current bottleneck is matrix multiply. It is a plain row-major triple loop
with optional transposed operands. That is appropriate for clarity, but the next
performance tier should focus on:

1. Batching training examples so matrix multiply works on multiple samples per
   call instead of one sample at a time.
2. Adding tiled matrix multiply for cache locality.
3. Adding contiguous transpose/cache buffers for repeated weight-gradient paths.
4. Separating inference-only graph execution from training graph execution.
5. Adding lightweight profiling around `mat_mul`, data copy, and evaluation.

Avoid adding many optimizers or activation functions before batching and matrix
multiply are improved; those will not move overall throughput much.

The current trainer supports plain SGD and momentum SGD. It can also emit a
CSV metrics stream with epoch, accuracy, cost, and learning rate. That is the
first visual-training bridge; a richer dashboard should consume this data before
the training runtime grows a live UI protocol.

## Module Strategy

The `.cppm` files in `ModuleInterfaces/` mirror the public headers. They are not
inside the Xcode synchronized source folder yet because the active target still
passes normal C++ compilation flags to `.cpp` files. Keeping module interfaces
separate lets the project evolve without breaking the current build.

The local Apple clang 21.0.0 command-line toolchain does not currently accept
the checked-in `export module` syntax with default flags, and it does not
provide `-fmodules-ts`. The files are therefore a migration contract rather than
part of the verified build until the compiler/build setting supports C++ module
interface units.

When the build target is ready for C++20 modules, use this dependency order:

```text
MLLibrary.Base
MLLibrary.Arena
MLLibrary.PRNG
MLLibrary.Matrix
MLLibrary.Model
MLLibrary.Data
MLLibrary.Training
MLLibrary
```

KairoMath's module target has also been extended with:

```text
Kairo.Foundation.Math.Tensor
```

This tensor module is the preferred long-term numeric substrate for MLLibrary.
The current app still uses its legacy `Matrix` and graph code, but future layers
should move data batches, weights, activations, and optimizer state onto
KairoMath tensors.

Implementation `.cpp` files can initially keep including headers. After module
interface compilation is stable, they can move to:

```cpp
module;

#include <cstring>

module MLLibrary.Matrix;
```

or remain normal implementation units that consume imported declarations,
depending on what Xcode's C++ module support expects.

## Pure C++ ML Roadmap

Good next features without external ML/DL dependencies:

1. Add tests for matrix ops and graph gradients.
2. Add batched inputs and batched losses.
3. Add Adam and momentum SGD.
4. Add model save/load for raw weights.
5. Add a tensor view type before adding convolution.
6. Add simple CNN layers once matrix/tensor layout is stable.
7. Add quantized inference only after float inference is clean and tested.

Defer these until the foundation is stronger:

- GPU acceleration.
- Transformer training.
- ONNX import.
- Multi-threaded scheduling.
- SIMD-heavy kernels.

Those are achievable in C++, but they become separate infrastructure projects
without external libraries.
