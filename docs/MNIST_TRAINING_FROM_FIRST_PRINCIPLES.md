# From first principles: trace an MNIST training step through MLLibrary

This is an implementation-guided reading companion to [SOURCE_READING_ORDER.md](SOURCE_READING_ORDER.md). It follows the **normal, classic C++** training route, not the optional KairoMath module integration. Inspect [main.cpp](../MLLibrary/main.cpp), [arena.cpp](../MLLibrary/Core/arena.cpp), [prng.cpp](../MLLibrary/Core/prng.cpp), [matrix.cpp](../MLLibrary/Math/matrix.cpp), [model.cpp](../MLLibrary/Model/model.cpp), and [training.cpp](../MLLibrary/Training/training.cpp) while reading.

## 1. Own memory before owning a model

`main` calls `MemArena::create(GiB(1), MiB(1))`, requesting a large virtual-address reserve with a smaller initial commit. The [arena implementation](../MLLibrary/Core/arena.cpp) uses platform reserve/commit/release functions and rounds sizes to operating-system page boundaries. Allocation via `push_struct` and `push_array` advances an arena position rather than creating independently freed heap objects. The key invariant is that all arena-allocated `Matrix`, `ModelVar`, and storage pointers must not be used after arena reset or `MemArena::destroy`.

`mat_create(arena, rows, cols)` allocates the matrix descriptor and the Float32 element buffer from the same arena. Its input dimensions are logically row/column counts; the `matrix.cpp` operations access `data` through the row-major contract. That descriptor is *not* an independently owning RAII vector. If an intermediate operation fails after allocating part of an object, investigate arena rollback and failure propagation rather than assuming ordinary destructor-based cleanup.

**Exercise:** diagram the lifetime of the image matrix, labels, parameters, gradient buffers and graph nodes. Mark where the owner is the arena, where a function only borrows a pointer, and where freed/reused arena storage would invalidate references.

## 2. Deterministic random number generation

[`prng.cpp`](../MLLibrary/Core/prng.cpp) implements PCG-style state transitions: `seed` initializes its state/odd increment and advances twice; `rand` transforms the old state into a rotated 32-bit integer; `randf` takes the high 24 bits and scales by `2^-24` into `[0,1)`. The file also exposes a single global PRNG used by calls such as `mat_fill_rand`.

Training uses an independently seeded `PRNG` through its training descriptor and has a bounded unbiased index-sampling helper using rejection of low values. Two notions of reproducibility therefore matter: **parameter initialization** can depend on global PRNG state, while **batch order** can depend on training-local seed and state. Preserving only the mini-batch seed does not reproduce a fresh run if the initial weights differ.

## 3. Construct the entire computation graph

[`main.cpp`](../MLLibrary/main.cpp) constructs the following fixed shapes (column vectors):

```text
input x              [784,1]
W0                   [16,784]
z0 = W0*x + b0       [16,1]
a0 = ReLU(z0)        [16,1]

W1                   [16,16]
z1 = W1*a0 + b1      [16,1]
a1 = a0 + ReLU(z1)   [16,1]   residual skip

W2                   [10,16]
z2 = W2*a1 + b2      [10,1]
output = softmax(z2) [10,1]
target y             [10,1]
loss = cross_entropy(y, output)
```

The `a0 + ReLU(z1)` term is a **residual connection**: it offers a direct gradient path from the second hidden representation to the first. The actual implementation is a small feedforward classifier, not a transformer or generic arbitrary-depth network.

`mv_create` in [model.cpp](../MLLibrary/Model/model.cpp) registers node indices, creates each value matrix and, only when `MV_FLAG_REQUIRES_GRAD` is set, creates gradient storage. `MV_FLAG_PARAMETER` identifies trainable values; `MV_FLAG_INPUT`, `MV_FLAG_OUTPUT`, `MV_FLAG_DESIRED_OUTPUT`, and `MV_FLAG_COST` bind distinguished nodes into the `ModelContext`.

The unary/binary graph helpers propagate the requires-gradient flag from differentiable inputs. A target/label node does not need to be a learnable parameter for the loss to deliver gradients to logits and weight matrices. The [model header](../MLLibrary/Model/model.hpp) defines up to two parent inputs per node and an operation enumeration; `model_compile` builds programs for execution over the graph.

**Checkpoint:** derive the shapes of all matrix products on paper, and show why `W1` can multiply `a0` but a `[10,16]` weight matrix cannot multiply the original `[784,1]` input.

## 4. Forward computation versus reverse-mode differentiation

The graph stores both `ModelVar::val` (the forward value) and, for differentiable nodes, `ModelVar::grad` (the incoming derivative accumulator). The forward program computes nodes in dependency order. Reverse-mode autodiff visits the graph in the reverse dependency direction, applying an operation-specific vector-Jacobian product.

For a scalar loss `L` and affine operation `z = Wx + b`, the chain rule gives `dL/dW = (dL/dz) x^T`, `dL/db = dL/dz`, and `dL/dx = W^T (dL/dz)`. With a residual addition, the gradient arriving at `a1` fans out into **both** its identity skip and its nonlinear subgraph, then recombines at `a0`. Any implementation that overwrites rather than accumulates contributions from both parents loses a valid gradient path.

The source offers both softmax-plus-elementwise-cross-entropy graph composition and a separate fused stable scalar softmax-cross-entropy operator; the shown `main.cpp` uses the former. Do not assume it uses the fused logit-level loss or that a generic `mv_softmax` automatically computes one distribution for every row; the header explicitly states the compact scalar runtime treats *all matrix elements* as one distribution.

**Exercise:** perform a two-node finite-difference gradient check on one weight: perturb a single parameter by `+h` and `-h`, recompute loss, estimate `(L(w+h)-L(w-h))/(2h)`, and compare against reverse-mode output. Use appropriately scaled `h` and tolerate Float32 error.

## 5. A validated training request

[`training.cpp`](../MLLibrary/Training/training.cpp) begins by validating that model nodes, forward/cost program, and training/test matrices exist and have nonempty storage. It also checks matching image/label row counts, feature widths, label widths, model input/target sizes, nonzero epochs/batch size, finite positive learning rate, supported optimizer kind, bounded momentum/betas, positive epsilon, nonnegative decay/clipping and at least one trainable parameter.

The descriptor in [training.hpp](../MLLibrary/Training/training.hpp) includes `epochs`, `batch_size`, `learning_rate`, `optimizer`, `momentum`, `beta1`, `beta2`, `epsilon`, `weight_decay`, `max_gradient_norm`, `seed`, `reject_non_finite`, and optional `metrics_csv_path`. `model_train` returns a `ModelTrainingResult` with success status, completed epochs/steps, final metrics, and error message.

The visible `main.cpp` calls `model_train(model, &desc)` without inspecting its returned result. A teaching or production caller should explicitly check this result before treating post-training outputs as evidence of successful optimization. This is an API-use observation, not a claim that training itself fails.

## 6. Gradient update and optimizer state

The default `ModelTrainingDesc` optimizer is SGD, but [`main.cpp`](../MLLibrary/main.cpp) selects momentum SGD with `momentum=0.9`, ten epochs, batch size 50 and learning rate 0.01. For a momentum-style optimizer, parameter updates depend on both the new gradient and stored velocity: this is why optimizer state is part of the semantic training state, not disposable cache. Adam/AdamW additionally track first/second moments, bias-correction step and decay semantics.

The header permits optional global gradient-norm clipping and rejection of non-finite training states. Those controls must be reasoned about at the **update boundary**: detecting a bad gradient only after mutating half of the parameters would violate an all-or-nothing optimizer-step expectation. Review the concrete training and checkpoint implementations before claiming transactional guarantees for each failure path.

With 60,000 training images and batch size 50, a full epoch contains 1,200 complete mini-batches if every sample is visited exactly once, and ten epochs nominally contain 12,000 such batches. Actual completed steps should be read from the returned `ModelTrainingResult`, not inferred when a run may fail or stop early.

## 7. Data and output lineage

`main` reads four raw binary files by expected shape, one-hot encodes labels to width ten, and prints one image and label. It runs pre-training inference, trains, then re-runs inference on a test image. `training_metrics.csv` can be transformed by `MLLibraryTrainingReport` into a standalone HTML training visualization. The data files' existence, encoding, classification correctness and measured accuracy must be verified from the actual host and dataset; none are guaranteed merely by source code.

Source-tree note: the normal build is classic C++23 compilation, whereas `ModuleInterfaces/` describes a prospective modular boundary and the `MLLIB_USE_KAIRO_MATH` option builds an additional module-consuming path. Do not imply that every classic `mat_mul` or `model_train` invocation is automatically dispatched to KairoSIMD/GPU.

## 8. Suggested tests and next implementation checkpoints

1. **Arena boundary:** allocate aligned blocks, verify addresses and content, exercise reserve/commit failure and arena lifetime without dereferencing invalidated pointers.
2. **PRNG repeatability:** same seed/sequence yields identical finite integer streams; different initialization should be tested independently of training order.
3. **Shape safety:** reject incompatible MatMul/add/loss inputs without reading past buffers. Differentiate null-result APIs from thrown errors.
4. **Autograd:** verify numerical gradients for matmul, ReLU away from its nondifferentiable kink, softmax and cross-entropy, including the residual branch.
5. **Training reproducibility:** start from identical weights, data order and descriptor seed; compare parameter values and metric histories.
6. **Failure propagation:** check `ModelTrainingResult` for missing input, mismatched labels, zero batch size and intentionally non-finite gradients.
7. **Resume correctness:** confirm saved weights, RNG progress, optimizer moments and step counters lead to the same next update when the checkpoint format promises it.
8. **Performance:** record wall-clock per step, allocation count, matrix multiply cost and final loss/accuracy separately; benchmark only on a known host and dataset.

**Evidence status:** source-backed explanation, not an executed compiler/test/accuracy claim.
