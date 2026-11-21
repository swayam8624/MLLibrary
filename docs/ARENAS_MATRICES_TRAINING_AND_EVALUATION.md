# A full MLLibrary execution trace: memory, numerics, training, and fair evaluation

This is a deep companion to [SOURCE_READING_ORDER.md](SOURCE_READING_ORDER.md), with links to *implemented* APIs. It is neither a claim that this research library matches modern industrial frameworks nor a promise that all optional C++23 modules and GPU backends are built in the same configuration.

## 1. How does a Matrix exist in memory?

Start from [MemArena::create and push_aligned](../MLLibrary/Core/arena.cpp) with declarations in [arena.h](../MLLibrary/Core/arena.h). The allocator reserves a contiguous range of **virtual address space**, initially commits a subset of its pages, and advances a bump pointer as allocations occur. This contrasts with making a separate `new[]`/heap allocation for each tensor. The public API returns a pointer or failure, and alignment must be a power of two. Virtual reserve is **not** the same as physical memory committed or already touched.

[`mat_create`](../MLLibrary/Math/matrix.cpp) requests both a `Matrix` header and its contiguous `float` array from the arena. The header contains shape metadata and a borrowed data pointer; it is not a reference-counted owner. Destroying/clearing/popping the arena invalidates all matrix headers or data stored in the released range. A nested temporary arena scope should never return a matrix whose payload resides inside a soon-to-be-rewound temporary allocation.

Consider an example `A` with shape `[2,3]`. Its six `float` values occupy one row-major contiguous block. To locate `A[i,j]`, the natural offset is `i*3 + j`; transpose flags in a later multiplication alter interpretation of indices but do not magically transpose the allocated bytes.

**Exercise:** draw a timeline of a training batch containing `mat_create(A)`, `begin_temp()`, an intermediate result, and scope exit. Which pointers survive? What changes if the optimizer stores a moment matrix in a temporary arena by mistake?

## 2. Understand matrix multiplication as a contract, not just three loops

[`mat_mul`](../MLLibrary/Math/matrix.cpp) accepts an output `Matrix*`, two operands, flags to transpose either operand, and a `zero_out` flag. Its effective shape check first computes `A'` and `B'` shapes. For valid `C = A' B'`, the contracted dimensions must match and the preallocated `out` must have the resulting shape.

Why is `zero_out` a serious argument? With `zero_out=true`, existing output is discarded and this call computes a fresh product. With `zero_out=false`, the routine **accumulates** onto its existing output, useful for some gradient paths but disastrous if a caller accidentally retains data from a previous mini-batch.

The correct mental model is **mathematical operation + storage side effects**. Shape validation alone does not tell you whether operand/output aliasing is permitted, whether nonfinite floats propagate, or whether the operation can accumulate stale values. Investigate those conditions in source and tests before treating the kernel as a generic BLAS substitute.

## 3. Softmax, cross entropy, and why naive formulas fail

[`mat_softmax`](../MLLibrary/Math/matrix.cpp) calculates a normalized exponential distribution but first subtracts the largest input value. The mathematical identity

\[
p_i = \frac{\exp(z_i - m)}{\sum_j \exp(z_j - m)},\quad m=\max_j z_j,
\]

produces the same normalized values in exact arithmetic as exponentiating raw logits, while reducing overflow risk for large positive logits. The implementation checks for finite inputs/intermediates and returns failure for invalid shapes or empty arrays. The API treats its input as one flattened distribution; it is **not inherently batched row-wise**.

Cross entropy and softmax may be combined as a stable fused operation. Read [`mat_softmax_cross_entropy`](../MLLibrary/Math/matrix.cpp) and its backward companion. A log of a very tiny naive softmax probability can blow up numerically; a stable logits-based loss guards against that. The backward pass often benefits from the cancellation of the softmax Jacobian with the loss derivative, but **check the actual target encoding and normalization in source** rather than applying a memorized derivative blindly.

**Exercise:** evaluate logits `[1000,999,998]` using a naive `exp(1000)` calculation and a max-subtracted computation. What happens under 32-bit floating point?

## 4. Follow one batch from a dataset into the optimizer

[`model_train`](../MLLibrary/Training/training.cpp) runs a real, inspectable training loop.

First it validates its training request and allocates optimizer moment matrices when required. It creates a deterministically seeded PRNG and shuffles the training-example index vector with a Fisher–Yates-style backward swap per epoch. The seed and data order belong in the experiment record. It then iterates mini-batches, including a potentially shorter last batch.

For each batch, the function clears parameter-required gradients. For each sample, it copies that sample's feature row and target row into the model's fixed input/desired-output matrices, invokes `model_prog_compute` for the forward graph and `model_prog_compute_grads` for the backward graph, then adds sample cost. Only **after** the batch does it divide gradients by batch size and calculate the global gradient norm for optional clipping.

That distinction is important: if the model's backward pass replaces gradients instead of accumulating them, the batch update will be incorrect. The training driver explicitly clears before the batch and scales afterwards; readers should confirm the reverse-mode accumulation behavior in [Model/model.cpp](../MLLibrary/Model/model.cpp).

Next, the optimizer calculates a proposed update. For SGD, the change follows `weight -= learning_rate * gradient`. Momentum variants store a running first moment; RMSProp stores a running squared-gradient estimate; Adam and AdamW use first/second moments with bias correction. In this implementation, Adam's weight decay is folded into the gradient, whereas AdamW applies decoupled decay to parameter values. They are *not* equivalent updates.

The interesting failure path is numerical safety: when `reject_non_finite` is enabled, this driver checks loss, parameter gradient, total squared norm, and **stages** candidate optimizer values to detect non-finite updates before applying parameter changes. This helps ensure a failed batch does not partially update half the model.

```text
Training/model_train
  -> shuffled example indices (Core/prng)
  -> copy sample features/targets into model input variables
  -> Model/model_prog_compute
      -> Math/matrix kernels
  -> Model/model_prog_compute_grads
      -> reverse-mode gradient accumulation
  -> scale + (optional) clip gradients
  -> validate prospective parameter/moment update
  -> commit optimizer step
  -> evaluate/report/checkpoint
```

**Exercise:** implement a tiny two-parameter linear network with a known gradient, then compare one SGD update, one momentum update and one AdamW update. Keep learning rate and data fixed. Verify the moments and weight decay term separately.

## 5. Preprocessing must be fitted on the training fold only

The repository also has a separate classical-data experimentation stack. [FittedPipeline::fit](../MLLibrary/Preprocessing/pipeline.cpp) consumes a `Dataset` and builds a snapshot of numeric imputation statistics, scaling ranges and categorical feature mappings. It records schema/fingerprint metadata. The counterpart [`transform`](../MLLibrary/Preprocessing/pipeline.cpp) applies that **frozen** state to another compatible dataset, preserving column order and target alignment.

The correct workflow is:

```text
split raw records into train and held-out evaluation sets
    |
    +-- training records -> FittedPipeline::fit
    |                        |
    |                        +-- transform(training)
    |
    +-- held-out records -----+-- transform(held_out), no fit()
```

If instead you normalize every record before splitting, the training fold learns from held-out distribution statistics. That is data leakage even if the model optimizer itself never sees held-out labels.

Unknown categories should obey a predetermined policy; adding a new category from the held-out fold to the learned feature vocabulary changes the feature space after training. Text columns may require an explicit transform or drop policy. Keep feature names, column ordering, schema version, and pipeline state together with the estimator in a reproducible experiment.

The [save](../MLLibrary/Preprocessing/pipeline.cpp) and [load](../MLLibrary/Preprocessing/pipeline.cpp) routines persist the fitted transform with a versioned format and validate it when loading. A model checkpoint without its matching preprocessing pipeline may be mathematically well-formed yet semantically wrong.

## 6. A scientific result needs more than successful compilation

The [Experiment layer](../MLLibrary/Experiment/experiment.cpp) turns prepared datasets into classical estimators and records. [Evaluation](../MLLibrary/Evaluation/metrics.cpp) and [CrossValidation](../MLLibrary/Evaluation/cross_validation.cpp) make evaluation criteria explicit. [Search](../MLLibrary/Search/search.cpp) and [AutoML](../MLLibrary/AutoML/automl.cpp) evaluate alternatives.

When using cross-validation, each fold's preprocessing must be fitted exclusively on that fold's training partition. Hyperparameter choices guided by validation should be evaluated once on a genuinely held-out test partition when making a generalization claim. A visually impressive training-loss curve can be compatible with severe overfitting or leakage.

For any reported metric, record the feature schema, label mapping, split/seed, preprocessing fingerprint, estimator config, checkpoint, optimizer state, compiler/build variant, and exact evaluation function. Include confidence intervals or repeated-seed variability when comparing close results.

## 7. What to test and how to read this repository

Use the existing test entry points in [Tests](../Tests) to examine arena ownership, PRNG sequences, matrix kernels, gradient correctness, training checkpoints, dataset/pipeline contracts and experiment workflow. For numerical kernels, compare both value and gradient against a small independent calculation, and deliberately include ill-conditioned or nonfinite edge cases.

For performance claims, separate cold allocation from steady-state kernel work, include the exact matrix sizes/layouts and backends, and measure repeatability. C++23 module files in [ModuleInterfaces](../ModuleInterfaces) are an alternative public surface and should be checked against CMake targets before asserting that a test exercises that module variant.

A good explanation of this library ends with **where the bytes live, which function changes them, why the math is stable, and how evaluation avoids leakage**. Those four questions remain relevant even when the implementation grows a GPU backend or a different model family.
