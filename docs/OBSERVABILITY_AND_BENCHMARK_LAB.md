# Observability laboratory — from MLLibrary results to reports and reproducible benchmarks

Read alongside [`Training/training.cpp`](../MLLibrary/Training/training.cpp), [`Visual/training_report.cpp`](../MLLibrary/Visual/training_report.cpp), [`Visual/dataset_report.cpp`](../MLLibrary/Visual/dataset_report.cpp), [`Visual/dataset_inspector.cpp`](../MLLibrary/Visual/dataset_inspector.cpp), [`Benchmark/matrix_benchmark.cpp`](../MLLibrary/Benchmark/matrix_benchmark.cpp), [`Benchmark/benchmark.cpp`](../MLLibrary/Benchmark/benchmark.cpp), and [`Contracts/runtime_contracts.cpp`](../MLLibrary/Contracts/runtime_contracts.cpp).

This chapter explains **what each report observes**, what it does not measure, and which numerical and platform assumptions are necessary before making a claim. All descriptions refer to the checked-in source and public APIs; no benchmark was executed in this documentation pass.

## 1. A training chart is a postprocessor of recorded measurements

The classic `model_train` function in [`Training/training.cpp`](../MLLibrary/Training/training.cpp) can write one CSV row per completed epoch with the exact header

```csv
epoch,accuracy,cost,learning_rate
```

The values are the current epoch index, held-out evaluation accuracy as a fraction, mean evaluation loss, and selected training learning rate. The training loop also displays batch costs in the terminal, but the visualization reads only the **epoch-level CSV**, not every per-batch event.

[`Visual/training_report.cpp`](../MLLibrary/Visual/training_report.cpp) validates the four-column header and parses every nonblank row using `std::from_chars`; it refuses an empty history. It then emits a standalone HTML document containing final numbers and canvas traces for loss, accuracy and learning rate. A report's final accuracy is calculated from the *last CSV row*, not from a fresh model inference. The HTML contains local JavaScript visualization code, not an API request that uploads training examples.

**Reasoning exercise:** if a run finishes three epochs and the CSV contains `0.70`, `0.80`, `0.84` accuracy, the final HTML should display 84%, with the three values plotted over time. It cannot tell you whether the data split leaked labels or whether another seed would succeed. Those questions are about the training protocol.

**Audit caution:** the parser checks numeric syntax but does not explicitly enforce finite bounds on every metric at its read boundary; passing strings representing non-finite values through some standard-library parsing implementations is a case to test. The plot is not a numerical validation oracle.

## 2. There are two different dataset reports

[`Visual/dataset_report.cpp`](../MLLibrary/Visual/dataset_report.cpp) operates on **raw Float32 MNIST-compatible arrays**: image data must have exactly 784 floats for every label; labels are checked for integral values from zero through nine; images must be finite. It computes sample count, pixel mean, standard deviation, extrema, class histogram and a deterministic sample grid. Only the rendered grayscale intensities are clamped into `[0,1]`; the summary statistics are calculated on the original numeric values.

The sample display is bounded to 1–500 images so one HTML file does not accidentally embed an unbounded dataset. Both CLI argument checks and the class-distribution calculation happen before the report is written.

[`Visual/dataset_inspector.cpp`](../MLLibrary/Visual/dataset_inspector.cpp), by contrast, uses the **typed CSV** stack: it invokes `load_inferred_csv`, `profile_dataset`, and both JSON/HTML report writers. It can accept a target-column name. Profile statistics include inferred schema, missingness, duplicates, distribution warnings and an FNV-style content fingerprint.

A visual image tile is not an explanation of dataset provenance; a text/categorical schema summary is not proof that feature types are semantically correct. They support different questions.

## 3. Matrix benchmarks measure an actual kernel, but not all performance

[`Benchmark/matrix_benchmark.cpp`](../MLLibrary/Benchmark/matrix_benchmark.cpp) parses a matrix dimension, iteration count, output file and seed. It bounds the dimension to 2048 and rejects nonpositive iteration counts. It allocates three square Float32 `Matrix` objects through `MemArena`, fills left/right operands with seeded pseudo-random values, warms the production `matrix.cpp::mat_mul` twice, and times repeated overwrite matrix multiplications with `steady_clock`.

It separately computes a **double-precision reference** for each output element and records the maximum absolute difference versus the Float32 result. Its throughput approximation counts `2 * N^3` arithmetic operations per square GEMM; this is a conventional FLOP accounting convention, not a hardware counter.

The executable also records host compiler/platform identifiers, p50/p95 timings, mean latency and process peak resident-memory high water mark. The C++ path uses `getrusage`; its memory units differ between macOS and Linux and are normalized by code. This reports process-wide high-water RSS, not per-kernel allocation or GPU memory residency.

**Important distinction:** this executable calls the classic scalar `mat_mul`, not `KairoSIMD::Dot`, `KairoScheduler` or Metal kernels. Neither `2*N^3` nor the `cpu.scalar.mat_mul` benchmark name proves vectorization, CPU utilization or GPU speedup.

## 4. Walk through a reproducible benchmark record

[Benchmark/benchmark.cpp](../MLLibrary/Benchmark/benchmark.cpp) defines a versioned JSON serializer. Its `ExperimentManifest::validate` requires benchmark/run identity and compiler/platform information. `BenchmarkResult::validate` checks measured-iteration count and rejects negative/non-finite latency, throughput or absolute-error fields. `write_json_report` validates inputs, handles output errors and escapes JSON strings.

Suggested command on a properly configured and built host:

```bash
cmake -S . -B build -G Ninja -DBUILD_TESTING=ON
cmake --build build --target MLLibraryBenchmark

# Data-generating work: this runs only on the user/CI machine, not in
# the documentation process that created this chapter.
./build/MLLibraryBenchmark \
  --output matrix-benchmark.json \
  --size 128 \
  --iterations 10 \
  --seed 5489
```

A good evidence ledger also records machine model, CPU frequency/governor and thermal state, OS and toolchain build, compiler optimization flags, commit SHA, number of repetitions and end-to-end program time. Compiler/platform labels alone are not enough to reproduce a published performance comparison.

### Why median, p95 and max error all matter

Mean latency is sensitive to scheduling interruptions. Median is a representative central sample, while p95 reports slower observations (with limited stability if very few repetitions are measured). Maximum absolute error checks numerical agreement with the implementation's double reference, but it does not by itself detect all pathological inputs or establish a valid relative-error scale.

If an algorithm is fast but produces incorrect outputs, it is not an optimization of the same computation. If it is numerically correct but uses different input initialization or different warmup, its performance comparison is not controlled.

For large dimensions, consider allocation costs and memory bandwidth, and compare with a matched scalar/optimized CPU/Metal pipeline that has the same mathematical input/output contract. GPU benchmarks must separately report upload, launch/compile, GPU completion, readback and end-to-end wall time. See [KairoGPU's Metal laboratory](https://github.com/swayam8624/KairoGPU/blob/main/docs/METAL_COMPUTE_PIPELINE_LAB.md).

## 5. A capability descriptor is not proof of device support

[`Contracts/runtime_contracts.cpp`](../MLLibrary/Contracts/runtime_contracts.cpp) validates dtype byte widths, tensor shape products, strides/layout and declared storage ownership; it also registers and searches model capability metadata.

For a tensor shape `[N,C]` with Float32 values, `TensorDescriptor::element_count` checks that `N*C` does not overflow `size_t`; `byte_size` checks that multiplication by four is safe. The code can distinguish a dense contiguous logical view from an explicit strided one. This is necessary before allocating a buffer or passing a span to an optimized kernel.

`ModelCapabilityRegistry::select` chooses a model metadata record matching the requested task, dtype, device, rank, input byte bound and inference/training support. **Matching a capability record does not create a GPU device, compile a model, or execute a kernel.** Actual host and backend availability still have to be established.

The `CheckpointFormat` metadata validation is separate from the classic `KMLCP01` weight-only binary serializer. It must not be represented as evidence that the latter saves optimizer state.

## 6. Report integrity and visualization limitations

Standalone HTML and JSON make results easy to inspect without depending on an external dashboard. But the renderer should never be the only surviving evidence: preserve the source CSV, benchmark JSON, code revision and input-data fingerprint.

For any generated HTML, verify that externally derived names/labels are escaped before insertion into markup or script; data-only reports have a much smaller attack surface than arbitrary string interpolation. For numerical arrays, reject or handle `NaN` and infinity explicitly rather than relying on browser behavior.

For large training histories, a fixed-size HTML canvas visualization may have different scaling and interactivity limitations than a dedicated plotting system. This is not a reason to claim the underlying metrics are less correct, but it informs what the report is useful for.

## 7. Source-linked verification and extension tasks

1. Produce a small valid training CSV and confirm exactly three chart series and the last-epoch summary. Test missing header, extra columns, empty file, NaN/Infinity and malformed values.
2. Inspect a tiny raw 28x28 image fixture. Check label bounds, finite pixels, exact image/label count and the 1–500 sample limit.
3. Run the scalar matrix benchmark at several sizes with a fixed seed. Record p50/p95 and error, and compare against a separately implemented high-precision reference. Avoid inferring speedup from a single result.
4. Use `TensorDescriptor` with empty, zero-dimension, overflowing and strided shapes; confirm explicit failures before allocating memory.
5. Register a capability and attempt to select it with an incompatible dtype/device/rank to observe the typed error contract.
6. For production-quality comparisons, add reproducible optimization flags, hardware identity, allocation breakdown, CPU/GPU sync evidence and matched reference outputs to the manifest.

**Evidence status:** source-level teaching and audit, not a benchmark run, compiled client, system profiler measurement or verified HTML render.
