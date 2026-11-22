# Checkpoint format, exact resumption, and reproducibility — teaching audit

**Read the real implementation:** [`Training/checkpoint.cpp`](../MLLibrary/Training/checkpoint.cpp), [`Training/training.cpp`](../MLLibrary/Training/training.cpp), [`Core/prng.cpp`](../MLLibrary/Core/prng.cpp), [`Model/model.hpp`](../MLLibrary/Model/model.hpp). This guide is an audit of the current **classic MLLibrary** route; it is **not** a description of KairoMath's separately implemented TensorTrainingCheckpoint.

## 1. The current KMLCP01 format saves *weights*

The file starts with seven magic bytes `KMLCP01`, then native-endian `u32 version = 1`, then native-endian `u32 parameterCount`. For every node marked `MV_FLAG_PARAMETER` in the **compiled cost program's traversal order**, it writes a native `u32 rows`, native `u32 columns`, and then `rows * columns` `f32` values. The serializer uses `std::fwrite` and `std::filesystem::rename` of a temporary path.

On load, the implementation checks magic/version/count/dimensions and reads each parameter into temporary `std::vector<f32>` storage. Only after the full payload was read successfully does it copy each vector's contents to the live arena-backed parameter matrix. This gives *all-or-nothing parameter mutation* for short reads and mismatched dimensions, not a complete end-to-end integrity guarantee.

The format currently **does not include**:
- momentum velocity or Adam/AdamW first/second moments;
- the optimizer kind, learning rate, weight decay or gradient clipping settings;
- the number of completed optimizer steps, epochs or partial-batch offset;
- the training-local PRNG state and exact shuffled sample order;
- the global initialization PRNG state, dataset version or preprocessing provenance;
- a checksum, endianness marker, parameter names or architecture fingerprint.

Therefore, loading a checkpoint reconstructs **inference weights** (assuming the same graph/parameter order), but it cannot reproduce the exact next stateful optimizer step of an interrupted training run. A new call to `model_train` allocates fresh moment arrays in a temporary scratch arena and sets its completed-step count to zero. This is particularly relevant to Adam bias correction, momentum and shuffled sample order.

## 2. Two RNG streams matter

[`mat_fill_rand`](../MLLibrary/Math/matrix.cpp) calls the global `prng_randf`, while `model_train` constructs a training-local PRNG from `ModelTrainingDesc::seed` and a derived sequence for shuffling. Two runs with the same descriptor seed may still start from different parameters when global initialization state differs. Matching only train/test data and optimizer name is not enough for reproducibility.

The training shuffle uses a rejection-sampled Fisher-Yates index permutation. For an exact experiment record, capture parameter initialization, sample-order stream, dataset, preprocessing transformations, optimizer settings, compiler, platform, and numerical mode.

## 3. Why the temporary file is helpful but not sufficient

Saving to `path.tmp` before rename reduces the risk of exposing a partly written destination file. However, do not equate rename with a guaranteed durable transaction across power loss or with cross-platform identical replacement semantics. The source does not flush all content through an explicit `fsync`/equivalent durability procedure.

**Source review issue to test:** the save function combines prior `success` and `std::fclose(file)` via short-circuit `&&`. If an earlier write fails and `success` is already false, `std::fclose(file)` is skipped, leaving the opened stream to be cleaned up by process teardown rather than that failure path. That is a likely resource-management defect, not a tested fault-injection result. A future runtime-fix pass should always close the stream, then report both write and close outcomes without changing checkpoint semantics silently.

**Other validation limitations:** loader shape equality does not prove that parameters have the same semantic names/order. The reader also does not enforce a trailing-byte policy or checksum; a compatible-looking but corrupted payload may be accepted. The binary has architecture-dependent native integer/Float32 representation assumptions.

## 4. A small, testable weight-only contract

Build a tiny graph with two learnable parameter matrices. Set them to known values, save, alter their live contents, load, and assert exact restoration. Then verify the file rejects a wrong magic, wrong version, truncated value data, wrong matrix count, and changed matrix shape **without modifying live parameter buffers**.

A separate test should save a model then build a graph with the same shapes but a different parameter order. It illustrates that structural dimension agreement alone does not establish semantic compatibility.

Use a temporary filesystem location, clean up files and record the error string returned on failure. Inject write/rename errors where feasible to test stream/resource cleanup. The test should not assume that saving a checkpoint automatically serializes optimizer state.

## 5. Requirements for future exact-resume checkpoints

To claim complete training resumption, a new versioned format would need to serialize and validate all state that influences the next update: parameters, per-parameter optimizer moments, optimizer step, epoch/batch position, PRNG internal state and sample traversal, hyperparameters, dataset identity and shape/provenance. It also needs bounded length validation, stable byte order, integrity checks, robust write/close/rename/durability behavior, and compatibility rules.

This is a **design target**, not current functionality. Do not reuse the existing version-1 magic to claim it does more than it actually stores.

**Evidence status:** source-level analysis; no checkpoint round-trip or fault-injection tests were executed during this documentation update.
