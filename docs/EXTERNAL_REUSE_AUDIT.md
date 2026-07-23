# External Reuse Audit

Updated: 2026-07-23

## Purpose

This is the living inventory for the Kairo local multimodal device-agent
program. It is not a claim to have found every repository on the internet.
That is neither finite nor useful. It records the mature upstream code and
official platform APIs that materially overlap the product, and it states the
integration decision before Kairo code is written.

The rule is simple: adopt a proven subsystem behind a narrow adapter when it is
not core differentiation. Build Kairo code only where the product requires a
stable local contract across perception, state grounding, policy, action
verification, replay, and correction-driven learning.

## Decision Classes

| Class | Meaning |
| --- | --- |
| Adopt | Consume an upstream API/library behind a Kairo interface; do not fork or reimplement its algorithms. |
| Reference | Study its architecture/tests/formats; Kairo needs a different scope or dependency boundary. |
| Defer | Keep the integration optional until the concrete vertical requires it. |
| Reject | Do not add it because it conflicts with the local deterministic control path or duplicates owned Kairo work. |

## Platform And Application Control

| Upstream | What already exists | Decision | Kairo boundary |
| --- | --- | --- |
| [Apple ScreenCaptureKit](https://developer.apple.com/documentation/screencapturekit/) | Native high-performance macOS display/window/audio capture with sample buffers and a user-facing content picker. | Adopt | `Kairo.Capture.Mac`; frame timestamps, bounded queues, no raw platform types above adapter. |
| [Apple Accessibility API](https://developer.apple.com/documentation/applicationservices/axuielement_h) | Structured UI hierarchy, state notifications, actions, and settable attributes for assistive apps. | Adopt | `Kairo.AppState.Mac`; read state before action, verify after action, never make OCR the primary source when AX data exists. |
| [Apple Vision](https://developer.apple.com/documentation/vision) | Pretrained local text recognition, image analysis, segmentation, and hand/body pose observations. | Adopt first | `Kairo.Perception.Vision`; use its hand-pose points before training a gesture detector. |
| [Adobe Premiere UXP API](https://developer.adobe.com/premiere-pro/uxp/) | Premiere's current JavaScript extensibility and application API. | Adopt | A signed, version-gated Premiere plugin is the action adapter; accessibility clicks are only a fallback for unsupported operations. |
| [OSWorld](https://os-world.github.io/) | Open task environment and execution-based evaluation for real desktop workflows. | Reference | Adapt its task/evaluation concepts into Kairo's replay and refusal regression suite; do not depend on its generic-agent architecture. |

## Capture, Timeline, And Privacy References

| Upstream | What already exists | Decision | Kairo boundary |
| --- | --- | --- |
| [Screenpipe](https://github.com/screenpipe/screenpipe) | MIT local screen/audio capture, accessibility-first event capture, OCR fallback, local timeline/search, and permission design. | Reference and selectively adopt concepts | Reuse its event-driven capture and retention ideas; do not embed its Rust application or adopt continuous surveillance as Kairo's default behavior. |
| [OBS Studio](https://github.com/obsproject/obs-studio) | Mature capture/encoding/scene application. | Defer | Use only if a future creator workflow needs recording/streaming, not for live device control. |
| [FFmpeg](https://ffmpeg.org/) | Codec/container/transcoding foundation. | Defer | Use for offline replay import/export, never on the latency-critical action loop. |

## Vision, Gesture, OCR, And Audio

| Upstream | What already exists | Decision | Kairo boundary |
| --- | --- | --- |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Real-time hand landmarks and gesture recognition, including C APIs and mobile/edge pipelines. | Reference; benchmark before adopting | Benchmark against Apple Vision. If it wins on required gesture coverage, wrap it in `Kairo.Perception.Hand`; do not rebuild landmark tracking. |
| [OpenCV](https://github.com/opencv/opencv) | Apache-2.0 computer vision primitives, video I/O, calibration, tracking, and image transforms. | Adopt selectively | Use for image preprocessing/calibration where native frameworks do not cover the need; keep tensors and scheduling Kairo-owned. |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | Mature OCR engine. | Defer | Apple Vision OCR is the macOS baseline; retain Tesseract only as a cross-platform fallback evaluation target. |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Plain C/C++ local ASR with Apple Silicon NEON/Metal/Core ML paths, VAD, quantization, and real-time examples. | Adopt behind an adapter | `Kairo.Perception.Speech`; do not train or implement a general ASR engine in Phase 1. |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Small speech-activity models. | Defer | Evaluate only if whisper.cpp VAD does not meet latency/accuracy needs. |

## ML And Inference Runtimes

| Upstream | What already exists | Decision | Kairo boundary |
| --- | --- | --- |
| [MLX](https://github.com/ml-explore/mlx) | Apple Silicon array/ML framework with C++, C, and Swift APIs, autodiff, graph optimization, and examples. | Adopt as an optional accelerator | `Kairo.Backend.MLX`; use it for model experimentation and selected local model execution, never let MLX types leak into the core action contracts. |
| [Core ML](https://developer.apple.com/documentation/coreml) | Apple's model deployment framework for CPU/GPU/Neural Engine. | Defer pending benchmark | Add a Core ML backend only after the same model is profiled against MLX/Metal on target hardware. |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Broad, high-performance ONNX inference and execution-provider ecosystem. | Reference and optional compatibility backend | KairoONNX remains a bounded native importer. Use ONNX Runtime only when model/operator coverage is urgent and native execution has measurable gaps. |
| [ncnn](https://github.com/Tencent/ncnn) | BSD-3 C++ inference runtime with NEON, multithreading, Vulkan, fp16/int8, and ONNX conversion. | Reference | Study allocator, operator lowering, and quantized-kernel design; not the primary Apple-Silicon path. |
| [MNN](https://github.com/alibaba/MNN) | Lightweight on-device inference/training engine with mobile and Metal support. | Reference | Benchmark only for cross-platform expansion; avoid a second full model runtime in the initial Mac product. |
| [ExecuTorch](https://pytorch.org/projects/executorch/) | PyTorch edge deployment/export/runtime with hardware backends. | Defer | Useful for models authored in PyTorch; not required for a C++-native Kairo initial vertical. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Local quantized LLM runtime across CPU/Metal/Vulkan/CUDA backends. | Defer and isolate | May power optional Ask/Plan assistance. It is prohibited from the real-time action path and cannot bypass KairoAI policy. |
| [AirLLM](https://github.com/lyogavin/airllm) | Apache-2.0 Python runtime that streams model layers to reduce accelerator-memory residency; its macOS path requires Apple Silicon, MLX, and PyTorch. | Adopt only as an optional isolated planner backend | Launch as a bounded local worker process with explicit model/disk/RAM budgets. Never link it into Kairo's C++ action runtime or schedule it on the perception critical path. |

## What Kairo Must Own

None of the upstream systems provides the complete differentiated loop below as
a focused, deterministic creator-workflow product:

```text
trusted screen/camera/audio capture
  -> calibrated task perception
  -> application-state grounding
  -> registered intent/action binding
  -> exact approval and capability policy
  -> native application command
  -> post-action verification and undo receipt
  -> replay, correction curation, regression evaluation
```

Kairo-owned code therefore remains limited to:

- adapter-neutral evidence, intent, action, receipt, verification, and replay
  contracts;
- state-aware action binding and exact authorization integration;
- application-specific command adapters and version capability tables;
- calibration/fusion policy and measurement tooling;
- a compact-model training/data path only where an upstream model is not good
  enough for a supported action;
- deterministic replay and refusal/action regression tests.

## Explicit Non-Goals

- Do not write another screen recorder, ASR engine, hand-landmark model,
  generic browser/desktop agent, or complete ONNX runtime.
- Do not fork a large upstream project merely to claim ownership; use versioned
  dependencies, adapters, benchmarks, and upstream patches where licensing and
  maintenance permit.
- Do not use a general LLM/VLM as the sole source of UI state or as direct
  execution authority.
- Do not integrate an external runtime until it passes a target-hardware
  latency, memory, correctness, license, supply-chain, and offline-operation
  evaluation.

## Integration Gate For Every Candidate

Before adding any dependency, record:

1. exact version, source URL, license, notices, and security update owner;
2. target-hardware benchmark for p50/p95 latency, memory, CPU/GPU/NE use, and
   cold-start behavior;
3. failure behavior when permissions, models, capture input, or application
   state are unavailable;
4. adapter API that prevents upstream types from entering Kairo action/policy
   modules;
5. deterministic fixture/replay tests and a removal/fallback strategy.

## Immediate Result

Phase 1 should use platform APIs and existing runtimes, not train perception
from scratch:

1. ScreenCaptureKit plus AXUIElement for read-only state grounding.
2. Apple Vision hand pose for the first gesture prototype.
3. whisper.cpp behind a local speech adapter only when voice is enabled.
4. Premiere UXP for supported Premiere actions.
5. KairoAI `DeviceAgent` plus `ToolPolicy` for proposal and approval.
6. Benchmark MLX/Core ML/Metal only after a narrow action has a real model and
   measured workload.

The current first vertical is implemented in
[`KairoMacPerception`](https://github.com/swayam8624/KairoMacPerception):
ScreenCaptureKit, Vision landmarks, a stable gesture recognizer, and an
approval-bound in-memory preview executor. Its companion protocol is shared
between the future Mac and iPad clients, but networking and UI remain Phase 4.

## AirLLM Decision

AirLLM can make very large open-weight models *loadable* on modest accelerator
memory by keeping only one layer resident at a time. This is valuable for
offline, low-frequency local planning or summarization experiments. It does not
make those models appropriate for a 32 GB real-time desktop agent: Apple
Silicon uses unified memory, and offloaded execution still depends on model
storage, layer-transfer bandwidth, KV-cache growth, process overhead, heat,
and generation latency.

Use the following deployment tiers:

| Tier | Model class | Role | Rule |
| --- | --- | --- | --- |
| Always-on | no LLM, or compact task model | vision/audio/gesture, policy, grounding, verification | must meet real-time budget without AirLLM/MLX-LM/llama.cpp dependency |
| Interactive local | roughly 3B-9B 4-bit model | Ask/Plan explanations and constrained intent suggestions | run in a separate process with a fixed context and memory cap |
| Deliberate local | roughly 14B-32B quantized model | optional longer planning or replay analysis | pause when capture/perception needs resources; never hold control-loop state |
| Offloaded experiment | 70B+ AirLLM model | offline research, batch review, or non-urgent planning | no action authority; benchmark disk/RAM/token latency before enabling |

On Apple Silicon, benchmark MLX and llama.cpp first for the interactive tier.
MLX is native to Apple unified memory and offers C++/C/Swift APIs; llama.cpp is
a dependency-light C/C++ runtime optimized for ARM NEON, Accelerate, Metal, and
quantized GGUF models. AirLLM remains the escape hatch for models that cannot
otherwise reside comfortably, not the default deployment choice.
