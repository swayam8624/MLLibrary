# Kairo Local Multimodal Device Agent

## Product Boundary

Kairo is not a generic chatbot that clicks a desktop. It is a local-first,
real-time perception-to-action runtime for explicitly supported workflows.
Its first vertical is visual creative work: screen-region capture, media-bin
organization, volume adjustment, and later verified Premiere timeline edits.

The novelty is the closed loop: compact local perception models propose a
bounded action; the active application adapter verifies the target state; the
existing KairoAI policy requires the exact approved invocation; the adapter
observes the result; and the run becomes replayable training evidence.

## Inputs And Outputs

| Boundary | Input | Output | Owner |
| --- | --- | --- | --- |
| Perception | camera frames, display frames, audio windows, accessibility selection state | timestamped typed evidence plus confidence | vision/audio/screen adapters |
| Intent | evidence window and active application identifier | typed `IntentKind`, evidence IDs, calibrated confidence | local intent model/rules |
| Grounding | intent plus visible UI state and app selection | canonical action arguments or refusal | application adapter |
| Policy | `ToolCall`, registered capability, exact user approval | allow/refuse decision | `Kairo.AI.ToolPolicy` |
| Execution | policy-approved action | command receipt, undo token where supported | OS/application adapter |
| Verification | post-action screen/application state | succeeded, failed, or ambiguous event | application adapter |
| Learning | replay record and explicit user correction | curated local training example | dataset/training pipeline |

No untrusted speech transcript, screen text, video pixel, web page, or model
output is execution authority. They can influence an intent proposal only.

## Reuse Map

| Existing code | Reuse now | Do not duplicate |
| --- | --- | --- |
| `KairoAI/Kairo.AI.ToolPolicy` | capability modes, exact approval binding, host tool registry | permission checks, credential policy, handler registry |
| `KairoAI/Kairo.AI.DeviceAgent` | evidence-to-intent-to-registered-action proposal contract | a second generic desktop-agent or action permission system |
| `KairoMath/Tensor.cppm` | frame/audio tensors, model weights, activations, losses, SGD reference path | a second tensor type or matrix allocator for new models |
| `KairoScheduler` | frame pipeline stages, batched preprocessing, background replay export | ad hoc thread pools per model or app adapter |
| `KairoSIMD` | image normalization, audio features, reductions, activation kernels | CPU feature checks scattered through model code |
| `KairoGPU` | future Metal command/buffer abstraction | fake GPU paths or app-specific GPU ownership |
| `KairoONNX` | bounded inference import once parser/executor exist | broad, untested ONNX compatibility claims |
| `KairoTransformers` | optional small temporal fusion/inference only | transformer training before tensor autodiff and kernels mature |
| `MLLibrary` | reference autograd/training behavior, metrics, future visual training tooling | a competing production runtime beside KairoMath |

## Current Implementation Truth

Implemented today:

- KairoMath scalar tensor operations and SGD reference update.
- MLLibrary matrix/autograd MNIST seed, SGD/momentum, and CSV metrics.
- Scheduler thread pool/range partitioning and scalar SIMD-compatible kernels.
- Backend-neutral GPU, ONNX graph, and transformer configuration contracts.
- KairoAI exact-call approval and a device-agent intent router. The router
  accepts only registered bindings and verified host evidence, then emits a
  `ToolCall` proposal. It cannot execute an OS action.

Not implemented today:

- camera, microphone, screen capture, gesture, ASR, or CV models;
- macOS accessibility/AppleScript/Adobe adapters;
- persistent replay store, undo integration, calibration, or visual dashboard;
- actual SIMD intrinsics, Metal kernels, ONNX protobuf parsing, or transformer
  execution/training.

## Non-Negotiable Constraints

- Local models and deterministic rules are the real-time path. A cloud or LLM
  provider may assist in Ask/Plan mode but never gates a time-critical action.
- A host adapter creates action arguments only after visual/state validation.
- Each mutation requires a registered capability and an exact-call approval
  until an explicit, narrowly scoped auto-execute policy is independently
  designed and tested.
- Every mutation must have a receipt, verification result, and an undo route
  where the host application exposes one.
- Perception models must expose calibration data; raw confidence alone is not a
  safety claim.

## Phase Plan And Exit Gates

### Phase 0: Deterministic Control Contract

Deliver typed evidence, registered app bindings, action proposal, KairoAI
authorization handoff, receipt/verification interfaces, deterministic replay
format, and tests for every refusal branch.

Exit gate: a synthetic Premiere trim proposal cannot reach a handler when any
required evidence is absent, weak, mismatched to the active app, or unapproved.

### Phase 1: Read-Only Visual Grounding

Add macOS screen capture and accessibility state adapters, an on-screen visual
debugger, frame timestamps, active-app identification, selected-control
verification, and replay recording. No mutations beyond a test harness.

Exit gate: replay deterministically reproduces the proposed action or refusal
for a captured session; visual grounding false-positive rate is measured for
each supported application state.

### Phase 2: Perception Adapters (Implemented Safe Vertical)

`KairoMacPerception` now provides a native macOS baseline: ScreenCaptureKit
one-shot display capture, Vision hand-pose landmarks, and an interpretable
two-pinch rectangle recognizer that requires stable evidence across frames.
It does not record continuously or write captured pixels to disk.

The adapter surface is complete for the first vertical. Future perception work
is model expansion, not a new control contract: calibration capture, measured
false-positive/negative rates, an iPad camera source, and optional local speech
via whisper.cpp.

### Phase 3: Single Safe Action Vertical (Implemented)

The initial action is creating an in-memory crop preview from a verified screen
region. It is reversible by `discard(id:)`; it cannot edit an application,
change a system setting, overwrite a file, or export an artifact. A
`PreviewActionExecutor` rejects stale state, accepts only an externally issued
approval identity, verifies preview ownership, and makes undo discard the only
owned reference. KairoAI records the proposal, exact authorization boundary,
receipt, verification, and replay identity.

The remaining Phase 3 work is target-hardware measurement and a host bridge
that turns KairoAI approval objects into `ApprovedPreviewRequest`; it does not
expand action authority.

Exit gate: 99%+ correct action verification in a controlled test suite, zero
unapproved mutations, and bounded end-to-end latency measured on target Macs.

### Phase 4: Visual Companion Control Surface (Contract Ready)

`KairoControlProtocol` is a shared, transport-neutral Swift target for the Mac
host and future iPad companion. It allows the companion to request a preview,
approve/reject a known call, or discard a known preview. It rejects arbitrary
arguments, while the Mac remains the sole policy, state-validation, execution,
verification, and replay authority.

Phase 4 implementation starts with a Mac host state viewer and an iPad pairing
screen over authenticated local networking. It then adds the proposal queue,
preview metadata, approval controls, and replay timeline. No companion message
may directly invoke an accessibility or application command.

### Phase 3: Creative Application Adapter

Add Premiere-specific state inspection and a single reversible operation,
starting with selecting/marking a clip before trim or split. Arguments must be
constructed from the verified timeline selection, never free-form language.

Exit gate: action replay, mismatch refusal, undo, app-version compatibility,
and recovery after a stale screen frame are all tested.

### Phase 4: Local Learning And Performance

Move preprocessing to scheduler/SIMD dispatch, implement Metal through
KairoGPU, train compact gesture/audio/visual classifiers with KairoMath-backed
data paths, add active-learning review, calibration, regression suites, and
resource budgets.

Exit gate: latency, memory, thermal behavior, and correction-driven accuracy
improvements are tracked per model and per supported workflow.

### Phase 5: Broader Multimodal Workflows

Add more applications only after each adapter meets the same state grounding,
approval, verification, undo, and replay standard. Small transformer-style
temporal models may fuse evidence only after baseline specialized models are
measured to be insufficient.

## Initial Performance Budget

On a 32 GB Apple Silicon machine, reserve the interactive path for compact
models and explicit budgets: capture-to-proposal p95 under 100 ms for gestures,
under 250 ms for short spoken commands, bounded frame queues with dropped stale
frames, and no unbounded video retention. These are targets to measure, not
claimed current performance.

## Deliberate Limitations

- It will not infer arbitrary UI semantics from pixels alone.
- It will not perform broad unattended desktop control.
- It will not execute destructive actions from a gesture, voice phrase, or
  text prompt without the future policy requirements above.
- It will not promise universal Premiere compatibility; each app version and
  action surface needs a tested adapter.
- It will not use a large language model as a substitute for visual grounding,
  deterministic policy, or verification.
