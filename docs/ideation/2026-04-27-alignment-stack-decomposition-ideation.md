---
date: 2026-04-27
topic: alignment-stack-decomposition
focus: Break up the production alignment stack, especially src/tomojax/align, pipeline.py, and losses.py
mode: repo-grounded
---

# Ideation: Alignment Stack Decomposition

## Grounding Context

TomoJAX is a Python/JAX CT reconstruction and alignment package. The production alignment stack already has useful leaf modules under `src/tomojax/align/`, including state, DOFs, schedules, objectives, optimizers, validation residuals, fold reconstruction, checkpointing, and geometry application.

The current pressure points are not just file size, but ownership concentration:

- `src/tomojax/align/pipeline.py` is about 3,015 lines and still owns orchestration, config, reconstruction stepping, pose optimization, multires, setup geometry calibration, observer/stat plumbing, checkpoint handling, and payload construction.
- `src/tomojax/align/losses.py` is about 1,128 lines and mixes loss specs, parsing, schedule resolution, target precompute, kernels, adapter construction, and Gauss-Newton weighting.
- Existing tests are strong enough for targeted CPU-friendly refactoring, but broad numerical behavior still needs manual heavy validation later.

Institutional learning from `docs/solutions/architecture-patterns/reuse-align-multires-for-geometry-calibration-2026-04-25.md` says setup geometry calibration should stay inside the unified alignment system. The refactor target is not a separate calibration product or a private solver family. The production contracts to preserve are stopped train-fold reconstruction, streamed validation residuals, `views_per_batch`, small normal-equation accumulation, active DOF provenance, loss provenance, optimizer provenance, and complete manifest/debug metadata.

External grounding points the same way: decompose in place first. JAX code benefits from pure functions, explicit state, clear differentiation boundaries, and static/runtime metadata separation. Scientific workflow systems and ML technical-debt literature reinforce that modularity only helps when inputs, outputs, config, memory behavior, tests, and provenance are explicit. Branch-by-abstraction and characterization tests are a better fit than a big package split.

## Ranked Ideas

### 1. Stage Engine And Objective Boundary First

**Description:** Keep `align()` and `align_multires()` as the public facade, but extract the execution of resolved stages into a private stage engine. The stage engine should run pose stages, setup-validation stages, reconstruction steps, checkpoint emission, observer adaptation, and stage-local stats through explicit inputs and outputs. In parallel, objective-specific evidence assembly should move out of `pipeline.py` so fixed-volume pose objectives and stopped train-fold validation objectives expose a consistent objective contract.

**Warrant:** `direct:` `pipeline.py` currently spans config resolution, reconstruction steps, pose optimization, setup geometry validation-LM, checkpoint callbacks, stage stats, and final info assembly. `objectives.py` already defines objective concepts, while setup validation orchestration still lives in the pipeline.

**Rationale:** This is the highest-leverage split because it attacks the main production risk: the workflow is unified in concept but over-centralized in implementation. It also preserves the successful architecture: setup geometry calibration stays inside alignment, but the pipeline stops owning the details of every objective and optimizer branch.

**Downsides:** Medium-to-high coordination cost. The first extraction must be conservative because stage stats, checkpoint payloads, observer actions, and resume behavior are compatibility-sensitive.

**Confidence:** 90%

**Complexity:** High

**Status:** Unexplored

### 2. Split `losses.py` Around A Stable Loss Capability Facade

**Description:** Keep `tomojax.align.losses` as the import-compatible facade, but split implementation into concern-owned private modules: specs/parsing, registry/descriptors, target precompute state, pure per-view kernels, adapter builders, and Gauss-Newton weighting. The public contract should remain `AlignmentLossSpec` and `LossAdapter`; internals should answer which losses support which capabilities.

**Warrant:** `direct:` `losses.py` mixes spec dataclasses, aliases, parsing, target-derived state, Otsu/KDE/Chamfer helpers, differentiable kernels, adapter construction, schedule parsing, and GN support. Institutional learning specifically says setup discovery should continue using the existing alignment loss machinery, normally `l2_otsu`, through `AlignmentLossSpec` and `LossAdapter`.

**Rationale:** This is a clean production-code target with lower workflow blast radius than `pipeline.py`. It reduces future AI-generated debt because adding a loss would have obvious homes: spec, precompute, kernel, builder, capability tests. It also protects setup validation-LM from accidental private loss shortcuts.

**Downsides:** Needs careful parity tests. Loss behavior is numerically sensitive, and a mechanical move can still alter device placement, mask slicing, or GN weights if done casually.

**Confidence:** 88%

**Complexity:** Medium

**Status:** Unexplored

### 3. Promote Provenance, Memory, And Result Contracts To First-Class Internals

**Description:** Extract production contracts from scattered dictionaries into typed internal builders: a memory/execution plan for batching, checkpointing, gather dtype, fold chunking, and validation streaming; a provenance/stat builder for objective kind, optimizer kind, loss kind, gauge policy, fold policy, and calibration state; and typed result records that serialize to the existing dict shape at the boundary.

**Warrant:** `direct:` `pipeline.py` repeatedly assembles `outer_stats`, optimizer metadata, reconstruction info, gauge stats, objective provenance, setup geometry summaries, and final `AlignInfo`/`AlignMultiresInfo` payloads. The documented production lessons call out memory behavior and manifest provenance as core invariants.

**Rationale:** Production readiness depends on debuggability and reproducibility. This move makes behavior visible without requiring the heavy reconstruction suite for every refactor. It also aligns with the current desloppify finding that result `TypedDict`s and actual payload keys have drifted.

**Downsides:** Can become bureaucracy if it overgeneralizes. The first version should be small and compatibility-driven, not an event framework.

**Confidence:** 86%

**Complexity:** Medium

**Status:** Unexplored

### 4. Add A Fast Alignment Contracts Test Pack Before Broad Moves

**Description:** Add a CPU-friendly test pack around public imports, `AlignConfig` defaults and aliases, loss spec canonicalization, schedule resolution, info dict keys, checkpoint/resume payload shape, observer action adaptation, and memory/provenance flags. Use tiny arrays and monkeypatching where useful rather than full reconstruction quality tests.

**Warrant:** `external:` Characterization tests are the standard safety net for behavior-preserving refactors. The repo’s testing guidance favors CPU-friendly owned-surface tests, and the user explicitly wants to avoid the heavy suite for now.

**Rationale:** This is the safest enabler for breaking up `pipeline.py` and `losses.py`. It lets each extraction preserve observable contracts while deferring manual heavy validation until the end.

**Downsides:** Tests alone do not improve architecture. They should be written against contracts we actually intend to preserve, not against incidental local variable behavior.

**Confidence:** 84%

**Complexity:** Low to Medium

**Status:** Unexplored

### 5. Introduce A Private Alignment Run Spec Or Context

**Description:** Extract the resolved execution-ready inputs from `AlignConfig` and setup preparation into a private `AlignmentRunSpec`, `AlignmentExecutionPlan`, or run context. It should hold resolved schedule, active pose/setup DOFs, bounds, gauge policy, batching knobs, loss adapter/spec, objective mode, resume state, and provenance seed data before runtime mutation starts.

**Warrant:** `direct:` `AlignConfig` is large and cross-cutting, and `_prepare_align_setup` already returns an explicit setup state. `pipeline.py` then threads many resolved values through nested closures and helper calls.

**Rationale:** This separates user intent, resolved execution policy, and mutable runtime state. That is especially important for JAX code, where static metadata, arrays, and side-effecting host concerns should not be blurred.

**Downsides:** The context must stay private and small. A broad mutable context object could hide coupling instead of reducing it.

**Confidence:** 80%

**Complexity:** Medium

**Status:** Unexplored

### 6. Extract Pure JAX Kernel And Differentiation Boundaries After The Workflow Contracts Settle

**Description:** Once stage/objective boundaries are clearer, extract pure kernels for projection scoring, smoothness gradients, candidate scoring, validation residual/JVP accumulation, gauge/state transitions, and loss evaluation. Each production objective path should declare its differentiation mode: fixed volume, stopped reconstruction, validation residual JVP, unrolled reference, or no differentiation.

**Warrant:** `external:` JAX’s core design favors pure functions and explicit state, while the architecture note says production setup discovery uses stopped train-fold reconstruction plus streamed validation LM, not memory-heavy unrolled bilevel gradients.

**Rationale:** This protects the most expensive production invariant: do not accidentally reintroduce huge autodiff tapes or materialized residual/Jacobian stacks. It also makes kernels separately benchmarkable and easier to inspect.

**Downsides:** It should not be the first move. Extracting kernels before objective/stage contracts are stable risks creating many thin wrappers around still-unclear ownership.

**Confidence:** 76%

**Complexity:** High

**Status:** Unexplored

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | Immediate `align/` package split | Premature. The repo already has leaf modules; the missing piece is internal contracts, not more directory structure. |
| 2 | Standalone compiler-style `AlignmentProgram` / pass runner | Useful analogy, but too heavy as the next step. It risks introducing architecture before the concrete stage/objective contracts are proven. |
| 3 | Scientific workflow node engine | Over-abstracted for an in-process library refactor. It would add framework surface before simple typed inputs/outputs have been extracted. |
| 4 | Full bounded-context/import-lint system | Potentially useful later, but too much process before the code has stable private boundaries. |
| 5 | AI-editable module headers as primary work | Helpful supplement, but not a primary production-readiness move. Add headers when extracting modules. |
| 6 | Refactor map document as primary work | Useful as support, but code contracts and tests should lead. |
| 7 | Config normalization split as the first move | Worth doing, but lower leverage than stage/objective and loss boundaries. It can ride along with the run-spec extraction. |
| 8 | Immutable loss precompute payloads immediately | Directionally good, but higher risk than a mechanical facade/registry split. Do after parity tests exist. |
| 9 | Typed run ledger/event system | The idea is strong if constrained, but a full event framework would be too broad. Keep the first version as result/provenance builders. |
| 10 | Public internal interfaces | Rejected as a framing. Keep new contracts private until at least two real consumers need them. |

## Recommendation

Yes: focus next on breaking up the production alignment stack, but do it as an in-place modular-monolith refactor rather than a package split. The first implementation batch should be:

1. Add fast alignment contract tests for current public/result/loss/schedule/provenance behavior.
2. Split `losses.py` behind a compatibility facade because it is bounded, high-signal, and less entangled with workflow execution.
3. Extract provenance/result/memory builders from `pipeline.py` so stage execution has stable payload contracts.
4. Then extract a private stage engine/objective boundary from `pipeline.py`.

The script should stay out of scope for now. It is large, but it is not the production API surface driving alignment correctness and extensibility.
