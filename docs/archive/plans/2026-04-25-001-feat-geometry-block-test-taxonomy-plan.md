---
title: "feat: Add geometry-block test taxonomy"
type: feat
status: active
date: 2026-04-25
origin: docs/brainstorms/geometry-calibration-solver-requirements.md
---

# Geometry-Block Test Taxonomy Plan

## Overview

Build a proper test and evidence suite for the new geometry-block calibration system.

The immediate correction is that geometry calibration demos must use TomoJAX's established
phantom generation path, not an ad hoc Shepp-Logan phantom. The historical
`doc-normal-scan-motion-128` run used a shared `random_shapes/lamino_disk` phantom with seed
`20260458`, 128 views, 128^3 volumes, and multires levels `8 4 2 1`. The new tests should follow
that discipline while validating the new implementation decision: calibration is staged inside
`align_multires` through reusable geometry parameter blocks.

This plan separates default CPU-friendly tests from optional GPU documentation runs. CI should prove
contracts and representative recovery on small problems. Laptop runs should generate rich before/after
images at 128^3 without becoming default test burden.

## Problem Frame

The current ad hoc before/after image generator is not a trustworthy test of the new system because:

- It introduced Shepp-Logan phantoms that do not match the project's documentation runs or phantom
  conventions.
- It used a short, low-iteration profile that does not match the normal 8/4/2/1 alignment pipeline.
- It was written as a separate demonstration path instead of a disciplined taxonomy-style evidence
  generator.
- It risks obscuring whether we are testing supplied corrections, estimated geometry blocks, or pose
  alignment.

The test suite needs to answer four concrete questions:

1. Does `align_multires` recover hidden instrument/lab geometry parameters through geometry blocks?
2. Does it do so without pose DOFs standing in for instrument errors?
3. Are supplied corrections, estimated corrections, frozen gauges, and final calibrated geometry
   recorded distinctly?
4. Can we produce documentation-quality before/after images using the same phantom family and run
   style as the historical taxonomy runs?

## Requirement Trace

This plan implements or validates these requirements from
`docs/brainstorms/geometry-calibration-solver-requirements.md`:

- R1: Separate instrument/lab geometry calibration from per-view pose alignment.
- R5: Label known centre correction as supplied metadata, not solver recovery.
- R12: Support detector/ray-grid centre offset as a `tomojax-align` geometry block.
- R14: Synthetic tests must prove hidden detector-centre recovery, not known correction replay.
- R15: Geometry blocks must use the differentiable projector and multires alignment machinery; grid
  search is diagnostic only.
- R16: Manifests must state that `det_u_px` is the canonical detector/ray-grid centre representation
  under the chosen gauge, not proof of detector/axis intercept separability.
- R18: Axis direction calibration belongs to instrument geometry.
- R19: Laminography tilt calibration belongs to instrument geometry.
- R20: Axis direction should be staged after detector centre by default, with optional detector-centre
  refinement after axis calibration.
- R31: Real-data acceptance needs visual/artifact evidence beyond scalar sharpness.
- R32: Manifests must distinguish supplied, estimated, frozen, gauges, objectives, and final
  calibrated geometry.
- AE1: A hidden `det_u_px` offset recovered through `tomojax-align --optimise-geometry det_u_px`
  across multires levels is the core acceptance example.

## Scope

In scope:

- CPU-friendly tests for geometry-block recovery and artifact contracts.
- A canonical scenario generator for 128^3 before/after images using `lamino_disk`.
- Scenarios covering detector centre, detector roll, axis direction, laminography tilt, and staged
  combinations.
- Manifest and summary schemas that make parameter provenance explicit.
- Optional laptop/GPU run commands for documentation-quality artifacts.

Out of scope:

- Reintroducing standalone calibration solvers outside `align_multires`.
- Using Shepp-Logan phantoms for geometry-block documentation tests.
- Making 128^3 GPU image generation part of default CI.
- Claiming physical detector/axis intercept separability from scalar `det_u_px`; the scalar remains a
  gauge-fixed detector/ray-grid centre representation.
- Solving real laminography quality in this phase. Real data can be used later as evidence, but this
  test plan is for the new synthetic geometry-block system.

## Current Evidence

The historical documentation run in `runs/doc-normal-scan-motion-128/manifest.json` recorded:

- `phantom.kind`: `random_shapes/lamino_disk`
- `phantom.seed`: `20260458`
- shared phantom across cases
- 128^3 volume
- 128 views
- multires levels `8 4 2 1`
- `views_per_batch=1`
- case artifacts including manifests, metrics, parameter traces, workflow comparisons, and quicklook
  images

The implementation learning in
`docs/solutions/architecture-patterns/reuse-align-multires-for-geometry-calibration-2026-04-25.md`
established that geometry calibration must be a thin layer inside `align_multires`, with reusable
geometry blocks and checkpoint metadata, not a parallel optimizer stack.

The current ad hoc generator at `scripts/generate_alignment_before_after_128.py` should be treated as
non-canonical until it is rewritten or replaced. Its Shepp-Logan path is specifically what this plan
prevents from becoming documentation evidence.

## Technical Decisions

### Use the Established Phantom Path

All canonical synthetic geometry-block evidence should use `src/tomojax/data/phantoms.py`:

- Use `lamino_disk()` for laminography-style documentation runs.
- Keep `random_cubes_spheres()` as the underlying random-shape primitive.
- Use seed `20260458` as the default documentation seed unless a scenario explicitly needs another
  seed.
- Record the phantom kind and seed in every manifest.
- Do not import or blend `shepp_logan_3d` in the canonical generator.

Rationale: the old taxonomy run is already interpretable and comparable. Reusing its phantom family
keeps before/after images meaningful and avoids inventing a new visual test target.

### Test the Product Path, Not a Side Path

All recovery scenarios should call `align_multires` with geometry blocks. The tests can use helper
functions, but the calibrated behavior must be the same path exposed by `tomojax-align
--optimise-geometry`.

Rationale: the first implementation mistake was creating parallel calibration machinery. The tests
should lock in the DRY architecture by making `align_multires` the only calibration route.

### Split Small CI Tests From Heavy Evidence Runs

Default tests should use small volumes and view counts so they run on CPU. The 128^3, 128-view,
8/4/2/1 evidence run should be an explicit laptop/GPU command.

Rationale: large visual evidence is necessary for confidence and docs, but default tests must stay
fast, deterministic, and cheap.

### Treat Supplied Corrections As Controls

Known correction replay can appear as a control case, but must be labeled as supplied/frozen metadata,
not as estimated geometry.

Rationale: this avoids repeating the earlier centre-of-rotation ambiguity where applying a known
offset looked like solver success.

### Prefer Scenario Taxonomy Over One-Off Panels

Before/after images should come from a scenario taxonomy with consistent manifests, not from a loose
image script.

Rationale: if a scenario fails, we should know which block, gauge, objective, levels, and recovered
parameters were involved.

## Implementation Units

### Unit 1: Canonical Geometry-Block Scenario Specification

Goal: define the scenario set and shared configuration used by both CPU tests and optional GPU image
generation.

Files:

- `scripts/generate_alignment_before_after_128.py` or a replacement script such as
  `scripts/generate_geometry_block_taxonomy_128.py`
- `tests/test_geometry_block_taxonomy_generator.py`
- `src/tomojax/data/phantoms.py` only if a small helper is needed; avoid changing phantom semantics

Work:

- Remove Shepp-Logan from the canonical path.
- Define a typed scenario spec with:
  - scenario id
  - scenario title
  - hidden geometry perturbation
  - supplied corrections
  - active geometry blocks
  - frozen gauges
  - levels
  - expected recovered parameter names
  - expected artifact names
- Use `lamino_disk(size=..., seed=20260458, ...)` by default.
- Make small-test and 128^3 profiles differ only by configuration, not by code path.
- Add a dry-run or metadata-only mode so tests can validate scenarios without running a full
  reconstruction.

Acceptance:

- The scenario list can be loaded without running JAX-heavy alignment.
- Every scenario records phantom kind and seed.
- No canonical generator path imports or selects Shepp-Logan.
- The default docs profile is 128^3, 128 views, `levels=(8, 4, 2, 1)`, and `views_per_batch=1`.

### Unit 2: CPU-Friendly Geometry Recovery Regression Tests

Goal: prove the new geometry blocks recover hidden errors through `align_multires` on small synthetic
problems.

Files:

- `tests/test_align_quick.py`
- `tests/test_geometry_block_workflows.py` if the existing file becomes too broad
- `tests/README.md` only if test ownership guidance needs a small update

Work:

- Keep or refine the existing detector-centre recovery test:
  - hidden `det_u_px`
  - no pose DOFs
  - active geometry block `det_u_px`
  - recovered value moves toward the hidden offset
  - checkpoint/metadata records it as estimated
- Add detector-roll recovery:
  - hidden detector roll
  - active roll block
  - no pose DOFs
  - recovered roll reduces loss and moves in the correct direction
- Add axis-direction recovery:
  - hidden small axis direction perturbation
  - active axis-direction block
  - staged after detector centre where relevant
- Add laminography-tilt recovery:
  - hidden tilt perturbation
  - active tilt block
  - verify it changes the lab/instrument geometry, not per-view pose arrays
- Add a staged combined test:
  - stage 1: `det_u_px`
  - stage 2: roll or axis direction
  - optional stage 3: refine `det_u_px`
  - run through multires levels, scaled down for CPU

Acceptance:

- Tests are deterministic on CPU.
- Tests do not require the laptop GPU.
- Tests do not assert unrealistic exact recovery on tiny problems; they assert directionality,
  meaningful loss reduction, parameter provenance, and absence of pose-DOF substitution.
- Failure messages include observed estimates and loss deltas.

### Unit 3: Manifest and Artifact Contract Tests

Goal: make artifacts auditable before running expensive image generation.

Files:

- `tests/test_geometry_block_taxonomy_generator.py`
- `scripts/generate_geometry_block_taxonomy_128.py` or the revised generator script

Work:

- Validate top-level run manifest fields:
  - phantom kind and seed
  - volume size
  - view count
  - levels
  - geometry block order
  - active/frozen/supplied parameters
  - gauge notes
  - command/config used
- Validate per-scenario manifest fields:
  - hidden truth
  - initial geometry
  - supplied corrections
  - estimated corrections
  - final calibrated geometry
  - objective/loss summaries
  - artifact paths
- Validate summary CSV columns:
  - scenario id/name
  - block sequence
  - hidden values
  - recovered values
  - initial/final loss
  - status
  - strongest preview artifact
- Ensure supplied known corrections are never reported as estimated values.
- Ensure `det_u_px` is described as detector/ray-grid centre under the gauge, not as a universal
  physical centre-of-rotation proof.

Acceptance:

- A metadata-only test can validate manifests and summary schema without generating 128^3 volumes.
- A small smoke run writes real artifacts into a temporary directory and passes schema validation.
- Existing run directories under `runs/` are not required for tests.

### Unit 4: Optional 128^3 Laptop Evidence Run

Goal: generate documentation-quality before/after images for the new geometry-block system.

Files:

- `scripts/generate_geometry_block_taxonomy_128.py` or revised
  `scripts/generate_alignment_before_after_128.py`
- `docs/internal/geometry-block-taxonomy-runbook.md` if a runbook is useful

Run profile:

- volume: 128^3
- views: 128
- phantom: `random_shapes/lamino_disk`
- seed: `20260458`
- levels: `8 4 2 1`
- `views_per_batch=1`
- save per-level and final previews
- save `before_after_panel.png` per scenario
- save a master contact sheet
- save `summary.csv`, `status.json`, and manifests

Initial scenarios:

- `det_u_px_hidden_m004`: hidden detector/ray-grid centre offset near the real-data correction scale.
- `det_u_px_hidden_p004`: opposite sign detector centre offset to catch sign conventions.
- `detector_roll_small`: detector roll error only.
- `axis_direction_yaw`: rotation axis not perfectly vertical, side-to-side component.
- `axis_direction_pitch`: rotation axis tipped forward/backward.
- `lamino_tilt_error`: laminography tilt estimate error.
- `det_u_then_roll`: staged detector centre followed by detector roll.
- `det_u_then_axis_then_det_u_refine`: staged centre, axis, centre refinement.
- `known_correction_control`: supplied known detector-centre correction, clearly labeled as supplied.

Acceptance:

- The run completes without out-of-memory by using the normal multires alignment path.
- Every scenario has a before/after panel and per-scenario manifest.
- The master contact sheet is visually useful and uses robust but honest display scaling.
- Parameter tables show hidden, initial, estimated, and final geometry.
- Failed scenarios remain visible in `summary.csv` with logs and partial artifacts.

### Unit 5: Cleanup and Documentation

Goal: remove misleading code paths and document how to run the test suite.

Files:

- `scripts/generate_alignment_before_after_128.py`
- `scripts/generate_geometry_block_taxonomy_128.py` if added
- `docs/brainstorms/geometry-calibration-solver-requirements.md`
- `docs/solutions/architecture-patterns/reuse-align-multires-for-geometry-calibration-2026-04-25.md`
- `tests/README.md`

Work:

- Either replace the current generator or rewrite it so its name and behavior match the canonical
  taxonomy.
- Remove Shepp-Logan imports from the canonical geometry-block evidence path.
- Document the three validation tiers:
  - default CPU tests
  - optional small GPU smoke
  - optional 128^3 laptop documentation run
- Add a short note to the requirements doc that geometry-block tests use the same `align_multires`
  staged block machinery as production alignment.
- Add a short note to the architecture learning if implementation reveals any additional DRY rules.

Acceptance:

- A maintainer can tell which command is for CI and which command is for laptop evidence.
- The repository no longer contains an endorsed geometry-block before/after generator that silently
  uses Shepp-Logan.
- Documentation does not present known centre correction replay as calibration recovery.

## Test Matrix

| Layer | Purpose | Example command/profile | Required by default |
| --- | --- | --- | --- |
| Unit | scenario spec and manifest schema | `uv run pytest -q tests/test_geometry_block_taxonomy_generator.py` | yes |
| Workflow | small CPU geometry recovery | `JAX_PLATFORM_NAME=cpu uv run pytest -q tests/test_align_quick.py tests/test_geometry_block_workflows.py` | yes |
| Full tests | repository regression | `uv run pytest -q tests` | yes before merge |
| GPU smoke | catches accelerator/JAX shape issues | small generator profile with 32^3 or 64^3 | no |
| Docs evidence | rich before/after images | 128^3, 128 views, levels `8 4 2 1` on laptop | no |

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Tiny CPU tests cannot exactly recover all coupled geometry parameters | Assert directionality, loss reduction, provenance, and gauge behavior; leave exact visual quality to GPU evidence runs. |
| Geometry parameters compensate for each other | Use staged tests and per-stage manifests; include single-error and combined-error scenarios. |
| Before/after images look better due to display scaling rather than alignment | Store numeric losses and parameter estimates beside every image; keep scaling logic consistent and documented. |
| Heavy 128^3 runs become too slow for routine development | Keep them optional and scriptable; default CI remains small. |
| The generator becomes a second product pipeline | Keep calibration execution inside `align_multires`; the script only builds scenarios, calls the normal path, and renders artifacts. |
| Centre-of-rotation terminology becomes misleading again | Use `det_u_px` and detector/ray-grid centre language in manifests; reserve "known correction" for supplied metadata controls. |

## Validation Sequence

1. Implement Unit 1 and run the scenario/manifest tests.
2. Implement Unit 2 and run targeted CPU workflow tests.
3. Implement Unit 3 and run a temporary-directory smoke test for artifact contracts.
4. Run full local tests with `uv run pytest -q tests`.
5. Push/sync cleanly to the laptop branch or worktree.
6. Run the optional 128^3 evidence profile on the laptop.
7. Inspect the master contact sheet and strongest/weakest scenarios before using any images in docs.

## Success Criteria

This phase is done when:

- Canonical geometry-block tests use `lamino_disk` or `random_cubes_spheres`, not Shepp-Logan.
- Hidden detector-centre recovery is tested through `align_multires`.
- Roll, axis direction, and tilt blocks have CPU-friendly coverage.
- Combined staged calibration is tested through multires levels.
- Manifests separate supplied, estimated, frozen, gauge, and final geometry fields.
- A 128^3 laptop run can generate before/after panels using levels `8 4 2 1`.
- The codebase has one reusable geometry-block execution path, not a pile of special-case solvers.

## Deferred Questions

- Exact numeric tolerances for roll, axis, and tilt recovery should be set after a first small CPU
  implementation pass. They should be strict enough to catch sign/convention regressions but not so
  strict that tiny phantoms become flaky.
- The final image layout for docs can be decided after the 128^3 evidence run produces real panels.
- Real laminography validation should come after synthetic geometry-block confidence is restored.
