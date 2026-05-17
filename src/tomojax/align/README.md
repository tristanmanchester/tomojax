# tomojax.align

## Purpose

`tomojax.align` owns alignment orchestration: alternating reconstruction and
geometry updates, continuation policy, observability reporting, gauge
canonicalisation, and solver traces.

This README defines the alignment boundary in two tiers:

- `tomojax.align` is the production facade. It is deliberately small and should
  be the default import for product callers.
- `tomojax.align.api` is the developer and advanced facade. It is broader so
  the CLI, benchmarks, focused tests, and advanced integrations can keep using
  typed schedule, loss, profile, geometry-state, objective, and solver helpers
  while the implementation is split into deeper owners.

Diagnostic-only helpers may remain directly importable during
productionization, but they are not part of `tomojax.align.api.__all__`; new
diagnostic entrypoints belong in `tomojax.verify`, `tomojax.bench`, or
`tomojax dev ...`, not in the product alignment namespace.

## Production facade

`tomojax.align` exports only:

- `AlignConfig`
- `align`
- `align_multires`

These names are the stable product entrypoints for running single-level or
multi-resolution alignment.

## Developer facade

`tomojax.align.api` intentionally remains broad, but it is not the product
namespace. It re-exports:

- product entrypoints and resume/checkpoint callback types;
- alignment schedules, continuation presets, active-DOF declarations, and gauge
  policies;
- loss specifications, loss schedules, profile policy helpers, and normalization
  helpers used by the CLI;
- geometry-state, pose-state, calibration-state, projection objective, and
  detector-grid helpers needed by benchmarks and artifact generation;
- reference LM/Schur solvers and damping/trust-radius helpers for focused
  solver tests and advanced integrations.

Prefer adding new user-facing product entrypoints to `tomojax.align` only when
they are stable enough for general callers. Prefer adding experimental,
diagnostic, benchmark, or solver-development names to deeper owning modules and
exposing them through `tomojax.align.api` only when existing callers need a
temporary typed facade.

Schur diagnostic report helpers are owned by `tomojax.verify`, and raw
`JointSchurDiagnostics` remains an internal solver type rather than an
alignment facade export.

## Dependencies

Allowed dependencies:

- `tomojax.core`
- `tomojax.geometry`
- `tomojax.motion`
- `tomojax.nuisance`
- `tomojax.forward`
- `tomojax.recon`
- `tomojax.verify`
- `tomojax.backends`
- `tomojax.io`

Forbidden dependencies:

- private implementation files from other deep modules
- generic utility modules
- new compatibility aliases outside the documented facade

## Invariants

- Default geometry optimisation must be gradient-first LM/GN, not grid search.
- Geometry updates must emit artifact/provenance data.
- Product code should import alignment helpers from `tomojax.align.api` or the
  package-root product facade, not nested compatibility aliases.
- `tomojax.align.checkpoint`, `tomojax.align.diagnostics`,
  `tomojax.align.motion_models`, `tomojax.align.params_export`, and
  `tomojax.align.losses` are intentionally not registered compatibility
  modules. Use their owning deep-module paths.
- `solve_pose_only_lm` defaults to `phi_residual_rad`, `dx_px`, and `dz_px`;
  `alpha_rad` and `beta_rad` are supported opt-in pose DOFs for focused stages.
- `solve_setup_only_lm` defaults to `theta_offset_rad`, `det_u_px`, active
  `det_v_px`, and detector roll. Axis rotations are supported opt-in setup
  parameters; theta scale is supported only as an explicit opt-in setup
  parameter until identifiable scale policy activates it automatically.
- `solve_joint_schur_lm` is the first reference Schur setup+pose slice for the
  supported setup and pose DOFs. It has accepted/rejected damping adaptation and
  ratio-based trust-radius adaptation, but is not yet the final trust-region
  engine.
- `solve_joint_schur_lm` can opt into per-view gain/offset variable projection
  so affine acquisition drift is modelled as nuisance rather than geometry.
- Schur diagnostics should be consumed through `tomojax.verify` artifacts or
  internal solver result objects, not through `tomojax.align.api`.

## Tests

- Existing alignment tests cover retained staged behavior.
- `tests/test_alternating_solver_smoke.py` covers the deterministic diagnostic
  artifact run.
- `tests/test_v2_module_skeleton.py` verifies the v2 facade exists and imports.
