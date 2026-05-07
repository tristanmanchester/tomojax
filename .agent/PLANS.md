# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 2 parallel laminography geometry vertical slice
- Goal: carry nominal parallel-laminography acquisition metadata through
  `GeometryState`, sidecar IO, and the `core_trilinear_ray` adapter so the same
  core projector can model tilted rotation-axis laminography without adding
  projector selectors or toy approximations.

### Scope

- In scope:
  - Add typed acquisition metadata to `GeometryState` for parallel tomography
    versus parallel laminography, including nominal laminography tilt and tilt
    axis.
  - Serialize/read the acquisition metadata in v2 geometry sidecars.
  - Build the v2-to-core nominal axis from acquisition metadata, then apply the
    already-supported setup axis x/y corrections.
  - Preserve existing defaults for parallel tomography.
  - Add focused adapter and sidecar readback tests.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Solving laminography tilt as an active setup parameter, automatic weak-DOF
    activation, benchmark reruns, object drift, or detector-boundary semantics.
- Deep module owners: `tomojax.geometry` for typed state/serialization,
  `tomojax.forward` for core adapter semantics, and `tomojax.datasets` for
  synthetic sidecar population.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add acquisition metadata to `GeometryState`.
- [x] Serialize and load acquisition metadata in v2 sidecars.
- [x] Wire laminography nominal axis into the core adapter.
- [x] Populate laminography metadata during synthetic sidecar generation.
- [x] Add focused adapter/sidecar tests.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_forward_reference.py::test_core_projection_geometry_matches_core_laminography_pose_convention
  tests/test_geometry_serialization.py::test_geometry_json_and_pose_csv_round_trip_contract_artifacts
  tests/test_synthetic_datasets.py::test_load_synthetic_dataset_sidecars_reads_manifest_index
  -q` passed: 3 tests in 3.02 seconds.
- `uv run ruff check ...` passed for touched Python source and tests.
- `uv run basedpyright ...` passed with 0 errors, 0 warnings, and 0 notes for
  touched source and tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- Laminography is represented as a different nominal rotation axis for the same
  parallel-ray detector/beam model; setup axis rotations remain correction DOFs
  applied on top of that nominal axis.

### Completed Previous Slices

- [x] Detector roll supported and committed: `2be6a99`.
- [x] Axis tilt supported and committed with GPU diagnostic pause:
  `ac347d2`.
- [x] Alpha/beta pose supported and committed: `aea525d`.
- [x] Supported geometry update DOFs exposed in `align-auto`: `19dd503`.
- [x] Theta-scale opt-in setup updates committed: `be3d059`.

### Risks

- Risk: laminography tilt convention must match existing core
  `LaminographyGeometry`.
- Mitigation: use the same tilt-about-x/z axis convention and compare adapter
  poses against the core laminography geometry in focused tests.
