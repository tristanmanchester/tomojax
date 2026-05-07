# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 calibrated-grid backend provenance slice
- Goal: carry benchmark manifest detector-grid policy into sidecar readback and
  benchmark backend provenance so `backend_policy:
  calibrated_grid_fallback_explicit` has concrete evidence on calibrated-grid
  synthetic datasets.

### Scope

- In scope:
  - Parse optional `detector_grid` from the benchmark manifest.
  - Write/read `detector_grid` through generated synthetic dataset manifests
    and align-auto sidecar readback.
  - Record explicit backend fallback provenance when a sidecar requests a
    calibrated noncanonical detector grid.
  - Add focused sidecar/backend policy tests.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Changing projector kernels, adding new detector-grid transforms, or
    rerunning benchmarks.
- Deep module owners: `tomojax.datasets` for manifest metadata and
  `tomojax.align`/`tomojax.cli` for benchmark provenance payloads.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Parse and write detector-grid manifest metadata.
- [x] Include detector-grid metadata in align-auto sidecar readback.
- [x] Emit calibrated-grid backend fallback provenance.
- [x] Add focused sidecar/backend policy tests.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_synthetic_datasets.py::test_load_synthetic_dataset_sidecars_reads_manifest_index
  tests/test_alternating_benchmark_criteria.py -q` passed: 8 tests in
  2.66 seconds.
- `uv run ruff check src/tomojax/datasets/_specs.py
  src/tomojax/datasets/_writer.py src/tomojax/cli/align_auto.py
  src/tomojax/align/_alternating_artifacts.py tests/test_synthetic_datasets.py
  tests/test_alternating_benchmark_criteria.py` passed.
- `uv run basedpyright` on the same focused source/test set passed with
  0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- Backend policy criteria must use explicit sidecar/provenance metadata, not
  infer fallback from benchmark names.

### Completed Previous Slices

- [x] Detector roll supported and committed: `2be6a99`.
- [x] Axis tilt supported and committed with GPU diagnostic pause:
  `ac347d2`.
- [x] Alpha/beta pose supported and committed: `aea525d`.
- [x] Supported geometry update DOFs exposed in `align-auto`: `19dd503`.
- [x] Theta-scale opt-in setup updates committed: `be3d059`.
- [x] Parallel laminography acquisition metadata committed: `7aa086c`.
- [x] det_v observability gating evidence committed: `7c1e0fe`.
- [x] Synthetic unsupported-term classification committed: `28e336f`.
- [x] Benchmark criterion aliases committed: `fe83427`.
- [x] Laminography solver residuals committed: `7002d42`.
- [x] Recovered det_v policy criterion committed: `f6fe3c4`.
- [x] Backend policy criterion evaluation committed: `b040829`.

### Risks

- Risk: provenance can claim a fallback without any manifest trigger.
- Mitigation: emit fallback rows only when sidecar readback explicitly reports
  `detector_grid="calibrated_noncanonical"`.
