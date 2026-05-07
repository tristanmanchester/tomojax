# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 synthetic manifest unsupported-term classification slice
- Goal: restore truthful synthetic sidecar classification for remaining
  unsupported benchmark manifest terms so generated datasets do not claim
  `all_supported` while object motion or unsupported nuisance/jump policies are
  still not represented by the v2 core geometry/solver path.

### Scope

- In scope:
  - Parse `true_object_motion` from the benchmark manifest.
  - Classify unsupported manifest terms for object motion, sparse pose jumps,
    bad views, partial-FOV invalid edges, and nuisance terms not realised by the
    current sidecar writer.
  - Preserve `supported_only` behaviour.
  - Add focused synthetic sidecar tests for unsupported-term classification.
  - Update docs/logs and commit the slice.
- Out of scope:
  - Implementing object motion, jump detection, bad-view handling, partial-FOV
    masks, or new nuisance fitting.
- Deep module owner: `tomojax.datasets` for synthetic sidecar manifests.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Parse object-motion benchmark manifest data.
- [x] Classify unsupported manifest terms in generated sidecars.
- [x] Add focused synthetic sidecar tests.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_synthetic_datasets.py::test_generate_object_motion_dataset_marks_unsupported_manifest_terms
  tests/test_synthetic_datasets.py::test_generate_combined_nuisance_dataset_marks_unmodelled_terms
  tests/test_synthetic_datasets.py::test_generate_supported_only_setup_global_dataset_removes_unsupported_truth
  -q` passed: 3 tests in 2.80 seconds.
- `uv run ruff check src/tomojax/datasets/_specs.py
  src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py` passed.
- `uv run basedpyright src/tomojax/datasets/_specs.py
  src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py` passed
  with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- Synthetic dataset manifests must distinguish supported geometry terms from
  remaining unsupported manifest terms; a generated sidecar should not report
  `all_supported` unless the current v2 path really evaluates all listed terms.

### Completed Previous Slices

- [x] Detector roll supported and committed: `2be6a99`.
- [x] Axis tilt supported and committed with GPU diagnostic pause:
  `ac347d2`.
- [x] Alpha/beta pose supported and committed: `aea525d`.
- [x] Supported geometry update DOFs exposed in `align-auto`: `19dd503`.
- [x] Theta-scale opt-in setup updates committed: `be3d059`.
- [x] Parallel laminography acquisition metadata committed: `7aa086c`.
- [x] det_v observability gating evidence committed: `7c1e0fe`.

### Risks

- Risk: over-broad classification could mark already-supported DOFs as
  unsupported again.
- Mitigation: classify only manifest terms that are still not modelled by
  `GeometryState`, sidecar projection generation, or the current solver path.
