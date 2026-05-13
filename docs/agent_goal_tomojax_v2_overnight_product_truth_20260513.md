# TomoJAX v2 Overnight Product Truth Goal - 2026-05-13

## Objective

Prepare TomoJAX v2 for a credible publication-facing MVP by proving what a user can
actually run, producing the artifacts they need to judge output quality, and fixing
the most important failures found in the original `128^3` plan rather than hiding
them behind oracle or diagnostic labels.

This is not a naming/report-only pass. If a supported production workflow fails, fix
it. If a workflow is still research, keep it out of production claims and document the
exact blocker with artifacts.

Use `vivobook-ts:/home/tristan/projects/tomojax-v2-goal` for CUDA-heavy runs. Keep
all synthetic volume/detector sizes at `128^3` or smaller unless explicitly noted.
Commit incrementally after each verified slice.

## Non-negotiable scope

Do all four workstreams below.

1. Public MVP polish
2. Honest original `128^3` synthetic gates
3. Real laminography production workflow cleanup
4. Stopped-alignment blocker advancement

Do not add broad new research features unless they are needed to make one of these
workstreams pass. Do not promote benchmark-specific tricks to the public path.

## Product rules

- Public user commands must be clean and boring:
  - `tomojax inspect`
  - `tomojax validate`
  - `tomojax ingest`
  - `tomojax preprocess`
  - `tomojax recon`
  - `tomojax align`
  - `tomojax simulate`
- Developer diagnostics belong under `tomojax dev`.
- User-facing reports must not say `mvp`, `smoke`, `v1 parity audit`, `fixed truth`,
  or `oracle` unless the report section is explicitly a developer diagnostic.
- Production status labels must distinguish:
  - `production_pass`
  - `oracle_geometry_pass`
  - `diagnostic_only`
  - `research_blocker`
  - `unsupported_model`
  - `failed`
- A green production claim must come with:
  - command
  - git commit
  - input dataset provenance
  - backend/device
  - peak memory where sampled
  - runtime
  - final report path
  - user-inspectable PNG artifacts

## Workstream 1 - Public MVP polish

Goal: make the cleaned public CLI feel usable and publication-ready.

Deliverables:

- Verify these commands on small deterministic fixtures:
  - `tomojax inspect`
  - `tomojax validate`
  - `tomojax ingest`
  - `tomojax preprocess`
  - `tomojax recon`
  - `tomojax align --mode cor`
  - `tomojax simulate`
- Ensure every successful command writes or reports clear artifact paths.
- Ensure common user mistakes fail with actionable messages, not raw internal tracebacks
  where practical.
- Add or update a concise first-run doc:
  - one TIFF-stack ingest path
  - one NX/HDF5/NXtomography path
  - one synthetic path
  - one real-laminography staged path
- Add a publication-facing support matrix that says what is supported, what is
  diagnostic, and what is research.
- Ensure the README/quickstart do not send users to `tomojax dev` for normal
  reconstruction/alignment.

Acceptance:

- Focused CLI tests pass.
- `just check` passes before the final goal is marked complete.
- The support matrix is honest about limitations.

## Workstream 2 - Honest original `128^3` synthetic gates

Goal: run the original five scenarios from
`docs/tomojax-v2/05_synthetic_128_benchmark_suite.md` honestly and fix supported
failures instead of only reporting them.

The five cases are:

1. `synth128_setup_global_tomo`
2. `synth128_pose_random_extreme`
3. `synth128_lamino_axis_roll_pose`
4. `synth128_thermal_object_drift`
5. `synth128_combined_nuisance_jumps`

Required actions:

- Re-run or validate existing artifacts for all five at `128^3` max.
- For cases 1 and 2, preserve green status only if the evidence still really supports
  the stated claim. If the pass is fixed-truth/oracle-only, label it as such.
- For cases 3-5, debug and fix what is feasible within the existing architecture:
  - laminography axis/roll/theta/det-v observability
  - object-motion suspicion/reporting versus actual object-frame recovery
  - nuisance/jump residual handling and status classification
- If a failure is due to an unsupported model, make the output status explicit and
  ensure the public support matrix does not claim support.
- Do not mark a red case green by excluding hard metrics or weakening criteria.
- Produce one consolidated report:
  `docs/benchmark_runs/2026-05-13-synthetic128-product-truth.md`.

Required artifacts per case:

- manifest JSON
- metric/status JSON
- geometry trace where available
- reconstruction or preview PNG slices where available
- command used
- backend/device/memory/runtime evidence

Acceptance:

- Cases 1-2 are either `production_pass` or explicitly `oracle_geometry_pass` with no
  ambiguous wording.
- Cases 3-5 are either improved/fixed or classified with a precise blocker and artifact
  evidence.
- No synthetic run exceeds `128^3` volume size.

## Workstream 3 - Real laminography production workflow cleanup

Goal: turn the real laminography path from a development-era parity runner into a
clean, user-facing staged workflow while preserving the evidence from the scan that
worked.

Reference run:

`runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525`

Required actions:

- Keep a production-facing staged real-laminography entrypoint:
  `scripts/real_laminography/run_real_lamino_staged.py`
- If a public `tomojax align` route can reasonably call this workflow for this data
  shape/profile, wire or document it cleanly. If not, document why the script remains
  the supported real-laminography path for now.
- Rename user-facing report labels away from `mvp`, `smoke`, and `v1 parity`.
- Produce user-inspectable artifacts:
  - contact sheet of stages
  - final reconstruction slices
  - COR-only comparator slices
  - residual trace
  - geometry trace
  - summary markdown
- Run the binned development profile first, then the full-size profile if the binned
  output is sane.
- Compare final staged output against the COR-only comparator and the retained reference
  report.
- Produce:
  `docs/benchmark_runs/2026-05-13-real-lamino-product-workflow.md`.

Acceptance:

- The report states whether v2 matches, exceeds, or falls short of the retained real
  reference case.
- The output can be judged visually from PNGs without digging through NumPy arrays.
- Any retained old/native runner logic is explicitly documented as allowed retained
  production implementation or developer-only reference logic.

## Workstream 4 - Stopped-alignment blocker advancement

Goal: make meaningful progress on the core truth-free detector-centre/COR gap without
turning the product into a COR-grid-search toolbox.

Required actions:

- Define one honest stopped detector-centre gate:
  - rich phantom
  - `128^3` max
  - no true volume for production decision
  - det_u/COR active
  - pose frozen initially
  - no weak-view exclusion as pass criterion
- Run or reuse the latest diagnostic artifacts and update the report with the current
  state.
- Try at least one bounded fix that is compatible with TomoJAX's differentiable
  optimisation identity, for example:
  - scout/alignment-volume support or low-frequency anchor
  - real multires preview carry if not already production-equivalent
  - loss/mask alignment between x-step and geometry step
  - v2-native differentiable detector-centre bootstrap, not sinogram/COR sweep
- If it still fails, document the blocker precisely and keep it out of the production
  support matrix.
- Produce:
  `docs/benchmark_runs/2026-05-13-stopped-alignment-product-blocker.md`.

Acceptance:

- The report says in plain terms whether truth-free stopped detector-centre recovery is
  production-supported today.
- Any improvement has metric evidence; any failure has artifact evidence.
- No sinogram, cross-correlation, entropy/sharpness sweep, or COR grid search is added
  to the default product path.

## Remote execution notes

Before GPU runs on `vivobook-ts`:

1. Sync the current Mac working tree to the laptop without deleting `.git`, `.venv`,
   `runs`, `.artifacts`, or untracked oracle notes.
2. Use the laptop checkout:
   `/home/tristan/projects/tomojax-v2-goal`
3. Use CUDA with explicit NVIDIA wheel library paths if needed:

```bash
NVLIB=$(find "$PWD/.venv/lib/python3.12/site-packages/nvidia" -type d -name lib | paste -sd: -)
env LD_LIBRARY_PATH="$NVLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  UV_CACHE_DIR=.uv-cache \
  JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ...
```

## Final completion audit

Before marking the goal complete:

- `git status --short` must be clean except intentionally documented external artifacts.
- All new source/docs/tests must be committed.
- `just check` must pass locally or on the laptop, with the location recorded.
- Reports for the four workstreams must exist.
- The support matrix must not overclaim stopped alignment, laminography, object motion,
  nuisance, or Pallas support.
- Artifacts from remote GPU runs must be synced back to the Mac if they are referenced
  by local docs.
