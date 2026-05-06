# Lightning Alignment Engine

TomoJAX alignment is now alignment-first by default. The default profile is
`lightning`: it spends computation on reaching useful geometry quickly, records
what actually ran, and uses verification metadata to decide whether the result
is acceptable. The explicit reference profile is `tortoise`.

## Profiles

`lightning` is the default profile for CLI and `AlignConfig` runs. It prefers:

- Pallas hot paths where support checks pass
- `huber_tv` inner reconstruction
- `gather_dtype = "auto"`
- smooth pose models before fully independent per-view freedom
- fast reconstruction diagnostics
- performance-only proposal stages followed by gradient-safe refinement

`tortoise` is the slow reference/debug profile. It prefers:

- JAX projectors
- FP32 gather
- `tv` regularisation
- per-view pose parameters
- reference diagnostics
- no performance-only proposal stage by default

Explicit user configuration wins over profile defaults. For example,
`--align-profile lightning --projector-backend jax --pose-model per_view` is a
valid run: the manifest should still record the lightning profile and the
explicit overrides.

## Stage Contract

Alignment schedules are staged. A stage records:

- role: `proposal`, `setup`, `reconstruction`, `refine`, or `verify`
- differentiability requirement: `performance_only`, `gradient_safe`, or
  `not_required`
- quality tier
- requested and actual backend
- fallback reason when the requested fast path could not run
- speed-claim eligibility

Lightning can use non-differentiable proposal scoring when that gets the pose or
setup state into a better basin. A proposal is not final evidence by itself; it
must feed into refinement or verification.

## Reconstruction Quality

Inner alignment reconstruction is not the final image-quality product. It is a
latent estimate used to expose enough structure for geometry solving. Fast tiers
may skip expensive final diagnostics. Verification or final reconstruction tiers
can spend more iterations and compute richer quality metrics.

The final reconstruction lane remains separate. Once geometry is solved, a user
can still run a slower high-quality reconstruction or export geometry to another
reconstructor.

## Pallas And Fallback

Pallas is the preferred lightning engine for supported GPU cases, but support is
checked at Python boundaries. Unsupported geometry, unsupported detector grids,
CPU execution, or gradient-critical surfaces can fall back to JAX unless strict
mode requests failure.

Every workflow result should distinguish:

- requested backend
- actual backend
- differentiability class
- fallback reason
- whether the row is eligible for a speed claim

## Benchmark Meaning

Workflow benchmarks are the headline evidence. Operator microbenchmarks are
regression guards and hypothesis tests.

Useful alignment evidence includes:

- wall time and stage wall time
- time-to-good-geometry where available
- reprojection or validation loss
- geometry/recovery metrics for synthetic cases
- final-quality metrics when final reconstruction is requested
- backend and precision provenance
- memory

Helper-only speedups must be labeled as helper-only. They should not be reported
as TomoJAX alignment progress unless a workflow benchmark shows the improvement
reaches the alignment run.
