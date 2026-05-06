# AGENTS.md

## SIFS Code Search

Use SIFS from the shell for codebase search before broad file reads when you need to find behavior, symbols, related implementations, or relevant files.

- Discover the current CLI contract with `sifs agent-context --json`.
- Search with `sifs search "<query>" --source <project> --limit 10`.
- Narrow by path with `--filter-path <repo-relative-path>` and use `--mode bm25` for exact symbols.
- Inspect results with `sifs get <file_path> <line> --source <project>` and `sifs find-related <file_path> <line> --source <project>`.
- Use `--source <project>` when the agent may not be running from the target checkout. If already running in the target checkout, `--source .` is usually appropriate.

## Project identity

TomoJAX is being reimagined as a fast, differentiable tomography and laminography alignment/reconstruction toolbox.

The goal is not to patch the old staged alignment engine. The goal is to implement the architecture described in:

- docs/tomojax-v2/01_high_level_architecture.md
- docs/tomojax-v2/02_loss_and_optimiser_spec.md
- docs/tomojax-v2/03_repo_layout.md
- docs/tomojax-v2/04_phased_implementation_plan.md
- docs/tomojax-v2/05_synthetic_128_benchmark_suite.md
- docs/tomojax-v2/06_verification_and_artifact_contract.md
- docs/tomojax-v2/07_synthetic_generator_pseudocode.md

If these files are absent, stop and ask for them.

Treat `docs/tomojax-v2/04_phased_implementation_plan.md` as the canonical phased plan. Use `.agent/PLANS.md` only as a living execution log for the active milestone, not as a competing plan.

## Architectural style

Use deep modules.

Each top-level package under src/tomojax is a deep module with a small public API and hidden implementation.

Allowed:
- tomojax.core
- tomojax.geometry
- tomojax.motion
- tomojax.nuisance
- tomojax.forward
- tomojax.recon
- tomojax.align
- tomojax.verify
- tomojax.datasets
- tomojax.backends
- tomojax.io
- tomojax.cli

Forbidden across module boundaries:
- importing tomojax.geometry._anything from outside tomojax.geometry
- importing tomojax.forward._anything from outside tomojax.forward
- adding generic utils/helpers/misc modules
- spreading one feature across many shallow files without a clear module owner

When new top-level modules become importable, update `.importlinter` in the same milestone so dependency direction stays executable.

Each deep module should have:
- api.py
- __init__.py that re-exports the public API
- private _*.py implementation files
- README.md describing purpose, public API, dependencies, invariants, and tests

## Rewrite policy

This is a reimagining, not a compatibility refactor.

Prefer deleting obsolete code to adapting it.

Do not preserve old CLI paths, old staged aligners, old experimental branches, or duplicated kernels unless they are required for benchmark comparison and have tests.

Do not create legacy/, old/, compat/, scratch/, helpers.py, utils.py, or misc.py.

Git history is the archive.

## Numerical policy

The default geometry solver is robust Schur LM/GN over setup plus per-view 5-DOF pose.

The default geometry loss is masked, whitened pseudo-Huber projection-domain reprojection loss with weak priors and gauge canonicalisation.

The default reconstruction step is FISTA / Huber-TV FISTA with stopped-gradient latent volumes during alignment.

Do not introduce grid-search alignment methods as default behaviour.

Non-gradient methods are allowed only as benchmarked optional accelerators or diagnostics.

## Development policy

Before making large changes:
1. Read the relevant design docs.
2. Create or update an execution plan.
3. Identify the deep module owner.
4. Add or update tests first where practical.
5. Keep diffs scoped to the current milestone.

After each milestone:
1. Run `just check`.
2. Fix failures immediately.
3. Update docs/implementation_log.md.
4. Record decisions and deviations from the design docs.

## Quality gates

Every public API must be typed.
Every deep module must have tests.
Every geometry update must emit artifact/provenance data.
Every backend fast path must have a JAX reference comparison.
Every synthetic recovery path must be deterministic from a seed.

## Done means

A milestone is done only when:
- implementation exists,
- tests exist,
- `just check` passes,
- docs/implementation_log.md is updated,
- dead code introduced by the change has been deleted.
