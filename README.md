# TomoJAX v2 Rewrite

This branch is reimagining TomoJAX as a fast, differentiable tomography and
laminography alignment/reconstruction toolbox.

The package-facing CLI is the grouped `tomojax` command:

```bash
uv run tomojax inspect scan.nxs
uv run tomojax ingest ./projections --angles angles.csv --out scan.nxs
uv run tomojax preprocess raw.nxs corrected.nxs --log
uv run tomojax recon corrected.nxs --out recon.nxs
uv run tomojax align corrected.nxs --out aligned.nxs --schedule cor
```

The installed package exposes a single `tomojax` console script. Developer
diagnostics and benchmark probes live under `tomojax dev ...`.

The canonical v2 design docs live in [`docs/tomojax-v2/`](docs/tomojax-v2/):

- [`01_high_level_architecture.md`](docs/tomojax-v2/01_high_level_architecture.md)
- [`02_loss_and_optimiser_spec.md`](docs/tomojax-v2/02_loss_and_optimiser_spec.md)
- [`03_repo_layout.md`](docs/tomojax-v2/03_repo_layout.md)
- [`04_phased_implementation_plan.md`](docs/tomojax-v2/04_phased_implementation_plan.md)
- [`05_synthetic_128_benchmark_suite.md`](docs/tomojax-v2/05_synthetic_128_benchmark_suite.md)
- [`06_verification_and_artifact_contract.md`](docs/tomojax-v2/06_verification_and_artifact_contract.md)
- [`07_synthetic_generator_pseudocode.md`](docs/tomojax-v2/07_synthetic_generator_pseudocode.md)

Current user-facing workflow docs:

- [`docs/quickstart.md`](docs/quickstart.md)
- [`docs/real-laminography.md`](docs/real-laminography.md)
- [`docs/synthetic-tomography.md`](docs/synthetic-tomography.md)
- [`docs/known-limitations.md`](docs/known-limitations.md)
- [`docs/benchmark_runs/2026-05-13-production-readiness.md`](docs/benchmark_runs/2026-05-13-production-readiness.md)

This is not a backwards-compatible refactor. The v2 work should prefer deep
modules, small public APIs, typed boundaries, executable architecture checks,
and deletion of obsolete staged-alignment compatibility code.

## Agent Workflow

Read [`AGENTS.md`](AGENTS.md) before making changes. Use
[`docs/tomojax-v2/04_phased_implementation_plan.md`](docs/tomojax-v2/04_phased_implementation_plan.md)
as the canonical phased plan and keep [`.agent/PLANS.md`](.agent/PLANS.md)
updated for the active milestone.

Record milestone decisions, validation commands, known failures, and design
deviations in [`docs/implementation_log.md`](docs/implementation_log.md).

## Guardrails

The branch has guardrails for the rewrite:

- Ruff formatting and linting
- basedpyright type checking
- import-linter dependency contracts
- `tools/check_public_imports.py` for private-module boundary checks
- pre-commit configuration
- `just` command recipes

Useful commands:

```bash
just --list
just imports
just typecheck
just check
```

`just check` is the target milestone gate. It may fail while retained internal
implementation code is being migrated; do not weaken the guardrails to make
that code pass. Instead, remove or migrate retained code into the v2 deep
module architecture and record temporary failures in
`docs/implementation_log.md`.
