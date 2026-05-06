# TomoJAX Implementation Log

This log records implementation milestones, validation commands, design
decisions, deviations from `docs/tomojax-v2/`, and unresolved risks.

## 2026-05-06 — Milestone 0 Guardrail Preparation

### Summary

- Added strict Ruff, pytest, basedpyright, import-linter, and pre-commit guardrails.
- Added `tools/check_public_imports.py` to prevent cross-boundary imports from
  private TomoJAX modules.
- Marked existing white-box tests with explicit
  `check-public-imports: allow-private` exceptions.
- Added `.agent/PLANS.md` as the active milestone execution-plan workspace.
- Added this implementation log.

### Decisions

- `docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased
  implementation plan.
- `.agent/PLANS.md` is not a competing plan. It is the active milestone work log
  and checklist.
- `.importlinter` currently reflects importable packages in the transitional
  tree. It must be updated as the v2 deep-module skeleton becomes real.

### Validation

- `uv lock` completed after adding `basedpyright` and `pre-commit`.
- `just --list` found the canonical command surface.
- `uv run pre-commit validate-config` passed.
- `uv run ruff check tools/check_public_imports.py pyproject.toml` passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run basedpyright --version` reported `basedpyright 1.39.3`.
- `just typecheck` currently fails on the transitional source and test tree.
  This is expected technical debt from the pre-restructure implementation, not a
  guardrail installation failure. The v2 implementation should make strict
  typechecking pass as modules are replaced.

### Risks

- The current codebase still contains transitional modules such as
  `tomojax.utils`, `tomojax.data`, and `tomojax.calibration`. The v2 restructure
  should delete or migrate them under the deep-module architecture rather than
  preserve them as compatibility surfaces.
- `just check` cannot pass until type errors in the existing implementation are
  removed or the old implementation is replaced by the v2 deep modules.
