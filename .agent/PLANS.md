# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: clear the first alignment I/O lint blockers after geometry cleanup.

### Scope

- In scope:
  - Add missing checkpoint and params-export docstrings.
  - Clean local checkpoint file handling and validation lint.
  - Reduce `_json_native` return-count lint in params export.
  - Run focused Ruff checks and checkpoint/export tests.
- Out of scope:
  - Alignment algorithm changes.
  - Alignment model lint cleanup.
  - Repository-wide legacy Ruff cleanup outside alignment I/O.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Clean checkpoint doc/file-handling lint.
- [x] Clean params-export doc/return-count lint.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff format src/tomojax/align/io/checkpoint.py src/tomojax/align/io/params_export.py`
  passed.
- `uv run ruff check src/tomojax/align/io/checkpoint.py src/tomojax/align/io/params_export.py`
  passed.
- `uv run pytest tests/test_align_checkpoint.py tests/test_align_quick.py -q`
  passed: 33 tests.
- `uv run pytest tests/test_align_params_export.py -q` passed: 8 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The touched alignment I/O files are no longer in the failure
  list; the first remaining blockers are in `src/tomojax/align/model/*`,
  followed by broader repository lint backlog. Formatter churn from
  `just check` was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Preserved checkpoint atomic-write semantics while switching to `Path.replace`
  and `contextlib.suppress` for temporary-file cleanup.
- Kept params-export JSON normalization behavior but collapsed scalar/list
  conversion into a single final return path.
- Deviation: none from the cleanup scope.

### Risks

- Risk: checkpoint write-path cleanup could change atomic save behavior.
- Mitigation: preserve fsync and cleanup semantics and run checkpoint tests.
- Proposed next fix for `just check`: alignment model lint cleanup after I/O is
  clear.
