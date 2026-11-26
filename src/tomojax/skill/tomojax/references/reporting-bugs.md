# Report a TomoJAX bug

Report an internal error (`kind` `internal` in a tool error), a result that is clearly wrong, or documentation that disagrees with what TomoJAX does. A bad reconstruction of bad data is not a bug: check the center, the corrections and the geometry first.

## What to collect

1. **Version.** `status` reports TomoJAX's version; in Python, `tj.__version__`.
2. **The exact call.** The tool and its arguments, or the Python code. `tomojax://handle/{id}` gives the provenance manifest of a handle: every operation and setting that produced it.
3. **What happened.** The error message, or a description of the slice that looks wrong and what you expected instead.
4. **A reproducer on simulated data**, whenever the problem does not depend on the user's scan: start from `simulate` (or `tj.simulate` with the same geometry kind, size and view count) and show the same failure. A reproducer that runs without the user's files is the most useful part of the report.

## Privacy

**Never attach the user's data, file paths, or images of their data without their permission.** This includes scan files, slices rendered from their scans, paths that name their facility, proposal or sample, and manifests that contain those paths. Facility data can be confidential or unpublished. Describe what you saw instead, and ask the user before including anything from their files. Images of simulated data are fine.

## File the issue

Show the user the draft and get their agreement before filing. Then:

```bash
gh issue create -R tristanmanchester/tomojax --title "<one line: what fails>" --body-file report.md
```

In `report.md`, give the version, the call, what happened, what you expected and the simulated reproducer, in that order. Check first whether an open issue already describes it: `gh issue list -R tristanmanchester/tomojax --search "<words>"`.
