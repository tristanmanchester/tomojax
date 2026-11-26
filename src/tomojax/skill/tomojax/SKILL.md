---
name: tomojax
description: Reconstructs, aligns and checks X-ray tomography with TomoJAX (parallel-beam, laminography and lab cone-beam CT) through its MCP server or Python API. Use when the user wants to load or correct a scan, find the rotation centre, align per-view motion, reconstruct a volume, compare or tune reconstruction settings without ground truth, point at or segment a feature in a slice, or report a TomoJAX bug.
license: GPL-3.0-only
compatibility: Python 3.12 or newer with the tomojax package; a CUDA 12 GPU is optional but much faster; the MCP tools need the tomojax[mcp] extra and a client running `tomojax mcp`.
---

# TomoJAX

TomoJAX turns X-ray projections into volumes: parallel-beam synchrotron scans, laminography and lab cone-beam CT. It loads the common facility and vendor formats, corrects raw frames, fits the rotation centre, setup angles and per-view motion, and reconstructs with analytical or iterative methods. Volumes are `(z, y, x)` and projections `(view, v, u)`; voxel and detector spacings share one length unit and angles are in degrees.

## MCP server or your own Python

Use the **TomoJAX MCP server** (tools such as `open`, `reconstruct`, `view`, `python`) for anything that touches TomoJAX data:

- operations on scans and volumes, which stay resident in the server's worker between calls, on the GPU when there is one;
- rendering slices with TomoJAX's display conventions (`view`), including marks and feature outlines;
- long jobs, which run as MCP tasks with progress and cancellation.

For compound work on that data (a sweep of settings, a loop over slices, a custom measure) use the server's `python` tool: one call runs your code in the worker beside the data and returns a small result. Its namespace has `tj`, `np`, `handles`, `register`, `register_feature`, `view` and `show`; [references/python.md](references/python.md) has the details and worked examples.

Use **your own Python** for everything general: plotting your own figures, statistics on numbers the tools returned, file management, writing reports. Do not copy large arrays out of the server to do what a `python` call can do in place.

Without the MCP server, the same operations are plain Python calls on `import tomojax as tj`.

## The five verbs

| Verb | MCP tool | Python |
| --- | --- | --- |
| Load | `inspect`, then `open` | `tj.inspect(path)`, `tj.load(path)` or `tj.load_frames(path)` |
| Correct | `correct` | `frames.corrected(...)` |
| Align | `align` | `tj.align(scan, fit=...)` |
| Reconstruct | `reconstruct` | `tj.reconstruct(scan, method, ...)` |
| Save | `save`, `export` | `tj.save(path, result)`, `tj.export(...)` |

Key choices:

- **Method.** `auto` picks the Fourier-slice inverse for unposed axial half-turn scans, CGLS for segmented cone geometry, and FBP/FDK otherwise. Its choice and limitations are in the result's `info["automatic"]` (`method`, `reason`, `status`, `reasons`, `center_status`); read it before trusting the result, since automatic selection checks neither geometry nor artefacts. For noisy or few-view data, `fista` with `tv_weight` and `nonnegative=True` usually gives a cleaner volume, and `warm_start` starts it from FBP. [references/solvers.md](references/solvers.md) covers the solvers.
- **Centre.** TomoJAX never estimates the centre during reconstruction: it comes from the file or from `align`. Fit it with `align` and `fit="center"` (or `method="opposite-views"` for a parallel scan with opposing views), or check by eye with a `centre_sweep` preview, whose offsets are signed horizontal detector pixels. The fitted centre is in the detector's length unit, matching `Detector.center`. See [references/find-the-centre.md](references/find-the-centre.md).
- **Alignment fits.** `fit` takes `center`, `tilt`, `roll`, `translations` (per-view shifts) and `rotations` (per-view wobble). Fit the centre first on a new instrument.
- **Before a large job**, call `reconstruct` or `align` with `dry_run=true` to see the method, grid and memory placement.

## Check a result without ground truth

A low residual is not a good reconstruction. Look at slices: `view` with no axis shows three central planes, and an off-centre slice and a profile across a sharp edge show blur, double edges, rings and streaks. Then:

- compare settings with `evaluate`, which scores candidates on held-out projections, and inspect the leading candidates' slices at the same display limits;
- for a decision about a specific structure, score candidates on that structure ([references/tuning-without-truth.md](references/tuning-without-truth.md));
- with a simulation, and only then, compare against the phantom.

Simulated data has ground truth; real scans do not. Do not propose optimising against ground truth unless the data is simulated.

## Agree a feature by pointing at it

When the user names a structure ("the bright thin shell", "that crack"), do not describe it back in words. Mark it, check the report, then ask:

1. **Mark.** `view` the slice with `marks`. Point at a pixel of the image you are looking at, `{"type": "point", "at": {"pixel": [x, y], "image": "render-…"}, "label": "shell?", "snap": "peak"}`, or give array indices, `"at": [z, y, x]`. `snap` (`peak`, `trough` or `edge`) locks the point onto the brightest or darkest sample or the steepest edge within `within` index units (default 5). For a small or thin target, zoom in with an `roi` around it first.
2. **Check.** A point or arrow on flat ground, with no peak, trough or edge within 2 samples, comes back with a `warning` and `nearest`: the array indices, distance, value and percentile of the closest structure on that plane. Mark `nearest.at` with `snap` and view again. On structure, compare the `value` and its `percentile` with the description: a "bright" structure under a mark at the 3rd percentile is the wrong structure. Never ask the user about a mark that has a warning or whose value contradicts what they described.
3. **Ask.** Show the image and ask whether that is the feature. Mark alternatives in different colours if unsure.
4. Once confirmed, make it a feature handle with the `feature` tool and outline it with `view` and `features`, on any volume on the same grid and side by side across candidates (`source` as a list).

[references/pointing-and-features.md](references/pointing-and-features.md) has the mark types and feature recipes, and [references/thin-shell.md](references/thin-shell.md) works through a thin bright shell from a mark to scores.

## Stay within the task

Work with the data, handles and files the task names, and the outputs you create. Do not read other agents' threads, sessions or working files, or files outside the task, to find hints or answers.

## Report a TomoJAX bug

If TomoJAX raises an internal error, gives a result that is clearly wrong, or documentation disagrees with behaviour, follow [references/reporting-bugs.md](references/reporting-bugs.md). Never attach the user's data, file paths or images of their data without their permission.

## When to read each reference

| Reference | Read it when |
| --- | --- |
| [reconstruct-scan.md](references/reconstruct-scan.md) | Reconstructing a scan from a file or handle |
| [lab-ct-start-to-finish.md](references/lab-ct-start-to-finish.md) | Processing a lab cone-beam acquisition (Nikon, Zeiss, TIFF) |
| [find-the-centre.md](references/find-the-centre.md) | The centre is unknown or the slices show rings or arcs |
| [check-a-reconstruction.md](references/check-a-reconstruction.md) | Judging a volume or comparing settings |
| [pointing-and-features.md](references/pointing-and-features.md) | Showing the user a location, or turning it into a region |
| [tuning-without-truth.md](references/tuning-without-truth.md) | Choosing reconstruction settings on a real scan |
| [thin-shell.md](references/thin-shell.md) | Segmenting or scoring a thin structure that touches two materials |
| [python.md](references/python.md) | Writing code for the `python` tool, custom solvers or analysis |
| [solvers.md](references/solvers.md), [geometry.md](references/geometry.md), [calibration.md](references/calibration.md) | Direct solver calls, custom geometry, acquisition metadata |
| [large-scans.md](references/large-scans.md) | The scan or volume is larger than GPU or host memory, or there are several GPUs |
| [reporting-bugs.md](references/reporting-bugs.md) | Filing an issue |

The MCP server also serves the workflow references as `tomojax://guide/{topic}` resources and prompts, `tomojax://handle/{id}` for a handle's provenance and `tomojax://describe/{operation}` for an operation's settings.
