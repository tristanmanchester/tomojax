---
name: tomojax
description: Reconstructs, aligns and checks X-ray tomography with TomoJAX (parallel-beam, laminography and lab cone-beam CT) through its MCP server or Python API. Use when the user wants to load or correct a scan, find the rotation center, align per-view motion, reconstruct a volume, compare or tune reconstruction settings without ground truth, point at or segment a feature in a slice, or report a TomoJAX bug.
license: GPL-3.0-only
compatibility: Python 3.12 or newer with the tomojax package; a CUDA 12 GPU is optional but much faster; the MCP tools need the tomojax[mcp] extra and a client running `tomojax mcp`.
---

# TomoJAX

TomoJAX turns X-ray projections into volumes: parallel-beam synchrotron scans, laminography and lab cone-beam CT. It loads the common facility and vendor formats, corrects raw frames, fits the rotation center, setup angles and per-view motion, and reconstructs with analytical or iterative methods. Volumes are `(z, y, x)` and projections `(view, v, u)`; voxel and detector spacings share one length unit and angles are in degrees.

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

- **Method.** `auto` picks the Fourier-slice inverse for well-sampled axial half-turn and full-turn scans that are unposed or carry axial rotations about `z` alone, CGLS for segmented cone geometry and for cone arcs shorter than 180° plus the fan angle (from the effective source orbit), and FBP/FDK otherwise. Its choice and limitations are in the result's `info["automatic"]` (`method`, `reason`, `status`, `reasons`, `center_status`); read it before trusting the result, since automatic selection checks neither geometry nor artifacts. For region-of-interest scans where the object extends past the detector, `fourier`, `fbp`, and cone-beam FDK continue the unmeasured exterior with a cylinder model by default (`exterior="cylinder"`; disable with `config={"exterior": "zero"}` and an explicit `method`). CGLS, SIRT, FISTA and SPDHG subtract the estimated contribution from outside the reconstruction grid before fitting the measured rays. MLEM adds its nonnegative estimate to the predicted counts. Set `exterior="zero"` in the iterative config to fit the original data. With `iterations` omitted, unpenalized `cgls` and full-data `sirt` stop at an estimated projection discrepancy capped at 50 updates (full-data `sirt` uses momentum when `relaxation <= 1`), while `fista` calibrates TV weight and stops at its proximal-gradient noise estimate up to 200 updates; read `info["termination"]`, `early_stop`, and `effective_iterations` to check whether the cap ended the run. For noisy or few-view data, `fista` with `nonnegative=True` usually gives a cleaner volume, and `warm_start` starts it from FBP. [references/solvers.md](references/solvers.md) covers the solvers and stopping rules.
- **Center.** TomoJAX never estimates the center during reconstruction: it comes from the file or from `align`. Fit it with `align` and `fit="center"` (no other options needed): on an unposed, unrolled, well-sampled axial parallel scan this measures the axis directly across the full detector width without reconstructing a volume (`image` is `None`); on other geometries, or with `method="joint"` or joint options, it runs the joint fit (seeded from the measured center on axial parallel scans). Use `method="opposite-views"` only for a pair-only check, or check by eye with a `center_sweep` preview, whose offsets are signed horizontal detector pixels. The fitted center is in the detector's length unit, matching `Detector.center`. See [references/find-the-center.md](references/find-the-center.md).
- **Rings and flat fields.** Use `Rings()` in Python (or `{"type": "Rings"}` in `correct` and `reconstruct_slabs`) with no width parameter to remove fine and wide detector rings. It handles stationary column errors on half and full turns; angle-varying swirls are not rings and are not removed. Bracketing multi-frame flat sets default to `flat_weights: "views"`, weighting local texture from a sample of the views while interpolating common row gain by position (single-frame sets interpolate by position). When views move the flat by more than 1% over more than 1% of the detector, `correct` and `reconstruct_slabs` return a warning in `notes` (and in the text summary). Inspect `flat_moved_by_views` and `flat_structure_unmatched` in the `flat_dark` record (`scan.corrections` in Python, `handles[id].corrections` in MCP `python`, or `inspect` after `save`). Shadows in both flat sets leave a broad error beyond what `Rings()` removes, and smooth beam-level changes cannot be separated from object attenuation; borrow clean flats from another scan by passing its path as `flats` on `open` or `tj.load_frames`, or pass `flat_weights: "position"` to blend by position alone.
- **Half-acquisition.** On an axial parallel full turn with an offset detector, keep all views and fit `center`. Inspect the aligned scan for `field_of_view_px` and `overlap_px` (or `central_gap_px` if the axis lies outside the detector). When the input file records no grid, the aligned scan's grid expands to the far edge; otherwise pass a grid explicitly. Ring correction uses the fitted axis. For multi-gigabyte outputs, fit `center` on a cropped row band, then open the full raw frames with `geometry: {"detector_center": [U, V]}` (or a full-height `Detector` in Python) and stream with `reconstruct_slabs`.
- **Alignment fits.** `fit` takes `center`, `tilt`, `roll`, `translations` (per-view shifts) and `rotations` (per-view wobble). Fit the center first on a new instrument. Every joint fit checks for objects crossing the field of view and returns `disturbed_views` (`score`, `ranges`, `angles`, `message`). Inspect any flagged projections before excluding confirmed half-open ranges via `config: {"exclude_views": [[start, stop], ...]}` (excluded views stay in `result.scan` with interpolated pose increments, and `fit_views` lists the retained indices). `AlignConfig` also accepts `detector_columns` (`"fixed"`, `"transient"`, or `"both"`) for column errors and `image_tv_weight="coverage"` for coverage-weighted TV on the working volume.
- **Before a large job**, call `reconstruct` or `align` with `dry_run=true` to see the method, grid and memory placement. A job that may be killed or canceled takes `checkpoint`, a directory it resumes from when called again with the same arguments; see [references/large-scans.md](references/large-scans.md).
- **What the file and the solver tell you.** `inspect` reports under `loading` what a file does not record (pixel size, rotation center, rotation axis), whether a stopped scan was cut short, and full-turn parallel orbit dimensions (`full_turn`, `axis_to_edge_px`, `field_of_view_px`, `overlap_px`, and `central_gap_px`); a file with `geometry: "not_recorded"` needs `geometry` declared on `open`. Every `reconstruct` and `reconstruct_slabs` result reports `truncation` (`outside_fraction` and `interior_shift`, plus `"N% of attenuation beyond the detector"` in the summary text from 1% up), and `reconstruct_slabs` saves that whole-scan record and iterative stopping diagnostics in `volume.reconstruction_info`. Iterative results report `termination` and `effective_iterations` (plus `slab_stopping` on independent axial slabs), and FISTA its automatic `resolved_tv_weight`; quote them when you report a result.

## Check a result without ground truth

A low residual is not a good reconstruction. Look at slices: `view` with no axis shows three central planes, and an off-center slice and a profile across a sharp edge show blur, double edges, rings and streaks. Then:

- compare settings with `evaluate`, which scores candidates on held-out projections, and inspect the leading candidates' slices at the same display limits;
- for a decision about a specific structure, score candidates on that structure ([references/tuning-without-truth.md](references/tuning-without-truth.md));
- with a simulation, and only then, compare against the phantom.

Simulated data has ground truth; real scans do not. Do not propose optimizing against ground truth unless the data is simulated.

## Agree a feature by pointing at it

When the user names a structure ("the bright thin shell", "that crack"), do not describe it back in words. Mark it, check the report, then ask:

1. **Mark.** `view` the slice with `marks`. Point at a pixel of the image you are looking at, `{"type": "point", "at": {"pixel": [x, y], "image": "render-…"}, "label": "shell?", "snap": "peak"}`, or give array indices, `"at": [z, y, x]`. `snap` (`peak`, `trough` or `edge`) locks the point onto the brightest or darkest sample or the steepest edge within `within` index units (default 5). For a small or thin target, zoom in with an `roi` around it first.
2. **Check.** A point or arrow on flat ground, with no peak, trough or edge within 2 samples, comes back with a `warning` and `nearest`: the array indices, distance, value and percentile of the closest structure on that plane. Mark `nearest.at` with `snap` and view again. On structure, compare the `value` and its `percentile` with the description: a "bright" structure under a mark at the 3rd percentile is the wrong structure. Never ask the user about a mark that has a warning or whose value contradicts what they described.
3. **Ask.** Show the image and ask whether that is the feature. Mark alternatives in different colors if unsure.
4. Once confirmed, make it a feature handle with the `feature` tool and outline it with `view` and `features`, on any volume on the same grid and side by side across candidates (`source` as a list).

[references/pointing-and-features.md](references/pointing-and-features.md) has the mark types and feature recipes, and [references/thin-shell.md](references/thin-shell.md) works through a thin bright shell from a mark to scores.

## Stay within the task

Work with the data, handles and files the task names, and the outputs you create. Do not read other agents' threads, sessions or working files, or files outside the task, to find hints or answers.

## Report a TomoJAX bug

If TomoJAX raises an internal error, gives a result that is clearly wrong, or documentation disagrees with behavior, follow [references/reporting-bugs.md](references/reporting-bugs.md). Never attach the user's data, file paths or images of their data without their permission.

## When to read each reference

| Reference | Read it when |
| --- | --- |
| [reconstruct-scan.md](references/reconstruct-scan.md) | Reconstructing a scan from a file or handle |
| [lab-ct-start-to-finish.md](references/lab-ct-start-to-finish.md) | Processing a lab cone-beam acquisition (Nikon, Zeiss, TIFF) |
| [find-the-center.md](references/find-the-center.md) | The center is unknown or the slices show rings or arcs |
| [check-a-reconstruction.md](references/check-a-reconstruction.md) | Judging a volume or comparing settings |
| [pointing-and-features.md](references/pointing-and-features.md) | Showing the user a location, or turning it into a region |
| [tuning-without-truth.md](references/tuning-without-truth.md) | Choosing reconstruction settings on a real scan |
| [thin-shell.md](references/thin-shell.md) | Segmenting or scoring a thin structure that touches two materials |
| [python.md](references/python.md) | Writing code for the `python` tool, custom solvers or analysis |
| [solvers.md](references/solvers.md), [geometry.md](references/geometry.md), [calibration.md](references/calibration.md) | Direct solver calls, custom geometry, acquisition metadata |
| [large-scans.md](references/large-scans.md) | The scan or volume is larger than GPU or host memory, there are several GPUs, or a cluster job needs checkpoint and resume |
| [reporting-bugs.md](references/reporting-bugs.md) | Filing an issue |

The MCP server also serves the workflow references as `tomojax://guide/{topic}` resources and prompts, `tomojax://handle/{id}` for a handle's provenance and `tomojax://describe/{operation}` for an operation's settings.
