# tomojax.alignment

Use `tj.align(scan, fit=..., grid=...)`. The [alignment guide](../../../docs/align-a-scan.md) explains how to choose parameters and inspect results. By default `fit="translations"` fits per-view lab-frame translations; `"center"`, `"tilt"`, `"roll"`, and `"rotations"` select additional parameter blocks. All geometries share one private joint solver in `_joint`.

The solver estimates the volume and selected geometry parameters together. With the default `motion="independent"`, each view moves freely; with `motion="smooth"`, per-view translations follow a smooth-drift and robust-jump prior and per-view rotations couple to neighbours. Shared centre and angle columns border the normal system. Exact Schur elimination gives the image update, and the backprojector is the exact transpose of the forward operator. Each working pixel traces one ray through its centre, binned no coarser than a voxel. Inside every step, each view's offset and vertical row ramp relative to the scan-wide mean level are profiled out so flux fluctuations do not mimic vertical motion. The returned `result.scan` carries the calibrated geometry and fitted per-view poses.

The line search compares each pose trial against the image-only trial at the same image step so an image improvement cannot pay for a worse pose. Per-view rotations carry a prior in radians (including their smooth components).

A pyramid level stops after two consecutive small geometry updates (following at least `min_iterations` steps) when damping is at or below its starting value. Step size is measured before line-search scaling.

`tomojax.alignment.api` also exposes the projection-only centre estimator and its diagnostic report (used by `tj.align(..., method="opposite-views")`).
