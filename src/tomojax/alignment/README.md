# tomojax.alignment

Use `tj.align(scan, fit=..., grid=...)`. The [alignment guide](../../../docs/align-a-scan.md)
explains how to choose the parameters and check the result. The default fits per-view
translations; `"centre"`, `"tilt"`, `"roll"` and `"rotations"` name the additional parameter
blocks. All geometries use one private joint solver in `_joint`.

The solver estimates the image and the requested geometry corrections together. Per-view
translation blocks are coupled by a smooth-drift and robust-jump prior. Shared centre and angle
columns border that system. Exact Schur elimination gives the image update; the backprojector
is the exact transpose of the selected forward operator, including detector sampling and
missing-ray masks. The returned scan carries the calibrated geometry and fitted per-view poses.

The line search compares a pose trial with the image-only trial at the **same image step**.
An image improvement cannot pay for a harmful pose move. Per-view rotations pay a size
prior in radians, including their smooth components, so smooth rotations are not free.

A level stops after two consecutive small geometry proposals, following at least four outer
steps. Both proposals must use damping at or below its initial value. The size is measured before
line-search scaling: rejecting a pose move, or suppressing it with heavy damping, does not
establish convergence. Further image improvement can continue in a subsequent reconstruction.

`tomojax.alignment.api` retains the projection-only centre preview and its diagnostic report.
They do not select a different alignment engine behind `tj.align`.
