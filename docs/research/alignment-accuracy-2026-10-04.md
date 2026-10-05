# Alignment accuracy diagnostics, 2026-10-04

Two distinct numerical limits remain after the [coupled pilot](public-free-voxel-joint-2026-10-04.md):
the central pose stencil loses precision, and the inner system is inadequately
solved. Exact pose-block elimination has mixed residual results with the current
volume preconditioner. A [subsequent nonlinear check](shared-conditioning-2026-10-04.md)
corrects the initial decision to reject it from residual norms alone: tilted
pose recovery can improve despite a larger normal residual. No solver default,
volume basis, acceptance gate, or gauge was changed.

[Raw results and diagnostic source](../../bench/reference/alignment-accuracy-2026-10-04.json.gz)
include all six cells, both Jacobian implementations, three linear budgets,
and the independent small dense checks. Terminal states were recaptured through
public `align` using frozen runtime source
`73a6925946726db21367d405b72593a25e9f16952177220322a62c3bf00e1d07`.
These fixed-state diagnostics are not new accepted-result timing baselines.
The published comparison retains the cold/warm timings and sampled process memory.

## Derivatives

At each terminal state, views 0, 15, 30, 45 and 60 were checked on every detector
pixel and all five pose coordinates. The reference uses independent FP64 SciPy
rigid transforms and piecewise polynomial ray integration. It bypasses the
FP32 output cast in `project_voxel_truth`. Physical central perturbations of
`1e-5`, `1e-6`, and `1e-7` times the minimum pitch (divided by the object radius
for rotations) check reference stability. Relative column errors aggregate
the sampled views/pixels per DOF; the table reports the largest DOF error.

| Cell | FP32 central relative error | JAX analytic relative error |
|---|---:|---:|
| parallel-clean | 0.006084 | 3.69e-6 |
| parallel-noisy | 0.006120 | 3.64e-6 |
| anisotropic-clean | 0.009302 | 1.14e-6 |
| anisotropic-noisy | 0.008981 | 1.28e-6 |
| lamino-clean | 0.007081 | 3.30e-6 |
| lamino-noisy | 0.006943 | 3.44e-6 |

The existing Pallas analytic implementation also agrees to a few parts per
million. Comparing the production central stencil with the FP64 reference at
the *same* step gives essentially the same error: arithmetic dominates the
measured discrepancy, rather than finite-difference truncation at these states.
The small-step FP64 comparisons agree within approximately 1e-7 relative error.
This is evidence for testing the existing analytic option, not evidence that
central differences caused the entire remaining recovery error.

Nominal parallel poses were checked separately. Rays can lie exactly in voxel
knot planes. There, the trilinear basis has different one-sided derivatives;
the analytic choice and a symmetric central stencil can differ by 36–42% in
some DOFs. That discrepancy is not an analytic-kernel accuracy claim at a
smooth point. Blindly substituting one derivative at every pose needs an
end-to-end check from the original initialization.

## Linear accuracy

Both solvers operate at the same saved state, with the same central columns,
matched exact FP/BP, positivity active set, damping, zero TV, and zero pose
smoothness. Pose elimination factors the 61 five-by-five blocks and applies
the true damped volume Schur operator; it does not square an incorrectly
assumed orthogonal residual projector. Each Krylov iteration uses one FP/BP
pair; setup, back-substitution and explicit verification add operator calls.
These are accuracy comparisons at iteration caps, not speed comparisons.

The entries below are explicitly recomputed **full joint normal residuals**,
relative to the starting right-hand side. The requested tolerance is `1e-4`.

| Cell | Stacked, 40 | Pose eliminated, 40 | Stacked, cap 640 | Pose eliminated, cap 640 |
|---|---:|---:|---:|---:|
| parallel-clean | 0.1799 | 0.1492 | 1.14e-4 | 9.20e-5 (326 iterations) |
| parallel-noisy | 0.3268 | 0.2645 | 4.98e-4 | 7.07e-5 (422) |
| anisotropic-clean | 0.3498 | 0.1646 | 1.51e-3 | 9.83e-5 (444) |
| anisotropic-noisy | 0.5354 | 0.5270 | 4.36e-3 | 5.13e-5 (522) |
| lamino-clean | 0.5832 | 1.6166 | 0.3424 | 0.1265 (640, unconverged) |
| lamino-noisy | 0.6450 | 2.1394 | 0.5197 | 0.1529 (640, unconverged) |

Elimination improves high-budget residuals but does not solve the tilted
conditioning problem. At the existing 40-iteration cap it worsens both tilted
cells. This alone does not determine nonlinear recovery: see the correction
above. Do not promote this variant, tighten only its nominal tolerance, or
tune its already better parallel cases. A shared volume preconditioner or
multilevel correction needs testing on plain reconstruction and the eliminated
system together before revisiting it. These residuals do not support publishing
a pilot reduced-Hessian spectrum from unconverged inverse applications.

## Gauge and scope of identifiability evidence

An independent FP64 dense check used an asymmetric 6×5×4 free-voxel object and
13 irregular views in each geometry. All 120 voxel columns were independent.
After best unconstrained voxel compensation, common rigid-frame perturbations
left 15–70% of their original projection perturbation in these small problems.
Continuous object-frame symmetry is therefore not an exact null direction of
this finite, fixed trilinear basis. Removing those directions inside the solve
would change the discrete problem.

Illustrative spectra after quotienting the six continuous frame directions had
condition numbers 586 (parallel), 775 (anisotropic), and 103 (laminography), with
no numerical nulls at a `1e-12` relative threshold. This different, tiny object
does **not** establish pilot identifiability, predict its noisy rotation floor,
or justify treating every weak eigenvector as gauge. The public solver gauge
and shared-frame acceptance verification remain unchanged.

Next: test the existing analytic Jacobian on all six original cells, holding
the rest of the workflow fixed. Then investigate a common volume conditioning
change across all 27 reconstruction cells and six alignment cells. Damping and
adaptive inner accuracy remain separate experiments; no new benchmark framework
is required.
