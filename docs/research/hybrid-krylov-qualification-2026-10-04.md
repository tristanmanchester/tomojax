# Filtered hybrid Krylov prototype: rejected qualification, 2026-10-04

The first filtered-adjoint/GCV prototype passes the independent algebra checks
but does not qualify for integration. At 64 steps it improves image error in
only three of twelve small physical controls; the other nine worsen compared
with the ordinary unregularized Krylov control. Allowing GCV to select an earlier
iterate improves four noisy controls, while the tilted noisy cases still worsen.
Its retained image and projection bases also exceed the 8 GB showcase budget.
This particular prototype stops here without tuning its passing cases.

No public reconstruction or alignment solver changed. The frozen 27-cell matrix
and six-cell recovery pilot were **not rerun**: this candidate failed the smaller
qualification that precedes such a run. All 33 scheduled cells are explicitly
retained as `not_run_failed_prototype_qualification`, with null cold/warm, quality
and measured GPU-memory fields. This report establishes no workflow speedup,
new accepted reconstruction, successful recovery baseline, or recovery rate.

## Algorithm and checks

The prototype builds an orthonormal basis V for K(Aᵀ F A, Aᵀ F y), then solves
`min_z ||A V z - y||² + lambda² ||z||²`. Thus it evaluates the original data
residual, not the filtered normal residual. Since V is orthonormal, its
coefficient penalty is the reconstructed-image Euclidean norm. The image starts
at zero. Directions stay in range(Aᵀ), so this construction does not introduce
the null component exposed by the previous image-space preconditioner.

F is a positive radial filter on each detector plane in physical frequency units,
with a nonzero floor equal to the reciprocal detector diagonal. It is identical
in construction across tilted, unequal-spacing and irregular geometry. Symmetry
and positivity are checked explicitly. It is an approximate conditioning choice,
not a demonstrated inverse of the physical normal operator.

GCV chooses lambda from the measured data and small projected SVD. Its residual
includes energy outside range(A V), and its denominator uses all measured rows.
A fixed logarithmic scan and scalar minimization inspect candidate minima;
lambda zero is also allowed. Truth never enters the solver or parameter choice.
Using this conditional projected fit does not prove that its degrees-of-freedom
estimate fully accounts for a data-dependent Krylov basis.

Independent checks compare the projected solution with an augmented dense least
squares solve and compare GCV with an explicit influence matrix. They include
overdetermined, underdetermined, and rank-deficient matrices. Every physical
iterate is checked for orthogonality, original projected-objective stationarity,
and absence of an added null-space component. All checks passed.

The experiment follows the direction suggested by
[filtered BA-GMRES](https://arxiv.org/abs/2201.07408) and
[hybrid ABBA-GMRES](https://arxiv.org/abs/2602.17892), but is not a reproduction
of either solver. The [weighted-GCV paper](https://www.nist.gov/publications/weighted-gcv-method-lanczos-hybrid-regularization)
also explains why a parameter-selection rule must be checked in its projected
setting. This prototype uses ordinary full-data GCV, not that paper's adaptive
weighted-GCV algorithm. Their published results do not validate this variant.

## Physical controls

Every geometry uses an independent FP64 exact-trilinear matrix on an 8×7×6 grid,
five irregular angles, and a shifted 9×7 detector. The unequal-spacing case uses
0.8/1.2/1.4 voxel spacing; laminography tilts the scan axis by 30°. All 336 image
coefficients are free. Objects contain smooth or sharp structure. Noisy data add
1% projection-RMS Gaussian noise with the same fixed seed policy across cases.
These are small matched-model qualification controls, not the independently
generated continuous-object acceptance fixtures.

All three methods run 64 steps with full reorthogonalization: ordinary Krylov
with no filter or regularization; filtered Krylov without regularization; and
filtered Krylov with GCV-selected image damping. The ordinary method is an
FP64 subspace reference, not a timing measurement of public FP32 CGLS.

| Geometry / object | Ordinary 8-step L2 | Hybrid 8-step L2 | Ordinary 64-step L2 | Filtered 64-step L2 | Hybrid 64-step L2 | GCV-selected step / L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lamino / smooth-clean | 0.340421 | 0.329050 | 0.234718 | 0.240090 | 0.240105 | 64 / 0.240105 |
| lamino / sharp-clean | 0.360312 | 0.347600 | 0.243206 | 0.243537 | 0.243556 | 64 / 0.243556 |
| lamino / smooth-noisy | 0.342425 | 0.331776 | 0.277252 | 0.281505 | 0.281350 | 64 / 0.281350 |
| lamino / sharp-noisy | 0.360019 | 0.347945 | 0.274582 | 0.277115 | 0.276935 | 64 / 0.276935 |
| anisotropic / smooth-clean | 0.126730 | 0.119227 | 0.087658 | 0.092063 | 0.092081 | 64 / 0.092081 |
| anisotropic / sharp-clean | 0.233245 | 0.228070 | 0.194361 | 0.201602 | 0.201619 | 64 / 0.201619 |
| anisotropic / smooth-noisy | 0.128002 | 0.121975 | 0.173019 | 0.163768 | 0.162094 | 42 / 0.147120 |
| anisotropic / sharp-noisy | 0.236180 | 0.229938 | 0.252174 | 0.250284 | 0.249202 | 41 / 0.238818 |
| parallel / smooth-clean | 0.126030 | 0.121813 | 0.095947 | 0.097902 | 0.097908 | 64 / 0.097908 |
| parallel / sharp-clean | 0.278448 | 0.273643 | 0.255717 | 0.255825 | 0.255825 | 63 / 0.255828 |
| parallel / smooth-noisy | 0.128064 | 0.124164 | 0.196712 | 0.218165 | 0.214528 | 21 / 0.110844 |
| parallel / sharp-noisy | 0.279965 | 0.274952 | 0.294190 | 0.295521 | 0.294166 | 39 / 0.277488 |

The earlier improvements at step eight are not accepted-result speed evidence.
GCV's selected iteration is a retrospective diagnostic over the computed
sequence; the prototype did not stop there or avoid the later computation.
Conditioning, regularization bias and data-driven stopping remain coupled.
This screen does not prove that every filtered or hybrid method will fail.

## Storage preflight

The prototype retains k image directions and k projected directions. For k=64,
even FP32 storage requires at least `64 * 4 * (image_voxels + projection_values)`
bytes, before the object, input data, orthogonalization temporaries, FFT workspace,
solver libraries, or allocator overhead. The calculations below apply to every
smooth, sharp and noisy suite cell of each geometry/size. They are array-storage
lower bounds, not measured peak process GPU memory or comparisons with ASTRA.

| Geometry / nominal size, 180 views | Basis-only GiB |
| --- | ---: |
| parallel-64-180 | 0.238 |
| anisotropic-64-180 | 0.133 |
| lamino-64-180 | 0.238 |
| parallel-128-180 | 1.203 |
| anisotropic-128-180 | 0.627 |
| lamino-128-180 | 1.203 |
| parallel-256-180 | 6.812 |
| anisotropic-256-180 | 3.444 |
| lamino-256-180 | 6.812 |

At 512³ / 720 views, those two FP32 bases alone require
**77.0 GiB**. No production implementation of this full-storage
prototype can satisfy the 8 GB showcase constraint. A different algorithm for
bounded storage would require its own accuracy and convergence qualification;
changing a storage constant does not establish that it preserves these iterates.

For the six alignment sizes, the corresponding hypothetical joint image/pose
basis lower bounds are listed below. The coupled nonlinear alignment path was
not implemented or run, so these sizes are not recovery evidence.

| Alignment cell | Basis-only MiB |
| --- | ---: |
| lamino-clean | 23.324 |
| lamino-noisy | 23.324 |
| anisotropic-clean | 14.169 |
| anisotropic-noisy | 14.169 |
| parallel-clean | 23.324 |
| parallel-noisy | 23.324 |

## Reproduction and retained evidence

The [compressed qualification record](../../bench/reference/hybrid-krylov-qualification-2026-10-04.json.gz)
contains all 2,304 physical iterate records, all three reference methods,
the exact prototype and qualification source, algebra checks, storage calculations,
and every unperformed frozen cell. The existing cold-fastest external workflow
is retained for each reconstruction cell; it was not rerun and no speed or memory
ratio is computed against this CPU qualification.

The scripts are one-off research artifacts under the ignored `.artifacts/`
directory. No benchmark framework, public API, library defaults, acceptance
threshold, or goal denominator changed. The previous 26/27 reconstruction and
5/6 public alignment outcomes remain authoritative; the stretch goals remain open.
## Subsequent physical-gradient GCV screen

A separate small experiment used an explicit physical-gradient penalty and a
direct generalized-eigenvalue solve, without the rejected filtered Krylov basis.
It selected one weight from the measured data using full-data GCV, with identical
bounds across tilted, anisotropic and parallel scans. Twenty-four controls covered
smooth/sharp objects, matched/independent continuous-object data, and clean/noisy
measurements. Independent dense solves and influence-matrix score checks passed.

The candidate improved image error in only **20/24 controls** against a
GCV-stopped, unregularized Krylov control. The four regressions were independent
tilted smooth clean, independent tilted sharp clean/noisy, and matched anisotropic
sharp clean. In the independent tilted sharp controls, candidate relative L2
errors were 18.96 and 19.63, versus 1.11 and 1.11 for the early-stopped control.
Both methods performed poorly there; comparing only with a fully converged,
overfitted least-squares solution would conceal the candidate's failure.

This variant is rejected without per-case weight retuning or public-workflow
integration. No frozen acceptance result or timing score changes. The
[complete screen and sources](../../bench/reference/gradient-gcv-qualification-2026-10-04.json.gz)
retain all successes, failures and selection settings.
