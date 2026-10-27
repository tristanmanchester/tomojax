# Measurements

This page holds the measured evidence behind the speed, accuracy and memory
claims in the guides. Each section names the date, the TomoJAX commit the
numbers were measured on and the protocol. The code has changed since some of
those commits, so a later commit can be faster or slower than the figures here.
Failed cases are reported with the rest.

## Test environment

Unless a section says otherwise, measurements use one NVIDIA GeForce RTX 4070
Laptop GPU (Ada, 8 GB) on Linux with Python 3.12, JAX 0.11.2, ASTRA 2.5 and
TIGRE 3.1.3. The several-GPU results use four H100 80 GB GPUs in one machine on
Modal.
Objects are synthetic except the FIPS walnuts, a real lab
cone-beam scan.

*Cold* time is a fresh process through the first accepted result, including
imports, compilation and transfers. *Warm* time repeats the same call in that
process. GPU memory is the process's peak, sampled every 10 ms; sampling can
miss short peaks, and it includes runtime and allocator overhead.

Commands, adapters and physical conventions are in the
[benchmark guide](../bench/README.md). Raw records are kept out of the
repository; each result names the command, commit and date to rerun it.

## Raw NXtomo loading and held-out split integrity

Measured 2026-10-27 on `a4a969cb` (loading) and `2626e6e5` (evaluation),
using the local Diamond files. These are functional/integrity tests, not image
or pose accuracy measurements. Commands in the lab shared directory are
`JAX_PLATFORMS=cpu uv run --no-sync python shared/agent02/loader_kill.py`
(the metadata/pixel section) and
`JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 uv run --no-sync python
shared/agent02/evaluation_kill.py`.

- I13 NXtomo scans 159791 and 159815 classify respectively 37/80/80 and
  3001/200/100 sample/flat/dark frames. First-sample row1080 columns441:449
  exactly match raw HDF5 pixels (maximum integer error 0). Discovery cold/warm
  times are 55.7/4.75 ms and 46.6/15.2 ms, with 260 MiB peak process RSS and no
  GPU allocation. Missing physical pitch requires explicit detector geometry.
- I12 physical row1080, every thirtieth sample view (61×1×2560), tests
  interleaved-view, angular-block and detector-block splits. Changing every
  excluded value to 1234 leaves a fixed four-coefficient least-squares fit
  bitwise unchanged. Streamed versus dense squared-error relative differences
  are 1.96e-16, 0 and 3.00e-16. Cold/warm fit-and-score times are 15.29/13.51,
  13.00/13.25 and 9.95/8.46 ms; peak RSS 69.37 MiB, GPU 0. This checks held-out
  accounting, not tomographic reconstruction or stopping-rule effectiveness.

The [held-out guide](held-out-evaluation.md) describes the split and leakage
contract. Public reconstruction/alignment workflows do not automatically enforce
arbitrary detector observation masks.

## Real-scan comparison criteria

Use the same calibrated geometry and physical output grid for each library,
and compare iterative methods at matched reconstruction error rather than a
fixed iteration count. Nikon and Zeiss comparisons use data residuals from one
common calibrated ASTRA forward projector for all libraries, together with
volume differences. Compare iterative times at matched residual; compare FDK
times with both diagnostics reported. Physical accuracy checks use the walnut
reference and independent analytic phantoms. The Zeiss vendor comparison uses
a fixed registered valid-data region and an intensity-affine fit as an image
diagnostic, not as a truth volume or an acceptance threshold.

## Resident cone volume storage

Measured 2026-10-27–28 on Omarchy, RTX 4070 Laptop 8 GB, under concurrent
CPU load. The control is `c7d8578d`, with public `(x, y, z)` volumes. The
candidate commits below use public `(z, y, x)` volumes and private
z-contiguous resident iterates. All original views and calibrated grids are
retained: Walnut one/three orbits at binning 2, Nikon and Zeiss at binning 2,
and both CuO scans at binning 4. Every solve starts from FDK. CGLS uses three
iterations; ordered-subset rows use one sweep with the subset counts below.

These are **T3 pipeline agreements, not reconstruction accuracy**. Image
differences compare physical voxels after axis permutation, without fitting
scale or alignment. The maximum relative peak difference over these pairs is
5.18e-7. Walnut-support agreement with the authors' reconstruction is unchanged
to the shown precision: CGLS 0.087821/0.061075, OS-SART 0.083891/0.055488,
and OSEM 0.071995/0.048422 for one/three orbits. No views are held out.

Each row uses two fresh processes per implementation, interleaved in one GPU
job. Cold includes loading, numerical imports, compilation, FDK, solving and
host output; warm repeats the complete reconstruction including FDK, but not
loading. Times average the two process cold times or warm medians. Scoring
and saving comparisons are outside timing and the reported memory boundary.
GPU peaks use 20 ms sampling; host peaks are process RSS high-water marks
before scoring. Memory columns give the maximum over the two processes,
control / candidate. Filesystem caches are uncontrolled.

| Scan, method | Control cold / warm (s) | Candidate cold / warm (s) | Image relative L2 | GPU peak MiB | Host peak GB | Candidate commit |
|---|---:|---:|---:|---:|---:|---|
| walnut1, cgls | 22.85 / 14.76 | 22.12 / 14.69 | 6.69e-08 | 4308 / 4308 | 5.757 / 5.757 | `11833821` |
| nikon, cgls | 29.61 / 22.95 | 28.12 / 22.51 | 4.09e-08 | 4574 / 4574 | 4.253 / 4.216 | `e2f8010f` |
| zeiss, cgls | 15.23 / 10.03 | 15.27 / 9.79 | 0 | 4576 / 4576 | 3.537 / 3.532 | `e2f8010f` |
| walnut3, cgls | 60.63 / 44.24 | 61.03 / 43.36 | 7.53e-08 | 4312 / 4312 | 7.539 / 7.537 | `11833821` |
| cuo-spl1, cgls | 64.66 / 38.86 | 62.91 / 37.39 | 0 | 4576 / 4576 | 6.194 / 6.193 | `11833821` |
| cuo2, cgls | 63.14 / 38.72 | 61.62 / 36.84 | 1.92e-08 | 4576 / 4717 | 6.176 / 6.171 | `e2f8010f` |
| walnut1, ossart | 16.51 / 8.50 | 15.92 / 8.40 | 1.82e-08 | 6152 / 6152 | 5.755 / 5.756 | `e2f8010f` |
| walnut1, osem | 16.16 / 7.59 | 17.72 / 7.57 | 2.19e-07 | 4282 / 4282 | 5.755 / 5.756 | `11833821` |
| nikon, ossart | 18.35 / 11.45 | 17.05 / 11.73 | 1.86e-08 | 3606 / 3606 | 4.181 / 4.154 | `11833821` |
| zeiss, ossart | 12.45 / 6.88 | 12.09 / 6.37 | 0 | 6204 / 6204 | 4.124 / 4.121 | `bd49f888` |
| zeiss, osem | 10.55 / 4.73 | 10.22 / 4.73 | 0 | 4546 / 4546 | 4.917 / 4.918 | `bd49f888` |
| walnut3, ossart | 40.69 / 25.04 | 44.18 / 23.79 | 1.95e-08 | 3318 / 3318 | 7.538 / 7.539 | `bd49f888` |
| walnut3, osem | 42.55 / 22.38 | 35.83 / 21.87 | 2.21e-07 | 3266 / 3266 | 7.885 / 7.880 | `bd49f888` |

The Nikon OS-SART pairs in the table are slower warm by 0.73% and 4.32%.
A balanced repeat on `e74bd197` (2026-10-28, three warm calls per process,
`shared/agent01/nikon_repeat.sh`) gives 11.162/11.270 s for the control and
11.110/11.124 s for the candidate: 0.47% and 1.29% faster. Cold times including
loading are 16.971/16.810 s and 16.184/17.262 s respectively. The image relative
L2 difference is at most 1.86e-8, both GPU peaks are 3606 MiB, and maximum
host RSS is 4.186 GB for both. These results show run-to-run timing variation;
they do not establish a uniform cold speedup. Walnut-one OSEM and Zeiss OSEM
also have mixed-sign warm timing differences.

One CuO2 candidate process in the table reaches 4717 MiB rather than 4576 MiB.
An instrumented four-process repeat on `e74bd197` (`shared/agent01/cuo_peak.sh`,
2026-10-28) gives 4576 MiB globally and 4566 MiB per compute process for every
arm, with identical JAX peak live allocation (3.712 GB) and pool (4.563 GB).
The extra 141 MiB in the table has no retained per-process trace and remains
unattributed. Prescoring host RSS spans 6.156–6.168 GB. Instrumentation includes
corrected-input hashing, so its cold times are not a loading benchmark.

CuO2 fresh-process comparisons can differ by 2.83e-6 relative L2 because
preprocessing uses sequential or balanced four-pixel sums during binning.
With identical pre-binning logarithms, those two sum trees reproduce every
corrected pixel and the differing FDK/final-image hashes; the solver is not
the first divergence. This is T3 arithmetic agreement, not an accuracy result.
The attribution commands are `shared/agent04-preprocess/run_tree.sh` and
`run_fingerprint.sh` on `c7d8578d` (2026-10-28). The instrumented repeat gives
the same 2.83e-6 difference between two control processes whose binning trees
differ, as between control and candidate.
The executable protocol is `bench/cone_matrix.py`; set the input environment
variables to the scans named in the full-view sections below. CGLS uses
`--iterations 3`; all ordered-subset cases use `--iterations 1`. Other options
are unchanged defaults, including FDK initialization and recomputed
normalization:

```bash
python bench/cone_matrix.py "$CASE" "$INPUT" --library tomojax \
  --method "$METHOD" --iterations "$ITERATIONS" --subsets "$SUBSETS" \
  --repeats "$REPEATS" --output result.json
```

Use `CASE=walnut1/walnut3/nikon/zeiss/cuo` and the corresponding path; both
CuO inputs use `CASE=cuo`. OS-SART subset counts are 12/17/13/13 for
Walnut-one/Walnut-three/Nikon/Zeiss; OSEM counts are 12/13/1 for
Walnut-one/Walnut-three/Zeiss. The lab pairing commands are
`shared/agent01/all13.sh 0 4 0,3`, `all13.sh 4 9 4,7,8`, and
`all13.sh 9 13`, with the corresponding original command list in
`shared/baseline/cmds.txt`. These use two warm CGLS repeats for Walnut-one
and one elsewhere. `shared/agent04-transfer/run.sh` measures Nikon CGLS
(two warm repeats) and CuO2 CGLS (one), control/candidate/control/candidate.
`shared/agent01/balanced_pair.sh` measures Walnut-one OS-SART and Zeiss CGLS
with three warm repeats, candidate/control/control/candidate. The all13
pairing uses control first on even command indices and candidate first on
odd indices, repeating that order twice. The drivers perform canonical
voxel comparisons and delete scratch volumes after scoring.

## Detector response and structure controls

Measured 2026-10-27 on `e71f900f` for response injections and `271fd6e3` for
independently scored structure increments. The `Rings` implementation is the
same in both. These are T1 known perturbations of real line-integral data,
not reconstruction accuracy or ring-energy rankings. All selected views are
used; there is no held-out split, fitted registration or pose-gauge removal.
Explicit detector centres are −5/−138/−15/0 pixels in table order; ESRF uses
the panel centre, not an independently calibrated axis.

| Scan and frozen cut | Injected-field residual, Rings | Injected-field residual, Algotom | Off-axis structure alteration, Rings | Off-axis structure alteration, Algotom |
|---|---:|---:|---:|---:|
| I13 159815, row 1080, columns 441:1869, every eighth of 3000 views | 0.2803 | 0.4033 | 0.1112 | 0.3501 |
| I12 3339629, row 1080, all 2560 columns, every eighth view | 0.3917 | 0.2406 | 0.0085 | 0.2998 |
| APS 2-BM `test_data.h5`, row 11, all 1536 columns, every eighth view | 0.6110 | 0.2302 | 0.0130 | 2.0735 |
| ESRF bamboo, row 270, all 640 columns, every eighth view | 0.4370 | 0.1696 | 0.0367 | 0.3529 |

Injected response is `a + b * (1 - exp(-q))`, with independent pixel fields,
RMS 0.004 and 0.006, seeds 608–611, and paired sparse support. These amplitudes
are declared perturbations, **not calibrated detector errors**. The field
residual is `RMS(C(q + error) - C(q)) / RMS(error)`: existing structure damage
common to both outputs cancels. Corresponding off-support ratios are
0.246/0.298/0.221/0.307 for Rings and 0.299/0.202/0.208/0.152 for Algotom.
Rings misses at least one of the frozen residual ≤0.5 and off-support ≤0.2
gates on every scan. Even/odd injection variants also miss those gates.

Structure increments are independent analytic Gaussian Radon traces, not
TomoJAX projections: widths 2 and 8 pixels, peak 0.05, radii 0.35 and 0.7
of detector half-width, azimuth seed 9027. The table gives the maximum
`norm(C(q + increment) - C(q) - increment) / norm(increment)` per scan.
Rings misses the 5% gate on I13; Algotom misses it on all four cuts. The I13
maximum remains 10.5% with all 3000 views. Axis-centred Gaussians and annuli
are reported separately as unidentifiable detector/object aliases, not as
failures to recover uniquely determined structure.

Even an off-axis Gaussian has a radial component. A follow-up physical-radial
projection on `09437d17` assigns 1.13–6.68%, 0.76–4.09%, 1.26–6.29% and
2.32–12.89% of the probes' energy to that component. Maximum error outside
the radial component, normalized by nonradial truth, is
0.04712/0.00601/0.01009/0.03771 for Rings. Thus the I13 full-increment failure
mostly concerns the radial prior, not its uniquely varying component. The
frozen full-increment gate above is not retrospectively changed.

Algotom 1.7 uses its canonical transmission-domain chain,
`-log(remove_all_stripe(exp(-q)))`, with unchanged defaults 3/51/21.
Commands, from this run's checkout with its external `shared/` harness:

```bash
JAX_PLATFORMS=cpu ALGO_DOMAIN=transmission LAB_ARTIFACTS=OUTPUT \
  uv run --no-sync python ../../shared/agent08/rings/injected_kill.py
JAX_PLATFORMS=cpu LAB_ARTIFACTS=OUTPUT \
  uv run --no-sync python ../../shared/agent02/ring_structure_score.py
JAX_PLATFORMS=cpu LAB_ARTIFACTS=OUTPUT \
  uv run --no-sync python ../../shared/agent02/i13_native_noharm.py
```

The response run's CPU first/warm correction times are 5.070/0.225 s,
1.220/0.141 s, 1.235/0.038 s and 0.990/0.028 s for the table's scan order;
these exclude imports and loading and do not follow the page's fresh-process protocol.
Peak host RSS is 1281 MiB and GPU use is zero. No speed comparison is claimed.
Final ring quality requires independent rows and seeds plus reconstruction
metrics; these perturbation controls do not establish removal of native rings.

### Response-inverse feasibility

On the I13 159815 runtime cut (every 25th sample view, rows 1080:1088,
columns 441:1873, count-binned 8×8), the former final response had no real
inverse for 166 observations. Its three Newton steps nevertheless returned
finite values spanning −8.77 to 9.11. The constrained fit on `694a9ae7`
changes seven of 179 final-stage coefficient pairs, gives a minimum response
derivative of 0.5011 and maximum forward/inverse closure error of 3.58e−7.
The final weighted data-fit SSE changes from 1325.891 to 1324.329; the first
fine stage increases by 0.00136%. These are numerical feasibility checks,
not reconstruction accuracy. The constraint changes the odd coefficient
components too: RMS 0.01785 for offset and 0.02860 for response slope.

```bash
JAX_PLATFORMS=cpu uv run --no-sync python ../../shared/agent08/rings/source_coarse.py
```

The four-scan finite-reference fixture and analytic structure controls retain
their previous pass/fail outcomes; the fit constraint repairs inversion, not
the statistical separation of detector errors and object structure.

CUDA A/B/A/B on the native I13 cut, baseline `c7d8578d` versus the constrained
implementation, gives median warm correction times 0.01403/0.01719 and
0.01312/0.01613 s: approximately **23% slower**, with sampled GPU peaks
704–706 versus 742 MiB (**about 36 MiB more**). Cold calls are 9.63–9.68
versus 9.73–9.92 s; peak host RSS is 1607 versus 1623–1628 MiB. On the same
runs, seed-622 dense log-gain residual is 0.919913 and the independent
Gaussian increment alteration is 0.104278 for both implementations; native
output disagreement is at most 7.62e−8 relative. These controls still fail
their quality gates. The correctness repair is not a speed improvement.

```bash
LAB_ARTIFACTS=OUTPUT uv run --no-sync python ../../shared/agent08/rings/paired_gpu.py
```

## Reconstruction against ASTRA

Measured 2026-10-05 on commit `6fa1585`. The matrix has 27 cells: Gaussian,
sharp (structured) and noisy structured objects, each scanned in parallel,
shifted anisotropic and 30° laminography geometry on 64³, 128³ and 256³ grids,
all with 180 views. Each method runs at budgets of 1, 2, 4, … 256 iterations
until its full-volume relative L2 error passes the cell's fixed gate, without
fitted amplitudes or clipping. Each side contributes its fastest accepted
workflow by cold time. Cold time runs through the first accepted budget,
including the failed lower budgets; warm time is the median of three repeats
of that budget. ASTRA was measured in the same session. TIGRE was not rerun,
because it was the fastest external method in no cell of the earlier
matrices.

| Speedup (ASTRA time / TomoJAX time), 26 paired cells | Cold | Warm |
|---|---:|---:|
| Geometric mean | 1.00 | 2.78 |
| Range | 0.44–2.81 | 0.59–20.4 |

The sharp anisotropic 64³ cell has no accepted result on either side. In the
parallel and anisotropic cells TomoJAX's fastest method is the opt-in Fourier
inverse on CUDA (`fourier_reconstruct` with CuPy), not the default FBP. In
laminography it is Joseph CGLS: plain, FBP-started or coarse-to-fine. Small
laminography scans are slower from a fresh process (0.44–0.59). There, JAX's
import and device setup (about 0.4 s) and tracing the solver (about 0.2 s)
outweigh solves of 30–180 ms.

| Object | Cell | TomoJAX fastest accepted, cold / warm ms | ASTRA fastest accepted, cold / warm ms | Cold speedup | Warm speedup |
|---|---|---|---|---:|---:|
| Gaussian | parallel 64 | Fourier: 368 / 3.0 | fbp2d: 334 / 24.6 | 0.91 | 8.18 |
| Gaussian | anisotropic 64 | Fourier: 372 / 1.5 | CGLS: 383 / 30.1 | 1.03 | 20.41 |
| Gaussian | laminography 64 | coarse-to-fine CGLS: 1777 / 239.0 | CGLS: 1728 / 676.4 | 0.97 | 2.83 |
| Gaussian | parallel 128 | Fourier: 417 / 11.2 | fbp2d: 391 / 59.8 | 0.94 | 5.32 |
| Gaussian | anisotropic 128 | Fourier: 394 / 8.1 | fbp3d (CuPy): 454 / 10.4 | 1.15 | 1.29 |
| Gaussian | laminography 128 | coarse-to-fine CGLS: 2942 / 763.5 | CGLS: 7066 / 3200.5 | 2.40 | 4.19 |
| Gaussian | parallel 256 | Fourier: 593 / 74.6 | fbp2d: 697 / 297.5 | 1.17 | 3.99 |
| Gaussian | anisotropic 256 | Fourier: 507 / 39.0 | fbp3d (CuPy): 539 / 77.5 | 1.06 | 1.99 |
| Gaussian | laminography 256 | coarse-to-fine CGLS: 14077 / 5572.4 | CGLS: 39612 / 18048.9 | 2.81 | 3.24 |
| Sharp | parallel 64 | Fourier: 422 / 3.6 | fbp2d: 367 / 28.4 | 0.87 | 7.85 |
| Sharp | anisotropic 64 | none accepted | none accepted | — | — |
| Sharp | laminography 64 | CGLS: 912 / 28.7 | CGLS: 405 / 29.2 | 0.44 | 1.02 |
| Sharp | parallel 128 | Fourier: 513 / 12.0 | fbp2d: 439 / 61.8 | 0.86 | 5.14 |
| Sharp | anisotropic 128 | Fourier: 416 / 8.0 | fbp3d (CuPy): 474 / 10.6 | 1.14 | 1.33 |
| Sharp | laminography 128 | CGLS: 1197 / 178.4 | FBP-started CGLS: 711 / 105.8 | 0.59 | 0.59 |
| Sharp | parallel 256 | Fourier: 614 / 69.5 | fbp2d: 678 / 308.0 | 1.10 | 4.43 |
| Sharp | anisotropic 256 | Fourier: 533 / 39.6 | fbp3d (CuPy): 555 / 76.5 | 1.04 | 1.93 |
| Sharp | laminography 256 | FBP-started CGLS: 2108 / 647.4 | CGLS: 3225 / 982.5 | 1.53 | 1.52 |
| Noisy | parallel 64 | Fourier: 404 / 3.3 | fbp2d: 394 / 26.2 | 0.98 | 7.99 |
| Noisy | anisotropic 64 | Fourier: 391 / 1.6 | FBP-started CGLS: 448 / 8.5 | 1.15 | 5.29 |
| Noisy | laminography 64 | CGLS: 861 / 29.2 | CGLS: 407 / 29.3 | 0.47 | 1.00 |
| Noisy | parallel 128 | Fourier: 504 / 11.4 | fbp2d: 396 / 62.2 | 0.79 | 5.46 |
| Noisy | anisotropic 128 | Fourier: 406 / 8.1 | fbp3d (CuPy): 450 / 10.0 | 1.11 | 1.24 |
| Noisy | laminography 128 | CGLS: 1043 / 104.2 | CGLS: 588 / 98.6 | 0.56 | 0.95 |
| Noisy | parallel 256 | Fourier: 593 / 70.4 | fbp2d: 678 / 298.7 | 1.14 | 4.24 |
| Noisy | anisotropic 256 | Fourier: 545 / 41.7 | fbp3d (CuPy): 532 / 77.9 | 0.98 | 1.87 |
| Noisy | laminography 256 | FBP-started CGLS: 2099 / 645.0 | CGLS: 2277 / 736.3 | 1.08 | 1.14 |

## TV-regularised reconstruction against TIGRE

Measured 2026-10-05 on commit `7dbf427` with
[`bench/compare_tv.py`](../bench/compare_tv.py). A 128³ structured parallel
phantom is scanned with 180 views and 3% noise. Each library runs 50
iterations over its own grid of regularisation weights, and the best relative
L2 error is reported with its warm time. TIGRE's times include its host
transfers.

| Method | Best relative L2 error | Warm time |
|---|---:|---:|
| TomoJAX FISTA-TV | 0.076 | 1.13 s |
| TIGRE FISTA | 0.101 | 9.7 s |
| TIGRE ASD-POCS | 0.146 | 5.7 s |

Larger TIGRE FISTA weights diverge. Unregularised CGLS reaches 0.138 at best
(10 iterations). This is one phantom at one noise level, not a general ranking.

## Cone-beam projection and FDK

Measured 2026-10-07 on commit `5ea7c22` with
[`bench/compare_cone.py`](../bench/compare_cone.py): five voxelised ellipsoids
(256³, 2×2×2 sub-samples) on a circular 360-view scan with a 384² detector at
1.5× magnification. Forward projections are compared with exact analytic line
integrals and FDK volumes with the phantom. Times are the best of three warm
calls; ASTRA and TIGRE take host arrays.

| Operation | TomoJAX | ASTRA 2.5 | TIGRE |
|---|---:|---:|---:|
| Forward projection | 0.093 s (error 0.0060) | 0.170 s (0.0060) | 0.477 s interpolated (0.0062), 1.24 s Siddon (0.0068) |
| Backprojection | 0.178 s, exact transpose | 0.075 s, approximate transpose | 0.103 s, "matched" |
| FDK | 0.048 s (error 0.0581) | 0.293 s (0.0581) | 0.513 s (0.0581) |
| Forward, axis tilted 0.5° | 0.092 s | 0.171 s | |
| Backprojection, axis tilted 0.5° | 0.232 s, exact transpose | 0.075 s, approximate transpose | |

TomoJAX's backprojection is the exact transpose of its forward projector,
which the iterative solvers and gradients rely on. ASTRA's voxel-driven
backprojector is cheaper but only approximately its transpose, so a CGLS
iteration (one of each) costs about 0.27 s in TomoJAX against ASTRA's 0.25 s.
The transpose has a faster path for views whose detector columns are parallel
to the rotation axis. A tilted axis, a rolled or pitched detector, or the
per-view poses of alignment rule that path out, and an iteration then costs
about 0.32 s at this size. ASTRA's times do not depend on the geometry.

On the same scan (2026-10-06, commit `dc411a2`, before the forward projector
was made faster), CGLS with the matched transpose took 3.4, 6.6 and 13.2 s for
10, 20 and 40 iterations, with volume errors of 0.090, 0.043 and 0.048.
ASTRA's `CGLS3D_CUDA` took 2.3, 4.4 and 8.7 s with errors of 0.088, 0.040 and
0.101. It diverges after about 20 iterations: its own data residual rises
from 0.0053 at 30 iterations to 0.0062 at 40 and 0.027 at 60, which CGLS with a
matched transpose cannot do.

On the binned FIPS walnut (2026-10-11, commit `f9b2c5b`; 900 views from the
three orbits, 501³ voxels, error inside the walnut against the authors'
published reconstruction, warm times including transfers) neither CGLS
diverges within 50 iterations, and ASTRA's is faster at equal error. From
zero, TomoJAX's reaches 0.040 at 25 iterations in 77 s and ASTRA's 0.043 at 35
in 72 s; ASTRA's then rises to 0.047 at 50. Started from the mean of the
orbits' FDK volumes, TomoJAX's reaches 0.046 in 15 iterations and 49 s, ASTRA's
0.047 in 18 and 39 s. The walnut's axis leans about half a degree, which rules
out the faster transpose path above, so each TomoJAX iteration costs more than
ASTRA's.

`fdk_host` reconstructed a 1024³ scan (1024 views of 1024² pixels, 4.3 GB each
way) in z-slabs in 8.0 s with the projections and volume in RAM, 11.1 s on the
first call (2026-10-07, commit `95f0530`). ASTRA's `FDK_CUDA` took 20.3 s and
TIGRE's `fdk` 30.7 s from arrays in RAM.

### Full-view walnut SIRT

Measured 2026-10-19 on `20f4a966` (TomoJAX), `9a743fc0` (ASTRA one orbit),
`e280d938` (ASTRA three orbits) and `e3bc2f63` (TIGRE). Walnut 1 uses all
1200 views of orbit 2 or all 3600 views of the three orbits, binned 2 × 2 onto
a 486 × 384 detector, and a 501³ grid with 0.1 mm voxels. Load the projections
with `bench/walnut.py`'s `load_orbits(walnut_path, [2], 1, binning=2)` or
`load_orbits(walnut_path, [1, 2, 3], 1, binning=2)` and use
`volume_geometry(10)` with `tj.Scan.from_astra`. The TomoJAX call is:

```python
from tomojax.recon import SIRTConfig

result = tj.reconstruct(
    scan, "sirt", iterations=12, warm_start=True,
    config=SIRTConfig(lower_bound=0, subsets=1, acceleration=True),
)
volume = np.asarray(result.volume)
```

Each solve includes its library's FDK initializer. TomoJAX uses 12 iterations
with `SIRTConfig(lower_bound=0, acceleration=True)`; ASTRA and TIGRE use 24
nonnegative SIRT iterations. The one-orbit ASTRA call is `SIRT3D_CUDA`; the three-orbit
call uses the same recurrence with ASTRA forward/backprojection in 512-view
batches. Cold includes imports, loading and the first solve with fresh
compilation caches; warm repeats the solve in that process. The filesystem
cache is not cleared.

| Orbits | Library | Cold / warm (s) | Common-projector relative residual | Full / walnut-support relative L2 error |
|---|---|---:|---:|---:|
| One | TomoJAX | 58.02 / 53.47 | 0.024742 | 0.106162 / 0.074939 |
| One | ASTRA | 70.36 / 67.27 | 0.025998 | 0.109991 / 0.077235 |
| One | TIGRE | 364.27 / 363.06 | 0.044170 | 0.142517 / 0.098411 |
| Three | TomoJAX | 179.07 / 174.22 | 0.026222 | 0.067624 / 0.045780 |
| Three | ASTRA | 237.79 / 234.35 | 0.027558 | 0.072746 / 0.049110 |
| Three | TIGRE | 1171.20 / 1120.54 | 0.045232 | 0.113725 / 0.076657 |

The residual uses one calibrated ASTRA forward projector for all three libraries.
Both image errors use the authors' reference, without intensity fitting.

### Full-view Nikon and Zeiss SIRT

Measured 2026-10-19 on `272684c7` (TomoJAX and Zeiss ASTRA) and `35f4055f`
(Nikon ASTRA), with TIGRE on `e3bc2f63` (Nikon) and `6149357d` (Zeiss).
Nikon uses all 1571 views and a 500³ grid with 0.1880371266 mm
voxels; Zeiss uses all 801 views and a 512³ grid with 0.0833853988647461 mm
voxels. The input is
`tj.load_frames(path, darks=0, grid=grid).corrected(binning=2)`, where `path`
is the Nikon seeds XTekCT file or the Zeiss HDtomo `tomo-A.txrm` file and
`grid` is `tj.Grid(n, n, n, pitch, pitch, pitch)` with those dimensions.
Each solve starts from its library's FDK. TomoJAX uses the same 12-iteration
accelerated SIRT call above; ASTRA uses 24 nonnegative iterations with its
forward/backprojectors in 512-view batches. TIGRE uses
`tigre.algorithms.sirt(data, geo, angles, 24, init=initial, verbose=False)`,
with `initial = tigre.algorithms.fdk(data, geo, angles, dowang=False)`.

| Scan | Library | Cold / warm (s) | Common-projector relative residual | Volume difference to ASTRA |
|---|---|---:|---:|---:|
| Nikon | TomoJAX | 95.52 / 89.63 | 0.028788 | 0.013977 |
| Nikon | ASTRA | 135.89 / 125.78 | 0.030057 | — |
| Nikon | TIGRE | 504.23 / 493.43 | 0.056112 | 0.102748 |
| Zeiss | TomoJAX | 43.86 / 36.15 | 0.187915 | 0.025439 |
| Zeiss | ASTRA | 69.11 / 62.76 | 0.189497 | — |
| Zeiss | TIGRE | 291.57 / 290.02 | 0.207728 | 0.137740 |

Cold includes imports, loading and the first host volume with fresh compilation
caches; warm repeats the solve. Filesystem caches are not cleared. Loading took
3.03 / 10.04 / 3.02 s for Nikon TomoJAX / ASTRA / TIGRE, and
4.99 / 5.77 / 2.05 s for Zeiss.
The same calibrated ASTRA forward projector scores all three libraries. Volume
differences are full-volume relative L2, without intensity fitting. On the
fixed Zeiss vendor region, intensity-affine NRMSE is 0.353989 for TomoJAX,
0.351702 for ASTRA and 0.343797 for TIGRE; this is an image diagnostic,
not an acceptance gate.

### CuO FDK at fourfold binning

Measured 2026-10-20 on `75435039`. Both Zeiss CuO scans use all 3201 views
with 4 × 4 detector binning: spl1 has a 509³ grid and CuO2 a 507³ grid.
Load `frames = tj.load_frames(path, darks=0)`, then
`scan = frames.corrected(binning=4)` and call
`tj.reconstruct(scan, "fbp", grid=frames.geometry.grid.resampled((n, n, n)))`
with the corresponding grid size. ASTRA uses `FDK_CUDA`; TIGRE uses
`tigre.algorithms.fdk(data, geo, angles, dowang=False)`, with the same calibrated
geometry. No library changes its geometry or input to match another image.

| Scan | Library | Cold / warm (s) | Common-projector relative residual | Full-volume difference to ASTRA |
|---|---|---:|---:|---:|
| spl1 | TomoJAX | 33.52 / 3.76 | 0.057908 | 0.126512 |
| spl1 | ASTRA | 32.78 / 4.74 | 0.104623 | — |
| spl1 | TIGRE | 38.01 / 11.59 | 0.104608 | 0.000039 |
| CuO2 | TomoJAX | 39.70 / 4.12 | 0.078255 | 0.126404 |
| CuO2 | ASTRA | 53.71 / 10.31 | 0.120631 | — |
| CuO2 | TIGRE | 50.14 / 12.38 | 0.120616 | 0.000039 |

Cold includes loading and the first host image in a fresh process with fresh
compilation caches; warm repeats the solve. Filesystem caches are not cleared.
Loading and setup take 25.13 / 27.49 / 23.25 s for spl1 TomoJAX / ASTRA / TIGRE,
and 30.13 / 44.41 / 29.63 s for CuO2. TomoJAX's spl1 cold workflow is slower
than ASTRA's despite its faster warm solve.

All views use the same calibrated ASTRA residual. Full-volume differences are
relative L2 without intensity fitting, not physical error. On spl1, differences
in virtual padding, filter discretization and full-turn coverage weights account
for the difference between TomoJAX and ASTRA. The fixed vendor comparison uses
501 complete four-plane blocks, covering 2004 native planes, and one global
affine intensity fit: NRMSE 0.154591 / 0.154654 / 0.154651 for TomoJAX / ASTRA /
TIGRE. This is an image diagnostic, not ground truth. CuO2 has no vendor volume.
On the full spl1 grid, TomoJAX–ASTRA relative L2 is 0.005718 within the
vendor-defined sample mask, 0.007387 inside the horizontal cylinder seen by
all views and 0.619622 outside it; 99.67% of the squared difference is outside.
The sample mask includes pores and is independent of both library images.

### Full-view Nikon OSEM

Measured 2026-10-19 on `44170fc0` (TomoJAX and TIGRE) and `20f4a966`
(ASTRA). All 1571 views of the Nikon seeds scan use the same load call,
2 × 2 detector binning and 500³ physical grid as above. Solve inputs are
clipped at zero. The TomoJAX call is
`tj.reconstruct(scan, "mlem", iterations=8, warm_start=True,
config=MLEMConfig(subsets=4))`, with `MLEMConfig` from `tomojax.recon`.
Each outer iteration visits four ordered subsets. TomoJAX and TIGRE start
from FDK; ASTRA starts from a uniform positive volume. The ASTRA and TIGRE
adapters apply ordered-subset updates using their forward and backprojectors.

| Library | Outer iterations | Cold / warm (s) | Common-projector relative residual | Volume difference to ASTRA |
|---|---:|---:|---:|---:|
| TomoJAX | 8 | 100.71 / 94.96 | 0.026650 | 0.319973 |
| ASTRA | 48 | 260.40 / 256.31 | 0.026732 | — |
| TIGRE | 8 | 210.11 / 196.94 | 0.026848 | 0.313299 |

Cold includes imports, loading and the first host volume with fresh compilation
caches; warm repeats the solve in that process. The filesystem cache is not
cleared. Loading took 3.48, 10.09 and 3.83 s respectively. Residuals use the
same calibrated ASTRA forward projector and the original corrected data,
including negative samples. Volume differences are full-volume relative L2,
without intensity fitting. There is no physical truth volume for this scan;
these are image diagnostics, not reconstruction errors.

### Full-view CGLS

Measured 2026-10-20 on `e2cd2101`, with ASTRA 2.5.0. Walnut uses all 1200 views
of orbit 2, detector binning 2 × 2 and a 501³ grid of 0.1 mm voxels. Nikon and
Zeiss use the full-view inputs and physical grids above; the Nikon input is
`47209 testscan korn01_recon.xtekct`. Both libraries start from their own FDK.
TomoJAX uses three iterations and ASTRA four. ASTRA's native CGLS fits Walnut
and Zeiss; Nikon uses an explicit FP32 recurrence over ASTRA's projectors,
with GPU residuals, host search projections and 128-view batches.

| Scan | Library | Cold / warm (s) | Full / support image error, or common-projector fit | Volume difference to ASTRA |
|---|---|---:|---:|---:|
| Walnut, one orbit | TomoJAX | 27.48 / 18.84 | 0.184222 / 0.087821 | 0.013025 |
| Walnut, one orbit | ASTRA | 22.28 / 16.30 | 0.182273 / 0.086669 | — |
| Nikon | TomoJAX | 31.74 / 26.74 | 0.034928 | 0.027400 |
| Nikon | ASTRA | 33.74 / 26.14 | 0.029934 | — |
| Zeiss | TomoJAX | 17.87 / 12.24 | 0.163861 | 0.042137 |
| Zeiss | ASTRA | 16.53 / 12.90 | 0.155535 | — |

Each pair uses two fresh processes per library in ASTRA–TomoJAX–TomoJAX–ASTRA
order, with two warm solves per process. The table averages all four warm
samples. Cold includes numerical imports, loading and the first host volume
with fresh compilation caches; warm includes FDK, setup, transfers and the host
volume. Filesystem caches are not cleared. Scoring is outside the timer.

Walnut errors use the authors' reference, in the full volume and its fixed
support. Nikon and Zeiss match data fit, not physical reconstruction error:
all original corrected log projections are scored with one common calibrated
ASTRA forward projector. Full-volume differences use no alignment or intensity
fit. The fixed orthogonal sections retain the same structures; differences
include smooth background fields, support boundaries, edges and streaks.
TomoJAX is 15.6% slower warm on Walnut, 2.3% slower on Nikon and 5.1% faster on
Zeiss at these quality targets.

Set `WALNUT`, `NIKON` and `ZEISS` to those inputs. These commands configure one
fresh process each; run each pair in the order above, with separate output
filenames for the two processes:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python bench/cone_matrix.py walnut1 "$WALNUT" --library astra \
  --method cgls --iterations 4 --storage native --repeats 2 \
  --output bench/results/walnut1-astra.json
python bench/cone_matrix.py walnut1 "$WALNUT" --library tomojax \
  --method cgls --iterations 3 --repeats 2 \
  --output bench/results/walnut1-tomojax.json
python bench/cone_matrix.py nikon "$NIKON" --library astra \
  --method cgls --iterations 4 --storage device-residual --batch 128 --repeats 2 \
  --output bench/results/nikon-astra.json
python bench/cone_matrix.py nikon "$NIKON" --library tomojax \
  --method cgls --iterations 3 --repeats 2 \
  --output bench/results/nikon-tomojax.json
python bench/cone_matrix.py zeiss "$ZEISS" --library astra \
  --method cgls --iterations 4 --storage native --repeats 2 \
  --output bench/results/zeiss-astra.json
python bench/cone_matrix.py zeiss "$ZEISS" --library tomojax \
  --method cgls --iterations 3 --repeats 2 \
  --output bench/results/zeiss-tomojax.json
```

Add `--save-volume reference.npy` to the first ASTRA process and
`--compare-to reference.npy` to the TomoJAX processes to reproduce the image
differences, then delete the temporary volume.

#### Three-orbit walnut and CuO CGLS

Measured 2026-10-20 UTC, with two fresh processes per library in the same
interleaved order, one warm solve per process, and the same timing boundaries.
The table averages both warm samples. Three-orbit Walnut uses all 3600 views
on the 501³, 0.1 mm grid. Both CuO scans use all 3201 views, detector binning
4 × 4, and grids of 509³ at 4.51024 µm (spl1) and 507³ at 4.53458 µm (CuO2).
ASTRA uses its bounded device-residual recurrence with 128-view batches.

| Input, source commit | Library, iterations | Cold / warm (s) | Full / support image error, or common-projector fit | Volume difference to ASTRA |
|---|---|---:|---:|---:|
| Walnut, three orbits, `3ba80328` | TomoJAX, 3 | 79.06 / 59.25 | 0.149590 / 0.061075 | 0.010133 |
| Same | ASTRA, 4 | 74.56 / 60.26 | 0.146930 / 0.058154 | — |
| CuO-spl1, `ed9641e7` | TomoJAX, 3 | 74.87 / 44.94 | 0.028927 | 0.073933 |
| Same | ASTRA, 5 | 91.56 / 65.42 | 0.028078 | — |
| CuO2, `ed9641e7` | TomoJAX, 3 | 80.78 / 48.60 | 0.061258 | 0.074565 |
| Same | ASTRA, 5 | 95.80 / 66.05 | 0.060670 | — |

Walnut warm times are effectively tied: ASTRA's two samples are 58.98 and
61.54 s; TomoJAX's are 59.257 and 59.252 s. Mean loading/setup and first solve
take 17.26 + 61.79 s for TomoJAX and 16.07 + 58.49 s for ASTRA.
At the fixed CuO fit targets, TomoJAX is 31.3% faster warm on spl1 and 26.4%
on CuO2. These are data-fit comparisons, not physical image-error comparisons.
Fixed orthogonal sections retain the specimen shapes and pores; their largest
differences are smooth fields and acquisition-support boundaries.

| Input | Peak solver GPU MiB, TomoJAX / ASTRA | Peak solver host GB, TomoJAX / ASTRA |
|---|---:|---:|
| Walnut, three orbits | 4404 / 5720 | 7.398 / 8.074 |
| CuO-spl1 | 4704 / 6776 | 6.071 / 9.593 |
| CuO2 | 4704 / 6694 | 6.048 / 9.528 |

Set `CUO_SPL1` to `Tristan Manchester_Copper_oxyde_spl1.txrm` and `CUO2`
to `CuO_2_tomo-B.txrm`. Run each pair in ASTRA–TomoJAX–TomoJAX–ASTRA order
with distinct output names:

```bash
python bench/cone_matrix.py walnut3 "$WALNUT" --library astra \
  --method cgls --iterations 4 --storage device-residual --batch 128 --repeats 1 \
  --output bench/results/walnut3-astra-cgls.json
python bench/cone_matrix.py walnut3 "$WALNUT" --library tomojax \
  --method cgls --iterations 3 --repeats 1 \
  --output bench/results/walnut3-tomojax-cgls.json
python bench/cone_matrix.py cuo "$CUO_SPL1" --library astra \
  --method cgls --iterations 5 --storage device-residual --batch 128 --repeats 1 \
  --output bench/results/spl1-astra-cgls.json
python bench/cone_matrix.py cuo "$CUO_SPL1" --library tomojax \
  --method cgls --iterations 3 --repeats 1 \
  --output bench/results/spl1-tomojax-cgls.json
python bench/cone_matrix.py cuo "$CUO2" --library astra \
  --method cgls --iterations 5 --storage device-residual --batch 128 --repeats 1 \
  --output bench/results/cuo2-astra-cgls.json
python bench/cone_matrix.py cuo "$CUO2" --library tomojax \
  --method cgls --iterations 3 --repeats 1 \
  --output bench/results/cuo2-tomojax-cgls.json
```

#### CGLS work counts

Separate warm profiles on `3ba80328` (2026-10-20 UTC) locate the one-orbit
Walnut and Nikon gaps in the matched transpose. Kernel times below are
diagnostic; the paired, uninstrumented solves supply the wall times.
T/A denotes TomoJAX / ASTRA.

| Input | Forward T/A (s) | Iterative transpose T/A (s) | FDK backprojection T/A (s) | GPU copies T/A (s) |
|---|---:|---:|---:|---:|
| Walnut, one orbit | 4.410 / 6.557 | 13.419 / 7.013 | 0.827 / 1.291 | 0.355 / 0.955 |
| Nikon | 7.464 / 12.735 | 20.512 / 8.403 | 1.093 / 1.690 | 0.412 / 2.527 |

TomoJAX makes 76 / 100 forward callbacks and 76 / 100 adjoint callbacks,
respectively, with 4864 / 6400 launched view slots: 4800 / 6284 physical
views plus shifted-tail overlap. Its wide transpose takes 13.311 / 20.349 s
across 304 / 400 kernels. ASTRA has 190 / 250 iterative backprojection
launches; its bounded Nikon recurrence records 7855 forward and 7855
backprojected views. Native ASTRA Walnut view counts are not inferred from
kernel counts.

Each driver solve has an NVTX range `cone-solve-N`, with `N=0` for the first
solve. Capture a warm solve with Nsight Systems, retaining at least one repeat;
replace the example's driver arguments with those of the selected row:

```bash
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 nsys profile --trace=cuda,nvtx \
  --capture-range=nvtx --nvtx-capture=cone-solve-1 --capture-range-end=stop \
  --sample=none --cpuctxsw=none -o bench/results/cone-profile \
  python bench/cone_matrix.py walnut1 "$WALNUT" --library tomojax \
  --method cgls --iterations 3 --repeats 1 --output bench/results/profile.json
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_gpu_mem_time_sum \
  --filter-nvtx cone-solve-1 bench/results/cone-profile.nsys-rep
```

### One-orbit walnut ordered subsets

Measured 2026-10-20 on `e2cd2101`, with the same full-view one-orbit input and
501³ grid as CGLS above. Each solve starts from FDK and makes one pass through
its subsets. OS-SART is nonnegative; OSEM clips solve inputs at zero. ASTRA
computes each subset's column normalization when it is used.

| Method | Library and subsets | Cold / warm (s) | Full / walnut-support relative L2 error | Volume difference to ASTRA |
|---|---|---:|---:|---:|
| OS-SART | TomoJAX, 12 | 16.75 / 9.95 | 0.121031 / 0.083891 | 0.012251 |
| OS-SART | ASTRA, 16 | 12.29 / 8.38 | 0.116176 / 0.081108 | — |
| OSEM | TomoJAX, 12 | 14.41 / 8.50 | 0.097780 / 0.071995 | 0.009635 |
| OSEM | ASTRA, 14 | 10.32 / 6.52 | 0.094714 / 0.069006 | — |

These use the same interleaved fresh-process protocol, with one warm solve per
process: the reported warm time averages two samples per library. Author
errors and full-volume differences use no intensity fitting. TomoJAX is
18.8% slower warm for OS-SART and 30.4% slower for OSEM at these settings.
Both ASTRA settings have lower author errors than the TomoJAX settings.

```bash
python bench/cone_matrix.py walnut1 "$WALNUT" --library astra \
  --method ossart --iterations 1 --subsets 16 --norms recompute \
  --output bench/results/walnut1-astra-os.json
python bench/cone_matrix.py walnut1 "$WALNUT" --library tomojax \
  --method ossart --iterations 1 --subsets 12 \
  --output bench/results/walnut1-tomojax-os.json
python bench/cone_matrix.py walnut1 "$WALNUT" --library astra \
  --method osem --iterations 1 --subsets 14 --norms recompute \
  --output bench/results/walnut1-astra-osem.json
python bench/cone_matrix.py walnut1 "$WALNUT" --library tomojax \
  --method osem --iterations 1 --subsets 12 \
  --output bench/results/walnut1-tomojax-osem.json
```

### Full-view Nikon and Zeiss ordered subsets

Measured 2026-10-20 UTC on `ed9641e7`, using the full-view inputs and grids
above, one sweep from FDK, and ASTRA recomputed column normalization.
Two fresh processes per library run in ASTRA–TomoJAX–TomoJAX–ASTRA order,
with one warm repeat per process. Cold and warm use the CGLS timing boundaries.
All rows meet their fixed common-projector fit target: 0.0354 for Nikon OS,
0.1971 for Zeiss OS and 0.2238 for Zeiss OSEM.

| Input, method | Library, subsets | Cold / warm (s) | Common-projector fit | Volume difference to ASTRA |
|---|---|---:|---:|---:|
| Nikon OS | TomoJAX, 13 | 18.63 / 13.08 | 0.034627 | 0.005683 |
| Same | ASTRA, 12 | 16.08 / 12.34 | 0.035392 | — |
| Zeiss OS | TomoJAX, 13 | 13.98 / 7.72 | 0.196145 | 0.022196 |
| Same | ASTRA, 16 | 10.68 / 7.29 | 0.193721 | — |
| Zeiss OSEM | TomoJAX, 1 | 12.19 / 5.48 | 0.217301 | 0.122820 |
| Same | ASTRA, 2 | 8.76 / 5.04 | 0.209633 | — |

TomoJAX's warm gaps are 6.0%, 5.9% and 8.7%, respectively. The fixed sections
retain the same objects. OS differences include edges, background and streaks;
the larger OSEM difference is mainly object contrast, with TomoJAX's one-subset
result dimmer than ASTRA's two-subset result. These target-fit comparisons
do not establish equal physical image error.

The subset counts include throughput probes, not just the first passing count.
Nikon ASTRA at 16 subsets also passes, with probe warm time 12.27 s versus
12.11 s at 12; their throughput difference is small. Zeiss ASTRA at 16 OS
subsets is faster than 13. For Zeiss OSEM, two subsets beat one; TomoJAX's
two-subset probe has fit 0.209491 and warm time 5.58 s, versus 5.49 s at one.
Single-subset TomoJAX OSEM uses 4738 MiB peak GPU and 4.782 GB peak host
memory in the paired runs.

```bash
python bench/cone_matrix.py nikon "$NIKON" --library astra \
  --method ossart --iterations 1 --subsets 12 --norms recompute --repeats 1 \
  --output bench/results/nikon-astra-os.json
python bench/cone_matrix.py nikon "$NIKON" --library tomojax \
  --method ossart --iterations 1 --subsets 13 --repeats 1 \
  --output bench/results/nikon-tomojax-os.json
python bench/cone_matrix.py zeiss "$ZEISS" --library astra \
  --method ossart --iterations 1 --subsets 16 --norms recompute --repeats 1 \
  --output bench/results/zeiss-astra-os.json
python bench/cone_matrix.py zeiss "$ZEISS" --library tomojax \
  --method ossart --iterations 1 --subsets 13 --repeats 1 \
  --output bench/results/zeiss-tomojax-os.json
python bench/cone_matrix.py zeiss "$ZEISS" --library astra \
  --method osem --iterations 1 --subsets 2 --norms recompute --repeats 1 \
  --output bench/results/zeiss-astra-osem.json
python bench/cone_matrix.py zeiss "$ZEISS" --library tomojax \
  --method osem --iterations 1 --subsets 1 --repeats 1 \
  --output bench/results/zeiss-tomojax-osem.json
```

### Ordered-subset work counts

Separate warm profiles on `ed9641e7` use the selected settings above:
2026-10-20 UTC for OS and Zeiss OSEM, and 2026-10-21 UTC for Walnut OSEM.
The one-orbit Walnut timing rows use `e2cd2101`.

| Input, method | Forward T/A (s) | Iterative transpose T/A (s) | FDK backprojection T/A (s) | GPU copies T/A (s) |
|---|---:|---:|---:|---:|
| Walnut, one-orbit OS | 2.414 / 2.222 | 5.876 / 2.513 | 0.796 / 1.226 | 0.217 / 0.857 |
| Walnut, one-orbit OSEM | 1.111 / 1.128 | 6.462 / 2.483 | 0.795 / 1.236 | 0.264 / 0.601 |
| Nikon OS | 2.038 / 4.671 | 8.824 / 3.257 | 1.039 / 1.515 | 0.380 / 1.092 |
| Zeiss OS | 2.784 / 2.095 | 3.128 / 1.663 | 0.506 / 0.839 | 0.230 / 0.932 |
| Zeiss OSEM | 0.891 / 1.223 | 3.209 / 1.659 | 0.500 / 0.841 | 0.235 / 0.415 |

Forward time includes OS row normalization: Nikon uses 0.277 s of specialized
uniform-support projections, while the other OS rows use ordinary forwards.
The matched transpose accounts for most of each gap; Zeiss OS also spends
longer on forwards.

Walnut, Nikon and Zeiss OS make 24 / 26 / 13 TomoJAX paired-adjoint callbacks,
carrying 2400 / 3172 / 1612 scalar view slots. Their data and support forwards
carry the same totals. These include padding and shifted-tail overlap; ASTRA
records 2400 / 3142 / 1602 physical forward and backprojected views.
Walnut and Nikon use 144 / 156 paired-plane kernels (5.836 / 8.764 s);
Zeiss uses 104 separable-adjoint kernels (3.099 s). ASTRA's iterative
backprojections use 96 / 120 / 64 launches.

One-orbit Walnut OSEM makes 48 TomoJAX forwards (1200 slots) and 96 adjoint
callbacks (2400 slots); its wide transpose uses 192 kernels taking 6.405 s.
ASTRA records 1200 physical forward views, 2400 backprojected views and 84
backprojection launches. A same-source paired probe of 12 versus 13 TomoJAX
subsets takes 8.391 versus 8.424 s warm: fewer callbacks do not improve
throughput here.

Zeiss OSEM makes 26 TomoJAX forwards (806 slots) and 52 adjoint callbacks
(1612 slots), including 104 separable-adjoint kernels (3.172 s).
ASTRA records 801 physical forward views and 1602 backprojected views,
with 52 backprojection launches. Use the warm NVTX capture above with the
corresponding ordered-subset command to reproduce these profiles.

## FIPS walnuts (real lab cone-beam CT)

`bench/walnut.py` reconstructs the walnut collection of Der Sarkissian et al.
(Scientific Data 6, 215, 2019; CC BY 4.0). Each walnut has three source orbits
of 1200 projections on a FleX-ray scanner, per-view geometry vectors, and a
reference reconstruction: 50 iterations of non-negative least squares on all
three orbits, 501³ voxels of 0.1 mm. Walnut 1 was measured on 2026-10-07
(commits `95f0530` to `d1f36f2`):

| | TomoJAX | ASTRA |
| --- | --- | --- |
| Geometry from the vectors | source 66.0014 mm, detector 199.0062 mm (the scanner's log: 66.0014, 199.0062) | |
| FDK, one orbit (1200 views) | 1.7–1.9 s on the first call, compilation included; 1.55 s after; within 0.05% of ASTRA's volume and 0.6% of the authors' published FDK | 2.13 s |
| FDK error against the reference, in the walnut | 0.099 | 0.099 |
| FDK, detector binned 2 × 2 | 1.23 s on the first call, 1.05 s after; error 0.091 | 1.43 s; 0.091 |
| Non-negative least squares, three orbits, every 4th view, 20 iterations | 304 s; error 0.152 | 217 s (the reference's own solver); 0.147 |
| The same, detector binned 2 × 2 | 86 s; error 0.154 | 91 s; 0.152 |

Times start with the projections in host memory and end with the volume
there. JAX's GPU start-up (about 2 s, once per process) is not included. FDK
filters each row by a matrix product on the GPU and backprojects through the
texture unit from half floats, scaled per batch to their peak; like ASTRA's,
its interpolation weights are rounded to 1/256. The volumes agree with ASTRA's
to 0.05% (FDK) and 1.3% (least squares).

Unbinned, TomoJAX's least squares is slower. The detector samples twice as
finely as the 0.1 mm voxels (a pixel spans 0.05 mm at the axis), so the
matched transpose, which follows every ray through every plane, does four
times the work the volume needs; ASTRA's backprojector samples the detector
once per voxel. Binning 2 × 2 (`scan.binned(2)`) costs nothing in accuracy here
and makes TomoJAX's reconstruction 3.5 times faster and ASTRA's 2.4 times. The
FleX-ray's rotation axis also leans 0.5° across the detector, so every view
takes the transpose's general path, about 1.3 times slower than its path for
an upright axis.

### Full-view ordered subsets

Measured 2026-10-20 on `75435039`: all 3600 views, 2 × 2 detector binning,
a 501³ grid with 0.1 mm voxels, and nonnegative OS-SART from FDK. The call is
`tj.reconstruct(scan, "sirt", iterations=1, warm_start=True,
config=SIRTConfig(subsets=17, lower_bound=0))`. ASTRA uses 16 interleaved
subsets, its native forward/backprojectors, and column normalization recomputed
at each update; its FDK start averages the three physical turns.

| Library | Cold / warm (s) | Full / walnut-support relative L2 error |
|---|---:|---:|
| TomoJAX, 17 subsets | 36.27 / 26.62 | 0.082527 / 0.055488 |
| ASTRA, 16 subsets | 31.59 / 22.54 | 0.083874 / 0.056272 |

Times are means of two interleaved fresh-process measurements; cold includes
loading and setup, warm repeats the whole solve, including FDK and normalization.
Filesystem caches are not cleared. TomoJAX has lower error but is 18.1% slower
warm. A separate profile spends 16.91 s in its exact transpose versus 7.63 s
in ASTRA's backprojection; this identifies an implementation cost, not a lower
bound on every exact-transpose implementation.

That profile records 408 launches of TomoJAX's paired residual-and-normalization
transpose kernel, versus 256 ASTRA iterative backprojection launches. TomoJAX
also launches 68 data forwards (4.67 s) and 68 uniform-volume forwards (0.42 s);
ASTRA's forwards, including row normalization, take 6.84 s. FDK backprojection
takes 2.41 / 3.76 s for TomoJAX / ASTRA, and GPU host/device copy time is
0.53 / 1.53 s. These are diagnostic kernel times, not the uninstrumented
whole-solve timings in the table.

The public benchmark configures these OS settings with:

```bash
python bench/cone_matrix.py walnut3 "$WALNUT" --library astra \
  --method ossart --iterations 1 --subsets 16 --norms recompute --repeats 1 \
  --output bench/results/walnut3-astra-os.json
python bench/cone_matrix.py walnut3 "$WALNUT" --library tomojax \
  --method ossart --iterations 1 --subsets 17 --repeats 1 \
  --output bench/results/walnut3-tomojax-os.json
```

#### Three-orbit OSEM

Measured 2026-10-21 UTC on `ed9641e7`, with the same 3600-view input,
2 × 2 detector binning and 501³ grid. Both libraries use one sweep from FDK;
ASTRA recomputes column normalization. Two fresh processes per library run
in ASTRA–TomoJAX–TomoJAX–ASTRA order, with one warm solve per process.
Cold includes numerical imports, loading and the first host volume; warm
includes FDK, normalization, transfers and the host result. Filesystem caches
are not cleared, and scoring is outside the timer.

| Library, subsets | Cold / warm (s) | Full / support image error | Volume difference to ASTRA |
|---|---:|---:|---:|
| TomoJAX, 13 | 42.85 / 27.96 | 0.070274 / 0.048422 | 0.004439 |
| ASTRA, 14 | 30.70 / 21.00 | 0.068634 / 0.047040 | — |

Both meet the fixed full/support error targets of 0.0724 / 0.0504.
The unfitted orthogonal sections show the same shell and internal structures;
differences lie mainly in shell contrast and smooth support fields.
TomoJAX is 33.2% slower warm. Peak solver memory is 3358 / 3892 MiB GPU and
7.733 / 9.045 GB host for TomoJAX / ASTRA. ASTRA's host-cached 13-subset
normalization fits in 14.69 GB host memory but takes 38.61 s warm in a policy
probe, versus 19.76 s for 14-subset recomputation.

```bash
python bench/cone_matrix.py walnut3 "$WALNUT" --library astra \
  --method osem --iterations 1 --subsets 14 --norms recompute --repeats 1 \
  --output bench/results/walnut3-astra-osem.json
python bench/cone_matrix.py walnut3 "$WALNUT" --library tomojax \
  --method osem --iterations 1 --subsets 13 --repeats 1 \
  --output bench/results/walnut3-tomojax-osem.json
```

Separate warm profiles on the same source locate the gap in the transpose:

| Forward T/A (s) | Iterative transpose T/A (s) | FDK backprojection T/A (s) | GPU copies T/A (s) |
|---:|---:|---:|---:|
| 3.413 / 4.176 | 19.765 / 8.138 | 2.590 / 3.890 | 0.774 / 1.651 |

TomoJAX makes 117 forward callbacks (3627 slots) and 234 adjoint callbacks
(7254 slots), including padding and shifted-tail overlap. Its wide transpose
uses 468 kernels taking 19.615 s. ASTRA records 3600 physical forward views,
7200 backprojected views and 252 backprojection launches. Both profile outputs
meet the same image-error targets. These counts describe the current
implementations, not an inherent cost of requiring an exact transpose.

### Orbit registration

The collection provides two geometries per walnut. `scan_geom_original.geom`
is as the scanner recorded it. In `scan_geom_corrected.geom` the authors
raised the source and detector of orbits 2 and 3 by 0.397 and 0.794 mm to bring
them into register with orbit 1. Seen from the object, which is how TomoJAX's
poses describe it, that is a vertical correction (`dz`) of −0.397 and
−0.794 mm. Measured 2026-10-07 on commit `0684880`, `bench/walnut_alignment.py`
gives `tj.align` the original record (three orbits, every 4th view, binned
2 × 2), which finds the correction itself in 4.4 minutes:

| | Orbit 2 | Orbit 3 |
| --- | --- | --- |
| Height relative to orbit 1, TomoJAX | −0.381 mm | −0.755 mm |
| The authors' correction | −0.397 mm | −0.794 mm |

Only the orbits' relative heights are observable. `tj.align` reports the
least-motion estimate, here 0.38 mm above the authors', who keep orbit 1 where
its record puts it. To compare volumes voxel by voxel the bench therefore holds
orbit 1 fixed, as they do. Then 20 iterations of non-negative least squares
reach the reference to an error of 0.155 with TomoJAX's geometry, against
0.154 with the authors' and 0.280 with the original record:

![Vertical and axial slices of the walnut reconstructed with the published, original, TomoJAX-aligned and corrected geometries](images/walnut_alignment_slices.png)

![Each orbit's height relative to orbit 1, recovered by tj.align, against the authors' correction](images/walnut_alignment_heights.png)

`tomojax.alignment.api.orbit_heights`, which compares the orbits' data without
reconstructing, finds +0.37 and +0.74 mm in 23 s on the same walnut
(2026-10-10, commit `944b2e0`). Pose alignment there finds +0.38 and +0.76 in
about 250 s, and held-out views favour about +0.39 and +0.76.

Across all 42 walnuts (2026-10-08), the median disagreement with the authors'
heights is under a fifth of a voxel; the
[benchmark guide](../bench/README.md#fips-walnuts-real-lab-cone-beam-ct) has
the table.

## Alignment

Unless stated, these are fresh `tomojax align` processes on analytic
projections of continuous objects, which are independent of TomoJAX's voxel
model. They were measured with the default settings and detector-frame
translations on 2026-10-06, commit `9d4c4d1`.

### Per-view pose recovery

181 views at 128³, 361 at 256³, 30° laminography and parallel:

| Scan | Motion | Time | Rotation RMSE |
|---|---|---:|---:|
| 128³ parallel / laminography | ±0.25°, ±0.5 px | 50 / 43 s | 0.0085 / 0.0026° |
| 128³ parallel / laminography / anisotropic | ±0.5°, ±8 px | 33 / 46 / 34 s | 0.0090 / 0.0029 / 0.022° |
| 256³ laminography, full resolution | ±0.25°, ±0.5 px | 202 s | 0.0030° |
| 256³ laminography, stopped at half resolution (`--levels 4 2`) | ±0.25°, ±0.5 px | 55 s | 0.0052° |
| 64³ parallel / laminography | ±0.5°, ±15 px (23% of the detector) | 19 / 27 s | 0.031 / 0.013° |

At 128³ the translations are recovered to 0.002 pixels, and the 256³ run fits
in 8 GB of GPU memory. At 64³ the 0.01–0.03° results are the discretisation
floor for that size. At 32³ the floor is 0.1–0.5°, even when the solve starts
from the true poses (2026-10-05, commit `f155b94`). The 64³ anisotropic scan
with ±15 px shifts is not captured (5.0° rotation error) in either translation
frame. An earlier record of 0.0076° and 0.016° for the ±8 px parallel and
anisotropic scans could not be reproduced: that commit now gives 0.0087° and
0.022°.

The global shift search is what captures large shifts (2026-10-05, commit
`84b941b`). With ±0.5° tilts and ±15 px shifts at 64³, rotations end 7.7–12.6°
wrong without it and 0.012–0.034° with it, in parallel and laminography scans.

After the half-resolution alignment of the 256³ laminography scan, the full
data were reconstructed with the recovered poses (2026-10-05, commit
`a7416a7`). Unregularised CGLS reaches 0.27 relative error. Positivity-
constrained FISTA reaches 0.12 after 100 iterations and 0.091 after 300
(109 s), against 0.059 for the volume of the full-resolution alignment.

### Detector centre

The projection-only estimator uses a one-pixel derivative bandwidth and a
Fourier-interpolated global correlation peak. On 2026-10-27, the fixed shift
matrix (`JAX_PLATFORMS=cpu uv run --no-sync python
../../shared/agent03/shift_kill.py`, implementation `a04e5ee7`) gave the following 95th-percentile absolute errors in the **change**
of centre after injected ±0.25, ±0.5 and ±1 px shifts (T1):

| Scan (three detector rows) | Centre-change error, px |
|---|---:|
| APS 2-BM test_data | 0.00085 |
| Elettra TomoBank 00024 | 0.00292 |
| ESRF bamboo | 0.00567 |
| I12 3339629, rows 1050/1080/1100 | 0.00117 |
| I13 159815, rows 690/1080/1470 | 0.00488 |

These errors test shift equivariance, not absolute centre accuracy. Motion,
truncation and competing correlation peaks can bias a single centre estimate.

`alignment.api.estimate_center_report` retains the weighted view-pair constraints,
conditional 90% matching intervals, other correlation peaks and apparent
row-centre slope. The uncertainty uses eight detector-column blocks, an observed
cross-Hessian and leverage correction; it does not claim a physical-centre
interval under arbitrary motion. Pair disagreement is reported separately.
The automatic workflow saves these diagnostics and reports conditional
inconsistency without replacing it by independent per-view poses.
Runtime fields label the intervals `local_matching_interval90_px`, with
`global_coverage_calibrated=false` and `uncertainty_status=local_matching_only`.
These are local matching diagnostics, not calibrated global centre intervals;
automatic alignment reports this limitation even without pair disagreement.

On 2026-10-27 (`a04e5ee7`), 64 real rows gave centre-change p95 errors
0.00422 px on Elettra 00024 and 0.00284 px on I13 159815. The report took
0.233/0.230 s and 2.30/2.27 s cold/warm on corrected inputs; cumulative peak
host memory was 348/663 MiB, GPU memory zero. Reading and correcting the
selected HDF5 observations took 0.035/52.61 s separately. Command:
`OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu uv run --no-sync python
../../shared/agent03/full_rows.py`. I13 used rows 300:1860 and columns 441:1872;
both scans sampled 64 evenly spaced rows. I13 pair centres ranged from
−3.76 to +0.59 px, with 1.75 px scatter and conditional standard errors
0.049–0.287 px. These conflicting pairs do not define one physical centre.
Elettra's 50.800 px estimate and conditional interval [49.235, 52.366] px
include the TomoBank 51.5 px centre convention (T3 agreement, not accuracy).

Independent analytic Gaussian line integrals with Poisson counts, seed 304,
100 pairs per category, gave conditional 90% coverage 93% dense, 100% sparse,
91% low contrast and 91% truncated. Their p95 location errors were
0.0103/0.0152/0.0861/0.0088 px. Command: `OPENBLAS_NUM_THREADS=1
JAX_PLATFORMS=cpu uv run --no-sync python
../../shared/agent03/production_coverage.py`. This calibration excludes wrong
matching modes, axis tilt and sample motion; it is not whole-scan coverage.

Independent gVXR cuboid projections expose failures outside that calibration
(2026-10-27, `a04e5ee7`, renderer seed 316, count seed 317, five cases per
category, 257 by 65 pixels, 30,000 incident photons). Absolute centre p95
errors were 0.086/0.156/0.120/0.052/0.173 px for dense, sparse, off-centre,
phase-fringe and truncated objects, respectively, but 3.907 px at low contrast
and 1.004 px with a one-degree axis tilt. Only one of the five low-contrast
truths lay inside its conditional 90% interval. Tilt is outside the axial
matching model; its apparent row-slope error reached 0.678 px p95 over the
32-row half-height. Command: isolated gVXR Python runs
`../../shared/agent03/gvxr_render.py`, then `JAX_PLATFORMS=cpu uv run --no-sync
python ../../shared/agent03/gvxr_score.py`, with `LAB_ARTIFACTS` set to the
output directory. The complete renderer-and-score job took 3.04 s; the CPU
scoring stage took 0.564 s with 273 MiB peak host memory.

Signed cSAXS porous-glass phase data gave centre-change p95 0.000512 px over
the same six injected shifts, versus 0.006150 px for the integer-peak baseline
(2026-10-27, `a04e5ee7`). The 51 selected views used 64 evenly spaced rows and
1312 columns. Point estimation took 0.410/0.410 s cold/warm versus
0.0393/0.0398 s for the baseline; data reading took 7.31 s, with 640 MiB peak
host memory and no GPU use. Command: `OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
uv run --no-sync python ../../shared/agent03/psi_phase.py`. This is shift
equivariance on phase data, not an absolute centre measurement.


| Mode and scan | Result |
|---|---|
| `cor`, +3.7 px offset, 128³ parallel / laminography (2026-10-05, `74923c1`) | 3.693 / 3.677 px in 41 / 84 s; with ±0.5°/±8 px motion as well, 2.2 / 3.3 px |
| `cor-then-pose`, +3.7 px offset with ±0.5°/±8 px motion, 128³ parallel / laminography | rotations to 0.005 / 0.006°, volume error 0.0007 against a true-geometry reconstruction; reports −3.573 px, where the identifiable value is −3.569 px (the random shifts have a mean of 0.131 px) |
| `cor-then-pose`, gVXR chip phantom (DIAD-like 25 keV laminography with phase contrast, blur and noise; 3.2 px offset, about 2 px shifts, 0.1° tilts) | 3.18 px (identifiable 3.178 px), rotations to 0.073°, shifts to 0.087 px RMS, in 127 s |

The earlier sequence, a centre-of-rotation search followed by a pose polish,
left rotation errors of 0.23–0.26° on the analytic scans. With object-frame
translations on the chip phantom's 720 views, `dx` reached 70 px RMS around
90° and 270°, and the per-view u error rose to 0.37 px, against 0.06 px with
detector-frame translations.

### Where the volume sits

Without the least-motion gauge (2026-10-06, commit `123dffe`), a 64³ cone-beam
alignment left a common 1-voxel shift along the axis, and a half-turn parallel
scan with a detector offset a 3-voxel shift. The relative volume errors were
0.26 and 0.55; with the gauge they fall to 0.0037 and 0.019. The README's 96³
laminography example falls from 0.099, with only a rigid shift removed, to
0.047.

### Cone-beam scans

`--mode cor` on 128³ and 256³ blob scans (360 views, 1% to 5% noise) with axis
offsets of 3.7 to 11.2 voxels and detector rolls up to 1.2° recovers the offset
to 0.08 voxels and the roll to 0.04°, in about 4 s at 256³. At 64³ the roll is
good to about 0.15° (2026-10-06, commit `0547ee7`).

On a 64³ cone scan (120 views, magnification 1.5) with ±0.3° and ±1.5 px of
random motion in all six parameters, pose alignment recovers the poses exactly
when the data come from its own projector (2026-10-06, commit `3c35507`). From
exact line integrals of smooth Gaussian blobs it recovers translations to
0.004–0.07 px and rotations to 0.16° RMS (0.13° at 128³), against 0.10°
(0.08°) for the same objects in a parallel beam. `phi`, about the rotation
axis, is the weakest parameter in both.

### Free-voxel pilot

The six-cell pilot (2026-10-04 to 10-06) uses independent random-voxel 32³
objects, 61 irregular views, parallel, shifted anisotropic and 30° tilted
geometry, each clean and noisy, with ±0.25°/±0.5 px motion. Its gates are
0.01° for rotations together with fixed image and shift gates. Its
projections integrate the voxel basis exactly. With the default Joseph
integrator, a different discretisation from the data, every cell misses the
0.01° gate at 0.10–0.26°, while volume errors stay at 0.009–0.085. With
`ray_integrator = "exact"`, the model the data were made with, five of six
cells pass in 8–20 s. That partly reflects an inverse crime. The noisy
anisotropic cell misses at about 0.016°; a local noise analysis predicts about
0.017° from the measurement noise alone. The exact integrator is 10–30 times
slower than Joseph sampling (2026-10-05, commit `f155b94`).

## Memory

### Scans larger than the GPU

All on the 8 GB laptop GPU:

| Run | Result | Date, commit |
|---|---|---|
| `fbp_host`, 1024³ laminography (30° tilt, 1024 views of 1024²), memmap in and out, 4.3 GB each | 37 s, in two x-slabs; laminography FBP ran out of memory at 768³ before | 2026-10-05, `8bc0653` |
| FISTA-TV, 512³ laminography, 3072 views (3.2 GB of projections), streamed, 10 iterations | 4.3 GB peak GPU memory | 2026-10-05, `6fa1585` |
| FISTA-TV, the same with 768 views, where both fit | 67.6 s streamed, 65.0 s with the projections on the GPU | 2026-10-05, `6fa1585` |
| CGLS, 3072 views, streamed, 10 iterations | 154 s, 3.2 GB peak; converges to the same solution | 2026-10-06, `9d4c4d1` |
| SPDHG-TV, 3072 views, streamed, 192 iterations (one view block each) | 297 s, 4.3 GB peak, with its 3.2 GB dual in host memory; bitwise identical to the device-resident solver | 2026-10-06, `9d4c4d1` |
| SIRT from raw frames with `tj.reconstruct_slabs`, the whole Diamond I12 scan below: 1801 views of 2160 × 2560 pixels (39.8 GB corrected), a 2560² × 2160 volume (56.6 GB) | 1436 s of solving, 1.41 GB peak GPU memory and 3.7 GB host memory; 1.13 times the per-voxel time of an 8-row reconstruction that fits | 2026-10-17, `440cc563` |
| `tj.reconstruct` of 352 rows of the same scan, returned as one 9.23 GB array | 211 s, 0.78 GB peak GPU memory, 17 GB host memory; 1.04 times the per-voxel time of 8 rows | 2026-10-17, `440cc563` |
| OSEM of the FIPS walnut at full resolution: all 3600 views of 972 × 768 pixels, a 1001³ volume, 4 subsets × 1 iteration from FDK | First / repeat solve 230.65 / 241.81 s in 128-plane host slabs; loading 7.52 s; peak GPU 5798 MiB, host 15.57 GB anonymous; full / walnut-support relative L2 error 0.124385 / 0.094026 | 2026-10-21, `30f9f1df` |
| FDK from raw CuO-spl1 frames: all 3201 views of 2036² pixels, a 2036³ volume, 32-plane output slabs | First / repeat correction and reconstruction 847.16 / 786.11 s; imports and loading 15.78 s; peak GPU 6056 MiB, host 12.634 GB, including scoring | 2026-10-20, `9be6e5ad` |
| ASTRA FDK on the same raw CuO-spl1 frames and output slabs | First / repeat correction and reconstruction 847.16 / 838.56 s; imports and loading 13.67 s; peak GPU 3036 MiB, host 13.193 GB, including scoring | 2026-10-20, `76a1f412` |
| FDK from raw CuO2 frames: all 3201 views of 2028² pixels, a 2028³ volume, 32-plane output slabs | First correction and reconstruction 1053.93 s; imports and loading 16.17 s; peak GPU 4494 MiB, host 14.240 GB, including scoring | 2026-10-20, `75435039` |
| OS-SART of CuO-spl1 at twofold binning: all 3201 views of 1018² pixels corrected into memory (13.3 GB), a 1018³ volume kept on the GPU, 3 iterations × 8 subsets from zero | 441 s solving, 30 s correcting; peak GPU 7484 MiB, host 13.78 GB while solving (15.40 GB with scoring); relative residual 0.0319 on every eighth view | 2026-10-20, `30f9f1df` |
| SIRT of a synthetic 1300³ volume (8.79 GB, more than the GPU) in host memory: 180 cone views of 1300² pixels, 3 iterations | 62.8 s solving in 128-plane slabs; peak GPU 6160 MiB, host 11.47 GB | 2026-10-20, `5cf50e6b` |
| CGLS of the same synthetic 1300³ volume with it and its three work volumes memory-mapped on a USB hard disk (35 GB), 2 iterations | 1514 s solving in 128-plane slabs; peak GPU 6476 MiB, host 14.55 GB anonymous; 113 GB read and 114 GB written, creating the files included; normal residual 7.79e8 → 8.69e7 | 2026-10-21, `30f9f1df` |
| OS-SART of native CuO-spl1 from raw frames: all 3201 views of 2036² pixels, corrected as the sweeps reach them, a 2036³ volume (33.76 GB) memory-mapped on a USB hard disk, 1 iteration × 4 subsets from zero | 2131 s solving, 2295 s in all; peak GPU 6398 MiB, host 9.39 GB anonymous (12.07 GB resident); 207 GB read (106 GB of raw frames, 101 GB of volume), 135 GB written | 2026-10-20, `5cf50e6b` |

The walnut call is `tj.reconstruct(scan, "mlem", warm_start=True,
config=MLEMConfig(iterations=1, subsets=4))`, using all three corrected
geometries, unbinned projections clipped at zero, and 0.05 mm voxels. The
volume is scored at the published 0.1 mm voxel centres. Both calls use
`XLA_PYTHON_CLIENT_PREALLOCATE=false`; the repeat uses the same process after
releasing the first volume.

The CuO call is `tj.load_frames("Tristan Manchester_Copper_oxyde_spl1.txrm",
darks=0)`, followed by `tj.reconstruct_slabs(frames, "fbp")`. Consume and
release each block before advancing the iterator; no whole volume is retained.
The grid uses the recorded 1.12756 µm sampling without binning. The first
process uses fresh compilation caches and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
Times sum the correction and reconstruction stages between scoring operations.
Including interleaved scoring, the first / repeat passes take 1455.72 / 1365.08 s,
with imports and loading additional to the first pass.

CuO's relative projection residual is 0.0586312 on 256 fixed, evenly spaced
views, using the full detector and an ASTRA forward projection summed over
the physical output slabs. Reconstruction uses all 3201 views. The vendor
comparison covers all 2005 planes with one fixed spatial registration and one
whole-volume affine intensity fit: normalized RMSE 0.209660. It is a diagnostic,
not physical ground truth; the reconstructed pores are visibly less sharp
than the vendor's. Both metrics and the six retained native cuts agree exactly
between repeats.

ASTRA uses `FDK_CUDA` in 256-view batches on the same 32-plane output slabs,
preserving the complete acquisition's preweighting and angular scale. Its
first / repeat scored passes take 1304.22 / 1314.98 s, with imports and loading
additional to the first pass. Its projection residual is 0.1171325 and vendor
NRMSE 0.209232, both repeatable. First correction and reconstruction times are
roughly equal; TomoJAX's repeated reconstruction is 6.3% shorter, but ASTRA's
scored passes are shorter. The lower TomoJAX projection residual does not
establish better physical image accuracy.
On six native axial cuts, the relative difference between the images is 0.143371
across the cuts and 0.008441 within an independently vendor-defined sample mask.
99.41% of their squared difference lies outside the horizontal cylinder seen
by every view. These are cut diagnostics, not a full-volume image comparison.

CuO2 uses the same load and slab calls with `CuO_2_tomo-B.txrm` and the
recorded 1.13365 µm sampling. Its first scored pass takes 1376.31 s, with
imports and loading additional. The same 256-view projection score is
0.0790636. There is no vendor volume, repeated pass or native library comparison
for this scan; this is a capacity and data-fit measurement.

The OS-SART rows of CuO-spl1 are `python bench/out_of_core.py cuo
"Tristan Manchester_Copper_oxyde_spl1.txrm" --binning 2 --iterations 3
--subsets 8`, which corrects the projections with `Frames.corrected(binning=2)`
and calls `tj.reconstruct(scan, "sirt", grid=..., config=SIRTConfig(iterations=3,
subsets=8))`, and the same script at native resolution with `--iterations 1
--subsets 4 --scratch DIR`. The synthetic rows are `python bench/out_of_core.py
synthetic --size 1300 --views 180 --method sirt --iterations 3` (and `--method
cgls --iterations 2 --scratch DIR`), spheres in a cylinder seen with the CuO
cone angle. Each CGLS iteration reads the four volumes six times and writes
them four times between them.

At native resolution the 2036³ float32 volume, 33.76 GB, is more than this
machine's 30 GB of RAM. In the native OS-SART row it is a memory-mapped file
in `DIR` given as `out=`, written by the first sweep and read and rewritten
once by each later one, in 32-plane slabs (the 8 GB GPU holds no thicker ones
beside a subset's 800 views). The corrected projections (53 GB) are never
stored: each sweep corrects 64-row bands of every view from the raw frames, a
band ahead on a worker thread. The raw frames are on the internal NVMe disk,
the volume on a USB hard disk; exFAT has no sparse files, so creating the
volume writes it once more. One iteration of four subsets from zero is a
capacity measurement, not a converged reconstruction; its central planes show
the sample cylinder as the twofold-binned reconstruction does.

The Diamond rows use `SIRTConfig(iterations=3, subsets=15, lower_bound=0)`
with `warm_start=True` (an FBP start) on frames from
`tj.load_frames(projections, angles=..., flats=..., darks=...)`, a 2560² grid
with the detector centre at −138 pixels. The full scan was run as three jobs of
720 rows, each from a fresh process, so the time is their sum; each job peaked
at the same GPU memory. Eight slab rows are bitwise equal to ordinary
`tj.reconstruct` of those rows. Stock ASTRA and TIGRE entry points would need
the 39.8 GB of corrected projections and the 56.6 GB volume in host memory,
more than the 30 GB of this machine, so no full-scan comparison was attempted;
either can be split by rows by hand for this geometry.

### Iterative solver memory

512³ laminography, 768 views of 512² pixels (0.8 GB of projections), with
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (2026-10-05, commit `4bdf52a`):

| Solver | Peak GPU memory before the change | After |
|---|---|---|
| CGLS | out of memory | 5.6 GB |
| FISTA-TV | 7.2 GB | 4.6 GB |
| SPDHG-TV | out of memory | 4.9 GB |

CGLS keeps about five volumes and three sinograms, FISTA-TV seven volumes, and
SPDHG-TV eight volumes plus its dual sinogram. JAX's default allocator limit is
75% of device memory, under which this CGLS case does not fit; the CLI raises
the limit to 90%.

### Fourier inverse

Measured 2026-10-04 on commit `776a8c9`, before later changes made the Fourier
inverse faster; the memory has not been remeasured since. Parallel Gaussian
phantom, 720 views, CUDA backend; warm time is the median of seven solves. At
512³ the default 16-slice slabs peak at 678 MiB. Allowing the CuPy queues to
be reused across calls brought that peak down from 3892 MiB.

| Case | Slices per slab | Warm time | Peak GPU memory | Fastest external warm, its peak |
|---|---:|---:|---:|---|
| 256³ | 16 | 88.0 ms | 284 MiB | CuPy/ASTRA: 432 ms, 814 MiB |
| 256³ | 8 | 87.2 ms | 220 MiB | CuPy/ASTRA: 431 ms, 814 MiB |
| 512³ | 16 | 518 ms | 678 MiB | CuPy/ASTRA: 3085 ms, 1262 MiB |
| 512³ | 8 | 615 ms | 430 MiB | native ASTRA 2D: 3072 ms, 156 MiB |

`FourierConfig(slices_per_batch=8, backend="cupy")` selects eight slices.
ASTRA's slice-wise native 2D FBP uses the least memory.

## Several GPUs

Measured 2026-10-08 on commit `a3a925b`, on four H100 80 GB GPUs in one Modal
machine. The run is 20 non-negative FISTA iterations on the binned FIPS walnut
(three orbits, every fourth view: 900 views of 486 × 384 pixels, 501³ voxels),
with views shared among the GPUs by `devices=`. Each time is the second call:

| GPUs | Time (s) | Speedup | Volume vs one GPU (relative L2) |
|---:|---:|---:|---:|
| 1 | 24.8 | 1.00 | — |
| 2 | 13.6 | 1.83 | 2.9e-7 |
| 4 | 7.4 | 3.34 | 2.9e-7 |

First calls, which compile, took 27.4, 15.8 and 9.5 s. The difference from one
GPU is the order of the cross-device sum. Each GPU still runs the solver's
volume-sized updates, which do not shrink with more GPUs, and the 0.5 GB volume
sum crosses NVLink once per iteration. ASTRA and TIGRE were not run on this
machine, so these numbers are not a comparison with them.

## Images

The README's [synthetic figure](../images/README.md#reproducible-synthetic-reconstruction)
uses the same model to simulate and reconstruct: it shows the API, not
accuracy. The [historical DIAD laminography images](../images/README.md#historical-real-data-illustrations)
show qualitative use on real data; their raw scan and configuration are not
bundled, so they cannot be reproduced.

## Calibrated detector count noise

`tomojax.datasets.DetectorModel` estimates effective photon gain and read noise
from repeated flats/darks and draws Poisson/read noise before flat correction.
The following **T2 held-out-flat** check calibrates on the first temporal half
and evaluates independent adjacent-frame differences in the second half. It
measures noise, not reconstruction accuracy or complete ring realism.

On 2026-10-27, commit `b608d187`, seed 705, the first/repeated calls generated
100 flat exposures on a 16-row cut. Both observed and generated log differences
used the same column high-pass (Gaussian sigma 8, 32-column edge exclusion).

| Scan and training rows | Simulated/observed RMS | Adjacent-row correlation error | First/repeated draw (s) |
|---|---:|---:|---:|
| I13 159815, 1080:1096, columns 441:1869 | 0.9601 | -0.0095 | 0.113/0.113 |
| I12 3339629, 1048:1064, all columns | 1.0022 | -0.0085 | 0.216/0.213 |
| ESRF bamboo reduced, 262:278, all columns | 1.0019 | -0.0395 | 0.052/0.050 |

All pass the frozen RMS-within-20% and correlation-error-within-0.1 thresholds.
An equally calibrated independent Gaussian baseline also passes; these data do
not require a more complicated spatial noise model at those tolerances. ESRF
has one dark frame: its read noise is explicitly assumed zero, not measured.
The effective gain can absorb unseparated noise under that assumption.

Command: `LAB_ARTIFACTS=artifacts/agent05-count-statistics JAX_PLATFORMS=cpu uv run
--no-sync python shared/agent05-detector/count_statistics_flux.py` in lab run
`tomojax-20261027a`. The combined run used 620 MiB peak host RSS and no GPU memory;
draw times exclude imports and reference loading. Finite-flat noise generates
persistent row-specific errors, but sample-ring statistics and blind-judge
indistinguishability remain unmeasured for this model.

A stricter tail check on the same frozen split (commit `b608d187`, seed 705,
`shared/agent05-detector/noise_tails_flux.py`) divides pixels into five first-half
signal quintiles. RMS ratios remain within 20% for every group, but ESRF's
excess-kurtosis discrepancy is 0.300–4.811, exceeding the 0.5 gate in four of
five groups. I12 and I13 pass all five groups, with maximum discrepancies
0.023 and 0.390. The model therefore fails ESRF's tail-distribution test despite
matching its RMS; sparse transient contamination is not represented by its
Poisson/read model.
