# Whole-system reconstruction matrix, 2026-10-04

26/27 cells have accepted TomoJAX and external results. The overall optimization goal remains open.

The primary score uses fresh-process time through the first accepted fixed budget, including initialization, transfers, compilation, unsuccessful lower budgets and quality verification. Warm values are medians of seven complete calls at the selected budget. The same cold-fastest accepted external workflow supplies both the timing and memory denominator.

All cases use 180 views and the original versioned physical geometry, phantoms, noise and full-volume relative L2 gates. No amplitude fitting, cropping or threshold changes were made. An absent accepted result blocks the whole-matrix geometric mean; passing-only averages cannot meet the goal.

This corrected protocol loads physical metadata without JAX/Pallas in external workers. Version 1 and older cold ratios included that unrelated startup cost and are superseded for goal scoring. Each raw worker records its loaded solver modules. Source was frozen in an isolated snapshot and verified unchanged after all 297 method records completed.

The per-suite `environment.source_tree_sha256` field mistakenly hashed the
caller checkout. Use the [source audit](../../bench/reference/system-matrix-v2-source-audit.json)
and [retained launch archive](../../bench/reference/system-matrix-v2-source.tar.gz)
for executable-source provenance. Every archived benchmark/library file and
the driver match the isolated snapshot byte for byte and reproduce the
launch digest. The raw records remain unchanged. Future environment records
resolve their source root from the executed script.

Measurements use the RTX 4070 Laptop GPU (8 GB); full versions and execution environment are in each raw file. Process GPU memory is sampled per PID every requested 10 ms across imports, search and repeats. This includes runtime/allocator overhead but can miss transient peaks. Fixtures are generated independently before workers; reading them is timed.

Whole-matrix cold geometric mean: —. Worst measured accepted-pair cold ratio: 0.21×. An undefined aggregate is not a passing result.

Raw records: [gaussian-v1](../../bench/reference/system-matrix-v2-gaussian-v1-64-128-256-180.json.gz), [structured-v1](../../bench/reference/system-matrix-v2-structured-v1-64-128-256-180.json.gz), [structured-noisy-v1](../../bench/reference/system-matrix-v2-structured-noisy-v1-64-128-256-180.json.gz); [complete score](../../bench/reference/system-matrix-v2-summary.json).

## Accepted workflows by cell

TomoJAX and external methods are selected independently by cold accepted time. Times are milliseconds, memory is MiB. “No accepted workflow” means its time to acceptance is undefined; measured failed attempts appear below. The raw files retain all successful alternatives and unsupported adapters.

### gaussian-v1

| Cell | Library / method | Cold / warm ms | Relative L2 | GPU MiB |
|---|---|---:|---:|---:|
| parallel-64-180 | TomoJAX: Fourier (CuPy) | 552.24 / 2.83 | 0.000115 | 146 |
| parallel-64-180 | External: ASTRA FBP2D | 291.19 / 23.88 | 0.004108 | 140 |
| anisotropic-64-180 | TomoJAX: Fourier (CuPy) | 556.13 / 1.41 | 0.013551 | 146 |
| anisotropic-64-180 | External: ASTRA CGLS | 333.57 / 29.80 | 0.020479 | 150 |
| lamino-64-180 | TomoJAX: Joseph CGLS | 2649.65 / 608.22 | 0.090326 | 214 |
| lamino-64-180 | External: ASTRA CGLS | 1655.14 / 674.19 | 0.093366 | 158 |
| parallel-128-180 | TomoJAX: Fourier (CuPy) | 633.95 / 11.24 | 0.000134 | 168 |
| parallel-128-180 | External: ASTRA FBP2D | 317.84 / 58.53 | 0.001045 | 140 |
| anisotropic-128-180 | TomoJAX: Fourier (CuPy) | 597.55 / 7.82 | 0.009877 | 170 |
| anisotropic-128-180 | External: ASTRA BP3D + CuPy filter | 361.80 / 10.42 | 0.009649 | 242 |
| lamino-128-180 | TomoJAX: Coarse-to-fine Joseph CGLS | 5691.87 / 708.09 | 0.091121 | 406 |
| lamino-128-180 | External: ASTRA CGLS | 6952.53 / 3191.47 | 0.094436 | 202 |
| parallel-256-180 | TomoJAX: Fourier (CuPy) | 778.13 / 70.67 | 0.000149 | 244 |
| parallel-256-180 | External: ASTRA FBP2D | 596.32 / 297.53 | 0.000285 | 140 |
| anisotropic-256-180 | TomoJAX: Fourier (CuPy) | 709.05 / 41.58 | 0.013707 | 222 |
| anisotropic-256-180 | External: ASTRA BP3D + CuPy filter | 463.27 / 76.94 | 0.013619 | 544 |
| lamino-256-180 | TomoJAX: Coarse-to-fine Joseph CGLS | 17052.17 / 5457.37 | 0.091455 | 1174 |
| lamino-256-180 | External: ASTRA CGLS | 39276.83 / 17920.08 | 0.097107 | 524 |

### structured-v1

| Cell | Library / method | Cold / warm ms | Relative L2 | GPU MiB |
|---|---|---:|---:|---:|
| parallel-64-180 | TomoJAX: Fourier (CuPy) | 691.16 / 3.34 | 0.135290 | 146 |
| parallel-64-180 | External: ASTRA FBP2D | 310.00 / 27.38 | 0.131569 | 140 |
| anisotropic-64-180 | TomoJAX: no accepted workflow | — / — | — | — |
| anisotropic-64-180 | External: no accepted workflow | — / — | — | — |
| lamino-64-180 | TomoJAX: Joseph CGLS | 1545.43 / 26.56 | 0.268145 | 214 |
| lamino-64-180 | External: ASTRA CGLS | 337.76 / 29.04 | 0.268893 | 158 |
| parallel-128-180 | TomoJAX: Fourier (CuPy) | 676.20 / 11.41 | 0.096056 | 168 |
| parallel-128-180 | External: ASTRA FBP2D | 341.42 / 61.06 | 0.092523 | 140 |
| anisotropic-128-180 | TomoJAX: Fourier (CuPy) | 628.08 / 8.02 | 0.124626 | 170 |
| anisotropic-128-180 | External: ASTRA BP3D + CuPy filter | 372.09 / 10.25 | 0.123795 | 242 |
| lamino-128-180 | TomoJAX: Joseph CGLS | 1912.15 / 173.50 | 0.257526 | 398 |
| lamino-128-180 | External: ASTRA FBP + CGLS | 633.30 / 107.28 | 0.275349 | 374 |
| parallel-256-180 | TomoJAX: Fourier (CuPy) | 782.79 / 70.07 | 0.068552 | 244 |
| parallel-256-180 | External: ASTRA FBP2D | 607.14 / 301.17 | 0.072488 | 140 |
| anisotropic-256-180 | TomoJAX: Fourier (CuPy) | 740.64 / 38.82 | 0.088213 | 222 |
| anisotropic-256-180 | External: ASTRA BP3D + CuPy filter | 465.42 / 75.54 | 0.093416 | 544 |
| lamino-256-180 | TomoJAX: Joseph CGLS | 4634.09 / 1419.98 | 0.254374 | 1166 |
| lamino-256-180 | External: ASTRA CGLS | 3068.35 / 987.89 | 0.254414 | 524 |

### structured-noisy-v1

| Cell | Library / method | Cold / warm ms | Relative L2 | GPU MiB |
|---|---|---:|---:|---:|
| parallel-64-180 | TomoJAX: Fourier (CuPy) | 617.44 / 3.22 | 0.137725 | 146 |
| parallel-64-180 | External: ASTRA FBP2D | 295.64 / 26.92 | 0.132838 | 140 |
| anisotropic-64-180 | TomoJAX: Fourier (CuPy) | 608.73 / 1.50 | 0.178906 | 146 |
| anisotropic-64-180 | External: ASTRA FBP + CGLS | 355.22 / 8.19 | 0.175221 | 182 |
| lamino-64-180 | TomoJAX: Joseph CGLS | 1536.37 / 26.72 | 0.268277 | 214 |
| lamino-64-180 | External: ASTRA CGLS | 327.67 / 28.64 | 0.268988 | 158 |
| parallel-128-180 | TomoJAX: Fourier (CuPy) | 664.92 / 11.05 | 0.105897 | 168 |
| parallel-128-180 | External: ASTRA FBP2D | 325.53 / 59.53 | 0.099352 | 140 |
| anisotropic-128-180 | TomoJAX: Fourier (CuPy) | 604.49 / 7.90 | 0.129038 | 170 |
| anisotropic-128-180 | External: ASTRA BP3D + CuPy filter | 363.43 / 10.33 | 0.128229 | 242 |
| lamino-128-180 | TomoJAX: Joseph CGLS | 1724.26 / 103.20 | 0.319490 | 398 |
| lamino-128-180 | External: ASTRA CGLS | 491.65 / 99.03 | 0.319828 | 202 |
| parallel-256-180 | TomoJAX: Fourier (CuPy) | 780.14 / 73.94 | 0.097765 | 244 |
| parallel-256-180 | External: ASTRA FBP2D | 587.41 / 296.83 | 0.102397 | 140 |
| anisotropic-256-180 | TomoJAX: Fourier (CuPy) | 760.52 / 40.49 | 0.102425 | 222 |
| anisotropic-256-180 | External: ASTRA BP3D + CuPy filter | 454.93 / 73.54 | 0.115666 | 544 |
| lamino-256-180 | TomoJAX: Joseph CGLS | 3243.36 / 795.75 | 0.317843 | 1166 |
| lamino-256-180 | External: ASTRA CGLS | 2110.23 / 712.91 | 0.317849 | 524 |

## Paired ratios

Speedup is external time divided by TomoJAX time. Memory fraction is TomoJAX memory divided by that same external workflow. Values above one indicate faster TomoJAX time or greater TomoJAX memory, respectively.

| Suite / cell | Cold speedup | Warm speedup, same pair | Memory fraction |
|---|---:|---:|---:|
| gaussian-v1 / parallel-64-180 | 0.53 | 8.44 | 1.04 |
| gaussian-v1 / anisotropic-64-180 | 0.60 | 21.15 | 0.97 |
| gaussian-v1 / lamino-64-180 | 0.62 | 1.11 | 1.35 |
| gaussian-v1 / parallel-128-180 | 0.50 | 5.20 | 1.20 |
| gaussian-v1 / anisotropic-128-180 | 0.61 | 1.33 | 0.70 |
| gaussian-v1 / lamino-128-180 | 1.22 | 4.51 | 2.01 |
| gaussian-v1 / parallel-256-180 | 0.77 | 4.21 | 1.74 |
| gaussian-v1 / anisotropic-256-180 | 0.65 | 1.85 | 0.41 |
| gaussian-v1 / lamino-256-180 | 2.30 | 3.28 | 2.24 |
| structured-v1 / parallel-64-180 | 0.45 | 8.19 | 1.04 |
| structured-v1 / anisotropic-64-180 | — | — | — |
| structured-v1 / lamino-64-180 | 0.22 | 1.09 | 1.35 |
| structured-v1 / parallel-128-180 | 0.50 | 5.35 | 1.20 |
| structured-v1 / anisotropic-128-180 | 0.59 | 1.28 | 0.70 |
| structured-v1 / lamino-128-180 | 0.33 | 0.62 | 1.06 |
| structured-v1 / parallel-256-180 | 0.78 | 4.30 | 1.74 |
| structured-v1 / anisotropic-256-180 | 0.63 | 1.95 | 0.41 |
| structured-v1 / lamino-256-180 | 0.66 | 0.70 | 2.23 |
| structured-noisy-v1 / parallel-64-180 | 0.48 | 8.37 | 1.04 |
| structured-noisy-v1 / anisotropic-64-180 | 0.58 | 5.48 | 0.80 |
| structured-noisy-v1 / lamino-64-180 | 0.21 | 1.07 | 1.35 |
| structured-noisy-v1 / parallel-128-180 | 0.49 | 5.39 | 1.20 |
| structured-noisy-v1 / anisotropic-128-180 | 0.60 | 1.31 | 0.70 |
| structured-noisy-v1 / lamino-128-180 | 0.29 | 0.96 | 1.97 |
| structured-noisy-v1 / parallel-256-180 | 0.75 | 4.01 | 1.74 |
| structured-noisy-v1 / anisotropic-256-180 | 0.60 | 1.82 | 0.41 |
| structured-noisy-v1 / lamino-256-180 | 0.65 | 0.90 | 2.23 |

## Failed attempts

These times describe attempts, never successful recovery. Failed methods repeat their last attempted budget seven times; every warm quality sample is retained. Unsupported adapters are listed separately in the raw files and have no meaningful solve latency or process-memory measurement.

| Suite / cell | Method | Status | Last budget | Cold attempted / warm attempted ms | Last L2 / target | GPU MiB |
|---|---|---|---:|---:|---:|---:|
| gaussian-v1 / lamino-64-180 | ASTRA BP3D + CuPy filter | target_not_reached | 1 | 344.10 / 3.97 | 0.566742 / 0.10 | 188 |
| gaussian-v1 / lamino-64-180 | ASTRA SIRT | target_not_reached | 256 | 1374.91 / 532.09 | 0.221506 / 0.10 | 158 |
| gaussian-v1 / lamino-128-180 | ASTRA BP3D + CuPy filter | target_not_reached | 1 | 399.81 / 24.38 | 0.568928 / 0.10 | 320 |
| gaussian-v1 / lamino-128-180 | ASTRA SIRT | target_not_reached | 256 | 6004.18 / 2694.48 | 0.222726 / 0.10 | 202 |
| gaussian-v1 / lamino-256-180 | Joseph CGLS | target_not_reached | 256 | 88846.83 / 44695.93 | 0.109139 / 0.10 | 1166 |
| gaussian-v1 / lamino-256-180 | ASTRA BP3D + CuPy filter | target_not_reached | 1 | 717.12 / 283.23 | 0.569485 / 0.10 | 684 |
| gaussian-v1 / lamino-256-180 | ASTRA SIRT | target_not_reached | 256 | 36977.90 / 16771.08 | 0.223661 / 0.10 | 524 |
| structured-v1 / anisotropic-64-180 | Fourier (CuPy) | target_not_reached | 1 | 637.25 / 1.56 | 0.177969 / 0.15 | 146 |
| structured-v1 / anisotropic-64-180 | Joseph CGLS | target_not_reached | 256 | 2187.28 / 281.89 | 0.359880 / 0.15 | 212 |
| structured-v1 / anisotropic-64-180 | Coarse-to-fine Joseph CGLS | target_not_reached | 256 | 4463.22 / 92.62 | 0.280919 / 0.15 | 284 |
| structured-v1 / anisotropic-64-180 | FBP + Joseph CGLS | target_not_reached | 256 | 2398.50 / 264.47 | 0.361210 / 0.15 | 228 |
| structured-v1 / anisotropic-64-180 | ASTRA BP3D + CuPy filter | target_not_reached | 1 | 362.57 / 3.32 | 0.175930 / 0.15 | 168 |
| structured-v1 / anisotropic-64-180 | ASTRA FBP + CGLS | target_not_reached | 256 | 1250.53 / 420.94 | 0.630801 / 0.15 | 186 |
| structured-v1 / anisotropic-64-180 | ASTRA CGLS | target_not_reached | 256 | 1136.10 / 419.06 | 0.906271 / 0.15 | 150 |
| structured-v1 / anisotropic-64-180 | ASTRA SIRT | target_not_reached | 256 | 1019.95 / 358.96 | 0.211675 / 0.15 | 150 |
| structured-v1 / lamino-64-180 | ASTRA BP3D + CuPy filter | target_not_reached | 1 | 357.18 / 3.97 | 0.490169 / 0.30 | 188 |
| structured-v1 / lamino-128-180 | ASTRA BP3D + CuPy filter | target_not_reached | 1 | 384.31 / 24.33 | 0.485728 / 0.30 | 320 |
| structured-v1 / lamino-256-180 | ASTRA BP3D + CuPy filter | target_not_reached | 1 | 680.30 / 280.36 | 0.484764 / 0.30 | 684 |
| structured-noisy-v1 / anisotropic-64-180 | Joseph CGLS | target_not_reached | 256 | 2128.36 / 280.45 | 0.389214 / 0.18 | 212 |
| structured-noisy-v1 / anisotropic-64-180 | Coarse-to-fine Joseph CGLS | target_not_reached | 256 | 3972.66 / 92.71 | 0.286208 / 0.18 | 284 |
| structured-noisy-v1 / anisotropic-64-180 | ASTRA CGLS | target_not_reached | 256 | 1119.68 / 417.07 | 1.390727 / 0.18 | 150 |
| structured-noisy-v1 / anisotropic-64-180 | ASTRA SIRT | target_not_reached | 256 | 1013.41 / 358.19 | 0.214500 / 0.18 | 150 |
| structured-noisy-v1 / lamino-64-180 | ASTRA BP3D + CuPy filter | target_not_reached | 1 | 349.63 / 3.95 | 0.490385 / 0.32 | 182 |
| structured-noisy-v1 / lamino-128-180 | ASTRA BP3D + CuPy filter | target_not_reached | 1 | 380.32 / 23.66 | 0.486574 / 0.32 | 320 |
| structured-noisy-v1 / lamino-256-180 | ASTRA BP3D + CuPy filter | target_not_reached | 1 | 695.08 / 277.99 | 0.488165 / 0.32 | 684 |

The real-data, free-voxel alignment, derivative-cost and large 720-view controls are separate requirements. This matrix alone cannot establish those goals. In particular, restricted Gaussian-mixture alignment contributes no evidence toward public free-voxel recovery.
