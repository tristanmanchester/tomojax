# Public free-voxel alignment baseline, 2026-10-04

All six scheduled cells completed one fresh-process and seven repeated public alignment calls. No cell met the joint acceptance gates. These are failed-attempt times; time to successful recovery and the required 20× denominator remain undefined.

The [frozen pilot specification](public-free-voxel-pilot.md) defines the independent random-voxel objects, 61 irregular angles, physical motion, noise and unchanged gates. Every voxel and all five per-view pose parameters are optimized from zero volume and nominal geometry. Verification permits one common rigid object frame, applied to both volume and poses; it permits no per-view registration or amplitude fitting.

The run used the public `tomojax.alignment.align` API, 64 outer iterations, 20 reconstruction iterations per outer step, FP32 sampling, central-difference Gauss–Newton, positivity, and no TV penalty. All unsuccessful calls exhausted the declared outer budget.

Cold timing starts before worker launch and includes imports, data loading, setup, transfers, compilation and every verification. Warm timing includes the same complete API solve and verification in the initialized process. Independent fixture generation precedes the worker. Per-PID GPU memory was sampled every requested 10 ms across the entire worker; brief allocation peaks can be missed. GPU jobs were serialized. CPU validation and diagnostic jobs ran separately during parts of this queue.

| Cell | Status | Cold s | Warm median [min, max] s | Image relative L2 | Rotation RMSE ° | Translation vector RMSE px | Peak GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| parallel-clean | target_not_reached | 64.460 | 61.022 [60.727, 61.495] | 0.010887 | 0.078435 | 0.004006 | 444 |
| parallel-noisy | target_not_reached | 64.486 | 61.278 [61.183, 61.830] | 0.011062 | 0.078196 | 0.003985 | 444 |
| anisotropic-clean | target_not_reached | 50.647 | 45.787 [45.756, 47.091] | 0.019529 | 0.570186 | 0.017940 | 444 |
| anisotropic-noisy | target_not_reached | 50.700 | 46.421 [45.840, 47.287] | 0.019491 | 0.571898 | 0.017825 | 444 |
| lamino-clean | target_not_reached | 64.565 | 60.909 [60.837, 61.189] | 0.144337 | 0.462657 | 0.004291 | 444 |
| lamino-noisy | target_not_reached | 65.180 | 60.919 [60.725, 61.159] | 0.144366 | 0.465505 | 0.004295 | 444 |

Quality columns show the cold attempt. The [complete raw record](../../bench/reference/public-free-voxel-v1-baseline.json.gz) retains every repeat, per-outer quality check, stopping decision, step bound and actual backend. Image gates are 0.10 parallel/anisotropic and 0.20 tilted; rotation is 0.01° and translation is 0.05 native pixels. Passing image and translation gates does not override a failed rotation gate.

Both reconstruction and pose evaluation actually ran on **JAX CUDA**. Pallas was requested, but reconstruction reported `dynamic_geometry_alignment_uses_jax_core`; pose objectives required gradient-safe projector semantics. This is GPU execution, not a CPU fallback, and it is not evidence of Pallas performance.

The [launch source archive](../../bench/reference/public-free-voxel-v1-baseline-source.tar.gz) and [source digest](../../bench/reference/public-free-voxel-v1-baseline-source.json) preserve the code used for these attempts. Its source hash was unchanged at completion. The active checkout has subsequent step-bound corrections; those changes do not retroactively alter this baseline.

Histories exposed an effective FISTA bound multiplied by 1.2 at every outer iteration. This eventually suppressed voxel updates. Separate oracle-volume diagnostics also exposed uniform-ray-step integration bias against the independent trilinear-basis data. Fixing either limiter still requires another complete public free-voxel measurement before any recovery-speed claim.

The pilot uses small motion and does not establish identifiability or 99% recovery on the declared larger-motion noisy distribution. The separate [reconstruction matrix](system-matrix-2026-10-04.md) contains the ASTRA/TIGRE comparisons; fixed-geometry reconstruction times cannot replace the missing successful joint-recovery denominator.

## Step-bound correction diagnostic

A subsequent frozen diagnostic corrected repeated step-bound inflation, initial physical operator scaling and fallback TV double-counting. It kept the sampled projector and all six fixtures, gates and iteration budgets. Each cell ran one cold and one warm call; this is a diagnostic, not a seven-repeat headline baseline. All 12 calls still failed the rotation gate. Image and translation errors decreased across all six cells, while parallel rotation error increased. The corrections remain correctness fixes; this performance line was stopped rather than tuned further.

| Cell | Status | Cold s | Warm s | Image relative L2 | Rotation RMSE ° | Translation vector RMSE px | Peak GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| parallel-clean | target_not_reached | 64.179 | 60.363 | 0.008869 | 0.097590 | 0.002695 | 444 |
| parallel-noisy | target_not_reached | 64.045 | 60.493 | 0.009593 | 0.098552 | 0.002731 | 444 |
| anisotropic-clean | target_not_reached | 48.259 | 44.763 | 0.009800 | 0.323083 | 0.008839 | 444 |
| anisotropic-noisy | target_not_reached | 48.350 | 44.950 | 0.010119 | 0.320626 | 0.008917 | 444 |
| lamino-clean | target_not_reached | 63.202 | 59.494 | 0.121312 | 0.245558 | 0.003167 | 444 |
| lamino-noisy | target_not_reached | 63.090 | 59.821 | 0.121388 | 0.252345 | 0.003217 | 444 |

The [complete diagnostic record](../../bench/reference/public-free-voxel-v1-bound-correction.json.gz) and [frozen source](../../bench/reference/public-free-voxel-v1-bound-correction-source.tar.gz) preserve both calls, every outer iteration and backend metadata. This run used JAX CUDA for reconstruction and pose scoring. It does not supply a successful-recovery denominator.
