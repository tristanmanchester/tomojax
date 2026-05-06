# 2026-05-06 Phase 8 Setup-Global GPU Ladder

Ran the realistic `synth128_setup_global_tomo` diagnostic on the laptop GPU
after isolating the 64^3/64-view memory blow-up.

## GPU

- JAX backend: `gpu`
- Selected JAX device: `cuda:0`
- Host GPU from `nvidia-smi`: NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB

JAX GPU required the Python CUDA wheel library paths in `LD_LIBRARY_PATH`; the
ambient `/usr/local/cuda-12.3/lib64` path could not initialize JAX CUDA because
cuSPARSE was not found by the plugin.

## Memory Isolation

Dataset: nuisance-free `synth128_setup_global_tomo`, 64^3 volume, view ladder
1/4/16/64, generated through the sidecar writer.

The original full benchmark failed in Schur finite differences with a 12.14 GiB
GPU allocation:

```text
RESOURCE_EXHAUSTED: Autotuning failed ... f32[194,64,64,64,64] ...
Out of memory while trying to allocate 12.14GiB.
```

Source: the shared finite-difference Jacobian evaluated all parameter
perturbations with a single `jax.vmap`, materializing parameter x view x volume
work arrays. Projector, backprojector, one FISTA step, nuisance fitting, and
single Schur updates all passed after changing finite differences to accumulate
columns sequentially.

| Views | Projector | Backprojector | FISTA 1 Iter | Schur Fixed Truth | Schur Stopped Volume | Schur Fixed Truth + Nuisance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.974s | 1.022s | 4.069s | 5.362s | 4.074s | 4.716s |
| 4 | 1.016s | 1.199s | 4.337s | 6.484s | 4.890s | 5.488s |
| 16 | 1.042s | 1.422s | 4.542s | 8.640s | 6.983s | 7.850s |
| 64 | 1.066s | 2.130s | 5.287s | 18.062s | 15.870s | 17.672s |

## Benchmark Results

Mode: `balanced`, nuisance disabled, existing sidecar ingestion path,
comparison rendered by `tomojax-synthetic-benchmark-compare`.

| Mode | Status | Criteria | Geometry | det_u RMSE px | det_v RMSE px | theta RMSE rad | Final Residual | Volume NMSE | Schur Accepted | Total Time s |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| fixed_synthetic_truth | failed | failed | failed | 6.9338 | 0.00666 | 0.02211 | 0.856277 | 0.686109 | true | 37.5096 |
| stopped_reconstruction | failed | failed | failed | 7.25 | 0 | 0.02182 | 0 | 0.686110 | true | 24.8489 |

## Interpretation

Fixed-truth also fails the setup-global recovery gate, so this ladder points to
setup/pose/theta coupling or geometry convention mapping rather than stopped
reconstruction or volume gauge handling alone. The stopped-reconstruction mode
does not improve `det_u`, while fixed-truth improves it only slightly and remains
far outside the manifest tolerance.

The 32^3/4-view smoke remains wiring/CI coverage only and should not be treated
as alignment-quality evidence.
