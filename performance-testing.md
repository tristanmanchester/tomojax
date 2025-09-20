## Baseline

```
✨ Pixi task (align): python -m tomojax.cli.align --data data/sim_misaligned.nxs --levels 1 --outer-iters 4 --recon-iters 10 --lambda-tv 0.003 --opt-method gn --gn-damping 1e-3 --gather-dtype bf16 --checkpoint-projector --log-summary --out out/align_misaligned.nxs --progress
INFO:2025-09-20 06:29:56,050:jax._src.xla_bridge:925: Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
2025-09-20 06:29:56,050 | INFO | Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
INFO:2025-09-20 06:29:56,051:jax._src.xla_bridge:925: Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
2025-09-20 06:29:56,051 | INFO | Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
2025-09-20 06:29:56,061 | INFO | JAX backend: gpu
2025-09-20 06:29:56,061 | INFO | Devices: [CudaDevice(id=0)]
Align: outer iters:   0%|                                         | 0/4 [00:00<?, ?it/s]2025-09-20 06:32:55,567 | INFO | Outer 1/4 | total 2m58.7s | elapsed 2m58.9s
2025-09-20 06:32:55,567 | INFO |   Recon | time 2m45.7s | L 4.573e+04->5.488e+04 | loss 9.743e+07->5.104e+06 (min 5.104e+06)
2025-09-20 06:32:55,567 | INFO |   Align | time 12.2s | |drot|_mean 1.640e-02 rad | |dtrans|_mean 4.978e+00 | loss 5.086e+06->1.259e+06 (Δ -3.826e+06, -75.24%)
Align: outer iters:  25%|████████                        | 1/4 [02:58<08:56, 178.68s/it]2025-09-20 06:35:09,174 | INFO | Outer 2/4 | total 2m13.6s | elapsed 5m12.5s
2025-09-20 06:35:09,174 | INFO |   Recon | time 2m06.0s | L 5.488e+04->6.585e+04 | loss 1.260e+06->3.144e+05 (min 3.144e+05)
2025-09-20 06:35:09,174 | INFO |   Align | time 6.8s | |drot|_mean 4.997e-03 rad | |dtrans|_mean 5.219e-01 | loss 3.027e+05->2.485e+05 (Δ -5.422e+04, -17.91%)
Align: outer iters:  50%|████████████████                | 2/4 [05:12<05:04, 152.17s/it]2025-09-20 06:37:23,253 | INFO | Outer 3/4 | total 2m14.1s | elapsed 7m26.6s
2025-09-20 06:37:23,253 | INFO |   Recon | time 2m06.6s | L 6.585e+04->7.902e+04 | loss 2.486e+05->1.224e+05 (min 1.224e+05)
2025-09-20 06:37:23,253 | INFO |   Align | time 6.7s | |drot|_mean 9.055e-04 rad | |dtrans|_mean 9.973e-02 | loss 1.187e+05->1.167e+05 (Δ -2.005e+03, -1.69%)
Align: outer iters:  75%|████████████████████████        | 3/4 [07:26<02:23, 143.91s/it]2025-09-20 06:39:38,054 | INFO | Outer 4/4 | total 2m14.8s | elapsed 9m41.4s
2025-09-20 06:39:38,054 | INFO |   Recon | time 2m07.2s | L 7.902e+04->9.483e+04 | loss 1.167e+05->7.409e+04 (min 7.409e+04)
2025-09-20 06:39:38,054 | INFO |   Align | time 6.8s | |drot|_mean 3.113e-04 rad | |dtrans|_mean 4.601e-02 | loss 7.248e+04->7.214e+04 (Δ -3.374e+02, -0.47%)
2025-09-20 06:39:38,054 | INFO | Alignment completed in 9m41.4s (recon 9m05.6s, align 32.4s over 4 outer iters)
2025-09-20 06:39:38,055 | INFO |   Loss 5.086e+06 -> 7.214e+04 (Δ -5.014e+06, -98.58%)
2025-09-20 06:39:39,287 | INFO | Saved alignment results to out/align_misaligned.nxs
```


## Test 1
- Recon barrier after FISTA return:
  - File: src/tomojax/align/pipeline.py:304
  - Added x.block_until_ready() (with a safe fallback to jax.block_until_ready(x))
before computing recon_time.
- Align barrier before capturing align_time:
  - File: src/tomojax/align/pipeline.py:372
  - Added jax.block_until_ready(params5) just before computing stat["align_time"].

  ```
✨ Pixi task (align): python -m tomojax.cli.align --data data/sim_misaligned.nxs --levels 1 --outer-iters 4 --recon-iters 10 --lambda-tv 0.003 --opt-method gn --gn-damping 1e-3 --gather-dtype bf16 --checkpoint-projector --log-summary --out out/align_misaligned.nxs --progress
INFO:2025-09-20 06:57:07,670:jax._src.xla_bridge:925: Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
2025-09-20 06:57:07,670 | INFO | Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
INFO:2025-09-20 06:57:07,671:jax._src.xla_bridge:925: Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
2025-09-20 06:57:07,671 | INFO | Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
2025-09-20 06:57:07,680 | INFO | JAX backend: gpu
2025-09-20 06:57:07,680 | INFO | Devices: [CudaDevice(id=0)]
Align: outer iters:   0%|                                         | 0/4 [00:00<?, ?it/s]2025-09-20 07:00:06,893 | INFO | Outer 1/4 | total 2m58.6s | elapsed 2m58.6s
2025-09-20 07:00:06,893 | INFO |   Recon | time 2m45.6s | L 4.573e+04->5.488e+04 | loss 9.743e+07->5.104e+06 (min 5.104e+06)
2025-09-20 07:00:06,893 | INFO |   Align | time 12.2s | |drot|_mean 1.640e-02 rad | |dtrans|_mean 4.978e+00 | loss 5.086e+06->1.259e+06 (Δ -3.826e+06, -75.24%)
Align: outer iters:  25%|████████                        | 1/4 [02:58<08:55, 178.58s/it]2025-09-20 07:02:22,913 | INFO | Outer 2/4 | total 2m16.0s | elapsed 5m14.6s
2025-09-20 07:02:22,913 | INFO |   Recon | time 2m08.5s | L 5.488e+04->6.585e+04 | loss 1.260e+06->3.144e+05 (min 3.144e+05)
2025-09-20 07:02:22,913 | INFO |   Align | time 6.8s | |drot|_mean 4.997e-03 rad | |dtrans|_mean 5.222e-01 | loss 3.027e+05->2.485e+05 (Δ -5.422e+04, -17.91%)
Align: outer iters:  50%|████████████████                | 2/4 [05:14<05:07, 153.54s/it]2025-09-20 07:04:39,326 | INFO | Outer 3/4 | total 2m16.4s | elapsed 7m31.0s
2025-09-20 07:04:39,326 | INFO |   Recon | time 2m08.7s | L 6.585e+04->7.902e+04 | loss 2.486e+05->1.224e+05 (min 1.224e+05)
2025-09-20 07:04:39,326 | INFO |   Align | time 6.9s | |drot|_mean 9.055e-04 rad | |dtrans|_mean 1.000e-01 | loss 1.187e+05->1.167e+05 (Δ -2.006e+03, -1.69%)
Align: outer iters:  75%|████████████████████████        | 3/4 [07:31<02:25, 145.72s/it]2025-09-20 07:06:56,655 | INFO | Outer 4/4 | total 2m17.3s | elapsed 9m48.4s
2025-09-20 07:06:56,655 | INFO |   Recon | time 2m09.6s | L 7.902e+04->9.483e+04 | loss 1.167e+05->7.409e+04 (min 7.409e+04)
2025-09-20 07:06:56,655 | INFO |   Align | time 6.9s | |drot|_mean 3.114e-04 rad | |dtrans|_mean 4.715e-02 | loss 7.248e+04->7.214e+04 (Δ -3.382e+02, -0.47%)
2025-09-20 07:06:56,656 | INFO | Alignment completed in 9m48.4s (recon 9m12.4s, align 32.8s over 4 outer iters)
2025-09-20 07:06:56,656 | INFO |   Loss 5.086e+06 -> 7.214e+04 (Δ -5.014e+06, -98.58%)
2025-09-20 07:06:57,892 | INFO | Saved alignment results to out/align_misaligned.nxs
  ```

## Test 2
- Replaced _gn_update_one to use Jv/J^T v operators with a 10‑step 5D CG solve,
avoiding explicit J and H allocations.
  - File: src/tomojax/align/pipeline.py:165
  - Keeps memory O(5 + nv·nu) and reduces kernel count.
  - Uses cfg.gn_damping as Tikhonov damping.


```
✨ Pixi task (align): python -m tomojax.cli.align --data data/sim_misaligned.nxs --levels 1 --outer-iters 4 --recon-iters 10 --lambda-tv 0.003 --opt-method gn --gn-damping 1e-3 --gather-dtype bf16 --checkpoint-projector --log-summary --out out/align_misaligned.nxs --progress
INFO:2025-09-20 07:14:53,103:jax._src.xla_bridge:925: Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
2025-09-20 07:14:53,103 | INFO | Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
INFO:2025-09-20 07:14:53,104:jax._src.xla_bridge:925: Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
2025-09-20 07:14:53,104 | INFO | Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
2025-09-20 07:14:53,114 | INFO | JAX backend: gpu
2025-09-20 07:14:53,114 | INFO | Devices: [CudaDevice(id=0)]
Align: outer iters:   0%|                                         | 0/4 [00:00<?, ?it/s]2025-09-20 07:18:25,053 | INFO | Outer 1/4 | total 3m31.3s | elapsed 3m31.3s
2025-09-20 07:18:25,053 | INFO |   Recon | time 2m46.6s | L 4.573e+04->5.488e+04 | loss 9.743e+07->5.104e+06 (min 5.104e+06)
2025-09-20 07:18:25,054 | INFO |   Align | time 43.9s | |drot|_mean 1.640e-02 rad | |dtrans|_mean 4.978e+00 | loss 5.086e+06->1.259e+06 (Δ -3.826e+06, -75.24%)
Align: outer iters:  25%|████████                        | 1/4 [03:31<10:33, 211.28s/it]2025-09-20 07:21:10,391 | INFO | Outer 2/4 | total 2m45.3s | elapsed 6m16.6s
2025-09-20 07:21:10,391 | INFO |   Recon | time 2m08.8s | L 5.488e+04->6.585e+04 | loss 1.260e+06->3.144e+05 (min 3.144e+05)
2025-09-20 07:21:10,391 | INFO |   Align | time 35.8s | |drot|_mean 4.997e-03 rad | |dtrans|_mean 5.221e-01 | loss 3.027e+05->2.485e+05 (Δ -5.422e+04, -17.91%)
Align: outer iters:  50%|████████████████                | 2/4 [06:16<06:08, 184.25s/it]2025-09-20 07:23:55,812 | INFO | Outer 3/4 | total 2m45.4s | elapsed 9m02.1s
2025-09-20 07:23:55,812 | INFO |   Recon | time 2m08.9s | L 6.585e+04->7.902e+04 | loss 2.486e+05->1.224e+05 (min 1.224e+05)
2025-09-20 07:23:55,812 | INFO |   Align | time 35.7s | |drot|_mean 9.055e-04 rad | |dtrans|_mean 9.995e-02 | loss 1.187e+05->1.167e+05 (Δ -2.005e+03, -1.69%)
Align: outer iters:  75%|████████████████████████        | 3/4 [09:02<02:55, 175.65s/it]2025-09-20 07:26:40,884 | INFO | Outer 4/4 | total 2m45.1s | elapsed 11m47.1s
2025-09-20 07:26:40,884 | INFO |   Recon | time 2m08.7s | L 7.902e+04->9.483e+04 | loss 1.167e+05->7.409e+04 (min 7.409e+04)
2025-09-20 07:26:40,884 | INFO |   Align | time 35.5s | |drot|_mean 3.114e-04 rad | |dtrans|_mean 4.730e-02 | loss 7.248e+04->7.214e+04 (Δ -3.382e+02, -0.47%)
2025-09-20 07:26:40,885 | INFO | Alignment completed in 11m47.1s (recon 9m13.0s, align 2m31.0s over 4 outer iters)
2025-09-20 07:26:40,885 | INFO |   Loss 5.086e+06 -> 7.214e+04 (Δ -5.014e+06, -98.58%)
2025-09-20 07:26:42,233 | INFO | Saved alignment results to out/align_misaligned.nxs
```


## Test 3
- Alignment gradient now donates params5 into the JIT:
  - File: src/tomojax/align/pipeline.py:152
  - Replaced grad_all = jax.jit(jax.grad(...)) with:
      - @jax.jit(donate_argnums=(0,))
      - def grad_all(params5, vol): return jax.grad(align_loss, argnums=0)(params5,
vol)
- FISTA already donates z in val_and_grad (confirmed in src/tomojax/recon/
fista_tv.py:310); left unchanged.

```
✨ Pixi task (align): python -m tomojax.cli.align --data data/sim_misaligned.nxs --levels 1 --outer-iters 4 --recon-iters 10 --lambda-tv 0.003 --opt-method gn --gn-damping 1e-3 --gather-dtype bf16 --checkpoint-projector --log-summary --out out/align_misaligned.nxs --progress
INFO:2025-09-20 07:38:52,430:jax._src.xla_bridge:925: Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
2025-09-20 07:38:52,430 | INFO | Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
INFO:2025-09-20 07:38:52,431:jax._src.xla_bridge:925: Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
2025-09-20 07:38:52,431 | INFO | Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
2025-09-20 07:38:52,441 | INFO | JAX backend: gpu
2025-09-20 07:38:52,441 | INFO | Devices: [CudaDevice(id=0)]
Align: outer iters:   0%|                                         | 0/4 [00:00<?, ?it/s]2025-09-20 07:41:54,245 | INFO | Outer 1/4 | total 3m01.2s | elapsed 3m01.2s
2025-09-20 07:41:54,245 | INFO |   Recon | time 2m48.2s | L 4.573e+04->5.488e+04 | loss 9.743e+07->5.104e+06 (min 5.104e+06)
2025-09-20 07:41:54,245 | INFO |   Align | time 12.2s | |drot|_mean 1.640e-02 rad | |dtrans|_mean 4.978e+00 | loss 5.086e+06->1.259e+06 (Δ -3.826e+06, -75.24%)
Align: outer iters:  25%|████████                        | 1/4 [03:01<09:03, 181.20s/it]2025-09-20 07:44:12,171 | INFO | Outer 2/4 | total 2m17.9s | elapsed 5m19.2s
2025-09-20 07:44:12,171 | INFO |   Recon | time 2m10.2s | L 5.488e+04->6.585e+04 | loss 1.260e+06->3.144e+05 (min 3.144e+05)
2025-09-20 07:44:12,171 | INFO |   Align | time 6.9s | |drot|_mean 4.997e-03 rad | |dtrans|_mean 5.221e-01 | loss 3.027e+05->2.485e+05 (Δ -5.422e+04, -17.91%)
Align: outer iters:  50%|████████████████                | 2/4 [05:19<05:11, 155.74s/it]2025-09-20 07:46:30,069 | INFO | Outer 3/4 | total 2m17.9s | elapsed 7m37.0s
2025-09-20 07:46:30,069 | INFO |   Recon | time 2m10.2s | L 6.585e+04->7.902e+04 | loss 2.486e+05->1.224e+05 (min 1.224e+05)
2025-09-20 07:46:30,069 | INFO |   Align | time 6.9s | |drot|_mean 9.055e-04 rad | |dtrans|_mean 9.993e-02 | loss 1.187e+05->1.167e+05 (Δ -2.005e+03, -1.69%)
Align: outer iters:  75%|████████████████████████        | 3/4 [07:37<02:27, 147.59s/it]2025-09-20 07:48:48,072 | INFO | Outer 4/4 | total 2m18.0s | elapsed 9m55.1s
2025-09-20 07:48:48,072 | INFO |   Recon | time 2m10.3s | L 7.902e+04->9.483e+04 | loss 1.167e+05->7.409e+04 (min 7.409e+04)
2025-09-20 07:48:48,073 | INFO |   Align | time 6.9s | |drot|_mean 3.113e-04 rad | |dtrans|_mean 4.583e-02 | loss 7.248e+04->7.214e+04 (Δ -3.377e+02, -0.47%)
2025-09-20 07:48:48,073 | INFO | Alignment completed in 9m55.1s (recon 9m18.9s, align 32.9s over 4 outer iters)
2025-09-20 07:48:48,073 | INFO |   Loss 5.086e+06 -> 7.214e+04 (Δ -5.014e+06, -98.58%)
2025-09-20 07:48:49,408 | INFO | Saved alignment results to out/align_misaligned.nxs
```


## Test 4
- Precompute reciprocals once:
  - inv_vx = jnp.float32(1.0 / grid.vx) (and inv_vy, inv_vz)
- Switch to incremental indices:
  - Compute ix0,iy0,iz0 from q0
  - Compute per-step deltas dix,diy,diz from dq
  - Carry (acc, ix, iy, iz) in scan and update by (dix, diy, diz) each step
- Removes per-step divides and some subtracts; numerics unchanged.


```

```											