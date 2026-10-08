# Multi-GPU reconstruction and alignment: a literature review for TomoJAX

October 2026. Four researchers surveyed one topic each: how to split the projectors,
distributed iterative algorithms, alignment at scale, and multi-device JAX. They checked
citations against the arXiv, DOI or publisher pages. Where a source could not be
confirmed it is marked *unverified*. Reported numbers are the authors', not
re-measured.

## The short answer

Split the **views** across GPUs first and keep a full copy of the volume on each.
Every GPU projects its own views, and the backprojections are summed. FISTA and CGLS
are unchanged, and the transpose stays exact: the sum of each GPU's exact partial
transpose is the exact transpose. Only the order of the summation changes. Alignment
splits the same way, because each view's pose Jacobian depends only on that view and
the volume.

Split the **volume into slabs** later, and only for volumes that do not fit on one
GPU. That is TIGRE's and MBIRJAX's design. It costs more: cone-beam slabs read
overlapping detector row bands, the forward projection must merge them, and TV needs
a one-voxel overlap ("halo") between neighbouring slabs. That cost is worth paying
only for memory.

Nobody has published multi-GPU joint 6-DOF pose alignment, or multi-orbit cone
alignment. TomoJAX would be the first to report scaling there.

## How the projectors are split elsewhere

- **TIGRE** (Biguri et al., arXiv:1905.03748, J. Parallel Distrib. Comput. 2020;
  TIGRE v3, arXiv:2412.10129).
  - Forward projection splits the views across GPUs, each with the full image. When
    the image does not fit, it is cut into axial stacks and the partial projections
    are summed on the host. Backprojection gives each GPU part of the image, and
    every GPU streams all projections.
  - Only two projection buffers are held per GPU, so copies overlap with kernels.
    Pinned memory raised transfers from about 4 to 12 GB/s.
  - Forward projection scales almost perfectly: 50%, 33% and 25% of the one-GPU
    time on 2, 3 and 4 GPUs. Backprojection scales sublinearly, because its kernel
    is fast enough that memory management dominates: over 50% of the time at 512³.
  - A 3340×3340×900 volume ran 30 CGLS iterations in 4 h 21 min.
  - This is the baseline to beat. Its lesson applies to us: once backprojection is
    fast, transfers set the limit.
- **ASTRA.**
  - The distributed ASTRA (Palenstijn et al., 2016, doi:10.1186/s40679-016-0032-z) is a
    separate MPI extension. It cuts the volume into z-slabs, each owning a detector row band.
    Cone-beam backprojection then needs no communication, but forward projection must
    merge overlapping rows. SIRT stopped improving at about 16 GPUs for 1024³ at a 7.8°
    cone angle.
  - Whether stock ASTRA splits 3D work across GPUs is disputed between the researchers.
    One found ASTRA's NEWS.txt listing multi-GPU 3D FP/BP/FDK since 1.8, via
    `astra.set_gpu_index` with a list. The other found no documentation of it. Check
    before benchmarking.
- **iFDK** (Chen et al., SC'19, arXiv:1909.02724).
  - It uses an R×C grid of GPUs: views along columns, volume parts along rows.
  - Filtering, communication and backprojection are pipelined.
  - It reconstructed 4K in about 30 s on 2,048 V100s.
  - The follow-up, Scalable FBP Decomposition (SC'21, doi:10.1145/3458817.3476139),
    reconstructs 4096³ in under 16 s on 1,024 GPUs, bound by storage.
  - This is the template if TomoJAX ever needs views × slabs together.
- **MBIRJAX** (docs only, <https://mbirjax.readthedocs.io/en/latest/dev_sharding_overview.html>).
  - It is the only JAX design found, and is now legacy, replaced by MBIRTorch.
  - The volume is split by slice and the sinogram by view. Forward projection
    broadcasts slice bands to the view owners, and backprojection reduce-scatters them
    back.
  - It runs one Python thread per GPU, in a single process.
  - It pads to a multiple of the device count, so results do not depend on it.
  - It reports about 2× on two GPUs. Its docs warn that `shard_map` gave wrong results
    and that `jax.device_put` between GPUs returned zeros, on an L40S but not an H100.
    That is their claim and was not reproduced, but it matters because our Modal GPU is
    an L40S.
- **Others.**
  - LEAP (arXiv:2307.05801) is multi-GPU via `set_gpus`, with no published scaling.
  - TomocuPy (arXiv:2209.08450) overlaps disk, transfer and compute to reconstruct 2048³
    in under 7 s on one A100. It is a model for transfer overlap, not for splitting
    across GPUs.
  - CIL delegates to ASTRA and TIGRE, tomosipo and ODL add nothing of their own, and RTK
    has no automatic multi-GPU support.

## Algorithms

- **The data term is a sum over views, so splitting views changes nothing in FISTA or
  CGLS.** Each iteration needs one volume-sized sum across GPUs. Petascale XCT
  (Hidayetoglu et al., arXiv:2009.07226) shows this sum dominates at scale. It
  reconstructed 9K×11K×11K on 24,576 Summit GPUs in under 3 minutes. A three-level
  sum (within a socket, a node, then across nodes) cut inter-node traffic by 58–64%.
  It used half precision, which we would not.
- **Slabs plus TV.** TIGRE keeps a halo as deep as the number of inner TV iterations it
  runs between exchanges, so the result stays exact.
  Kumar and Donatelli (arXiv:2603.28756, March 2026) split along the rotation axis,
  exchanging a one-voxel halo for the prior after every iteration. They report
  2048×2447×2447 in about 30 min on 16 Perlmutter nodes and under 9 min on 128. That
  is 3.4× for 8× the hardware, weaker than the abstract suggests. Coarse-to-fine
  levels cut their run time about 5×, and an FBP start saved 20–25 iterations.
- **SPDHG maps naturally to one view subset per GPU** (Chambolle et al.,
  arXiv:1706.04957). A fixed per-GPU assignment is a valid sampling scheme
  (Gutierrez et al., arXiv:2207.12291), and step sizes can adapt (arXiv:2301.02511).
  But it changes the iterates, and its dual state costs a sinogram per subset.
- **Avoid for now**, because each trades exactness or reproducibility for speed:
  - consensus methods that solve inner problems per node (MACE, arXiv:1911.09278:
    4.55× on 1200 nodes);
  - quantised exchange (arXiv:2410.06106);
  - mixed-precision communication;
  - pipelined CG (arXiv:1905.06850), which reorders floating-point work and loses
    accuracy.

## Alignment

- **Nikitin et al., distributed nonrigid nano-tomography** (arXiv:2008.03375, IEEE TCI
  2021) is the closest precedent.
  - ADMM alternates optical-flow deformation, split by angle, with tomography, split by
    slice, on 4 P100s.
  - 4 GPUs gave at most 2× over one, because host–device transfers dominated.
  - Their schedule stops at 2×2 binning. That matches our walnut result on an L40S: the
    full-resolution level moved no orbit's height by more than a micrometre and cost 5×
    the time.
- **PtyGer** (Yu et al., arXiv:2106.07575) runs CG for ptychography split by scan
  position across 8 GPUs. It exchanges halos with neighbours and all-reduces only for
  the line search, and gets up to 1.7× per doubling. It is a template for a CG solver
  that needs only small all-reduces.
- **MegBA** (Ren et al., arXiv:2112.01349, ECCV 2022) is distributed bundle adjustment.
  It has the same structure as ours: many small per-view blocks coupled to one large
  shared block. It solves this with distributed preconditioned CG and Schur
  elimination, matching single-node precision. It is the best model for splitting the
  Gauss–Newton step.
- Cryo-ET tools (AreTomo, doi:10.1101/2022.02.15.480593, and successors) use one GPU per
  tilt series for throughput. They do not split one problem across GPUs.
- SAMCIRT and rMIRT (arXiv:2402.04480, 2301.11029) support the coupled update with
  exact adjoints, but neither is multi-GPU.

What this means for TomoJAX's coupled Gauss–Newton step:
- Per-view Jacobians and their 6×6 blocks are local to each GPU, and the 6N pose vector
  is tiny and replicated.
- The volume side needs a volume-sized sum on every CG iteration. That is the cost to
  measure.
- MegBA's Schur approach might keep most of the work local; that is an open question.

## Engineering with JAX

The JAX researcher checked these against the installed JAX 0.11.2, the jaxlib source,
and runs on 4 simulated CPU devices and the RTX 4070.

- `pmap` is in maintenance mode and its C++ path was removed in 0.10. **`shard_map` is
  the tool.**
- An opaque FFI or `buffer_callback` call under plain `jit` with sharded inputs is not
  split. XLA gathers everything onto every device and runs the call redundantly
  (reproduced). Inside `shard_map`, the callback runs once per shard, on separate
  threads.
- The callback's `ExecutionContext` gives a stream but no device number. Each buffer
  does carry its device, through `__dlpack_device__()` and `__cuda_array_interface__`
  (checked).
- What our CuPy kernels therefore need:
  - Read the device number from a buffer.
  - Enter `cp.cuda.Device(ordinal)` before the stream context.
  - Allocate all scratch inside that.
  - Drop the `jax.devices()[0]` checks in the kernel-availability gating.
  - Key per-device state by ordinal: the FDK cuBLAS handle and the texture objects.
    CuPy already caches compiled modules per device.
- **Python holds the GIL through each callback.** Launches are serialised across GPUs.
  That is harmless while callbacks only enqueue asynchronous work, but any
  synchronisation inside one stalls every GPU.
- Put `custom_vjp` outside the `shard_map`, so the backward pass is our explicit
  adjoint `shard_map` rather than collectives JAX derives itself (untested).
- **Interconnect cost.** An all-reduce moves about 2V per GPU for a V-byte volume. My
  estimate for the walnut is 501³ × 4 B ≈ 0.5 GB, so about 1 GB per GPU per transpose.
  That is roughly 80 ms on PCIe at 12 GB/s, and a few ms on NVLink. The
  single-GPU binned transpose takes about 3 s, or about 0.8 s split four ways. PCIe
  would add about 10%, and the sum is cheap on NVLink. GeForce cards have direct
  GPU-to-GPU copies disabled in recent drivers, so rent datacentre cards.
- **Reproducibility.** NCCL's ring order is fixed by topology, so results repeat run
  to run on the same machine (from a search result, not checked in the NCCL source).
  They will differ at round-off between GPU counts, unless we sum in a fixed tree
  ourselves.

## Proposed plan

1. **Device-correct kernels (no new hardware needed).**
   - Each CuPy launch resolves its device from its buffers, and per-device state is
     keyed by device.
   - The `jax.devices()[0]` assumptions go.
   - Test the sharding with `JAX_PLATFORMS=cpu` and
     `XLA_FLAGS=--xla_force_host_platform_device_count=4`, against single-device
     results.
2. **View-split operators.**
   - Wrap forward projection and the transpose in `shard_map`: views sharded, volume
     replicated, a `psum` after the transpose.
   - Pad the view count to a multiple of the device count with inert views.
   - FISTA, CGLS and FDK then work on several GPUs unchanged, and so does the
     alignment's view loop.
   - Give users a way to choose devices (for example `devices=` or a mesh), defaulting
     to one GPU so nothing changes for current users.
3. **Check on real GPUs (Modal, short runs).**
   - Kernel results on GPU 1 versus GPU 0.
   - Scaling on 2 and 4 GPUs.
   - Prefer H100s over L40Ss, given MBIRJAX's L40S warnings.
   - Benchmark against TIGRE on the same GPUs, and ASTRA's multi-GPU mode once it is
     confirmed to exist.
4. **Slabs, for volumes beyond one GPU.**
   - Follow TIGRE's double-buffered streaming and MBIRJAX's slice/view layout.
   - Use a one-voxel TV halo exchanged every iteration, so FISTA stays exact.
   - This extends the host-slab design `fbp_host` already uses to iterative methods.
5. **Alignment.**
   - After step 2, measure whether the volume sum in the coupled CG step limits
     scaling.
   - Look at MegBA-style Schur elimination only if it does.
   - Keep stopping before full resolution by default: our walnut run and Nikitin et al.
     both support it.

## Open questions

- Does XLA set the CUDA current device on the callback thread? If so, entering the
  device context is only a safeguard.
- Do the per-device launches actually run concurrently at 4–8 GPUs, given that the GIL
  serialises them?
- Is MBIRJAX's L40S problem real, and does it affect buffer callbacks?
- Laminography slabs do not map to detector row bands (`fbp_host` uses x-slabs there).
  Is their forward-projection overlap acceptable?
- Is our transpose kernel bit-reproducible on one GPU? Its atomics bound any
  cross-device guarantee.
