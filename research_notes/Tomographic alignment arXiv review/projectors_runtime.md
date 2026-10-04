# Differentiable tomographic projectors, exact voxel-basis integration, fast GPU operators, pose gradients, and JAX compile/startup overhead

Scope note: these notes cover papers, libraries and docs relevant to TomoJAX's projector/runtime work: the exact trilinear-basis integrator, matched CUDA forward/adjoint, pose Jacobians, cold-start compile cost and memory-bounded gradients. Evidence labels used below:
- **[PR]** peer-reviewed (journal or conference).
- **[arXiv]** preprint only.
- **[README/docs]** claims from project documentation, not independently measured.
- **[older]** foundational work from before 2022.

All arXiv IDs below were checked against the arXiv abstract page or arXiv PDF during this session unless marked "(ID from search snippet only)".

---

## 1. Differentiable CT/projector libraries and how they compute geometry/pose derivatives

### Takeaway
Most differentiable CT libraries (LEAP, tomosipo, TorchRadon, PYRO-NN, CTorch, diffct, TIGRE's PyTorch wrapper) differentiate only with respect to the **volume or sinogram**: the VJP of A is Aᵀ, so a gradient costs about one extra projector call. **Geometry/pose** derivatives come from two places. One is autodiff through a tensorised ray tracer (DiffDRR); it is memory-hungry and slow. The other is analytic derivatives that reuse the projector itself (Thies et al. for backprojection; Jiang/Stayman et al. for forward and back projection). The best measured result is Jiang et al. 2025 (arXiv 2508.13304, IEEE TBME): their analytic pose gradient used 0.77 GB against DiffDRR's 5.74 GB for a 256³ volume and was about 7.8× faster in 2D/3D registration. That is the most direct template for TomoJAX's unused fused analytic pose-normal kernel.

### Cited Findings

**Pose/geometry-gradient methods (most relevant to TomoJAX)**
- **Jiang, Wang, Uneri, Zbijewski, Stayman (2025), "Differentiable Forward and Back-Projector for Rigid Motion Estimation in X-ray Imaging", arXiv 2508.13304; IEEE Trans. Biomed. Eng., DOI 10.1109/TBME.2025.3643742 [PR].**
  - Technique: analytic gradients, not autodiff. "The gradients of both forward and back-projection can be expressed directly in terms of the forward and back-projection operations themselves." — [arXiv abs](https://arxiv.org/abs/2508.13304)
  - Forward gradient: ∇_M l(d,M) = ∫ G_f(r) δ(p(r)−d) dr, where G_f(r) = (∂x′μ, ∂y′μ, ∂z′μ)·(∇_M x′, ∇_M y′, ∇_M z′). In words, it projects the volume's spatial gradient weighted by the motion Jacobian, which amounts to a standard forward projection. Backprojection uses an analogous detector-domain gradient G_b. — [arXiv HTML](https://arxiv.org/html/2508.13304)
  - Implemented for ray-driven, distance-driven, separable-footprint and voxel-driven projectors. — [arXiv HTML](https://arxiv.org/html/2508.13304)
  - Speed:
    - 2D/3D registration with 500 views took 18.45 min, against 143.95 min for DiffDRR (about 7.8×).
    - The voxel-driven backprojector gradient took 1.21 s/iteration against 27.10 s/iteration for Thies et al. (about 22.4×, 256 views). — [arXiv HTML](https://arxiv.org/html/2508.13304)
  - Memory: 256³ volume, 384² projections, 100 iterations. The proposed ray-driven method used 0.77 GB; DiffDRR used 5.74 GB (86.6% less). — [arXiv HTML](https://arxiv.org/html/2508.13304)
  - Limitations stated by the authors:
    - Rigid motion only.
    - The continuous-domain gradient is then discretised, which gives an "approximation error due to mismatch between continuous formulation and discrete application".
    - Narrow capture range in 2D/3D registration.
    - Calibration is 6-DoF only, not 9-DoF. — [arXiv HTML](https://arxiv.org/html/2508.13304)
  - No explicit "gradient cost = k × forward" ratio is reported. — [arXiv abs](https://arxiv.org/abs/2508.13304)
- **Thies et al. (2024), "A gradient-based approach to fast and accurate head motion compensation in cone-beam CT", arXiv 2401.09283; IEEE Trans. Med. Imaging [PR].**
  - Analytic Jacobian of the cone-beam **backprojection** with respect to the geometry ("generalized derivatives of the backprojection operator"), released as code.
  - 19-fold speed-up over existing methods; reprojection error fell from about 3 mm to 0.61 mm. — [arXiv abs](https://arxiv.org/abs/2401.09283)
  - GPU code: [mareikethies/geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT) — [arXiv PDF](https://arxiv.org/pdf/2401.09283)
- **Thies et al. (2022), "Gradient-Based Geometry Learning for Fan-Beam CT Reconstruction", arXiv 2212.02177 [arXiv; abstract later at BVM 2024, DOI 10.1007/978-3-658-44037-4_58].**
  - Propagates gradients from an image-domain loss into fan-beam geometry parameters.
  - With an autofocus-style learned quality metric: 35.5% lower MSE and 12.6% higher SSIM than the motion-affected reconstruction. — [arXiv abs](https://arxiv.org/abs/2212.02177)
- **Gopalakrishnan & Golland (2022), "Fast Auto-Differentiable Digitally Reconstructed Radiographs for Solving Inverse Problems in Intraoperative Imaging" (DiffDRR), arXiv 2208.12737; MICCAI CLIP workshop [PR, workshop].**
  - Siddon's method rewritten as vectorised PyTorch tensor ops; pose gradients come from PyTorch autodiff.
  - Claims rendering speed "equivalent to" CUDA/C++ DRR generators. — [arXiv abs](https://arxiv.org/abs/2208.12737)
  - Independent measurements (Jiang et al. above) show it is about 7.5× more memory-hungry and about 7.8× slower than analytic pose gradients. — [arXiv HTML 2508.13304](https://arxiv.org/html/2508.13304)
- **Choi, Cho, Kim, Kim (2026), "Revisiting Pose Sensitivity in Splat-based Computed Tomography under Sparse-view Reconstruction", arXiv 2608.04752 [arXiv].**
  - Finds that splat-CT artefacts on real systems "primarily originate from pose inaccuracies in the acquisition geometry rather than from view sparsity itself".
  - Proposes a "stable gradient-based framework that jointly refines geometric parameters during reconstruction". The abstract gives no numbers. — [arXiv abs](https://arxiv.org/abs/2608.04752)

**Volume-gradient libraries (gradient = adjoint projector)**
- **LEAP**, Kim & Champley (2023), "Differentiable Forward Projector for X-ray Computed Tomography", arXiv 2307.05801; ICML 2023 Workshop on Differentiable Almost Everything [PR, workshop].
  - C++/CUDA multi-GPU and multi-core CPU projectors with PyTorch bindings; "minimizing the GPU memory footprint". No timings in the abstract. — [arXiv abs](https://arxiv.org/abs/2307.05801); [LLNL/LEAP GitHub](https://github.com/LLNL/LEAP)
  - The LEAP README cites Champley et al., "Methods for Few-View CT Image Reconstruction", arXiv 2410.07552 (ID from search snippet only). — [LEAP README](https://github.com/LLNL/LEAP/blob/main/README.md)
- **CTorch**, Jiang, Gang, Stayman (2025), "CTorch: PyTorch-Compatible GPU-Accelerated Auto-Differentiable Projector Toolbox for Computed Tomography", arXiv 2503.16741 [arXiv].
  - Voxel-driven, ray-driven, distance-driven and separable-footprint projectors in CUDA C, wrapped as PyTorch modules. The abstract gives no benchmarks. — [arXiv abs](https://arxiv.org/abs/2503.16741)
- **PYRO-NN update**, Schneider, Sun, Ye, Michen, Maier (2025), "An update to PYRO-NN: A Python Library for Differentiable CT Operators", arXiv 2511.08427 [arXiv].
  - Adds PyTorch alongside TensorFlow, with CUDA kernels for parallel, fan and cone geometry and arbitrary trajectories.
  - Ray- and voxel-driven operators computed on the fly, with no stored system matrix.
  - **No runtime benchmarks.** — [arXiv HTML](https://arxiv.org/html/2511.08427)
- **DRACO**, Ye et al. (2024), "DRACO: Differentiable Reconstruction for Arbitrary CBCT Orbits", arXiv 2410.14900 [arXiv].
  - A shift-variant FBP network for arbitrary orbits (built on PYRO-NN-style operators). For sinusoidal orbits: MSE −38.6%, PSNR +7.7%, SSIM +5.0%, and >97% less computation time than iterative methods. — [arXiv abs](https://arxiv.org/abs/2410.14900)
- **TorchRadon**, Ronchetti (2020) [older, arXiv 2009.14788].
  - CUDA library with PyTorch `backward()` support; claims "up to 125" times faster than ASTRA. — [arXiv abs](https://arxiv.org/abs/2009.14788v1)
- **tomosipo**, Hendriksen et al. (2021), Optics Express, DOI 10.1364/OE.439909 [PR, older-ish].
  - Python front end on ASTRA with PyTorch/ODL integration that keeps data on the GPU. — [GitHub](https://github.com/ahendriksen/tomosipo); [ResearchGate](https://www.researchgate.net/publication/355421042_Tomosipo_fast_flexible_and_convenient_3D_tomography_for_complex_scanning_geometries_in_Python)
- **diffct** (sypsyp97, GitHub; no arXiv paper found) [README/docs].
  - Numba-CUDA kernels with torch.autograd for circular orbits, including a separable-footprint projector family.
  - Claims "every projector/backprojector pair is a byte-accurate adjoint, verified by tests". — [GitHub](https://github.com/sypsyp97/diffct)
- **TIGRE v3**, Biguri et al. (2024), arXiv 2412.10129 [arXiv; Cambridge repository copy].
  - More flexible geometry, multi-GPU support for large volumes, 23 iterative algorithms and a **PyTorch wrapper**. — [arXiv abs](https://arxiv.org/abs/2412.10129)
- **ASTRA 2.2–2.5** (release notes) [README/docs].
  - **2.5.0 (2026-06-17)** adds `direct_FP`/`direct_BP` on DLPack arrays "avoiding the overhead of creating intermediate ASTRA data and algorithm objects". It also adds zero-copy DLPack links from PyTorch, TensorFlow, CuPy **and JAX**, plus experimental ROCm/HIP.
  - **2.4 (2025-08-04)** adds curved detectors (`cyl_cone_vec`).
  - **2.2 (2024-07-12)** has faster FDK. — [ASTRA news](https://astra-toolbox.com/_sources/news.rst.txt); [ASTRA docs](https://astra-toolbox.com/)
- **CIL**, Jørgensen et al. (2021), "Core Imaging Library — Part I", arXiv 2102.04560 [older; journal venue (believed Phil. Trans. R. Soc. A 2021) not re-verified]. A framework that wraps ASTRA/TIGRE back ends. — [arXiv PDF](https://arxiv.org/pdf/2102.04560)

**Splat/differentiable-rendering projectors**
- **FaCT-GS**, Pieta, Pedersen, Borgi, Jørgensen, Andreasen, Dahl (2026), "FaCT-GS: Fast and Scalable CT Reconstruction with Gaussian Splatting", arXiv 2604.01844 [arXiv].
  - Fused CUDA rasteriser and voxeliser, plus fused SSIM/TV loss kernels.
  - The biggest win came from replacing the per-pixel backward pass with a **per-Gaussian backward pass**, plus axis-aligned bounding boxes.
  - Timings at 512², 75 views: about 112 s, against about 9 min for R²-Gaussian and about 55 s for FISTA. About 13× faster than R²-Gaussian at 2k. — [arXiv HTML](https://arxiv.org/html/2604.01844)
- **R²-Gaussian** (2024), arXiv 2405.20693 (ID from search snippet/title only). Gaussian kernels with X-ray rasterisation and a CUDA differentiable voxeliser. — [arXiv abs](https://arxiv.org/abs/2405.20693)

### Inferences
- **For TomoJAX's fused pose-normal kernel.** In Jiang et al., the pose gradient is the projection of ∇μ contracted with the pose Jacobian. For a rigid pose with rotation R and translation t, ∂x′/∂t is constant and ∂x′/∂ω is linear in position.
  - The 6-DoF pose gradient for one ray is therefore a residual-weighted sum of 3 "gradient-projections" plus 3 position-weighted moments of ∇μ.
  - All of these can be accumulated **in the same ray pass** as the forward (more registers, no extra memory traffic). That should put "forward + pose gradient" well within the 3× forward budget.
  - Volume gradient (the adjoint) is one extra backprojection, about 1× forward.
  - This is consistent with the 0.77 GB result: no autodiff tape, memory scales with volume plus detector only.
- **Exact discrete gradients.** Jiang et al.'s acknowledged continuous-to-discrete mismatch matters for TomoJAX. TomoJAX's exact trilinear integrator lets it differentiate the *discrete* operator exactly: per-cell two-point Gauss–Legendre is exact for the cubic-in-t integrand, so derivatives of the node positions are analytic. The pose kernel can then be gradient-consistent with the forward, which DD/SF-based analytic gradients are not.
  - Use the existing differentiable JAX reference as the oracle: run finite-difference and JVP checks of the fused kernel against it.
- **Where to spend effort.** DiffDRR-style autodiff through a JAX ray tracer is the pattern to *avoid* in production paths (7.5× memory, 7.8× time in the Jiang benchmark). TomoJAX's JAX reference should stay a test oracle, with the CUDA kernels exposed through `jax.custom_vjp`/`custom_jvp`.
- **ASTRA comparison.** ASTRA 2.5's `direct_FP/BP` on DLPack (including JAX arrays) reduces ASTRA's per-call setup. Re-run cold/warm comparisons against ASTRA ≥2.5, because ASTRA's object-creation overhead is now lower than in older baselines.

### Gaps
- No library reports a measured "pose-gradient cost / forward cost" ratio. Jiang et al. report end-to-end times only.
- Could not verify how LEAP, CTorch or TIGRE-PyTorch handle geometry gradients. Their abstracts and READMEs describe volume gradients only, so they likely do not support pose gradients.
- No Mitsuba/Dr.Jit-based 3D CT projector with pose gradients and benchmarks was found. Haouchat et al. (section 2) use Dr.Jit for a 2D spline projector only.
- Thies et al. (2022), "Calibration by differentiation" (J. Microscopy), was not verified here.

---

## 2. Exact or high-accuracy voxel/spline-basis projectors

### Takeaway
The closest recent analogue to TomoJAX's exact trilinear integrator is Haouchat, Kashani, Thévenaz, Unser (2025, arXiv 2503.20907). They use closed-form X-ray transforms of box-splines and tensor B-splines (degree 1 is the trilinear/bilinear basis) via ray tracing in Dr.Jit, with **matched adjoints** and geometry-independent runtime. Compared with pixels, runtime rises 2.6× for degree 1 and 4.1× for degree 2, and quality improves. The work is 2D only, and 3D is left as future work. The Cutting Voxel Projector (Kulvait et al., J. Comput. Sci. 2025) gives near-exact cone-beam voxel-footprint integration and beats TT footprints at large cone angles.

### Cited Findings
- **Haouchat, Kashani, Thévenaz, Unser (2025), "Generalized Ray Tracing with Basis functions for Tomographic Projections", arXiv 2503.20907 (v2 Sep 2025) [arXiv].**
  - "We propose an exact method to compute the x-ray transform of an image with arbitrary geometry"; images are linear combinations of overlapping shifted basis functions. — [arXiv abs](https://arxiv.org/abs/2503.20907); [arXiv PDF](https://arxiv.org/pdf/2503.20907)
  - Bases: box-splines of degree 0 (pixels), 1 and 2, and separable B-splines. Closed-form X-ray transforms are derived for arbitrary lines. Implemented on the Dr.Jit JIT (Jakob et al., ACM TOG 41(4), 2022). — [arXiv PDF](https://arxiv.org/pdf/2503.20907)
  - "Our projector and back-projector form matched adjoint pairs and allow for any projection geometry without affecting performance." — [arXiv PDF](https://arxiv.org/pdf/2503.20907)
  - Runtime: on average ×2.6 (degree 1) and ×4.1 (degree 2) relative to the pixel basis, GPU forward + back, RTX A5000.
  - ASTRA's performance "drops significantly for arbitrary geometries", while theirs is geometry-independent.
  - Degree-2 box-spline (support of 3 cells) is recommended as the best quality/runtime trade-off; degree >2 saturates. — [arXiv PDF](https://arxiv.org/pdf/2503.20907)
  - **2D only**: "can be naturally extended to 3D settings, which we leave for future work." — [arXiv PDF](https://arxiv.org/pdf/2503.20907)
- **Kulvait, Moosmann, Rose (2021/2025), "Cutting Voxel Projector a New Approach to Construct 3D Cone Beam CT Operator", arXiv 2110.09841; J. Comput. Sci. 87:102573 (2025) [PR].**
  - Analytical voxel-to-detector-pixel volume formulas give a "near-exact projector and backprojector".
  - Beats the TT footprint projector, especially at large cone angles. The relaxed variant matches TT accuracy with a speed advantage and is faster than Siddon at equal accuracy.
  - GPU and open source (KCT_cbct). — [arXiv abs](https://arxiv.org/abs/2110.09841); [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S187775032500050X)
- **Interpolation-then-integrate framing** [older, Turbell PhD thesis, ch. 5].
  - Forward projection is split into continuous interpolation of the voxel data and line integration through the result.
  - Trilinear interpolation plus equidistant sampling gives the standard sampled ray-marcher, which is what TomoJAX's sampled projector is. — [Turbell ch. 5 PDF](https://www.cvl.isy.liu.se/education/undergraduate/tsbb31/download/TurbellPhDCh5.pdf)
- **Cubic-in-t property** [older, volume-rendering literature].
  - Along a ray segment inside a trilinear cell the interpolant is a cubic polynomial in the ray parameter, used for exact ray/isosurface intersection. — [Wald et al. 2004 (VMV), PDF](https://www.sci.utah.edu/~wald/Publications/2004/iso/IsoIsec_VMV2004.pdf)
  - "Second Order Pre-Integrated Volume Rendering" (PacificVis 2008) uses polynomial integration of trilinearly reconstructed volumes. — [PDF](http://icps.u-strasbg.fr/~marchesin/pacificvis08.pdf)
- **Separable-footprint / distance-driven** projectors with matched pairs are now standard in CTorch, PYRO-NN-adjacent tools, diffct and Jiang et al.'s analytic gradients. — [CTorch arXiv 2503.16741](https://arxiv.org/abs/2503.16741); [diffct](https://github.com/sypsyp97/diffct); [arXiv HTML 2508.13304](https://arxiv.org/html/2508.13304)
- **Why matched adjoints matter.** Bentley, Pasha, Sabaté Landman, Yang, Zhang (2026), "Hybrid ABBA-GMRES for Unmatched Backprojectors in Large Scale X-Ray Computerized Tomography", arXiv 2602.17892 [arXiv].
  - Unmatched forward/back pairs "violate adjointness assumptions underlying classical least-squares solvers".
  - Hybrid AB/BA-GMRES with automatic regularisation mitigates the resulting semi-convergence. 2D GPU experiments. — [arXiv abs](https://arxiv.org/abs/2602.17892)

### Inferences
- **Fit with TomoJAX's exact integrator.** Integrating the zero-extended trilinear basis exactly per cell with two-point Gauss–Legendre is the 3D, rigid-pose analogue of Haouchat et al.'s degree-1 tensor-B-spline exact X-ray transform. It is exact because the integrand is cubic in t between voxel-centre planes.
  - Haouchat et al.'s ×2.6 runtime for degree 1 over degree 0, in 2D, is a useful reference for what "exactness" should cost relative to a nearest/box projector.
  - It also suggests TomoJAX could reasonably add a 3D degree-2 box-spline mode later if accuracy at coarse multires levels matters.
- **Publishable gap.** No 3D, arbitrary-per-view-pose, matched exact trilinear projector with analytic pose gradients was found in the 2022–2026 literature. Haouchat et al. is 2D, and Kulvait et al. is a voxel footprint without pose gradients. TomoJAX's integrator appears to fill that gap.
- **Matched adjoints.** Keeping forward/adjoint exactly matched (as TomoJAX does) avoids the ABBA-GMRES class of problems, so CGLS/LSQR-type and pose-gradient methods keep their theory.

### Gaps
- Could not extract the per-N GPU timing table values (ms) from Haouchat et al. Only the averaged 2.6× and 4.1× ratios were recoverable from the PDF text.
- No recent paper found that benchmarks exact-trilinear versus Joseph versus SF accuracy in 3D with modern GPUs.
- Long, Fessler & Balter (separable footprint, 2010) and De Man & Basu (distance-driven, 2004) are foundational [older] but were not re-fetched here.

---

## 3. Fast GPU forward/back-projection: kernel design, laminography, mixed precision

### Takeaway
Recent measured speed gains come from fusion (loss and gradient inside the projection kernel), per-primitive rather than per-pixel backward passes, asynchronous I/O overlap and FP16 storage. FP16 storage halves memory with reported negligible accuracy loss in direct reconstruction. FP16 arithmetic inside iterative inner products risks overflow unless the data are rescaled. For tilted/laminography geometry, the APS group (TomocuPy) uses Fourier-based O(N³ log N) methods on GPU rather than ray-driven backprojection.

### Cited Findings
- **FaCT-GS (arXiv 2604.01844):** the largest speed-up came from reformulating the backward pass per primitive (per Gaussian) instead of per pixel, plus **fused loss kernels** (SSIM, TV), giving 4.5–13× over R²-Gaussian. — [arXiv HTML](https://arxiv.org/html/2604.01844)
- **Nikitin (2022/2023), "TomocuPy: efficient GPU-based tomographic reconstruction with asynchronous data processing", arXiv 2209.08450 [arXiv verified; journal version (believed J. Synchrotron Rad. 2023) not re-verified in this session].**
  - FP16 arithmetic and storage, justified by sub-16-bit detectors.
  - A 2048³ reconstruction, including I/O and initialisation, takes <7 s on one A100. 20–30× faster than TomoPy on CPU.
  - Overlaps disk I/O, H2D/D2H transfers and GPU compute. — [arXiv abs](https://arxiv.org/abs/2209.08450)
  - Another summary reports a 2× drop in total memory use with FP16. — [search result citing arXiv PDF](https://arxiv.org/pdf/2209.08450)
- **Nikitin, Wildenberg, Mittone, Shevchenko, Deriy, De Carlo (2024), "Laminography as a tool for imaging large-size samples with high resolution", arXiv 2401.11101 [arXiv].**
  - A "low computational complexity" laminography reconstruction with multi-GPU processing; a whole mouse brain in 4 slabs, about 12 TB. — [arXiv abs](https://arxiv.org/abs/2401.11101)
  - The search snippet describes a Fourier-based O(N³ log N) GPU method with chunked asynchronous processing. — [arXiv HTML](https://arxiv.org/html/2401.11101v1)
- **Tensor Cores / FP16 in iterative CT:** "Accelerating iterative CT reconstruction algorithms using Tensor Cores", J. Real-Time Image Processing (2021), DOI 10.1007/s11554-020-01069-5 [PR]. About 5× speed-up on an RTX 2080 Ti, with mixed-precision error "nearly equal" to FP32 (from search snippet; full text paywalled). — [Springer](https://link.springer.com/article/10.1007/s11554-020-01069-5)
- **Half-precision hazards:** "Iterative Methods at Lower Precision", arXiv 2210.03844 (ID from search snippet). Half-precision tomography gave NaNs from overflow in inner products (5-bit exponent). Rescaling by 0.01 gave meaningful but less clear reconstructions. — [arXiv PDF](https://arxiv.org/pdf/2210.03844)
- **FP32 vs FP64:** Chillarón, Quintana-Ortí, Vidal, Verdú (2024), arXiv 2412.07631 (QR-based direct CT). FP32 halves time (1.3 vs 2.5 min for 2048 slices) with SSIM 0.99998 against FP64; noise concentrates in air regions. — [arXiv HTML](https://arxiv.org/html/2412.07631v2)
- **Geometry-independent performance:** ASTRA is "highly optimized for parallel- and cone-beam geometries" but slows significantly for arbitrary geometries, whereas a generic ray tracer's cost is geometry-independent (Haouchat et al.). — [arXiv PDF 2503.20907](https://arxiv.org/pdf/2503.20907)
- **Memory-bounded analytic kernels:** Jiang et al. report 0.77 GB versus 5.74 GB for autodiff (256³). — [arXiv HTML 2508.13304](https://arxiv.org/html/2508.13304)

### Inferences
- **Fuse the loss into the projector.** For TomoJAX's 256³ laminography solves, which are projector-bound, fusing residual computation and the data-term loss into the forward kernel removes one full sinogram read/write per iteration. FaCT-GS shows fused-loss gains, and TomoJAX already has `_pallas_loss.py`.
- **BF16/FP16 for volume and sinogram storage, FP32 accumulation along rays**, is the low-risk variant supported by TomocuPy-style evidence. Pure FP16 accumulation in CG/LSQR inner products is risky (the 2210.03844 overflow finding).
  - Validate against the FP32 exact integrator using residual-norm curves.
- **ASTRA's geometry advantage.** Per-view rigid poses are effectively "arbitrary geometry", where ASTRA's specialised paths lose their edge. Benchmarks against ASTRA should include arbitrary-pose (`*_vec`) geometries, not just circular parallel/cone.

### Gaps
- No 2022–2026 paper found specifically on slab/plane-based backprojection kernels for tilted laminography with per-view rigid poses.
- No peer-reviewed measurement found of BF16 accuracy in iterative CT with matched adjoints.
- Could not access full text of the Tensor Core paper to confirm the details.

---

## 4. JAX/XLA compilation and startup overhead

### Takeaway
JAX offers three tiers.
1. The **persistent compilation cache** stores XLA executables keyed by HLO. By default it only stores entries that took ≥1 s to compile, so TomoJAX should lower `jax_persistent_cache_min_compile_time_secs`. It can also persist the GPU autotune and kernel caches. Python tracing and lowering still run on every fresh process.
2. **`jax.export`** serialises StableHLO, which removes tracing and lowering but **still requires XLA compilation**. It supports Pallas/Mosaic-GPU/Triton custom calls subject to a stability allowlist.
3. **`jax.experimental.serialize_executable`** pickles a fully compiled executable, skipping both tracing and compile for an exact shape and device.

Scientific JAX imaging codes (cryoJAX, phaser, mbirjax) report iteration speed-ups but publish no cold-start measurements.

### Cited Findings
- **Persistent cache settings** [docs]:
  - Enabled with `jax_compilation_cache_dir`.
  - Entries are written only if compile time exceeds `jax_persistent_cache_min_compile_time_secs` (default **1.0 s**).
  - `jax_persistent_cache_min_entry_size_bytes` sets a minimum entry size (−1 disables the restriction).
  - `jax_persistent_cache_enable_xla_caches` options include `xla_gpu_kernel_cache_file` and `xla_gpu_per_fusion_autotune_cache_dir` (the default).
  - Primitives using `custom_partitioning` produce new cache keys every run, so they never hit. — [JAX docs](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
- **Cache hits still trace:** "Persistent compilation cache hit but tracing cache miss?" — [jax-ml/jax #22281](https://github.com/jax-ml/jax/issues/22281). A known correctness bug with optimistix + vmap on a cache hit was reported. — [jax-ml/jax #31733](https://github.com/jax-ml/jax/issues/31733)
- **Diagnosing slow tracing and compilation** [docs]:
  - `jax_log_compiles` reports tracing, lowering and compile times separately; `jax_explain_cache_misses` explains retraces; `jax_dump_ir_to` with `eqn_count_pprof` gives a flame graph of jaxpr equation counts.
  - Gotchas: functions recreated in loops or lambdas change `id()` and defeat the trace cache (use `functools.partial` or module-level functions); Python loops unroll into huge jaxprs (use `lax.scan`/`fori_loop`); shape variability recompiles (pad or bucket, or use `jax.export` shape polymorphism).
  - Long `HLO_PASSES` time indicates unrolling. — [JAX docs](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html)
- **`jax.export`** [docs]:
  - Serialises StableHLO plus metadata (flatbuffers). "Deserialized artifacts still require XLA compilation."
  - Supports symbolic dimensions.
  - Pallas, Mosaic GPU and Triton kernels are exportable, but only custom calls on a stability allowlist are permitted by default (`DisabledSafetyCheck.custom_call()` to override).
  - Compatibility: 6 months backward, 3 weeks forward. — [JAX export docs](https://docs.jax.dev/en/latest/export/export.html)
  - Shape polymorphism "can reduce the number of times code needs to be traced and lowered, but does not reduce the number of compilations." — [search summary of JAX docs](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
- **`jax.experimental.serialize_executable`** [docs]:
  - `serialize()` / `deserialize_and_load()` build a `jax.stages.Compiled` from bytes, letting a new process skip tracing *and* compiling for a known shape.
  - Loading an executable can run arbitrary code, so the bytes must come from a trusted source. — [JAX API docs](https://docs.jax.dev/en/latest/jax.experimental.serialize_executable.html)
  - Example use: [uwplasma/LMhdX PR #194](https://github.com/uwplasma/LMhdX/pull/194) ("keep a shape's program across processes beside the compilation cache").
  - Caveats: CPU deserialisation SIGBUS bug — [jax #40944](https://github.com/jax-ml/jax/issues/40944); AOT GPU compile needs a device — [jax #23971](https://github.com/jax-ml/jax/issues/23971).
- **XLA GPU autotuning:** `XLA_FLAGS=--xla_gpu_autotune_level=0` disables compile-time autotune benchmarking. This cuts compile time and memory but may choose slower kernels. `--xla_gpu_experimental_enable_fusion_autotuner=false` turns off the fusion autotuner. — [TensorCircuit write-up (dev.to)](https://dev.to/refractionray/large-scale-tensorcircuit-contractions-on-gpus-disabling-xla-gpu-autotuning-3p2); [NVIDIA JAX-Toolbox GPU perf docs](https://docs.nvidia.com/jax-toolbox/performance-profiling/gpu-performance)
- **Measured cache benefit (non-imaging):** Tunix/MLPerf on TPU v7x reported precompilation falling from 711 s to 109 s, and engine startup from 15.8 to 4.78 min, with a restored persistent cache. — [google/tunix PR #2599](https://github.com/google/tunix/pull/2599)
- **JAX scientific imaging codes:**
  - **cryoJAX**: O'Brien et al., bioRxiv 10.1101/2025.10.23.682564; Acta Cryst. D (2026), DOI 10.1107/S2059798326000550 [PR]. A JAX image-formation modelling library relying on JIT, autodiff and vmap; no cold-start numbers found. — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.23.682564v2); [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1107/S2059798326000550)
  - **phaser**: Gilgenbach, Zhu, LeBeau (2025), arXiv 2505.14372. JAX backend that JIT-compiles the whole inner loop; 6× faster per iteration than fold_slice; no compile-time figures. — [arXiv abs](https://arxiv.org/abs/2505.14372); [arXiv PDF](https://arxiv.org/pdf/2505.14372)
  - **mbirjax** (Bouman & Buzzard; GitHub plus docs, no arXiv paper found) [README/docs]. Vectorised coordinate descent in JAX. — [GitHub](https://github.com/cabouman/mbirjax)
  - **JAX-AMG** (arXiv 2606.09001) wraps AmgX as a native JAX primitive compatible with jit and reverse-mode AD. This is the pattern for external CUDA kernels as primitives. — [arXiv HTML](https://arxiv.org/html/2606.09001v1)
  - **DESC** (stellarator code) has a "Performance Tips" page on compilation caching. — [DESC docs](https://desc-docs.readthedocs.io/en/v0.14.2/performance_tips.html)

### Inferences (concrete actions for TomoJAX's 2–5 s cold compile)
1. **Enable the persistent cache with low thresholds.** Set `jax_persistent_cache_min_compile_time_secs=0` (or 0.1) and `min_entry_size_bytes=-1`. Each TomoJAX kernel probably compiles in under 1 s, so with the default threshold most kernels are *never cached*. Set `jax_persistent_cache_enable_xla_caches="all"` so GPU autotune results and the kernel cache persist too. This is the quickest route to the "<1 s cached startup" target.
2. **Measure before optimising.** Use `JAX_LOG_COMPILES=1` and `jax_explain_cache_misses` to split the 2–5 s into tracing, lowering and backend compile. Pallas→Triton/Mosaic lowering and PTX→SASS (ptxas) may dominate for custom kernels.
3. **Set `--xla_gpu_autotune_level=0` for small-cell interactive runs.** The heavy work is in Pallas kernels, which XLA's GEMM/fusion autotuner does not tune, so the downside should be small. Confirm by A/B testing.
4. **Bucket shapes.** Pad detector and volume shapes to a small set of buckets (for example multiples of 32/64) and pass the per-view pose arrays as traced arguments rather than static ones. Multires levels then reuse a bounded set of executables. This addresses "<5 s compile for new shapes" by turning many shapes into few.
5. **Pre-build an executable bundle.** For shipped defaults, `serialize_executable` executables per (GPU arch, bucket) skip both tracing and compile. The persistent cache plus `jax.export` still pays XLA compile on first load.
6. **Avoid trace-cache defeaters.** Do not create closures or lambdas in solver loops. Express iteration with `lax.fori_loop`/`scan` so unrolling does not inflate HLO_PASSES time.

### Gaps
- No paper or report found that quantifies cold-start compile time for a JAX/Pallas tomography or ptychography code. Phaser, cryoJAX and mbirjax report warm per-iteration speed only.
- Not verified whether Pallas-GPU (Triton or Mosaic GPU) kernels hit the persistent cache reliably across processes, or which Pallas custom-call targets are on the `jax.export` stability allowlist. Check this empirically with `jax_log_compiles`.
- No source found that breaks down Triton/ptxas compile time inside JAX GPU compiles.

---

## 5. Memory-bounded differentiation (implicit differentiation, checkpointing, adjoint-state)

### Takeaway
For gradients *through* a reconstruction (for example pose gradients of a loss on an inner-solved volume), the established options are:
- implicit differentiation of the optimality conditions (JAXopt-style), with memory independent of iteration count;
- deep-equilibrium fixed-point backpropagation;
- reversible or recomputation unrolling (Kellman et al.).

For TomoJAX's alternating pose/volume updates, the analytic-adjoint approach (Jiang et al.) gives bounded memory per projector call. Implicit differentiation covers the outer "volume depends on pose" coupling.

### Cited Findings
- **Blondel, Berthet, Cuturi, Frostig, Hoyer, Llinares-López, Pedregosa, Vert (2021/2022), "Efficient and Modular Implicit Differentiation", arXiv 2105.15183 [arXiv verified; the JAXopt paper, published at NeurIPS 2022 (venue not re-verified in this session)].** The user writes an optimality-condition function; autodiff of that function plus the implicit function theorem gives gradients of the solution, "added on top of any state-of-the-art solver". — [arXiv abs](https://arxiv.org/abs/2105.15183)
- **Gilton, Ongie, Willett (2021), "Deep Equilibrium Architectures for Inverse Problems in Imaging", arXiv 2102.07944 [arXiv verified; journal venue (believed IEEE TCI 2021) not re-verified].** Fixed-point (infinite-depth) unrolled networks incorporating the forward model; the compute budget is selectable at test time. — [arXiv abs](https://arxiv.org/abs/2102.07944)
- **Kellman, Zhang, Tamir, Bostan, Lustig, Waller (2020), "Memory-efficient Learning for Large-scale Computational Imaging", arXiv 2003.05551 (workshop version 1912.05098) [arXiv verified; journal venue (believed IEEE TCI 2020) not re-verified].** Exploits the reversibility of unrolled layers to recompute activations backwards rather than storing them. Demonstrated on CS, multi-channel MRI and super-resolution microscopy. — [arXiv abs](https://arxiv.org/abs/2003.05551); [workshop PDF](https://arxiv.org/pdf/1912.05098)
- **Analytic adjoint projectors versus autodiff:** 0.77 GB (analytic) versus 5.74 GB (DiffDRR autodiff) at 256³ / 384² / 100 iterations. — [arXiv HTML 2508.13304](https://arxiv.org/html/2508.13304)
- **On-the-fly operators without stored system matrices** (PYRO-NN) keep memory at volume plus sinogram scale. — [arXiv HTML 2511.08427](https://arxiv.org/html/2511.08427)

### Inferences
- **Bounded memory for gradients through a reconstruction.** If TomoJAX ever differentiates a pose objective *through* an inner reconstruction (for example a fold/recon-layer objective), use implicit differentiation. At the inner optimum x*(θ) of ½‖A(θ)x − y‖² + R(x), dL/dθ needs one linear solve with the Hessian AᵀA + ∇²R, which is CG using the matched forward and adjoint. Memory is O(volume), independent of inner iterations. Implement it with `jax.custom_vjp` or `optimistix`/`lineax`-style adjoints.
- **Checkpoint at view-batch granularity.** Where unrolling is unavoidable, `jax.checkpoint` (remat) at view-batch granularity, plus matched CUDA adjoints exposed via `custom_vjp`, avoids storing per-ray intermediates.
- **Gradient-cost estimate.** Forward plus volume gradient (one adjoint) plus pose gradient (fused in the forward ray pass) is about 2–2.5× forward with no tape. This would meet the "≤3× forward, bounded memory" target, assuming the fused pose kernel adds less than 0.5× forward through register-level accumulation (an inference, not measured).

### Gaps
- No tomography-specific paper found that applies implicit differentiation to joint pose/volume alignment and reports memory and time against unrolling.
- Measured `jax.checkpoint` overheads for projector-heavy JAX code were not found.
