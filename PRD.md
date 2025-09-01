Here’s a first-principles design and a minimal, fast, memory-efficient, physically correct parallel-beam projector in JAX that is fully differentiable with respect to the five rigid-body view parameters (α, β, φ, Δx, Δz). It streams ray integration via a scanned loop (no n_rays × n_steps tensor), uses tri-linear interpolation, includes the step-size factor, and provides value+gradient of a per-view loss.

Code: projector_parallel_jax.py


Product Requirements Document (PRD): Differentiable Parallel-Beam CT Projector (JAX)

1) Goal
- Provide a lightweight, physically accurate, memory-efficient, and differentiable parallel-beam CT projector to serve as the core engine for joint alignment-and-reconstruction on Mac CPU and larger GPU systems.
- Must compute projections and exact gradients of a data loss with respect to five view-wise rigid parameters Θ = (α, β, φ, Δx, Δz). Emphasis on correctness, speed, and low memory for 256^3 tests on CPU.

2) Scope
- Parallel-beam X-ray geometry (beam direction +y).
- Rigid-body per-view misalignment modeled by T x = R_y(β) R_x(α) R_z(φ) x + t, t = (Δx, 0, Δz).
- Forward operator A(Θ) integrating along +y with tri-linear interpolation and Δs scaling.
- Differentiation w.r.t. Θ via JAX autodiff.
- JIT compilation for speed; checkpointing for memory efficiency.
- Batch processing across views in a low-memory sequential loop (optionally vmap on GPU).
- Provide a minimal adjoint sanity test via VJP (small sizes).

Out of scope initially
- Cone/fan-beam and curved detectors (can be added).
- Non-rigid motion.
- TV-regularized recon in this module (the module focuses on A and ∂/∂Θ).

3) Functional requirements
- Inputs:
  - Volume: flattened jnp.ndarray (nx*ny*nz,) float32; grid: nx,ny,nz and voxel sizes vx,vy,vz; volume origin or center.
  - Detector: nu,nv and pixel sizes du,dv; detector center (x,z).
  - View parameters: Θ = (α, β, φ, Δx, Δz) per view (radians, same units as grid).
- Outputs:
  - Projection image (nv, nu) float32.
  - Loss value and ∂loss/∂Θ for each view.
- API:
  - forward_project_view(...)
  - view_loss_value_and_grad(...)
  - batch_forward_project(...)
- Numerical properties:
  - Tri-linear sampling; piecewise-differentiable everywhere except on voxel boundaries (measure-zero).
  - Includes Δs scaling of the integral.
- Memory:
  - No materialization of points_on_ray; integration uses lax.scan over steps and remat/checkpoint to bound backward memory.
- Performance targets (guidance on Mac CPU):
  - 256^3 volume, 256×256 detector, ~256 steps: seconds per view (exact number depends on CPU).
  - Low peak memory (~tens to hundreds of MB for single view).
  - On large GPU systems, vmap across views achieves multi-view throughput.

4) Architecture and design decisions
- Geometry model: Transform the object (sample at q = R^T (w − t)) and integrate along +y in world space.
  - This avoids per-ray direction re-computation and keeps r_hat constant (parallel beam).
  - It aligns with Eq. (1) and ensures physical fidelity.
- Streaming integration:
  - Use jax.lax.scan to iterate over y, accumulating contributions without storing per-step arrays (O(n_rays) memory).
  - Include step_size in the integral.
  - Optionally mask steps; by default, integrate across the volume y-extent.
- Differentiation:
  - Use JAX reverse-mode AD over the scanned computation for exact derivatives of the discretized operator.
  - stop_gradient(recon) to compute only ∂/∂Θ for alignment.
  - jax.checkpoint the step function to reduce backward memory (trade compute for memory).
- Validation:
  - Inner-product (adjoint) test on small problems via VJP w.r.t. recon.
  - Finite-difference checks of ∂loss/∂Θ on a single view (scripts TBD).
  - Visual checks: forward projections of simple phantoms; consistency under known transforms.

5) Performance/memory plan
- CPU (Mac):
  - Use single-view jit; sequential loop over views to cap memory.
  - Default n_steps = ceil((ny * vy) / step_size); default step_size = vy.
  - Checkpointed scan to bound reverse-mode tape.
- GPU (50 GB VRAM):
  - Optionally vmap across views (or sub-batches) for high throughput.
  - Keep code identical; JAX picks GPU kernels.

6) Risks and mitigations
- Backward memory through scan: mitigated by jax.checkpoint/remat; also consider using value_and_grad only w.r.t. Θ and stop_gradient(recon).
- Numerical kinks at voxel boundaries: expected with tri-linear interpolation; tolerance handled via robust optimization (outside this module).
- Scale mismatches: step_size must be included; origin and detector center must be consistent; provide defaults and documentation.

7) Roadmap
- v0.1 (this deliverable):
  - Parallel-beam A(Θ) with scan + checkpoint, ∂loss/∂Θ, batch helpers, adjoint sanity test.
- v0.2:
  - Exact matched adjoint A^T (streaming scatter-add or segmented-sum).
  - Finite-difference test harness; simple iterative recon example using A/A^T.
- v0.3:
  - Optional custom_vjp for Θ to reduce backward memory further (recompute strategy).
  - Optional per-ray masks and variable path lengths.
- v0.4+:
  - Distance-driven/SF kernels for higher accuracy.
  - Fan/cone-beam extensions.

8) Alternatives considered (ranked)
- Custom CUDA + PyTorch autograd (e.g., CTorch-style): fastest on GPU; higher dev overhead; not ideal for Mac CPU-only use.
- JAX with custom_vjp for Θ: best memory control with exact J^T r; more code; consider for v0.3.
- Plain NumPy/Numba: simpler but no AD; would require manual Jacobians.

How to use (example outline)
- Define grid = {nx,ny,nz,vx,vy,vz, vol_center or vol_origin}
- Define det = {nu,nv,du,dv, det_center}
- recon_flat = jnp.asarray(volume).ravel()
- params = jnp.array([alpha, beta, phi, dx, dz], dtype=jnp.float32)
- proj = forward_project_view(params, recon_flat, grid..., step_size=vy, n_steps=ny)
- loss, grad = view_loss_value_and_grad(params, measured, recon_flat, grid, det, step_size=vy, n_steps=ny)

If you want, I can also provide a tiny driver script that builds a cube phantom, projects a few views, runs the adjoint test (small volume), and prints gradient finite-difference checks for α and φ.