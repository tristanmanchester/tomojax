# JAX Parallel-Beam CT Projector – README

**Goal**  
Provide a fast, memory-efficient, and **fully differentiable** parallel-beam CT projector that can be dropped into any Python workflow (CPU or GPU) to simulate X-ray projections and to compute exact gradients with respect to rigid-body view parameters.  It is intended for alignment, iterative reconstruction, and deep-learning pipelines where data consistency is critical.

**What the projector does**  
- **Forward projection**  
  For each view, it integrates the 3D attenuation map along parallel rays that travel along the **+y** direction in world space.  
  The object can be rotated and translated rigidly before integration:  
  ```
  T x = R_y(β) R_x(α) R_z(φ) x + t
  ```
  where `t = (Δx, 0, Δz)` and the five parameters `(α, β, φ, Δx, Δz)` are **differentiable** via JAX autodiff.

- **Interpolation**  
  Trilinear sampling with exact analytical derivatives through the interpolation weights.

- **Memory & speed**  
  Uses a streaming `lax.scan` over ray steps, so peak memory is **O(detector pixels)** rather than **O(detector × steps)**.  
  Runs on CPU or GPU with JIT compilation.

- **Adjoint**  
  A matched backprojection is available slice-by-slice for FBP or iterative reconstruction.

In short: plug in a 3D volume, get projections and gradients; plug in projections, get a volume.

## Visual Examples

| Phantom | Projections | Sinogram | Reconstruction |
|---------|-------------|----------|----------------|
| <img src="images/phantom_slice.png" width="200"><br><img src="images/phantom_volume.png" width="200"> | <img src="images/projections.gif" width="200"> | <img src="images/sinogram.png" width="200"> | <img src="images/recon_slice.png" width="200"><br><img src="images/recon_volume.png" width="200"> |
| Top: slice<br>Bottom: volume | Animated over 180° | Angle vs detector | Top: slice<br>Bottom: volume |