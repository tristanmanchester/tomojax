# TomoJAX: Differentiable CT Projector with Alignment

TomoJAX v2 is the primary package at import path `tomojax`. CLIs run via `python -m tomojax.cli.<cmd>` and a full, copy‑paste tutorial lives in `docs/tutorial_end_to_end.md`.

**TomoJAX** is a fully differentiable, memory-efficient parallel-beam CT projector implemented in JAX. It provides exact gradients for **5-DOF rigid-body alignment optimization**, making it ideal for CT reconstruction, view alignment, and deep learning applications requiring data consistency.

#### Alignment demonstration
<img src="images/montage_scroll.gif" width="1000">

**Left to Right**: Ground truth phantom → Naive reconstructions (misaligned & added noise) → Aligned reconstructions


## Key Features

### **Core Projector**
- **Differentiable forward projection** with exact gradients via JAX autodiff
- **5-DOF rigid-body parameterization**: `T x = R_y(β) R_x(α) R_z(φ) x + t` where `t = (Δx, 0, Δz)`
- **Memory efficient**: Streaming integration via `lax.scan` — O(detector pixels) memory usage
- **Trilinear interpolation** with analytical derivatives
- **CPU/GPU support** with JIT compilation

### **Joint Reconstruction & Alignment**
- **Multi-resolution approach**: Hierarchical optimization (4x → 2x → 1x binning)
- **Alternating optimization**: FISTA-TV reconstruction + per-view alignment
- **Multiple optimisers**: AdaBelief (recommended), Adam, L-BFGS, hybrid approaches
- **4-DOF or 5-DOF**: Option to optimise projection angle φ or keep fixed

### **Alignment Capabilities**
- **Per-view misalignment correction** for all 5 rigid-body parameters:
  - **α** (pitch): rotation around x-axis  
  - **β** (roll): rotation around y-axis
  - **φ** (yaw): rotation around z-axis (projection angle)
  - **Δx**: horizontal translation
  - **Δz**: vertical translation
- **Automatic parameter scaling** for different parameter sensitivities
- **Alignment accuracy metrics** (RMSE in degrees/pixels)

### **Reconstruction**
- **FISTA-TV regularization** with Chambolle-Pock proximal operator
- **Automatic Lipschitz estimation** for optimal step sizes
- **Alignment-aware reconstruction** with exact gradients
- **FBP initialization** and iterative refinement

## Quick Start

```bash
# Inside pixi environment
pixi run install-root

# Simulate, misalign/noise, reconstruct, and align
python -m tomojax.cli.simulate --help
python -m tomojax.cli.misalign --help
python -m tomojax.cli.recon --help
python -m tomojax.cli.align --help

# Or use pixi tasks
pixi run simulate
pixi run misalign
pixi run recon
pixi run align

# Full tutorial
less docs/tutorial_end_to_end.md
```

## Alignment Algorithm

The implementation uses multi-resolution alternating optimisation:

1. **Multi-resolution pyramid**: Start at coarse resolution for global alignment
2. **Alternating steps**: 
   - Fix alignment → FISTA-TV reconstruction
   - Fix reconstruction → optimise alignment parameters
3. **Parameter transfer**: Scale alignment between resolution levels
4. **Convergence**: Monitor objective function and parameter changes


## Notes

- Legacy experimental scripts were removed. Use the CLIs above or the Python APIs under `tomojax.*`.

## Visual Examples

### Basic Projector Workflow
| Phantom | Projections | Sinogram | Reconstruction |
|---------|-------------|----------|----------------|
| <img src="images/phantom_slice.png" width="200"><br><img src="images/phantom_volume.png" width="200"> | <img src="images/projections.gif" width="200"> | <img src="images/sinogram.png" width="200"> | <img src="images/recon_slice.png" width="200"><br><img src="images/recon_volume.png" width="200"> |
| Top: slice<br>Bottom: volume projection | Animated over 360° | Angle vs detector | Top: slice<br>Bottom: volume projection |

### Alignment & Reconstruction Workflow

#### Misaligned Input Data
| Clean Misaligned Projections | Noisy Misaligned Projections |
|------------------------------|------------------------------|
| <img src="images/spin_projections_misaligned.gif" width="300"> | <img src="images/spin_projections_noisy.gif" width="300"> |
| Random rigid-body misalignments | Same misalignments + Poisson noise |

#### Sinogram Analysis
| Clean Misaligned Sinogram | Noisy Misaligned Sinogram |
|---------------------------|---------------------------|
| <img src="images/misaligned_sinogram.png" width="300"> | <img src="images/noisy_sinogram.png" width="300"> |
| Clear view-to-view inconsistencies | Inconsistencies + noise artifacts |

#### Multi-Resolution Alignment Process
| Clean Data Alignment | Noisy Data Alignment |
|---------------------|---------------------|
| <img src="images/alignment_process_misaligned.gif" width="300"> | <img src="images/alignment_process_noisy.gif" width="300"> |
| 4x → 2x → 1x resolution refinement | Robust alignment despite noise |
