# Runnable examples

Run these from the checkout root after [installation](../docs/installation.md).
They use public APIs and write no data unless an output is requested.

## Project and reconstruct a volume

```bash
uv run --no-sync python examples/simulate_and_reconstruct.py
```

[simulate_and_reconstruct.py](simulate_and_reconstruct.py) creates a 32³
Shepp–Logan object, generates 60 parallel projections with the Joseph model,
and runs 24 CGLS iterations with that same model. It prints the reference and
reconstruction shapes and the full-volume relative L2 error.
The iteration budget may end before convergence. The JAX backend runs on CPU or
an available GPU; prefix the command with `JAX_PLATFORMS=cpu` to require CPU.

Simulation and reconstruction share a discretization, so this is a usage example,
not independent accuracy evidence. See the [measurements](../docs/measurements.md)
for independent data and accepted-result comparisons.

## Reproduce the README figures

The README's figures image one phantom: a 29 mm skeletonised mechanical watch
movement with a mainspring barrel, a going train, a club-tooth escapement and a
screwed balance with its hairspring, modelled in Blender as one closed mesh per
material ([watch_blender.py](../bench/phantoms/watch_blender.py)). gVXR traces
each ray's path length through the meshes, so the projections are exact line
integrals of the meshes and share nothing with TomoJAX's projector; each
detector pixel averages the transmission of 2 × 2 rays before Poisson noise and
the logarithm. The beam is monochromatic at 120 keV, with xraylib attenuation
for titanium, brass, steel, silicon, ruby, Glucydur, 18-carat gold, Nivarox and
Nivaflex. The truth on each reconstruction grid is the exact voxel average of
the meshes ([mesh_voxels.py](../bench/phantoms/mesh_voxels.py)).

The scripts need a CUDA GPU, Blender 4.5 and the gVXR environment of the
[watch phantom](../bench/README.md#watch-movement-phantom), found under
`.artifacts` or given with `--blender` and `--gvxr-python`:

```bash
uv sync --locked --extra cuda12 --group examples --no-dev
uv run --no-sync python examples/reconstruct_watch.py
uv run --no-sync python examples/align_watch.py
uv run --no-sync python examples/render_watch.py
```

Each writes its figure to `images/` (or `--out`) with a JSON record of its
settings, errors and run times beside it. Meshes, projections and truths are
cached under `.artifacts/figures/watch`; the first run builds the meshes in
about 40 s. Times are on an RTX 4070 Laptop GPU.

### Reconstruct a few-view, noisy scan

[reconstruct_watch.py](reconstruct_watch.py) scans the movement with 200 views
over a half turn and 10⁴ photons per pixel, and reconstructs a 32-slice slab
through the hairspring on 38.5 µm voxels with filtered backprojection (Hann
filter) and with 200 iterations of non-negative FISTA-TV (weight 0.01) started
from it. Over the whole slab the relative L2 error is 0.357 for FBP and
0.125 for FISTA-TV, which takes 15 s. The script takes about a minute.

### Align a scan with per-view motion

[align_watch.py](align_watch.py) scans the whole movement on 77 µm voxels with
400 views over a half turn and 2 × 10⁴ photons per pixel. The rotation axis
projects 5.3 px off the detector centre, and every view has its own pose: a
slow wobble plus jitter of up to 0.25° in each rotation, 2.4 px across the axis
and 1.5 px along it. `tj.align(scan, mode="cor-then-pose")` takes 300 s. It
recovers the rotations to 0.06° RMS or better and the shifts to 0.07 px
(across) and 0.04 px (along) RMS, and the error of a FISTA-TV reconstruction
falls from 0.928 with the nominal geometry to 0.235. The script takes about
6 minutes.

Moving the whole movement rigidly, with every pose moved back, predicts the
same data, as does trading the detector centre against a shift of every view,
so `tj.align` reports the estimate with the least per-view motion. The truth,
its poses and its centre are moved to that estimate before they are compared
(see [where the aligned volume sits](../docs/alignment-guide.md#where-the-aligned-volume-sits));
there the centre is found at 3.26 px against 3.31 px.

### Volume-render a reconstruction

[render_watch.py](render_watch.py) scans the whole movement with 600 views and
3 × 10⁴ photons per pixel, reconstructs it on a 768 × 768 × 164 grid of 38.5 µm
voxels with 60 FISTA-TV iterations (55 s), and draws the reconstruction
by direct volume rendering in JAX ([_volume_render.py](_volume_render.py)).
Rays march through the volume, and a transfer function of the reconstructed
attenuation sets colour and opacity: steel white, brass and the balance gold,
the timing screws yellow, the ruby jewels red, and titanium blue glass drawn by
its boundaries. The picture is drawn from the reconstruction alone; the meshes
only make the projections. The script takes about 3 minutes.

## Prepare and align complex ptychography projections

```bash
uv sync --locked --extra cuda12 --extra fourier-cuda12 --no-dev
uv run --no-sync python examples/align_ptychography.py /path/to/porous_glass_dataset.mat
```

[align_ptychography.py](align_ptychography.py) reads either PSI MATLAB example:
`porous_glass_dataset.mat` (v7.3, measured and already aligned) or
`example_data.mat` (v5, simulated with motion). The measured-data route crops
unreliable borders; the smaller simulation uses its full detector. Both prepare
complex transmission in bounded device batches. For a CPU-sized preview of the
simulation, keep all angles but bin the **unwrapped phase**, not complex pixels:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python examples/align_ptychography.py \
  /path/to/example_data.mat --binning 4 --iterations 5 -o phase-slices.npz
```

The five-iteration run is a preview, not a convergence or accuracy guarantee.
The same workflow in Python, from the checkout root, is:

```python
from examples.align_ptychography import run_example

volume, poses, report = run_example("example_data.mat", binning=4, iterations=5)
print(volume.shape, report["value_unit"])
```

Its core steps are public APIs:

```python
import tomojax as tj
from tomojax.alignment import ProjectionMatchingConfig
from tomojax.corrections import remove_phase_ramp, unwrap_phase

phase = -remove_phase_ramp(unwrap_phase(objects), air_mask=air * abs(objects)**2)
scan = tj.Scan(phase, geometry)  # (view, v, u), geometry in pixel units
aligned = tj.align(scan, mode="pose", config=ProjectionMatchingConfig(iterations=20))
reconstruction = tj.reconstruct(aligned.scan, "fourier")
```

The air mask must cover actual air, not sample pixels. The example's measured-data
crop and air strips describe these supplied scans; they are not per-view alignment
corrections and must be reconsidered for other samples.
The joint estimator returns per-view translations without resampling the input.
It can fit noise or model mismatch as motion, including on an already aligned
scan; inspect pose estimates and physical slices rather than treating improved
consistency or convergence as proof of physical accuracy.

No supplied aligned phase or reference image is used in inference. For the
measured file, wavelength and pixel size convert reconstructed phase to
refractive-index decrement, `wavelength / (2*pi*pixel_size)`. The simulation has
no physical calibration: its output is phase per original pixel (`rad/pixel`),
not refractive index. Binning keeps geometry and detector-frame translations in
original-pixel units; one output voxel spans `binning` original pixels.

The example reports read, preparation, alignment, inverse and whole-workflow
times. It writes nothing unless `-o slices.npz` is supplied; that archive holds
central `delta_xy/xz/yz` or `phase_per_pixel_xy/xz/yz` slices, poses and unit
metadata. `pixel_size_m` is NaN when the input lacks physical calibration.
Open all three planes and check `value_unit` before interpreting values.

To keep the simulated example as a TomoJAX dataset and use the standard CLI,
prepare phase in Python (do not apply an absorption logarithm):

```python
import numpy as np
import tomojax as tj
from tomojax.alignment import ProjectionMatchingConfig
from examples.align_ptychography import read_psi_objects, prepare_phase, make_scan

objects, angles, _, _ = read_psi_objects("example_data.mat")
air = np.zeros(objects.shape[-2:], dtype=np.float32)
air[:, :50] = air[:, -50:] = 1  # known air in this supplied example
scan = make_scan(prepare_phase(objects, air), angles, binning=4)
aligned = tj.align(scan, mode="pose", config=ProjectionMatchingConfig(iterations=5))
tj.save("aligned-phase.nxs", aligned.scan)
```

```bash
uv run --no-sync tomojax recon aligned-phase.nxs -o phase-recon.nxs --method fourier --roi off
uv run --no-sync tomojax inspect phase-recon.nxs --preview phase-previews
```

These datasets retain original-pixel geometry: reconstructed values remain
`rad/pixel`. The joint translation configuration is a Python API; `tomojax align`
uses the general rigid-pose configuration, not this specialized estimator.


## Preview a real cone-beam orbit on a CPU

[import_walnut.py](import_walnut.py) prepares a small single-orbit FIPS walnut
scan for FDK or an iterative reconstruction, without ASTRA or a GPU. It uses
shared TIFF/geometry loading with the multi-orbit example below. Follow the
[complete Python and CLI commands](../docs/lab-ct.md#try-one-fips-walnut-orbit-on-a-cpu).

## Bring a real multi-orbit scan into register

This one needs data: download a walnut from the FIPS collection (Der
Sarkissian et al., Scientific Data 6, 215, 2019; CC BY 4.0;
[Zenodo record 2686726](https://zenodo.org/records/2686726)) and unzip it, then

```bash
uv run --no-sync python examples/align_walnut_orbits.py ~/data/walnuts/Walnut1 -o walnut.nxs
```

[align_walnut_orbits.py](align_walnut_orbits.py) reads the three orbits'
projections and the scanner's uncorrected geometry with `tj.Scan.from_astra`,
bins the detector 2 x 2, aligns every 4th view (900 in all) with `tj.align`, prints each
orbit's height relative to orbit 1 and reconstructs with 20 iterations of
non-negative least squares. On Walnut 1 it finds -0.38 and -0.76 mm for orbits
2 and 3, where the authors' corrected geometry has -0.40 and -0.79 mm, in about
six minutes on an 8 GB laptop GPU. The data fix the orbits' heights relative to
one another but can't pin down the walnut's absolute height: moving the whole
object, and every view with it, predicts the same projections, so the volume
can sit a fraction of a millimetre from where the scanner's record would put
it. [The measurements](../docs/measurements.md#orbit-registration) compare the
reconstruction with the authors' reference.

## Reconstruct raw laminography from nominal geometry

[align_laminography.py](align_laminography.py) combines public setup/motion
initialization, data-selected region, sparse pose fitting, untouched-view checks
and positive reconstruction. It is a conservative example, not a new alignment
mode or default. See the [automatic laminography section](../docs/real-laminography.md#automatic-nexus-laminography)
for the command and limitations. Its HDF5 dataset paths and nominal tilt are
explicit arguments; it writes nothing unless `-o` is supplied.

## Gallery figures

The scripts in [gallery](gallery) make the figures on the
[gallery page](../docs/gallery.md): laminography alignment of a gVXR-rendered
chip package, ring removal on a real Diamond I12 scan, few-view sandstone
micro-CT, an offset-detector battery scan and wide-cone FDK against CGLS on a
lattice part. Each takes an output path; the phantoms are built on first use
and cached under `.artifacts/figures`.

## Train a denoiser through the PyTorch projector

Install a CUDA-enabled PyTorch separately, then run
[train_denoiser.py](train_denoiser.py) on a Diamond I12 NXtomo row cut (raw
counts with flat and dark frames, Zenodo 3339629):

```bash
python examples/train_denoiser.py full_rows1048-1111.nxs --center -138
```

The small residual CNN learns from two disjoint, angle-interleaved view sets:
one supplies an FBP input, the other an independent image and projection target.
The loss sums image and projection mean-square errors in dimensionless units.
The CUDA PyTorch projector backpropagates with its exact transpose. A third view set is used
only for final projection-error reporting, never for training or input FBP.
The default bins 8×8, uses all views and takes 500 training steps.
Do not undersample each split: missing angular detail can teach the CNN to
blur structures even while its held-out projection error improves. Binning preserves physical
units, so the calibrated detector centre stays −138, not −138/8. Omit
`--center` to estimate it from the scan before splitting.

No clean image is needed for training. This is a single-scan demonstration;
projection residual is not image accuracy, and shared calibration errors,
rings and truncation can bias the result. Validate against an independent
reconstruction before interpreting learned features. Nothing is written
unless `--output result.npz` is given.
