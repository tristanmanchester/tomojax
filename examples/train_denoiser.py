"""Train a small self-supervised denoiser through the CUDA PyTorch projector.

Example (Diamond I12, Zenodo 3339629, full-resolution 64-row NXtomo cut)::

    python examples/train_denoiser.py full_rows1048-1111.nxs --center -138

Requires a CUDA-enabled PyTorch installation alongside TomoJAX. Two disjoint
interleaved view sets supply input and target reconstructions plus projection targets;
a third set is held out until final evaluation. Image and projection losses
are balanced once using their initial image gradients, without test views.
No clean training image is needed. This is a single-scan demonstration, not a
pretrained denoiser or a claim of unbiased recovery: truncation, rings, geometry error and shared
flat-field errors violate the independent-noise model. Check image quality
against an independent reference, not only the printed projection residual.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Let JAX and PyTorch share the GPU rather than reserve most of it for JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch
from torch import nn

import tomojax as tj
from tomojax.torch import Projector


class Denoiser(nn.Module):
    """A slice-wise residual CNN, initially the identity in attenuation units."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def _slice_batch_size(self, slices: torch.Tensor) -> int:
        """Keep two half-precision hidden feature maps within CUDA's L2.

        Slice groups do not change this network: it has no batch-dependent
        layers. CPU inference and already-small stacks retain one batch.
        """
        batch = slices.shape[0]
        if slices.is_cuda:
            pair_bytes = 4 * self.layers[0].out_channels * slices.shape[-2] * slices.shape[-1]
            cache_bytes = torch.cuda.get_device_properties(slices.device).L2_cache_size
            batch = min(batch, max(1, cache_bytes // pair_bytes))
        return batch

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        slices = volume.permute(2, 0, 1).unsqueeze(1).contiguous(memory_format=torch.channels_last)
        batch = self._slice_batch_size(slices)
        # Only the CNN uses mixed precision. The skip keeps attenuation and
        # the exact projector/transpose in float32, including during inference.
        with torch.autocast("cuda", dtype=torch.float16, enabled=volume.is_cuda):
            residual = (
                self.layers(slices)
                if batch == slices.shape[0]
                else torch.cat([self.layers(part) for part in slices.split(batch)])
            )
        return (slices + residual).squeeze(1).permute(1, 2, 0).contiguous()


def split_views(scan: tj.Scan) -> tuple[tj.Scan, tj.Scan, tj.Scan]:
    """Disjoint angle-interleaved input, target and final-test views."""
    order = np.argsort(np.asarray(scan.geometry.angles), kind="stable")
    return tuple(scan.selected(np.sort(order[k::3])) for k in range(3))


def prepare(path: Path, *, binning: int, every: int, center: float | None) -> tj.Scan:
    frames = tj.load_frames(path).selected(slice(None, None, every))
    scan = frames.corrected(binning=binning)
    if not isinstance(scan.geometry, tj.ParallelGeometry):
        raise ValueError("This slice-wise example requires an axial parallel-beam scan")
    if center is None:
        # Geometry calibration only; denoiser optimization never sees test views.
        scan = tj.estimate_center(scan)
    else:
        detector = replace(scan.detector, center=(center, scan.detector.center[1]))
        scan = replace(scan, geometry=replace(scan.geometry, detector=detector))
    det = scan.detector
    # Include the off-centre detector's complete lateral field, with voxel
    # spacing matched to binned pixels. Binning does not change physical units.
    side = int(np.ceil(det.nu + 2 * abs(det.center[0]) / det.du))
    grid = tj.Grid(
        side,
        side,
        det.nv,
        det.du,
        det.du,
        det.dv,
        vol_center=(0.0, 0.0, det.center[1]),
    )
    return replace(scan, geometry=replace(scan.geometry, grid=grid))


def projection_loss_weight(
    inputs: Sequence[torch.Tensor],
    operators: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    targets: Sequence[torch.Tensor],
    scale: torch.Tensor,
    normalization: torch.Tensor,
) -> torch.Tensor:
    """Balance the initial gradients in normalized image units, using training views only."""
    gradients = []
    for source, target in ((0, 1), (1, 0)):
        estimate = inputs[source].detach().requires_grad_(True)
        losses = (
            (estimate - inputs[target]).square().mean(),
            (operators[target](estimate * scale) - targets[target]).square().mean() / normalization,
        )
        gradients.append(
            torch.stack(
                [torch.autograd.grad(loss, estimate)[0].square().mean().sqrt() for loss in losses]
            )
        )
    image_gradient, projection_gradient = torch.stack(gradients).mean(dim=0)
    # An already stationary loss has no gradient scale to match.
    active = (image_gradient > 0) & (projection_gradient > 0)
    return torch.where(
        active, image_gradient / projection_gradient, torch.ones_like(image_gradient)
    )


def train(
    scan: tj.Scan, *, steps: int, learning_rate: float
) -> tuple[torch.Tensor, torch.Tensor, Denoiser, dict[str, float]]:
    """Fit balanced independent-half image/projection losses, then test once."""
    first, second, test = split_views(scan)
    inputs = [
        torch.from_dlpack(tj.reconstruct(part, "fbp").volume).detach() for part in (first, second)
    ]
    scale = torch.stack([volume.square().mean().sqrt() for volume in inputs]).mean()
    inputs = [volume / scale for volume in inputs]
    operators = [Projector(part.geometry) for part in (first, second)]
    targets = [
        torch.as_tensor(np.asarray(part.projections), device="cuda") for part in (first, second)
    ]
    normalization = torch.stack([target.square().mean() for target in targets]).mean()
    projection_weight = projection_loss_weight(inputs, operators, targets, scale, normalization)
    model = Denoiser().to(device="cuda", memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(steps):
        # Each input is reconstructed solely from the OTHER target's views.
        source, target = step % 2, 1 - step % 2
        optimizer.zero_grad(set_to_none=True)
        estimate = model(inputs[source]) * scale
        # Noise2Inverse image supervision emphasizes high-frequency noise that
        # a projection-only loss barely sees. Fix their image-gradient balance
        # at the identity: relative MSEs alone do not balance their effects.
        image_loss = (estimate / scale - inputs[target]).square().mean()
        projection_loss = (
            operators[target](estimate) - targets[target]
        ).square().mean() / normalization
        loss = image_loss + projection_weight * projection_loss
        # Scale small CNN gradients; the projector still applies its exact transpose.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if (step + 1) % 25 == 0:
            print(f"step {step + 1}: training relative MSE {float(loss.detach()):.6f}", flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        baseline = (inputs[0] + inputs[1]) * (scale / 2)
        # Average predictions in the same input distribution used for fitting.
        denoised = (model(inputs[0]) + model(inputs[1])) * (scale / 2)
        held_out = Projector(test.geometry)
        measured = torch.as_tensor(np.asarray(test.projections), device="cuda")
        norm = torch.linalg.vector_norm(measured)
        metrics = {
            "train_seconds": train_seconds,
            "projection_loss_weight": float(projection_weight),
            "held_out_baseline_nrmse": float(
                torch.linalg.vector_norm(held_out(baseline) - measured) / norm
            ),
            "held_out_denoised_nrmse": float(
                torch.linalg.vector_norm(held_out(denoised) - measured) / norm
            ),
        }
    return baseline, denoised, model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--center", type=float, help="calibrated detector-u centre, in input length units"
    )
    parser.add_argument("--binning", type=int, default=8)
    parser.add_argument("--every", type=int, default=1, help="use every Nth view")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--output", type=Path, help="optional NPZ of baseline/denoised volumes and grid"
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    torch.manual_seed(0)
    started = time.perf_counter()
    scan = prepare(args.input, binning=args.binning, every=args.every, center=args.center)
    baseline, denoised, _, metrics = train(scan, steps=args.steps, learning_rate=args.learning_rate)
    metrics["load_to_result_seconds"] = time.perf_counter() - started
    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        np.savez_compressed(
            args.output,
            baseline=baseline.cpu().numpy(),
            denoised=denoised.cpu().numpy(),
            grid=json.dumps(scan.grid.to_dict()),
            metrics=json.dumps(metrics),
        )


if __name__ == "__main__":
    main()
