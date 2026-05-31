"""Output and manifest persistence for the reconstruction CLI."""

from __future__ import annotations

from dataclasses import asdict
import logging
import sys
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from tomojax.cli.config import ConfigValue
from tomojax.cli.manifest import build_manifest, save_manifest
from tomojax.geometry import Detector, Grid
from tomojax.io import save_projection_payload
from tomojax.recon.quicklook import save_quicklook_png

from ._recon_command import ReconCommand

if TYPE_CHECKING:
    from tomojax.io import JsonValue


def write_reconstruction_outputs(
    command: ReconCommand,
    *,
    config_metadata: dict[str, ConfigValue],
    meta: Any,
    geometry_meta: dict[str, object],
    input_grid: Grid,
    recon_grid: Grid,
    detector: Detector,
    detector_center_override: JsonValue,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    roi_mode: str,
    is_parallel: bool,
    resolved_views_per_batch: int,
    views_per_batch_mode: str,
    gather_dtype: str,
    volume_mask: jnp.ndarray | None,
    algorithm_config: dict[str, object],
    volume: jnp.ndarray,
) -> None:
    """Persist reconstruction volume, optional preview, and optional manifest."""
    volume_np = np.asarray(volume)
    save_meta = meta.copy_metadata()
    save_meta.grid = recon_grid.to_dict()
    save_meta.detector = detector
    save_meta.geometry_meta = dict(save_meta.geometry_meta or {})
    save_meta.geometry_meta["detector_center_override"] = detector_center_override
    save_meta.volume = volume_np
    save_meta.frame = str(command.frame)
    save_meta.volume_axes_order = str(command.volume_axes)
    save_projection_payload(
        command.out,
        projections=meta.projections,
        metadata=save_meta,
    )
    logging.info("Saved reconstruction to %s", command.out)
    if command.quicklook is not None:
        _ = save_quicklook_png(command.quicklook, volume_np)
        logging.info("Saved reconstruction quicklook to %s", command.quicklook)
    if command.save_manifest is None:
        return
    manifest = build_reconstruction_manifest(
        command,
        config_metadata=config_metadata,
        meta=meta,
        geometry_meta=geometry_meta,
        input_grid=input_grid,
        recon_grid=recon_grid,
        detector=detector,
        detector_center_override=detector_center_override,
        det_grid=det_grid,
        roi_mode=roi_mode,
        is_parallel=is_parallel,
        resolved_views_per_batch=resolved_views_per_batch,
        views_per_batch_mode=views_per_batch_mode,
        gather_dtype=gather_dtype,
        volume_mask=volume_mask,
        algorithm_config=algorithm_config,
        volume_shape=list(volume_np.shape),
    )
    save_manifest(command.save_manifest, manifest)
    logging.info("Saved reproducibility manifest to %s", command.save_manifest)


def build_reconstruction_manifest(
    command: ReconCommand,
    *,
    config_metadata: dict[str, ConfigValue],
    meta: Any,
    geometry_meta: dict[str, object],
    input_grid: Grid,
    recon_grid: Grid,
    detector: Detector,
    detector_center_override: JsonValue,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    roi_mode: str,
    is_parallel: bool,
    resolved_views_per_batch: int,
    views_per_batch_mode: str,
    gather_dtype: str,
    volume_mask: jnp.ndarray | None,
    algorithm_config: dict[str, object],
    volume_shape: list[int],
) -> dict[str, JsonValue]:
    """Build the stable JSON manifest payload for a reconstruction run."""
    return build_manifest(
        "tomojax recon",
        list(sys.argv),
        asdict(command),
        {
            "input_path": command.data,
            "output_path": command.out,
            "quicklook_path": command.quicklook,
            "manifest_path": command.save_manifest,
            "config_path": config_metadata["config_path"],
            "config_file_values": config_metadata["config_file_values"],
            "explicit_cli_keys": config_metadata["explicit_cli_keys"],
            "effective_options": config_metadata["effective_options"],
            "algorithm": str(command.algo),
            "algorithm_config": algorithm_config,
            "geometry_type": str(meta.geometry_type),
            "input_projection_shape": list(meta.projections.shape),
            "reconstruction_grid": recon_grid.to_dict(),
            "detector": detector.to_dict(),
            "detector_center_override": detector_center_override,
            "detector_roll_deg": geometry_meta.get("detector_roll_deg"),
            "detector_grid_replayed": det_grid is not None,
            "roi": {
                "requested": roi_mode,
                "is_parallel": bool(is_parallel),
                "grid_changed": recon_grid != input_grid,
            },
            "requested_views_per_batch": command.views_per_batch,
            "views_per_batch": int(resolved_views_per_batch),
            "views_per_batch_mode": views_per_batch_mode,
            "requested_gather_dtype": str(command.gather_dtype),
            "gather_dtype": gather_dtype,
            "checkpoint_projector": bool(command.checkpoint_projector),
            "transfer_guard": str(command.transfer_guard),
            "mask_vol": str(command.mask_vol),
            "apply_saved_alignment": bool(command.apply_saved_alignment),
            "mask_applied": volume_mask is not None,
            "volume_shape": volume_shape,
            "volume_axes": str(command.volume_axes),
            "frame": str(command.frame),
        },
    )


__all__ = ["write_reconstruction_outputs"]
