from __future__ import annotations

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.align.objectives.loss_adapters import build_loss_adapter
from tomojax.align.objectives.loss_specs import parse_loss_spec
from tomojax.core.geometry import Detector, Grid, ParallelGeometry


def test_alignment_public_facade_stays_narrow() -> None:
    align_api = importlib.import_module("tomojax.align")

    assert set(align_api.__all__) == {"AlignConfig", "align", "align_multires"}
    assert align_api.AlignConfig is importlib.import_module("tomojax.align.pipeline").AlignConfig
    assert callable(align_api.align)
    assert callable(align_api.align_multires)


@pytest.mark.parametrize(
    "symbol",
    [
        "AlignConfig",
        "AlignResumeState",
        "AlignMultiresResumeState",
        "AlignCheckpointCallback",
        "AlignMultiresCheckpointCallback",
        "align",
        "align_multires",
        "adapt_legacy_observer",
        "ObserverAction",
        "ObserverCallback",
        "OuterStat",
        "OuterStatValue",
        "MultiresLevel",
    ],
)
def test_pipeline_compatibility_symbols_remain_importable(symbol: str) -> None:
    pipeline = importlib.import_module("tomojax.align.pipeline")

    assert hasattr(pipeline, symbol)


@pytest.mark.parametrize(
    "symbol",
    [
        "AlignmentLossConfig",
        "LossState",
        "LossAdapter",
        "L2OtsuLossSpec",
        "parse_loss_spec",
        "parse_loss_schedule",
        "resolve_loss_for_level",
        "validate_loss_schedule_levels",
        "loss_spec_name",
        "loss_is_within_relative_tolerance",
        "build_loss",
        "build_loss_adapter",
    ],
)
def test_loss_compatibility_symbols_remain_importable(symbol: str) -> None:
    losses = importlib.import_module("tomojax.align.losses")

    assert hasattr(losses, symbol)


@pytest.mark.parametrize(
    "module_name",
    [
        "tomojax.align.checkpoint",
        "tomojax.align.diagnostics",
        "tomojax.align.motion_models",
        "tomojax.align.params_export",
    ],
)
def test_bounded_alignment_legacy_module_aliases_remain_importable(module_name: str) -> None:
    importlib.import_module("tomojax.align")

    assert importlib.import_module(module_name)


@pytest.mark.parametrize(
    "module_name",
    [
        "tomojax.align.detector_center",
        "tomojax.align.dof_specs",
        "tomojax.align.dofs",
        "tomojax.align.gauge",
        "tomojax.align.geometry_applier",
        "tomojax.align.geometry_blocks",
        "tomojax.align.initializers",
        "tomojax.align.parametrizations",
        "tomojax.align.schedules",
        "tomojax.align.state",
    ],
)
def test_undocumented_alignment_legacy_module_aliases_are_not_registered(module_name: str) -> None:
    importlib.import_module("tomojax.align")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize(
    "module_name",
    [
        "tomojax.align._loss_adapters",
        "tomojax.align._loss_kernels",
        "tomojax.align._loss_specs",
        "tomojax.align._loss_state",
    ],
)
def test_private_loss_shims_are_not_compatibility_surface(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_loss_adapter_exposes_distinct_setup_validation_lm_capability() -> None:
    targets = jnp.zeros((2, 3, 3), dtype=jnp.float32)

    l2_adapter = build_loss_adapter(parse_loss_spec("l2"), targets)
    phasecorr_adapter = build_loss_adapter(parse_loss_spec("phasecorr"), targets)

    assert l2_adapter.supports_gauss_newton
    assert l2_adapter.supports_setup_validation_lm
    assert not phasecorr_adapter.supports_gauss_newton
    assert not phasecorr_adapter.supports_setup_validation_lm


def test_alignment_stat_float_conversion_records_errors() -> None:
    results = importlib.import_module("tomojax.align._results")
    stat: dict[str, object] = {}

    results._set_float_stat(stat, "rot_mean", object())

    assert "rot_mean" not in stat
    assert stat["stat_errors"][0]["key"] == "rot_mean"
    assert "TypeError:" in stat["stat_errors"][0]["error"]


def test_multires_elapsed_stat_conversion_records_errors() -> None:
    results = importlib.import_module("tomojax.align._results")

    stat = results.enrich_multires_stage_stat(
        {"outer_idx": 1, "cumulative_time": object()},
        level_factor=1,
        level_index=0,
        global_outer_idx=1,
        elapsed_offset=10.0,
        loss_name="l2",
        schedule_name="default",
        stage=None,
    )

    assert stat["level_elapsed_seconds"] is None
    assert stat["global_elapsed_seconds"] is None
    assert stat["level_stats_errors"][0]["key"] == "cumulative_time"
    assert "TypeError:" in stat["level_stats_errors"][0]["error"]


def test_multires_wall_time_conversion_records_errors() -> None:
    stage_loop = importlib.import_module("tomojax.align._stage_loop")
    info: dict[str, object] = {"wall_time_total": object()}

    level_wall_time = stage_loop._accumulate_stage_wall_time(2.5, info)

    assert level_wall_time == 2.5
    assert info["level_stats_errors"][0]["key"] == "wall_time_total"
    assert "TypeError:" in info["level_stats_errors"][0]["error"]


def test_setup_stage_rejects_unsupported_loss_before_fold_reconstruction(monkeypatch) -> None:
    pipeline = importlib.import_module("tomojax.align.pipeline")
    setup_stage = importlib.import_module("tomojax.align._setup_stage")

    def fail_fold_recon(*args, **kwargs):
        raise AssertionError("setup loss validation must run before fold reconstruction")

    monkeypatch.setattr(setup_stage, "reconstruct_train_fold_nograd", fail_fold_recon)

    grid = Grid(nx=3, ny=3, nz=3, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=3, nv=3, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=np.linspace(0.0, 180.0, 4, endpoint=False, dtype=np.float32),
    )
    projections = jnp.zeros((4, 3, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="level 1.*direct.*phasecorr"):
        pipeline.align_multires(
            geometry,
            grid,
            detector,
            projections,
            factors=(1,),
            cfg=pipeline.AlignConfig(
                outer_iters=1,
                recon_iters=1,
                tv_prox_iters=1,
                optimise_dofs=("det_u_px",),
                loss=parse_loss_spec("phasecorr"),
                views_per_batch=1,
                checkpoint_projector=False,
                gather_dtype="fp32",
                recon_positivity=False,
                early_stop=False,
            ),
        )
