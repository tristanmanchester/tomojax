from __future__ import annotations

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.align.losses import build_loss_adapter, parse_loss_spec
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
        "align",
        "align_multires",
        "_normalize_observer_action",
        "adapt_legacy_observer",
        "_should_prefer_gn_candidate",
        "_select_gn_candidate",
        "_is_expected_align_eval_failure",
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
        "_loss_cauchy",
        "_loss_chamfer_edge",
        "_loss_l2_otsu_soft",
        "_loss_mi_kde",
        "_loss_renyi_mi",
        "_loss_ssim_otsu",
        "_loss_tversky",
        "_loss_welsch",
    ],
)
def test_loss_compatibility_symbols_remain_importable(symbol: str) -> None:
    losses = importlib.import_module("tomojax.align.losses")

    assert hasattr(losses, symbol)


def test_loss_adapter_exposes_distinct_setup_validation_lm_capability() -> None:
    targets = jnp.zeros((2, 3, 3), dtype=jnp.float32)

    l2_adapter = build_loss_adapter(parse_loss_spec("l2"), targets)
    phasecorr_adapter = build_loss_adapter(parse_loss_spec("phasecorr"), targets)

    assert l2_adapter.supports_gauss_newton
    assert l2_adapter.supports_setup_validation_lm
    assert not phasecorr_adapter.supports_gauss_newton
    assert not phasecorr_adapter.supports_setup_validation_lm


def test_setup_stage_rejects_unsupported_loss_before_fold_reconstruction(monkeypatch) -> None:
    pipeline = importlib.import_module("tomojax.align.pipeline")
    setup_stage = importlib.import_module("tomojax.align._setup_stage")

    def fail_fold_recon(*args, **kwargs):
        raise AssertionError("setup loss validation must run before fold reconstruction")

    monkeypatch.setattr(pipeline, "reconstruct_train_fold_nograd", fail_fold_recon, raising=False)
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
