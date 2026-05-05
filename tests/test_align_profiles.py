from __future__ import annotations

import pytest

from tomojax.align._profiles import (
    alignment_profile_policy,
    normalize_alignment_profile,
    profile_policy_from_config,
    resolve_profiled_cli_defaults,
)
from tomojax.align._quality_policy import (
    normalize_quality_tier,
    reconstruction_quality_policy,
)
from tomojax.align.pipeline import AlignConfig


def test_default_align_config_uses_lightning_profile_defaults():
    cfg = AlignConfig()

    assert cfg.align_profile == "lightning"
    assert cfg.projector_backend == "pallas"
    assert cfg.regulariser == "huber_tv"
    assert cfg.gather_dtype == "auto"
    assert cfg.views_per_batch == 0
    assert cfg.pose_model == "spline"
    assert cfg.quality_tier == "fast"
    assert cfg.fallback_policy == "fallback"


def test_tortoise_align_config_uses_reference_profile_defaults():
    cfg = AlignConfig(align_profile="tortoise")

    assert cfg.align_profile == "tortoise"
    assert cfg.projector_backend == "jax"
    assert cfg.regulariser == "tv"
    assert cfg.gather_dtype == "fp32"
    assert cfg.views_per_batch == 1
    assert cfg.pose_model == "per_view"
    assert cfg.quality_tier == "reference"


def test_align_profile_normalization_accepts_reference_aliases():
    assert normalize_alignment_profile("default") == "lightning"
    assert normalize_alignment_profile("reference") == "tortoise"

    with pytest.raises(ValueError, match="align_profile"):
        normalize_alignment_profile("unknown")


def test_profiled_cli_defaults_do_not_override_configured_values():
    resolved = resolve_profiled_cli_defaults(
        align_profile="lightning",
        current={
            "projector_backend": "jax",
            "gather_dtype": "fp32",
            "regulariser": "tv",
            "recon_algo": "fista",
            "views_per_batch": 1,
            "checkpoint_projector": True,
            "pose_model": "per_view",
        },
        configured_keys={"projector_backend", "gather_dtype"},
    )

    assert resolved["align_profile"] == "lightning"
    assert resolved["projector_backend"] == "jax"
    assert resolved["gather_dtype"] == "fp32"
    assert resolved["regulariser"] == "huber_tv"
    assert resolved["views_per_batch"] == 0
    assert resolved["pose_model"] == "spline"
    assert resolved["profile_defaults"] == alignment_profile_policy("lightning").to_dict()


def test_profile_policy_snapshot_records_effective_config():
    cfg = AlignConfig(align_profile="lightning", gather_dtype="fp32")
    policy = profile_policy_from_config(cfg).to_dict()

    assert policy["align_profile"] == "lightning"
    assert policy["projector_backend"] == "pallas"
    assert policy["gather_dtype"] == "fp32"
    assert policy["quality_tier"] == "fast"


def test_reconstruction_quality_policy_names_stage_diagnostics():
    fast = reconstruction_quality_policy("fast")
    final = reconstruction_quality_policy("final")

    assert fast.prefer_mixed_precision is True
    assert fast.compute_final_data_loss is False
    assert final.final_quality is True
    assert final.compute_final_regulariser_value is True
    assert normalize_quality_tier("tortoise") == "reference"
