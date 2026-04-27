from __future__ import annotations

"""Compatibility facade for alignment loss specs, kernels, and adapters."""

from ._loss_adapters import (
    GaussNewtonWeightFn,
    LossAdapter,
    LossBuilderFn,
    PerViewLossFn,
    build_loss,
    build_loss_adapter,
    loss_supports_setup_validation_lm,
)
from ._loss_kernels import (
    _compute_otsu_threshold,
    _loss_barron,
    _loss_cauchy,
    _loss_chamfer_edge,
    _loss_charbonnier,
    _loss_correntropy,
    _loss_edge_aware_l2,
    _loss_fft_mag,
    _loss_grad_l1,
    _loss_grad_orient,
    _loss_huber,
    _loss_l2,
    _loss_l2_otsu_soft,
    _loss_mi_kde,
    _loss_mind,
    _loss_ms_ssim,
    _loss_ngf,
    _loss_phase_corr_soft,
    _loss_poisson_nll,
    _loss_pwls,
    _loss_renyi_mi,
    _loss_ssim,
    _loss_ssim_otsu,
    _loss_student_t,
    _loss_swd,
    _loss_tversky,
    _loss_welsch,
    _loss_zncc,
    _safe_epsilon,
    _sobel,
    _validated_renyi_alpha,
)
from ._loss_specs import (
    AlignmentLossConfig,
    AlignmentLossSchedule,
    AlignmentLossSpec,
    CorrelationLossSpec,
    EdgeL2LossSpec,
    GradientLossSpec,
    InformationLossSpec,
    L2LossSpec,
    L2OtsuLossSpec,
    LossScheduleEntry,
    MindLossSpec,
    PWLSLossSpec,
    PoissonLossSpec,
    RobustLossSpec,
    SSIMLossSpec,
    SWDLossSpec,
    TverskyLossSpec,
    canonicalize_loss_kind,
    loss_is_within_relative_tolerance,
    loss_spec_name,
    loss_spec_params,
    parse_loss_schedule,
    parse_loss_spec,
    resolve_loss_for_level,
    validate_loss_schedule_levels,
)
from ._loss_state import LossState

__all__ = [
    name
    for name in globals()
    if not name.startswith("__") and name != "annotations"
]
