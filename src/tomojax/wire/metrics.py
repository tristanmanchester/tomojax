"""Scalar diagnostics carried by completed progress updates."""

from enum import Enum


class Metric(Enum):
    """The member name is the wire key. Each value gives its meaning and unit."""

    loss = ("Sampled reconstruction objective, null between samples", "objective units")
    normal_residual_norm = (
        "Euclidean norm of the CGLS normal-equation residual",
        "normal-equation units",
    )
    objective = ("Joint image and pose objective at the accepted step", "objective units")
    pose_step_rms = ("RMS accepted pose displacement", "working pixels")
    proposed_pose_step_rms = ("RMS pose displacement before line search", "working pixels")
    undamped_proposed_pose_step_rms = (
        "RMS proposal at initial damping, when evaluated",
        "working pixels",
    )
    line_step = ("Accepted fraction of the trial step", "fraction")
    pose_fraction = ("Pose fraction in the accepted trial", "fraction")
    lm = ("Levenberg-Marquardt damping multiplier", "1")
    gain_ratio = ("Actual objective decrease divided by predicted decrease", "1")
    predicted_reduction = ("Objective decrease predicted by the local model", "objective units")
    cg_iterations = ("Iterations in the Schur conjugate-gradient solve", "iterations")
    cg_relative_residual = ("Final residual norm relative to the initial norm", "1")
    cg_residual_kind = ("How the CG residual was evaluated", "category")
    cg_status = ("Convergence or budget exhaustion of the CG solve", "category")
    cg_matvecs = ("Matrix-vector products in the CG solve", "products")
    fraction_downweighted = (
        "Fraction of valid measurements downweighted by the robust loss",
        "fraction",
    )


def table() -> str:
    """Render the client reference from the registered wire names."""
    rows = [f"| `{metric.name}` | {metric.value[0]} | {metric.value[1]} |" for metric in Metric]
    return "\n".join(["| Metric | Meaning | Unit |", "|---|---|---|", *rows])
