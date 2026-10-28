"""Validate NumPy-style docstrings on the supported Python API."""

from __future__ import annotations

import argparse
from importlib import import_module
import inspect
from typing import TYPE_CHECKING, Any, TypeGuard

from numpydoc.docscrape import get_doc_object
from numpydoc.validate import validate

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

PUBLIC_MODULES = (
    "tomojax",
    "tomojax.recon",
    "tomojax.corrections",
    "tomojax.io",
    "tomojax.alignment",
    "tomojax.geometry",
    "tomojax.evaluation",
    "tomojax.datasets",
    "tomojax.forward",
    "tomojax.forward.api",
    "tomojax.recon.api",
    "tomojax.io.api",
    "tomojax.alignment.api",
    "tomojax.geometry.api",
)
# Immutable values and typing aliases cannot own docstrings. Their reference
# entries live here so every export is covered by either numpydoc or this prose.
PUBLIC_DATA = {
    "tomojax.forward.api.PROJECTION_OPERATOR": (
        "Reference projection operator name, core_trilinear_ray."
    ),
    "tomojax.forward.api.ProjectionOperatorName": (
        "Literal name of the reference projection operator."
    ),
    "tomojax.forward.api.ResidualFilterKind": (
        "Residual filter: raw, lowpass_gaussian or bandpass_difference_of_gaussians."
    ),
    "tomojax.forward.api.ResidualLossMode": "Residual loss: l2 or pseudo_huber.",
    "tomojax.__version__": "Installed TomoJAX release version.",
    "tomojax.alignment.MODES": (
        "Explicit workflow alignment modes: pose, cor, cor-then-pose and full."
    ),
    "tomojax.recon.api.MethodConfig": (
        "A method's configuration: FBPConfig, FourierConfig, CGLSConfig, FistaConfig, "
        "SPDHGConfig, SIRTConfig or MLEMConfig."
    ),
    "tomojax.io.api.JsonValue": (
        "Recursive JSON value: null, boolean, number, string, list or mapping."
    ),
    "tomojax.alignment.api.MODES": "Explicit alignment modes, also exported by tomojax.alignment.",
    "tomojax.alignment.api.POSE_WIDTH": "Six pose columns: alpha, beta, phi, dx, dz and dy.",
    "tomojax.alignment.api.PUBLIC_SCHEDULE_PRESETS": (
        "Names accepted as public alignment schedules."
    ),
    "tomojax.alignment.api.AlignmentLossConfig": (
        "One alignment loss specification or a loss schedule."
    ),
    "tomojax.alignment.api.AlignmentLossSpec": (
        "Typed specification of a projection-domain alignment loss."
    ),
    "tomojax.alignment.api.AlignmentMode": (
        "Literal pose, cor, cor-then-pose or full workflow mode."
    ),
    "tomojax.alignment.api.DofBounds": (
        "Tuple of (parameter name, lower bound, upper bound) triples."
    ),
    "tomojax.alignment.api.GaugePolicy": (
        "Gauge policy: reject, anchor_mean, prior_required or diagnose_only."
    ),
    "tomojax.alignment.api.OuterStat": (
        "Mapping of outer-iteration diagnostic names to recorded values."
    ),
    "tomojax.alignment.api.PoseTranslationFrame": "Translation frame: object or detector.",
    "tomojax.alignment.api.QualityTier": "Alignment quality profile: fast or reference.",
    "tomojax.geometry.api.DISK_VOLUME_AXES": "Default on-disk volume axis order, zyx.",
    "tomojax.geometry.api.INTERNAL_VOLUME_AXES": "Internal Python volume axis order, xyz.",
    "tomojax.geometry.api.VOLUME_AXES_ATTR": (
        "File attribute recording volume axis order: volume_axes_order."
    ),
}


def _is_library_callable(obj: Any) -> TypeGuard[Callable[..., Any]]:
    return callable(obj) and (getattr(obj, "__module__", "") or "").startswith("tomojax")


def public_data(modules: Sequence[str] = PUBLIC_MODULES) -> Iterator[str]:
    """Yield the exported constants and typing aliases that need prose entries."""
    for module_name in modules:
        module = import_module(module_name)
        for name in module.__all__:
            if not _is_library_callable(getattr(module, name)):
                yield f"{module_name}.{name}"


MAIN_ENTRY_POINTS = frozenset(
    f"tomojax.{name}"
    for name in (
        "Scan",
        "load",
        "load_frames",
        "load_reconstruction",
        "save",
        "project",
        "backproject",
        "reconstruct",
        "reconstruct_slabs",
        "align",
        "estimate_center",
        "estimate_region",
    )
)
# Cross-references and extra prose are useful when needed, not mandatory filler.
IGNORED = {
    "ES01": "A summary plus parameter descriptions can be sufficient.",
    "SA01": "Related APIs can be linked in prose instead of a See Also section.",
}


def public_objects(modules: Sequence[str] = PUBLIC_MODULES) -> Iterator[tuple[str, Any]]:
    """Yield exported callables and their public methods, without duplicate aliases."""
    seen: set[int] = set()
    for module_name in modules:
        module = import_module(module_name)
        for name in module.__all__:
            obj = getattr(module, name)
            # Constants and typing aliases have no callable documentation contract.
            if not _is_library_callable(obj):
                continue
            candidates: list[tuple[str, Any]] = [(f"{module_name}.{name}", obj)]
            if inspect.isclass(obj):
                for member, value in inspect.getmembers(obj):
                    if not member.startswith("_") and (
                        callable(value) or isinstance(value, property)
                    ):
                        candidates.append((f"{module_name}.{name}.{member}", value))
            for qualified_name, candidate in candidates:
                value = candidate
                if isinstance(value, property):
                    value = value.fget
                if inspect.ismethod(value):
                    value = value.__func__
                if not _is_library_callable(value):
                    continue
                value = inspect.unwrap(value)
                if id(value) not in seen:
                    seen.add(id(value))
                    yield qualified_name, value


def docstring_errors(name: str, obj: Any) -> list[tuple[str, str]]:
    """Return actionable numpydoc errors for one public callable."""
    # Resolve objects ourselves: numpydoc's string resolver can import a same-named
    # submodule instead of validating the exported function (for example cgls).
    result = validate(get_doc_object(obj))
    ignored = set(IGNORED)
    if name not in MAIN_ENTRY_POINTS:
        ignored.add("EX01")  # Small helpers need examples only where they add information.
    if getattr(obj, "_is_protocol", False):
        ignored.add("PR01")  # Protocols describe interfaces, not (*args, **kwargs) constructors.
    return [(code, message) for code, message in result["errors"] if code not in ignored]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public docstring check, optionally restricted to package roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("modules", nargs="*", help="Public modules (default: supported API).")
    args = parser.parse_args(argv)
    failed = False
    count = 0
    for name, obj in public_objects(args.modules or PUBLIC_MODULES):
        count += 1
        for code, message in docstring_errors(name, obj):
            failed = True
            print(f"{name}: {code} {message}")
    for name in public_data(args.modules or PUBLIC_MODULES):
        if name not in PUBLIC_DATA:
            failed = True
            print(f"{name}: missing constant/type-alias reference entry")
    print(f"Checked {count} public docstrings.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
