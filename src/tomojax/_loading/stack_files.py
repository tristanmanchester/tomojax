"""Names belonging to one exported TIFF stack."""

from pathlib import Path


def slice_name(prefix: str, index: int) -> str:
    return f"{prefix}_{index:05d}.tif"


def tiff_stack_entry(path: Path, staged: Path) -> bool:
    """Replace staged entries and stale numbered slices belonging to their stack sidecar."""
    if not (path.is_file() or path.is_symlink()):
        return False
    if (staged / path.name).is_file():
        return True
    prefix, _, digits = path.stem.rpartition("_")
    return (
        digits.isascii()
        and digits.isdigit()
        and path.name == slice_name(prefix, int(digits))
        and (staged / f"{prefix}.json").is_file()
    )
