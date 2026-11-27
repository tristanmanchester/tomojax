"""Native destinations and dataset selectors supported by each projection reader."""

from pathlib import Path

from tomojax.errors import InvalidParameter

HDF5_SUFFIXES = frozenset((".nx", ".nxs", ".h5", ".hdf", ".hdf5"))


# Each reader declares which selectors it can resolve. The load contract tests
# exercise every format here with independently constructed files.
_HDF_PATHS = frozenset(("data_path", "angles_path", "image_key_path"))
DATASET_PATHS: dict[str, frozenset[str]] = {
    "nxtomo": _HDF_PATHS,
    "data_exchange": _HDF_PATHS,
    "nexus": _HDF_PATHS,
    "saved.nxs": _HDF_PATHS,
    "esrf_bliss": frozenset(("data_path",)),
    **dict.fromkeys(
        (
            "nikon_xtekct",
            "zeiss_ole",
            "skyscan",
            "esrf_edf",
            "esrf_pyhst",
            "tiff",
            "phase",
        ),
        frozenset[str](),
    ),
}


def check_dataset_paths(family: str, paths: dict[str, str | None]) -> None:
    """Verify every explicit selector has a corresponding dataset in this format."""
    supported = DATASET_PATHS[family]
    for name, value in paths.items():
        if value is not None and name not in supported:
            raise ValueError(
                f"{name} is unavailable for {family}: the acquisition fixes that field"
            )


def check_native_destination(path: Path) -> None:
    """Check that a destination uses a supported NeXus filename.

    Parameters
    ----------
    path : Path
        Native dataset destination.

    Raises
    ------
    InvalidParameter
        If the suffix cannot hold a saved TomoJAX dataset.
    """
    if path.suffix.lower() not in HDF5_SUFFIXES:
        known = ", ".join(sorted(HDF5_SUFFIXES))
        raise InvalidParameter("path", f"cannot save to {path.name}: give a path ending in {known}")
