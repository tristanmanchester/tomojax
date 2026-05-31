from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tomojax.io._preprocess_impl.config import PreprocessConfig


def _view_tokens_from_text(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.split("#", 1)[0]
        if not line.strip():
            continue
        tokens.extend(part for part in line.replace(",", " ").split() if part)
    return tokens


def _read_view_spec_file(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read view index file {path!s}: {exc}") from exc


def _parse_nonnegative_int(text: str, *, label: str, token: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"invalid {label} {text!r} in view spec token {token!r}") from exc
    if value < 0:
        raise ValueError(f"negative view indices are not supported in token {token!r}")
    return value


def _parse_view_spec(text: str | None, *, n_views: int, label: str) -> np.ndarray:
    if text is None or not str(text).strip():
        return np.asarray([], dtype=np.int64)

    selected: dict[int, None] = {}
    for token in _view_tokens_from_text(str(text)):
        if ":" in token:
            parts = token.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(f"malformed {label} range {token!r}")
            if parts[0] == "" or parts[1] == "":
                raise ValueError(f"{label} ranges must provide explicit start and stop: {token!r}")
            start = _parse_nonnegative_int(parts[0], label="range start", token=token)
            stop = _parse_nonnegative_int(parts[1], label="range stop", token=token)
            step = 1
            if len(parts) == 3:
                if parts[2] == "":
                    raise ValueError(f"{label} ranges must provide an explicit step: {token!r}")
                step = _parse_nonnegative_int(parts[2], label="range step", token=token)
            if step <= 0:
                raise ValueError(f"{label} range step must be positive in token {token!r}")
            if stop <= start:
                raise ValueError(f"{label} range must be non-empty in token {token!r}")
            if start >= n_views or stop > n_views:
                raise ValueError(
                    f"{label} range {token!r} is out of bounds for {n_views} sample views"
                )
            values = range(start, stop, step)
            for value in values:
                selected[int(value)] = None
        else:
            value = _parse_nonnegative_int(token, label="view index", token=token)
            if value >= n_views:
                raise ValueError(
                    f"{label} index {value} is out of bounds for {n_views} sample views"
                )
            selected[int(value)] = None

    return np.asarray(sorted(selected), dtype=np.int64)


def _combine_view_specs(
    spec: str | None,
    file_path: str | Path | None,
    *,
    n_views: int,
    label: str,
) -> np.ndarray | None:
    parts: list[str] = []
    if spec is not None and str(spec).strip():
        parts.append(str(spec))
    if file_path is not None:
        parts.append(_read_view_spec_file(file_path))
    if not parts:
        return None
    return _parse_view_spec("\n".join(parts), n_views=n_views, label=label)


def _resolve_sample_view_indices(
    *,
    n_sample_views: int,
    config: PreprocessConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    select = _combine_view_specs(
        config.select_views,
        config.select_views_file,
        n_views=n_sample_views,
        label="select-views",
    )
    reject = _combine_view_specs(
        config.reject_views,
        config.reject_views_file,
        n_views=n_sample_views,
        label="reject-views",
    )

    if select is None:
        candidate = np.arange(n_sample_views, dtype=np.int64)
    else:
        candidate = select.astype(np.int64, copy=False)

    explicit_rejected = np.asarray([], dtype=np.int64)
    if reject is not None and reject.size:
        reject_set = {int(v) for v in reject.tolist()}
        keep_mask = np.asarray([int(v) not in reject_set for v in candidate], dtype=bool)
        explicit_rejected = candidate[~keep_mask]
        candidate = candidate[keep_mask]

    if candidate.size == 0:
        raise ValueError("view selection/rejection removed all sample views")

    meta = {
        "select_views": None if select is None else select.tolist(),
        "reject_views": [] if reject is None else reject.tolist(),
        "explicit_rejected_sample_view_indices": explicit_rejected.tolist(),
        "candidate_sample_view_indices": candidate.tolist(),
    }
    return candidate, meta


def _parse_crop_spec(
    crop: str | None,
    *,
    nv: int,
    nu: int,
) -> tuple[int, int, int, int] | None:
    if crop is None or not str(crop).strip():
        return None
    text = str(crop).strip()
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError("--crop must be formatted as y0:y1,x0:x1")

    def parse_axis(axis_text: str, limit: int, axis_name: str) -> tuple[int, int]:
        axis_parts = axis_text.split(":")
        if len(axis_parts) != 2 or axis_parts[0] == "" or axis_parts[1] == "":
            raise ValueError(f"--crop {axis_name} range must be formatted as start:stop")
        start = _parse_nonnegative_int(axis_parts[0], label=f"{axis_name} start", token=axis_text)
        stop = _parse_nonnegative_int(axis_parts[1], label=f"{axis_name} stop", token=axis_text)
        if stop <= start:
            raise ValueError(f"--crop {axis_name} range must be non-empty")
        if stop > limit:
            raise ValueError(
                f"--crop {axis_name} range {axis_text!r} is out of bounds for size {limit}"
            )
        return start, stop

    y0, y1 = parse_axis(parts[0].strip(), nv, "y")
    x0, x1 = parse_axis(parts[1].strip(), nu, "x")
    return y0, y1, x0, x1
