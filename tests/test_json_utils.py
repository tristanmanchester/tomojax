from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tomojax.utils.json import drop_none, normalize_json


@dataclass
class _Config:
    path: Path
    values: np.ndarray


class _DictBacked:
    def to_dict(self) -> dict[str, object]:
        return {"scale": np.float32(1.5)}


class _BrokenDict:
    def to_dict(self) -> dict[str, object]:
        raise RuntimeError("broken")


def test_normalize_json_handles_common_runtime_values() -> None:
    payload = {
        "config": _Config(Path("input.nxs"), np.asarray([1, 2, 3])),
        "custom": _DictBacked(),
        "bad_float": float("inf"),
    }

    assert normalize_json(payload) == {
        "config": {"path": "input.nxs", "values": [1, 2, 3]},
        "custom": {"scale": pytest.approx(1.5)},
        "bad_float": "inf",
    }


def test_normalize_json_keeps_call_site_policy_explicit() -> None:
    namespace = argparse.Namespace(path=Path("out.nxs"))
    unsorted = {"b": 2, "a": 1}

    assert normalize_json(namespace, namespace=True) == {"path": "out.nxs"}
    assert list(normalize_json(unsorted, sort_mapping_keys=True)) == ["a", "b"]
    assert normalize_json(_BrokenDict(), catch_to_dict_errors=True).startswith("<")
    with pytest.raises(RuntimeError, match="broken"):
        normalize_json(_BrokenDict())


def test_drop_none_uses_shared_normalization_options() -> None:
    assert drop_none({"keep": np.float32(2.5), "skip": None}) == {"keep": pytest.approx(2.5)}
