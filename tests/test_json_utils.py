from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tomojax.io import drop_none, normalize_json, read_json_object, write_json_object


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


def test_json_object_artifact_helpers_normalize_and_validate(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "payload.json"

    write_json_object(path, {"path": Path("scan.nxs"), "bad": float("inf")})

    assert path.read_text(encoding="utf-8").endswith("\n")
    assert read_json_object(path) == {"bad": "inf", "path": "scan.nxs"}

    list_path = tmp_path / "list.json"
    list_path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        read_json_object(list_path)
    with pytest.raises(ValueError, match="must normalize to a mapping"):
        write_json_object(tmp_path / "invalid.json", [1, 2, 3])
