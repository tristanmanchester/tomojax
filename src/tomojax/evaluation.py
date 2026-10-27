"""Frozen observation splits and additive, matched-prediction held-out scores.

This module neither fits a model nor chooses a projector. Predictions must use
exactly the fit's discretisation, geometry and intensity convention. Exclude test
and guard observations from *all* fitted stages, including initialisation and
corrections; a mask on the final loss alone does not establish independence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

__all__ = ["Score", "Split"]


@dataclass(frozen=True)
class Score:
    """Additive squared-error statistics, with no fitted intensity scale.

    Parameters
    ----------
    squared_error : float
        Sum of precision times squared prediction error.
    target_energy : float
        Sum of precision times squared observed values, for relative error.
    weight_sum : float
        Sum of positive precisions.
    count : int
        Number of test observations with positive precision.
    """

    squared_error: float
    target_energy: float
    weight_sum: float
    count: int

    @property
    def mse(self) -> float:
        """Weighted mean squared error; NaN for an empty scored batch.

        Returns
        -------
        float
            Squared error divided by precision sum.
        """
        return self.squared_error / self.weight_sum if self.weight_sum else float("nan")

    @property
    def relative_l2(self) -> float:
        """Relative weighted L2 error; NaN if the target has no energy.

        Returns
        -------
        float
            Square root of squared error divided by weighted target energy.
        """
        return (
            (self.squared_error / self.target_energy) ** 0.5 if self.target_energy else float("nan")
        )

    def __add__(self, other: Score) -> Score:
        return Score(
            self.squared_error + other.squared_error,
            self.target_energy + other.target_energy,
            self.weight_sum + other.weight_sum,
            self.count + other.count,
        )


def _shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(shape) != 3 or any(int(n) != n or n < 1 for n in shape):
        raise ValueError("shape must contain positive (views, rows, columns)")
    return int(shape[0]), int(shape[1]), int(shape[2])


def _immutable(mask: np.ndarray) -> np.ndarray:
    # An immutable bytes owner prevents callers from re-enabling writeability.
    array = np.asarray(mask)
    if array.dtype != bool or array.ndim != 3:
        raise ValueError("split masks must be three-dimensional boolean arrays")
    return np.frombuffer(array.tobytes(), dtype=bool).reshape(array.shape)


@dataclass(frozen=True, kw_only=True, eq=False)
class Split:
    """Immutable training/test masks in acquisition coordinates.

    Masks are broadcastable to ``shape`` and may omit detector rows or columns
    by using singleton axes. False in both masks means excluded (a guard band).
    The fingerprint identifies this split, not the data: archive it with the
    source, selected frame IDs, geometry, correction settings and fit settings.

    Parameters
    ----------
    shape : tuple of int
        Full projection shape, (views, rows, columns).
    train : numpy.ndarray
        Boolean training mask, broadcastable to shape.
    test : numpy.ndarray
        Disjoint boolean held-out mask, broadcastable to shape.
    specification : str
        JSON description of how the split was frozen, without fitted values.

    Raises
    ------
    ValueError
        If masks overlap, cannot broadcast, or leave no training/test data.
    """

    shape: tuple[int, int, int]
    train: np.ndarray
    test: np.ndarray
    specification: str

    def __post_init__(self) -> None:
        shape = _shape(self.shape)
        train, test = _immutable(self.train), _immutable(self.test)
        if np.broadcast_shapes(shape, train.shape, test.shape) != shape:
            raise ValueError("split masks must broadcast to the projection shape")
        if np.any(train & test) or not train.any() or not test.any():
            raise ValueError("training and test masks must be nonempty and disjoint")
        spec = json.dumps(json.loads(self.specification), sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "test", test)
        object.__setattr__(self, "specification", spec)

    @property
    def fingerprint(self) -> str:
        """SHA256 of shape, specification and the compact masks.

        Returns
        -------
        str
            Hexadecimal digest identifying this frozen split.
        """
        digest = hashlib.sha256(str(self.shape).encode() + self.specification.encode())
        for mask in (self.train, self.test):
            digest.update(str(mask.shape).encode())
            digest.update(np.packbits(mask).tobytes())
        return digest.hexdigest()

    @classmethod
    def interleaved(cls, shape: tuple[int, int, int], *, stride: int, offset: int = 0) -> Split:
        """Hold every ``stride``-th view out for reconstruction selection.

        Parameters
        ----------
        shape : tuple of int
            Full projection shape (views, rows, columns).
        stride : int
            At least two; repeat at several strides to test angular sampling bias.
        offset : int, optional
            Held-out residue in [0, stride).

        Returns
        -------
        Split
            View masks with singleton detector axes.
        """
        shape = _shape(shape)
        if stride < 2 or int(stride) != stride or not 0 <= offset < stride or int(offset) != offset:
            raise ValueError("stride must be an integer >=2 and offset an integer in [0,stride)")
        test = (np.arange(shape[0]) % stride == offset)[:, None, None]
        return cls(
            shape=shape,
            train=~test,
            test=test,
            specification=json.dumps({"kind": "interleaved", "stride": stride, "offset": offset}),
        )

    @classmethod
    def angular(
        cls,
        shape: tuple[int, int, int],
        angles: np.ndarray,
        *,
        start: float,
        width: float,
        guard: float = 0,
        period: float = 360,
    ) -> Split:
        """Hold a periodic angular block out for global or smooth parameters.

        Parameters
        ----------
        shape : tuple of int
            Full projection shape (views, rows, columns).
        angles : numpy.ndarray
            One recorded angle per view in degrees; acquisition order is retained.
        start : float
            First test angle in degrees, modulo period.
        width : float
            Test interval width in degrees; its upper endpoint is excluded.
        guard : float, optional
            Excluded angular margin on either side of each test block.
        period : float, optional
            Block repetition period in degrees, default 360.

        Returns
        -------
        Split
            Angular test block and guarded training views, never per-view pose validation.
        """
        shape = _shape(shape)
        angles = np.asarray(angles, np.float64)
        if angles.shape != (shape[0],) or not np.isfinite(angles).all():
            raise ValueError("angles must contain one finite degree value per view")
        train, test = _blocks(angles, start=start, width=width, guard=guard, period=period)
        spec = {
            "kind": "angular",
            "start": start,
            "width": width,
            "guard": guard,
            "period": period,
            "angles_sha256": hashlib.sha256(angles.tobytes()).hexdigest(),
        }
        return cls(
            shape=shape,
            train=train[:, None, None],
            test=test[:, None, None],
            specification=json.dumps(spec),
        )

    @classmethod
    def detector(
        cls,
        shape: tuple[int, int, int],
        *,
        start: int,
        width: int,
        period: int,
        guard: int = 0,
        offsets: np.ndarray | float = 0,
    ) -> Split:
        """Hold detector-column blocks out within each view for per-view poses.

        Parameters
        ----------
        shape : tuple of int
            Full projection shape (views, rows, columns).
        start : int
            First test column, modulo period, before adding offsets.
        width : int
            Number of consecutive test columns in each block.
        period : int
            Block repetition period in detector pixels.
        guard : int, optional
            Columns excluded on each side of a test block. Choose this from the
            support of preprocessing and permitted motion, before fitting.
        offsets : numpy.ndarray or float, optional
            Frozen offset per view or a common offset, in detector pixels.

        Returns
        -------
        Split
            Masks with a singleton row axis and a test block in each view.
        """
        shape = _shape(shape)
        if any(int(v) != v for v in (start, width, period, guard)):
            raise ValueError("detector block boundaries must be integer pixels")
        offsets = np.broadcast_to(np.asarray(offsets, np.float64), (shape[0],))
        if not np.isfinite(offsets).all() or np.any(offsets != np.floor(offsets)):
            raise ValueError("offsets must be finite integer pixel offsets")
        train, test = _blocks(
            np.arange(shape[2])[None, :] - offsets[:, None],
            start=start,
            width=width,
            guard=guard,
            period=period,
        )
        if not np.all(test.any(axis=1)) or not np.all(train.any(axis=1)):
            raise ValueError("detector split needs training and test columns in every view")
        spec = {
            "kind": "detector",
            "start": start,
            "width": width,
            "period": period,
            "guard": guard,
            "offsets": offsets.tolist(),
        }
        return cls(
            shape=shape,
            train=train[:, None, :],
            test=test[:, None, :],
            specification=json.dumps(spec),
        )

    def score(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        *,
        weights: np.ndarray | float = 1,
        views: slice | None = None,
    ) -> Score:
        """Score a complete prediction or a view batch without refitting anything.

        Parameters
        ----------
        prediction : numpy.ndarray
            Matched forward prediction (batch views, rows, columns).
        target : numpy.ndarray
            Held-out observations with the same shape and intensity convention.
        weights : numpy.ndarray or float, optional
            Nonnegative precision, broadcastable to the batch; not square-root
            precision. Freeze data-adaptive weights using training data only.
        views : slice or None, optional
            Batch's original view indices; default is the complete acquisition.

        Returns
        -------
        Score
            Additive float64 statistics; sum batches before forming ratios.

        Raises
        ------
        ValueError
            If shapes disagree, weights are negative, or scored values are nonfinite.
        """
        views = slice(None) if views is None else views
        indices = np.arange(self.shape[0])[views]
        expected = (len(indices), *self.shape[1:])
        if np.shape(prediction) != expected or np.shape(target) != expected:
            raise ValueError(f"prediction and target must have batch shape {expected}")
        compact = np.broadcast_to(self.test, (self.shape[0], *self.test.shape[1:]))[views]
        mask = np.broadcast_to(compact, expected)
        w = np.broadcast_to(np.asarray(weights), expected)[mask].astype(np.float64)
        observed = np.asarray(target)[mask].astype(np.float64)
        predicted = np.asarray(prediction)[mask].astype(np.float64)
        if not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("scored weights must be finite and nonnegative")
        keep = w > 0
        observed, predicted, w = observed[keep], predicted[keep], w[keep]
        if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
            raise ValueError("positive-weight held-out observations/predictions must be finite")
        residual = predicted - observed
        return Score(
            float(np.sum(w * residual**2)),
            float(np.sum(w * observed**2)),
            float(np.sum(w)),
            int(keep.sum()),
        )


def _blocks(
    coordinates: np.ndarray,
    *,
    start: float,
    width: float,
    guard: float,
    period: float,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        not np.isfinite([start, width, guard, period]).all()
        or width <= 0
        or guard < 0
        or width + 2 * guard >= period
    ):
        raise ValueError(
            "block width must be positive, guard nonnegative, and width+2*guard < period"
        )
    phase = (coordinates - start) % period
    return (phase >= width + guard) & (phase < period - guard), phase < width
