"""Single-material beam hardening linearisation and its coefficient sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

import numpy as np

from tomojax.core import hardening as mixture

from ._steps import Domain, Step

if TYPE_CHECKING:
    from collections.abc import Sequence

    import jax

    from tomojax.geometry import Detector, Geometry, Grid


class _Source(Protocol):
    @property
    def geometry_metadata(self) -> dict[str, object]: ...


class _Scan(Protocol):
    """What the self-calibration reads from a :class:`tomojax.Scan`."""

    @property
    def projections(self) -> np.ndarray | jax.Array: ...
    @property
    def geometry(self) -> Geometry: ...
    @property
    def grid(self) -> Grid: ...
    @property
    def detector(self) -> Detector: ...
    @property
    def source(self) -> _Source | None: ...


@dataclass(frozen=True)
class BeamHardening(Step):
    """Linearise single-material beam hardening.

    A polychromatic beam's line integral through one material is
    ``q = g(p) = -ln sum_j w_j exp(-k_j p)``, where ``p`` is the material's
    line integral and the mixture ``(k_j, w_j)`` is the detected spectrum seen
    through the material's attenuation. Every such curve is increasing and
    concave and becomes linear at large ``p``, and every curve with those
    properties that a single material can make has this form; this step
    replaces each ``q`` by ``p = g^-1(q)``.

    ``rates`` are the ``k_j`` and ``weights`` the ``w_j`` (scaled to sum to one).
    With ``energy_kev=None`` the rates are scaled so that ``g'(0) = 1``: thin
    paths keep their attenuation, and the corrected line integrals are in the
    units of the unhardened beam's mean attenuation. With ``energy_kev`` the
    rates are the material's attenuation at each component's energy divided by
    its attenuation at ``energy_kev``, and the corrected line integrals are
    that energy's.

    The coefficients come from :meth:`fit` (self-calibration from a first
    reconstruction), :meth:`recorded` (the vendor's recorded linearisation),
    :meth:`from_polynomial` (any polynomial linearisation) or
    :meth:`from_spectrum` (the source spectrum, filters and detector).
    ``BeamHardening()`` leaves line integrals unchanged.

    Parameters
    ----------
    rates : tuple of float, optional
        Positive attenuation of each spectral component relative to the reference.
    weights : tuple of float, optional
        Nonnegative detected share of each component; scaled to sum to one.
    energy_kev : float or None, optional
        Reference energy whose attenuation the corrected line integrals give;
        None uses the unhardened beam's mean attenuation (unit initial slope).

    Raises
    ------
    ValueError
        If rates and weights differ in length, a rate is not positive, a weight
        is negative or all weights are zero, or a value is not finite.

    Examples
    --------
    >>> import numpy as np
    >>> from tomojax.corrections import BeamHardening
    >>> step = BeamHardening(rates=(0.5, 2.0), weights=(0.5, 0.5))
    >>> step.rates
    (0.4, 1.6)
    >>> float(step.inverse(step.curve(3.0)))
    3.0
    """

    domain: ClassVar[Domain] = "line integrals"
    rates: tuple[float, ...] = (1.0,)
    weights: tuple[float, ...] = (1.0,)
    energy_kev: float | None = None

    def __post_init__(self) -> None:
        if self.energy_kev is not None and not (
            np.isfinite(self.energy_kev) and self.energy_kev > 0
        ):
            raise ValueError("BeamHardening energy_kev must be positive and finite")
        rates, weights = mixture.normalise(
            tuple(self.rates), tuple(self.weights), unit_slope=self.energy_kev is None
        )
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "weights", weights)

    def apply(self, batch: jax.Array) -> jax.Array:
        """Replace measured line integrals by the material's.

        Parameters
        ----------
        batch : jax.Array
            Float32 line integrals of shape ``(views, rows, columns)``.

        Returns
        -------
        jax.Array
            Linearised line integrals, retaining the input shape.
        """
        if self.rates == (1.0,):
            return batch
        return mixture.invert(self.rates, self.weights, batch)

    def curve(self, p: np.ndarray | float) -> np.ndarray:
        """Measured line integrals ``g(p)`` of material line integrals ``p``.

        Parameters
        ----------
        p : array-like or float
            Material line integrals in this step's units.

        Returns
        -------
        numpy.ndarray
            The polychromatic line integrals, in float64.
        """
        return mixture.curve(self.rates, self.weights, np.asarray(p))

    def inverse(self, q: np.ndarray | float) -> np.ndarray:
        """Material line integrals ``g^-1(q)`` of measured ``q``, in float64 on the host.

        Parameters
        ----------
        q : array-like or float
            Measured line integrals.

        Returns
        -------
        numpy.ndarray
            The linearised line integrals that :meth:`apply` approximates in float32.
        """
        return mixture.inverse(self.rates, self.weights, np.asarray(q))

    @classmethod
    def fit(
        cls,
        scan: _Scan,
        *,
        threshold: float | None = None,
        iterations: int = 2,
    ) -> BeamHardening:
        """Self-calibrate from the scan: segment a reconstruction and reproject it.

        Each pass reconstructs the scan by FBP (FDK for cone beams) and divides
        the image into air and material levels, each a peak of the histogram
        with an interior, so that a second material is a level of its own and
        cupping within one material is not. Each level is flattened to its
        mean, the scan's own projector turns that into each ray's material
        line integral, and the curve is fitted to the measured line integrals
        against those (weighted least squares in transmission, each ray
        weighted by its Poisson precision). The curve is the most common
        material's, fitted on the rays that cross it alone; other materials
        are corrected as denser or lighter versions of it, and a phase too
        rare to be a level (a thin metal rod) keeps its own values. Later passes
        reconstruct the data corrected by the previous fit, so the levels see
        less cupping. The result is in the unit-initial-slope gauge.

        The object should sit inside the reconstruction grid in every view,
        clear of the background. Scatter, a truncated object and materials
        whose spectral shapes differ (metal in plastic) bias the curve. The
        gauge rests on the shortest paths through the material, so a binned
        scan, or voxels coarser than the pixels, can change the image's scale.

        Parameters
        ----------
        scan : Scan
            Line integrals with their geometry and reconstruction grid.
        threshold : float or None, optional
            Attenuation separating the material from the background (air,
            pores) in the reconstruction; None takes the deepest valley of its
            histogram between them.
        iterations : int, optional
            Reconstruct-segment-fit passes, at least one. Default is 2.

        Returns
        -------
        BeamHardening
            The fitted linearisation.

        Raises
        ------
        ValueError
            If ``iterations`` is not positive, no material stands out from the
            background, or no ray crosses one material alone.
        """
        from tomojax.recon.polychromatic import segmented_curve

        rates, weights, _ = segmented_curve(
            scan.geometry,
            scan.grid,
            scan.detector,
            np.asarray(scan.projections, np.float32),
            threshold=threshold,
            iterations=iterations,
        )
        return cls(rates=rates, weights=weights)

    @classmethod
    def recorded(cls, scan: _Scan) -> BeamHardening:
        """The linearisation the scan's vendor recorded, as the closest physical curve.

        Nikon CT Pro records ``p = sum_n CoefXn q^n`` in the ``.xtekct`` file;
        it becomes :meth:`from_polynomial` over the scan's measured line
        integrals (identity coefficients give ``BeamHardening()``). Zeiss's
        constant and Bruker's percentage are recorded with the scan's metadata
        (``scan.source.geometry_metadata["beam_hardening"]``) but enter
        formulas their vendors do not publish, so they are not applied.

        Parameters
        ----------
        scan : Scan
            A scan loaded from a vendor's files.

        Returns
        -------
        BeamHardening
            The vendor's linearisation.

        Raises
        ------
        ValueError
            If the scan records no linearisation TomoJAX can apply.
        """
        source = scan.source
        settings = None if source is None else source.geometry_metadata.get("beam_hardening")
        if not isinstance(settings, dict):
            raise ValueError("the scan records no beam hardening linearisation")
        if settings.get("vendor") != "nikon":
            raise ValueError(
                f"{settings.get('vendor')} records beam hardening setting {settings}, whose "
                "formula is not published; fit the curve with BeamHardening.fit(scan)"
            )
        coefficients = [float(c) for c in cast("list[float]", settings["coefficients"])]
        if coefficients[:2] == [0.0, 1.0] and not any(coefficients[2:]):
            return cls()
        measured = np.asarray(scan.projections, np.float32)
        return cls.from_polynomial(coefficients, upto=float(np.nanmax(measured)))

    @classmethod
    def from_polynomial(cls, coefficients: Sequence[float], *, upto: float) -> BeamHardening:
        """The physical curve closest to a polynomial linearisation ``p = sum_n c_n q^n``.

        Vendors record beam hardening as a polynomial in the measured line
        integral (Nikon's ``CoefX0`` to ``CoefX4``). This fits the mixture whose
        inverse best matches it for ``0 <= q <= upto`` (weighted as
        :meth:`fit` weights rays), then rescales it to unit initial slope; a
        polynomial that is not the inverse of a physical curve (one with zero
        initial slope, say) is replaced by the closest one that is.

        Parameters
        ----------
        coefficients : sequence of float
            ``(c0, c1, c2, ...)``, lowest power first.
        upto : float
            Largest measured line integral the curve must cover, such as the
            scan's maximum.

        Returns
        -------
        BeamHardening
            The fitted linearisation.

        Raises
        ------
        ValueError
            If ``upto`` is not positive or the polynomial gives no positive line integral.
        """
        if not (np.isfinite(upto) and upto > 0):
            raise ValueError("from_polynomial needs a positive, finite upto")
        q = np.linspace(0.0, float(upto), 2049)
        p = np.polynomial.polynomial.polyval(q, np.asarray(coefficients, np.float64))
        rates, weights, _ = mixture.fit(np.maximum(p, 0.0), q)
        return cls(rates=rates, weights=weights)

    @classmethod
    def from_spectrum(
        cls,
        energies: Sequence[float] | np.ndarray,
        weights: Sequence[float] | np.ndarray,
        material: str,
        *,
        energy_kev: float | None = None,
    ) -> BeamHardening:
        """The curve of one material seen by a known detected spectrum.

        Each energy bin is a component whose rate is the material's
        attenuation there (from xraylib), relative to its attenuation at
        ``energy_kev`` or, with None, to the spectrum's mean.
        :func:`~tomojax.corrections.tube_spectrum` models a tube's spectrum
        from its voltage, filters and scintillator; a measured or simulated
        spectrum works the same way.

        Parameters
        ----------
        energies : array-like
            Bin energies in keV.
        weights : array-like
            Detected signal per bin (for a scintillator, absorbed energy).
        material : str
            Chemical formula ("CuO") or NIST material name of the sample.
        energy_kev : float or None, optional
            Energy whose attenuation the corrected line integrals give; None
            gives unit initial slope.

        Returns
        -------
        BeamHardening
            The linearisation for that material and spectrum.

        Raises
        ------
        ValueError
            If the spectrum is empty or the material unknown.
        ImportError
            If xraylib is not installed.
        """
        from ._spectrum import attenuation

        e = np.asarray(energies, np.float64).ravel()
        w = np.asarray(weights, np.float64).ravel()
        if e.size == 0 or e.shape != w.shape or (e <= 0).any():
            raise ValueError("from_spectrum needs equal-length positive energies and weights")
        mu = attenuation(material, e)
        reference = (
            1.0 if energy_kev is None else float(attenuation(material, np.asarray([energy_kev]))[0])
        )
        return cls(rates=tuple(mu / reference), weights=tuple(w), energy_kev=energy_kev)
