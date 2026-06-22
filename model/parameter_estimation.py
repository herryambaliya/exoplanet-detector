"""
parameter_estimation.py
-----------------------
Member 3 (ML Engineer) — model/ folder
Fits a trapezoidal transit model to a phase-folded light curve and
extracts:
  • orbital period      (T)   — passed in from M2's BLS output
  • transit duration    (τ)   — fitted
  • transit depth       (δ)   — fitted
  • ingress/egress time (t12) — fitted
  • SNR / confidence    (σ)

Inputs  : 1-D numpy arrays  (phase, flux, flux_err)  from M2's detect.py
Outputs : TransitParams dataclass  +  diagnostic dict
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import curve_fit, minimize
from scipy.stats import chi2
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Transit Model
# ─────────────────────────────────────────────────────────────────────────────

def trapezoidal_transit(
    phase: np.ndarray,
    depth: float,
    duration: float,
    ingress_frac: float,
    center: float,
) -> np.ndarray:
    """
    Trapezoidal box model for a single transit.

    Parameters
    ----------
    phase        : Phase array (units: same as period, typically days).
                   Should be centered near 0 for the transit.
    depth        : Transit depth (fractional flux drop, positive value).
    duration     : Full transit duration  T14  (same units as phase).
    ingress_frac : Fraction of duration that is ingress/egress  (0 < f < 0.5).
                   ingress time  t12 = ingress_frac * duration.
    center       : Phase of transit center (usually ≈ 0 after folding).

    Returns
    -------
    flux_model : np.ndarray  (1 = out-of-transit baseline)
    """
    ingress_frac = np.clip(ingress_frac, 1e-4, 0.499)
    t12 = ingress_frac * duration          # ingress / egress half-width
    t14 = duration / 2.0                   # half total duration

    p = phase - center
    flux = np.ones_like(p, dtype=float)

    # Full flat bottom
    in_flat = np.abs(p) <= (t14 - t12)
    flux[in_flat] = 1.0 - depth

    # Ingress ramp (left side)
    in_ingress = (np.abs(p) > (t14 - t12)) & (np.abs(p) <= t14)
    ramp = (t14 - np.abs(p[in_ingress])) / t12
    flux[in_ingress] = 1.0 - depth * ramp

    return flux


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Result Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransitParams:
    """All estimated parameters for one transit event."""

    # ── Core physical parameters ──────────────────────────────────────────
    period_days: float            # Orbital period  (from BLS, passed in)
    depth_ppm: float              # Transit depth   in parts-per-million
    duration_hours: float         # Transit duration T14 in hours
    ingress_fraction: float       # t12 / T14  (0–0.5)
    center_phase: float           # Phase of best-fit transit center

    # ── Uncertainties  (1-σ) ─────────────────────────────────────────────
    depth_ppm_err: float = 0.0
    duration_hours_err: float = 0.0
    ingress_fraction_err: float = 0.0
    center_phase_err: float = 0.0

    # ── Quality / confidence ──────────────────────────────────────────────
    snr: float = 0.0              # Signal-to-noise ratio
    chi2_reduced: float = 0.0     # Reduced χ²  (goodness of fit)
    confidence_pct: float = 0.0   # Confidence level in %  (from Δχ² test)
    n_points_in_transit: int = 0  # Data points inside transit window

    # ── Derived quantities ────────────────────────────────────────────────
    ingress_duration_min: float = field(init=False)
    depth_fraction: float = field(init=False)

    def __post_init__(self):
        self.ingress_duration_min = self.ingress_fraction * self.duration_hours * 60
        self.depth_fraction = self.depth_ppm / 1e6

    def summary(self) -> str:
        lines = [
            "═══════════  Transit Parameter Estimation  ═══════════",
            f"  Period           :  {self.period_days:.6f}  days",
            f"  Depth            :  {self.depth_ppm:.1f} ± {self.depth_ppm_err:.1f}  ppm",
            f"  Duration (T14)   :  {self.duration_hours:.3f} ± {self.duration_hours_err:.3f}  h",
            f"  Ingress fraction :  {self.ingress_fraction:.3f} ± {self.ingress_fraction_err:.3f}",
            f"  Ingress time     :  {self.ingress_duration_min:.1f}  min",
            f"  Center phase     :  {self.center_phase:.5f} ± {self.center_phase_err:.5f}",
            "───────────────────────────────────────────────────────",
            f"  SNR              :  {self.snr:.2f}",
            f"  Reduced χ²       :  {self.chi2_reduced:.3f}",
            f"  Confidence       :  {self.confidence_pct:.2f} %",
            f"  Points in transit:  {self.n_points_in_transit}",
            "═══════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Parameter Estimator
# ─────────────────────────────────────────────────────────────────────────────

class TransitParameterEstimator:
    """
    Fits a trapezoidal transit model to a phase-folded light curve.

    Usage
    -----
    estimator = TransitParameterEstimator(period_days=3.14)
    params    = estimator.fit(phase, flux, flux_err)
    print(params.summary())
    """

    def __init__(
        self,
        period_days: float,
        phase_window: float = 0.15,   # Fraction of period to search for transit
    ):
        """
        Parameters
        ----------
        period_days  : Orbital period in days  (BLS output from M2).
        phase_window : ± fraction of full phase to consider as the transit
                       search region  (default 15 % on each side of center).
        """
        self.period_days = period_days
        self.phase_window = phase_window

    # ── Public entry point ────────────────────────────────────────────────

    def fit(
        self,
        phase: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
    ) -> TransitParams:
        """
        Estimate transit parameters from a phase-folded light curve.

        Parameters
        ----------
        phase    : Phase array in days, folded around the transit (center ≈ 0).
        flux     : Normalized flux  (median ≈ 1.0  out of transit).
        flux_err : Per-point flux uncertainty. If None, estimated from scatter.

        Returns
        -------
        TransitParams  dataclass with all fitted values and quality metrics.
        """
        phase = np.asarray(phase, dtype=float)
        flux  = np.asarray(flux,  dtype=float)

        if flux_err is None:
            flux_err = self._estimate_noise(phase, flux)
        flux_err = np.asarray(flux_err, dtype=float)
        flux_err = np.where(flux_err <= 0, np.median(flux_err[flux_err > 0]), flux_err)

        # ── 1. Coarse initial guess from the data ─────────────────────────
        p0, bounds = self._initial_guess(phase, flux)

        # ── 2. Least-squares fit  ─────────────────────────────────────────
        popt, pcov = self._run_curve_fit(phase, flux, flux_err, p0, bounds)

        depth_fit, dur_fit, ing_fit, cen_fit = popt

        # ── 3. Uncertainties from covariance matrix  ──────────────────────
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(4)

        # ── 4. Goodness-of-fit metrics ────────────────────────────────────
        flux_model   = trapezoidal_transit(phase, *popt)
        residuals    = flux - flux_model
        chi2_val     = np.sum((residuals / flux_err) ** 2)
        dof          = max(len(flux) - 4, 1)
        chi2_red     = chi2_val / dof

        # Δχ² confidence: compare fitted model vs flat (no transit) model
        flux_flat   = np.ones_like(flux)
        chi2_flat   = np.sum(((flux - flux_flat) / flux_err) ** 2)
        delta_chi2  = chi2_flat - chi2_val
        conf_pct    = chi2.cdf(delta_chi2, df=4) * 100   # 4 free params

        # ── 5. SNR ────────────────────────────────────────────────────────
        snr = self._compute_snr(phase, flux, flux_err, popt)

        # ── 6. Unit conversions ───────────────────────────────────────────
        depth_ppm      = depth_fit  * 1e6
        depth_ppm_err  = perr[0]    * 1e6
        dur_h          = dur_fit    * 24.0            # days → hours
        dur_h_err      = perr[1]    * 24.0

        # Points inside the transit window
        half_dur = dur_fit / 2.0
        in_transit = np.abs(phase - cen_fit) <= half_dur
        n_pts = int(np.sum(in_transit))

        return TransitParams(
            period_days          = self.period_days,
            depth_ppm            = depth_ppm,
            duration_hours       = dur_h,
            ingress_fraction     = ing_fit,
            center_phase         = cen_fit,
            depth_ppm_err        = depth_ppm_err,
            duration_hours_err   = dur_h_err,
            ingress_fraction_err = perr[2],
            center_phase_err     = perr[3],
            snr                  = snr,
            chi2_reduced         = chi2_red,
            confidence_pct       = conf_pct,
            n_points_in_transit  = n_pts,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _estimate_noise(
        self, phase: np.ndarray, flux: np.ndarray
    ) -> np.ndarray:
        """Estimate per-point noise from out-of-transit scatter (MAD-based)."""
        oot = np.abs(phase) > self.phase_window * self.period_days
        if oot.sum() < 10:
            oot = np.ones(len(phase), dtype=bool)
        mad  = np.median(np.abs(flux[oot] - np.median(flux[oot])))
        sigma = mad * 1.4826  # convert MAD → σ
        return np.full(len(flux), max(sigma, 1e-6))

    def _initial_guess(
        self,
        phase: np.ndarray,
        flux: np.ndarray,
    ) -> Tuple[list, dict]:
        """
        Coarse initial parameter guess from the raw folded light curve.
        """
        # Depth: difference between median flux and minimum
        median_flux = np.median(flux)
        min_flux    = np.min(flux)
        depth0      = max(median_flux - min_flux, 1e-5)

        # Transit center: phase of minimum flux (smoothed)
        cen0 = phase[np.argmin(flux)]

        # Duration: width of region below  median − 0.5 * depth
        threshold    = median_flux - 0.5 * depth0
        below        = phase[flux < threshold]
        if len(below) >= 2:
            dur0 = float(below.max() - below.min())
        else:
            dur0 = 0.05 * self.period_days   # fallback: 5 % of period

        dur0 = np.clip(dur0, 0.005 * self.period_days, 0.4 * self.period_days)
        ing0 = 0.15    # 15 % of duration for ingress/egress

        p0 = [depth0, dur0, ing0, cen0]

        bounds = (
            [1e-6,  0.002 * self.period_days, 0.02,  phase.min()],   # lower
            [0.5,   0.45  * self.period_days, 0.49,  phase.max()],   # upper
        )
        return p0, bounds

    def _run_curve_fit(
        self,
        phase: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        p0: list,
        bounds: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Wrapper around scipy curve_fit with fallback."""
        try:
            popt, pcov = curve_fit(
                trapezoidal_transit,
                phase, flux,
                p0      = p0,
                sigma   = flux_err,
                bounds  = bounds,
                maxfev  = 10_000,
                method  = "trf",
            )
            return popt, pcov
        except RuntimeError:
            # Fallback: Nelder-Mead minimisation (no covariance)
            def objective(params):
                try:
                    model = trapezoidal_transit(phase, *params)
                    return np.sum(((flux - model) / flux_err) ** 2)
                except Exception:
                    return 1e12

            result = minimize(
                objective, p0,
                method  = "Nelder-Mead",
                options = {"maxiter": 50_000, "xatol": 1e-7, "fatol": 1e-7},
            )
            return result.x, None

    def _compute_snr(
        self,
        phase: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        popt: np.ndarray,
    ) -> float:
        """
        SNR = (mean in-transit depth) / (RMS out-of-transit noise / √N_in)
        """
        depth, dur, ing, cen = popt
        half = dur / 2.0

        in_transit  = np.abs(phase - cen) <= half
        out_transit = np.abs(phase - cen) >  half * 1.5   # safe OOT region

        if in_transit.sum() < 2 or out_transit.sum() < 2:
            return 0.0

        mean_dip = np.median(1.0 - flux[in_transit])         # measured depth
        oot_rms  = np.std(flux[out_transit])                  # OOT noise
        n_in     = in_transit.sum()

        if oot_rms <= 0:
            return 0.0

        return float(mean_dip / (oot_rms / np.sqrt(n_in)))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Convenience wrapper  (called by run_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_transit_parameters(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray],
    period_days: float,
    phase_window: float = 0.15,
) -> TransitParams:
    """
    Top-level function for the pipeline.

    Parameters
    ----------
    phase        : Phase-folded time array  (days, transit near 0).
    flux         : Normalised flux  (OOT ≈ 1.0).
    flux_err     : Flux uncertainties  (or None to auto-estimate).
    period_days  : Orbital period from BLS  (M2 output).
    phase_window : Search window as fraction of period.

    Returns
    -------
    TransitParams
    """
    estimator = TransitParameterEstimator(
        period_days  = period_days,
        phase_window = phase_window,
    )
    return estimator.fit(phase, flux, flux_err)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Quick self-test  (python parameter_estimation.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Running self-test with synthetic transit …\n")

    rng = np.random.default_rng(42)

    # ── Synthetic ground-truth parameters ────────────────────────────────
    TRUE_PERIOD   = 3.52          # days
    TRUE_DEPTH    = 0.012         # ~12 000 ppm  (Jupiter-like around small star)
    TRUE_DURATION = 0.12          # days  (~2.9 h)
    TRUE_INGRESS  = 0.18          # 18 % of duration
    TRUE_CENTER   = 0.003         # slight phase offset

    N = 800
    phase = np.linspace(-0.25, 0.25, N)
    flux_clean = trapezoidal_transit(
        phase, TRUE_DEPTH, TRUE_DURATION, TRUE_INGRESS, TRUE_CENTER
    )
    noise_level = 5e-4
    flux = flux_clean + rng.normal(0, noise_level, N)
    flux_err = np.full(N, noise_level)

    # ── Run estimator ─────────────────────────────────────────────────────
    params = estimate_transit_parameters(
        phase       = phase,
        flux        = flux,
        flux_err    = flux_err,
        period_days = TRUE_PERIOD,
    )
    print(params.summary())

    # ── Compare to truth ──────────────────────────────────────────────────
    print("\nComparison to ground truth:")
    print(f"  Depth     :  est {params.depth_ppm:7.1f}  ppm   |  true {TRUE_DEPTH*1e6:7.1f}  ppm")
    print(f"  Duration  :  est {params.duration_hours:.3f} h  |  true {TRUE_DURATION*24:.3f} h")
    print(f"  Ingress   :  est {params.ingress_fraction:.3f}     |  true {TRUE_INGRESS:.3f}")
    print(f"  Center    :  est {params.center_phase:.5f}  |  true {TRUE_CENTER:.5f}")

    # ── Quick plot ────────────────────────────────────────────────────────
    model_fit = trapezoidal_transit(
        phase,
        params.depth_fraction,
        params.duration_hours / 24.0,
        params.ingress_fraction,
        params.center_phase,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(phase * 24, flux,        ".", color="#aaaaaa", ms=3, label="Data")
    ax.plot(phase * 24, flux_clean,  "-", color="#4fc3f7", lw=1.5, label="True model")
    ax.plot(phase * 24, model_fit,   "-", color="#ff7043", lw=2,   label="Fitted model")
    ax.set_xlabel("Phase  (hours)")
    ax.set_ylabel("Normalised flux")
    ax.set_title(f"Transit Fit  —  SNR {params.snr:.1f}  |  Confidence {params.confidence_pct:.1f} %")
    ax.legend()
    plt.tight_layout()
    plt.savefig("parameter_estimation_test.png", dpi=150)
    print("\nPlot saved → parameter_estimation_test.png")
