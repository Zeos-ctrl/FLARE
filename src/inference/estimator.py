"""
Parameter estimation for GWOSC events using the FLARE surrogate.

The surrogate generates candidate waveforms ~100x faster than PyCBC, which
makes it attractive as the template engine inside an MCMC over the intrinsic
binary parameters. This module provides:

* :class:`Prior` - bounded uniform prior over a chosen subset of parameters,
  with the remaining parameters held fixed.
* :class:`WaveformBank` - a pre-generated bank of surrogate templates used to
  warm-start the sampler (and reused by the dashboard for dataset previews).
* :class:`GWEventEstimator` - a matched-filter likelihood over real detector
  data plus an :mod:`emcee` driver.
* :class:`PosteriorResult` - posterior samples with summary statistics and a
  corner-plot helper.

The likelihood uses the matched-filter SNR maximised over coalescence time and
phase (the standard detection statistic), so distance, phase and arrival time
are handled analytically rather than sampled. For multiple detectors the
per-detector maximised SNR^2 values are summed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.inference.gwosc import EventData, fetch_event_data
from src.inference.predictor import WaveformPredictor

logger = logging.getLogger(__name__)

# Canonical order of the surrogate's parameter vector.
PARAM_ORDER = ["m1", "m2", "s1z", "s2z", "inc", "ecc"]
PARAM_INDEX = {name: i for i, name in enumerate(PARAM_ORDER)}

# ---------------------------------------------------------------------------
# Nested-sampling pool workers.
#
# dynesty parallelises by pickling the likelihood/prior-transform and mapping
# them across a process pool. We can't pickle a CUDA torch model or a bound
# closure, so instead the pool is created with the FORK start method and the
# estimator + transform are stashed in this module global: forked workers
# inherit it via copy-on-write, and only these top-level functions (picklable by
# reference) and the plain-array arguments cross the queue. This is why the
# pycbc reference -- CPU-bound SEOBNRv4 generation, embarrassingly parallel --
# pools cleanly across cores, while the GPU surrogate stays single-process.
# Not re-entrant: one pooled run at a time (the benchmark runs engines serially).
# ---------------------------------------------------------------------------
_POOL_STATE: Dict[str, object] = {}


def _pool_loglike(theta):
    est = _POOL_STATE["est"]
    kind = _POOL_STATE["kind"]
    if kind == "coherent":
        return est.coherent_log_posterior(theta)
    if kind == "gaussian":
        return est.gaussian_log_posterior(theta)
    return est.log_likelihood(theta)


def _pool_ptform(u):
    return _POOL_STATE["ptform"](u)


@dataclass
class Prior:
    """Uniform prior over a subset of parameters; others held fixed.

    Args:
        bounds: Mapping ``param_name -> (low, high)`` for the free parameters.
        fixed: Mapping ``param_name -> value`` for parameters held constant.
    """
    bounds: Dict[str, Tuple[float, float]]
    fixed: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.free_names: List[str] = list(self.bounds.keys())
        self.ndim = len(self.free_names)
        for name in self.free_names + list(self.fixed):
            if name not in PARAM_INDEX:
                raise ValueError(f"Unknown parameter '{name}'")

    @classmethod
    def default(cls) -> "Prior":
        """Broad prior over masses and aligned spins; inc/ecc fixed to zero."""
        return cls(
            bounds={
                "m1": (20.0, 100.0),
                "m2": (20.0, 100.0),
                "s1z": (-0.9, 0.9),
                "s2z": (-0.9, 0.9),
            },
            fixed={"inc": 0.0, "ecc": 0.0},
        )

    @property
    def low(self) -> np.ndarray:
        return np.array([self.bounds[n][0] for n in self.free_names])

    @property
    def high(self) -> np.ndarray:
        return np.array([self.bounds[n][1] for n in self.free_names])

    def log_prior(self, theta: np.ndarray) -> float:
        """Uniform log-prior; ``-inf`` outside the box or if m2 > m1."""
        if np.any(theta < self.low) or np.any(theta > self.high):
            return -np.inf
        full = self.to_full(theta)
        if full[PARAM_INDEX["m2"]] > full[PARAM_INDEX["m1"]]:
            return -np.inf  # enforce m1 >= m2 convention
        return 0.0

    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Draw ``n`` samples of the free parameters from the prior box."""
        rng = np.random.default_rng(seed)
        return rng.uniform(self.low, self.high, size=(n, self.ndim))

    def to_full(self, theta: np.ndarray) -> np.ndarray:
        """Expand a free-parameter vector into the full 6-vector."""
        full = np.zeros(len(PARAM_ORDER))
        for name, val in self.fixed.items():
            full[PARAM_INDEX[name]] = val
        for i, name in enumerate(self.free_names):
            full[PARAM_INDEX[name]] = theta[i]
        return full

    def to_full_batch(self, thetas: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`to_full` for an array of shape (N, ndim)."""
        n = len(thetas)
        full = np.zeros((n, len(PARAM_ORDER)))
        for name, val in self.fixed.items():
            full[:, PARAM_INDEX[name]] = val
        for i, name in enumerate(self.free_names):
            full[:, PARAM_INDEX[name]] = thetas[:, i]
        return full


@dataclass
class WaveformBankResult:
    """Templates and parameters produced by :meth:`WaveformBank.build`."""
    parameters: np.ndarray            # (N, ndim) free parameters
    full_parameters: np.ndarray       # (N, 6) full parameter vectors
    templates: np.ndarray             # (N, waveform_length) h_plus templates
    prior: Prior


class WaveformBank:
    """Pre-generate a bank of surrogate templates over a prior.

    Used to (a) warm-start MCMC walkers near the likelihood peak via a coarse
    matched-filter search, and (b) provide a fast, inspectable set of waveforms
    for the dashboard's dataset explorer.
    """

    def __init__(self, predictor: WaveformPredictor, prior: Optional[Prior] = None):
        self.predictor = predictor
        self.prior = prior or Prior.default()
        self.result: Optional[WaveformBankResult] = None

    def build(self, n_samples: int = 2000, batch_size: int = 256,
              seed: Optional[int] = None) -> WaveformBankResult:
        """Sample the prior and generate ``n_samples`` surrogate templates."""
        thetas = self.prior.sample(n_samples, seed=seed)
        full = self.prior.to_full_batch(thetas)

        logger.info("Building waveform bank: %d templates...", n_samples)
        h_plus_list, _ = self.predictor.batch_predict(full, batch_size=batch_size)
        templates = np.stack([hp.data for hp in h_plus_list])

        self.result = WaveformBankResult(
            parameters=thetas, full_parameters=full,
            templates=templates, prior=self.prior,
        )
        return self.result

    def best_matches(self, log_likelihood: Callable[[np.ndarray], float],
                     top_k: int = 32) -> np.ndarray:
        """Return the ``top_k`` bank parameters ranked by a likelihood callable."""
        if self.result is None:
            raise RuntimeError("Call build() before best_matches()")
        scores = np.array([log_likelihood(theta) for theta in self.result.parameters])
        order = np.argsort(scores)[::-1]
        return self.result.parameters[order[:top_k]]


@dataclass
class PosteriorResult:
    """Posterior samples and summary statistics from an MCMC run."""
    samples: np.ndarray               # (N, ndim) flattened post-burn samples
    param_names: List[str]
    log_prob: np.ndarray
    prior: Prior
    acceptance_fraction: float
    event_name: str = "unknown"
    sampler: str = "emcee"            # "emcee" or "dynesty"
    log_evidence: Optional[float] = None      # ln Z, nested sampling only
    log_evidence_err: Optional[float] = None  # ln Z uncertainty

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Median and 90% credible interval per parameter."""
        out = {}
        for i, name in enumerate(self.param_names):
            col = self.samples[:, i]
            lo, med, hi = np.percentile(col, [5, 50, 95])
            out[name] = {
                "median": float(med),
                "lower_90": float(lo),
                "upper_90": float(hi),
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
            }
        return out

    def to_dict(self) -> dict:
        d = {
            "event_name": self.event_name,
            "param_names": self.param_names,
            "n_samples": int(len(self.samples)),
            "acceptance_fraction": float(self.acceptance_fraction),
            "sampler": self.sampler,
            "summary": self.summary(),
        }
        if self.log_evidence is not None:
            d["log_evidence"] = float(self.log_evidence)
            d["log_evidence_err"] = float(self.log_evidence_err or 0.0)
        return d

    def corner_plot(self, path: Optional[str] = None, truths: Optional[Sequence] = None):
        """Render a corner plot of the posterior (requires ``corner``)."""
        import corner
        fig = corner.corner(
            self.samples, labels=self.param_names, truths=truths,
            quantiles=[0.05, 0.5, 0.95], show_titles=True,
        )
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig


class GWEventEstimator:
    """Estimate intrinsic binary parameters for a GWOSC event via MCMC.

    Args:
        predictor: A trained :class:`WaveformPredictor`.
        event: Either a GWOSC event name (fetched automatically) or a
            pre-fetched :class:`EventData` object.
        detectors: Detectors to use when fetching by name.
        prior: Parameter prior; defaults to :meth:`Prior.default`.
        duration, sample_rate: Data-conditioning settings when fetching.
    """

    def __init__(
        self,
        predictor: WaveformPredictor,
        event,
        detectors: Sequence[str] = ("H1", "L1"),
        prior: Optional[Prior] = None,
        duration: float = 32.0,
        sample_rate: Optional[float] = None,
        physical: bool = False,
        duration_model=None,
        likelihood: str = "matched_filter",
        offsource_psd: bool = False,
        distance_bounds: Tuple[float, float] = (50.0, 3000.0),
        snr_window_s: float = 0.1,
    ):
        self.predictor = predictor
        self.prior = prior or Prior.default()
        self.f_lower = 20.0
        # Likelihood: "matched_filter" = 0.5*sum_det max_t|rho|^2 (the detection
        # statistic; maximises distance implicitly). "gaussian" = the proper
        # Gaussian log-likelihood ``sum_det [alpha*rho*sigma - 0.5*alpha^2*sigma^2]``
        # with an explicit sampled ``distance`` (alpha = 1/distance); the
        # -0.5*sigma^2 term penalises loud high-mass templates, removing the
        # merger-dominance bias that skews mass recovery for short, loud signals.
        self.likelihood_kind = likelihood
        self.distance_bounds = distance_bounds
        # Fixed-window (physical-time) models reconstruct a matched-filterable
        # waveform directly; read the resize convention from the checkpoint.
        meta = getattr(predictor, "meta", {}) or {}
        self.td_fixed_window = bool(meta.get("td_fixed_window", False))
        self.merger_frac = float(meta.get("merger_frac", 0.9))
        # ``physical``: reconstruct physical-time templates (invert the surrogate's
        # merger-aligned time warp) instead of feeding the normalised-grid waveform
        # straight to the matched filter. Required for meaningful PE against real
        # data -- the normalised template has no absolute chirp timescale.
        self.physical = physical
        self.duration_model = duration_model
        if physical and duration_model is None:
            from src.inference.physical_time import DurationModel
            self.duration_model = DurationModel.load()
        # Templates must share the surrogate's sample rate.
        self.sample_rate = sample_rate or (1.0 / predictor.delta_t)
        self.delta_t = 1.0 / self.sample_rate

        if isinstance(event, EventData):
            self.event = event
        else:
            self.event = fetch_event_data(
                event, detectors=list(detectors), duration=duration,
                sample_rate=self.sample_rate, f_lower=self.f_lower,
            )
        self.event_name = self.event.event_name
        self._data_len = len(next(iter(self.event.detectors.values())).strain)
        # Data is centred on the event GPS time, so the merger is at the segment
        # centre; the windowed matched filter maximises tc only near there.
        self._center = self._data_len // 2
        self.snr_window = max(1, int(snr_window_s * self.sample_rate))
        # Seconds trimmed from each end of the matched-filter SNR series before
        # peak-finding: the inverse-spectrum-truncated PSD corrupts ~psd_seconds
        # at each edge, so discard generously (front > back for the whitening
        # filter's acausal wing).
        self.snr_crop_front = 8.0
        self.snr_crop_back = 4.0
        if offsource_psd:
            self._reestimate_psd_offsource()

    def _reestimate_psd_offsource(self, offset: float = 4.0, length: float = 64.0):
        """Replace each detector PSD with one estimated from OFF-SOURCE data
        (``length`` s ending ``offset`` s before the event). The on-source Welch
        PSD is biased by the loud signal itself, which distorts the matched filter
        and shifts the mass posterior; off-source removes that."""
        from gwpy.timeseries import TimeSeries as GW
        from pycbc.psd import welch, interpolate, inverse_spectrum_truncation
        from pycbc.types import TimeSeries as PT
        gps = self.event.gps_time
        for name, d in self.event.detectors.items():
            ts = GW.fetch_open_data(name, gps - offset - length, gps - offset,
                                    sample_rate=int(self.sample_rate),
                                    cache=True).highpass(self.f_lower)
            seg = min(int(4 * self.sample_rate), len(ts.value))
            p = welch(PT(ts.value.astype(np.float64), delta_t=self.delta_t),
                      seg_len=seg, seg_stride=seg // 2)
            p = interpolate(p, d.strain.delta_f)
            # Inverse-spectrum-truncate so the whitening filter is finite length
            # (otherwise wrap-around leakage inflates the matched-filter SNR).
            d.psd = inverse_spectrum_truncation(
                p, seg, low_frequency_cutoff=self.f_lower, trunc_method="hann")

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------
    def _template(self, full_params: np.ndarray):
        """Surrogate h_plus as a zero-padded PyCBC TimeSeries matching the data."""
        from pycbc.types import TimeSeries as PyCBCTimeSeries

        if self.physical:
            h = self.predictor.predict_physical(
                m1=full_params[0], m2=full_params[1],
                duration_model=self.duration_model)
            # keep the merger + most recent inspiral that fits the data window.
            if len(h) > self._data_len:
                h = h[-self._data_len:]
            ts = PyCBCTimeSeries(np.asarray(h, dtype=np.float64), delta_t=self.delta_t)
            ts.resize(self._data_len)
            return ts

        hp, _ = self.predictor.predict(
            m1=full_params[0], m2=full_params[1],
            s1z=full_params[2], s2z=full_params[3],
            inc=full_params[4], ecc=full_params[5],
        )
        # Fixed-window models reconstruct a physical-time waveform: place it the
        # standard pycbc way so the matched filter is calibrated.
        if self.td_fixed_window:
            h = np.asarray(hp.data, dtype=np.float64)
            return self._place_template(h, int(self.merger_frac * len(h)))
        ts = PyCBCTimeSeries(hp.data.astype(np.float64), delta_t=self.delta_t)
        ts.resize(self._data_len)
        return ts

    def log_likelihood(self, theta: np.ndarray) -> float:
        """Network log-likelihood = 0.5 * sum_det max_t |rho(t)|^2.

        ``rho`` is the complex matched-filter SNR (maximised over phase), and
        the maximum over the time series maximises over coalescence time.
        """
        from pycbc.filter import matched_filter

        full = self.prior.to_full(theta)
        try:
            template = self._template(full)
        except Exception as exc:  # surrogate/pycbc failure on bad params
            logger.debug("Template generation failed for %s: %s", full, exc)
            return -np.inf

        loglike = 0.0
        for det in self.event.detectors.values():
            snr = matched_filter(
                template, det.strain, psd=det.psd,
                low_frequency_cutoff=self.f_lower,
            )
            if self.td_fixed_window:
                # Crop the IST-corrupted edges, then peak within snr_window of the
                # known merger time (rejects distant glitches).
                snr = snr.crop(self.snr_crop_front, self.snr_crop_back)
                a = np.abs(snr.numpy())
                ctr = int(round((self.event.gps_time - float(snr.start_time)) / self.delta_t))
                lo, hi = max(0, ctr - self.snr_window), min(len(a), ctr + self.snr_window)
                rho2 = np.max(a[lo:hi] ** 2) if hi > lo else np.max(a ** 2)
            else:
                # Trim filter wrap-around edges before taking the peak.
                edge = len(snr) // 8
                rho2 = np.max(np.abs(snr.numpy()[edge:-edge]) ** 2)
            loglike += 0.5 * float(rho2)
        return loglike

    def log_posterior(self, theta: np.ndarray) -> float:
        lp = self.prior.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    # ------------------------------------------------------------------
    # Proper Gaussian likelihood (fixed-window physical-time templates)
    # ------------------------------------------------------------------
    def _place_template(self, h: np.ndarray, merger_sample: int):
        """Turn a fixed-window waveform array into a matched-filterable template
        the standard pycbc way: put the merger at ``t = 0`` (negative epoch),
        resize to the data length, then ``cyclic_time_shift`` so the start sits at
        sample 0. This mirrors ``get_td_waveform`` + ``resize`` + ``cyclic_time_
        shift`` exactly; a manual zero-pad embed instead produces a spurious,
        time-offset SNR peak with inflated amplitude, so do NOT hand-embed."""
        from pycbc.types import TimeSeries as PyCBCTimeSeries
        ts = PyCBCTimeSeries(np.asarray(h, dtype=np.float64), delta_t=self.delta_t,
                             epoch=-merger_sample * self.delta_t)
        ts.resize(self._data_len)
        return ts.cyclic_time_shift(ts.start_time)

    def _template_centered(self, m1: float, m2: float):
        """Matched-filterable physical-time template for a fixed-window model."""
        hp, _ = self.predictor.predict(m1=m1, m2=m2)
        h = np.asarray(hp.data, dtype=np.float64)
        return self._place_template(h, int(self.merger_frac * len(h)))

    def _matched_quantities(self, m1: float, m2: float):
        """Per-detector ``(rho_peak, sigma)``: matched-filter SNR (phase-max, time
        maxed within ``snr_window`` of the event GPS) and template norm
        ``sqrt(<h|h>)``. The SNR series is cropped first to drop the inverse-
        spectrum-truncation edge corruption."""
        from pycbc.filter import matched_filter, sigma as pycbc_sigma
        tmpl = self._template_centered(m1, m2)
        gps, w = self.event.gps_time, self.snr_window
        out = []
        for d in self.event.detectors.values():
            snr = matched_filter(tmpl, d.strain, psd=d.psd,
                                 low_frequency_cutoff=self.f_lower)
            snr = snr.crop(self.snr_crop_front, self.snr_crop_back)
            a = np.abs(snr.numpy())
            # peak within +/- snr_window of the known merger time
            ctr = int(round((gps - float(snr.start_time)) / self.delta_t))
            lo, hi = max(0, ctr - w), min(len(a), ctr + w)
            rho = float(np.max(a[lo:hi])) if hi > lo else float(np.max(a))
            sig = float(pycbc_sigma(tmpl, psd=d.psd, low_frequency_cutoff=self.f_lower))
            out.append((rho, sig))
        return out

    def gaussian_log_posterior(self, theta_full: np.ndarray) -> float:
        """Posterior over ``[free intrinsic..., distance]`` using the proper
        Gaussian likelihood. ``h_ref`` is the surrogate template at its training
        reference distance (~1 Mpc); a candidate at ``distance`` scales it by
        ``alpha = 1/distance``. logL = sum_det alpha*rho*sigma - 0.5*alpha^2*sigma^2."""
        nd = self.prior.ndim
        theta, distance = theta_full[:nd], float(theta_full[nd])
        lp = self.prior.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        dlo, dhi = self.distance_bounds
        if not (dlo <= distance <= dhi):
            return -np.inf
        full = self.prior.to_full(theta)
        try:
            quants = self._matched_quantities(full[0], full[1])
        except Exception as exc:  # pragma: no cover
            logger.debug("template failed for %s: %s", full, exc)
            return -np.inf
        alpha = 1.0 / distance
        logl = sum(alpha * rho * sig - 0.5 * alpha * alpha * sig * sig
                   for rho, sig in quants)
        return lp + logl

    # ------------------------------------------------------------------
    # Coherent antenna-response likelihood (both polarizations + sky)
    # ------------------------------------------------------------------
    def _place_polarizations(self, full: np.ndarray):
        """Place BOTH surrogate polarizations as matched-filterable templates.

        The surrogate returns (h_plus, h_cross) at the sampled inclination (the
        higher-mode content makes h_cross carry real information); each is placed
        the standard pycbc way (see :meth:`_place_template`)."""
        hp_p, hc_p = self.predictor.predict(
            m1=full[0], m2=full[1], s1z=full[2], s2z=full[3], inc=full[4])
        n = len(hp_p.data)
        ms = int(self.merger_frac * n)
        hp = self._place_template(np.asarray(hp_p.data, dtype=np.float64), ms)
        hc = self._place_template(np.asarray(hc_p.data, dtype=np.float64), ms)
        return hp, hc

    def coherent_log_posterior(self, theta_full: np.ndarray) -> float:
        """Coherent multi-detector Gaussian log-posterior with antenna response.

        Samples ``[free intrinsic (incl. inclination)..., distance, ra, dec, psi]``.
        Each detector sees ``h_d = (1/D)[F+·h_plus + F×·h_cross]`` delayed by the
        geometric time-of-flight, with F+, F× the antenna patterns. The coalescence
        phase is analytically MARGINALISED (Bessel-I0), which is the standard proper
        treatment; the coalescence time is taken at the SNR peak near the trigger.

        Unlike the single-polarization matched-filter likelihood, this constrains
        distance and inclination (they stop being a pure amplitude degeneracy) and
        localises the source on the sky -- the payoff of the higher-mode h_cross and
        a detector network.

            logL = log I0(|Z|) + |Z| - 0.5 <h|h>,
            Z    = Σ_d (1/D)[F+_d <d|h+>_d + F×_d <d|h×>_d]  (phase-coherent sum)
            <h|h>= Σ_d (1/D)^2 [F+^2 σ_pp + F×^2 σ_xx + 2 F+ F× σ_px]
        """
        from pycbc.detector import Detector
        from pycbc.filter import matched_filter, overlap_cplx, sigma as pycbc_sigma
        from scipy.special import i0e

        nd = self.prior.ndim
        theta = theta_full[:nd]
        fixed_sky = getattr(self, "fixed_sky", None)
        if fixed_sky is not None:
            # sky held fixed (localised independently); sample only distance
            distance = float(theta_full[nd])
            ra, dec, psi = (float(x) for x in fixed_sky)
        else:
            distance, ra, dec, psi = (float(x) for x in theta_full[nd:nd + 4])
        lp = self.prior.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        dlo, dhi = self.distance_bounds
        if not (dlo <= distance <= dhi and 0.0 <= ra <= 2 * np.pi
                and -np.pi / 2 <= dec <= np.pi / 2 and 0.0 <= psi <= np.pi):
            return -np.inf
        full = self.prior.to_full(theta)
        try:
            hp, hc = self._place_polarizations(full)
        except Exception as exc:  # pragma: no cover
            logger.debug("polarization template failed for %s: %s", full, exc)
            return -np.inf

        gps = self.event.gps_time
        w = self.snr_window
        # Distance scaling: the surrogate templates are at the model's reference
        # distance, so a candidate at ``distance`` scales by ref/distance (NOT
        # 1/distance -- that dropped a ~100x factor and killed the <h|h> penalty).
        ref = float(getattr(self.predictor, "ref_distance_mpc", 1.0))
        alpha = ref / distance

        # Per detector: complex <d|h+>,<d|h×> windows about the candidate arrival
        # time gps+Δt, and the template norms. The coalescence time is maximised
        # COHERENTLY over a SINGLE geocenter shift shared by all detectors (the
        # sky enters through the per-detector delay Δt, so a wrong sky evaluates
        # off-peak -> lower |Z| -> the network localises the source).
        zp_w, zx_w, hh = [], [], 0.0
        wlen = None
        for name, d in self.event.detectors.items():
            det = Detector(name)
            fp, fc = det.antenna_pattern(ra, dec, psi, gps)
            dt = det.time_delay_from_earth_center(ra, dec, gps)
            sp = matched_filter(hp, d.strain, psd=d.psd, low_frequency_cutoff=self.f_lower)
            sx = matched_filter(hc, d.strain, psd=d.psd, low_frequency_cutoff=self.f_lower)
            sp = sp.crop(self.snr_crop_front, self.snr_crop_back)
            sx = sx.crop(self.snr_crop_front, self.snr_crop_back)
            sig_p = float(pycbc_sigma(hp, psd=d.psd, low_frequency_cutoff=self.f_lower))
            sig_x = float(pycbc_sigma(hc, psd=d.psd, low_frequency_cutoff=self.f_lower))
            ctr = int(round((gps + dt - float(sp.start_time)) / self.delta_t))
            lo, hi = ctr - w, ctr + w
            ap, ax = sp.numpy(), sx.numpy()
            if lo < 0 or hi > len(ap):
                return -np.inf
            zp_w.append(ap[lo:hi] * (alpha * fp * sig_p))
            zx_w.append(ax[lo:hi] * (alpha * fc * sig_x))
            sig_px = float(np.real(overlap_cplx(
                hp, hc, psd=d.psd, low_frequency_cutoff=self.f_lower, normalized=False)))
            hh += alpha * alpha * (fp * fp * sig_p * sig_p + fc * fc * sig_x * sig_x
                                   + 2.0 * fp * fc * sig_px)
            wlen = len(ap[lo:hi])
        # coherent sum over the shared time offset; |Z| = max over offset
        Zt = np.zeros(wlen, dtype=complex)
        for a_p, a_x in zip(zp_w, zx_w):
            Zt += a_p + a_x
        absZ = float(np.max(np.abs(Zt)))
        # phase marginalised: log I0(|Z|) = log i0e(|Z|) + |Z|
        logl = float(np.log(i0e(absZ)) + absZ - 0.5 * hh)
        return lp + logl

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def initialize_walkers(self, nwalkers: int, bank: Optional[WaveformBank] = None,
                           seed: Optional[int] = None) -> np.ndarray:
        """Initial walker positions, optionally warm-started from a bank."""
        rng = np.random.default_rng(seed)
        if bank is not None and bank.result is not None:
            seeds = bank.best_matches(self.log_likelihood, top_k=nwalkers)
            jitter = 0.01 * (self.prior.high - self.prior.low)
            p0 = seeds + rng.normal(0.0, jitter, size=seeds.shape)
            return np.clip(p0, self.prior.low, self.prior.high)
        return self.prior.sample(nwalkers, seed=seed)

    def _sanitize_walkers(self, p0: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Resample any walkers that start at zero prior/posterior support."""
        rng = np.random.default_rng(seed)
        p0 = np.clip(np.asarray(p0, dtype=float), self.prior.low, self.prior.high)
        for i in range(len(p0)):
            tries = 0
            while not np.isfinite(self.prior.log_prior(p0[i])) and tries < 100:
                p0[i] = self.prior.sample(1, seed=None)[0]
                tries += 1
        return p0

    def run_mcmc(
        self,
        nwalkers: int = 32,
        nsteps: int = 2000,
        burn_frac: float = 0.3,
        bank: Optional[WaveformBank] = None,
        p0: Optional[np.ndarray] = None,
        progress: bool = True,
        on_step: Optional[Callable[[dict], None]] = None,
        seed: Optional[int] = None,
    ) -> PosteriorResult:
        """Run emcee and return a :class:`PosteriorResult`.

        Args:
            on_step: Optional callback ``fn(dict)`` invoked each step with
                progress info; used by the dashboard to stream chains live.
        """
        import emcee

        gaussian = self.likelihood_kind == "gaussian"
        coherent = self.likelihood_kind == "coherent"
        # gaussian samples an extra distance; coherent samples distance + sky
        # (ra, dec, psi) and uses the antenna-response likelihood.
        fixed_sky = getattr(self, "fixed_sky", None)
        if coherent:
            extra = ["distance"] if fixed_sky is not None else ["distance", "ra", "dec", "psi"]
        elif gaussian:
            extra = ["distance"]
        else:
            extra = []
        ndim = self.prior.ndim + len(extra)
        logpost = (self.coherent_log_posterior if coherent
                   else self.gaussian_log_posterior if gaussian else self.log_posterior)
        if p0 is None:
            p0 = self.initialize_walkers(nwalkers, bank=bank, seed=seed)
            rng = np.random.default_rng(seed)
            cols = []
            if coherent or gaussian:
                cols.append(rng.uniform(*self.distance_bounds, size=(nwalkers, 1)))
            if coherent and fixed_sky is None:
                cols.append(rng.uniform(0.0, 2 * np.pi, size=(nwalkers, 1)))     # ra
                cols.append(np.arcsin(rng.uniform(-1.0, 1.0, size=(nwalkers, 1))))  # dec (uniform on sphere)
                cols.append(rng.uniform(0.0, np.pi, size=(nwalkers, 1)))         # psi
            if cols:
                p0 = np.concatenate([p0] + cols, axis=1)
        if not gaussian and not coherent:
            p0 = self._sanitize_walkers(p0, seed=seed)

        # A DE move mix is more robust than the default stretch move on the
        # sharply-peaked likelihoods typical of gravitational-wave data.
        moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, moves=moves)

        logger.info("Running MCMC: %d walkers x %d steps (ndim=%d, %s)",
                    nwalkers, nsteps, ndim, self.likelihood_kind)
        for step, _ in enumerate(sampler.sample(p0, iterations=nsteps, progress=progress)):
            if on_step is not None and (step % 10 == 0 or step == nsteps - 1):
                on_step({
                    "step": step,
                    "total_steps": nsteps,
                    "mean_acceptance": float(np.mean(sampler.acceptance_fraction)),
                })

        burn = int(burn_frac * nsteps)
        samples = sampler.get_chain(discard=burn, flat=True)
        log_prob = sampler.get_log_prob(discard=burn, flat=True)
        names = list(self.prior.free_names) + extra

        return PosteriorResult(
            samples=samples,
            param_names=names,
            log_prob=log_prob,
            prior=self.prior,
            acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
            event_name=self.event_name,
            sampler="emcee",
        )

    def run_nested(
        self,
        nlive: int = 500,
        dlogz: float = 0.1,
        sample: str = "rwalk",
        maxcall: Optional[int] = None,
        progress: bool = True,
        on_step: Optional[Callable[[dict], None]] = None,
        seed: Optional[int] = None,
        pool_processes: Optional[int] = None,
    ) -> PosteriorResult:
        """Estimate the posterior with dynesty nested sampling.

        Nested sampling explores broad, shallow and multimodal likelihoods far
        more reliably than an affine-invariant ensemble -- the regime the short,
        high-mass events fall into, where emcee medians scatter within otherwise-
        overlapping credible intervals. It also returns the Bayesian evidence
        ``ln Z`` (with an uncertainty), which the ensemble sampler does not.

        The same three likelihoods are reused (``matched_filter`` / ``gaussian`` /
        ``coherent``); dynesty needs the prior supplied as a unit-cube transform
        rather than an additive log-prior, so the parameter box, the ``m1 >= m2``
        convention, and the distance / sky priors are encoded in the transform.
        """
        import dynesty

        gaussian = self.likelihood_kind == "gaussian"
        coherent = self.likelihood_kind == "coherent"
        fixed_sky = getattr(self, "fixed_sky", None)
        if coherent:
            extra = ["distance"] if fixed_sky is not None else ["distance", "ra", "dec", "psi"]
        elif gaussian:
            extra = ["distance"]
        else:
            extra = []
        names = list(self.prior.free_names) + extra
        ndim = len(names)

        # Pure log-likelihood (no prior term): inside the transformed region the
        # additive log-prior is 0 and every in-bounds check passes, so the
        # existing ``*_log_posterior`` callables return exactly the log-likelihood.
        loglike = (self.coherent_log_posterior if coherent
                   else self.gaussian_log_posterior if gaussian else self.log_likelihood)

        low, high, span = self.prior.low, self.prior.high, self.prior.high - self.prior.low
        free = list(self.prior.free_names)
        nfree = len(free)
        im1 = free.index("m1") if "m1" in free else None
        im2 = free.index("m2") if "m2" in free else None
        dlo, dhi = self.distance_bounds

        def prior_transform(u):
            x = np.asarray(u, dtype=float).copy()
            # free intrinsic parameters: uniform over the prior box
            x[:nfree] = low + span * u[:nfree]
            # enforce m1 >= m2 (matches Prior.log_prior); mass bounds are equal so
            # the ordered pair stays inside the box
            if im1 is not None and im2 is not None:
                hi_m, lo_m = max(x[im1], x[im2]), min(x[im1], x[im2])
                x[im1], x[im2] = hi_m, lo_m
            j = nfree
            for name in extra:
                if name == "distance":
                    x[j] = dlo + (dhi - dlo) * u[j]        # uniform in distance
                elif name == "ra":
                    x[j] = 2 * np.pi * u[j]
                elif name == "dec":
                    x[j] = np.arcsin(2 * u[j] - 1)         # uniform on the sphere
                elif name == "psi":
                    x[j] = np.pi * u[j]
                j += 1
            return x

        rng = np.random.default_rng(seed)
        # Optional process pool: parallelises the (CPU-bound) likelihood across
        # cores. Fork-inherits the estimator via ``_POOL_STATE`` rather than
        # pickling it (see the module-level note); only used for CPU engines --
        # never fork a live CUDA context, so the GPU surrogate passes 1 here.
        pool = None
        nproc = int(pool_processes or 1)
        if nproc > 1:
            import multiprocessing as mp
            _POOL_STATE["est"] = self
            _POOL_STATE["kind"] = self.likelihood_kind
            _POOL_STATE["ptform"] = prior_transform
            pool = mp.get_context("fork").Pool(nproc)
            sampler = dynesty.NestedSampler(
                _pool_loglike, _pool_ptform, ndim, nlive=nlive, sample=sample,
                rstate=rng, pool=pool, queue_size=nproc)
        else:
            sampler = dynesty.NestedSampler(
                loglike, prior_transform, ndim, nlive=nlive, sample=sample, rstate=rng)

        logger.info("Running nested sampling: nlive=%d dlogz=%.3g (ndim=%d, %s, nproc=%d)",
                    nlive, dlogz, ndim, self.likelihood_kind, nproc)
        try:
            sampler.run_nested(dlogz=dlogz, maxcall=maxcall, print_progress=progress)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
                _POOL_STATE.clear()
        res = sampler.results
        if on_step is not None:
            on_step({"iteration": int(res["niter"]),
                     "logz": float(res["logz"][-1]),
                     "logz_err": float(res["logzerr"][-1])})

        samp = np.asarray(res["samples"])
        logl = np.asarray(res["logl"])
        logz = float(res["logz"][-1])
        # importance weights -> equal-weight posterior (carry logl via the same idx)
        w = np.exp(np.asarray(res["logwt"]) - logz)
        w = np.clip(w, 0.0, None)
        w /= w.sum()
        ntarget = max(4000, len(samp))
        idx = rng.choice(len(samp), size=ntarget, p=w)

        return PosteriorResult(
            samples=samp[idx],
            param_names=names,
            log_prob=logl[idx],
            prior=self.prior,
            acceptance_fraction=float(res["eff"]) / 100.0,  # sampling efficiency (%)->frac
            event_name=self.event_name,
            sampler="dynesty",
            log_evidence=logz,
            log_evidence_err=float(res["logzerr"][-1]),
        )

    # ------------------------------------------------------------------
    # Time-domain reconstruction diagnostic
    # ------------------------------------------------------------------
    def reconstruction_plot(self, result: "PosteriorResult", path: Optional[str] = None,
                            detector: Optional[str] = None,
                            band: Tuple[float, float] = (20.0, 400.0)):
        """Post-PE time-domain diagnostic for fixed-window (physical-time) models.

        Reconstructs the best-fit (MAP) surrogate strain, fits its amplitude,
        phase and arrival time to the data, and produces a 2x2 figure:
          (a) whitened detector strain + surrogate reconstruction,
          (b) the time-domain subtraction residual,
          (c) a Q-scan of the data with the surrogate's instantaneous frequency
              track f(t) overlaid,
          (d) a Q-scan of the residual (the signal should be removed).

        Only meaningful for ``td_fixed_window`` models (the reconstruction must be
        physical-time). Returns the figure, or ``None`` if it cannot be produced.
        """
        if not self.td_fixed_window:
            logger.info("reconstruction_plot skipped: model is not fixed-window TD")
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scipy.signal import hilbert
            from gwpy.timeseries import TimeSeries as GTS
            from pycbc.types import TimeSeries as PT
        except Exception as exc:  # pragma: no cover - optional deps
            logger.warning("reconstruction_plot skipped (missing deps): %s", exc)
            return None

        flow, fhigh = band
        # Best-fit intrinsic parameters at the posterior maximum.
        nd = self.prior.ndim
        theta_map = np.asarray(result.samples[int(np.argmax(result.log_prob))][:nd])
        full = self.prior.to_full(theta_map)
        m1, m2 = float(full[0]), float(full[1])

        dets = self.event.detectors
        if detector is None:  # pick the loudest detector for the clearest chirp
            quants = self._matched_quantities(m1, m2)
            detector = list(dets)[int(np.argmax([r for r, _ in quants]))]
        d = dets[detector]
        DT, SR, DL = self.delta_t, self.sample_rate, self._data_len
        ctr, gps, psd = DL // 2, self.event.gps_time, d.psd
        strain = d.strain.numpy().astype(np.float64)
        start_t = float(d.strain.start_time)

        # Centred surrogate template (merger at segment centre) + 90-deg quadrature.
        hp, _ = self.predictor.predict(m1=m1, m2=m2)
        h = np.asarray(hp.data, dtype=np.float64)
        pk = int(self.merger_frac * len(h))
        h0 = np.zeros(DL); off = ctr - pk
        lo, hi = max(0, off), min(DL, off + len(h)); h0[lo:hi] = h[lo - off:hi - off]
        h90 = np.imag(hilbert(h0))

        def whiten(arr):
            w = PT(arr.astype(np.float64), delta_t=DT).to_frequencyseries() / psd ** 0.5
            f = w.sample_frequencies.numpy(); w.data[(f < flow) | (f > fhigh)] = 0.0
            return w.to_timeseries().numpy()
        dw, h0w, h90w = whiten(strain), whiten(h0), whiten(h90)

        # Fit integer time-shift + amplitude/phase by least squares near the merger
        # (whitening is linear so the coefficients apply to the raw templates too).
        half = int(0.15 * SR); sl = slice(ctr - half, ctr + half); best = None
        for s in range(-int(0.05 * SR), int(0.05 * SR), 2):
            X = np.column_stack([np.roll(h0w, s)[sl], np.roll(h90w, s)[sl]])
            ab, *_ = np.linalg.lstsq(X, dw[sl], rcond=None)
            r2 = float((dw[sl] - X @ ab) @ (dw[sl] - X @ ab))
            if best is None or r2 < best[0]:
                best = (r2, s, ab)
        _, shift, (a, b) = best
        resid = strain - (a * np.roll(h0, shift) + b * np.roll(h90, shift))
        fitw = a * np.roll(h0w, shift) + b * np.roll(h90w, shift); residw = dw - fitw

        an = hilbert(np.roll(h0, shift)); env = np.abs(an)
        finst = np.gradient(np.unwrap(np.angle(an)), DT) / (2 * np.pi)
        tt = (np.arange(DL) - ctr) * DT
        keep = (env > 0.05 * env.max()) & (finst > flow) & (finst < fhigh) & (np.abs(tt) < 0.5)

        def qscan(arr):
            return GTS(arr, sample_rate=SR, t0=start_t).q_transform(
                frange=(flow, fhigh), qrange=(4, 32),
                outseg=(gps - 0.35, gps + 0.10), whiten=True)
        q_data, q_res = qscan(strain), qscan(resid)

        fig, ax = plt.subplots(2, 2, figsize=(14, 9))
        m = np.abs(tt) < 0.20
        ax[0, 0].plot(tt[m], dw[m], color="#888", lw=1.0, label="whitened data")
        ax[0, 0].plot(tt[m], fitw[m], color="#1d4ed8", lw=1.6, label="surrogate reconstruction")
        ax[0, 0].set_title("(a) Whitened strain + surrogate reconstruction")
        ax[0, 0].set_xlim(-0.2, 0.1); ax[0, 0].legend(fontsize=9)
        ax[0, 1].plot(tt[m], dw[m], color="#ccc", lw=0.9, label="data")
        ax[0, 1].plot(tt[m], residw[m], color="#c1121f", lw=1.1, label="residual (signal subtracted)")
        ax[0, 1].set_title("(b) Time-domain subtraction residual")
        ax[0, 1].set_xlim(-0.2, 0.1); ax[0, 1].legend(fontsize=9)
        for a_, q, ttl in ((ax[1, 0], q_data, "(c) Q-scan of data + surrogate f(t)"),
                           (ax[1, 1], q_res, "(d) Q-scan of residual")):
            T = q.times.value - gps; F = q.frequencies.value
            pc = a_.pcolormesh(T, F, q.value.T, shading="auto", cmap="viridis",
                               vmin=0, vmax=float(np.percentile(q_data.value, 99.5)))
            a_.set_yscale("log"); a_.set_ylim(flow, fhigh); a_.set_xlim(-0.35, 0.10)
            a_.set_xlabel("time from merger [s]"); a_.set_ylabel("frequency [Hz]")
            a_.set_title(ttl); fig.colorbar(pc, ax=a_, label="normalised energy")
        ax[1, 0].plot(tt[keep], finst[keep], color="#ff3b3b", lw=2.2, label="surrogate f(t)")
        ax[1, 0].legend(fontsize=9, loc="upper left")
        for a_ in (ax[0, 0], ax[0, 1]):
            a_.set_xlabel("time from merger [s]")
        fig.suptitle(f"{self.event_name} ({detector}): TD reconstruction  "
                     f"m1={m1:.1f}, m2={m2:.1f} M_sun", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        if path:
            fig.savefig(path, dpi=140, bbox_inches="tight")
            plt.close(fig)
        return fig
