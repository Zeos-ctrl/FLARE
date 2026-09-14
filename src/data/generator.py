# src/data/generator.py
import os
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Tuple
from pycbc.waveform import get_td_waveform
from scipy.signal import hilbert
from scipy.stats import qmc

logger = logging.getLogger(__name__)


def _gen_worker(args):
    """Picklable module-level worker for parallel waveform generation.

    Each pycbc ``get_td_waveform`` call is independent and CPU-bound, so
    generation (the pipeline's real bottleneck now that ROM training is
    seconds) parallelises cleanly across cores.
    """
    params, wl, dt, fl, approx, clean, smooth, fixed_window = args
    gen = WaveformGenerator(waveform_length=wl, delta_t=dt, f_lower=fl,
                            approximant=approx, smooth_resize=smooth,
                            fixed_window=fixed_window)
    return gen._generate_single_waveform(np.asarray(params), clean=clean)

@dataclass
class WaveformDataset:
    """Clean data container for waveform datasets"""
    inputs: np.ndarray
    targets_amplitude: np.ndarray
    targets_phase: np.ndarray
    parameters: np.ndarray
    time_array: np.ndarray
    amplitude_scale: float
    parameter_means: np.ndarray
    parameter_stds: np.ndarray
    metadata: dict


@dataclass
class ReducedOrderDataset:
    """Reduced-order (SVD) training data.

    Instead of per-(sample, time) rows, each waveform is one row: normalised
    features ``theta`` map to standardised SVD coefficients for amplitude and
    phase. The bases expand those coefficients back to full (L,) amplitude/phase
    curves at inference. ``*_coeff_mean/std`` un-standardise the network output.
    """
    theta: np.ndarray                # (N, d) normalised features
    amp_coeffs: np.ndarray           # (N, k_a) standardised targets
    phase_coeffs: np.ndarray         # (N, k_p) standardised targets
    amp_basis: np.ndarray            # (L, k_a)
    phase_basis: np.ndarray          # (L, k_p)
    amp_coeff_mean: np.ndarray
    amp_coeff_std: np.ndarray
    phase_coeff_mean: np.ndarray
    phase_coeff_std: np.ndarray
    amplitude_scale: float
    parameters: np.ndarray
    time_array: np.ndarray
    parameter_means: np.ndarray
    parameter_stds: np.ndarray
    metadata: dict

class WaveformGenerator:
    """Handles waveform generation and data preparation"""
    
    def __init__(self, waveform_length: int = 2048, delta_t: float = 1/2048,
                 f_lower: float = 20.0, approximant: str = "SEOBNRv4",
                 smooth_resize: bool = True, fixed_window: bool = False):
        self.waveform_length = waveform_length
        self.delta_t = delta_t
        self.f_lower = f_lower
        self.approximant = approximant
        # Fixed physical-time window (for parameter estimation): place the merger
        # at MERGER_FRAC*L on a FIXED delta_t grid and keep the surrounding L real
        # samples -- NO time warp and NO amplitude trimming, so the waveform stays
        # a faithful physical-time signal (the warp/trim used for the match metric
        # discard the absolute timescale and the early inspiral, which breaks
        # matched-filter PE against real detector data). Overrides smooth_resize.
        self.fixed_window = fixed_window
        # Use the C1-smooth (PCHIP) merger-aligned time warp rather than the
        # two-segment linear resample. The two-segment map has a derivative kink
        # at the merger, which glitches the Hilbert phase (~4 rad spike at the
        # peak) and injects across-mass jitter into the phase/amplitude that caps
        # learnability -- badly for long low-mass waveforms. The smooth warp keeps
        # the merger pinned at MERGER_FRAC*L but removes the kink, so the learned
        # phase becomes a smooth function of the masses. (Low-mass SEOBNRv4
        # campaign: mean match 0.79 -> 0.97 on the many-cycle regime.)
        self.smooth_resize = smooth_resize
        
    def sample_parameters(self, n_samples: int, method: str = 'lhs',
                         mass_range: Tuple[float, float] = (30, 100),
                         spin_range: Tuple[float, float] = (-0.7, 0.7),
                         incl_range: Tuple[float, float] = (0.0, np.pi),
                         ecc_range: Tuple[float, float] = (0.0, 0.3),
                         seed: Optional[int] = None) -> np.ndarray:
        """Sample the [m1, m2, s1z, s2z, inclination, eccentricity] space."""

        lows = np.array([mass_range[0], mass_range[0], spin_range[0],
                        spin_range[0], incl_range[0], ecc_range[0]])
        highs = np.array([mass_range[1], mass_range[1], spin_range[1],
                         spin_range[1], incl_range[1], ecc_range[1]])

        rng = np.random.default_rng(seed)

        # Dimensions with an empty range (low == high) are *pinned* to that
        # constant — e.g. fixing spin to 0 so the waveform is fully determined
        # by mass alone. qmc.scale requires strictly increasing bounds, so we
        # scale against a widened placeholder and then overwrite the pinned
        # columns with their constant.
        fixed = highs <= lows
        if method == 'lhs':
            sampler = qmc.LatinHypercube(d=6, seed=seed)
            safe_highs = np.where(fixed, lows + 1.0, highs)
            samples = qmc.scale(sampler.random(n_samples), lows, safe_highs)
        else:
            safe_highs = np.where(fixed, lows + 1.0, highs)
            samples = rng.uniform(lows, safe_highs, size=(n_samples, 6))

        if fixed.any():
            samples[:, fixed] = lows[fixed]

        return samples

    def generate_dataset(self, n_samples: int, clean: bool = True,
                        feature_names: list = None,
                        sampling_ranges: Optional[dict] = None,
                        direct_strain: bool = False) -> WaveformDataset:
        """Generate complete dataset for training.

        Args:
            sampling_ranges: Optional dict with any of ``mass_range``,
                ``spin_range``, ``incl_range``, ``ecc_range`` to override the
                parameter-space bounds used when drawing samples.
            direct_strain: If True, the "amplitude" target is the signed
                normalised strain itself (for a single network that predicts the
                raw waveform) and the phase target is zeros. See
                :meth:`_process_waveforms`.
        """

        feature_names = feature_names or ['chirp_mass', 'symmetric_mass_ratio']
        parameters = self.sample_parameters(n_samples, **(sampling_ranges or {}))

        waveforms = self._generate_all(parameters, clean)

        # Process into training format
        dataset = self._process_waveforms(waveforms, parameters, feature_names,
                                          direct_strain=direct_strain)
        return dataset

    def generate_reduced_dataset(self, n_samples: int, clean: bool = True,
                                 feature_names: list = None,
                                 sampling_ranges: Optional[dict] = None,
                                 energy: float = 0.99999,
                                 max_rank: int = 64,
                                 min_rank: int = 1,
                                 scale_factor: bool = True,
                                 scale_weight: float = 20.0,
                                 scale_window: Tuple[float, float] = (0.1, 0.8),
                                 bank_path: str = ""
                                 ) -> ReducedOrderDataset:
        """Generate a reduced-order (SVD) dataset: one row per waveform.

        Builds the amplitude and phase matrices, compresses each with an SVD
        basis, and returns per-waveform *coefficient* targets instead of
        per-(sample, time) rows -- so the ``theta -> coeffs`` network trains on N
        rows rather than N*L. See :mod:`src.data.reduced_basis`.

        ``scale_factor`` splits the phase into a per-waveform scale (total
        accumulated phase over the clean interior window ``scale_window``, a
        smooth ~Mc^-5/3 function of the masses) and a near-universal SHAPE, then
        SVD-compresses only the shape; the phase coefficient vector becomes
        ``[shape_coeffs..., log_scale]``, with reconstruction
        ``phase = exp(log_scale) * (shape_coeffs @ shape_basis.T)``. Without this
        the dominant raw-phase coefficient *is* the total phase and would need
        ~1e-4 relative accuracy from the net (unreachable for long low-mass
        waveforms). ``scale_weight`` up-weights the single critical log-scale
        target in the coeff MSE (baked into its standardisation).
        """
        from src.data.features import FeatureExtractor
        from src.data.reduced_basis import build_basis

        feature_names = feature_names or ['chirp_mass', 'symmetric_mass_ratio']
        if bank_path:
            # Reuse a saved bank: load the first n_samples already-resized
            # waveforms + their params instead of regenerating (see
            # ``ProjectConfig.waveform_bank``). Everything downstream (Hilbert ->
            # SVD -> scale-factor) is identical, so one bank serves all configs.
            parameters, waveforms = self._load_bank(bank_path, n_samples)
        else:
            parameters = self.sample_parameters(n_samples, **(sampling_ranges or {}))
            waveforms = self._generate_all(parameters, clean)

        N = len(waveforms)
        L = self.waveform_length

        # Amplitude / unwrapped phase via the analytic (Hilbert) signal.
        amp = np.zeros((N, L), dtype=np.float64)
        phase = np.zeros((N, L), dtype=np.float64)
        for i, h in enumerate(waveforms):
            analytic = hilbert(np.asarray(h, dtype=np.float64))
            amp[i] = np.abs(analytic)
            phase[i] = np.unwrap(np.angle(analytic))
        amplitude_scale = float(amp.max()) or 1.0
        amp_norm = amp / amplitude_scale

        amp_basis, amp_coeffs, k_a = build_basis(amp_norm, energy, max_rank, min_rank)

        if scale_factor:
            # phase = scale * shape; anchor the scale on a clean window.
            if self.fixed_window:
                # Anchor RELATIVE TO THE MERGER so the window is always inside the
                # signal regardless of waveform length. A fixed (0.1,0.8)L interior
                # anchor works for long waveforms (IMRPhenomD) but lands in the
                # zero-padding for SHORT ones (SEOBNRv4 starts closer to f_lower):
                # the unwrapped Hilbert phase of the zero-pad random-walks, so
                # phase[0.8L]-phase[0.1L] collapses to ~0/negative (40% negative for
                # SEOBNRv4) and log(scale) diverges (phase val loss ~500, match 0.39).
                # The merger is always at MERGER_FRAC*L, and even the shortest
                # high-mass BBH spans >0.1L before it, so [merger-0.1L, merger] is
                # in-signal for every mass.
                merger = int(round(self.MERGER_FRAC * L))
                d = int(round(0.10 * L))
                i0, i1 = merger - d, merger
            else:
                # Warp/normalised representation: the signal fills the window, so a
                # fixed interior window is a clean, consistent scale anchor.
                i0, i1 = int(scale_window[0] * L), int(round(scale_window[1] * L)) - 1
            a0 = phase[:, i0:i0 + 1]
            pscale = (phase[:, i1:i1 + 1] - a0)
            pscale = np.where(np.abs(pscale) < 1e-6, 1.0, pscale)
            shape = (phase - a0) / pscale
            phase_basis, shape_coeffs, k_p = build_basis(shape, energy, max_rank, min_rank)
            log_scale = np.log(np.abs(pscale[:, 0]))
            # phase coeff vector = [shape_coeffs, log_scale]
            phase_coeffs = np.concatenate([shape_coeffs, log_scale[:, None]], axis=1)
        else:
            phase_basis, phase_coeffs, k_p = build_basis(phase, energy, max_rank, min_rank)

        # Centre per component, then scale by a SINGLE global factor per curve.
        # The SVD basis is orthonormal, so reconstruction error == coefficient
        # error (Parseval); per-component std-scaling would mis-weight the
        # dominant coefficient (which carries most of the waveform, e.g. the huge
        # low-mass phase ramp) and wreck the fit. Global scaling keeps the MSE
        # proportional to the actual reconstruction error.
        amp_cmean = amp_coeffs.mean(0)
        amp_cstd = np.full(amp_coeffs.shape[1], float(amp_coeffs.std()) + 1e-30)
        amp_coeffs_std = (amp_coeffs - amp_cmean) / amp_cstd

        phase_cmean = phase_coeffs.mean(0)
        phase_cstd = np.full(phase_coeffs.shape[1], float(phase_coeffs[:, :k_p].std()) + 1e-30)
        if scale_factor:
            # Standardise the log-scale by its OWN std / sqrt(weight): dividing the
            # std shrinks it, inflating the standardised target so its MSE counts
            # ~scale_weight x more (the scale is the single precision-critical
            # output). Un-standardising with the same std recovers it at predict.
            phase_cstd[-1] = (float(phase_coeffs[:, -1].std()) + 1e-30) / max(scale_weight, 1e-6) ** 0.5
        phase_coeffs_std = (phase_coeffs - phase_cmean) / phase_cstd

        # Standardised input features.
        features = FeatureExtractor.compute_features(parameters, feature_names)
        feat_norm, means, stds = FeatureExtractor.normalize_features(features)

        logger.info(f"Reduced-order dataset: N={N}, amp_rank={k_a}, "
                    f"phase_rank={k_p} (scale_factor={scale_factor})")

        metadata = {
            "n_samples": N,
            "waveform_length": L,
            "delta_t": self.delta_t,
            "approximant": self.approximant,
            "feature_names": list(feature_names),
            "reduced_order": True,
            "amp_rank": k_a,
            "phase_rank": k_p,
            "phase_scale_factor": bool(scale_factor),
            "td_fixed_window": bool(self.fixed_window),
            "merger_frac": float(self.MERGER_FRAC),
        }

        return ReducedOrderDataset(
            theta=feat_norm.astype(np.float32),
            amp_coeffs=amp_coeffs_std.astype(np.float32),
            phase_coeffs=phase_coeffs_std.astype(np.float32),
            amp_basis=amp_basis.astype(np.float32),
            phase_basis=phase_basis.astype(np.float32),
            amp_coeff_mean=amp_cmean, amp_coeff_std=amp_cstd,
            phase_coeff_mean=phase_cmean, phase_coeff_std=phase_cstd,
            amplitude_scale=amplitude_scale,
            parameters=parameters,
            time_array=np.linspace(0.0, 1.0, L, dtype=np.float64),
            parameter_means=means,
            parameter_stds=stds,
            metadata=metadata,
        )
    
    def _load_bank(self, bank_path: str, n_samples: int):
        """Load the first ``n_samples`` (params, resized-strain) rows from a saved
        waveform bank (.npz with ``strain`` (M, L) and ``params`` (M, 6)).

        The bank stores merger-aligned/resized strain at a fixed length; it must
        match this generator's ``waveform_length`` so the downstream Hilbert/SVD
        run on the same grid the evaluator uses.
        """
        if not os.path.isfile(bank_path):
            raise FileNotFoundError(f"waveform_bank not found: {bank_path}")
        d = np.load(bank_path)
        strain, params = d["strain"], d["params"]
        if strain.shape[1] != self.waveform_length:
            raise ValueError(
                f"bank waveform_length {strain.shape[1]} != generator "
                f"{self.waveform_length}; rebuild the bank or fix waveform_length.")
        M = len(strain)
        if n_samples > M:
            logger.warning(f"waveform_bank has {M} < requested {n_samples}; "
                           f"using all {M}.")
            n_samples = M
        strain = np.asarray(strain[:n_samples], dtype=np.float64)
        params = np.asarray(params[:n_samples], dtype=np.float64)
        logger.info(f"Loaded {n_samples} waveforms from bank {bank_path}")
        return params, [strain[i] for i in range(n_samples)]

    def _generate_single_waveform(self, params: np.ndarray, clean: bool) -> np.ndarray:
        """Generate a single waveform"""
        m1, m2, s1z, s2z, inc, ecc = params
        
        hp, _ = get_td_waveform(
            mass1=m1, mass2=m2,
            spin1z=s1z, spin2z=s2z,
            inclination=inc, eccentricity=ecc,
            delta_t=self.delta_t, f_lower=self.f_lower,
            approximant=self.approximant
        )
        
        # Process and resize
        h = self._resize_waveform(hp.numpy())

        if not clean:
            h = self._add_noise(h)

        return h

    def _generate_all(self, parameters: np.ndarray, clean: bool) -> list:
        """Generate every waveform, in parallel across CPU cores when worth it.

        pycbc waveform generation is CPU-bound and single-threaded per call, and
        is the pipeline bottleneck now that ROM training runs in seconds. Each
        call is independent, so we fan out over a process pool. Set
        ``FLARE_GEN_WORKERS`` to override the worker count (1 = serial).
        """
        n = len(parameters)
        default_workers = min(os.cpu_count() or 1, 12)
        workers = int(os.environ.get('FLARE_GEN_WORKERS', default_workers))

        if workers <= 1 or n < 32:
            return [self._generate_single_waveform(p, clean=clean)
                    for p in parameters]

        args = [(np.asarray(p), self.waveform_length, self.delta_t,
                 self.f_lower, self.approximant, clean, self.smooth_resize,
                 self.fixed_window)
                for p in parameters]
        chunk = max(1, n // (workers * 8))
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(_gen_worker, args, chunksize=chunk))
        except Exception as exc:  # pragma: no cover - robustness fallback
            logger.warning(f"Parallel generation failed ({exc}); using serial.")
            return [self._generate_single_waveform(p, clean=clean)
                    for p in parameters]

    # Fraction of the output window allocated to inspiral+merger; the merger
    # (peak amplitude) is pinned here so every waveform's merger lands at the
    # same normalised time. Matches the original fixed-window convention
    # (merger at L - L//10).
    MERGER_FRAC = 0.9

    @staticmethod
    def _resample(seg: np.ndarray, m: int) -> np.ndarray:
        """Linearly resample a 1-D segment onto ``m`` points (robust to len<2)."""
        if m <= 0:
            return np.empty(0, dtype=np.float64)
        if len(seg) == 0:
            return np.zeros(m, dtype=np.float64)
        if len(seg) == 1:
            return np.full(m, seg[0], dtype=np.float64)
        return np.interp(np.linspace(0.0, 1.0, m),
                         np.linspace(0.0, 1.0, len(seg)), seg)

    def _resize_waveform(self, h: np.ndarray) -> np.ndarray:
        """Trim near-zero padding and resample onto ``waveform_length`` samples
        with every waveform's merger aligned to a fixed normalised time.

        PyCBC time-domain waveforms carry a long, near-zero pre-inspiral and
        post-ringdown tail, and the merger sits at a *different* fraction of
        the active span depending on mass (the ringdown-to-inspiral ratio
        varies). Simply stretching the whole active span onto L would leave
        the mergers misaligned across the dataset. Instead we resample the
        inspiral ([start, peak]) and the ringdown ((peak, end]) separately so
        the merger always lands at ``MERGER_FRAC * L`` -- padding removed,
        every sample signal-bearing, and all mergers aligned so the network
        sees a consistent time reference.
        """
        L = self.waveform_length
        h = np.asarray(h, dtype=np.float64)
        n = len(h)
        if n == 0:
            return np.zeros(L, dtype=np.float64)

        envelope = np.abs(h)
        peak_amp = envelope.max()
        if peak_amp == 0:
            return np.zeros(L, dtype=np.float64)

        active = np.flatnonzero(envelope > 1e-3 * peak_amp)
        start, end = active[0], active[-1] + 1
        if end - start < 2:
            start, end = 0, n
        peak = int(np.clip(np.argmax(envelope), start, end - 1))

        if self.fixed_window:
            return self._fixed_window_resize(h, peak)
        if self.smooth_resize:
            return self._smooth_warp(h, start, peak, end)

        n_pre = max(2, int(round(self.MERGER_FRAC * L)))   # inspiral + merger
        n_post = L - n_pre                                 # ringdown
        pre = self._resample(h[start:peak + 1], n_pre)
        post = self._resample(h[peak + 1:end], n_post)
        return np.concatenate([pre, post])

    def _fixed_window_resize(self, h: np.ndarray, peak: int) -> np.ndarray:
        """Place the merger (``peak``) at ``MERGER_FRAC*L`` on the native
        ``delta_t`` grid and keep the L surrounding physical samples (zero-padded
        if the waveform is shorter than the window). No resampling/warp, no
        amplitude trim -- the output is a true physical-time waveform, so the
        reduced-order reconstruction is directly matched-filterable against real
        data. The window length is ``L * delta_t`` seconds; pick ``waveform_length``
        so the in-band signal for the mass range fits it."""
        L = self.waveform_length
        pk = int(round(self.MERGER_FRAC * L))
        out = np.zeros(L, dtype=np.float64)
        s0 = peak - pk                       # h index that maps to window start
        lo = max(0, s0)
        hi = min(len(h), s0 + L)
        if hi > lo:
            out[lo - s0:hi - s0] = h[lo:hi]
        return out

    def _smooth_warp(self, h: np.ndarray, start: int, peak: int,
                     end: int) -> np.ndarray:
        """Merger-aligned resample via a C1-smooth monotonic (PCHIP) time warp.

        The two-segment linear resample pins the merger at ``MERGER_FRAC*L`` but
        with a derivative kink there (inspiral and ringdown sampled at different
        rates). That kink glitches the analytic-signal phase at the merger and
        makes the extracted amplitude/phase a jittery, hard-to-learn function of
        the masses. A monotone cubic (PCHIP) interpolant through the three knots
        ``t_norm -> raw_index`` = ``{0:start, MERGER_FRAC:peak, 1:end-1}`` keeps
        the merger pinned but removes the kink, so the representation is smooth.
        """
        from scipy.interpolate import PchipInterpolator
        L = self.waveform_length
        e = max(peak + 1, end - 1)
        warp = PchipInterpolator([0.0, self.MERGER_FRAC, 1.0],
                                 [float(start), float(peak), float(e)])
        idx = warp(np.linspace(0.0, 1.0, L))
        return np.interp(idx, np.arange(len(h)), h)

    def _add_noise(self, h: np.ndarray, snr: Optional[float] = None,
                   seed: Optional[int] = None) -> np.ndarray:
        """Add white Gaussian noise scaled to a target matched-to-signal SNR.

        A crude white-noise model (not a coloured detector PSD) is sufficient
        for augmenting the training set; ``snr`` controls the noise level
        relative to the signal's RMS amplitude.
        """
        rng = np.random.default_rng(seed)
        snr = 15.0 if snr is None else snr
        signal_rms = np.sqrt(np.mean(h ** 2)) + 1e-30
        noise_sigma = signal_rms / max(snr, 1e-6)
        return h + rng.normal(0.0, noise_sigma, size=h.shape)

    def _process_waveforms(self, waveforms: list, parameters: np.ndarray,
                          feature_names: list,
                          direct_strain: bool = False) -> WaveformDataset:
        """Flatten waveforms into per-(sample, time) training rows.

        Two target representations:

        * **decomposition** (default): extract instantaneous amplitude and
          unwrapped phase via the analytic (Hilbert) signal; a smooth-amplitude
          net and a smooth-phase net are trained separately and recombined as
          ``h = A*cos(phase)``.
        * **direct_strain**: the amplitude target *is* the signed strain
          (normalised by a single global scale) and the phase target is zeros.
          A single network then predicts the raw waveform directly -- pair a
          linear-output model (e.g. a SIREN) in the amplitude slot with the
          ``zero_phase`` model so the recombination ``A*cos(0) = A`` returns the
          strain. Match is scale-invariant, so the global scale is arbitrary.

        Either way the design matrix rows are ``[t_norm, f_1, ..., f_k]``.
        """
        from src.data.features import FeatureExtractor

        n_samples = len(waveforms)
        L = self.waveform_length

        if direct_strain:
            strain = np.stack([np.asarray(h, dtype=np.float64) for h in waveforms])
            amplitude_scale = float(np.abs(strain).max()) or 1.0
            amp_norm = strain / amplitude_scale      # signed, in [-1, 1]
            phase = np.zeros((n_samples, L), dtype=np.float64)
        else:
            # Instantaneous amplitude/phase via the analytic signal.
            amp = np.zeros((n_samples, L), dtype=np.float64)
            phase = np.zeros((n_samples, L), dtype=np.float64)
            for i, h in enumerate(waveforms):
                analytic = hilbert(np.asarray(h, dtype=np.float64))
                amp[i] = np.abs(analytic)
                phase[i] = np.unwrap(np.angle(analytic))

            # Global amplitude scale keeps normalised targets in [0, 1] to match
            # the sigmoid output activation of the amplitude network.
            amplitude_scale = float(amp.max()) or 1.0
            amp_norm = amp / amplitude_scale

        # Normalised time grid in [0, 1] shared by every waveform.
        t_norm = np.linspace(0.0, 1.0, L, dtype=np.float64)

        # Derived, standardised features.
        features = FeatureExtractor.compute_features(parameters, feature_names)
        feat_norm, means, stds = FeatureExtractor.normalize_features(features)

        # Broadcast to per-(sample, time) rows.
        theta_rows = np.repeat(feat_norm, L, axis=0)            # (N*L, k)
        t_rows = np.tile(t_norm, n_samples).reshape(-1, 1)      # (N*L, 1)
        inputs = np.hstack([t_rows, theta_rows]).astype(np.float32)

        targets_amplitude = amp_norm.reshape(-1, 1).astype(np.float32)
        targets_phase = phase.reshape(-1, 1).astype(np.float32)

        metadata = {
            "n_samples": n_samples,
            "waveform_length": L,
            "delta_t": self.delta_t,
            "approximant": self.approximant,
            "feature_names": list(feature_names),
        }

        return WaveformDataset(
            inputs=inputs,
            targets_amplitude=targets_amplitude,
            targets_phase=targets_phase,
            parameters=parameters,
            time_array=t_norm,
            amplitude_scale=amplitude_scale,
            parameter_means=means,
            parameter_stds=stds,
            metadata=metadata,
        )
