# src/inference/predictor.py
import torch
import numpy as np
import json
import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

logger = logging.getLogger(__name__)

@dataclass
class WaveformPrediction:
    """Container for waveform predictions with optional uncertainty"""
    data: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    time: np.ndarray = None
    sample_rate: float = 1/2048
    approximant: str = "unknown"
    
    @property
    def has_uncertainty(self) -> bool:
        return self.uncertainty is not None
    
    def to_timeseries(self):
        """Convert to PyCBC/GWpy TimeSeries format"""
        from pycbc.types import TimeSeries
        return TimeSeries(self.data, delta_t=self.sample_rate)

class WaveformPredictor:
    """Unified predictor for gravitational waveform generation"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._load_checkpoint()
        self._setup_models()
        
    def _load_checkpoint(self):
        """Load models and metadata from checkpoint"""
        # Load metadata
        with open(os.path.join(self.checkpoint_path, 'meta.json')) as f:
            self.meta = json.load(f)
        
        # Load normalization parameters
        self.param_means = np.load(os.path.join(self.checkpoint_path, 'param_means.npy'))
        self.param_stds = np.load(os.path.join(self.checkpoint_path, 'param_stds.npy'))
        self.time_norm = np.load(os.path.join(self.checkpoint_path, 't_norm_array.npy'))
        
        # Load amplitude scale
        self.amp_scale = self.meta['amp_scale']
        self.waveform_length = self.meta['waveform_length']
        self.delta_t = self.meta['delta_t']
        self.feature_names = self.meta.get('feature_names', ['chirp_mass', 'symmetric_mass_ratio'])

        # Reduced-order (SVD) checkpoints carry per-curve bases + coeff stats.
        self.reduced_order = bool(self.meta.get('reduced_order', False))
        if self.reduced_order:
            def _t(name):
                return torch.from_numpy(
                    np.load(os.path.join(self.checkpoint_path, f'{name}.npy'))
                ).float().to(self.device)
            self.amp_basis = _t('amp_basis')          # (L, k_a)
            self.phase_basis = _t('phase_basis')      # (L, k_p)
            self.amp_coeff_mean = _t('amp_coeff_mean')
            self.amp_coeff_std = _t('amp_coeff_std')
            self.phase_coeff_mean = _t('phase_coeff_mean')
            self.phase_coeff_std = _t('phase_coeff_std')

        # Load uncertainty parameters if available
        self._load_uncertainty_params()
        
    def _load_uncertainty_params(self):
        """Load Laplace approximation parameters for uncertainty"""
        try:
            self.amp_weight_var = torch.from_numpy(
                np.load(os.path.join(self.checkpoint_path, 'amp_last_weight_variances.npy'))
            ).to(self.device)
            self.amp_bias_var = torch.from_numpy(
                np.load(os.path.join(self.checkpoint_path, 'amp_last_bias_variance.npy'))
            ).to(self.device)
            self.phase_weight_var = torch.from_numpy(
                np.load(os.path.join(self.checkpoint_path, 'phase_last_weight_variances.npy'))
            ).to(self.device)
            self.phase_bias_var = torch.from_numpy(
                np.load(os.path.join(self.checkpoint_path, 'phase_last_bias_variance.npy'))
            ).to(self.device)
            self.has_uncertainty = True
        except FileNotFoundError:
            self.has_uncertainty = False
            self.logger.info("Uncertainty parameters not found - predictions without uncertainty only")
    
    def _setup_models(self):
        """Initialize and load model weights"""
        from src.models.model_factory import make_amp_model, make_phase_model

        # Load hyperparameters
        with open(os.path.join(self.checkpoint_path, 'amp_params.json')) as f:
            amp_params = json.load(f)
        with open(os.path.join(self.checkpoint_path, 'phase_params.json')) as f:
            phase_params = json.load(f)

        if self.reduced_order:
            from src.models.rom import CoeffMLP

            def _coeff_net(p):
                return CoeffMLP(p['in_param_dim'], p['out_dim'], hidden=p['hidden'],
                                depth=p['depth'], fourier_bands=p['fourier_bands'],
                                fourier_max_freq=p['fourier_max_freq']).to(self.device)
            self.amp_model = _coeff_net(amp_params)
            self.phase_model = _coeff_net(phase_params)
            self.amp_model.load_state_dict(torch.load(
                os.path.join(self.checkpoint_path, 'amp_model.pt'), map_location=self.device))
            self.phase_model.load_state_dict(torch.load(
                os.path.join(self.checkpoint_path, 'phase_model.pt'), map_location=self.device))
            self.amp_model.eval()
            self.phase_model.eval()
            # dedicated-scale checkpoints carry a third net for the log-scale
            self.scale_model = None
            if self.meta.get('separate_scale'):
                with open(os.path.join(self.checkpoint_path, 'scale_params.json')) as f:
                    scale_params = json.load(f)
                self.scale_model = _coeff_net(scale_params)
                self.scale_model.load_state_dict(torch.load(
                    os.path.join(self.checkpoint_path, 'scale_model.pt'),
                    map_location=self.device))
                self.scale_model.eval()
            return

        # Create models
        feature_dim = len(self.feature_names)
        self.amp_model = make_amp_model(feature_dim, amp_params).to(self.device)
        self.phase_model = make_phase_model(feature_dim, phase_params).to(self.device)
        
        # Load weights
        self.amp_model.load_state_dict(
            torch.load(os.path.join(self.checkpoint_path, 'amp_model.pt'), 
                      map_location=self.device)
        )
        self.phase_model.load_state_dict(
            torch.load(os.path.join(self.checkpoint_path, 'phase_model.pt'),
                      map_location=self.device)
        )
        
        self.amp_model.eval()
        self.phase_model.eval()
    
    def predict(self, m1: float, m2: float, s1z: float = 0, s2z: float = 0,
               inc: float = 0, ecc: float = 0, 
               sigma_level: Optional[int] = None) -> Tuple[WaveformPrediction, WaveformPrediction]:
        """
        Predict gravitational waveform polarizations
        
        Args:
            m1, m2: Component masses in solar masses
            s1z, s2z: Spin components along orbital angular momentum
            inc: Inclination angle in radians
            ecc: Eccentricity
            sigma_level: If provided, compute uncertainty at this sigma level (1, 2, or 3)
        
        Returns:
            Tuple of (h_plus, h_cross) WaveformPrediction objects
        """
        # Prepare parameters
        params = np.array([[m1, m2, s1z, s2z, inc, ecc]])
        
        if sigma_level is not None and self.has_uncertainty:
            return self._predict_with_uncertainty(params[0], sigma_level)
        else:
            return self._predict_standard(params[0])
    
    def _predict_standard(self, params: np.ndarray) -> Tuple[WaveformPrediction, WaveformPrediction]:
        """Standard prediction without uncertainty"""
        if self.reduced_order:
            hp, hc = self._batch_predict_reduced(params.reshape(1, -1))
            return hp[0], hc[0]
        # Compute features
        features = self._compute_features(params)
        normalized = self._normalize_features(features)
        
        # Prepare input
        input_tensor = self._prepare_input(normalized)
        
        # Forward pass
        with torch.no_grad():
            amp_pred = self.amp_model(input_tensor[:, :1], input_tensor[:, 1:])
            phase_pred = self.phase_model(input_tensor[:, :1], input_tensor[:, 1:])
        
        # Process outputs
        amplitude = self._unscale_amplitude(amp_pred.cpu().numpy().ravel())
        phase = phase_pred.cpu().numpy().ravel()
        
        # Compute polarizations
        h_plus, h_cross = self._compute_polarizations(amplitude, phase, params[4])
        
        return (
            WaveformPrediction(h_plus, time=self.time_norm, sample_rate=self.delta_t),
            WaveformPrediction(h_cross, time=self.time_norm, sample_rate=self.delta_t)
        )
    
    def _predict_with_uncertainty(self, params: np.ndarray, 
                                 sigma_level: int) -> Tuple[WaveformPrediction, WaveformPrediction]:
        """Prediction with uncertainty quantification"""
        # Similar to standard but capture hidden features
        features = self._compute_features(params)
        normalized = self._normalize_features(features)
        input_tensor = self._prepare_input(normalized)
        
        # Capture hidden features for uncertainty
        captured = {}
        
        def hook_fn(name):
            def hook(module, inp, out):
                captured[name] = inp[0].detach()
            return hook
        
        # Find last linear layers
        amp_last = self._find_last_linear(self.amp_model)
        phase_last = self._find_last_linear(self.phase_model)
        
        # Register hooks
        amp_hook = amp_last.register_forward_hook(hook_fn('amp'))
        phase_hook = phase_last.register_forward_hook(hook_fn('phase'))
        
        # Forward pass
        with torch.no_grad():
            amp_mean = self.amp_model(input_tensor[:, :1], input_tensor[:, 1:])
            phase_mean = self.phase_model(input_tensor[:, :1], input_tensor[:, 1:])
        
        # Remove hooks
        amp_hook.remove()
        phase_hook.remove()
        
        # Compute variances
        amp_var = (captured['amp']**2 * self.amp_weight_var).sum(1, True) + self.amp_bias_var
        phase_var = (captured['phase']**2 * self.phase_weight_var).sum(1, True) + self.phase_bias_var
        
        # Convert to numpy
        amp_mean_np = amp_mean.cpu().numpy().ravel()
        amp_std_np = np.sqrt(amp_var.cpu().numpy().ravel()) * sigma_level
        phase_mean_np = phase_mean.cpu().numpy().ravel()
        phase_std_np = np.sqrt(phase_var.cpu().numpy().ravel()) * sigma_level
        
        # Compute polarizations with uncertainty
        amplitude = self._unscale_amplitude(amp_mean_np)
        h_plus, h_cross, h_plus_unc, h_cross_unc = self._compute_polarizations_with_uncertainty(
            amplitude, phase_mean_np, amp_std_np, phase_std_np, params[4]
        )
        
        return (
            WaveformPrediction(h_plus, h_plus_unc, self.time_norm, self.delta_t),
            WaveformPrediction(h_cross, h_cross_unc, self.time_norm, self.delta_t)
        )
    
    def batch_predict(self, parameters: np.ndarray, batch_size: int = 32,
                     sigma_level: Optional[int] = None) -> Tuple[List[WaveformPrediction], List[WaveformPrediction]]:
        """
        Batch prediction for multiple parameter sets
        
        Args:
            parameters: Array of shape (N, 6) with parameter sets
            batch_size: Batch size for GPU processing
            sigma_level: Optional uncertainty level
        
        Returns:
            Lists of h_plus and h_cross predictions
        """
        n_samples = len(parameters)
        h_plus_list = []
        h_cross_list = []
        
        for i in range(0, n_samples, batch_size):
            batch = parameters[i:i+batch_size]
            
            if sigma_level is not None and self.has_uncertainty:
                h_plus_batch, h_cross_batch = self._batch_predict_with_uncertainty(
                    batch, sigma_level
                )
            else:
                h_plus_batch, h_cross_batch = self._batch_predict_standard(batch)
            
            h_plus_list.extend(h_plus_batch)
            h_cross_list.extend(h_cross_batch)
        
        return h_plus_list, h_cross_list
    
    def _reduced_amp_phase(self, parameters: np.ndarray):
        """Reduced-order decode: theta -> standardised SVD coeffs -> full
        amplitude and unwrapped phase curves on the normalised grid. Returns
        ``(amp_matrix, phase_matrix)`` each ``(n, L)``. Shared by the polarization
        predictor and the physical-time reconstruction."""
        features = self._compute_features_vectorized(parameters)
        normalized = self._normalize_features(features)
        theta = torch.from_numpy(normalized.astype(np.float32)).to(self.device)
        with torch.no_grad():
            amp_c = self.amp_model(theta) * self.amp_coeff_std + self.amp_coeff_mean
            amp_norm = amp_c @ self.amp_basis.T        # (n, L)
            if self.meta.get('separate_scale'):
                # shape coeffs from the phase net; log-scale from its own net.
                ks = self.phase_basis.shape[1]
                shape_c = (self.phase_model(theta) * self.phase_coeff_std[:ks]
                           + self.phase_coeff_mean[:ks])
                log_scale = (self.scale_model(theta) * self.phase_coeff_std[ks:]
                             + self.phase_coeff_mean[ks:])
                phase_mat = torch.exp(log_scale) * (shape_c @ self.phase_basis.T)
            elif self.meta.get('phase_scale_factor'):
                # phase coeff vector = [shape_coeffs, log_scale]; reconstruct
                # phase = exp(log_scale) * (shape_coeffs @ shape_basis.T).
                phase_c = self.phase_model(theta) * self.phase_coeff_std + self.phase_coeff_mean
                log_scale = phase_c[:, -1:]
                shape = phase_c[:, :-1] @ self.phase_basis.T   # (n, L)
                phase_mat = torch.exp(log_scale) * shape
            else:
                phase_c = self.phase_model(theta) * self.phase_coeff_std + self.phase_coeff_mean
                phase_mat = phase_c @ self.phase_basis.T   # (n, L)
        amp_matrix = np.clip(amp_norm.cpu().numpy(), 0.0, None) * self.amp_scale
        phase_matrix = phase_mat.cpu().numpy()
        return amp_matrix, phase_matrix

    def _batch_predict_reduced(self, parameters: np.ndarray):
        """Reduced-order reconstruction: theta -> coeffs -> basis expansion, then
        polarizations on the merger-aligned normalised grid. ``h = amp*cos(phase)``."""
        amp_matrix, phase_matrix = self._reduced_amp_phase(parameters)

        h_plus_list, h_cross_list = [], []
        for i in range(len(parameters)):
            h_plus, h_cross = self._compute_polarizations(
                amp_matrix[i], phase_matrix[i], parameters[i, 4])
            h_plus_list.append(WaveformPrediction(h_plus, time=self.time_norm, sample_rate=self.delta_t))
            h_cross_list.append(WaveformPrediction(h_cross, time=self.time_norm, sample_rate=self.delta_t))
        return h_plus_list, h_cross_list

    def predict_physical(self, m1: float, m2: float,
                         durations=None, duration_model=None):
        """Reconstruct a PHYSICAL-time strain (uniform ``delta_t``) for one binary.

        The reduced-order predictor outputs amplitude/phase on the merger-aligned
        NORMALISED grid, which is not a physical-time waveform (the absolute chirp
        timescale is warped out). This inverts that warp using the physical
        inspiral/ringdown durations -- from ``durations=(T_pre, T_post)`` if given,
        else predicted from the masses by a
        :class:`~src.inference.physical_time.DurationModel` -- so the result can be
        matched-filtered against real detector data. See that module for the
        duration-precision caveat. Requires reduced-order mode.

        Returns a 1-D ``np.ndarray`` (``h_plus`` at the stored inclination).
        """
        if not self.reduced_order:
            raise RuntimeError("predict_physical requires a reduced-order model.")
        from src.inference.physical_time import reconstruct_physical
        params = np.array([[m1, m2, 0.0, 0.0, 0.0, 0.0]], dtype=float)
        amp_matrix, phase_matrix = self._reduced_amp_phase(params)
        return reconstruct_physical(
            amp_matrix[0], phase_matrix[0], m1, m2, self.delta_t,
            duration_model=duration_model, durations=durations)

    def _batch_predict_standard(self, parameters: np.ndarray) -> Tuple[List[WaveformPrediction], List[WaveformPrediction]]:
        """Efficient batch prediction without uncertainty"""
        if self.reduced_order:
            return self._batch_predict_reduced(parameters)
        # Vectorized feature computation
        features = self._compute_features_vectorized(parameters)
        normalized = self._normalize_features(features)
        
        # Prepare batch input
        batch_input = self._prepare_batch_input(normalized)
        
        # GPU forward pass
        with torch.no_grad():
            amp_out = self.amp_model(batch_input[:, :1], batch_input[:, 1:])
            phase_out = self.phase_model(batch_input[:, :1], batch_input[:, 1:])
        
        # Reshape outputs
        n_batch = len(parameters)
        amp_matrix = amp_out.reshape(n_batch, self.waveform_length).cpu().numpy()
        phase_matrix = phase_out.reshape(n_batch, self.waveform_length).cpu().numpy()
        
        # Unscale amplitudes
        amp_matrix = self._unscale_amplitude(amp_matrix)
        
        # Compute polarizations for each sample
        h_plus_list = []
        h_cross_list = []
        
        for i in range(n_batch):
            h_plus, h_cross = self._compute_polarizations(
                amp_matrix[i], phase_matrix[i], parameters[i, 4]
            )
            h_plus_list.append(WaveformPrediction(h_plus, time=self.time_norm, sample_rate=self.delta_t))
            h_cross_list.append(WaveformPrediction(h_cross, time=self.time_norm, sample_rate=self.delta_t))
        
        return h_plus_list, h_cross_list
    
    # Helper methods
    def _compute_features(self, params: np.ndarray) -> np.ndarray:
        """Compute derived features from parameters"""
        from src.data.features import FeatureExtractor
        return FeatureExtractor.compute_features(params.reshape(1, -1), self.feature_names)[0]
    
    def _compute_features_vectorized(self, parameters: np.ndarray) -> np.ndarray:
        """Vectorized feature computation"""
        from src.data.features import FeatureExtractor
        return FeatureExtractor.compute_features(parameters, self.feature_names)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using stored statistics"""
        return (features - self.param_means) / self.param_stds
    
    def _unscale_amplitude(self, amp_norm: np.ndarray) -> np.ndarray:
        """Convert normalized amplitude to physical scale"""
        return amp_norm * self.amp_scale
    
    def _compute_polarizations(self, amplitude: np.ndarray, phase: np.ndarray, 
                              inclination: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute h_plus and h_cross from amplitude and phase"""
        cos_inc = np.cos(inclination)
        h_plus = amplitude * ((1 + cos_inc**2) / 2) * np.cos(phase)
        h_cross = amplitude * cos_inc * np.sin(phase)
        return h_plus, h_cross
    
    def _find_last_linear(self, model: torch.nn.Module) -> torch.nn.Linear:
        """Find the last linear layer in a model"""
        linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        return linears[-1]

    def _prepare_input(self, normalized: np.ndarray) -> torch.Tensor:
        """Build the (T, 1+k) design matrix for a single parameter set.

        Column 0 is the shared normalized time grid; the remaining columns are
        the standardized features tiled across every time sample.
        """
        T = self.waveform_length
        t = self.time_norm.reshape(-1, 1)
        theta = np.tile(normalized.reshape(1, -1), (T, 1))
        x = np.hstack([t, theta]).astype(np.float32)
        return torch.from_numpy(x).to(self.device)

    def _prepare_batch_input(self, normalized: np.ndarray) -> torch.Tensor:
        """Build the (n_batch*T, 1+k) design matrix for a batch of parameters.

        Rows are ordered sample-major (all T time samples of sample 0, then
        sample 1, ...) to match the ``reshape(n_batch, T)`` in the callers.
        """
        T = self.waveform_length
        n = normalized.shape[0]
        theta = np.repeat(normalized, T, axis=0)
        t = np.tile(self.time_norm, n).reshape(-1, 1)
        x = np.hstack([t, theta]).astype(np.float32)
        return torch.from_numpy(x).to(self.device)

    def _compute_polarizations_with_uncertainty(
            self, amplitude: np.ndarray, phase: np.ndarray,
            amp_std: np.ndarray, phase_std: np.ndarray,
            inclination: float):
        """Polarizations plus first-order propagated uncertainty bands.

        ``amp_std`` is in normalized-amplitude units (network output space) and
        is rescaled here; ``phase_std`` is already in radians. Both are assumed
        pre-scaled to the requested sigma level.
        """
        amp_std_phys = amp_std * self.amp_scale
        cos_inc = np.cos(inclination)
        fp = (1 + cos_inc ** 2) / 2
        fc = cos_inc
        cos_p, sin_p = np.cos(phase), np.sin(phase)

        h_plus = amplitude * fp * cos_p
        h_cross = amplitude * fc * sin_p

        hp_var = (fp * cos_p) ** 2 * amp_std_phys ** 2 \
            + (amplitude * fp * sin_p) ** 2 * phase_std ** 2
        hc_var = (fc * sin_p) ** 2 * amp_std_phys ** 2 \
            + (amplitude * fc * cos_p) ** 2 * phase_std ** 2

        return h_plus, h_cross, np.sqrt(hp_var), np.sqrt(hc_var)

    def _batch_predict_with_uncertainty(self, parameters: np.ndarray, sigma_level: int):
        """Batched prediction with Laplace-approximation uncertainty bands."""
        features = self._compute_features_vectorized(parameters)
        normalized = self._normalize_features(features)
        batch_input = self._prepare_batch_input(normalized)

        captured = {}

        def hook_fn(name):
            def hook(module, inp, out):
                captured[name] = inp[0].detach()
            return hook

        amp_last = self._find_last_linear(self.amp_model)
        phase_last = self._find_last_linear(self.phase_model)
        amp_hook = amp_last.register_forward_hook(hook_fn('amp'))
        phase_hook = phase_last.register_forward_hook(hook_fn('phase'))

        with torch.no_grad():
            amp_out = self.amp_model(batch_input[:, :1], batch_input[:, 1:])
            phase_out = self.phase_model(batch_input[:, :1], batch_input[:, 1:])

        amp_hook.remove()
        phase_hook.remove()

        amp_var = (captured['amp'] ** 2 * self.amp_weight_var).sum(1, True) + self.amp_bias_var
        phase_var = (captured['phase'] ** 2 * self.phase_weight_var).sum(1, True) + self.phase_bias_var

        n = len(parameters)
        T = self.waveform_length
        amp_matrix = self._unscale_amplitude(amp_out.reshape(n, T).cpu().numpy())
        phase_matrix = phase_out.reshape(n, T).cpu().numpy()
        amp_std_matrix = (np.sqrt(amp_var.cpu().numpy()) * sigma_level).reshape(n, T)
        phase_std_matrix = (np.sqrt(phase_var.cpu().numpy()) * sigma_level).reshape(n, T)

        h_plus_list, h_cross_list = [], []
        for i in range(n):
            hp, hc, hp_u, hc_u = self._compute_polarizations_with_uncertainty(
                amp_matrix[i], phase_matrix[i],
                amp_std_matrix[i], phase_std_matrix[i], parameters[i, 4]
            )
            h_plus_list.append(WaveformPrediction(hp, hp_u, self.time_norm, self.delta_t))
            h_cross_list.append(WaveformPrediction(hc, hc_u, self.time_norm, self.delta_t))

        return h_plus_list, h_cross_list
