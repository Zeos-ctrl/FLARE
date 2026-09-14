# src/core/trainer.py
import copy
import logging
import json
import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Dict, Optional

from src.core.config import ProjectConfig
from src.data.config import DEVICE
from src.data.generator import WaveformGenerator, WaveformDataset
from src.data.features import FeatureExtractor
from src.utils.io import save_checkpoint
from src.utils.uncertainty import compute_last_layer_hessian

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        # Honour the requested device but fall back to CPU when CUDA is absent.
        self.device = DEVICE if config.device.startswith("cuda") else torch.device(config.device)
        self.checkpoint_path = config.project_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # Per-model optimizers, cached so repeated _train_epoch calls (e.g.
        # from the tuner) reuse optimizer state across epochs.
        self._optimizers: Dict[int, torch.optim.Optimizer] = {}
        
        # Initialize data generator
        self.generator = WaveformGenerator(
            waveform_length=config.waveform_length,
            delta_t=config.delta_t,
            f_lower=config.f_lower,
            approximant=config.waveform,
            fixed_window=config.td_fixed_window,
        )
        
    def prepare_data(self, n_samples: Optional[int] = None) -> WaveformDataset:
        """Generate or load training data"""
        n_samples = n_samples or self.config.num_samples

        if self.config.reduced_order:
            return self._prepare_reduced_data(n_samples)

        # Check for cached dataset
        cache_path = os.path.join(self.checkpoint_path, 'dataset.npz')
        if os.path.exists(cache_path) and not self.config.force_regenerate:
            logger.info(f"Loading cached dataset from {cache_path}")
            return self._load_cached_dataset(cache_path)

        # Generate new dataset
        logger.info(f"Generating {n_samples} samples with {self.config.waveform}")
        dataset = self.generator.generate_dataset(
            n_samples=n_samples,
            clean=self.config.clean_data,
            feature_names=self.config.feature_names,
            sampling_ranges=self.config.sampling_ranges,
            direct_strain=self.config.direct_strain
        )

        # Cache dataset
        self._save_dataset(dataset, cache_path)
        return dataset

    def _prepare_reduced_data(self, n_samples: int):
        """Generate (or load cached) reduced-order SVD dataset."""
        cache_path = os.path.join(self.checkpoint_path, 'reduced_dataset.npz')
        if os.path.exists(cache_path) and not self.config.force_regenerate:
            logger.info(f"Loading cached reduced dataset from {cache_path}")
            return self._load_reduced_dataset(cache_path)

        logger.info(f"Generating {n_samples} reduced-order samples with "
                    f"{self.config.waveform}")
        dataset = self.generator.generate_reduced_dataset(
            n_samples=n_samples,
            clean=self.config.clean_data,
            feature_names=self.config.feature_names,
            sampling_ranges=self.config.sampling_ranges,
            energy=self.config.svd_energy,
            max_rank=self.config.svd_max_rank,
            min_rank=self.config.svd_min_rank,
            scale_factor=self.config.phase_scale_factor,
            scale_weight=self.config.phase_scale_weight,
            scale_window=(self.config.phase_scale_lo, self.config.phase_scale_hi),
            bank_path=self.config.waveform_bank,
        )
        self._save_reduced_dataset(dataset, cache_path)
        return dataset
    
    def create_dataloaders(self, dataset: WaveformDataset) -> Dict:
        """Create train/val dataloaders from dataset"""
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        
        # Convert to tensors
        X = torch.from_numpy(dataset.inputs).float().to(self.device)
        A = torch.from_numpy(dataset.targets_amplitude).float().to(self.device)
        phi = torch.from_numpy(dataset.targets_phase).float().to(self.device)
        
        # Split indices
        indices = np.arange(len(X))
        train_idx, val_idx = train_test_split(
            indices, test_size=self.config.val_split,
            random_state=42, shuffle=True
        )
        
        # Create datasets
        loaders = {
            'amp': {
                'train': DataLoader(
                    TensorDataset(X[train_idx], A[train_idx]),
                    batch_size=self.config.batch_size, shuffle=True
                ),
                'val': DataLoader(
                    TensorDataset(X[val_idx], A[val_idx]),
                    batch_size=self.config.batch_size, shuffle=False
                )
            },
            'phase': {
                'train': DataLoader(
                    TensorDataset(X[train_idx], phi[train_idx]),
                    batch_size=self.config.batch_size, shuffle=True
                ),
                'val': DataLoader(
                    TensorDataset(X[val_idx], phi[val_idx]),
                    batch_size=self.config.batch_size, shuffle=False
                )
            }
        }
        return loaders
    
    def run_training(self, data: Optional[WaveformDataset] = None, on_epoch_end=None,
                     amp_model: Optional[nn.Module] = None,
                     phase_model: Optional[nn.Module] = None):
        """Complete training pipeline.

        Args:
            data: Optional pre-generated dataset; generated/loaded if omitted.
            on_epoch_end: Optional per-epoch callback forwarded to
                :meth:`train_model` for live progress streaming.
            amp_model, phase_model: Optional pre-built networks. When supplied
                (e.g. from the CLI), they are trained as-is and no model code is
                constructed here. When omitted, models are built from the config
                (built-in MLP, or an operator file named by ``config.*_module``).
        """
        from src.models.model_factory import make_amp_model, make_phase_model

        # Prepare data
        dataset = data or self.prepare_data()

        if self.config.reduced_order:
            return self._run_training_reduced(dataset, on_epoch_end=on_epoch_end)

        loaders = self.create_dataloaders(dataset)

        # Create models
        feature_dim = len(self.config.feature_names)

        # Load best hyperparameters if they exist
        amp_params = self._load_best_params('amp')
        phase_params = self._load_best_params('phase')

        if amp_model is None:
            amp_model = make_amp_model(feature_dim, amp_params)
        if phase_model is None:
            phase_model = make_phase_model(feature_dim, phase_params)
        amp_model = amp_model.to(self.device)
        phase_model = phase_model.to(self.device)

        # Train models
        logger.info("Training amplitude model...")
        amp_model = self.train_model(amp_model, loaders['amp'], 'amp',
                                     on_epoch_end=on_epoch_end)

        logger.info("Training phase model...")
        phase_model = self.train_model(phase_model, loaders['phase'], 'phase',
                                       on_epoch_end=on_epoch_end)
        
        # Compute uncertainty estimates
        logger.info("Computing uncertainty estimates...")
        amp_hessian = compute_last_layer_hessian(amp_model, loaders['amp']['train'], self.device)
        phase_hessian = compute_last_layer_hessian(phase_model, loaders['phase']['train'], self.device)
        
        # Save complete checkpoint
        self._save_final_checkpoint(
            amp_model, phase_model, dataset,
            amp_hessian, phase_hessian
        )
        
        logger.info(f"Training complete! Models saved to {self.checkpoint_path}")
        return amp_model, phase_model

    # ------------------------------------------------------------------
    # Reduced-order (SVD) training path
    # ------------------------------------------------------------------
    def _run_training_reduced(self, dataset, on_epoch_end=None):
        """Train small ``theta -> SVD coefficients`` networks (one row/waveform).

        Reconstruction at inference is ``amp = amp_basis @ amp_coeffs`` and
        ``phase = phase_basis @ phase_coeffs`` (see the predictor). Orders of
        magnitude faster than the per-(sample, time) path.
        """
        from src.models.rom import CoeffMLP
        from sklearn.model_selection import train_test_split

        d = dataset.theta.shape[1]
        k_a = dataset.amp_coeffs.shape[1]
        k_p = dataset.phase_coeffs.shape[1]
        hidden = self.config.amp_hidden_layers[0]
        depth = len(self.config.amp_hidden_layers)

        theta = torch.from_numpy(dataset.theta).float().to(self.device)
        amp_c = torch.from_numpy(dataset.amp_coeffs).float().to(self.device)
        phase_c = torch.from_numpy(dataset.phase_coeffs).float().to(self.device)

        idx = np.arange(len(theta))
        tr, va = train_test_split(idx, test_size=self.config.val_split,
                                  random_state=42, shuffle=True)
        tr = torch.from_numpy(tr).to(self.device)
        va = torch.from_numpy(va).to(self.device)

        def make(out_dim):
            return CoeffMLP(d, out_dim, hidden=hidden, depth=depth,
                            fourier_bands=self.config.fourier_bands,
                            fourier_max_freq=self.config.fourier_max_freq).to(self.device)

        amp_model = self._train_coeff_model(make(k_a), theta, amp_c, tr, va,
                                            'amp', self.config.amp_lr,
                                            self.config.amp_weight_decay, on_epoch_end)

        # Dedicated scale network: split the phase coeff vector [shape..., log_scale]
        # so the precision-critical log-scale gets its own net, and the shape net
        # only predicts the k_shape shape coeffs.
        separate = (self.config.rom_separate_scale
                    and dataset.metadata.get("phase_scale_factor"))
        scale_model = None
        if separate:
            k_shape = k_p - 1
            scale_model = self._train_coeff_model(
                make(1), theta, phase_c[:, k_shape:], tr, va, 'scale',
                self.config.phase_lr, self.config.phase_weight_decay, on_epoch_end)
            shape_c = phase_c[:, :k_shape]
            if self.config.rom_phase_loss == "curve":
                phase_model = self._train_phase_curve(
                    make(k_shape), theta, phase_c, amp_c, tr, va, dataset,
                    on_epoch_end, shape_only=True)
            else:
                phase_model = self._train_coeff_model(
                    make(k_shape), theta, shape_c, tr, va, 'phase',
                    self.config.phase_lr, self.config.phase_weight_decay, on_epoch_end)
        elif self.config.rom_phase_loss == "curve":
            phase_model = self._train_phase_curve(
                make(k_p), theta, phase_c, amp_c, tr, va, dataset, on_epoch_end)
        elif self.config.rom_phase_loss == "match":
            # MSE warm-start to a good basin (the match loss is non-convex; a
            # cold start on it stalls), then fine-tune the phase net on the
            # ACTUAL match against the real bank waveform -- optimising the metric
            # itself rather than an MSE/curve proxy. Requires raw-phase mode
            # (no scale-factoring) and a waveform_bank to supply match targets.
            phase_model = self._train_coeff_model(
                make(k_p), theta, phase_c, tr, va, 'phase', self.config.phase_lr,
                self.config.phase_weight_decay, on_epoch_end)
            phase_model = self._finetune_phase_match(
                phase_model, amp_model, theta, tr, va, dataset, d, on_epoch_end)
        else:
            phase_model = self._train_coeff_model(make(k_p), theta, phase_c, tr, va,
                                                  'phase', self.config.phase_lr,
                                                  self.config.phase_weight_decay, on_epoch_end)

        self._save_reduced_checkpoint(amp_model, phase_model, dataset,
                                      d, k_a, k_p, hidden, depth, scale_model=scale_model)
        logger.info(f"Reduced-order training complete! Saved to {self.checkpoint_path}")
        return amp_model, phase_model

    def _train_coeff_model(self, model, theta, coeffs, tr, va, model_type,
                           lr, weight_decay, on_epoch_end=None):
        """Mini-batch train a theta->coeffs network (cosine LR) with early stop.

        Mini-batching + cosine annealing reaches the far higher coefficient
        precision the scale-factored phase needs (the log-scale must be accurate
        to ~1e-4 relative); plain full-batch Adam plateaus well short of it.
        """
        import copy
        from torch.optim.lr_scheduler import CosineAnnealingLR

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = CosineAnnealingLR(opt, self.config.num_epochs)
        criterion = nn.MSELoss()
        best_val, best_state, patience = float('inf'), None, 0
        n_tr = len(tr)
        bs = min(4096, n_tr)

        for epoch in range(self.config.num_epochs):
            model.train()
            perm = tr[torch.randperm(n_tr, device=tr.device)]
            for k in range(0, n_tr, bs):
                b = perm[k:k + bs]
                opt.zero_grad()
                loss = criterion(model(theta[b]), coeffs[b])
                loss.backward()
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(theta[va]), coeffs[va]).item()

            if val_loss < best_val - 1e-9:
                best_val, best_state, patience = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1

            if on_epoch_end is not None:
                on_epoch_end({
                    'model_type': model_type, 'epoch': epoch,
                    'total_epochs': self.config.num_epochs,
                    'val_loss': float(val_loss), 'best_val_loss': float(best_val),
                    'lr': float(opt.param_groups[0]['lr']), 'patience_counter': patience,
                })
            if patience >= self.config.patience:
                logger.info(f"Early stopping {model_type} coeff net at epoch {epoch} "
                            f"(best val {best_val:.3e})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _train_phase_curve(self, model, theta, phase_c, amp_c, tr, va, dataset,
                           on_epoch_end=None, shape_only=False):
        """Train the phase coeff net with an amplitude-weighted phase loss in
        waveform space (reconstruct phase from the predicted coeffs, penalise
        error where the amplitude is loud) plus a light coeff-MSE anchor. This is
        the reduced-order form of the offline 'curve loss' that lifts the hard
        low-mass tail past what plain coeff-MSE reaches.

        ``shape_only`` (dedicated-scale mode): the net predicts only the k_shape
        shape coeffs, and the phase is reconstructed with the TRUE log-scale (the
        scale is fit by its own dedicated net) -- this decouples the two so each
        gets a clean objective."""
        import copy
        from torch.optim.lr_scheduler import CosineAnnealingLR

        dev = self.device
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
        a_basis = t(dataset.amp_basis); p_basis = t(dataset.phase_basis)
        a_mean = t(dataset.amp_coeff_mean); a_std = t(dataset.amp_coeff_std)
        p_mean = t(dataset.phase_coeff_mean); p_std = t(dataset.phase_coeff_std)
        scale_factor = bool(dataset.metadata.get("phase_scale_factor"))
        pow_ = self.config.rom_curve_amp_pow
        k_shape = p_basis.shape[1]

        def phase_from(coeff_std):                       # (B, k_p) standardised
            c = coeff_std * p_std + p_mean
            if scale_factor:
                return torch.exp(c[:, -1:]) * (c[:, :-1] @ p_basis.T)
            return c @ p_basis.T

        def phase_from_shape(shape_std, log_scale_true_std):
            shape = shape_std * p_std[:k_shape] + p_mean[:k_shape]
            log_scale = log_scale_true_std * p_std[-1] + p_mean[-1]
            return torch.exp(log_scale[:, None]) * (shape @ p_basis.T)

        def amp_from(coeff_std):
            return torch.clamp((coeff_std * a_std + a_mean) @ a_basis.T, min=0)

        opt = torch.optim.AdamW(model.parameters(), lr=self.config.phase_lr,
                                weight_decay=self.config.phase_weight_decay)
        sched = CosineAnnealingLR(opt, self.config.num_epochs)
        best_val, best_state, patience = float('inf'), None, 0
        n_tr = len(tr); bs = min(4096, n_tr)

        use_freq = self.config.rom_curve_weight == "freq"

        def curve_loss(idx):
            out = model(theta[idx])
            ph_true = phase_from(phase_c[idx])
            if shape_only:
                ph_pred = phase_from_shape(out, phase_c[idx][:, -1])
                reg = ((out - phase_c[idx][:, :k_shape]) ** 2).mean(-1) * 0.02
            else:
                ph_pred = phase_from(out)
                reg = ((out - phase_c[idx]) ** 2).mean(-1) * 0.02
            amp_w = amp_from(amp_c[idx])
            if use_freq:
                # per-cycle weight: instantaneous frequency |dphi/dt|, gated to the
                # signal-bearing region (each GW cycle counts ~equally to the match).
                gate = (amp_w > 1e-2 * amp_w.amax(-1, keepdim=True)).float()
                dphi = torch.diff(ph_true, dim=-1)
                dphi = torch.cat([dphi[:, :1], dphi], dim=-1)
                w = dphi.abs() * gate
            else:
                w = amp_w ** pow_
            d = ph_pred - ph_true
            d = d - (w * d).sum(-1, keepdim=True) / (w.sum(-1, keepdim=True) + 1e-30)
            lp = (w * d * d).sum(-1) / (w.sum(-1) + 1e-30)
            return (lp + reg).mean()

        for epoch in range(self.config.num_epochs):
            model.train()
            perm = tr[torch.randperm(n_tr, device=tr.device)]
            for k in range(0, n_tr, bs):
                b = perm[k:k + bs]
                opt.zero_grad(); loss = curve_loss(b); loss.backward(); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                val_loss = curve_loss(va).item()
            if val_loss < best_val - 1e-12:
                best_val, best_state, patience = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
            if on_epoch_end is not None:
                on_epoch_end({'model_type': 'phase', 'epoch': epoch,
                              'total_epochs': self.config.num_epochs,
                              'val_loss': float(val_loss), 'best_val_loss': float(best_val),
                              'lr': float(opt.param_groups[0]['lr']), 'patience_counter': patience})
            if patience >= self.config.patience:
                logger.info(f"Early stopping phase curve net at epoch {epoch} (best {best_val:.3e})")
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _finetune_phase_match(self, phase_model, amp_model, theta, tr, va,
                              dataset, d, on_epoch_end=None):
        """Fine-tune the phase coeff net on ``1 - match`` against the real bank
        waveform (differentiable match, ``src/utils/metrics.TorchMatch``).

        Every MSE/curve loss is a proxy: coeff MSE == waveform phase MSE
        (Parseval), which is not what the match measures, and the amplitude/freq
        curve weightings trade the loud merger against the quiet inspiral. The
        match itself is differentiable, so optimise it directly. Only valid for
        raw-phase mode (``phase = coeffs @ phase_basis.T``); scale-factored and
        dedicated-scale checkpoints keep their curve/coeff losses.

        The match target is the original bank strain, loaded in the SAME row
        order as the reduced dataset (``_load_bank`` returns the first N rows, so
        rows align with ``dataset.parameters``). Matching against the true strain
        (not the rank-truncated oracle reconstruction) is what lets the trained
        match exceed the L2 oracle -- the oracle is optimal for MSE, not match.
        """
        import copy
        from torch.optim.lr_scheduler import CosineAnnealingLR

        from src.utils.metrics import TorchMatch

        if dataset.metadata.get("phase_scale_factor"):
            raise ValueError("rom_phase_loss='match' requires raw-phase mode "
                             "(set rom_heads='amp_phase'); scale-factoring is incompatible.")
        if not self.config.waveform_bank:
            raise ValueError("rom_phase_loss='match' needs a waveform_bank for match targets.")

        dev = self.device
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
        a_basis = t(dataset.amp_basis); p_basis = t(dataset.phase_basis)
        a_mean = t(dataset.amp_coeff_mean); a_std = t(dataset.amp_coeff_std)
        p_mean = t(dataset.phase_coeff_mean); p_std = t(dataset.phase_coeff_std)
        amp_scale = float(dataset.amplitude_scale)

        # Bank strain in dataset-row order; match target.
        params, waveforms = self.generator._load_bank(
            self.config.waveform_bank, self.config.num_samples)
        strain = t(np.asarray(waveforms, dtype=np.float32))
        L = strain.shape[1]
        matcher = TorchMatch(L, self.config.delta_t, self.config.f_lower, device=dev)

        amp_model.eval()
        for p in amp_model.parameters():
            p.requires_grad_(False)

        def match_of(idx):
            with torch.no_grad():
                amp = torch.clamp((amp_model(theta[idx]) * a_std + a_mean) @ a_basis.T,
                                  min=0) * amp_scale
            phase = (phase_model(theta[idx]) * p_std + p_mean) @ p_basis.T
            return matcher(amp * torch.cos(phase), strain[idx])

        epochs = int(self.config.rom_match_epochs)
        lr = float(self.config.rom_match_lr)
        opt = torch.optim.AdamW(phase_model.parameters(), lr=lr,
                                weight_decay=self.config.phase_weight_decay)
        sched = CosineAnnealingLR(opt, epochs)
        bs = min(256, len(tr))
        best_val, best_state, patience = -1.0, None, 0

        for epoch in range(epochs):
            phase_model.train()
            perm = tr[torch.randperm(len(tr), device=tr.device)]
            for k in range(0, len(perm), bs):
                b = perm[k:k + bs]
                opt.zero_grad()
                (1.0 - match_of(b).mean()).backward()
                torch.nn.utils.clip_grad_norm_(phase_model.parameters(), 1.0)
                opt.step()
            sched.step()
            phase_model.eval()
            with torch.no_grad():
                val_match = float(np.mean([match_of(va[k:k + bs]).mean().item()
                                           for k in range(0, len(va), bs)]))
            if val_match > best_val:
                best_val, best_state, patience = val_match, copy.deepcopy(phase_model.state_dict()), 0
            else:
                patience += 1
            if on_epoch_end is not None:
                on_epoch_end({'model_type': 'phase', 'epoch': epoch,
                              'total_epochs': epochs, 'val_loss': float(1.0 - val_match),
                              'best_val_loss': float(1.0 - best_val),
                              'lr': float(opt.param_groups[0]['lr']),
                              'patience_counter': patience, 'phase': 'match_finetune'})
            if patience >= self.config.patience:
                logger.info(f"Early stopping phase match fine-tune at epoch {epoch} "
                            f"(best match {best_val:.4f})")
                break

        if best_state is not None:
            phase_model.load_state_dict(best_state)
        logger.info(f"Phase match fine-tune done (val match {best_val:.4f})")
        return phase_model

    def _save_reduced_checkpoint(self, amp_model, phase_model, dataset,
                                 d, k_a, k_p, hidden, depth, scale_model=None):
        """Persist coeff nets, SVD bases, coeff norm-stats and metadata."""
        models = {'amp': amp_model, 'phase': phase_model}
        # dedicated-scale mode: the phase net predicts only the k_shape shape
        # coeffs; a separate scale net predicts the single log-scale.
        k_phase_out = (k_p - 1) if scale_model is not None else k_p
        if scale_model is not None:
            models['scale'] = scale_model
        data_arrays = {
            'param_means': dataset.parameter_means,
            'param_stds': dataset.parameter_stds,
            't_norm_array': dataset.time_array,
            'amp_basis': dataset.amp_basis,
            'phase_basis': dataset.phase_basis,
            'amp_coeff_mean': dataset.amp_coeff_mean,
            'amp_coeff_std': dataset.amp_coeff_std,
            'phase_coeff_mean': dataset.phase_coeff_mean,
            'phase_coeff_std': dataset.phase_coeff_std,
        }
        metadata = {
            'waveform': self.config.waveform,
            'waveform_length': self.config.waveform_length,
            'delta_t': self.config.delta_t,
            'amp_scale': float(dataset.amplitude_scale),
            'feature_names': self.config.feature_names,
            'train_samples': self.config.num_samples,
            'project_name': self.config.project_name,
            'reduced_order': True,
            'amp_rank': k_a, 'phase_rank': k_p,
            'phase_scale_factor': bool(self.config.phase_scale_factor),
            'separate_scale': scale_model is not None,
            'td_fixed_window': bool(self.config.td_fixed_window),
            'merger_frac': float(dataset.metadata.get('merger_frac', 0.9)),
            'f_lower': float(self.config.f_lower),
        }
        save_checkpoint(self.checkpoint_path, models, data_arrays, metadata)

        # Hyperparameters the predictor needs to rebuild the coeff nets.
        common = {'reduced_order': True, 'in_param_dim': d, 'hidden': hidden,
                  'depth': depth, 'fourier_bands': self.config.fourier_bands,
                  'fourier_max_freq': self.config.fourier_max_freq}
        self._save_params('amp', {**common, 'out_dim': k_a})
        self._save_params('phase', {**common, 'out_dim': k_phase_out})
        if scale_model is not None:
            self._save_params('scale', {**common, 'out_dim': 1})
        self.config.save()

    def _save_reduced_dataset(self, dataset, cache_path):
        np.savez(cache_path,
                 theta=dataset.theta, amp_coeffs=dataset.amp_coeffs,
                 phase_coeffs=dataset.phase_coeffs, amp_basis=dataset.amp_basis,
                 phase_basis=dataset.phase_basis,
                 amp_coeff_mean=dataset.amp_coeff_mean, amp_coeff_std=dataset.amp_coeff_std,
                 phase_coeff_mean=dataset.phase_coeff_mean, phase_coeff_std=dataset.phase_coeff_std,
                 amplitude_scale=np.array(dataset.amplitude_scale),
                 parameters=dataset.parameters, time_array=dataset.time_array,
                 parameter_means=dataset.parameter_means, parameter_stds=dataset.parameter_stds,
                 metadata=json.dumps(dataset.metadata))

    def _load_reduced_dataset(self, cache_path):
        from src.data.generator import ReducedOrderDataset
        d = np.load(cache_path, allow_pickle=False)
        return ReducedOrderDataset(
            theta=d['theta'], amp_coeffs=d['amp_coeffs'], phase_coeffs=d['phase_coeffs'],
            amp_basis=d['amp_basis'], phase_basis=d['phase_basis'],
            amp_coeff_mean=d['amp_coeff_mean'], amp_coeff_std=d['amp_coeff_std'],
            phase_coeff_mean=d['phase_coeff_mean'], phase_coeff_std=d['phase_coeff_std'],
            amplitude_scale=float(d['amplitude_scale']),
            parameters=d['parameters'], time_array=d['time_array'],
            parameter_means=d['parameter_means'], parameter_stds=d['parameter_stds'],
            metadata=json.loads(str(d['metadata'])),
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _make_optimizer(self, model: nn.Module, model_type: str) -> torch.optim.Optimizer:
        if model_type == 'amp':
            lr, wd = self.config.amp_lr, self.config.amp_weight_decay
        else:
            lr, wd = self.config.phase_lr, self.config.phase_weight_decay
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    def _get_optimizer(self, model: nn.Module, model_type: str) -> torch.optim.Optimizer:
        key = id(model)
        if key not in self._optimizers:
            self._optimizers[key] = self._make_optimizer(model, model_type)
        return self._optimizers[key]

    def _train_epoch(self, model: nn.Module, loaders: Dict, model_type: str) -> float:
        """Run one training epoch and return the mean validation loss.

        Shared by the full training loop and the HPO objective in the tuner.
        The optimizer is cached per model so state persists across calls.
        """
        criterion = nn.MSELoss()
        optimizer = self._get_optimizer(model, model_type)
        clip = self.config.amp_clip if model_type == 'amp' else self.config.phase_clip

        model.train()
        for X, Y in loaders['train']:
            optimizer.zero_grad()
            pred = model(X[:, :1], X[:, 1:])
            loss = criterion(pred, Y)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for X, Y in loaders['val']:
                pred = model(X[:, :1], X[:, 1:])
                total += criterion(pred, Y).item() * len(X)
                count += len(X)
        return total / max(count, 1)

    def train_model(self, model: nn.Module, loaders: Dict, model_type: str,
                    on_epoch_end=None) -> nn.Module:
        """Train a single model with early stopping and LR scheduling.

        Args:
            model: Amplitude or phase network.
            loaders: Dict with 'train' and 'val' DataLoaders.
            model_type: 'amp' or 'phase'.
            on_epoch_end: Optional callback ``fn(dict)`` invoked after every
                epoch with progress metrics. Used by the dashboard/API to
                stream live training curves.
        """
        optimizer = self._get_optimizer(model, model_type)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      patience=10, min_lr=1e-6)

        best_val = float('inf')
        best_state = None
        patience_counter = 0
        min_delta = 1e-6

        pbar = tqdm(range(self.config.num_epochs), desc=f"train {model_type}")
        for epoch in pbar:
            val_loss = self._train_epoch(model, loaders, model_type)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            improved = val_loss < best_val - min_delta
            if improved:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            pbar.set_postfix(val_loss=f"{val_loss:.3e}", lr=f"{current_lr:.1e}")

            if on_epoch_end is not None:
                on_epoch_end({
                    'model_type': model_type,
                    'epoch': epoch,
                    'total_epochs': self.config.num_epochs,
                    'val_loss': float(val_loss),
                    'best_val_loss': float(best_val),
                    'lr': float(current_lr),
                    'patience_counter': patience_counter,
                })

            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping {model_type} at epoch {epoch} "
                            f"(best val loss {best_val:.3e})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    # ------------------------------------------------------------------
    # Dataset caching
    # ------------------------------------------------------------------
    def _save_dataset(self, dataset: WaveformDataset, cache_path: str):
        """Persist a generated dataset to a compressed .npz cache."""
        np.savez(
            cache_path,
            inputs=dataset.inputs,
            targets_amplitude=dataset.targets_amplitude,
            targets_phase=dataset.targets_phase,
            parameters=dataset.parameters,
            time_array=dataset.time_array,
            amplitude_scale=np.array(dataset.amplitude_scale),
            parameter_means=dataset.parameter_means,
            parameter_stds=dataset.parameter_stds,
            metadata=json.dumps(dataset.metadata),
        )
        logger.info(f"Cached dataset to {cache_path}")

    def _load_cached_dataset(self, cache_path: str) -> WaveformDataset:
        """Load a dataset previously written by :meth:`_save_dataset`."""
        d = np.load(cache_path, allow_pickle=False)
        return WaveformDataset(
            inputs=d['inputs'],
            targets_amplitude=d['targets_amplitude'],
            targets_phase=d['targets_phase'],
            parameters=d['parameters'],
            time_array=d['time_array'],
            amplitude_scale=float(d['amplitude_scale']),
            parameter_means=d['parameter_means'],
            parameter_stds=d['parameter_stds'],
            metadata=json.loads(str(d['metadata'])),
        )
    
    def _save_final_checkpoint(self, amp_model, phase_model, dataset,
                               amp_hessian, phase_hessian):
        """Save all components needed for inference"""
        
        models = {
            'amp': amp_model,
            'phase': phase_model
        }
        
        data_arrays = {
            'param_means': dataset.parameter_means,
            'param_stds': dataset.parameter_stds,
            't_norm_array': dataset.time_array,
            'amp_last_weight_variances': amp_hessian['weight_var'].cpu().numpy(),
            'amp_last_bias_variance': amp_hessian['bias_var'].cpu().numpy(),
            'phase_last_weight_variances': phase_hessian['weight_var'].cpu().numpy(),
            'phase_last_bias_variance': phase_hessian['bias_var'].cpu().numpy()
        }
        
        metadata = {
            'waveform': self.config.waveform,
            'waveform_length': self.config.waveform_length,
            'delta_t': self.config.delta_t,
            'amp_scale': float(dataset.amplitude_scale),
            'feature_names': self.config.feature_names,
            'train_samples': self.config.num_samples,
            'project_name': self.config.project_name
        }
        
        save_checkpoint(self.checkpoint_path, models, data_arrays, metadata)
        
        # Save hyperparameters
        self._save_params('amp', self._load_best_params('amp'))
        self._save_params('phase', self._load_best_params('phase'))
        
        # Save config
        self.config.save()
    
    def _load_best_params(self, model_type: str) -> dict:
        """Load best hyperparameters or use defaults"""
        param_file = os.path.join(self.checkpoint_path, f'{model_type}_params.json')
        
        fourier = {
            'fourier_bands': self.config.fourier_bands,
            'fourier_max_freq': self.config.fourier_max_freq,
            'fourier_learnable': self.config.fourier_learnable,
        }
        # Custom-model selection travels alongside the hyperparameters so the
        # model factory can pick built-in vs. an operator file in custom_models/.
        custom = {
            'model_kind': self.config.model_kind,
            'amp_module': self.config.amp_module,
            'phase_module': self.config.phase_module,
        }

        if os.path.exists(param_file):
            with open(param_file) as f:
                params = json.load(f)
            # Fill in Fourier/custom settings if a tuned param file predates them.
            return {**fourier, **custom, **params}
        else:
            # Use defaults from config
            if model_type == 'amp':
                return {
                    'amp_hidden_size': self.config.amp_hidden_layers[0],
                    'layers': len(self.config.amp_hidden_layers),
                    'banks': self.config.amp_banks,
                    'dropout': self.config.amp_dropout,
                    **fourier,
                    **custom,
                }
            else:
                return {
                    'phase_hidden_size': self.config.phase_hidden_layers[0],
                    'layers': len(self.config.phase_hidden_layers),
                    'banks': self.config.phase_banks,
                    'dropout': self.config.phase_dropout,
                    **fourier,
                    **custom,
                }
    
    def _save_params(self, model_type: str, params: dict):
        """Save hyperparameters"""
        param_file = os.path.join(self.checkpoint_path, f'{model_type}_params.json')
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=2)
