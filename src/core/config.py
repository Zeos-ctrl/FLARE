# src/core/config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os

@dataclass
class ProjectConfig:
    """Central configuration for a training/tuning project"""
    # Project settings
    project_name: str = "default_project"
    checkpoint_dir: str = "checkpoints"
    
    # Data settings
    num_samples: int = 1000
    waveform: str = "IMRPhenomD"
    waveform_length: int = 2048

    # Fixed physical-time window representation (for matched-filter PE): place the
    # merger at MERGER_FRAC*L on a fixed delta_t grid, no warp/trim (see
    # WaveformGenerator.fixed_window). The reduced-order reconstruction is then a
    # faithful physical-time waveform. Pick waveform_length so the in-band signal
    # for the mass range fits ``waveform_length * delta_t`` seconds.
    td_fixed_window: bool = False

    # Optional path to a saved raw-ish waveform bank (.npz with ``strain`` of shape
    # (M, waveform_length) already merger-aligned/resized, and ``params`` (M, 6)).
    # When set, the reduced-order path LOADS the first ``num_samples`` waveforms
    # from the bank instead of regenerating them -- so an expensive-to-build bank
    # (e.g. 50k IMRPhenomD waveforms) is generated once and reused across every
    # experiment/SVD/loss configuration. Build one with scratchpad/build_bank.py.
    waveform_bank: str = ""
    delta_t: float = 1/2048
    f_lower: float = 20.0
    val_split: float = 0.3
    clean_data: bool = True
    force_regenerate: bool = False

    feature_names: List[str] = field(
        default_factory=lambda: ["chirp_mass", "symmetric_mass_ratio"]
    )

    # When True the dataset target is the signed normalised strain (single
    # network predicts the raw waveform) instead of the amplitude/phase
    # decomposition. Pair with a linear-output model in the amp slot and the
    # ``zero_phase`` model in the phase slot.
    direct_strain: bool = False

    # Reduced-order (SVD) mode: compress amplitude/phase into an SVD basis and
    # train a small theta -> coefficients network (one row per waveform) instead
    # of the per-(sample, time) design matrix. Much faster; see
    # src/data/reduced_basis.py. ``svd_energy`` picks the basis rank by retained
    # spectral energy, capped at ``svd_max_rank``.
    reduced_order: bool = False
    svd_energy: float = 0.999999
    svd_max_rank: int = 64
    svd_min_rank: int = 12

    # Reduced-order phase handling. With ``phase_scale_factor`` the phase is split
    # into a per-waveform scale (total accumulated phase, a smooth ~Mc^-5/3
    # function of the masses) and a near-universal SHAPE, and only the shape is
    # SVD-compressed; the network predicts the shape coeffs plus log-scale.
    # Raw-phase SVD instead makes the dominant coeff *be* the total phase, which
    # then needs ~1e-4 relative accuracy from the net -- unreachable, so raw-phase
    # ROM caps ~0.65 on long low-mass waveforms while scale-factored ROM reaches
    # ~0.96. ``phase_scale_weight`` up-weights the (single, critical) log-scale in
    # the coeff MSE. Scale is anchored on the clean interior window below.
    phase_scale_factor: bool = True
    phase_scale_weight: float = 20.0
    phase_scale_lo: float = 0.1
    phase_scale_hi: float = 0.8

    # Reduced-order phase training loss. "coeff" = plain MSE on the SVD
    # coefficients (fast, ~0.92 on the low-mass benchmark). "curve" reconstructs
    # the phase curve from the predicted coeffs and minimises an amplitude-weighted
    # phase error in *waveform* space -- it penalises phase error where the signal
    # is loud (what the match actually cares about) and lifts the hard low-mass
    # tail (~0.92 -> ~0.96). ``rom_curve_amp_pow`` sets the amplitude weighting
    # exponent.
    # "coeff" = SVD-coeff MSE. "curve" = amplitude/freq-weighted phase error in
    # waveform space. "match" = MSE warm-start then fine-tune the phase net on the
    # ACTUAL differentiable match against the bank waveform (optimises the metric,
    # not a proxy). "match" requires raw-phase mode (rom_heads="amp_phase") and a
    # waveform_bank. Fine-tune budget/LR set by rom_match_epochs / rom_match_lr.
    rom_phase_loss: str = "coeff"          # "coeff" | "curve" | "match"
    rom_match_epochs: int = 400
    rom_match_lr: float = 5e-5
    rom_curve_amp_pow: float = 2.0
    # Curve-loss weighting: "amp" weights phase error by amplitude^amp_pow (focuses
    # on the loud merger); "freq" weights by instantaneous frequency, i.e. per GW
    # cycle (better for the many-cycle low-mass inspiral, where most SNR lives).
    rom_curve_weight: str = "amp"          # "amp" | "freq"

    # Reduced-order HEAD ARCHITECTURE, chosen by name from src/models/rom_heads.py
    # (swappable like custom_models/). Sets ``phase_scale_factor`` and
    # ``rom_separate_scale`` below. Leave "" to use those two flags directly.
    #   amp_phase          -> 2 heads, raw-phase SVD (classic)
    #   amp_shape_scale_2h -> 2 heads, scale-factored, shared [shape, log-scale]
    #   amp_shape_scale    -> 3 heads, scale-factored, dedicated scale net (best)
    rom_heads: str = ""

    # Give the phase-scale (the single precision-critical scalar: total accumulated
    # phase) its OWN dedicated network instead of sharing a head with the shape
    # coefficients (only used when phase_scale_factor). Usually set via rom_heads.
    rom_separate_scale: bool = False

    # Parameter-space sampling ranges
    mass_min: float = 30.0
    mass_max: float = 100.0
    spin_min: float = -0.7
    spin_max: float = 0.7
    incl_min: float = 0.0
    incl_max: float = 3.141592653589793
    ecc_min: float = 0.0
    ecc_max: float = 0.3

    # Model architecture
    # "builtin" instantiates the parametric MLP; "custom" loads an operator
    # file from custom_models/<name>.py that defines
    # build_model(in_param_dim, time_dim) -> nn.Module.
    model_kind: str = "builtin"
    amp_module: str = ""
    phase_module: str = ""
    amp_hidden_layers: List[int] = field(default_factory=lambda: [256, 256, 256])
    amp_banks: int = 3
    phase_hidden_layers: List[int] = field(default_factory=lambda: [256, 256, 256, 256])
    phase_banks: int = 6

    # Fourier-feature encoding (shared by amp/phase builders)
    fourier_bands: int = 16
    fourier_max_freq: float = 10.0
    fourier_learnable: bool = False

    @property
    def sampling_ranges(self) -> Dict[str, tuple]:
        """Bounds dict consumable by WaveformGenerator.generate_dataset."""
        return {
            "mass_range": (self.mass_min, self.mass_max),
            "spin_range": (self.spin_min, self.spin_max),
            "incl_range": (self.incl_min, self.incl_max),
            "ecc_range": (self.ecc_min, self.ecc_max),
        }
    
    # Training settings
    batch_size: int = 32768
    num_epochs: int = 200
    patience: int = 50
    
    # Optimization
    amp_lr: float = 0.0005
    amp_weight_decay: float = 0.1
    amp_dropout: float = 0.15
    amp_clip: float = 3.0
    
    phase_lr: float = 0.0005
    phase_weight_decay: float = 0.1
    phase_dropout: float = 0.15
    phase_clip: float = 4.0
    
    # HPO settings
    hpo_trials: int = 50
    hpo_samples: int = 1000
    
    # Device
    device: str = "cuda"

    def __post_init__(self):
        # A named ROM head scheme (rom_heads) resolves to the low-level flags, so
        # architectures are swappable by name (src/models/rom_heads.py).
        if self.rom_heads:
            from src.models.rom_heads import resolve
            spec = resolve(self.rom_heads)
            self.phase_scale_factor = spec["scale_factor"]
            self.rom_separate_scale = spec["separate_scale"]

    @property
    def project_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.project_name)
    
    def save(self):
        os.makedirs(self.project_path, exist_ok=True)
        with open(os.path.join(self.project_path, "config.json"), "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)
    
    @classmethod
    def load(cls, project_path: str) -> "ProjectConfig":
        with open(os.path.join(project_path, "config.json")) as f:
            return cls(**json.load(f))
