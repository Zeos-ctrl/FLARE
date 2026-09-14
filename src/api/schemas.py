"""Pydantic request/response models for the FLARE dashboard API."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class SettingsModel(BaseModel):
    """Global system/data settings used to generate training data."""
    waveform: str = "IMRPhenomD"
    # Default: a 4 s @ 4096 Hz fixed physical window (16384 samples). The in-band
    # signal for BBH masses >= ~20 Msun from 20 Hz fits 4 s.
    waveform_length: int = 16384
    sample_rate: float = 4096.0      # delta_t = 1 / sample_rate
    f_lower: float = 20.0
    num_samples: int = 1000
    # Fixed physical-time window representation (no warp/trim) -- keeps the FULL
    # waveform on a fixed grid with the merger placed by shift, so the reduced-
    # order reconstruction is a faithful physical-time waveform that is directly
    # matched-filterable for parameter estimation. This is the DEFAULT (the
    # warp/trim representation maximises the match metric but is not PE-faithful:
    # it discards the early inspiral and the absolute timescale). Set False only
    # for a pure match-accuracy campaign. Pick waveform_length so the in-band
    # signal fits ``waveform_length / sample_rate`` seconds.
    td_fixed_window: bool = True
    # Optional saved waveform bank (.npz with resized ``strain`` + ``params``);
    # the reduced-order path loads the first ``num_samples`` rows instead of
    # regenerating. Lets one expensive bank be reused across experiments.
    waveform_bank: str = ""
    val_split: float = 0.3
    clean_data: bool = True
    device: str = "cuda"
    feature_names: List[str] = Field(default_factory=lambda: ["chirp_mass", "symmetric_mass_ratio"])
    # Predict the raw strain with a single network instead of amp/phase decomposition.
    direct_strain: bool = False
    # Reduced-order (SVD) training: compress amp/phase to an SVD basis and learn
    # theta -> coefficients (one row per waveform). Much faster than per-point.
    reduced_order: bool = False
    svd_energy: float = 0.999999
    svd_max_rank: int = 64
    svd_min_rank: int = 12
    # Reduced-order phase options: split the phase into scale x shape (needed for
    # long low-mass waveforms), and choose the phase loss -- "coeff" (SVD-coeff
    # MSE, fast) or "curve" (amplitude-weighted phase error in waveform space,
    # higher match on the hard low-mass tail).
    phase_scale_factor: bool = True
    # "coeff" | "curve" | "match". "match" fine-tunes the phase net on the actual
    # differentiable match (raw-phase mode only; needs a waveform_bank).
    rom_phase_loss: str = "coeff"
    rom_match_epochs: int = 400
    rom_match_lr: float = 5e-5
    rom_curve_amp_pow: float = 2.0
    rom_curve_weight: str = "amp"          # "amp" | "freq" (per-cycle weighting)
    # Head architecture, by name (src/models/rom_heads.py): "amp_phase" (2 heads),
    # "amp_shape_scale_2h" (2 heads, scale-factored), "amp_shape_scale" (3 heads).
    rom_heads: str = ""
    # Give the precision-critical phase-scale its own network (usually via rom_heads).
    rom_separate_scale: bool = False
    # Parameter-space sampling ranges. mass_min 20 keeps every waveform's in-band
    # inspiral inside the default 4 s window (lower masses need a longer window).
    mass_min: float = 20.0
    mass_max: float = 100.0
    spin_min: float = -0.7
    spin_max: float = 0.7
    incl_min: float = 0.0
    incl_max: float = 3.141592653589793
    ecc_min: float = 0.0
    ecc_max: float = 0.3


class ModelDesignModel(BaseModel):
    """A named neural-network architecture design."""
    name: str = "default"
    # "builtin" uses the parametric MLP below; "custom" loads operator-authored
    # model files from custom_models/ (referenced here by file stem, no .py).
    model_kind: str = "builtin"
    amp_module: str = ""
    phase_module: str = ""
    amp_hidden_size: int = 256
    amp_layers: int = 3
    amp_banks: int = 3
    amp_dropout: float = 0.15
    amp_lr: float = 0.0005
    amp_weight_decay: float = 0.1
    phase_hidden_size: int = 256
    phase_layers: int = 4
    phase_banks: int = 6
    phase_dropout: float = 0.15
    phase_lr: float = 0.0005
    phase_weight_decay: float = 0.1
    fourier_bands: int = 16
    fourier_max_freq: float = 10.0
    fourier_learnable: bool = False
    batch_size: int = 32768
    num_epochs: int = 200
    patience: int = 50
    hpo_trials: int = 50
    hpo_samples: int = 1000


class TrainRequest(BaseModel):
    project_name: str
    model_name: str = "default"


class TuneRequest(BaseModel):
    project_name: str
    model_name: str = "default"
    model_type: str = "both"
    n_trials: Optional[int] = None


class EvaluateRequest(BaseModel):
    project_name: str
    n_samples: int = 1000
    batch_size: int = 32
    device: str = "cuda"


class DatasetPreviewRequest(BaseModel):
    """Preview generated waveforms/parameters without launching a full run."""
    waveform: str = "IMRPhenomD"
    n_samples: int = 8
    waveform_length: int = 16384
    f_lower: float = 20.0
    sample_rate: float = 4096.0
    td_fixed_window: bool = True     # preview on the same fixed physical window as training
    mass_range: Tuple[float, float] = (20.0, 100.0)
    spin_range: Tuple[float, float] = (-0.7, 0.7)
    incl_range: Tuple[float, float] = (0.0, 3.141592653589793)
    ecc_range: Tuple[float, float] = (0.0, 0.3)


class EstimateRequest(BaseModel):
    project_name: str
    event: str = "GW150914"
    detectors: List[str] = Field(default_factory=lambda: ["H1", "L1"])
    nwalkers: int = 32
    nsteps: int = 2000
    bank_size: int = 2000
    duration: float = 8.0
    free_params: Optional[Dict[str, Tuple[float, float]]] = None
    fixed_params: Optional[Dict[str, float]] = None
    device: str = "cuda"
    # Likelihood: "matched_filter" (detection statistic) or "gaussian" (proper
    # Gaussian likelihood with a sampled distance -- unbiased masses for loud
    # short signals; use with a fixed-window/physical-time model). Off-source PSD
    # avoids the on-source-signal bias in the noise estimate.
    likelihood: str = "matched_filter"
    offsource_psd: bool = False
    distance_min: float = 50.0
    distance_max: float = 3000.0
    snr_window_s: float = 0.1


class JobRef(BaseModel):
    id: str
    type: str
    status: str


# --- Experiments (agent-driven automated runs) --------------------------
class ExperimentSpec(BaseModel):
    """One automated experiment: what to train, how to score it.

    Reference a saved design by ``model_name`` OR supply an inline
    ``model_design`` (so an agent can vary architecture without persisting it).
    ``settings_overrides`` is a partial patch over the saved global settings
    (e.g. ``{"num_samples": 20, "device": "cpu"}``).
    """
    name: Optional[str] = None                       # auto-generated if omitted
    model_name: Optional[str] = None
    model_design: Optional[ModelDesignModel] = None
    settings_overrides: Dict[str, object] = Field(default_factory=dict)
    evaluate: bool = True
    eval_n_samples: int = 500
    group: Optional[str] = None                      # sweep / campaign label
    tags: List[str] = Field(default_factory=list)
    source: str = "agent"                            # agent | ui | searcher
    notes: Optional[str] = None


class ExperimentRequest(ExperimentSpec):
    """Body for POST /api/experiments (identical to the spec)."""


class AutoRunConfig(BaseModel):
    """Body for POST /api/agent/auto — one autonomous train->eval search campaign."""
    group: Optional[str] = None                      # auto-generated if omitted
    budget: int = 5                                  # number of experiments to run
    proposer: str = "local"                          # local | claude | random
    objective: Optional[str] = None                  # free-text steering, e.g. "focus on the phase network"
    local_base_url: Optional[str] = None             # OpenAI-compatible server URL (proposer=local)
    local_model: Optional[str] = None                # model id on that server
    local_api_key: Optional[str] = None               # bearer key, if the server requires one
    num_epochs: int = 200                            # training budget per run
    eval_n_samples: int = 500
    evaluate: bool = True
    settings_overrides: Dict[str, object] = Field(default_factory=dict)
    notes: Optional[str] = None


class ExperimentRecord(BaseModel):
    """Durable, on-disk record of an experiment and its outcome."""
    id: str
    name: str
    status: str = "pending"                          # + "interrupted" (reconciled)
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    group: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source: str = "agent"
    job_id: Optional[str] = None
    project_name: str = ""
    spec: ExperimentSpec
    progress: Dict[str, object] = Field(default_factory=dict)
    metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    error: Optional[str] = None
