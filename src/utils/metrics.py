import numpy as np
from pycbc.filter import match
from pycbc.types import TimeSeries
from pycbc.psd import aLIGOZeroDetHighPower


class TorchMatch:
    """Differentiable batched match under the aLIGO design PSD (torch).

    The match is ``max_t |z(t)| / sqrt(<h1|h1><h2|h2>)`` with
    ``z = ifft(h1~ conj(h2~) / S(f))``; ``|z|`` maximises over the phase offset
    and ``max_t`` over the time offset -- both degeneracies handled exactly, so
    it can be used directly as a training objective (``1 - match``) instead of an
    MSE proxy. Agrees with :func:`compute_match` to ~1e-10 (see
    ``scratch/matchloss.py::validate``).

    Two numerical traps this guards against: the design PSD is ~1e-46 so ``1/S``
    overflows float32 -> keep the inverse PSD in float64 and rescale it to O(1);
    raw strain ~1e-18 underflows when squared -> normalise each waveform first
    (the match is scale-invariant, so both rescalings cancel).
    """

    def __init__(self, length: int, delta_t: float, f_low: float = 20.0,
                 device: str = "cuda"):
        import torch
        self.length = int(length)
        self.delta_f = 1.0 / (self.length * delta_t)
        n = self.length // 2 + 1
        s = np.asarray(aLIGOZeroDetHighPower(n, self.delta_f, f_low).data, dtype=np.float64)
        inv = np.zeros_like(s)
        good = np.isfinite(s) & (s > 0)
        inv[good] = 1.0 / s[good]
        inv[np.arange(n) * self.delta_f < f_low] = 0.0
        if np.any(inv > 0):
            inv = inv / inv[inv > 0].max()
        self.inv_psd = torch.from_numpy(inv).to(device).double()

    def __call__(self, h1, h2):
        """Match between (B, L) batches ``h1`` and ``h2``; returns (B,)."""
        import torch
        h1 = h1.double(); h2 = h2.double()
        h1 = h1 / (h1.abs().amax(-1, keepdim=True) + 1e-300)
        h2 = h2 / (h2.abs().amax(-1, keepdim=True) + 1e-300)
        A = torch.fft.rfft(h1); B = torch.fft.rfft(h2)
        w = self.inv_psd
        na = (A.real ** 2 + A.imag ** 2) * w
        nb = (B.real ** 2 + B.imag ** 2) * w
        norm = torch.sqrt(na.sum(-1) * nb.sum(-1)) + 1e-300
        prod = A * torch.conj(B) * w
        z = torch.fft.ifft(torch.nn.functional.pad(prod, (0, self.length - prod.shape[-1])),
                           n=self.length, dim=-1)
        peak = (z.real ** 2 + z.imag ** 2).amax(-1).sqrt() * self.length
        return peak / norm


def compute_match(h1: np.ndarray, h2: np.ndarray,
                  delta_t: float = 1/2048, f_low: float = 20.0) -> float:
    """Compute the match between two waveforms"""
    h1_ts = TimeSeries(h1, delta_t=delta_t)
    h2_ts = TimeSeries(h2, delta_t=delta_t)
    
    # Ensure same length
    tlen = max(len(h1_ts), len(h2_ts))
    h1_ts.resize(tlen)
    h2_ts.resize(tlen)
    
    # Generate PSD
    delta_f = 1.0 / h1_ts.duration
    flen = tlen // 2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)
    
    # Compute match
    m, _ = match(h1_ts, h2_ts, psd=psd, low_frequency_cutoff=f_low)
    return float(m)
