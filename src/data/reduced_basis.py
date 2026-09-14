"""SVD reduced-order basis utilities.

Gravitational waveforms (once merger-aligned) live on a very low-dimensional
manifold: the amplitude and phase matrices are highly compressible, so a handful
of SVD basis vectors reconstruct them to >0.99 match. This lets the surrogate
train a small ``theta -> coefficients`` network (N rows) instead of the full
per-(sample, time) design matrix (N*L rows) -- orders of magnitude faster.

A "basis" here is a matrix ``B`` of shape ``(L, k)`` whose columns are the top-k
right singular vectors of the data matrix ``M`` (shape ``(N, L)``). Projection is
``coeffs = M @ B`` (shape ``(N, k)``) and reconstruction is ``coeffs @ B.T``.
"""
from __future__ import annotations

import numpy as np


def choose_rank(singular_values: np.ndarray, energy: float, max_rank: int,
                min_rank: int = 1) -> int:
    """Smallest k whose cumulative squared-energy fraction reaches ``energy``.

    ``min_rank`` floors the result: energy alone under-counts curves like the
    low-mass phase, whose spectral energy is dominated by a single huge ramp so
    that the *match-relevant* fine structure sits below the threshold.
    """
    if len(singular_values) == 0:
        return 1
    cumulative = np.cumsum(singular_values ** 2) / np.sum(singular_values ** 2)
    k = int(np.searchsorted(cumulative, energy) + 1)
    return int(np.clip(k, min(min_rank, len(singular_values)),
                       min(max_rank, len(singular_values))))


def build_basis(matrix: np.ndarray, energy: float = 0.99999,
                max_rank: int = 64, min_rank: int = 1):
    """SVD ``matrix`` (N, L) and return ``(basis (L, k), coeffs (N, k), k)``.

    ``k`` is chosen so the retained singular values capture ``energy`` of the
    total spectral energy, floored at ``min_rank`` and capped at ``max_rank``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    # For a large data matrix (e.g. a 50k-waveform bank at L=4096), the economy
    # SVD's U (N, L) and LAPACK workspace alone are several GB and OOM a modest
    # box. We only ever keep the top ``max_rank`` (~64) singular vectors, so use a
    # randomized SVD there -- O(N*L*max_rank) memory/time and accurate to well
    # past the retained rank. Exact SVD is kept for small matrices (identical
    # results, no sklearn dependency for the common case).
    if max(matrix.shape) > 8000:
        from sklearn.utils.extmath import randomized_svd
        n_comp = int(min(max_rank, min(matrix.shape)))
        _, s, vt = randomized_svd(matrix, n_components=n_comp,
                                  n_oversamples=16, random_state=0)
        # Energy fraction needs the TOTAL spectral energy; the randomized s only
        # covers the top n_comp, so use the exact Frobenius energy as denominator.
        total_energy = float((matrix ** 2).sum())
        cumulative = np.cumsum(s ** 2) / (total_energy + 1e-30)
        k = int(np.searchsorted(cumulative, energy) + 1)
        k = int(np.clip(k, min(min_rank, len(s)), min(max_rank, len(s))))
    else:
        # Economy SVD: U (N, r), s (r,), Vt (r, L) with r = min(N, L).
        _, s, vt = np.linalg.svd(matrix, full_matrices=False)
        k = choose_rank(s, energy, max_rank, min_rank)
    basis = vt[:k].T.copy()                 # (L, k)
    coeffs = matrix @ basis                 # (N, k)
    return basis, coeffs, k


def reconstruct(coeffs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Reconstruct the (N, L) matrix from coefficients and basis."""
    return np.asarray(coeffs) @ np.asarray(basis).T
