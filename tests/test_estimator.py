"""Tests for the parameter-estimation prior logic (no network/model needed)."""
import numpy as np
import pytest

from src.inference.estimator import PARAM_INDEX, Prior


def test_prior_default_shapes():
    prior = Prior.default()
    assert prior.ndim == 4
    assert prior.free_names == ["m1", "m2", "s1z", "s2z"]
    assert prior.low.shape == (4,)
    assert prior.high.shape == (4,)


def test_prior_to_full_places_fixed_and_free():
    prior = Prior(bounds={"m1": (30, 90), "m2": (30, 90)},
                  fixed={"s1z": 0.1, "inc": 0.5, "ecc": 0.0})
    full = prior.to_full(np.array([70.0, 60.0]))
    assert full[PARAM_INDEX["m1"]] == 70.0
    assert full[PARAM_INDEX["m2"]] == 60.0
    assert full[PARAM_INDEX["s1z"]] == 0.1
    assert full[PARAM_INDEX["inc"]] == 0.5


def test_prior_log_prior_bounds_and_mass_ordering():
    prior = Prior(bounds={"m1": (30, 90), "m2": (30, 90)})
    assert prior.log_prior(np.array([70.0, 60.0])) == 0.0      # inside, m1 > m2
    assert prior.log_prior(np.array([60.0, 70.0])) == -np.inf  # m2 > m1 rejected
    assert prior.log_prior(np.array([100.0, 60.0])) == -np.inf  # out of bounds


def test_prior_to_full_batch_matches_single():
    prior = Prior(bounds={"m1": (30, 90), "m2": (30, 90)}, fixed={"inc": 0.3})
    thetas = np.array([[70.0, 60.0], [80.0, 40.0]])
    batch = prior.to_full_batch(thetas)
    for i in range(2):
        assert np.allclose(batch[i], prior.to_full(thetas[i]))


def test_prior_sample_within_bounds():
    prior = Prior.default()
    s = prior.sample(50, seed=0)
    assert s.shape == (50, 4)
    assert np.all(s >= prior.low) and np.all(s <= prior.high)
