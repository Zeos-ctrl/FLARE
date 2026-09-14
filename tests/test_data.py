# tests/test_data.py
import pytest
import numpy as np
from src.data.generator import WaveformGenerator, WaveformDataset
from src.data.features import FeatureExtractor

def test_waveform_generator_init():
    """Test WaveformGenerator initialization"""
    gen = WaveformGenerator(
        waveform_length=1024,
        delta_t=1/2048,
        approximant="SEOBNRv4"
    )
    
    assert gen.waveform_length == 1024
    assert gen.delta_t == 1/2048
    assert gen.approximant == "SEOBNRv4"

def test_sample_parameters():
    """Test parameter sampling"""
    gen = WaveformGenerator()
    
    # Test different sampling methods
    for method in ['lhs', 'uniform']:
        params = gen.sample_parameters(
            n_samples=10,
            method=method,
            seed=42
        )
        
        assert params.shape == (10, 6)
        assert np.all(params[:, 0] >= 30)  # m1 >= 30
        assert np.all(params[:, 0] <= 100)  # m1 <= 100
        assert np.all(params[:, 4] >= 0)  # inclination >= 0
        assert np.all(params[:, 4] <= np.pi)  # inclination <= pi

def test_sample_parameters_reproducibility():
    """Test that sampling with seed is reproducible"""
    gen = WaveformGenerator()
    
    params1 = gen.sample_parameters(n_samples=5, seed=42)
    params2 = gen.sample_parameters(n_samples=5, seed=42)
    
    np.testing.assert_array_equal(params1, params2)

def test_feature_extractor():
    """Test feature computation"""
    # Create dummy parameters
    params = np.array([[35, 30, 0.1, 0.2, 0.5, 0.01]])
    
    # Compute features
    features = FeatureExtractor.compute_features(
        params,
        ['chirp_mass', 'symmetric_mass_ratio']
    )
    
    assert features.shape == (1, 2)
    
    # Check chirp mass calculation
    m1, m2 = 35, 30
    expected_chirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    np.testing.assert_almost_equal(features[0, 0], expected_chirp)

def test_feature_normalization():
    """Test feature normalization"""
    features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    normalized, means, stds = FeatureExtractor.normalize_features(features)
    
    # Check normalization properties
    np.testing.assert_almost_equal(normalized.mean(axis=0), [0, 0, 0])
    np.testing.assert_almost_equal(normalized.std(axis=0), [1, 1, 1])
    
    # Test denormalization
    denormalized = normalized * stds + means
    np.testing.assert_almost_equal(denormalized, features)
