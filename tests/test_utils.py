# tests/test_utils.py
import pytest
import tempfile
import json
import torch
import numpy as np
import os
from unittest.mock import Mock, patch
from src.utils.io import save_checkpoint, load_checkpoint
from src.utils.metrics import compute_match

def test_save_load_checkpoint():
    """Test checkpoint saving and loading"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        models = {
            'amp': Mock(state_dict=lambda: {'weight': torch.randn(10, 10)})
        }
        data_arrays = {
            'param_means': np.array([1, 2, 3]),
            'param_stds': np.array([0.1, 0.2, 0.3])
        }
        metadata = {
            'waveform': 'SEOBNRv4',
            'project_name': 'test'
        }
        
        # Save checkpoint
        save_checkpoint(tmpdir, models, data_arrays, metadata)
        
        # Check files exist
        assert os.path.exists(os.path.join(tmpdir, 'meta.json'))
        assert os.path.exists(os.path.join(tmpdir, 'param_means.npy'))
        assert os.path.exists(os.path.join(tmpdir, 'param_stds.npy'))
        
        # Load checkpoint
        checkpoint = load_checkpoint(tmpdir)
        
        assert checkpoint['metadata']['waveform'] == 'SEOBNRv4'
        np.testing.assert_array_equal(checkpoint['param_means'], data_arrays['param_means'])

def test_compute_match():
    """Test match computation"""
    # Create two similar waveforms
    t = np.linspace(0, 1, 1000)
    h1 = np.sin(2 * np.pi * 10 * t)
    h2 = np.sin(2 * np.pi * 10 * t + 0.1)  # Slightly phase shifted
    
    # Use actual PyCBC FrequencySeries for PSD
    from pycbc.types import FrequencySeries
    
    with patch('src.utils.metrics.aLIGOZeroDetHighPower') as mock_psd:
        # Mock PSD as FrequencySeries
        mock_psd.return_value = FrequencySeries(np.ones(501), delta_f=1.0)
        
        match = compute_match(h1, h2, delta_t=0.001)
        
        assert 0 <= match <= 1
        assert match > 0.9  # Should be high for similar waveforms

def test_compute_match_identical():
    """Test match computation for identical waveforms"""
    h = np.random.randn(1000)
    
    from pycbc.types import FrequencySeries
    
    with patch('src.utils.metrics.aLIGOZeroDetHighPower') as mock_psd:
        # Mock PSD as FrequencySeries
        mock_psd.return_value = FrequencySeries(np.ones(501), delta_f=1.0)
        
        match = compute_match(h, h, delta_t=0.001)
        
        assert match == pytest.approx(1.0, rel=1e-5)
