# tests/test_imports.py
"""Test that all modules can be imported."""

def test_core_imports():
    """Test core module imports"""
    from src.core.config import ProjectConfig
    from src.core.trainer import Trainer
    from src.core.tuner import Tuner
    from src.core.evaluator import Evaluator
    
    assert ProjectConfig is not None
    assert Trainer is not None
    assert Tuner is not None
    assert Evaluator is not None

def test_data_imports():
    """Test data module imports"""
    from src.data.generator import WaveformGenerator, WaveformDataset
    from src.data.features import FeatureExtractor
    
    assert WaveformGenerator is not None
    assert WaveformDataset is not None
    assert FeatureExtractor is not None

def test_utils_imports():
    """Test utils module imports"""
    from src.utils.io import save_checkpoint, load_checkpoint
    from src.utils.metrics import compute_match
    from src.utils.uncertainty import compute_last_layer_hessian
    
    assert save_checkpoint is not None
    assert load_checkpoint is not None
    assert compute_match is not None
    assert compute_last_layer_hessian is not None

def test_inference_imports():
    """Test inference module imports"""
    from src.inference.predictor import WaveformPredictor, WaveformPrediction
    
    assert WaveformPredictor is not None
    assert WaveformPrediction is not None
