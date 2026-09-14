# tests/test_config.py
import pytest
import tempfile
import os
import json
from src.core.config import ProjectConfig

def test_project_config_defaults():
    """Test that ProjectConfig initializes with correct defaults"""
    config = ProjectConfig()
    
    assert config.project_name == "default_project"
    assert config.num_samples == 1000
    assert config.waveform == "IMRPhenomD"
    assert config.device == "cuda"
    assert config.val_split == 0.3

def test_project_config_custom():
    """Test custom configuration"""
    config = ProjectConfig(
        project_name="test_project",
        num_samples=500,
        waveform="IMRPhenomD"
    )
    
    assert config.project_name == "test_project"
    assert config.num_samples == 500
    assert config.waveform == "IMRPhenomD"

def test_project_config_save_load():
    """Test saving and loading configuration"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save config
        config = ProjectConfig(
            project_name="test",
            checkpoint_dir=tmpdir,
            num_samples=250
        )
        config.save()
        
        # Check file exists
        config_file = os.path.join(config.project_path, "config.json")
        assert os.path.exists(config_file)
        
        # Load and verify
        loaded_config = ProjectConfig.load(config.project_path)
        assert loaded_config.project_name == "test"
        assert loaded_config.num_samples == 250

def test_project_path():
    """Test project path generation"""
    config = ProjectConfig(
        project_name="my_model",
        checkpoint_dir="/tmp/checkpoints"
    )
    
    assert config.project_path == "/tmp/checkpoints/my_model"
