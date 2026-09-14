# src/utils/io.py
import json
import torch
import numpy as np
import os

def save_checkpoint(path: str, models: dict, data: dict, metadata: dict):
    """Save a complete checkpoint with all necessary files"""
    os.makedirs(path, exist_ok=True)
    
    # Save models
    for name, model in models.items():
        torch.save(model.state_dict(), os.path.join(path, f'{name}_model.pt'))
    
    # Save data arrays
    for name, array in data.items():
        np.save(os.path.join(path, f'{name}.npy'), array)
    
    # Save metadata
    with open(os.path.join(path, 'meta.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def load_checkpoint(path: str) -> dict:
    """Load a checkpoint and return all components"""
    checkpoint = {}
    
    # Load metadata
    with open(os.path.join(path, 'meta.json')) as f:
        checkpoint['metadata'] = json.load(f)
    
    # Load numpy arrays
    for file in os.listdir(path):
        if file.endswith('.npy'):
            name = file[:-4]
            checkpoint[name] = np.load(os.path.join(path, file))
    
    return checkpoint

