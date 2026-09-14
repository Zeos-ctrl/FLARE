# src/utils/uncertainty.py
"""Uncertainty quantification utilities for neural network predictions."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_last_layer_hessian(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_var: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Compute the diagonal of the Hessian of MSE loss w.r.t. the final linear layer.
    
    This is used for Laplace approximation-based uncertainty quantification.
    
    Args:
        model: Neural network model
        loader: DataLoader for the training data
        device: Device to run computations on
        noise_var: Noise variance for the likelihood
        
    Returns:
        Dictionary with 'weight_var' and 'bias_var' tensors
    """
    # Find the last linear layer with output dimension 1
    linear_layers = [
        module for module in model.modules()
        if isinstance(module, nn.Linear) and module.out_features == 1
    ]
    
    if not linear_layers:
        raise ValueError("No final Linear layer with out_features=1 found in model")
    
    final_layer = linear_layers[-1]
    
    # Initialize accumulators for Hessian diagonal
    weight_shape = final_layer.weight.shape
    bias_shape = final_layer.bias.shape if final_layer.bias is not None else (1,)
    
    H_weight = torch.zeros(weight_shape, device=device)
    H_bias = torch.zeros(bias_shape, device=device)
    
    # Hook to capture input features to the final layer
    captured_features = {'phi': None}
    
    def hook_fn(module, inputs, outputs):
        captured_features['phi'] = inputs[0].detach()
    
    hook = final_layer.register_forward_hook(hook_fn)
    
    # Accumulate Hessian diagonal over the dataset
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                X, Y = batch
            else:
                # Handle different batch formats
                X = batch[0]
                Y = batch[1] if len(batch) > 1 else None
            
            X = X.to(device)
            
            # Split input into time and parameters if needed
            if X.shape[1] > 1:
                t_norm = X[:, :1]
                theta = X[:, 1:]
                _ = model(t_norm, theta)
            else:
                _ = model(X)
            
            # Get captured features
            phi = captured_features['phi']
            batch_size = phi.shape[0]
            
            # Accumulate Hessian diagonal
            # H_ii = sum_n (phi_n^2) / noise_var for weights
            H_weight += (phi ** 2).sum(dim=0, keepdim=True) / noise_var
            
            # H_bias = N / noise_var for bias
            if final_layer.bias is not None:
                H_bias += batch_size / noise_var
    
    # Remove hook
    hook.remove()
    
    # Compute variances as inverse of Hessian diagonal
    weight_variances = 1.0 / (H_weight + 1e-10)  # Add small epsilon for numerical stability
    bias_variance = 1.0 / (H_bias + 1e-10) if final_layer.bias is not None else torch.zeros(1, device=device)
    
    return {
        'weight_var': weight_variances,
        'bias_var': bias_variance
    }


def compute_prediction_uncertainty(
    model: nn.Module,
    inputs: torch.Tensor,
    weight_var: torch.Tensor,
    bias_var: torch.Tensor,
    capture_layer_name: str = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute prediction mean and variance using Laplace approximation.
    
    Args:
        model: Neural network model
        inputs: Input tensor
        weight_var: Weight variances from Hessian
        bias_var: Bias variance from Hessian
        capture_layer_name: Name of layer to capture features from
        
    Returns:
        Tuple of (mean predictions, variance predictions)
    """
    # Find the last linear layer
    linear_layers = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.out_features == 1
    ]
    
    if not linear_layers:
        raise ValueError("No final Linear layer found")
    
    layer_name, final_layer = linear_layers[-1]
    
    # Hook to capture features
    captured = {}
    
    def hook_fn(module, inputs, outputs):
        captured['features'] = inputs[0].detach()
    
    hook = final_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(inputs)
    
    # Remove hook
    hook.remove()
    
    # Compute variance
    features = captured['features']
    variance = (features ** 2 * weight_var).sum(1, keepdim=True) + bias_var
    
    return predictions, variance


class LaplacePosterior:
    """
    Laplace approximation for uncertainty quantification in neural networks.
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 device: torch.device, noise_var: float = 1.0):
        """
        Initialize Laplace posterior approximation.
        
        Args:
            model: Trained neural network
            train_loader: Training data loader
            device: Computation device
            noise_var: Observation noise variance
        """
        self.model = model
        self.device = device
        self.noise_var = noise_var
        
        # Compute Hessian diagonal
        self.hessian_info = compute_last_layer_hessian(
            model, train_loader, device, noise_var
        )
        
        self.weight_var = self.hessian_info['weight_var']
        self.bias_var = self.hessian_info['bias_var']
        
        logger.info(f"Laplace approximation initialized with noise_var={noise_var}")
    
    def predict_with_uncertainty(self, inputs: torch.Tensor, 
                                 sigma_level: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            inputs: Input tensor
            sigma_level: Number of standard deviations for uncertainty
            
        Returns:
            Tuple of (mean predictions, uncertainties)
        """
        mean, variance = compute_prediction_uncertainty(
            self.model, inputs, self.weight_var, self.bias_var
        )
        
        uncertainty = torch.sqrt(variance) * sigma_level
        
        return mean, uncertainty


def ensemble_uncertainty(models: list, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute uncertainty using model ensemble.
    
    Args:
        models: List of trained models
        inputs: Input tensor
        
    Returns:
        Tuple of (mean predictions, standard deviation)
    """
    predictions = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            pred = model(inputs)
            predictions.append(pred)
    
    predictions = torch.stack(predictions, dim=0)
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)
    
    return mean, std


def mc_dropout_uncertainty(model: nn.Module, inputs: torch.Tensor, 
                          n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute uncertainty using Monte Carlo dropout.
    
    Args:
        model: Model with dropout layers
        inputs: Input tensor
        n_samples: Number of forward passes
        
    Returns:
        Tuple of (mean predictions, standard deviation)
    """
    # Enable dropout during inference
    model.train()
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(inputs)
            predictions.append(pred)
    
    predictions = torch.stack(predictions, dim=0)
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)
    
    # Return model to eval mode
    model.eval()
    
    return mean, std
