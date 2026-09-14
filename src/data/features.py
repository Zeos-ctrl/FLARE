# src/data/features.py
import numpy as np
from typing import Dict, List

class FeatureExtractor:
    """Handles parameter to feature transformations"""
    
    @staticmethod
    def compute_features(parameters: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Compute derived features from raw parameters
        
        Args:
            parameters: Array of shape (N, 6) with [m1, m2, s1z, s2z, inc, ecc]
            feature_names: List of features to compute
        
        Returns:
            Array of shape (N, len(feature_names))
        """
        m1, m2, s1z, s2z, inc, ecc = parameters.T
        
        chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        eta = (m1 * m2) / (m1 + m2)**2
        total_mass = m1 + m2
        chi_eff = (m1 * s1z + m2 * s2z) / (m1 + m2)
        features = {
            'chirp_mass': chirp_mass,
            'symmetric_mass_ratio': eta,
            'effective_spin': chi_eff,
            'inclination': inc,
            'eccentricity': ecc,
            'mass_ratio': m2 / m1,
            'total_mass': total_mass,
            # Raw aligned component spins (the surrogate's direct spin inputs).
            'spin1z': s1z,
            'spin2z': s2z,
            # PN reduced-spin parameter: the leading spin combination that enters
            # the inspiral phase (IMRPhenomD is parameterised by ~this). Aligned
            # spin causes orbital hang-up -> more inspiral cycles -> a larger total
            # accumulated phase (the precision-critical reduced-order `scale`), so
            # this is the key spin feature for the phase-scale head.
            'chi_pn': chi_eff - (38.0 * eta / 113.0) * (s1z + s2z),
            # Antisymmetric spin (subdominant; helps unequal-spin cases).
            'chi_a': 0.5 * (s1z - s2z),
            # Log features: the accumulated-phase scale of a chirp is
            # log(scale) ~ const - (5/3) log(chirp_mass), i.e. LINEAR in
            # log(chirp_mass), so these make the (precision-critical) reduced-order
            # phase-scale a near-linear, easily-learned target.
            'log_chirp_mass': np.log(chirp_mass),
            'log_total_mass': np.log(total_mass),
            'log_eta': np.log(eta),
            # Post-Newtonian scaling of the accumulated phase (~Mc^-5/3): a direct
            # feature for the reduced-order phase-scale target.
            'chirp_mass_m53': chirp_mass ** (-5.0 / 3.0),
        }
        
        return np.stack([features[name] for name in feature_names], axis=1)
    
    @staticmethod
    def normalize_features(features: np.ndarray, 
                          means: np.ndarray = None, 
                          stds: np.ndarray = None) -> tuple:
        """Normalize features to zero mean and unit variance"""
        if means is None:
            means = features.mean(axis=0)
        if stds is None:
            stds = features.std(axis=0)
            
        normalized = (features - means) / (stds + 1e-10)
        return normalized, means, stds
