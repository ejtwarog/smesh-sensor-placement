"""
Height distribution modeling for smoke simulation.

This module provides functionality for modeling height distributions in smoke patterns,
using a Gamma distribution to ensure positive values and convergence to normal for large Q_release.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Union, List


class HeightDistribution:
    """Class for modeling height distributions in smoke patterns.
    
    Uses a Gamma distribution since height > 0 and should converge to the Normal
    distribution for large Q_release and small h_std.
    """
    
    def __init__(self, Q_release: float, h_std: float):
        """Initialize height distribution model.
        
        Args:
            Q_release: Release rate parameter (affects mean height)
            h_std: Standard deviation of height distribution
        """
        # Convert Q_release to mean height
        self.h_mu = self._Q_to_h(Q_release)
        self.h_std = h_std
        
        # Calculate Gamma distribution parameters
        # Using shape (α) = (μ/σ)², scale (θ) = σ²/μ
        self.gamma_alpha = (self.h_mu ** 2) / (self.h_std ** 2)
        self.gamma_theta = (self.h_std ** 2) / self.h_mu
        
        # Create the Gamma distribution
        self.distribution = stats.gamma(
            a=self.gamma_alpha, 
            scale=self.gamma_theta
        )
    
    @staticmethod
    def _Q_to_h(Q: float) -> float:
        """Convert release rate Q to mean height.
        
        This is a simplified model that could be replaced with more complex models
        like the Briggs Plume Rise model in the future.
        """
        return Q / 10.0
    
    def sample(self, size: int = 1) -> Union[float, np.ndarray]:
        """Sample from the height distribution.
        
        Args:
            size: Number of samples to generate
            
        Returns:
            Single float if size=1, otherwise numpy array of sampled heights
        """
        samples = self.distribution.rvs(size=size)
        return samples[0] if size == 1 else samples
    
    def likelihood(self, height: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the probability density at given height(s).
        
        Args:
            height: Height value(s) at which to evaluate the PDF
            
        Returns:
            Probability density value(s)
        """
        return self.distribution.pdf(height)
    
    def log_likelihood(self, height: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the log probability density at given height(s).
        
        Args:
            height: Height value(s) at which to evaluate the log-PDF
            
        Returns:
            Log probability density value(s)
        """
        return self.distribution.logpdf(height)
    
    # Aliases for consistency with Julia interface
    pdf = likelihood
    logpdf = log_likelihood
