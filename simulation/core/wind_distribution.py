"""
Wind distribution modeling for smoke simulation.

This module provides functionality for modeling wind distributions in smoke patterns,
similar to the wind_distribution.jl file in the original Julia codebase.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import stats


@dataclass
class WindDistribution:
    """Class for modeling wind distributions in smoke patterns.
    
    This class combines wind speed and direction modeling into a single interface
    for use in smoke simulation. It handles both sampling and probability calculations.
    
    Args:
        mean_speed: Mean wind speed (default: 5.0)
        speed_std: Standard deviation of wind speed (default: 2.0)
        mean_direction: Mean wind direction in radians (default: 0.0)
        direction_std: Standard deviation of wind direction (default: 0.5)
        speed_dist: Type of distribution for wind speed (default: 'weibull')
        direction_dist: Type of distribution for wind direction (default: 'vonmises')
        **params: Additional parameters for the distributions
    """
    mean_speed: float = 5.0
    speed_std: float = 2.0
    mean_direction: float = 0.0
    direction_std: float = 0.5
    speed_dist: str = 'weibull'
    direction_dist: str = 'vonmises'
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the wind distributions with the specified parameters."""
        # Set up speed distribution
        if self.speed_dist.lower() == 'weibull':
            self._speed_dist = stats.weibull_min(
                c=self.params.get('shape', 2.0),
                scale=self.params.get('scale', self.mean_speed),
                loc=self.params.get('loc', 0.0)
            )
        else:
            raise ValueError(f"Unsupported speed distribution: {self.speed_dist}")
        
        # Set up direction distribution
        if self.direction_dist.lower() == 'vonmises':
            self._direction_dist = stats.vonmises(
                kappa=1.0 / (self.direction_std ** 2) if self.direction_std > 0 else 1e6,
                loc=self.mean_direction
            )
        else:
            raise ValueError(f"Unsupported direction distribution: {self.direction_dist}")
    
    def sample(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Sample wind speeds and directions.
        
        Args:
            size: Number of samples to generate
            
        Returns:
            Tuple of (speeds, directions) arrays
        """
        return self.sample_speeds(size), self.sample_directions(size)
    
    def sample_speeds(self, n_samples: int) -> List[float]:
        """Sample wind speeds.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of wind speeds
        """
        return np.maximum(0, self._speed_dist.rvs(size=n_samples)).tolist()
    
    def sample_directions(self, n_samples: int) -> List[float]:
        """Sample wind directions.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of wind directions in radians [0, 2π)
        """
        samples = self._direction_dist.rvs(size=n_samples)
        return (samples % (2 * np.pi)).tolist()
    
    def sample_trajectory(self, n_timesteps: int) -> Tuple[List[float], List[float]]:
        """Sample a trajectory of wind directions and speeds.
        
        Args:
            n_timesteps: Number of timesteps to sample
            
        Returns:
            Tuple of (wind_directions, wind_speeds) lists
        """
        return self.sample_directions(n_timesteps), self.sample_speeds(n_timesteps)
    
    def speed_pdf(self, speed: float) -> float:
        """Calculate the probability density of a wind speed."""
        return float(self._speed_dist.pdf(speed))
    
    def direction_pdf(self, direction: float) -> float:
        """Calculate the probability density of a wind direction."""
        x = direction % (2 * np.pi)
        return float(self._direction_dist.pdf(x))
    
    def speed_logpdf(self, speed: float) -> float:
        """Calculate the log probability density of a wind speed."""
        return float(self._speed_dist.logpdf(speed))
    
    def direction_logpdf(self, direction: float) -> float:
        """Calculate the log probability density of a wind direction."""
        x = direction % (2 * np.pi)
        return float(self._direction_dist.logpdf(x))
    
    def likelihood_trajectory(self, wind_directions: List[float], 
                            wind_speeds: List[float]) -> float:
        """Calculate the likelihood of a trajectory of wind directions and speeds.
        
        Args:
            wind_directions: List of wind directions in radians
            wind_speeds: List of wind speeds
            
        Returns:
            The joint probability density of the trajectory
        """
        dir_likelihood = np.prod([self.direction_pdf(d) for d in wind_directions])
        speed_likelihood = np.prod([self.speed_pdf(s) for s in wind_speeds])
        return dir_likelihood * speed_likelihood
    
    def log_likelihood_trajectory(self, wind_directions: List[float], 
                                wind_speeds: List[float]) -> float:
        """Calculate the log-likelihood of a trajectory of wind directions and speeds.
        
        Args:
            wind_directions: List of wind directions in radians
            wind_speeds: List of wind speeds
            
        Returns:
            The log of the joint probability density of the trajectory
        """
        dir_ll = sum([self.direction_logpdf(d) for d in wind_directions])
        speed_ll = sum([self.speed_logpdf(s) for s in wind_speeds])
        return dir_ll + speed_ll


@dataclass
class WindDistribution:
    """Class for modeling wind distributions in smoke patterns.
    
    This class provides methods to sample wind speeds and directions,
    and to calculate their likelihoods.
    """
    mean_speed: float = 5.0
    speed_std: float = 2.0
    mean_direction: float = 0.0
    direction_std: float = 0.5
    
    def __post_init__(self):
        """Initialize the wind model with default parameters."""
        self.model = WindModel(
            speed_dist='weibull',
            direction_dist='vonmises',
            scale=self.mean_speed,
            shape=2.0,  # Typical shape parameter for wind speeds
            kappa=1.0 / (self.direction_std ** 2) if self.direction_std > 0 else 1e6,
            mean_direction=self.mean_direction
        )
    
    def sample_trajectory(self, n_timesteps: int) -> Tuple[List[float], List[float]]:
        """Sample a trajectory of wind directions and speeds.
        
        Args:
            n_timesteps: Number of timesteps to sample
            
        Returns:
            Tuple of (wind_directions, wind_speeds) lists
        """
        wind_directions = self.model.sample_directions(n_timesteps)
        wind_speeds = self.model.sample_speeds(n_timesteps)
        return wind_directions, wind_speeds
    
    def likelihood_trajectory(self, wind_directions: List[float], 
                            wind_speeds: List[float]) -> float:
        """Calculate the likelihood of a trajectory of wind directions and speeds.
        
        Args:
            wind_directions: List of wind directions in radians
            wind_speeds: List of wind speeds
            
        Returns:
            The joint probability density of the trajectory
        """
        # Calculate the product of individual likelihoods
        dir_likelihood = np.prod([self.model.direction_pdf(d) for d in wind_directions])
        speed_likelihood = np.prod([self.model.speed_pdf(s) for s in wind_speeds])
        return dir_likelihood * speed_likelihood
    
    def log_likelihood_trajectory(self, wind_directions: List[float], 
                                 wind_speeds: List[float]) -> float:
        """Calculate the log-likelihood of a trajectory of wind directions and speeds.
        
        Args:
            wind_directions: List of wind directions in radians
            wind_speeds: List of wind speeds
            
        Returns:
            The log of the joint probability density of the trajectory
        """
        # Sum the log-likelihoods for numerical stability
        dir_ll = sum([self.model.direction_logpdf(d) for d in wind_directions])
        speed_ll = sum([self.model.speed_logpdf(s) for s in wind_speeds])
        return dir_ll + speed_ll

class WindModel:
    """Class for modeling wind distributions in smoke patterns."""
    
    def __init__(self, speed_dist='weibull', direction_dist='vonmises', **params):
        """Initialize wind model with speed and direction distributions.
        
        Args:
            speed_dist: Type of distribution for wind speed
            direction_dist: Type of distribution for wind direction
            **params: Parameters for the distributions
        """
        self.speed_dist = self._create_speed_distribution(speed_dist, params)
        self.direction_dist = self._create_direction_distribution(direction_dist, params)
        self.params = params
    
    def _create_speed_distribution(self, dist_type, params):
        """Create wind speed distribution."""
        if dist_type.lower() == 'weibull':
            return stats.weibull_min(
                c=params.get('shape', 2.0),
                scale=params.get('scale', 5.0),
                loc=params.get('loc', 0.0)
            )
        else:
            raise ValueError(f"Unsupported speed distribution: {dist_type}")
    
    def _create_direction_distribution(self, dist_type, params):
        """Create wind direction distribution."""
        if dist_type.lower() == 'vonmises':
            return stats.vonmises(
                kappa=params.get('kappa', 1.0),
                loc=params.get('mean_direction', 0.0)
            )
        else:
            raise ValueError(f"Unsupported direction distribution: {dist_type}")
    
    def sample(self, size=1):
        """Sample wind speed and direction.
        
        Args:
            size: Number of samples to generate
            
        Returns:
            Tuple of (speeds, directions) arrays
        """
        speeds = self.speed_dist.rvs(size=size)
        directions = self.direction_dist.rvs(size=size)
        return speeds, directions
    
    def sample_speeds(self, n_samples: int) -> List[float]:
        """Sample wind speeds.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of wind speeds
        """
        return np.maximum(0, self.speed_dist.rvs(size=n_samples)).tolist()
    
    def sample_directions(self, n_samples: int) -> List[float]:
        """Sample wind directions.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of wind directions in radians [0, 2π)
        """
        samples = self.direction_dist.rvs(size=n_samples)
        # Ensure directions are in [0, 2π)
        return (samples % (2 * np.pi)).tolist()
    
    def speed_pdf(self, speed: float) -> float:
        """Calculate the probability density of a wind speed."""
        return float(self.speed_dist.pdf(speed))
    
    def direction_pdf(self, direction: float) -> float:
        """Calculate the probability density of a wind direction."""
        # Handle circular nature of directions
        x = direction % (2 * np.pi)
        return float(self.direction_dist.pdf(x))
    
    def speed_logpdf(self, speed: float) -> float:
        """Calculate the log probability density of a wind speed."""
        return float(self.speed_dist.logpdf(speed))
    
    def direction_logpdf(self, direction: float) -> float:
        """Calculate the log probability density of a wind direction.
        
        Args:
            direction: Wind direction in radians
            
        Returns:
            Log probability density
        """
        # Handle circular nature of directions
        x = direction % (2 * np.pi)
        return float(self.direction_dist.logpdf(x))
