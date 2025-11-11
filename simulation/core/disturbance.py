"""
Disturbance modeling for smoke simulation.

This module provides functionality for modeling disturbances in smoke patterns,
including height and wind disturbances for smoke simulation trajectories.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from .wind_distribution import WindDistribution


@dataclass
class FullDisturbanceTrajectory:
    """Represents a trajectory of disturbances over time.
    
    Attributes:
        height: Initial height (only one value as it's part of the initial state)
        wind_dirs: List of wind directions for each timestep
        wind_speeds: List of wind speeds for each timestep
    """
    height: float
    wind_dirs: List[float]
    wind_speeds: List[float]


class FullDisturbances:
    """Container for all disturbance models and their operations.
    
    This class combines height and wind distribution models and provides methods
    to sample trajectories and calculate their likelihoods.
    
    Args:
        height_dist: Height distribution model
        wind_dist: Wind distribution model
    """
    
    def __init__(self, height_dist, wind_dist: WindDistribution):
        self.height_dist = height_dist
        self.wind_dist = wind_dist
    
    def sample_disturbances(self, n_timesteps: int) -> FullDisturbanceTrajectory:
        """Sample a trajectory of disturbances.
        
        Args:
            n_timesteps: Number of timesteps to sample
            
        Returns:
            A FullDisturbanceTrajectory containing the sampled disturbances
        """
        # Sample height (single value for initial state)
        height = self.height_dist.sample(1)[0]
        
        # Sample wind directions and speeds for all timesteps
        wind_dirs, wind_speeds = self.wind_dist.sample_trajectory(n_timesteps)
        
        return FullDisturbanceTrajectory(
            height=height,
            wind_dirs=wind_dirs,
            wind_speeds=wind_speeds
        )
    
    def disturbance_trajectory_likelihood(
        self, 
        disturb_traj: FullDisturbanceTrajectory
    ) -> float:
        """Calculate the likelihood of a disturbance trajectory.
        
        Args:
            disturb_traj: The disturbance trajectory to evaluate
            
        Returns:
            The probability density of the trajectory
        """
        height_pdf = self.height_dist.likelihood(disturb_traj.height)
        wind_pdf = self.wind_dist.likelihood_trajectory(
            disturb_traj.wind_dirs, 
            disturb_traj.wind_speeds
        )
        return height_pdf * wind_pdf
    
    def disturbance_trajectory_log_likelihood(
        self, 
        disturb_traj: FullDisturbanceTrajectory
    ) -> float:
        """Calculate the log-likelihood of a disturbance trajectory.
        
        Args:
            disturb_traj: The disturbance trajectory to evaluate
            
        Returns:
            The log-probability density of the trajectory
        """
        height_ll = self.height_dist.log_likelihood(disturb_traj.height)
        wind_ll = self.wind_dist.log_likelihood_trajectory(
            disturb_traj.wind_dirs,
            disturb_traj.wind_speeds
        )
        return height_ll + wind_ll


def most_likely_disturbance_trajectories(
    loglikelihoods: List[float],
    trajectory_list: List[FullDisturbanceTrajectory],
    num_most_likely: int
) -> Tuple[List[FullDisturbanceTrajectory], List[int]]:
    """Find the most likely disturbance trajectories.
    
    Args:
        loglikelihoods: List of log-likelihoods for each trajectory
        trajectory_list: List of FullDisturbanceTrajectory objects
        num_most_likely: Number of most likely trajectories to return
        
    Returns:
        Tuple of (list of most likely trajectories, their indices)
    """
    # Get indices of top num_most_likely log-likelihoods
    top_indices = np.argpartition(loglikelihoods, -num_most_likely)[-num_most_likely:]
    # Sort them in descending order
    top_indices = top_indices[np.argsort(-np.array(loglikelihoods)[top_indices])]
    
    return [trajectory_list[i] for i in top_indices], top_indices.tolist()


def find_most_likely_trajectories(
    disturbances: FullDisturbances,
    trajectory_list: List[FullDisturbanceTrajectory],
    num_most_likely: int
) -> Tuple[List[FullDisturbanceTrajectory], List[int]]:
    """Find the most likely trajectories from a list.
    
    Args:
        disturbances: FullDisturbances instance
        trajectory_list: List of trajectories to evaluate
        num_most_likely: Number of most likely trajectories to return
        
    Returns:
        Tuple of (list of most likely trajectories, their log-likelihoods)
    """
    loglikelihoods = [
        disturbances.disturbance_trajectory_log_likelihood(traj)
        for traj in trajectory_list
    ]
    
    return most_likely_disturbance_trajectories(
        loglikelihoods, trajectory_list, num_most_likely
    )
