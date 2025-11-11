"""
Trajectory utilities for particle path calculation in vector fields.
"""

import numpy as np
from typing import List, Tuple


def calculate_trajectory(
    start_x: float,
    start_y: float,
    vector_field,
    max_steps: int = 500,
    step_size: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Calculate a single particle's trajectory from a starting point using a vector field.
    
    Args:
        start_x: Starting X coordinate in pixel space
        start_y: Starting Y coordinate in pixel space
        vector_field: TerrainVectorField object with get_vector_at() method
        max_steps: Maximum number of steps to simulate
        step_size: Distance to move per step in pixels
        
    Returns:
        List of (x, y) tuples representing the particle path
    """
    path = [(start_x, start_y)]
    current_x, current_y = start_x, start_y
    
    for step in range(max_steps):
        # Check if particle is out of bounds
        if (current_x < 0 or current_x >= vector_field.width or
            current_y < 0 or current_y >= vector_field.height):
            break
        
        # Get terrain vector at current position
        u, v = vector_field.get_vector_at(current_x, current_y)
        
        # If vector is negligible, stop the particle
        if abs(u) < 1e-6 and abs(v) < 1e-6:
            break
        
        # Move particle in direction of vector
        next_x = current_x + u * step_size
        next_y = current_y + v * step_size
        
        # Clamp to world boundaries
        next_x = np.clip(next_x, 0, vector_field.width - 1)
        next_y = np.clip(next_y, 0, vector_field.height - 1)
        
        path.append((next_x, next_y))
        current_x, current_y = next_x, next_y
    
    return path


def calculate_trajectories(
    start_points: np.ndarray,
    vector_field,
    max_steps: int = 500,
    step_size: float = 0.5
) -> List[List[Tuple[float, float]]]:
    """
    Calculate trajectories for multiple starting points.
    
    Args:
        start_points: Array of shape (N, 2) with starting coordinates [x, y]
        vector_field: TerrainVectorField object with get_vector_at() method
        max_steps: Maximum number of steps to simulate
        step_size: Distance to move per step in pixels
        
    Returns:
        List of N trajectories, each a list of (x, y) tuples
    """
    trajectories = []
    for start_x, start_y in start_points:
        trajectory = calculate_trajectory(start_x, start_y, vector_field, max_steps, step_size)
        trajectories.append(trajectory)
    return trajectories
