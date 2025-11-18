"""
Wind trajectory simulation for particle movement in terrain-based vector fields.

This module manages the creation, initialization, and rollout of wind trajectories
that are sampled from burn areas and evolve based on terrain gradients.
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from simulation.core.terrain_vectors import TerrainVectorField
from simulation.core.smoke_points import sample_points_in_burn_area, initialize_particle_trajectories_from_course
from simulation.utils.trajectory_utils import calculate_trajectory


class WindTrajectory:
    """Represents a single wind trajectory with position, velocity, and history.
    
    Trajectories follow a primary wind vector (e.g., 270° at 5 kts) that is blended
    with terrain gradient influence. The terrain_influence_factor controls how much
    the terrain gradient modulates the primary wind direction.
    """
    
    def __init__(
        self,
        start_x: float,
        start_y: float,
        speed: float,
        direction_deg: float,
        terrain_influence_factor: float = 0.2
    ):
        """
        Initialize a wind trajectory.
        
        Args:
            start_x: Starting X coordinate (pixel)
            start_y: Starting Y coordinate (pixel)
            speed: Initial speed magnitude (e.g., 5 knots)
            direction_deg: Initial direction in degrees (0=N, 90=E, 180=S, 270=W)
            terrain_influence_factor: Blend factor for terrain gradient influence (0-1).
                0 = pure primary wind, 1 = pure terrain gradient
        """
        self.start_x = start_x
        self.start_y = start_y
        self.x = start_x
        self.y = start_y
        self.speed = speed
        self.direction_deg = direction_deg
        self.terrain_influence_factor = terrain_influence_factor
        
        # Convert direction to unit vector
        # Note: direction_deg uses standard compass convention (0=N, 90=E, 180=S, 270=W)
        # In image coordinates with inverted y-axis, we negate the angle
        th = np.deg2rad(-direction_deg)
        self.u = np.sin(th) * speed  # eastward
        self.v = np.cos(th) * speed  # northward
        
        # Track history
        self.history = [(start_x, start_y)]
        self.time_steps = 0
    
    def step(self, terrain_vector_field: TerrainVectorField, step_size: float = 0.5):
        """
        Advance trajectory by one time step, influenced by terrain gradient.
        
        The trajectory follows a blended direction:
        - Primary wind: maintains the initial direction and speed
        - Terrain influence: causes directional deflection based on terrain gradient
        - Blend: terrain_influence_factor controls the deflection angle (0=no deflection, 1=full terrain direction)
        
        Args:
            terrain_vector_field: TerrainVectorField object for terrain vectors
            step_size: Distance to move per step in pixels
        """
        # Get terrain vector at current position
        terrain_u, terrain_v = terrain_vector_field.get_vector_at(self.x, self.y)
        
        # Blend velocity vectors directly (more stable than angle blending)
        # Primary wind: (self.u, self.v) already has correct speed
        # Terrain influence: scale terrain vector by influence factor and speed
        blended_u = self.u + terrain_u * self.speed * self.terrain_influence_factor
        blended_v = self.v + terrain_v * self.speed * self.terrain_influence_factor
        
        # Normalize to maintain constant speed
        magnitude = np.sqrt(blended_u**2 + blended_v**2)
        if magnitude > 1e-6:
            self.u = (blended_u / magnitude) * self.speed
            self.v = (blended_v / magnitude) * self.speed
        else:
            # Fallback: keep current velocity
            pass
        
        # Update position
        self.x += self.u * step_size
        self.y += self.v * step_size
        
        # Record history
        self.history.append((self.x, self.y))
        self.time_steps += 1
    
    def get_path(self) -> List[Tuple[float, float]]:
        """Get the full trajectory path."""
        return self.history
    
    def is_out_of_bounds(self, width: int, height: int) -> bool:
        """Check if trajectory is out of bounds."""
        return self.x < 0 or self.x >= width or self.y < 0 or self.y >= height


class WindTrajectoryField:
    """Manages a collection of wind trajectories sampled from a burn area."""
    
    def __init__(
        self,
        terrain_vector_field: TerrainVectorField,
        burn_area_geojson: str,
        num_trajectories: int = 50,
        initial_speed: float = 5.0,
        initial_direction_deg: float = 270,
        terrain_influence_factor: float = 0.02
    ):
        """
        Initialize a wind trajectory field.
        
        Args:
            terrain_vector_field: TerrainVectorField object
            burn_area_geojson: Path to GeoJSON file with burn area
            num_trajectories: Number of trajectories to sample
            initial_speed: Initial speed for all trajectories (default: 5 knots)
            initial_direction_deg: Initial direction for all trajectories in degrees
                (default: 270 = westerly wind)
            terrain_influence_factor: Blend factor for terrain gradient influence (0-1).
                Controls how much terrain modulates the primary wind direction.
        """
        self.terrain_vector_field = terrain_vector_field
        self.burn_area_geojson = burn_area_geojson
        self.num_trajectories = num_trajectories
        self.initial_speed = initial_speed
        self.initial_direction_deg = initial_direction_deg
        self.terrain_influence_factor = terrain_influence_factor
        
        self.trajectories: List[WindTrajectory] = []
        self.active_trajectories: List[WindTrajectory] = []
        
        self._initialize_trajectories()
    
    def _initialize_trajectories(self):
        """Sample starting points and initialize trajectories."""
        # Sample starting points from burn area
        sampled_points = sample_points_in_burn_area(
            self.burn_area_geojson,
            self.num_trajectories,
            self.terrain_vector_field.transform,
            self.terrain_vector_field.height,
            self.terrain_vector_field.width
        )
        
        # Create trajectories with terrain influence factor
        for col, row in sampled_points:
            trajectory = WindTrajectory(
                col, row,
                self.initial_speed,
                self.initial_direction_deg,
                self.terrain_influence_factor
            )
            self.trajectories.append(trajectory)
        
        self.active_trajectories = self.trajectories.copy()
    
    def reinitialize(
        self,
        num_trajectories: Optional[int] = None,
        initial_speed: Optional[float] = None,
        initial_direction_deg: Optional[float] = None,
        terrain_influence_factor: Optional[float] = None
    ):
        """
        Reinitialize trajectories with new parameters.
        
        Args:
            num_trajectories: Number of trajectories to sample (uses existing if None)
            initial_speed: Initial speed for trajectories (uses existing if None)
            initial_direction_deg: Initial direction in degrees (uses existing if None)
            terrain_influence_factor: Terrain influence blend factor (uses existing if None)
        """
        # Update parameters
        if num_trajectories is not None:
            self.num_trajectories = num_trajectories
        if initial_speed is not None:
            self.initial_speed = initial_speed
        if initial_direction_deg is not None:
            self.initial_direction_deg = initial_direction_deg
        if terrain_influence_factor is not None:
            self.terrain_influence_factor = terrain_influence_factor
        
        # Clear and reinitialize
        self.trajectories = []
        self.active_trajectories = []
        self._initialize_trajectories()
    
    def step(self, step_size: float = 0.5):
        """
        Advance all active trajectories by one time step.
        
        Args:
            step_size: Distance to move per step in pixels
        """
        # Remove out-of-bounds trajectories
        self.active_trajectories = [
            t for t in self.active_trajectories
            if not t.is_out_of_bounds(
                self.terrain_vector_field.width,
                self.terrain_vector_field.height
            )
        ]
        
        # Step remaining trajectories
        for trajectory in self.active_trajectories:
            trajectory.step(self.terrain_vector_field, step_size)
    
    def rollout(self, num_steps: int, step_size: float = 0.5):
        """
        Rollout all trajectories for a specified number of steps.
        
        Args:
            num_steps: Number of time steps to simulate
            step_size: Distance to move per step in pixels
        """
        for _ in range(num_steps):
            self.step(step_size)
    
    def get_all_paths(self) -> List[List[Tuple[float, float]]]:
        """Get all trajectory paths."""
        return [t.get_path() for t in self.trajectories]
    
    def get_active_paths(self) -> List[List[Tuple[float, float]]]:
        """Get paths for active (in-bounds) trajectories."""
        return [t.get_path() for t in self.active_trajectories]
    
    def get_current_positions(self) -> np.ndarray:
        """Get current positions of all active trajectories."""
        if not self.active_trajectories:
            return np.array([])
        return np.array([(t.x, t.y) for t in self.active_trajectories])
    
    def get_statistics(self) -> dict:
        """Get statistics about the trajectory field."""
        return {
            'total_trajectories': len(self.trajectories),
            'active_trajectories': len(self.active_trajectories),
            'avg_time_steps': np.mean([t.time_steps for t in self.trajectories]) if self.trajectories else 0,
            'max_time_steps': max([t.time_steps for t in self.trajectories]) if self.trajectories else 0,
        }
