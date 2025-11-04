"""
Smoke point sampling for burn regions.

This module provides functionality for sampling points from burn regions
to simulate smoke sources in wildfire modeling.
"""

from typing import List, Optional, Union, Tuple
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, Point
from shapely.ops import unary_union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sample_points_from_polygon(
    polygon: ShapelyPolygon,
    n_samples: int,
    max_attempts: int = 10,
    verbose: bool = False
) -> np.ndarray:
    """Sample points from the interior of a polygon.
    
    Args:
        polygon: Input polygon to sample from
        n_samples: Number of points to sample
        max_attempts: Maximum number of sampling attempts
        verbose: Whether to print debug information
        
    Returns:
        Array of sampled points with shape (n_samples, 2)
    """
    if not isinstance(polygon, ShapelyPolygon):
        raise TypeError("Input must be a Shapely Polygon")
    
    if n_samples <= 0:
        return np.zeros((0, 2))
    
    # Get the bounds of the polygon
    min_x, min_y, max_x, max_y = polygon.bounds
    
    points = []
    attempts = 0
    
    while len(points) < n_samples and attempts < max_attempts:
        # Generate random points within the bounding box
        x = np.random.uniform(min_x, max_x, n_samples * 2)
        y = np.random.uniform(min_y, max_y, n_samples * 2)
        
        # Check which points are inside the polygon
        for xi, yi in zip(x, y):
            point = Point(xi, yi)
            if polygon.contains(point):
                points.append([xi, yi])
                if len(points) >= n_samples:
                    break
        
        attempts += 1
    
    if len(points) < n_samples and verbose:
        logger.warning(
            f"Only sampled {len(points)}/{n_samples} points after "
            f"{max_attempts} attempts"
        )
    
    return np.array(points[:n_samples])


class BurnScene:
    """Class representing a burn scene with multiple time steps of burn polygons."""
    
    def __init__(self):
        """Initialize an empty burn scene."""
        self.burn_polys = []  # List of Shapely Polygons
    
    def add_burn_polygon(self, polygon: ShapelyPolygon, time_step: int):
        """Add a burn polygon at a specific time step.
        
        Args:
            polygon: Polygon representing the burn area
            time_step: Time step index (0-based)
        """
        # Ensure we have enough space in the list
        while len(self.burn_polys) <= time_step:
            self.burn_polys.append(None)
        
        self.burn_polys[time_step] = polygon
    
    def get_burn_polygon(self, time_step: int) -> Optional[ShapelyPolygon]:
        """Get the burn polygon at a specific time step.
        
        Args:
            time_step: Time step index (0-based)
            
        Returns:
            Burn polygon at the specified time step, or None if not available
        """
        if 0 <= time_step < len(self.burn_polys):
            return self.burn_polys[time_step]
        return None
    
    def sample_smoke_points(
        self,
        n_samples_per_step: int,
        current_time_step: int,
        max_attempts: int = 10,
        verbose: bool = False
    ) -> np.ndarray:
        """Sample smoke points from all burn polygons up to the current time step.
        
        Args:
            n_samples_per_step: Number of samples to generate per time step
            current_time_step: Current time step (samples from previous steps)
            max_attempts: Maximum number of sampling attempts per polygon
            verbose: Whether to print debug information
            
        Returns:
            Array of sampled points with shape (n_points, 2)
        """
        if current_time_step <= 0:
            return np.zeros((0, 2))
        
        all_points = []
        
        for t in range(min(current_time_step, len(self.burn_polys))):
            poly = self.burn_polys[t]
            if poly is not None:
                points = sample_points_from_polygon(
                    poly, n_samples_per_step, max_attempts, verbose
                )
                all_points.append(points)
        
        if not all_points:
            return np.zeros((0, 2))
        
        return np.vstack(all_points)
    
    def get_cumulative_burn_area(
        self,
        current_time_step: int
    ) -> Optional[ShapelyPolygon]:
        """Get the cumulative burn area up to the current time step.
        
        Args:
            current_time_step: Current time step (0-based)
            
        Returns:
            Combined polygon of all burn areas up to current_time_step
        """
        if current_time_step <= 0 or not self.burn_polys:
            return None
            
        # Get all non-None polygons up to current_time_step
        valid_polys = [
            p for p in self.burn_polys[:current_time_step]
            if p is not None and not p.is_empty
        ]
        
        if not valid_polys:
            return None
            
        # Combine all polygons using unary_union
        return unary_union(valid_polys)


def create_burn_scene_from_polygons(
    polygons: List[ShapelyPolygon]
) -> BurnScene:
    """Create a burn scene from a list of polygons.
    
    Args:
        polygons: List of Shapely Polygons, one per time step
        
    Returns:
        Initialized BurnScene object
    """
    scene = BurnScene()
    for t, poly in enumerate(polygons):
        if poly is not None:
            scene.add_burn_polygon(poly, t)
    return scene
