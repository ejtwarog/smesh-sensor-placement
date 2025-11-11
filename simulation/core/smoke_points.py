"""
Smoke point sampling for burn regions.

This module provides functionality for sampling points from burn regions
to simulate smoke sources in wildfire modeling.
"""

from typing import List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, Point
from shapely.ops import unary_union
import logging
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parents[2]  # Go up to smesh-sensor-placement
sys.path.insert(0, str(project_root))

from simulation.utils.geo_utils import load_geojson_polygons, sample_points_in_polygon

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
    
    def sample_points_in_burn_area_at_time(
        self,
        time_step: int,
        num_samples: int,
        transform,
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Sample points from the burn area at a specific time step in pixel coordinates.
        
        Args:
            time_step: Time step index (0-based)
            num_samples: Number of points to sample
            transform: Rasterio transform for coordinate conversion
            height: Raster height in pixels
            width: Raster width in pixels
            
        Returns:
            Array of shape (num_samples, 2) with pixel coordinates [col, row]
        """
        poly = self.get_burn_polygon(time_step)
        if poly is None or poly.is_empty:
            return np.array([])
        
        # Convert Shapely polygon to list of (lon, lat) tuples
        exterior_coords = list(poly.exterior.coords)
        
        # Use the geo_utils function to sample
        sampled_pts = sample_points_in_polygon(
            exterior_coords, num_samples, transform, height, width
        )
        return sampled_pts
    
    def sample_points_in_cumulative_burn_area(
        self,
        current_time_step: int,
        num_samples: int,
        transform,
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Sample points from the cumulative burn area up to a specific time step.
        
        Args:
            current_time_step: Current time step (0-based)
            num_samples: Number of points to sample
            transform: Rasterio transform for coordinate conversion
            height: Raster height in pixels
            width: Raster width in pixels
            
        Returns:
            Array of shape (num_samples, 2) with pixel coordinates [col, row]
        """
        cumulative_poly = self.get_cumulative_burn_area(current_time_step)
        if cumulative_poly is None or cumulative_poly.is_empty:
            return np.array([])
        
        # Convert Shapely polygon to list of (lon, lat) tuples
        if hasattr(cumulative_poly, 'exterior'):
            # Single polygon
            exterior_coords = list(cumulative_poly.exterior.coords)
        else:
            # MultiPolygon - use the first polygon
            exterior_coords = list(cumulative_poly.geoms[0].exterior.coords)
        
        # Use the geo_utils function to sample
        sampled_pts = sample_points_in_polygon(
            exterior_coords, num_samples, transform, height, width
        )
        return sampled_pts
    
    def initialize_trajectories_for_burn_area(
        self,
        time_step: int,
        num_samples: int,
        transform,
        height: int,
        width: int,
        initial_course_deg: float = 270
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points from burn area and initialize trajectories for them.
        
        Args:
            time_step: Time step index (0-based)
            num_samples: Number of points to sample
            transform: Rasterio transform for coordinate conversion
            height: Raster height in pixels
            width: Raster width in pixels
            initial_course_deg: Initial course in degrees (0=N, 90=E, 180=S, 270=W)
            
        Returns:
            Tuple of (sampled_points, initial_directions) arrays
        """
        sampled_pts = self.sample_points_in_burn_area_at_time(
            time_step, num_samples, transform, height, width
        )
        
        if len(sampled_pts) == 0:
            return np.array([]), np.array([])
        
        directions = initialize_particle_trajectories_from_course(
            sampled_pts, initial_course_deg
        )
        
        return sampled_pts, directions


def sample_points_in_burn_area(
    geojson_path: str,
    num_samples: int,
    transform,
    height: int,
    width: int
) -> np.ndarray:
    """
    Randomly sample points within a burn area polygon from a GeoJSON file.
    
    Args:
        geojson_path: Path to GeoJSON file with burn area polygon
        num_samples: Number of points to sample
        transform: Rasterio transform for coordinate conversion
        height: Raster height in pixels
        width: Raster width in pixels
        
    Returns:
        Array of shape (num_samples, 2) with pixel coordinates [col, row]
    """
    polygons = load_geojson_polygons(geojson_path)
    if not polygons:
        return np.array([])
    
    # Use first polygon
    sampled_pts = sample_points_in_polygon(
        polygons[0], num_samples, transform, height, width
    )
    return sampled_pts


def initialize_particle_trajectories_from_course(
    sampled_points: np.ndarray,
    initial_course_deg: float = 270
) -> np.ndarray:
    """
    Initialize particle trajectories at sampled points with a fixed course direction.
    
    Args:
        sampled_points: Array of shape (N, 2) with pixel coordinates [col, row]
        initial_course_deg: Initial course in degrees (0=N, 90=E, 180=S, 270=W)
        
    Returns:
        Array of shape (N, 2) with initial direction vectors [u, v]
    """
    if len(sampled_points) == 0:
        return np.array([])
    
    # Convert course to radians
    th = np.deg2rad(initial_course_deg)
    u = -np.sin(th)  # eastward
    v = -np.cos(th)  # northward
    direction_vector = np.array([u, v])
    
    # Return the same direction for all particles
    trajectories = np.tile(direction_vector, (len(sampled_points), 1))
    
    return trajectories
