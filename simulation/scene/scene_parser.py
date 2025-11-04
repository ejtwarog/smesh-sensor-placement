"""
Scene parsing and management for smoke simulation.

This module provides functionality for parsing burn scene data from files
and managing the state of burn scenes over time.
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from shapely.geometry import Polygon as ShapelyPolygon, Point
import logging

logger = logging.getLogger(__name__)

# Regular expression for matching coordinate strings
COORD_REGEX = re.compile(r"\[([\d\.\-+eE,\s]+)\]")

@dataclass
class BurnScene:
    """Class representing a burn scene with multiple time steps of burn polygons.
    
    Attributes:
        burn_polys: List of Shapely Polygons representing the burn area at each time step
        time_steps: List of time step indices
        snode_locations: List of sensor node locations as [x, y] coordinates
        reference_bounds: Tuple of (x_min, x_max, y_min, y_max) for the scene bounds
        perimeter_sample_points: List of points sampled along the perimeter
        emission_rate: Emission rate Q in g/acre/sec
    """
    burn_polys: List[ShapelyPolygon] = field(default_factory=list)
    time_steps: List[int] = field(default_factory=list)
    snode_locations: List[List[float]] = field(default_factory=list)
    reference_bounds: Tuple[float, float, float, float] = (0, 0, 0, 0)
    perimeter_sample_points: List[List[float]] = field(default_factory=list)
    emission_rate: float = 0.0
    
    def __post_init__(self):
        """Initialize time steps if not provided."""
        if not self.time_steps and self.burn_polys:
            self.time_steps = list(range(len(self.burn_polys)))
    
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
        
        # Update time steps if needed
        if time_step not in self.time_steps:
            self.time_steps.append(time_step)
            self.time_steps.sort()
    
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
    
    def get_cumulative_burn_area(self, current_time_step: int) -> Optional[ShapelyPolygon]:
        """Get the cumulative burn area up to the current time step.
        
        Args:
            current_time_step: Current time step index
            
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
        from shapely.ops import unary_union
        return unary_union(valid_polys)

def parse_coord(coord_str: str) -> Tuple[float, float]:
    """Parse a coordinate string of the form "x, y" into a tuple (x, y)."""
    try:
        x, y = map(float, (s.strip() for s in coord_str.split(',')))
        return (x, y)
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing coordinate string '{coord_str}': {e}")
        return (0.0, 0.0)

def parse_coords_file(filename: str, start_line: int = 2) -> List[List[float]]:
    """Parse a file containing coordinate data.
    
    Args:
        filename: Path to the coordinates file
        start_line: Line number to start parsing from (0-based)
        
    Returns:
        List of coordinate pairs as [x, y] lists
    """
    coords = []
    try:
        with open(filename, 'r') as f:
            for line in f.readlines()[start_line:]:
                match = COORD_REGEX.search(line)
                if match:
                    coord = parse_coord(match.group(1))
                    coords.append(list(coord))
    except Exception as e:
        logger.error(f"Error reading coordinates file {filename}: {e}")
    
    return coords

def parse_burn_area_file(filename: str) -> List[ShapelyPolygon]:
    """Parse a burn area file containing polygon data.
    
    Args:
        filename: Path to the burn area file
        
    Returns:
        List of Shapely Polygons representing burn areas at each time step
    """
    polygons = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            # Skip header and total area lines
            for line in lines[2:]:
                coords = []
                for match in COORD_REGEX.finditer(line):
                    coord = parse_coord(match.group(1))
                    coords.append(coord)
                
                if coords and coords[0] == coords[-1]:  # Close the polygon if needed
                    coords = coords[:-1]
                
                if len(coords) >= 3:  # Need at least 3 points for a valid polygon
                    polygons.append(ShapelyPolygon(coords))
    except Exception as e:
        logger.error(f"Error parsing burn area file {filename}: {e}")
    
    return polygons

def create_burn_scene(
    burn_area_file: str,
    snode_locations_file: str,
    reference_bounds_file: Optional[str] = None,
    total_burn_area_file: Optional[str] = None,
    n_perimeter_samples: int = 100
) -> BurnScene:
    """Create a BurnScene from input files.
    
    Args:
        burn_area_file: Path to the burn area file
        snode_locations_file: Path to the sensor node locations file
        reference_bounds_file: Optional path to the reference bounds file
        total_burn_area_file: Optional path to the total burn area file
        n_perimeter_samples: Number of points to sample along the perimeter
        
    Returns:
        Initialized BurnScene object
    """
    # Parse burn polygons
    burn_polys = parse_burn_area_file(burn_area_file)
    if not burn_polys:
        logger.warning(f"No valid burn polygons found in {burn_area_file}")
    
    # Parse sensor node locations
    snode_locations = parse_coords_file(snode_locations_file, start_line=2)
    
    # Parse reference bounds if provided
    reference_bounds = (0.0, 0.0, 0.0, 0.0)
    if reference_bounds_file:
        bounds = parse_coords_file(reference_bounds_file, start_line=0)
        if len(bounds) >= 2:
            x_coords = [x for x, _ in bounds]
            y_coords = [y for _, y in bounds]
            reference_bounds = (min(x_coords), max(x_coords), min(y_coords), max(y_coords))
    
    # Sample perimeter points if total burn area file is provided
    perimeter_samples = []
    if total_burn_area_file:
        try:
            perimeter_samples = sample_perimeter_from_file(total_burn_area_file, n=n_perimeter_samples)
        except Exception as e:
            logger.error(f"Error sampling perimeter from {total_burn_area_file}: {e}")
    
    # Create and return the burn scene
    scene = BurnScene(
        burn_polys=burn_polys,
        snode_locations=snode_locations,
        reference_bounds=reference_bounds,
        perimeter_sample_points=perimeter_samples
    )
    
    return scene
