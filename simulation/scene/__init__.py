"""
Scene parsing and management for smoke simulation.

This module provides functionality for loading and managing burn scenes,
including parsing from files and coordinate transformations.
"""

from .scene_parser import BurnScene, parse_burn_area_file, parse_coords_file
from .perimeter_sampler import sample_perimeter_points, sample_perimeter_from_file

__all__ = [
    'BurnScene',
    'parse_burn_area_file',
    'parse_coords_file',
    'sample_perimeter_points',
    'sample_perimeter_from_file'
]
