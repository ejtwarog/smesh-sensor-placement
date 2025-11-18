"""
Core functionality for smoke simulation and analysis.

Active Modules:
- drift_scene: Wind drift dashboard visualization
- terrain_vectors: DEM and terrain vector field
- wind_trajectories: Particle trajectory simulation
- smoke_points: Burn area sampling
- plume_model: Atmospheric stability classes
"""

from .terrain_vectors import TerrainVectorField
from .plume_model import AIR_COLUMN_STABILITY_CLASSES

__all__ = [
    "TerrainVectorField",
    "AIR_COLUMN_STABILITY_CLASSES"
]
