"""
Visualization tools for smoke simulation.

This package provides functionality for visualizing smoke plumes, fire perimeters,
and other simulation results in 2D and 3D.
"""

from .plot_3d_plume import plot_3d_plume
from .plot_3d_scene import plot_3d_scene
from .plotting import plot_plume, plot_scene, animate_plume, animate_scene
from .plume_plotting import plot_plume_2d, plot_plume_3d

__all__ = [
    'plot_3d_plume',
    'plot_3d_scene',
    'plot_plume',
    'plot_scene',
    'animate_plume',
    'animate_scene',
    'plot_plume_2d',
    'plot_plume_3d'
]
