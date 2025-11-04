"""
Plume visualization utilities for smoke simulation.

This module provides functions for visualizing smoke plumes and concentration fields.
"""

from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import logging

from shapely.geometry import Polygon as ShapelyPolygon

from ..scene import BurnScene
from ..simulation import PlumeSimulator

logger = logging.getLogger(__name__)

# Custom colormap for smoke plumes
SMOKE_CMAP = LinearSegmentedColormap.from_list(
    'smoke', 
    ['#00000000', '#f0f0f0', '#d0d0d0', '#b0b0b0', '#808080', '#505050', '#202020']
)

def plot_smoke_plume(
    simulator: PlumeSimulator,
    time_step: int,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    resolution: int = 100,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    title: Optional[str] = None,
    **plot_kwargs
) -> plt.Figure:
    """Plot the smoke plume concentration field at the specified time step.
    
    Args:
        simulator: PlumeSimulator instance
        time_step: Time step to plot
        x_lim: X-axis limits (min, max)
        y_lim: Y-axis limits (min, max)
        resolution: Number of grid points in each dimension
        ax: Optional matplotlib Axes to plot on
        show: Whether to call plt.show()
        title: Plot title
        **plot_kwargs: Additional arguments passed to pcolormesh
        
    Returns:
        The matplotlib Figure containing the plot
    """
    # Get the burn scene for reference
    scene = simulator.smoke_simulator.scene
    
    # Determine plot limits if not provided
    if x_lim is None or y_lim is None:
        # Get bounds from burn area if available
        if scene.burn_polys and scene.burn_polys[0] is not None:
            x_min, y_min, x_max, y_max = scene.burn_polys[0].bounds
            padding = max(x_max - x_min, y_max - y_min) * 0.2  # 20% padding
            x_lim = (x_min - padding, x_max + padding)
            y_lim = (y_min - padding, y_max + padding)
        else:
            x_lim = (-100, 100)
            y_lim = (-100, 100)
    
    # Create grid for concentration field
    x = np.linspace(x_lim[0], x_lim[1], resolution)
    y = np.linspace(y_lim[0], y_lim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate concentration at each grid point
    Z = simulator.get_concentration(X, Y, time_step)
    
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Default plot style
    plot_kwargs.setdefault('cmap', SMOKE_CMAP)
    plot_kwargs.setdefault('norm', LogNorm(vmin=1e-6, vmax=1.0))
    
    # Plot concentration field
    im = ax.pcolormesh(X, Y, Z, shading='auto', **plot_kwargs)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Concentration (g/m³)')
    
    # Plot burn area if available
    if time_step < len(scene.burn_polys) and scene.burn_polys[time_step] is not None:
        burn_poly = scene.burn_polys[time_step]
        if hasattr(burn_poly, 'exterior'):
            x, y = burn_poly.exterior.xy
            ax.fill(x, y, color='red', alpha=0.5, label='Burn Area')
    
    # Plot sensor nodes if available
    if hasattr(scene, 'snode_locations') and scene.snode_locations:
        snode_x = [p[0] for p in scene.snode_locations]
        snode_y = [p[1] for p in scene.snode_locations]
        ax.scatter(snode_x, snode_y, c='blue', marker='o', label='Sensor Nodes')
    
    # Set plot properties
    ax.set_aspect('equal')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    if title is None:
        title = f'Smoke Plume at Time Step {time_step}'
    ax.set_title(title)
    
    ax.legend()
    
    if show:
        plt.show()
    
    return fig
