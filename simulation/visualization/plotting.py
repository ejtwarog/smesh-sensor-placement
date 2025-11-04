"""
Plotting utilities for smoke simulation visualization.

This module provides functions for creating various types of plots to visualize
smoke simulation results, including burn areas, smoke plumes, and concentration fields.
"""

from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm, Normalize
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from shapely.geometry import Polygon as ShapelyPolygon

from ..scene import BurnScene
from ..simulation import PlumeSimulator

logger = logging.getLogger(__name__)

def plot_burn_area(
    scene: BurnScene,
    time_step: int = -1,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    title: Optional[str] = None,
    **plot_kwargs
) -> plt.Figure:
    """Plot the burn area at the specified time step.
    
    Args:
        scene: BurnScene containing the burn area data
        time_step: Time step to plot (-1 for the last time step)
        ax: Optional matplotlib Axes to plot on
        show: Whether to call plt.show()
        title: Plot title
        **plot_kwargs: Additional arguments passed to matplotlib's plot function
        
    Returns:
        The matplotlib Figure containing the plot
    """
    if time_step < 0:
        time_step = len(scene.burn_polys) + time_step
    
    if not 0 <= time_step < len(scene.burn_polys):
        raise ValueError(f"Invalid time step {time_step}")
    
    burn_poly = scene.burn_polys[time_step]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Default plot style
    plot_kwargs.setdefault('color', 'red')
    plot_kwargs.setdefault('alpha', 0.5)
    plot_kwargs.setdefault('label', 'Burn Area')
    
    if burn_poly is not None and not burn_poly.is_empty:
        x, y = burn_poly.exterior.xy
        ax.fill(x, y, **plot_kwargs)
    
    # Plot sensor nodes if available
    if hasattr(scene, 'snode_locations') and scene.snode_locations:
        snode_x = [p[0] for p in scene.snode_locations]
        snode_y = [p[1] for p in scene.snode_locations]
        ax.scatter(snode_x, snode_y, c='blue', marker='o', label='Sensor Nodes')
    
    # Set plot properties
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    if title is None:
        title = f'Burn Area at Time Step {time_step}'
    ax.set_title(title)
    
    ax.legend()
    
    if show:
        plt.show()
    
    return fig
