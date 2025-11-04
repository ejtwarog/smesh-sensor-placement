"""
3D Plume visualization tools for smoke simulation data.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Union
import plotly.graph_objects as go
from plotly.graph_objects import Figure, Layout, scatter3d

@dataclass
class PlumeFromPointSource:
    """A class representing a Gaussian plume model from a point source.
    
    Attributes:
        Q: Emission rate (concentration × L³/T)
        h: Effective stack height (m)
        stability_class: Atmospheric stability class (A-F)
        source_pos: [x, y, z] coordinates of the source
    """
    Q: float
    h: float
    stability_class: str
    source_pos: List[float] = None
    
    def __post_init__(self):
        if self.source_pos is None:
            self.source_pos = [0.0, 0.0, 0.0]


def query_plume_model(plume: PlumeFromPointSource, point: List[float], wind_vector: List[float]) -> float:
    """
    Query the plume model at a specific 3D point.
    
    Args:
        plume: Plume model instance
        point: [x, y, z] coordinates to query
        wind_vector: [u, v, w] wind vector components
        
    Returns:
        Concentration at the specified point
    """
    # This is a simplified model - you'll need to implement the actual
    # Gaussian plume model calculations here based on your requirements
    x, y, z = point
    u, v, w = wind_vector
    
    # Simple falloff model - replace with actual Gaussian plume model
    distance = np.sqrt(x**2 + y**2 + (z - plume.h)**2)
    return (plume.Q / (4 * np.pi * distance)) * np.exp(-distance / 100.0)


def plot_single_plume_scene(
    plume: PlumeFromPointSource,
    bounds: List[float],
    wind_vector: List[float],
    x_num: int = 101,
    y_num: int = 103,
    z_num: int = 105,
    vmin: float = 1e-12,
    vmax: float = 3.0,
    output_html: str = "gaussian_plume.html"
) -> None:
    """
    Create a 3D visualization of a plume with value-dependent transparency.
    
    Args:
        plume: Plume model instance
        bounds: [x_min, x_max, y_min, y_max, z_min, z_max] spatial bounds
        wind_vector: [u, v, w] wind vector components
        x_num: Number of points in x-direction
        y_num: Number of points in y-direction
        z_num: Number of points in z-direction
        vmin: Minimum value for density thresholding
        vmax: Maximum value for scaling
        output_html: Output HTML file path
    """
    # Unpack bounds
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    z_max_display = z_max + 100.0  # Add some headroom above max z
    
    # Create coordinate grids
    x = np.linspace(x_min, x_max, x_num)
    y = np.linspace(y_min, y_max, y_num)
    z = np.linspace(z_min, z_max_display, z_num)
    
    # Pre-allocate density array
    density = np.zeros((x_num, y_num, z_num))
    
    # Fill density array
    for i in range(x_num):
        for j in range(y_num):
            for k in range(z_num):
                density[i, j, k] = query_plume_model(
                    plume, [x[i], y[j], z[k]], wind_vector
                )
    
    # Apply floor to avoid log of zeros/very small numbers
    density = np.maximum(density, vmin)
    log10_density = np.log10(density)
    
    # Print some diagnostics
    print(f"Density max: {np.max(density)}, min: {np.min(density)}")
    print(f"Log10 density range: {np.min(log10_density)}, {np.max(log10_density)}")
    
    # Downsample for visualization
    dx, dy, dz = 4, 4, 4  # Downsampling factors
    
    # Create downsampled arrays
    xs, ys, zs, vals = [], [], [], []
    
    for i in range(0, x_num, dx):
        for j in range(0, y_num, dy):
            for k in range(0, z_num, dz):
                xs.append(x[i])
                ys.append(y[j])
                zs.append(z[k])
                vals.append(log10_density[i, j, k])
    
    print(f"→ Scatter point count: {len(xs)}")
    
    # Normalize values for opacity mapping
    vals_arr = np.array(vals)
    min_val, max_val = np.min(vals_arr), np.max(vals_arr)
    normalized = (vals_arr - min_val) / (max_val - min_val)
    opacities = normalized  # Values in [0, 1]
    
    # Define custom colorscale (similar to Julia version)
    # Using rgba for colors to include transparency in the colorscale
    custom_colorscale = [
        [0.0, 'rgba(0, 0, 0, 0.0)'],         # fully transparent
        [0.01, 'rgba(255, 255, 255, 0.1)'],   # white with low opacity
        [0.5, 'rgba(164, 164, 164, 0.5)'],    # mid-gray with medium opacity
        [0.75, 'rgba(82, 82, 82, 0.8)'],      # dark gray with high opacity
        [1.0, 'rgba(0, 0, 0, 1.0)']           # black with full opacity
    ]
    
    # Create 3D scatter plot
    fig = Figure()
    
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers',
            marker=dict(
                size=3,
                color=vals,
                colorscale=custom_colorscale,
                cmin=min_val,
                cmax=max_val,
                colorbar=dict(title="log₁₀(density)"),
                opacity=0.7  # Set a single opacity value for all markers
            )
        )
    )
    
    # Update layout
    fig.update_layout(
        title="3D Scatter of log₁₀(density) (downsampled, value-dependent opacity)",
        scene=dict(
            xaxis_title="X (Downwind)",
            yaxis_title="Y (Crosswind)",
            zaxis_title="Z (Height)",
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            zaxis=dict(range=[z_min, z_max_display]),
            camera=dict(eye=dict(x=1.5, y=1.2, z=0.8))
        )
    )
    
    # Save to HTML
    fig.write_html(output_html)
    print(f"✔ 3D scatter plot saved to '{output_html}'")


def example_usage():
    """Example usage of the plume visualization."""
    # Define plume parameters
    Q = 10.0  # Emission rate (units: concentration × L³/T)
    h = 50.0  # Effective stack height (m)
    stability_class = "D"  # One of "A" through "F"
    
    # Create plume model
    source_pos = [0.0, 0.0, 0.0]  # Source at origin
    wind_global = [1.0, 0.0, 0.0]  # Wind along +X
    
    plume = PlumeFromPointSource(
        Q=Q,
        h=h,
        stability_class=stability_class,
        source_pos=source_pos
    )
    
    # Define spatial bounds [x_min, x_max, y_min, y_max, z_min, z_max]
    bounds = [
        -100.0,  # x_min
        100.0,   # x_max
        -50.0,   # y_min
        50.0,    # y_max
        0.0,     # z_min
        100.0    # z_max
    ]
    
    # Generate and save the visualization
    plot_single_plume_scene(
        plume=plume,
        bounds=bounds,
        wind_vector=wind_global,
        x_num=51,  # Reduced for faster testing
        y_num=51,
        z_num=26,
        vmin=1e-12,
        vmax=1e3,
        output_html="gaussian_plume_scatter.html"
    )


if __name__ == "__main__":
    example_usage()
