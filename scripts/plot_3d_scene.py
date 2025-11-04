"""
3D visualization of burn areas with DEM and sensor nodes.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import numpy as np
import rasterio
import geopandas as gpd
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator

# Import local utilities
import sys
from pathlib import Path

# Add the simulation directory to the Python path
simulation_dir = Path(__file__).parent.parent / 'simulation'
sys.path.insert(0, str(simulation_dir))

# Now import the geo_utils module
from utils.geo_utils import (
    transform_raster,
    transform_geodataframe,
    get_crs_epsg
)

def plot_burn_areas_3d(
    burn_files,
    base_dem_path,
    snode_file_path,
    output_html='burnarea_smolder_slider.html',
    target_epsg: int = None,
    cleanup_temp_files: bool = True
):
    """Create an interactive 3D visualization of burn areas with DEM and sensor nodes.
    
    Args:
        burn_files: List of paths to burn area GeoTIFF files
        base_dem_path: Path to base DEM GeoTIFF file
        snode_file_path: Path to sensor node GeoJSON file
        output_html: Path to save the output HTML file
    
    Returns:
        plotly Figure object
    """
    # 1. Load and optionally transform base DEM
    temp_files = []
    try:
        if target_epsg is not None:
            # Create a temporary file for the transformed DEM
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
                temp_dem_path = temp_file.name
                temp_files.append(temp_dem_path)
            
            # Transform the DEM to target CRS
            transform_raster(base_dem_path, temp_dem_path, f'EPSG:{target_epsg}')
            dem_path = temp_dem_path
        else:
            dem_path = base_dem_path
            
        # Open the (possibly transformed) DEM
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype('float32')
            transform = src.transform
            crs = src.crs
    except Exception as e:
        # Clean up any temporary files if an error occurs
        if cleanup_temp_files:
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as cleanup_error:
                    print(f'Error cleaning up temporary file {temp_file}: {cleanup_error}')
        raise e
    
    # Convert no data values to NaN
    dem[dem <= 0] = np.nan
    
    # Create x and y coordinates
    nyd, nxd = dem.shape
    x0d, pxd, _, y0d, _, pyd = transform.to_gdal()
    xs_dem = x0d + pxd * np.arange(nxd)
    ys_dem = y0d + pyd * np.arange(nyd)
    
    # Create DEM surface trace
    dem_surface = go.Surface(
        x=xs_dem,
        y=ys_dem,
        z=dem,
        colorscale='Greys',
        showscale=False,
        opacity=0.7,
        name='Base DEM',
        showlegend=True
    )
    
    # 3. Load and transform sensor node locations if needed
    gdf = gpd.read_file(snode_file_path)
    
    # Transform nodes to target CRS if specified
    if target_epsg is not None:
        gdf = transform_geodataframe(gdf, f'EPSG:{target_epsg}')
    # Or to DEM CRS if no target specified
    elif gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    
    # Filter nodes to be within DEM bounds
    xmin, xmax = xs_dem.min(), xs_dem.max()
    ymin, ymax = ys_dem.min(), ys_dem.max()
    
    # Filter points within DEM bounds
    gdf = gdf.cx[xmin:xmax, ymin:ymax]
    
    # Get coordinates
    xs_snode = gdf.geometry.x.values
    ys_snode = gdf.geometry.y.values
    
    # 4. Interpolate elevations for sensor nodes
    # Create a grid of coordinates
    yy, xx = np.meshgrid(ys_dem, xs_dem, indexing='ij')
    
    # Flatten the arrays for interpolation
    points = np.column_stack((yy[~np.isnan(dem)], xx[~np.isnan(dem)]))
    values = dem[~np.isnan(dem)]
    
    # Create interpolation function
    itp = RegularGridInterpolator(
        (ys_dem, xs_dem),
        dem,
        method='nearest',
        bounds_error=False
    )
    
    # Interpolate z values for sensor nodes
    zs_snode = itp(np.column_stack((ys_snode, xs_snode)))
    
    # Create sensor node trace
    snode_trace = go.Scatter3d(
        x=xs_snode,
        y=ys_snode,
        z=zs_snode,
        mode='markers',
        marker=dict(
            size=4,
            color='blue',
            symbol='circle'
        ),
        name='SNODE'
    )
    
    # 5. Load burn areas
    active_surfaces = []
    smolder_surfaces = []
    
    # Process burn files
    for i, burn_file in enumerate(burn_files, 1):
        if target_epsg is not None:
            # Create a temporary file for the transformed burn area
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
                temp_burn_path = temp_file.name
                temp_files.append(temp_burn_path)
            
            # Transform the burn area to target CRS
            transform_raster(burn_file, temp_burn_path, f'EPSG:{target_epsg}')
            burn_path = temp_burn_path
        else:
            burn_path = burn_file
            
        with rasterio.open(burn_path) as src:
            burn_data = src.read(1).astype('float32')
            burn_transform = src.transform
            
        # Convert no data values to NaN
        burn_data[burn_data <= 0] = np.nan
        
        # Create coordinates
        x0, px, _, y0, _, py = burn_transform.to_gdal()
        ny, nx = burn_data.shape
        xs = x0 + px * np.arange(nx)
        ys = y0 + py * np.arange(ny)
        
        # Create surface traces for active burn
        # Add a small z-offset to prevent z-fighting
        # Since elevation is in meters and we're working with degrees, make this very small
        z_offset = 0.1  # meters
        active_surfaces.append(go.Surface(
            x=xs,
            y=ys,
            z=burn_data + z_offset,
            colorscale=[[0, 'rgba(255,0,0,0)'], [1, 'rgba(255,200,0,0.8)']],  # Brighter orange-red for better visibility
            surfacecolor=np.ones_like(burn_data),
            showscale=False,
            opacity=0.9,
            name=f'Burn {i} (active)',
            visible=(i == 0),  # Only first burn visible initially
            showlegend=True
        ))
        
        # Create surface traces for smoldering areas
        smolder_surfaces.append(go.Surface(
            x=xs,
            y=ys,
            z=burn_data + z_offset,
            colorscale=[[0, 'rgba(100,100,100,0)'], [1, 'rgba(50,50,50,0.6)']],  # Darker gray for better contrast
            surfacecolor=np.ones_like(burn_data) * 0.7,
            showscale=False,
            opacity=0.7,
            name=f'Burn {i} (smolder)',
            visible=False,  # None visible initially
            showlegend=True
        ))
    
    # 6. Combine all traces - DEM and SNODEs first, then burn areas
    traces = [dem_surface, snode_trace] + active_surfaces + smolder_surfaces
    
    # 7. Create slider steps
    n_burns = len(burn_files)
    steps = []
    
    # DEM and SNODEs are always visible (first two traces)
    base_visible = [True, True]
    
    for i in range(1, n_burns + 1):
        # Start with base visibility (DEM and SNODEs)
        vis = base_visible.copy()
        
        # Add active burn area (only one active at a time)
        vis += [j + 1 == i for j in range(n_burns)]
        
        # Add smolder layers (all previous burns except current)
        vis += [j + 1 < i for j in range(n_burns)]
        
        steps.append(dict(
            label=f'Burn {i}',
            method='update',
            args=[
                {'visible': vis},
                {'title': f'Burn Progression - Step {i}/{n_burns}'}
            ]
        ))
    
    # 8. Create layout with slider
    # Calculate data ranges for proper aspect ratio
    x_range = [xs_dem.min(), xs_dem.max()]
    y_range = [ys_dem.min(), ys_dem.max()]
    z_range = [np.nanmin(dem), np.nanmax(dem)]
    
    # Calculate spans for each axis
    x_span = x_range[1] - x_range[0]  # degrees longitude
    y_span = y_range[1] - y_range[0]  # degrees latitude
    z_span = z_range[1] - z_range[0]  # meters
    
    # Convert lat/long spans to approximate meters (at equator)
    # 1 degree ≈ 111,320 meters (for longitude at equator, less as you move toward poles)
    # For simplicity, we'll use this approximation since we're dealing with small areas
    x_span_m = x_span * 111320  # Approximate meters per degree at equator
    y_span_m = y_span * 111320  # Approximate meters per degree latitude
    
    # Calculate aspect ratio that works well for geographic data
    # Scale z-axis more aggressively to make elevation changes visible
    aspect_ratio = dict(
        x=1.0,  # Reference
        y=y_span/x_span if x_span > 0 else 1.0,  # Maintain lat/long ratio
        z=0.5 * (x_span_m + y_span_m) / (2 * z_span) if z_span > 0 else 0.1
    )
    
    layout = go.Layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Elevation (m)',
            aspectratio=aspect_ratio,
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=-1.5, z=0.8),  # Adjusted for better 3D view
                projection=dict(type='perspective')
            ),
            xaxis=dict(range=x_range, autorange=False),
            yaxis=dict(range=y_range, autorange=False),
            zaxis=dict(range=z_range, autorange=False)
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=50),
        sliders=[{
            'active': 0,
            'currentvalue': {'prefix': 'Time Step: '},
            'pad': {'t': 50},
            'steps': steps,
            'transition': {'duration': 300, 'easing': 'cubic-in-out'}
        }],
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': '▶️',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
            }],
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }]
    )
    
    # 9. Set initial visibility (first burn active, no smoldering yet)
    initial_visibility = [True, True]  # DEM and SNODEs
    initial_visibility += [i == 0 for i in range(n_burns)]  # First burn active
    initial_visibility += [False] * n_burns  # No smoldering initially
    
    # Update traces with initial visibility
    for i, trace in enumerate(traces):
        trace.visible = initial_visibility[i] if i < len(initial_visibility) else False
    
    # 10. Create and save figure
    fig = go.Figure(data=traces, layout=layout)
    
    # Set initial title
    fig.update_layout(title_text='Burn Progression - Step 1/{}'.format(n_burns))
    
    fig.write_html(output_html)
    print(f'✔ 3-D burn progression visualization saved to {output_html}')
    
    # Clean up temporary files if requested
    if cleanup_temp_files:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f'Warning: Could not remove temporary file {temp_file}: {e}')
    
    return fig


def main():
    """Example usage of the plot_burn_areas_3d function."""
    # File paths - update these to your actual file paths
    burn_file_dir = 'data/BurnData/HenryCoe/BurnAreas/'
    burn_files = [
        'BurnArea1.tif', 'BurnArea2.tif', 'BurnArea3.tif',
        'BurnArea4.tif', 'BurnArea5.tif', 'BurnArea6.tif'
    ]
    burn_files = [os.path.join(burn_file_dir, f) for f in burn_files]
    
    base_dem_path = 'data/BurnData/DEMs_and_Buffered_Burns/HenryCoe_Updated.tif'
    snode_file_path = 'data/BurnData/HenryCoe/SNodeLocations/Sim1.json'
    
    # Target EPSG code (e.g., 3857 for Web Mercator, 4326 for WGS84)
    # Set to None to keep original CRS
    target_epsg = 3857  # Web Mercator is good for web visualization
    
    # Create the plot with coordinate transformation
    plot_burn_areas_3d(
        burn_files=burn_files,
        base_dem_path=base_dem_path,
        snode_file_path=snode_file_path,
        target_epsg=target_epsg,
        cleanup_temp_files=True
    )


if __name__ == '__main__':
    main()
