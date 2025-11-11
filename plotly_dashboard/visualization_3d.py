"""
3D visualization components for the dashboard.
"""
from pathlib import Path
import plotly.graph_objects as go
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.plot_3d_scene import plot_burn_areas_3d

def create_3d_visualization(data_dir, location):
    """Create a 3D visualization of burn areas and sensor nodes.
    
    Args:
        data_dir: Path to the data directory
        location: Location name (e.g., 'HenryCoe')
        
    Returns:
        plotly.graph_objects.Figure: 3D visualization figure
    """
    # Define paths to data files
    base_dir = data_dir / location
    # Look for DEM file in the DEMs_and_Buffered_Burns directory
    dems_dir = data_dir / "DEMs_and_Buffered_Burns"
    dem_file = next(dems_dir.glob("*.tif"), None)
    print(f"Looking for DEM in: {dems_dir}")
    print(f"Found DEM file: {dem_file}")
    if dem_file is None:
        raise FileNotFoundError(f"No DEM file found in {dems_dir}")
    dem_path = dem_file
    snode_dir = base_dir / "SNodeLocations"
    burn_area_dir = base_dir / "BurnAreas"
    
    # Get list of burn area and snode files
    burn_files = list(burn_area_dir.glob("*.tif")) + list(burn_area_dir.glob("*.tiff"))
    snode_files = list(snode_dir.glob("*.json"))  # Look for JSON files instead of GeoJSON
    
    if not burn_files:
        return go.Figure(layout={
            'title': 'Error: No burn area files found',
            'title_x': 0.5
        })
        
    if not snode_files:
        # Try to find the first available SNODE file
        snode_files = list(snode_dir.glob("*"))
        if not snode_files:
            return go.Figure(layout={
                'title': 'Error: No SNODE files found',
                'title_x': 0.5
            })
    
    # For now, use the first snode file found
    snode_file = snode_files[0]
    
    try:
        # Generate the 3D plot
        # Create a temporary file for the output
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Generate the 3D plot
        fig = plot_burn_areas_3d(
            burn_files=burn_files,
            base_dem_path=dem_path,
            snode_file_path=snode_file,
            output_html=output_path,  # This is still needed by plot_burn_areas_3d
            cleanup_temp_files=True
        )
        
        # Ensure the figure has a proper layout
        if not fig.layout:
            fig.update_layout(
                title=f"3D Visualization - {location}",
                scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title='Elevation',
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
        # Add camera controls
        fig.update_scenes(
            xaxis_autorange=True,
            yaxis_autorange=True,
            zaxis_autorange=True
        )
        
        return fig
    except Exception as e:
        print(f"Error creating 3D visualization: {e}")
        # Return an empty figure with an error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading 3D visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
