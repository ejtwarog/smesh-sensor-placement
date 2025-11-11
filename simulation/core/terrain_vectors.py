import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import rasterio.transform

# Add the project root to the Python path
project_root = Path(__file__).parents[2]  # Go up to smesh-sensor-placement
sys.path.insert(0, str(project_root))

# Now we can import from simulation
from simulation.utils.geo_utils import transform_raster, load_geojson_polygons
from simulation.core.smoke_points import sample_points_in_burn_area, initialize_particle_trajectories_from_course


class TerrainVectorField:
    """
    A terrain-based vector field that generates unit vectors pointing downslope.
    Vectors are purely based on terrain gradients with no wind component.
    """
    
    def __init__(self, dem_path: str, target_epsg: int = 4326, sampling_resolution_m: float = 1.0):
        """
        Initialize the terrain vector field from a DEM.
        
        Args:
            dem_path: Path to DEM GeoTIFF file
            target_epsg: Target CRS for DEM transformation (default: 4326 for lat/lon)
            sampling_resolution_m: Resolution of each DEM cell in meters
        """
        self.dem_path = dem_path
        self.target_epsg = target_epsg
        self.sampling_resolution_m = sampling_resolution_m
        self.dem = None
        self.vectors = None
        self.transform = None
        self.crs = None
        self.height = None
        self.width = None
        
        self._load_dem()
        self._compute_vectors()
    
    def _load_dem(self):
        """Load and transform DEM if needed."""
        temp_files = []
        
        try:
            # Handle coordinate transformation if needed
            if self.target_epsg is not None:
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
                    temp_dem_path = temp_file.name
                    temp_files.append(temp_dem_path)
                
                print(f"Transforming DEM to EPSG:{self.target_epsg}...")
                transform_raster(str(self.dem_path), temp_dem_path, f'EPSG:{self.target_epsg}')
                dem_path_to_load = temp_dem_path
            else:
                dem_path_to_load = self.dem_path
            
            # Open the (possibly transformed) DEM
            with rasterio.open(dem_path_to_load) as src:
                self.dem = src.read(1).astype('float32')
                self.crs = src.crs
                self.transform = src.transform
                print(f"DEM loaded. CRS: {self.crs}, Shape: {self.dem.shape}")
            
            # Convert no data values to NaN
            self.dem[self.dem <= 0] = np.nan
            self.height, self.width = self.dem.shape
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    print(f'Error cleaning up temporary file {temp_file}: {e}')
    
    def _compute_vectors(self):
        """Compute terrain-based unit vectors."""
        if self.dem is None:
            raise ValueError("DEM not loaded")
        
        self.vectors = generate_2d_wind_vectors(
            terrain_dem=self.dem,
            sampling_resolution_m=self.sampling_resolution_m,
        )
    
    def get_vector_at(self, col: float, row: float) -> np.ndarray:
        """
        Get the terrain vector at a specific pixel location.
        
        Args:
            col: Column (x) coordinate
            row: Row (y) coordinate
            
        Returns:
            (u, v) unit vector at that location
        """
        return get_vector_at_point(self.vectors, col, row)
    


def calculate_slope_components(dem: np.ndarray, resolution_m: float):
    """
    Returns dz/dx and dz/dy (m/m). Assumes x = Easting, y = Northing.
    """
    # np.gradient returns gradients along each axis: axis=0 -> rows (y), axis=1 -> cols (x)
    dZ_dy, dZ_dx = np.gradient(dem, resolution_m, resolution_m)
    return dZ_dx, dZ_dy

def generate_2d_wind_vectors(
    terrain_dem: np.ndarray,
    sampling_resolution_m: float = 1.0,
    eps: float = 1e-9,
):
    """
    Produce a (H, W, 2) array of terrain-based unit vectors (u, v) based purely on topography.
    
    Vectors point in the upslope direction with unit magnitude (length 1).

    Args:
        terrain_dem: Digital Elevation Model array (H, W)
        sampling_resolution_m: Resolution of each DEM cell in meters
        eps: Small value to prevent division by zero
        
    Returns:
        (H, W, 2) array of unit terrain vectors (u, v)
    """
    dem = np.asarray(terrain_dem, dtype=float)
    H, W = dem.shape

    # Terrain gradients
    dzdx, dzdy = calculate_slope_components(dem, sampling_resolution_m)  # (H,W) each

    # Upslope direction at each cell: positive gradient (toward increasing elevation)
    up_u = dzdx
    up_v = dzdy
    
    # Normalize to unit vectors
    up_mag = np.sqrt(up_u**2 + up_v**2) + eps
    U = up_u / up_mag
    V = up_v / up_mag

    # Stack to (H, W, 2) -> (u, v)
    terrain_vectors = np.stack([U, V], axis=-1)
    return terrain_vectors


def get_vector_at_point(vectors, col, row):
    """
    Get the terrain vector at a specific pixel location.
    
    Args:
        vectors: (H, W, 2) array of terrain vectors
        col: Column (x) coordinate
        row: Row (y) coordinate
        
    Returns:
        (u, v) vector at that location, or (0, 0) if out of bounds or NaN
    """
    # Handle NaN values
    if np.isnan(col) or np.isnan(row):
        return np.array([0.0, 0.0])
    
    row = int(np.round(row))
    col = int(np.round(col))
    
    if 0 <= row < vectors.shape[0] and 0 <= col < vectors.shape[1]:
        vec = vectors[row, col]
        # Check if vector contains NaN
        if np.any(np.isnan(vec)):
            return np.array([0.0, 0.0])
        return vec
    return np.array([0.0, 0.0])


def visualize_wind_field(
    dem_path: str,
    target_epsg: int = 4326,
    stride: int = 2,
    grid_spacing: int = None,
    show_grid: bool = True,
    burn_area_geojson: str = None,
    num_samples: int = 0,
    return_trajectories: bool = False,
):
    """
    Visualize terrain-based unit vectors with optional burn area and sampled points.
    
    Args:
        dem_path: Path to DEM GeoTIFF file
        target_epsg: Target CRS for DEM transformation
        stride: Subsample stride for visualization
        grid_spacing: Grid cell spacing in pixels (default: same as stride)
        show_grid: Whether to show grid lines overlay (default: True)
        burn_area_geojson: Path to GeoJSON file with burn area polygon (optional)
        num_samples: Number of random points to sample within burn area (default: 0)
        return_trajectories: Whether to return sampled points and their initial directions
        
    Returns:
        If return_trajectories is True: (sampled_points, initial_directions)
        Otherwise: None
    """
    # Create terrain vector field
    tvf = TerrainVectorField(dem_path, target_epsg=target_epsg)
    
    dem = tvf.dem
    wind = tvf.vectors
    height, width = tvf.height, tvf.width
    dem_path_to_load = dem_path
    
    sampled_points_all = None
    initial_directions_all = None
    
    try:
        # Create coordinate grids
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Subsample for better visualization
        X_sub = X[::stride, ::stride]
        Y_sub = Y[::stride, ::stride]
        wind_u_sub = wind[::stride, ::stride, 0]
        wind_v_sub = wind[::stride, ::stride, 1]
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Plot DEM
        dem_plot = plt.contourf(X, Y, dem, levels=20, cmap='terrain')
        plt.colorbar(dem_plot, label='Elevation (m)')
        
        # Plot wind vectors
        # wind_u is eastward, wind_v is northward
        # Since we invert the y-axis below, negate v-component so vectors point correctly
        plt.quiver(X_sub, Y_sub, wind_u_sub, -wind_v_sub,
                  color='red', scale=100, width=0.002, headwidth=3)
        
        # Add grid overlay if requested
        if show_grid:
            grid_spacing = grid_spacing if grid_spacing is not None else stride
            # Draw vertical grid lines
            for x_line in np.arange(0, width, grid_spacing):
                plt.axvline(x=x_line, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            # Draw horizontal grid lines
            for y_line in np.arange(0, height, grid_spacing):
                plt.axhline(y=y_line, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Add burn area overlay if provided
        if burn_area_geojson and os.path.exists(burn_area_geojson):
            # Load burn area polygons
            polygons = load_geojson_polygons(burn_area_geojson)
            
            for polygon_coords in polygons:
                # Convert lat/lon coordinates to pixel coordinates
                pixel_coords = []
                for lon, lat in polygon_coords:
                    # Use rasterio transform to convert geographic to pixel coordinates
                    row, col = rasterio.transform.rowcol(tvf.transform, lon, lat)
                    # Only include points that are within the DEM bounds
                    if 0 <= row < height and 0 <= col < width:
                        pixel_coords.append([col, row])
                
                if pixel_coords:
                    pixel_coords = np.array(pixel_coords)
                    # Plot the polygon boundary
                    plt.plot(pixel_coords[:, 0], pixel_coords[:, 1], 
                            color='blue', linewidth=2, label='Burn Area')
                    
                    # Sample random points within the polygon if requested
                    if num_samples > 0:
                        sampled_pts = sample_points_in_burn_area(burn_area_geojson, num_samples, tvf.transform, height, width)
                        if len(sampled_pts) > 0:
                            plt.scatter(sampled_pts[:, 0], sampled_pts[:, 1], 
                                       color='green', s=30, marker='o', 
                                       label=f'Sampled Points ({len(sampled_pts)})', zorder=5)
                            
                            # Initialize particle trajectories with fixed course direction
                            if return_trajectories:
                                sampled_points_all = sampled_pts
                                initial_directions_all = initialize_particle_trajectories_from_course(sampled_pts, initial_course_deg=270)
                                
                                # Plot trajectory arrows from each sampled point
                                arrow_length = 5  # pixels
                                for i, (col, row) in enumerate(sampled_pts):
                                    u, v = initial_directions_all[i]
                                    # Negate v to account for inverted y-axis
                                    plt.arrow(col, row, u * arrow_length, -v * arrow_length,
                                             head_width=1.5, head_length=1, fc='cyan', ec='cyan', 
                                             linewidth=1.5, zorder=4)
        
        plt.gca().invert_yaxis()  # Invert y-axis so north is up
        
        # Format title
        plt.title("Terrain-based Unit Vectors (Upslope Direction)")
        plt.xlabel("Easting (pixels)")
        plt.ylabel("Northing (pixels)")
        plt.tight_layout()
        plt.show()
        
        # Return trajectories if requested
        if return_trajectories and sampled_points_all is not None:
            return sampled_points_all, initial_directions_all
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        raise


if __name__ == "__main__":
    # Path to DEM
    base_dem_path = "/Users/evantwarog/Documents/Coursework/AA228V/smesh-sensor-placement/data/BurnData/DEMs_and_Buffered_Burns/DEM_HenryCoe.tif"
    
    # Path to burn area
    burn_area_path = "/Users/evantwarog/Documents/Coursework/AA228V/smesh-sensor-placement/data/BurnData/HenryCoe/BurnAreas/BurnArea2.json"
    
    # Visualize terrain-based unit vectors with burn area overlay and sampled points
    visualize_wind_field(
        dem_path=base_dem_path,
        burn_area_geojson=burn_area_path,
        num_samples=5,  # Sample 50 random points within the burn area
    )
