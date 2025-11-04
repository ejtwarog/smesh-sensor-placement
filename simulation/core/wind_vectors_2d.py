import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[2]  # Go up to smoke_sim_py
sys.path.insert(0, str(project_root))

# Now we can import from smoke_sim
from smoke_sim.utils.geo_utils import transform_raster


def calculate_slope_components(dem: np.ndarray, resolution_m: float):
    """
    Returns dz/dx and dz/dy (m/m). Assumes x = Easting, y = Northing.
    """
    # np.gradient returns gradients along each axis: axis=0 -> rows (y), axis=1 -> cols (x)
    dZ_dy, dZ_dx = np.gradient(dem, resolution_m, resolution_m)
    return dZ_dx, dZ_dy

def unit_vector_from_met_direction(deg_from_north: float):
    """
    Meteorological convention: direction is the angle FROM which the wind blows,
    measured clockwise from North. Returns the *toward* unit vector (u east, v north).
    """
    th = np.deg2rad(deg_from_north)
    u = -np.sin(th)  # eastward
    v = -np.cos(th)  # northward
    return np.array([u, v])  # shape (2,)

def generate_2d_wind_vectors(
    predominate_wind_direction_deg_from_north: float,
    predominant_wind_speed: float,
    terrain_dem: np.ndarray,
    sampling_resolution_m: float = 1.0,
    downslope_weight_scale: float = 0.6,
    max_downslope_blend: float = 0.7,
    eps: float = 1e-9,
):
    """
    Produce a (H, W, 2) array of terrain-aware wind vectors (u, v) in m/s.

    Heuristic:
      final_dir = normalize( (1 - w) * base_dir + w * downslope_dir )
      where w = clip(downslope_weight_scale * slope_mag, 0, max_downslope_blend)
      slope_mag = sqrt((dz/dx)^2 + (dz/dy)^2)

    Speed is kept constant (predominant_wind_speed).
    """
    dem = np.asarray(terrain_dem, dtype=float)
    H, W = dem.shape

    # Base (uniform) wind direction, unit vector
    base_uv = unit_vector_from_met_direction(predominate_wind_direction_deg_from_north)  # (2,)

    # Terrain gradients
    dzdx, dzdy = calculate_slope_components(dem, sampling_resolution_m)  # (H,W) each

    # Downslope direction at each cell: negative gradient (toward decreasing elevation)
    down_u = -dzdx
    down_v = -dzdy
    down_mag = np.sqrt(down_u**2 + down_v**2) + eps
    down_u /= down_mag
    down_v /= down_mag

    # Blend weight based on slope magnitude (m/m)
    slope_mag = np.sqrt(dzdx**2 + dzdy**2)
    w = np.clip(downslope_weight_scale * slope_mag, 0.0, max_downslope_blend)

    # Blend base direction with downslope direction, then normalize and scale to speed
    U = (1 - w) * base_uv[0] + w * down_u
    V = (1 - w) * base_uv[1] + w * down_v
    mag = np.sqrt(U**2 + V**2) + eps
    U = predominant_wind_speed * U / mag
    V = predominant_wind_speed * V / mag

    # Stack to (H, W, 2) -> (u, v)
    wind_vectors = np.stack([U, V], axis=-1)
    return wind_vectors



if __name__ == "__main__":
    import rasterio
    from pathlib import Path
    
    # Configuration
    target_epsg = 4326  # WGS84 - adjust as needed
    temp_files = []
    
    try:
        # Path to DEM
        base_dem_path = "/Users/evantwarog/Documents/Coursework/AA228V/smoke_sim_py/data/BurnData/DEMs_and_Buffered_Burns/DEM_HenryCoe.tif"
        
        # Handle coordinate transformation if needed
        if target_epsg is not None:
            # Create a temporary file for the transformed DEM
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
                temp_dem_path = temp_file.name
                temp_files.append(temp_dem_path)
            
            # Transform the DEM to target CRS
            print(f"Transforming DEM to EPSG:{target_epsg}...")
            transform_raster(str(base_dem_path), temp_dem_path, f'EPSG:{target_epsg}')
            dem_path = temp_dem_path
        else:
            dem_path = base_dem_path
            
        # Open the (possibly transformed) DEM
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype('float32')
            transform = src.transform
            crs = src.crs
            print(f"DEM loaded. CRS: {crs}, Shape: {dem.shape}")
            
    except Exception as e:
        # Clean up any temporary files if an error occurs
        if temp_files:
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as cleanup_error:
                    print(f'Error cleaning up temporary file {temp_file}: {cleanup_error}')
        raise e
    
    # Convert no data values to NaN
    dem[dem <= 0] = np.nan
    
    # Create coordinate grids
    height, width = dem.shape
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Rest of your visualization code...
    try:
        # Generate wind vectors (wind from West at 5 m/s)
        wind = generate_2d_wind_vectors(
            predominate_wind_direction_deg_from_north=240.0,  # Wind from West
            predominant_wind_speed=5.0,  # 5 m/s
            terrain_dem=dem,
            sampling_resolution_m=1.0  # Assuming 10m resolution
        )
        
        # Subsample for better visualization
        stride = 4
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
        plt.quiver(X_sub, Y_sub, wind_u_sub, -wind_v_sub,  # Negative v for correct y-direction
                  color='red', scale=100, width=0.002, headwidth=3)
        
        plt.title("Terrain-aware Wind Vectors (Wind from West at 5 m/s)")
        plt.xlabel("Easting (pixels)")
        plt.ylabel("Northing (pixels)")
        plt.tight_layout()
        plt.show()
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {temp_file}: {e}")