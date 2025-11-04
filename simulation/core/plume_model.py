"""
Gaussian plume model for smoke dispersion simulation.

This module implements a Gaussian plume model based on atmospheric stability classes
for simulating smoke dispersion from point sources.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union
import numpy as np
from numpy.typing import NDArray

# Vertical column parameters for atmospheric stability classes
# See: https://en.wikipedia.org/wiki/Atmospheric_dispersion_modeling
_VERTICAL_COLUMN_VALUES_TABLE = np.array([
    [-1.104,  0.9878, -0.0076,  4.679, -1.7172,  0.2770],
    [-1.634,  1.0350, -0.0096, -1.999,  0.8752,  0.0136],
    [-2.054,  1.0231, -0.0076, -2.341,  0.9477, -0.0020],
    [-2.555,  1.0423, -0.0087, -3.186,  1.1737, -0.0316],
    [-2.754,  1.0106, -0.0064, -3.783,  1.3010, -0.0450],
    [-3.143,  1.0148, -0.0070, -4.490,  1.4024, -0.0540]
])


@dataclass
class AtmoColumnParams:
    """Parameters for atmospheric column stability classes.
    
    The parameters (i,j,k) for y and z directions where:
    - "y" is the cross-wind direction
    - "z" is the vertical direction
    """
    iy: float
    jy: float
    ky: float
    iz: float
    jz: float
    kz: float


# Dictionary of stability classes (A-F) with their parameters
AIR_COLUMN_STABILITY_CLASSES: Dict[str, AtmoColumnParams] = {
    cls: AtmoColumnParams(*params) 
    for cls, params in zip(
        ["A", "B", "C", "D", "E", "F"],
        _VERTICAL_COLUMN_VALUES_TABLE
    )
}


def sigma_change(x_local: float, iyz: float, jyz: float, kyz: float) -> float:
    """Calculate the change in standard deviation of the Gaussian plume.
    
    Args:
        x_local: Downwind distance from source
        iyz, jyz, kyz: Coefficients from the atmospheric stability class
        
    Returns:
        Standard deviation of the plume at distance x_local
    """
    return np.exp(iyz + jyz * np.log(x_local) + kyz * np.log(x_local) ** 2)


def cross_wind_dispersion(x_local: float, y_local: float, 
                         sigma_y: Callable[[float], float]) -> float:
    """Calculate cross-wind dispersion of the Gaussian plume.
    
    Args:
        x_local: Downwind distance from source
        y_local: Cross-wind distance from centerline
        sigma_y: Function to compute cross-wind standard deviation
        
    Returns:
        Cross-wind dispersion factor
    """
    sigma_y_curr = sigma_y(x_local)
    return np.exp(-y_local ** 2 / (2 * sigma_y_curr ** 2)) / (sigma_y_curr * np.sqrt(2 * np.pi))


def along_wind_dispersion(x_local: float, z_local: float, 
                         sigma_z: Callable[[float], float],
                         h_center: float) -> float:
    """Calculate along-wind (vertical) dispersion of the Gaussian plume.
    
    Args:
        x_local: Downwind distance from source
        z_local: Vertical distance from ground
        sigma_z: Function to compute vertical standard deviation
        h_center: Plume centerline height
        
    Returns:
        Vertical dispersion factor
    """
    sigma_z_curr = sigma_z(x_local)
    return np.exp(-(z_local - h_center) ** 2 / (2 * sigma_z_curr ** 2)) / (sigma_z_curr * np.sqrt(2 * np.pi))


def plume_model_point_estimate(x_local: float, y_local: float, z_local: float,
                             Q: float, u_wind_local: float, h_local: float,
                             sigma_y: Callable[[float], float],
                             sigma_z: Callable[[float], float]) -> float:
    """Calculate the point estimate of the Gaussian plume model.
    
    Args:
        x_local: Downwind distance from source (must be >= 0)
        y_local: Cross-wind distance from centerline
        z_local: Vertical distance from ground
        Q: Emission rate (mass/time)
        u_wind_local: Wind speed at source height
        h_local: Plume centerline height
        sigma_y: Function for cross-wind standard deviation
        sigma_z: Function for vertical standard deviation
        
    Returns:
        Concentration at point (x_local, y_local, z_local)
    """
    if x_local < 0:
        return 0.0
        
    cross_wind = cross_wind_dispersion(x_local, y_local, sigma_y)
    along_wind = along_wind_dispersion(x_local, z_local, sigma_z, h_local)
    
    return (Q / u_wind_local) * cross_wind * along_wind


class PlumeFromPointSource:
    """Gaussian plume model for a point source.
    
    This class models the dispersion of a plume from a point source using
    a Gaussian plume model with atmospheric stability classes.
    """
    
    def __init__(self, Q: float, h: float, stability_class: str,
                 source_xyz: Union[List[float], NDArray[np.float64]]):
        """Initialize the plume model.
        
        Args:
            Q: Emission rate (mass/time)
            h: Plume centerline height
            stability_class: Atmospheric stability class (A-F)
            source_xyz: [x, y, z] coordinates of the source
        """
        self.Q = Q
        self.h = h
        self.source_xyz = np.asarray(source_xyz, dtype=np.float64)
        
        if stability_class not in AIR_COLUMN_STABILITY_CLASSES:
            raise ValueError(f"Invalid stability class: {stability_class}")
            
        atmo_params = AIR_COLUMN_STABILITY_CLASSES[stability_class]
        
        # Create sigma functions with the appropriate parameters
        self.sigma_y = lambda x: sigma_change(x, atmo_params.iy, atmo_params.jy, atmo_params.ky)
        self.sigma_z = lambda x: sigma_change(x, atmo_params.iz, atmo_params.jz, atmo_params.kz)
    
    @staticmethod
    def global_to_local_coords(xyz_local: Union[List[float], NDArray[np.float64]],
                              xyz_global: Union[List[float], NDArray[np.float64]],
                              wind_global: Union[List[float], NDArray[np.float64]]) -> NDArray[np.float64]:
        """Convert global coordinates to local plume coordinates.
        
        Args:
            xyz_local: Reference point (source) in global coordinates
            xyz_global: Point to convert to local coordinates
            wind_global: Wind vector in global coordinates [u, v, w]
            
        Returns:
            Point in local coordinates [x_downwind, y_crosswind, z_vertical]
        """
        xyz_centered = np.asarray(xyz_global) - np.asarray(xyz_local)
        wind_global = np.asarray(wind_global, dtype=np.float64)
        
        # Local x-axis aligned with wind direction
        wind_speed = np.linalg.norm(wind_global)
        if wind_speed < 1e-10:  # Avoid division by zero
            return np.zeros(3)
            
        wind_x_local = wind_global / wind_speed
        
        # Local z-axis is global z-axis
        z_local = np.array([0, 0, 1])
        
        # Local y-axis is cross product of z and wind direction
        y_local = np.cross(z_local, wind_x_local)
        y_local = y_local / np.linalg.norm(y_local)  # Normalize
        
        # Project global coordinates to local coordinate system
        x_local = np.dot(xyz_centered, wind_x_local)
        y_local = np.dot(xyz_centered, y_local)
        z_local = np.dot(xyz_centered, z_local)
        
        return np.array([x_local, y_local, z_local])
    
    def query_plume_model(self, xyz_global: Union[List[float], NDArray[np.float64]],
                         wind_global: Union[List[float], NDArray[np.float64]]) -> float:
        """Query the plume model at a point in global coordinates.
        
        Args:
            xyz_global: [x, y, z] coordinates of query point
            wind_global: [u, v, w] wind vector at query point
            
        Returns:
            Concentration at the query point
        """
        xyz_local = self.global_to_local_coords(
            self.source_xyz, xyz_global, wind_global
        )
        
        return plume_model_point_estimate(
            x_local=xyz_local[0],
            y_local=xyz_local[1],
            z_local=xyz_local[2],
            Q=self.Q,
            u_wind_local=np.linalg.norm(wind_global),
            h_local=self.h,
            sigma_y=self.sigma_y,
            sigma_z=self.sigma_z
        )


def query_multiple_plumes_points(
    plumes: List[PlumeFromPointSource],
    xyz_global_points: List[Union[List[float], NDArray[np.float64]]],
    wind_global: Union[List[float], NDArray[np.float64]]
) -> NDArray[np.float64]:
    """Query multiple plume models at multiple points.
    
    Args:
        plumes: List of PlumeFromPointSource objects
        xyz_global_points: List of [x, y, z] coordinates to query
        wind_global: [u, v, w] wind vector
        
    Returns:
        Array of concentrations at each query point
    """
    density = np.zeros(len(xyz_global_points))
    
    for i, point in enumerate(xyz_global_points):
        for plume in plumes:
            density[i] += plume.query_plume_model(point, wind_global)
    
    return density
