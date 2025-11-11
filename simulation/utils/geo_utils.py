"""
Geospatial utilities for coordinate transformations, projections, and polygon sampling.
"""

import rasterio
import rasterio.transform
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import numpy as np
from typing import Union, Tuple, Optional, List
import json

def transform_raster(
    input_path: str,
    output_path: str,
    dst_crs: Union[str, int],
    resolution: Optional[Tuple[float, float]] = None,
    resampling: Resampling = Resampling.nearest
) -> None:
    """Reproject a raster file to a different CRS.
    
    Args:
        input_path: Path to input raster file
        output_path: Path to save the reprojected raster
        dst_crs: Target CRS (can be EPSG code as int or string, or WKT/Proj4 string)
        resolution: Optional (x, y) resolution in target CRS units. If None, calculates automatically.
        resampling: Resampling method to use. Defaults to nearest neighbor.
    """
    with rasterio.open(input_path) as src:
        # Calculate the transform and dimensions for the output
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=resolution
        )
        
        # Set the metadata for the output
        meta = src.meta.copy()
        meta.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Create the output file
        with rasterio.open(output_path, 'w', **meta) as dst:
            # Reproject each band
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling
                )

def transform_geodataframe(
    gdf: gpd.GeoDataFrame,
    dst_crs: Union[str, int]
) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to a different CRS.
    
    Args:
        gdf: Input GeoDataFrame
        dst_crs: Target CRS (can be EPSG code as int or string, or WKT/Proj4 string)
        
    Returns:
        Reprojected GeoDataFrame
    """
    return gdf.to_crs(dst_crs)

def get_crs_epsg(crs) -> int:
    """Extract EPSG code from a CRS object.
    
    Args:
        crs: CRS object, string, or int
        
    Returns:
        int: EPSG code, or None if not available
    """
    if crs is None:
        return None
    if isinstance(crs, int):
        return crs
    if isinstance(crs, str) and crs.lower().startswith('epsg:'):
        return int(crs.split(':')[1])
    try:
        from pyproj import CRS
        crs_obj = CRS(crs)
        if crs_obj.to_epsg() is not None:
            return crs_obj.to_epsg()
    except ImportError:
        pass
    return None


def load_geojson_polygons(geojson_path: str) -> List[List[Tuple[float, float]]]:
    """
    Load polygon coordinates from a GeoJSON file.
    
    Args:
        geojson_path: Path to GeoJSON file
        
    Returns:
        List of polygon coordinate lists, where each polygon is a list of (lon, lat) tuples
    """
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    polygons = []
    for feature in geojson_data.get('features', []):
        if feature['geometry']['type'] == 'Polygon':
            # Extract exterior ring (first coordinate list)
            coords = feature['geometry']['coordinates'][0]
            polygons.append(coords)
    
    return polygons


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: (x, y) tuple
        polygon: List of (x, y) tuples representing polygon vertices
        
    Returns:
        Boolean indicating if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def sample_points_in_polygon(
    polygon_coords: List[Tuple[float, float]],
    num_samples: int,
    transform: rasterio.transform.Affine,
    height: int,
    width: int
) -> np.ndarray:
    """
    Randomly sample points within a polygon in pixel space.
    
    Args:
        polygon_coords: List of (lon, lat) tuples defining polygon in geographic coordinates
        num_samples: Number of points to sample
        transform: Rasterio transform for coordinate conversion from geographic to pixel space
        height: Raster height in pixels
        width: Raster width in pixels
        
    Returns:
        Array of shape (num_samples, 2) with pixel coordinates [col, row]
    """
    # Convert polygon to pixel coordinates
    pixel_polygon = []
    for lon, lat in polygon_coords:
        row, col = rasterio.transform.rowcol(transform, lon, lat)
        if 0 <= row < height and 0 <= col < width:
            pixel_polygon.append((col, row))
    
    if len(pixel_polygon) < 3:
        return np.array([])
    
    # Get bounding box of polygon
    pixel_polygon_array = np.array(pixel_polygon)
    min_col = int(np.floor(pixel_polygon_array[:, 0].min()))
    max_col = int(np.ceil(pixel_polygon_array[:, 0].max()))
    min_row = int(np.floor(pixel_polygon_array[:, 1].min()))
    max_row = int(np.ceil(pixel_polygon_array[:, 1].max()))
    
    # Clamp to image bounds
    min_col = max(0, min_col)
    max_col = min(width - 1, max_col)
    min_row = max(0, min_row)
    max_row = min(height - 1, max_row)
    
    # Randomly sample points within bounding box until we have enough inside polygon
    sampled_points = []
    max_attempts = num_samples * 100
    attempts = 0
    
    while len(sampled_points) < num_samples and attempts < max_attempts:
        col = np.random.uniform(min_col, max_col)
        row = np.random.uniform(min_row, max_row)
        
        if point_in_polygon((col, row), pixel_polygon):
            sampled_points.append([col, row])
        
        attempts += 1
    
    return np.array(sampled_points)
