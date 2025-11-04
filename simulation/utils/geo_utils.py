"""
Geospatial utilities for coordinate transformations and projections.
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import numpy as np
from typing import Union, Tuple, Optional
import os

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
