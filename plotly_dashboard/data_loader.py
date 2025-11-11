"""Data loading utilities for the Smoke Simulation Dashboard."""

import os
import json
from pathlib import Path
import geopandas as gpd
from typing import List, Optional, Union, Dict, Any
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, shape

# Configure GeoPandas to use the 'pyogrio' engine for better performance and reliability
gpd.options.io_engine = 'pyogrio'

def convert_geometry(geom_data: Dict[str, Any]) -> Optional[Any]:
    """Convert geometry data to a Shapely geometry object."""
    try:
        if 'coordinates' in geom_data:
            # Handle different geometry types
            geom_type = geom_data.get('type', '').lower()
            
            if geom_type == 'point':
                coords = geom_data['coordinates']
                # Take only x,y coordinates (first 2 elements)
                return Point(coords[:2])
                
            elif geom_type == 'polygon':
                # Get the exterior ring (first element) and take x,y coordinates
                coords = np.array(geom_data['coordinates'][0])
                if coords.shape[1] > 2:  # If 3D or 4D coordinates
                    coords = coords[:, :2]  # Take only x,y
                return Polygon(coords)
                
            elif geom_type == 'multipolygon':
                polygons = []
                for polygon in geom_data['coordinates']:
                    for ring in polygon:  # First ring is exterior, rest are holes
                        coords = np.array(ring)
                        if coords.shape[1] > 2:
                            coords = coords[:, :2]
                        polygons.append(Polygon(coords))
                return MultiPolygon(polygons)
                
    except Exception as e:
        print(f"Error converting geometry: {e}")
    return None

def load_geojson(file_path: Path) -> Optional[gpd.GeoDataFrame]:
    """Load a GeoJSON file and return a GeoDataFrame with proper CRS handling."""
    try:
        # Read the raw JSON first to handle custom structures
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Handle different GeoJSON structures
        if 'features' in data:  # FeatureCollection
            features = data['features']
            geometries = []
            properties = []
            
            for feature in features:
                if 'geometry' in feature:
                    geom = convert_geometry(feature['geometry'])
                    if geom is not None:
                        geometries.append(geom)
                        properties.append(feature.get('properties', {}))
            
            if not geometries:
                print(f"No valid geometries found in {file_path}")
                return None
                
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
            
        else:  # Single geometry or custom format
            # Try to parse as a single geometry
            geom = convert_geometry(data)
            if geom is not None:
                gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
            else:
                # Try to read with GeoPandas directly as fallback
                gdf = gpd.read_file(file_path)
                
        # Ensure we have a valid CRS
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
            
        # Convert to WGS84 if needed
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
            
        return gdf
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_snode_data(file_path: str, sim_num: int) -> gpd.GeoDataFrame:
    """Load SNode data from a GeoJSON file."""
    gdf = load_geojson(file_path)
    if gdf is not None:
        gdf['simulation'] = f'Sim {sim_num}'
        gdf['label'] = [f'Sim {sim_num} Node {i+1}' for i in range(len(gdf))]
    return gdf

def load_burn_area(file_path: str, area_num: int) -> gpd.GeoDataFrame:
    """Load burn area data from a GeoJSON file."""
    gdf = load_geojson(file_path)
    if gdf is not None:
        gdf['area'] = f'Burn Area {area_num}'
    return gdf

def find_geojson_files(directory: str, pattern: str = '*.json') -> List[str]:
    """Find all GeoJSON files in a directory matching the given pattern."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    return sorted([str(p) for p in Path(directory).rglob(pattern)])

def load_all_snodes(snode_dir: str) -> List[gpd.GeoDataFrame]:
    """Load all SNode data from a directory."""
    snode_files = find_geojson_files(snode_dir, 'Sim*.json')
    return [
        load_snode_data(f, i+1) 
        for i, f in enumerate(snode_files)
    ]

def load_all_burn_areas(burn_area_dir: str) -> List[gpd.GeoDataFrame]:
    """Load all burn area data from a directory."""
    burn_area_files = find_geojson_files(burn_area_dir, 'BurnArea*.json')
    return [
        load_burn_area(f, i+1)
        for i, f in enumerate(burn_area_files)
    ]
