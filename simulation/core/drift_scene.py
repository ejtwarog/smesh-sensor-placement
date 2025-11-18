"""
Wind Drift Dashboard module for interactive visualization.

Provides the WindDriftDashboard class for creating interactive visualizations
of wind drift with terrain influence, burn area progression, and trajectory
sampling over time.
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
import rasterio.transform

from .terrain_vectors import TerrainVectorField
from .wind_trajectories import WindTrajectoryField
from simulation.utils.viz_utils import hsl_to_rgb
from simulation.utils.geo_utils import load_geojson_polygons


class WindDriftDashboard:
    """Interactive dashboard for wind drift visualization with burn area progression."""
    
    def __init__(self, dem_path: str = None, burn_area_dir: str = None):
        """Initialize the dashboard.
        
        Args:
            dem_path: Path to DEM GeoTIFF file
            burn_area_dir: Path to directory containing burn area GeoJSON files
        """
        if dem_path is None:
            dem_path = "/Users/evantwarog/Documents/Coursework/AA228V/smesh-sensor-placement/data/BurnData/DEMs_and_Buffered_Burns/DEM_HenryCoe.tif"
        if burn_area_dir is None:
            burn_area_dir = "/Users/evantwarog/Documents/Coursework/AA228V/smesh-sensor-placement/data/BurnData/HenryCoe/BurnAreas"
        
        self.terrain_vector_field = TerrainVectorField(dem_path)
        self.burn_area_dir = burn_area_dir
        self.WORLD_WIDTH = self.terrain_vector_field.width
        self.WORLD_HEIGHT = self.terrain_vector_field.height
        self.wind_trajectory_field = None
        self.transform = self.terrain_vector_field.transform
        
        # Load all burn area JSON files
        self.burn_area_files = sorted([
            os.path.join(burn_area_dir, f) for f in os.listdir(burn_area_dir)
            if f.endswith('.json')
        ])
        self.current_burn_area_index = 0
        self.max_burn_area_index_visited = 0
    
    def _get_dem_geographic_bounds(self):
        """Get DEM bounds in geographic coordinates (lon/lat)."""
        lon_min, lat_max = rasterio.transform.xy(self.transform, 0, 0)
        lon_max, lat_min = rasterio.transform.xy(self.transform, self.WORLD_HEIGHT - 1, self.WORLD_WIDTH - 1)
        return lon_min, lon_max, lat_min, lat_max
    
    def _create_terrain_heatmap(self):
        """Create terrain heatmap trace in geographic coordinates."""
        dem = self.terrain_vector_field.dem
        height, width = dem.shape
        lon_min, lon_max, lat_min, lat_max = self._get_dem_geographic_bounds()
        
        return go.Heatmap(
            z=dem,
            colorscale='Gray',
            showscale=False,
            hoverinfo='skip',
            name='Terrain',
            y=np.linspace(lat_max, lat_min, height),
            x=np.linspace(lon_min, lon_max, width)
        )
    
    def _create_terrain_vectors(self):
        """Create terrain vector field in geographic coordinates."""
        stride = max(1, self.WORLD_WIDTH // 15)
        arrow_scale = 0.0005
        
        xs, ys, dxs, dys = [], [], [], []
        
        for col in range(0, self.WORLD_WIDTH, stride):
            for row in range(0, self.WORLD_HEIGHT, stride):
                u, v = self.terrain_vector_field.get_vector_at(col, row)
                magnitude = np.sqrt(u**2 + v**2)
                
                if magnitude > 1e-6:
                    lon, lat = rasterio.transform.xy(self.transform, row, col)
                    xs.append(lon)
                    ys.append(lat)
                    dxs.append(u * arrow_scale)
                    dys.append(-v * arrow_scale)
        
        return go.Scatter(
            x=xs, y=ys,
            mode='markers',
            marker=dict(size=6, color='#888888', opacity=0.6),
            name='Downhill',
            hoverinfo='skip'
        ), dxs, dys, xs, ys
    
    def _create_wind_trajectories(self):
        """Create wind trajectory traces in geographic coordinates."""
        if self.wind_trajectory_field is None:
            return []
        
        paths = self.wind_trajectory_field.get_all_paths()
        traces = []
        
        for index, path in enumerate(paths):
            if len(path) < 2:
                continue
            
            hue = (index * (360 / max(len(paths), 1))) % 360
            rgb = hsl_to_rgb(hue, 80, 60)
            color = f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
            
            xs = [rasterio.transform.xy(self.transform, row, col)[0] for col, row in path]
            ys = [rasterio.transform.xy(self.transform, row, col)[1] for col, row in path]
            
            traces.append(go.Scatter(
                x=xs, y=ys,
                mode='lines',
                line=dict(color=color, width=2),
                name=f'Traj {index+1}',
                hoverinfo='skip',
                showlegend=False
            ))
            
            traces.append(go.Scatter(
                x=[xs[0]], y=[ys[0]],
                mode='markers',
                marker=dict(size=8, color=color),
                hoverinfo='skip',
                showlegend=False
            ))
            
            if len(path) >= 2:
                x_end, y_end = xs[-1], ys[-1]
                x_prev, y_prev = xs[-2], ys[-2]
                arrow_end_x = x_end + (x_end - x_prev) * 1.2
                arrow_end_y = y_end + (y_end - y_prev) * 1.2
                
                traces.append(go.Scatter(
                    x=[x_end, arrow_end_x],
                    y=[y_end, arrow_end_y],
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        return traces
    
    def _create_burn_area_boundary(self, burn_area_file: str, is_active: bool = False):
        """Create burn area boundary traces from a GeoJSON file."""
        burn_area_polygons = load_geojson_polygons(burn_area_file)
        traces = []
        min_lon = float("inf")
        max_lon = float("-inf")
        min_lat = float("inf")
        max_lat = float("-inf")
        
        color = 'red' if is_active else '#cccccc'
        line_width = 2 if is_active else 1
        opacity = 1.0 if is_active else 0.5
        
        for polygon_coords in burn_area_polygons:
            geo_coords = []
            for lon, lat in polygon_coords:
                row, col = rasterio.transform.rowcol(self.transform, lon, lat)
                if 0 <= row < self.WORLD_HEIGHT and 0 <= col < self.WORLD_WIDTH:
                    geo_coords.append((lon, lat))
            
            if geo_coords:
                xs = [p[0] for p in geo_coords] + [geo_coords[0][0]]
                ys = [p[1] for p in geo_coords] + [geo_coords[0][1]]

                min_lon = min(min_lon, min(xs))
                max_lon = max(max_lon, max(xs))
                min_lat = min(min_lat, min(ys))
                max_lat = max(max_lat, max(ys))
                
                traces.append(go.Scatter(
                    x=xs, y=ys,
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    name='Burn Area' if is_active else 'Other Areas',
                    hoverinfo='skip',
                    showlegend=is_active,
                    opacity=opacity
                ))
        
        if min_lon == float("inf"):
            min_lon, max_lon, min_lat, max_lat = self._get_dem_geographic_bounds()
        
        return traces, (min_lon, max_lon, min_lat, max_lat)
    
    def _create_smoldering_area_boundary(self, burn_area_file: str):
        """Create smoldering (previous) burn area boundary in black."""
        burn_area_polygons = load_geojson_polygons(burn_area_file)
        traces = []
        min_lon = float("inf")
        max_lon = float("-inf")
        min_lat = float("inf")
        max_lat = float("-inf")
        
        for polygon_coords in burn_area_polygons:
            geo_coords = []
            for lon, lat in polygon_coords:
                row, col = rasterio.transform.rowcol(self.transform, lon, lat)
                if 0 <= row < self.WORLD_HEIGHT and 0 <= col < self.WORLD_WIDTH:
                    geo_coords.append((lon, lat))
            
            if geo_coords:
                xs = [p[0] for p in geo_coords] + [geo_coords[0][0]]
                ys = [p[1] for p in geo_coords] + [geo_coords[0][1]]

                min_lon = min(min_lon, min(xs))
                max_lon = max(max_lon, max(xs))
                min_lat = min(min_lat, min(ys))
                max_lat = max(max_lat, max(ys))
                
                traces.append(go.Scatter(
                    x=xs, y=ys,
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Smoldering',
                    hoverinfo='skip',
                    showlegend=False,
                    opacity=0.8
                ))
        
        if min_lon == float("inf"):
            min_lon, max_lon, min_lat, max_lat = self._get_dem_geographic_bounds()
        
        return traces, (min_lon, max_lon, min_lat, max_lat)
    
    def _create_smoldering_trajectory_field(self, num_trajectories, wind_direction, terrain_influence):
        """Create trajectory field for smoldering phase (sample from all areas at reduced rate)."""
        from .smoke_points import sample_points_in_burn_area
        from .wind_trajectories import WindTrajectory
        
        temp_field = WindTrajectoryField(
            self.terrain_vector_field,
            self.burn_area_files[0],
            num_trajectories=1,
            initial_speed=5.0,
            initial_direction_deg=wind_direction,
            terrain_influence_factor=terrain_influence
        )
        temp_field.trajectories = []
        
        for burn_file in self.burn_area_files:
            sampled_points = sample_points_in_burn_area(
                burn_file,
                num_trajectories,
                self.terrain_vector_field.transform,
                self.terrain_vector_field.height,
                self.terrain_vector_field.width
            )
            
            for col, row in sampled_points:
                trajectory = WindTrajectory(
                    col, row,
                    5.0,
                    wind_direction,
                    terrain_influence
                )
                temp_field.trajectories.append(trajectory)
        
        temp_field.active_trajectories = temp_field.trajectories.copy()
        return temp_field
    
    def _create_cumulative_trajectory_field(self, num_trajectories, previous_area_count, 
                                           burn_area_index, wind_direction, terrain_influence):
        """Create trajectory field with cumulative sampling from current and previous areas."""
        from .smoke_points import sample_points_in_burn_area
        from .wind_trajectories import WindTrajectory
        
        current_burn_area = self.burn_area_files[burn_area_index]
        temp_field = WindTrajectoryField(
            self.terrain_vector_field,
            current_burn_area,
            num_trajectories=num_trajectories,
            initial_speed=5.0,
            initial_direction_deg=wind_direction,
            terrain_influence_factor=terrain_influence
        )
        
        for prev_idx in range(burn_area_index):
            prev_burn_area = self.burn_area_files[prev_idx]
            sampled_points = sample_points_in_burn_area(
                prev_burn_area,
                previous_area_count,
                self.terrain_vector_field.transform,
                self.terrain_vector_field.height,
                self.terrain_vector_field.width
            )
            
            for col, row in sampled_points:
                trajectory = WindTrajectory(
                    col, row,
                    5.0,
                    wind_direction,
                    terrain_influence
                )
                temp_field.trajectories.append(trajectory)
        
        temp_field.active_trajectories = temp_field.trajectories.copy()
        return temp_field
    
    def create_figure(self, num_trajectories, wind_direction, terrain_influence, burn_area_index: int = 0, is_smoldering: bool = False):
        """Create figure with current parameters."""
        self.current_burn_area_index = burn_area_index
        
        if burn_area_index > self.max_burn_area_index_visited:
            self.max_burn_area_index_visited = burn_area_index
        
        if is_smoldering:
            smoldering_count = max(1, int(num_trajectories * 0.1))
            self.wind_trajectory_field = self._create_smoldering_trajectory_field(
                num_trajectories=smoldering_count,
                wind_direction=wind_direction,
                terrain_influence=terrain_influence
            )
        else:
            current_burn_area = self.burn_area_files[burn_area_index] if self.burn_area_files else None
            
            if current_burn_area:
                current_area_count = num_trajectories
                previous_area_count = max(1, int(num_trajectories * 0.1))
                
                self.wind_trajectory_field = self._create_cumulative_trajectory_field(
                    num_trajectories=current_area_count,
                    previous_area_count=previous_area_count,
                    burn_area_index=burn_area_index,
                    wind_direction=wind_direction,
                    terrain_influence=terrain_influence
                )
        
        if self.wind_trajectory_field:
            self.wind_trajectory_field.rollout(num_steps=8, step_size=0.25)
        
        fig = go.Figure()
        fig.add_trace(self._create_terrain_heatmap())
        
        terrain_scatter, dxs, dys, xs, ys = self._create_terrain_vectors()
        fig.add_trace(terrain_scatter)
        
        annotations = []
        for x, y, dx, dy in zip(xs, ys, dxs, dys):
            annotations.append(
                dict(
                    x=x + dx,
                    y=y + dy,
                    ax=x,
                    ay=y,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor='#888888',
                    opacity=0.6
                )
            )
        
        for trace in self._create_wind_trajectories():
            fig.add_trace(trace)
        
        all_bounds = []
        for idx, burn_file in enumerate(self.burn_area_files):
            if is_smoldering:
                burn_traces, bounds = self._create_smoldering_area_boundary(burn_file)
            elif idx == burn_area_index:
                burn_traces, bounds = self._create_burn_area_boundary(burn_file, is_active=True)
            elif idx < burn_area_index:
                burn_traces, bounds = self._create_smoldering_area_boundary(burn_file)
            else:
                burn_traces, bounds = self._create_burn_area_boundary(burn_file, is_active=False)
            
            for trace in burn_traces:
                fig.add_trace(trace)
            all_bounds.append(bounds)
        
        dem_min_lon, dem_max_lon, dem_min_lat, dem_max_lat = self._get_dem_geographic_bounds()
        min_lon = dem_min_lon
        max_lon = dem_max_lon
        min_lat = dem_min_lat
        max_lat = dem_max_lat
        
        for burn_min_lon, burn_max_lon, burn_min_lat, burn_max_lat in all_bounds:
            min_lon = min(min_lon, burn_min_lon)
            max_lon = max(max_lon, burn_max_lon)
            min_lat = min(min_lat, burn_min_lat)
            max_lat = max(max_lat, burn_max_lat)

        fig.update_layout(
            title=f'Wind Drift Visualization | Dir: {wind_direction}° | Influence: {terrain_influence:.2f}',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#544948'),
            height=800,
            width=1200,
            yaxis=dict(scaleanchor='x', scaleratio=1, range=[min_lat, max_lat]),
            xaxis=dict(scaleanchor='y', scaleratio=1, range=[min_lon, max_lon]),
            annotations=annotations
        )
        
        return fig
