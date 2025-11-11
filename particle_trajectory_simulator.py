"""
Particle Trajectory Simulator in 2D Terrain Vector Field
Simulates particle movement through a terrain-based vector field derived from a DEM.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
import sys
import rasterio.transform

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from simulation.core.terrain_vectors import TerrainVectorField
from simulation.core.wind_trajectories import WindTrajectoryField
from simulation.utils.trajectory_utils import calculate_trajectory
from simulation.utils.viz_utils import hsl_to_rgb
from simulation.core.smoke_points import sample_points_in_burn_area
from simulation.utils.geo_utils import load_geojson_polygons


class ParticleTrajectorySimulator:
    """Simulates and visualizes particle trajectories in a terrain-based vector field."""
    
    # Configuration constants
    MAX_TRAJECTORY_STEPS = 500
    
    def __init__(self, dem_path: str = None, burn_area_geojson: str = None):
        """
        Initialize the simulator with a terrain vector field and optional burn area.
        
        Args:
            dem_path: Path to DEM GeoTIFF file. If None, uses default Henry Coe DEM.
            burn_area_geojson: Path to GeoJSON file with burn area polygon. If None, uses default.
        """
        # Set default DEM path
        if dem_path is None:
            dem_path = "/Users/evantwarog/Documents/Coursework/AA228V/smesh-sensor-placement/data/BurnData/DEMs_and_Buffered_Burns/DEM_HenryCoe.tif"
        
        # Set default burn area path
        if burn_area_geojson is None:
            burn_area_geojson = "/Users/evantwarog/Documents/Coursework/AA228V/smesh-sensor-placement/data/BurnData/HenryCoe/BurnAreas/BurnArea2.json"
        
        # Load terrain vector field
        self.terrain_vector_field = TerrainVectorField(dem_path)
        self.WORLD_WIDTH = self.terrain_vector_field.width
        self.WORLD_HEIGHT = self.terrain_vector_field.height
        
        # Store burn area path
        self.burn_area_geojson = burn_area_geojson
        
        self.num_particles = 50
        self.trajectories = []
        
        # Wind trajectory field (optional)
        self.wind_trajectory_field = None
        self.use_wind_trajectories = False
        
        self._setup_ui()
        self._initialize_simulation()
    
    def _setup_ui(self):
        """Set up the matplotlib figure and interactive controls."""
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor('#111827')  # bg-gray-900
        
        # Main canvas
        self.ax = plt.subplot(111)
        self.ax.set_facecolor('#1f2937')  # bg-gray-800
        self.ax.set_xlim(0, self.WORLD_WIDTH)
        self.ax.set_ylim(0, self.WORLD_HEIGHT)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # Match web version coordinate system
        
        # Title
        self.fig.suptitle('Particle Trajectory Simulator', 
                         fontsize=16, fontweight='bold', color='#22d3ee', y=0.98)
        
        # Adjust layout to make room for sliders
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.93)
        
        # Create slider axes
        ax_num_particles = plt.axes([0.15, 0.08, 0.3, 0.03], facecolor='#374151')
        ax_run_button = plt.axes([0.15, 0.02, 0.3, 0.04], facecolor='#06b6d4')
        
        # Create sliders
        self.slider_num_particles = Slider(
            ax_num_particles, 'Particles', 1, 500, valinit=50, valstep=1,
            color='#06b6d4', track_color='#4b5563'
        )
        
        # Style slider labels
        self.slider_num_particles.label.set_color('#d1d5db')
        self.slider_num_particles.valtext.set_color('#22d3ee')
        
        # Run button
        self.btn_run = Button(ax_run_button, 'Reseed & Run', 
                             color='#06b6d4', hovercolor='#0891b2')
        self.btn_run.label.set_color('white')
        self.btn_run.label.set_fontweight('bold')
        
        # Connect event handlers
        self.slider_num_particles.on_changed(self._on_slider_change)
        self.btn_run.on_clicked(self._on_run_clicked)
    
    def _on_slider_change(self, val):
        """Handle slider changes."""
        self.num_particles = int(self.slider_num_particles.val)
        self.run_simulation()
    
    def _on_run_clicked(self, event):
        """Handle run button click."""
        self.run_simulation()
    
    def _initialize_simulation(self):
        """Initialize and run the first simulation."""
        self.run_simulation()
    
    def calculate_trajectory(self, start_x, start_y):
        """
        Calculate a single particle's trajectory from a starting point using the terrain vector field.
        
        Args:
            start_x: Starting X coordinate in pixel space
            start_y: Starting Y coordinate in pixel space
            
        Returns:
            List of (x, y) tuples representing the particle path
        """
        return calculate_trajectory(start_x, start_y, self.terrain_vector_field, 
                                   self.MAX_TRAJECTORY_STEPS, step_size=0.5)
    
    def _sample_points_from_burn_area(self, num_points: int) -> np.ndarray:
        """
        Sample random points from within the burn area polygon.
        
        Args:
            num_points: Number of points to sample
            
        Returns:
            Array of shape (num_points, 2) with pixel coordinates [col, row]
        """
        if not self.burn_area_geojson:
            # Fallback: sample from entire domain
            points = np.random.rand(num_points, 2)
            points[:, 0] *= self.WORLD_WIDTH
            points[:, 1] *= self.WORLD_HEIGHT
            return points
        
        # Use the smoke_points function to sample from burn area
        sampled_pts = sample_points_in_burn_area(
            self.burn_area_geojson,
            num_points,
            self.terrain_vector_field.transform,
            self.WORLD_HEIGHT,
            self.WORLD_WIDTH
        )
        return sampled_pts
    
    def initialize_wind_trajectories(self, num_trajectories: int = None, 
                                     initial_speed: float = 1.0, 
                                     initial_direction_deg: float = 270):
        """
        Initialize wind trajectory field.
        
        Args:
            num_trajectories: Number of trajectories to sample (defaults to num_particles)
            initial_speed: Initial speed for trajectories
            initial_direction_deg: Initial direction in degrees
        """
        if num_trajectories is None:
            num_trajectories = self.num_particles
        
        self.wind_trajectory_field = WindTrajectoryField(
            self.terrain_vector_field,
            self.burn_area_geojson,
            num_trajectories=num_trajectories,
            initial_speed=initial_speed,
            initial_direction_deg=initial_direction_deg
        )
        self.use_wind_trajectories = True
    
    def rollout_wind_trajectories(self, num_steps: int = 100, step_size: float = 0.5):
        """
        Rollout wind trajectories for specified number of steps.
        
        Args:
            num_steps: Number of time steps to simulate
            step_size: Distance to move per step in pixels
        """
        if self.wind_trajectory_field is None:
            print("Wind trajectory field not initialized. Call initialize_wind_trajectories() first.")
            return
        
        self.wind_trajectory_field.rollout(num_steps, step_size)
    
    def get_wind_trajectory_stats(self) -> dict:
        """Get statistics about wind trajectories."""
        if self.wind_trajectory_field is None:
            return {}
        return self.wind_trajectory_field.get_statistics()
    
    def run_simulation(self):
        """Run the full simulation: seed particles from burn area and calculate trajectories."""
        self.trajectories = []
        
        # Sample all particle starting points from burn area
        sampled_points = self._sample_points_from_burn_area(self.num_particles)
        
        for col, row in sampled_points:
            self.trajectories.append(self.calculate_trajectory(col, row))
        
        self._redraw()
    
    def _redraw(self):
        """Redraw the entire visualization."""
        self.ax.clear()
        self.ax.set_xlim(0, self.WORLD_WIDTH)
        self.ax.set_ylim(0, self.WORLD_HEIGHT)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.ax.set_facecolor('#1f2937')
        
        # Draw terrain vector field
        self._draw_terrain_vectors()
        
        # Draw wind trajectories if enabled
        self._draw_wind_trajectories()
        
        # Draw particle trajectories
        self._draw_trajectories()
        
        # Draw seed points
        self._draw_seed_points()
        
        self.fig.canvas.draw_idle()
    
    def _draw_terrain_vectors(self):
        """Draw DEM, terrain vector field, and burn area boundary."""
        # Draw DEM as background
        dem = self.terrain_vector_field.dem
        im = self.ax.imshow(dem, cmap='terrain', origin='upper', alpha=0.7,
                           extent=[0, self.WORLD_WIDTH, self.WORLD_HEIGHT, 0])
        
        # Draw burn area boundary
        self._draw_burn_area_boundary()
        
        # Draw terrain vector field with sparse sampling
        stride = max(1, self.WORLD_WIDTH // 15)  # Show ~15 vectors across width
        arrow_length = 1.5  # pixels (smaller)
        
        for col in range(0, self.WORLD_WIDTH, stride):
            for row in range(0, self.WORLD_HEIGHT, stride):
                u, v = self.terrain_vector_field.get_vector_at(col, row)
                
                magnitude = np.sqrt(u**2 + v**2)
                if magnitude < 1e-6:
                    continue
                
                # Negate v to account for inverted y-axis
                dx = u * arrow_length
                dy = -v * arrow_length
                
                self.ax.arrow(col, row, dx, dy,
                            head_width=0.8, head_length=0.6,
                            fc='#22d3ee', ec='#22d3ee', alpha=0.8, linewidth=0.5)
    
    def _draw_wind_trajectories(self):
        """Draw wind trajectories if they exist."""
        if self.wind_trajectory_field is None or not self.use_wind_trajectories:
            return
        
        paths = self.wind_trajectory_field.get_all_paths()
        
        for index, path in enumerate(paths):
            if len(path) < 2:
                continue
            
            # Calculate color using HSL
            hue = (index * (360 / max(len(paths), 1))) % 360
            rgb = hsl_to_rgb(hue, 80, 60)
            
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            
            # Draw trajectory line
            self.ax.plot(xs, ys, color=rgb, alpha=0.7, linewidth=1.5,
                        solid_capstyle='round', solid_joinstyle='round')
            
            # Draw starting point
            self.ax.plot(xs[0], ys[0], 'o', color=rgb, markersize=6, alpha=0.9)
            
            # Draw ending point
            self.ax.plot(xs[-1], ys[-1], 's', color=rgb, markersize=5, alpha=0.6)
    
    def _draw_burn_area_boundary(self):
        """Draw the burn area polygon boundary."""
        if not self.burn_area_geojson:
            return
        
        burn_area_polygons = load_geojson_polygons(self.burn_area_geojson)
        
        for polygon_coords in burn_area_polygons:
            # Convert lat/lon coordinates to pixel coordinates
            pixel_coords = []
            for lon, lat in polygon_coords:
                row, col = rasterio.transform.rowcol(self.terrain_vector_field.transform, lon, lat)
                if 0 <= row < self.WORLD_HEIGHT and 0 <= col < self.WORLD_WIDTH:
                    pixel_coords.append([col, row])
            
            if pixel_coords:
                pixel_coords = np.array(pixel_coords)
                # Plot the polygon boundary
                self.ax.plot(pixel_coords[:, 0], pixel_coords[:, 1],
                           color='red', linewidth=2, label='Burn Area', alpha=0.8)
    
    def _draw_trajectories(self):
        """Draw all calculated particle trajectories."""
        for index, path in enumerate(self.trajectories):
            if len(path) < 2:
                continue
            
            # Calculate color using HSL
            hue = (index * (360 / max(self.num_particles, 1))) % 360
            # Convert HSL to RGB
            rgb = hsl_to_rgb(hue, 80, 60)
            
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            
            self.ax.plot(xs, ys, color=rgb, alpha=0.85, linewidth=1.5,
                        solid_capstyle='round', solid_joinstyle='round')
    
    def _draw_seed_points(self):
        """Draw the starting seed point for each trajectory."""
        for path in self.trajectories:
            if len(path) > 0:
                seed_x, seed_y = path[0]
                self.ax.plot(seed_x, seed_y, 'o', color='#d1d5db', 
                           markersize=4, alpha=0.8)
    
    def show(self):
        """Display the interactive visualization."""
        plt.show()


if __name__ == '__main__':
    simulator = ParticleTrajectorySimulator()
    simulator.show()
