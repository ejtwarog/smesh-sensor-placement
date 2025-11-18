# Simulation Aid: Sensor Placement for Prescribed Wildfire Monitoring

A Python-based tool for simulating smoke dispersion and optimizing sensor placement for prescribed wildfire monitoring. This project provides both a command-line interface and an interactive web dashboard for analyzing burn areas, wind data, and sensor node placement.

## Overview

This project is a Python port of the SMeshSmokeValidation.jl package, enhanced with:
- Interactive 2D map visualization with burn areas and sensor nodes
- Wind data analysis (forecasted and historic)
- 3D burn progression visualization
- Sensor placement optimization tools
- Real-time dashboard for monitoring and analysis

## Features

- **2D Map View**: Interactive map showing burn areas, simulation nodes, and sensor locations
- **Wind Data Analysis**: View forecasted and historic wind data with filtering and statistics
- **Wind Drift Visualization**: Interactive simulation of smoke particle trajectories with terrain influence
  - Cumulative sampling from multiple burn areas over time
  - Smoldering phase visualization (6-hour post-fire period)
  - Real-time parameter adjustment (wind direction, terrain influence, particle count)
- **3D Visualization**: 3D burn progression with DEM and sensor node overlays
- **Sensor Placement Optimization**: Tools for optimizing sensor node placement
- **Data Export**: Export analysis results and visualizations

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd smoke_sim_py
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode (optional):
   ```bash
   pip install -e .
   ```

## Usage

### Running the Dashboard

Start the interactive web dashboard:
```bash
python plotly_dashboard/dashboard.py
```

Then open your browser to `http://127.0.0.1:8052/`

### Project Structure

```
smesh-sensor-placement/
├── plotly_dashboard/          # Interactive web dashboard
│   ├── dashboard.py           # Main dashboard application (Dash)
│   ├── data_loader.py         # Data loading utilities
│   ├── visualization_3d.py    # 3D visualization components
│   └── assets/                # Static files (logos, styles)
├── simulation/                # Core simulation package
│   ├── __init__.py            # Package initialization
│   ├── core/                  # Core simulation modules
│   │   ├── drift_scene.py     # Wind drift dashboard visualization
│   │   ├── terrain_vectors.py # DEM and terrain vector field
│   │   ├── wind_trajectories.py # Particle trajectory simulation
│   │   ├── smoke_points.py    # Burn area sampling
│   │   ├── plume_model.py     # Atmospheric stability classes
│   │   ├── disturbance.py     # Disturbance modeling (legacy)
│   │   └── wind_distribution.py # Wind distribution (legacy)
│   ├── scene/                 # Scene and geometry utilities
│   │   └── scene_parser.py    # Scene parsing (legacy)
│   ├── utils/                 # Utility functions
│   │   ├── geo_utils.py       # Geographic utilities (GeoJSON loading)
│   │   ├── viz_utils.py       # Visualization utilities (color conversion)
│   │   └── trajectory_utils.py # Trajectory utilities (legacy)
│   └── visualization/         # Plotting and visualization (legacy)
│       ├── plot_3d_plume.py
│       ├── plotting.py
│       └── plume_plotting.py
├── scripts/                   # Utility scripts
│   ├── get_historic_wind.py   # Fetch historic wind data
│   ├── get_forecasted_wind.py # Fetch forecasted wind data
│   ├── simulation_parameters.py # Simulation parameter management
│   └── ...
├── data/                      # Data directory
│   ├── BurnData/              # Burn area GeoTIFF files and DEMs
│   │   ├── HenryCoe/
│   │   │   ├── BurnAreas/     # GeoJSON burn area files
│   │   │   ├── SNodeLocations/ # Sensor node locations
│   │   │   └── ReferencePoints/
│   │   └── DEMs_and_Buffered_Burns/
│   ├── FuelConsumptionProfiles/
│   └── StatusUpdateData/
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project configuration
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Dashboard Tabs

### Map View
- Interactive map with burn areas and sensor nodes
- Toggle simulation nodes and burn areas on/off
- Pan and zoom to explore the terrain
- Hover for detailed information

### Wind Data
- **Forecasted Wind**: View predicted wind conditions for the burn location
- **Historic Wind**: Analyze historical wind patterns
- Filter by date range and location
- Wind rose visualization with von Mises distribution overlay
- Export data for further analysis

### Simulation Parameters
- Configure burn time windows
- Set atmospheric stability class
- Adjust simulation parameters
- Save/load parameter configurations

### Wind Drift
- **Interactive Trajectory Visualization**: Real-time particle trajectory simulation
- **Controls**:
  - Particles: Adjust number of trajectories (1-50)
  - Wind Direction: Set wind direction in degrees (0-360°)
  - Terrain Influence: Control terrain effect on trajectories (0-1)
  - Time: Advance through burn areas with 15-minute increments
- **Features**:
  - Cumulative sampling: Current area at 100%, previous areas at 10%
  - Smoldering phase: 6-hour post-fire period with reduced sampling
  - Terrain visualization: DEM heatmap with downhill vectors
  - Burn area progression: Red (active), Black (smoldering), Gray (future)

## Simulation Module

The `simulation/` package provides core functionality for wind drift modeling and trajectory simulation:

### Core Components

**Active Modules:**
- `drift_scene.py` - Wind drift dashboard visualization with cumulative sampling
- `terrain_vectors.py` - DEM loading and terrain gradient computation
- `wind_trajectories.py` - Particle trajectory simulation with terrain influence
- `smoke_points.py` - Burn area polygon loading and point sampling
- `plume_model.py` - Atmospheric stability classification

**Utilities:**
- `geo_utils.py` - GeoJSON polygon loading and parsing
- `viz_utils.py` - Color space conversion for trajectory visualization

### Key Concepts

**Cumulative Sampling**: As time progresses through burn areas:
- Current burn area: 100% of trajectories sampled
- Previous burn areas: 10% of trajectories sampled each
- Creates visual effect of smoke accumulation from past events

**Smoldering Phase**: After all burn areas are processed:
- Lasts 6 hours (24 × 15-minute steps)
- Samples from all areas at 10% rate each
- Represents post-fire cooling and dispersal

### Usage Example

```python
from simulation.core.drift_scene import WindDriftDashboard

# Initialize dashboard
dashboard = WindDriftDashboard(
    dem_path="data/BurnData/DEMs_and_Buffered_Burns/DEM_HenryCoe.tif",
    burn_area_dir="data/BurnData/HenryCoe/BurnAreas"
)

# Create visualization
fig = dashboard.create_figure(
    num_trajectories=10,
    wind_direction=270,
    terrain_influence=0.3,
    burn_area_index=0,
    is_smoldering=False
)
```

## Development

### Install Development Dependencies
```bash
pip install -e ".[dev]"
```

### Run Tests
```bash
pytest
```

### Code Style
The project follows PEP 8 conventions. Format code with:
```bash
black .
isort .
```

## Data Requirements

The project expects the following data structure:
- DEM (Digital Elevation Model) GeoTIFF files in `data/BurnData/DEMs_and_Buffered_Burns/`
- Burn area GeoTIFF files in `data/BurnData/HenryCoe/BurnAreas/`
- Sensor node locations in `data/BurnData/HenryCoe/SNodeLocations/`

## Dependencies

Key dependencies include:
- **numpy, scipy**: Scientific computing
- **pandas**: Data manipulation
- **geopandas, rasterio, shapely**: Geospatial operations
- **plotly, matplotlib**: Visualization
- **dash**: Web dashboard framework
- **h5py, netCDF4**: Data I/O

See `requirements.txt` for complete list.

## License

MIT License - Copyright (c) 2025 Evan Twarog and Daniel Neamati, Stanford University

See LICENSE file for details.

## Authors

- **Evan Twarog** - Stanford University
- **Daniel Neamati** - Stanford University

## Contributing

When contributing to this project:

1. **Code Style**: Follow PEP 8 conventions
   - Use `black` for formatting
   - Use `isort` for import organization
   
2. **Documentation**: Add docstrings to all public functions
   - Include parameter types and descriptions
   - Include return value documentation
   - Add usage examples for complex functions

3. **Module Organization**:
   - Keep related functionality together
   - Use relative imports within the simulation package
   - Minimize external dependencies

4. **Testing**: Add tests for new features
   - Use `pytest` for unit tests
   - Test edge cases and error conditions

5. **Project Structure**: Maintain the existing directory layout
   - Active modules in `simulation/core/`
   - Utilities in `simulation/utils/`
   - Dashboard components in `plotly_dashboard/`

## Acknowledgments

This project is based on the SMeshSmokeValidation.jl package and developed as part of Stanford University coursework.
