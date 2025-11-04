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
smoke_sim_py/
├── plotly_dashboard/          # Interactive web dashboard
│   ├── dashboard.py           # Main dashboard application
│   ├── visualization_3d.py    # 3D visualization components
│   └── assets/                # Static files (logos, styles)
├── simulation/                # Main package directory
│   ├── core/                  # Core simulation logic
│   │   ├── wind_distribution.py
│   │   ├── smoke_model.py
│   │   └── ...
│   ├── scene/                 # Scene and geometry utilities
│   ├── utils/                 # Utility functions (geo_utils, etc.)
│   └── visualization/         # Plotting and visualization utilities
├── scripts/                   # Utility scripts
│   ├── get_historic_wind.py   # Fetch historic wind data
│   ├── plot_3d_scene.py       # Generate 3D visualizations
│   └── ...
├── data/                      # Data directory
│   ├── BurnData/              # Burn area GeoTIFF files and DEMs
│   ├── FuelConsumptionProfiles/
│   └── StatusUpdateData/
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project configuration
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Dashboard Tabs

### 2D Map View
- Interactive map with burn areas and sensor nodes
- Toggle simulation nodes and burn areas on/off
- Pan and zoom to explore the terrain
- Hover for detailed information

### Wind Data
- **Forecasted Wind**: View predicted wind conditions for the burn location
- **Historic Wind**: Analyze historical wind patterns
- Filter by date range and location
- Export data for further analysis

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

## Acknowledgments

This project is based on the SMeshSmokeValidation.jl package and developed as part of Stanford University coursework.
