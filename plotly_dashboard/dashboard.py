# Standard library
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
import os
import sys
import traceback

# Third-party
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import geopandas as gpd
import pandas as pd
import numpy as np
from dateutil import parser as dtparser
import plotly.graph_objects as go

# Add the scripts directory to the Python path
scripts_dir = Path(__file__).parent.parent / "scripts"
DATA_DIR = Path(__file__).parent.parent / "data" / "BurnData"
sys.path.append(str(scripts_dir))

# Add simulation directory to path
simulation_dir = Path(__file__).parent.parent / "simulation"
sys.path.insert(0, str(simulation_dir))

# Import wind data functions
from get_forecasted_wind import get_wind_data, get_wind_summary, create_wind_dataframe
from get_historic_wind import (
    get_points, get_stations_by_distance, is_asos_station,
    fetch_station_observations, obs_features_to_df, circ_mean_var_deg
)
from simulation_parameters import (
    get_simulation_parameters, update_simulation_parameters, 
    cache_wind_data, get_cached_wind_data, has_wind_data
)
#from visualization_3d import create_3d_visualization
from data_loader import load_all_snodes, load_all_burn_areas, load_geojson, load_snode_data, load_burn_area

# Import atmospheric stability classes
from core.plume_model import AIR_COLUMN_STABILITY_CLASSES

# Import wind drift dashboard
from core.drift_scene import WindDriftDashboard

# ==============================================
# USER CONFIGURATION
# ==============================================

# Data source configuration
LOCATION = "HenryCoe"  
SNODE_DIR = "SNodeLocations"
BURN_AREA_DIR = "BurnAreas"

# Map configuration
MAPBOX_STYLE = "open-street-map"  # Other options: "stamen-terrain", "carto-positron", etc.
DEFAULT_BOUNDS = [-122.5, 37.2, -121.5, 37.8]  # Fallback bounds [minx, miny, maxx, maxy]
MAP_PADDING = 0.1  # 10% padding around data bounds
NODE_SIZE = 20
NODE_OPACITY = 1.0
LINE_WIDTH = 1
AREA_OPACITY = 0.2

# Color scheme
STANFORD_RED = '#8C1515'
STANFORD_LIGHT_RED = '#B83A4B'
STANFORD_DARK_RED = '#620F15'
STANFORD_SAND = '#D2C295'
STANFORD_STONE = '#544948'

# Server configuration
DEBUG = True
PORT = 8052  # Changed from 8051 to 8052 to avoid port conflicts

# ==============================================
# END OF USER CONFIGURATION
# ==============================================

# Initialize the Dash app with callback exceptions suppressed
assets_dir = Path(__file__).parent.parent / "data"
app = dash.Dash(__name__, suppress_callback_exceptions=True, assets_folder=str(assets_dir))
app.title = "Smoke Simulation Dashboard"

# Load data
snodes = load_all_snodes(DATA_DIR / LOCATION / SNODE_DIR)
burn_areas = load_all_burn_areas(DATA_DIR / LOCATION / BURN_AREA_DIR)

# Initialize wind drift dashboard
wind_drift_dashboard = WindDriftDashboard()

# Cache for 3D visualization to avoid regenerating on every tab click
_3d_visualization_cache = {
    'figure': None,
    'timestamp': None
}

# Filter out any None values that might have occurred during loading
snodes = [df for df in snodes if df is not None and not df.empty]
burn_areas = [df for df in burn_areas if df is not None and not df.empty]
# Combine all data for bounds calculation
all_geoms = []
for gdf in snodes + burn_areas:
    if hasattr(gdf, 'geometry') and not gdf.empty:
        all_geoms.extend(gdf.geometry)

# Calculate bounds with fallback values
if all_geoms:
    try:
        minx, miny, maxx, maxy = gpd.GeoSeries(all_geoms).total_bounds
        padding = (maxx - minx) * MAP_PADDING
        bounds = [minx - padding, miny - padding, maxx + padding, maxy + padding]
    except Exception as e:
        print(f"Error calculating bounds: {e}")
        bounds = DEFAULT_BOUNDS
else:
    print("No valid geometries found, using default bounds")
    bounds = DEFAULT_BOUNDS

# Create initial layout with all required components
def serve_layout():
    # Initial loading message for 3D visualization
    initial_3d_content = html.Div('Loading 3D visualization...', id='3d-plot-loading')
    
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Img(src='/assets/nav_logo.png', style={'height': '100px'})
                ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '10px'}),
                html.H1('Simulation Aid: Sensor Placement for Prescribed Wildfire Monitoring', 
                       style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': 'white', 'margin': '0'})
            ], style={'textAlign': 'left', 'backgroundColor': STANFORD_RED, 'padding': '10px 10px', 'display': 'flex', 'alignItems': 'center'}),
            dcc.Tabs(id='tabs', value='map', children=[
                dcc.Tab(label='Map View', value='map'),
                dcc.Tab(label='Wind Data', value='data'),
                dcc.Tab(label='Simulation Parameters', value='sim-params'),
                dcc.Tab(label='Wind Drift', value='wind-drift'),
            ]),
            html.Div(id='tabs-content')
        ], style={'font-family': 'Arial, sans-serif'}),
        
        # Hidden div to store initial data
        html.Div(id='dummy-output', style={'display': 'none'}),
        
        # Hidden div to force initial load
        html.Div(id='initial-load', style={'display': 'none'})
    ])

# Set the layout
app.layout = serve_layout

# Initial callback to ensure components are loaded
@app.callback(
    Output('dummy-output', 'children'),
    [Input('tabs', 'value')]
)
def initialize_components(tab):
    return ''

# Callback for tab content
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'map':
        return html.Div([
            html.Div([
                # Left sidebar for controls
                html.Div([
                    # Simulation Nodes Toggle
                    html.Div([
                        html.H4('Simulation Nodes', style={'marginBottom': '10px', 'fontSize': '1.2em', 'fontWeight': 'bold', 'color': STANFORD_RED}),
                        dcc.Checklist(
                            id='simulation-toggle',
                            options=[
                                {
                                    'label': html.Span(f'Sim {i+1} Nodes', 
                                                     style={'marginLeft': '5px', 'fontSize': '1.05em'}),
                                    'value': f'sim{i+1}'
                                } for i in range(len(snodes))
                            ],
                            value=[f'sim{i+1}' for i in range(len(snodes))],
                            labelStyle={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'},
                            inputStyle={'marginRight': '5px'}
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    # Divider
                    html.Hr(style={'borderTop': '1px solid #ddd', 'margin': '15px 0'}),
                    
                    # Burn Areas Toggle
                    html.Div([
                        html.H4('Burn Areas', style={'marginBottom': '10px', 'fontSize': '1.2em', 'fontWeight': 'bold', 'color': STANFORD_RED}),
                        dcc.Checklist(
                            id='burn-area-toggle',
                            options=[
                                {
                                    'label': html.Span(f'Burn Area {i+1}', 
                                                     style={'marginLeft': '5px', 'fontSize': '1.05em'}),
                                    'value': f'area{i+1}'
                                } for i in range(len(burn_areas))
                            ],
                            value=[f'area{i+1}' for i in range(len(burn_areas))],
                            labelStyle={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'},
                            inputStyle={'marginRight': '5px'}
                        )
                    ])
                ], style={
                    'width': '20%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'padding': '15px',
                    'background': '#f9f9f9',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginRight': '15px'
                }),
                
                # Map container
                html.Div([
                    dcc.Graph(
                        id='map-plot',
                        style={
                            'height': '80vh',
                            'width': '100%',
                            'borderRadius': '8px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        }
                    )
                ], style={
                    'width': '75%',
                    'display': 'inline-block',
                    'verticalAlign': 'top'
                })
            ], style={'display': 'flex'})
        ])
    elif tab == '3d':
        return html.Div([
            html.Div([
                html.H3('3D Visualization', style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div(id='3d-plot-container', children=[
                    dcc.Loading(
                        id='loading-3d',
                        type='circle',
                        children=html.Div(id='3d-plot-content')
                    )
                ])
            ], style={
                'width': '100%',
                'padding': '20px',
                'boxSizing': 'border-box'
            })
        ])
    elif tab == 'data':
        # Calculate center of the map bounds for default values
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        return html.Div([
            html.Div([
                # Left side: Input controls
                html.Div([
                    html.H3('Prescribed Burn Location', style={'color': STANFORD_RED}),
                    html.Div([
                        html.Div([
                            html.Label('Latitude:', style={'marginRight': '10px', 'fontSize': '1.1em', 'fontWeight': 'bold'}),
                            dcc.Input(
                                id='wind-lat-input',
                                type='number',
                                value=round(center_lat, 4),
                                step=0.0001,
                                style={'width': '120px', 'marginRight': '20px', 'fontSize': '1.05em', 'padding': '6px'}
                            ),
                            html.Label('Longitude:', style={'marginRight': '10px', 'fontSize': '1.1em', 'fontWeight': 'bold'}),
                            dcc.Input(
                                id='wind-lon-input',
                                type='number',
                                value=round(center_lon, 4),
                                step=0.0001,
                                style={'width': '120px', 'fontSize': '1.05em', 'padding': '6px'}
                            ),
                        ], style={'marginBottom': '20px'}),
                        
                        html.Div([
                            html.Label('Forecasted Burn Time:', style={'marginBottom': '15px', 'fontWeight': 'bold', 'fontSize': '1.1em', 'color': STANFORD_RED, 'display': 'block'}),
                            html.Div([
                                html.Label('Start Time:', style={'marginBottom': '5px', 'fontWeight': 'bold'}),
                                html.Div([
                                    dcc.Dropdown(
                                        id='start-year',
                                        placeholder='Year',
                                        style={
                                            'width': '100px',
                                            'marginRight': '10px',
                                            'display': 'inline-block',
                                            'verticalAlign': 'top',
                                            'minWidth': '100px'
                                        },
                                        optionHeight=30
                                    ),
                                    dcc.Dropdown(
                                        id='start-month',
                                        placeholder='Month',
                                        style={
                                            'width': '140px',
                                            'marginRight': '10px',
                                            'display': 'inline-block',
                                            'verticalAlign': 'top',
                                            'minWidth': '140px'
                                        },
                                        optionHeight=30
                                    ),
                                    dcc.Dropdown(
                                        id='start-day',
                                        placeholder='Day',
                                        style={
                                            'width': '100px',
                                            'marginRight': '10px',
                                            'display': 'inline-block',
                                            'verticalAlign': 'top',
                                            'minWidth': '100px'
                                        },
                                        optionHeight=30
                                    ),
                                    dcc.Dropdown(
                                        id='start-hour',
                                        placeholder='Hour',
                                        style={
                                            'width': '100px',
                                            'marginRight': '0',
                                            'display': 'inline-block',
                                            'verticalAlign': 'top',
                                            'minWidth': '100px'
                                        },
                                        optionHeight=30
                                    )
                                ], style={'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '20px'}),
                            
                            html.Div([
                                html.Label('End Time:', style={'marginBottom': '5px', 'fontWeight': 'bold'}),
                                html.Div([
                                    dcc.Dropdown(
                                        id='end-year',
                                        placeholder='Year',
                                        style={
                                            'width': '100px',
                                            'marginRight': '10px',
                                            'display': 'inline-block',
                                            'verticalAlign': 'top',
                                            'minWidth': '100px'
                                        },
                                        optionHeight=30
                                    ),
                                    dcc.Dropdown(
                                        id='end-month',
                                        placeholder='Month',
                                        style={
                                            'width': '140px',
                                            'marginRight': '10px',
                                            'display': 'inline-block',
                                            'verticalAlign': 'top',
                                            'minWidth': '140px'
                                        },
                                        optionHeight=30
                                    ),
                                    dcc.Dropdown(
                                        id='end-day',
                                        placeholder='Day',
                                        style={
                                            'width': '100px',
                                            'marginRight': '10px',
                                            'display': 'inline-block',
                                            'verticalAlign': 'top',
                                            'minWidth': '100px'
                                        },
                                        optionHeight=30
                                    ),
                                    dcc.Dropdown(
                                        id='end-hour',
                                        placeholder='Hour',
                                        style={
                                            'width': '100px',
                                            'marginRight': '0',
                                            'display': 'inline-block',
                                            'verticalAlign': 'top',
                                            'minWidth': '100px'
                                        },
                                        optionHeight=30
                                    )
                                ], style={'whiteSpace': 'nowrap'})
                            ], style={'marginBottom': '20px'}),
                            
                        ], style={'marginBottom': '20px'}),
                        
                        html.Button(
                            'Get Forecasted and Historic Wind',
                            id='get-wind-data',
                            style={
                                'backgroundColor': STANFORD_RED,
                                'color': 'white',
                                'border': 'none',
                                'padding': '12px 24px',
                                'borderRadius': '4px',
                                'cursor': 'pointer',
                                'marginBottom': '20px',
                                'fontSize': '1.1em',
                                'fontWeight': 'bold'
                            }
                        ),
                        
                        # Source information
                        html.Div([
                            html.P(
                                'Forecasted wind data is sourced from the National Weather Service (NWS) API. '
                                'Historic wind data is retrieved from ASOS weather stations via the NWS observations API.',
                                style={
                                    'fontSize': '0.9em',
                                    'color': '#666',
                                    'marginTop': '15px',
                                    'marginBottom': '0',
                                    'lineHeight': '1.5'
                                }
                            )
                        ]),
                        
                        # Status message will appear here
                        html.Div(id='time-range-status', style={'color': '#666', 'fontSize': '0.8em', 'marginTop': '10px'})
                    ])
                ], style={
                    'width': '30%',
                    'padding': '20px',
                    'borderRight': '1px solid #ddd',
                    'boxSizing': 'border-box',
                    'height': '100%',
                    'overflowY': 'auto'
                }),
                
                # Right side: Wind roses and data table
                html.Div([
                    # Wind Roses (top)
                    html.Div([
                        # Historic Wind Rose
                        html.Div([
                            html.H3('Historic Wind Rose', style={'color': STANFORD_RED, 'marginTop': '0'}),
                            html.Div(id='wind-rose-plot-historic')
                        ], style={
                            'width': '50%',
                            'display': 'inline-block',
                            'verticalAlign': 'top',
                            'paddingRight': '15px',
                            'boxSizing': 'border-box'
                        }),
                        
                        # Von Mises Wind Rose
                        html.Div([
                            html.H3('Von Mises Wind Rose', style={'color': STANFORD_RED, 'marginTop': '0'}),
                            html.Div(id='wind-rose-plot-vonmises')
                        ], style={
                            'width': '50%',
                            'display': 'inline-block',
                            'verticalAlign': 'top',
                            'boxSizing': 'border-box'
                        })
                    ], style={
                        'width': '100%',
                        'marginBottom': '20px'
                    }),
                    
                    # Wind Data Table (bottom)
                    html.Div([
                        html.H3('Wind Data', style={'color': STANFORD_RED, 'marginTop': '0'}),
                        html.Div(id='wind-data-table-forecasted', style={'marginBottom': '30px'}),
                        html.Div(id='wind-data-table-historic')
                    ], style={
                        'width': '100%',
                        'boxSizing': 'border-box',
                        'overflowY': 'auto',
                        'maxHeight': '40vh'
                    })
                ], style={
                    'width': '70%',
                    'padding': '20px',
                    'boxSizing': 'border-box',
                    'height': '100%',
                    'overflowY': 'auto'
                })
            ], style={
                'display': 'flex',
                'height': 'calc(100vh - 200px)'
            })
        ])
    elif tab == 'sim-params':
        return html.Div([
            html.Div([
                # Main content area
                html.Div([
                    html.H2('Simulation Parameters', style={'color': STANFORD_RED, 'marginBottom': '30px'}),
                    
                    # Forecasted Burn Time Section
                    html.Div([
                        html.H3('Forecasted Burn Time', style={'color': STANFORD_RED, 'marginBottom': '20px'}),
                        html.Div([
                            html.Label('Start Time:', style={'marginBottom': '10px', 'fontWeight': 'bold', 'fontSize': '1.05em', 'display': 'block'}),
                            html.Div([
                                dcc.Dropdown(
                                    id='sim-start-year',
                                    placeholder='Year',
                                    style={
                                        'width': '100px',
                                        'marginRight': '10px',
                                        'display': 'inline-block',
                                        'verticalAlign': 'top',
                                        'minWidth': '100px'
                                    },
                                    optionHeight=30
                                ),
                                dcc.Dropdown(
                                    id='sim-start-month',
                                    placeholder='Month',
                                    style={
                                        'width': '140px',
                                        'marginRight': '10px',
                                        'display': 'inline-block',
                                        'verticalAlign': 'top',
                                        'minWidth': '140px'
                                    },
                                    optionHeight=30
                                ),
                                dcc.Dropdown(
                                    id='sim-start-day',
                                    placeholder='Day',
                                    style={
                                        'width': '100px',
                                        'marginRight': '10px',
                                        'display': 'inline-block',
                                        'verticalAlign': 'top',
                                        'minWidth': '100px'
                                    },
                                    optionHeight=30
                                ),
                                dcc.Dropdown(
                                    id='sim-start-hour',
                                    placeholder='Hour',
                                    style={
                                        'width': '100px',
                                        'marginRight': '0',
                                        'display': 'inline-block',
                                        'verticalAlign': 'top',
                                        'minWidth': '100px'
                                    },
                                    optionHeight=30
                                )
                            ], style={'whiteSpace': 'nowrap', 'marginBottom': '20px'}),
                        ], style={'marginBottom': '30px'}),
                        
                        html.Div([
                            html.Label('End Time:', style={'marginBottom': '10px', 'fontWeight': 'bold', 'fontSize': '1.05em', 'display': 'block'}),
                            html.Div([
                                dcc.Dropdown(
                                    id='sim-end-year',
                                    placeholder='Year',
                                    style={
                                        'width': '100px',
                                        'marginRight': '10px',
                                        'display': 'inline-block',
                                        'verticalAlign': 'top',
                                        'minWidth': '100px'
                                    },
                                    optionHeight=30
                                ),
                                dcc.Dropdown(
                                    id='sim-end-month',
                                    placeholder='Month',
                                    style={
                                        'width': '140px',
                                        'marginRight': '10px',
                                        'display': 'inline-block',
                                        'verticalAlign': 'top',
                                        'minWidth': '140px'
                                    },
                                    optionHeight=30
                                ),
                                dcc.Dropdown(
                                    id='sim-end-day',
                                    placeholder='Day',
                                    style={
                                        'width': '100px',
                                        'marginRight': '10px',
                                        'display': 'inline-block',
                                        'verticalAlign': 'top',
                                        'minWidth': '100px'
                                    },
                                    optionHeight=30
                                ),
                                dcc.Dropdown(
                                    id='sim-end-hour',
                                    placeholder='Hour',
                                    style={
                                        'width': '100px',
                                        'marginRight': '0',
                                        'display': 'inline-block',
                                        'verticalAlign': 'top',
                                        'minWidth': '100px'
                                    },
                                    optionHeight=30
                                )
                            ], style={'whiteSpace': 'nowrap'})
                        ], style={'marginBottom': '30px'}),
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginBottom': '30px'}),
                    
                    # Time Step Duration Section
                    html.Div([
                        html.H3('Time Step Duration', style={'color': STANFORD_RED, 'marginBottom': '20px'}),
                        dcc.Dropdown(
                            id='timestep-duration',
                            options=[
                                {'label': '15 minutes', 'value': '15min'},
                                {'label': '30 minutes', 'value': '30min'},
                                {'label': '1 hour', 'value': '1hr'},
                                {'label': '2 hours', 'value': '2hr'}
                            ],
                            style={
                                'width': '250px',
                                'fontSize': '1.05em'
                            }
                        ),
                        html.P('Select the time step duration for the simulation', 
                               style={'marginTop': '10px', 'color': '#666', 'fontSize': '0.95em'})
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginBottom': '30px'}),
                    
                    # Atmospheric Stability Class Section
                    html.Div([
                        html.H3('Atmospheric Stability Class', style={'color': STANFORD_RED, 'marginBottom': '20px'}),
                        dcc.Dropdown(
                            id='atmospheric-stability',
                            options=[
                                {'label': 'Class A - Very Unstable', 'value': 'A'},
                                {'label': 'Class B - Unstable', 'value': 'B'},
                                {'label': 'Class C - Slightly Unstable', 'value': 'C'},
                                {'label': 'Class D - Neutral', 'value': 'D'},
                                {'label': 'Class E - Slightly Stable', 'value': 'E'},
                                {'label': 'Class F - Very Stable', 'value': 'F'}
                            ],
                            style={
                                'width': '300px',
                                'fontSize': '1.05em'
                            }
                        ),
                        html.P('Select the atmospheric stability class for dispersion modeling', 
                               style={'marginTop': '10px', 'color': '#666', 'fontSize': '0.95em'})
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginBottom': '30px'}),
                    
                ], style={
                    'maxWidth': '800px',
                    'margin': '0 auto',
                    'padding': '40px 20px'
                })
            ], style={
                'height': 'calc(100vh - 200px)',
                'overflowY': 'auto'
            })
        ])

    elif tab == 'wind-drift':
        return html.Div([
            html.Div([
                # Left sidebar with controls
                html.Div([
                    # Control panel
                    html.Div(
                        style={
                            'backgroundColor': '#f9f9f9',
                            'padding': '12px',
                            'borderRadius': '8px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
                            'fontSize': '12px'
                        },
                        children=[
                            # Particles slider
                            html.Div(
                                style={'marginBottom': '15px'},
                                children=[
                                    html.Label('Particles', style={'fontWeight': 'bold', 'marginBottom': '4px', 'display': 'block', 'color': STANFORD_STONE, 'fontSize': '0.85rem'}),
                                    dcc.Slider(
                                        id='wd-particles-slider',
                                        min=1,
                                        max=50,
                                        value=10,
                                        step=1,
                                        marks={i: str(i) for i in range(1, 51, 5)},
                                        tooltip={'placement': 'bottom', 'always_visible': True}
                                    ),
                                    html.Div(id='wd-particles-display', style={'marginTop': '6px', 'color': STANFORD_RED, 'fontSize': '11px'})
                                ]
                            ),
                            
                            # Wind direction slider
                            html.Div(
                                style={'marginBottom': '15px'},
                                children=[
                                    html.Label('Wind Dir (°)', style={'fontWeight': 'bold', 'marginBottom': '4px', 'display': 'block', 'color': STANFORD_STONE, 'fontSize': '0.85rem'}),
                                    dcc.Slider(
                                        id='wd-direction-slider',
                                        min=0,
                                        max=360,
                                        value=270,
                                        step=1,
                                        marks={i: str(i) for i in range(0, 361, 45)},
                                        tooltip={'placement': 'bottom', 'always_visible': True}
                                    ),
                                    html.Div(id='wd-direction-display', style={'marginTop': '6px', 'color': STANFORD_RED, 'fontSize': '11px'})
                                ]
                            ),
                            
                            # Terrain influence slider
                            html.Div(
                                style={'marginBottom': '15px'},
                                children=[
                                    html.Label('Terrain Influence', style={'fontWeight': 'bold', 'marginBottom': '4px', 'display': 'block', 'color': STANFORD_STONE, 'fontSize': '0.85rem'}),
                                    dcc.Slider(
                                        id='wd-influence-slider',
                                        min=0,
                                        max=1,
                                        value=0.3,
                                        step=0.05,
                                        marks={i/10: f'{i/10:.1f}' for i in range(0, 11)},
                                        tooltip={'placement': 'bottom', 'always_visible': True}
                                    ),
                                    html.Div(id='wd-influence-display', style={'marginTop': '6px', 'color': STANFORD_RED, 'fontSize': '11px'})
                                ]
                            ),
                            
                            # Time slider
                            html.Div(
                                style={'marginBottom': '15px'},
                                children=[
                                    html.Label('Time (15min increments)', style={'fontWeight': 'bold', 'marginBottom': '4px', 'display': 'block', 'color': STANFORD_STONE, 'fontSize': '0.85rem'}),
                                    dcc.Slider(
                                        id='wd-time-slider',
                                        min=0,
                                        max=40,
                                        value=0,
                                        step=1,
                                        marks={},
                                        tooltip={'placement': 'bottom', 'always_visible': True}
                                    ),
                                    html.Div(id='wd-time-display', style={'marginTop': '6px', 'color': STANFORD_RED, 'fontSize': '11px'})
                                ]
                            ),
                            
                            # Reseed button
                            html.Div(
                                style={'marginTop': '12px', 'textAlign': 'center'},
                                children=[
                                    html.Button(
                                        'Reseed & Run',
                                        id='wd-reseed-button',
                                        n_clicks=0,
                                        style={
                                            'backgroundColor': STANFORD_RED,
                                            'color': 'white',
                                            'padding': '8px 16px',
                                            'fontSize': '13px',
                                            'fontWeight': 'bold',
                                            'border': 'none',
                                            'borderRadius': '6px',
                                            'cursor': 'pointer',
                                            'transition': 'background-color 0.2s',
                                            'width': '100%'
                                        }
                                    )
                                ]
                            )
                        ]
                    ),
                    
                    # Hidden store for reseed trigger
                    dcc.Store(id='wd-reseed-store', data=0)
                ], style={
                    'width': '20%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'paddingRight': '10px',
                    'boxSizing': 'border-box',
                    'maxHeight': 'calc(100vh - 200px)',
                    'overflowY': 'auto'
                }),
                
                # Right side with graph (80% width)
                html.Div([
                    dcc.Graph(
                        id='wd-wind-drift-graph',
                        style={
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
                            'height': 'calc(100vh - 200px)'
                        }
                    )
                ], style={
                    'width': '80%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'boxSizing': 'border-box'
                })
            ], style={'display': 'flex', 'height': 'calc(100vh - 200px)', 'padding': '10px', 'gap': '10px'})
        ])

def get_geometry_coordinates(geom):
    """Extract coordinates from different geometry types."""
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.coords.xy
        return list(x), list(y)
    elif geom.geom_type == 'Point':
        return [geom.x], [geom.y]
    return [], []

def create_wind_rose_figure(df: pd.DataFrame, title: str = "Wind Rose", overlay_vonmises: bool = True, show_observed: bool = True):
    """Create a Plotly polar bar chart wind rose from wind data with optional von Mises overlay.
    
    Args:
        df: DataFrame with 'wind_speed_mps' and 'wind_dir_deg' columns
        title: Title for the plot
        overlay_vonmises: If True, overlay the fitted von Mises distribution
        show_observed: If True, show the observed data bars; if False, show only von Mises curve
    
    Returns:
        plotly.graph_objects.Figure or None if insufficient data
    """
    if df.empty:
        return None
    
    # Ensure numeric and valid values
    if "wind_speed_mps" not in df.columns or "wind_dir_deg" not in df.columns:
        return None
    
    spd = pd.to_numeric(df["wind_speed_mps"], errors="coerce")
    direc = pd.to_numeric(df["wind_dir_deg"], errors="coerce")
    m = spd.notna() & direc.notna()
    spd = spd[m]
    direc = (direc[m] % 360).values
    
    if len(spd) == 0:
        return None
    
    # Direction sectors (16 sectors, 22.5° each)
    sector = 22.5
    bins_deg = np.arange(0, 360 + sector, sector)
    counts, _ = np.histogram(direc, bins=bins_deg)
    total = counts.sum()
    
    if total == 0:
        return None
    
    # Calculate sector centers and frequencies
    sector_centers = (bins_deg[:-1] + bins_deg[1:]) / 2.0
    frac = counts / total
    
    # Cardinal directions for labels (centered at sector midpoints)
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    # Initialize figure data
    fig_data = []
    max_radial = max(frac) * 1.15
    
    # Add observed data bars if requested
    if show_observed:
        fig_data.append(
            go.Barpolar(
                r=frac,
                theta=sector_centers,
                width=sector,
                marker_color='#8C1515',
                marker_line_color='white',
                marker_line_width=1,
                hovertemplate='<b>%{customdata}</b><br>Frequency: %{r:.1%}<extra></extra>',
                customdata=directions,
                name='Observed',
                opacity=0.7
            )
        )
    
    # Create figure
    fig = go.Figure(data=fig_data)
    
    # Add von Mises overlay if requested
    if overlay_vonmises:
        try:
            from get_historic_wind import fit_vonmises_regression
            
            # Fit von Mises to the data
            vm_result = fit_vonmises_regression(df, speed_col='wind_speed_mps', dir_col='wind_dir_deg', n_components=1)
            mu = vm_result['mu']
            kappa = vm_result['kappa']
            
            # Generate smooth curve for von Mises
            theta_deg = np.linspace(0, 360, 360)
            theta_rad = np.deg2rad(theta_deg)
            
            # Shift mu to be in [0, 2π] range
            mu_shifted = mu % (2 * np.pi)
            
            # Calculate von Mises PDF
            from scipy.stats import vonmises
            vm_model = vonmises(kappa, loc=mu_shifted)
            pdf_values = vm_model.pdf(theta_rad)
            
            # Normalize PDF to match histogram scale
            pdf_normalized = pdf_values / (2 * np.pi) * len(direc)
            
            # Update max radial if von Mises extends further
            if not show_observed:
                max_radial = max(pdf_normalized) * 1.15
            
            # Add von Mises curve as a scatter trace
            fig.add_trace(go.Scatterpolar(
                r=pdf_normalized,
                theta=theta_deg,
                mode='lines',
                name=f'von Mises (κ={kappa:.2f})',
                line=dict(color='red', width=3),
                hovertemplate='<b>von Mises</b><br>Direction: %{theta:.0f}°<br>Value: %{r:.4f}<extra></extra>'
            ))
        except Exception as e:
            print(f"Warning: Could not fit von Mises distribution: {e}", file=sys.stderr)
    
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_radial],
                tickangle=90
            ),
            angularaxis=dict(
                tickvals=[0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5],
                ticktext=directions,
                rotation=90,
                direction='clockwise'
            )
        ),
        height=700,
        margin=dict(l=80, r=80, t=100, b=80),
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

# Single callback for 3D visualization
@app.callback(
    Output('3d-plot-content', 'children'),
    [Input('tabs', 'value'),
     Input('refresh-3d', 'n_clicks')],
    [State('tabs', 'value')],
    prevent_initial_call=True
)
def update_3d_visualization(tab_trigger, refresh_clicks, current_tab):
    # Get the context to see what triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
        
    # Only proceed if we're on the 3D tab or refresh was clicked
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'tabs' and current_tab != '3d':
        raise dash.exceptions.PreventUpdate
    
    # Check if we should use cached figure
    use_cache = trigger_id == 'tabs' and _3d_visualization_cache['figure'] is not None
    
    try:
        if use_cache:
            # Use cached figure if available and tab was clicked (not refresh)
            fig = _3d_visualization_cache['figure']
        else:
            # Generate new 3D visualization (refresh button clicked or first load)
            fig = create_3d_visualization(DATA_DIR, LOCATION)
            
            # Cache the figure for future tab clicks
            _3d_visualization_cache['figure'] = fig
            _3d_visualization_cache['timestamp'] = datetime.now()
        
        if fig is None:
            raise ValueError("3D visualization returned None")
        
        # Ensure the figure has a proper height
        if not fig.layout.height:
            fig.update_layout(height=800)
            
        # Return the graph component with loading state and cache status
        cache_status = html.Div(
            f"{'(Cached)' if use_cache else '(Freshly loaded)'}",
            style={
                'textAlign': 'center',
                'fontSize': '0.85em',
                'color': '#666',
                'marginBottom': '10px'
            }
        ) if _3d_visualization_cache['timestamp'] else None
        
        return html.Div([
            cache_status,
            dcc.Loading(
                id="loading-3d",
                type="circle",
                children=[
                    dcc.Graph(
                        id='3d-plot',
                        figure=fig,
                        style={
                            'height': '80vh',
                            'width': '100%',
                            'borderRadius': '8px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        },
                        config={
                            'displayModeBar': True,
                            'scrollZoom': True,
                            'displaylogo': False
                        }
                    )
                ]
            ),
            # Add a refresh button
            html.Div([
                html.Button('Refresh 3D View', 
                          id='refresh-3d', 
                          n_clicks=0,
                          className='btn btn-primary',
                          style={
                              'margin': '10px', 
                              'padding': '10px 15px', 
                              'cursor': 'pointer',
                              'backgroundColor': STANFORD_RED,
                              'color': 'white',
                              'border': 'none',
                              'borderRadius': '4px',
                              'fontSize': '1.05em'
                          })
            ], style={'textAlign': 'center'})
        ])
    except Exception as e:
        error_msg = f"Error in 3D visualization: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return html.Div([
            html.H4('Error loading 3D visualization'),
            html.Pre(str(error_msg), style={
                'whiteSpace': 'pre-wrap',
                'wordBreak': 'break-all',
                'backgroundColor': '#f8f9fa',
                'padding': '10px',
                'borderRadius': '5px',
                'maxHeight': '300px',
                'overflow': 'auto'
            }),
            html.Button('Try Again', id='retry-3d', n_clicks=0, 
                      style={'margin': '10px', 'padding': '10px 15px', 'cursor': 'pointer'})
        ], style={'color': 'red', 'padding': '20px'})

def get_days_in_month(year, month):
    """Get the number of days in a given month, accounting for leap years."""
    if month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29
        return 28
    if month in [4, 6, 9, 11]:
        return 30
    return 31

def generate_time_options():
    now = datetime.now()
    max_date = now + timedelta(days=10)
    
    # Current year and next year if within 10 days of year end
    years = [{'label': str(now.year), 'value': now.year}]
    if now.month == 12 and now.day >= 22:  # If we're within 10 days of year end
        years.append({'label': str(now.year + 1), 'value': now.year + 1})
    
    # Months with proper labels and values
    months = [
        {'label': 'January', 'value': 1}, {'label': 'February', 'value': 2},
        {'label': 'March', 'value': 3}, {'label': 'April', 'value': 4},
        {'label': 'May', 'value': 5}, {'label': 'June', 'value': 6},
        {'label': 'July', 'value': 7}, {'label': 'August', 'value': 8},
        {'label': 'September', 'value': 9}, {'label': 'October', 'value': 10},
        {'label': 'November', 'value': 11}, {'label': 'December', 'value': 12}
    ]
    
    # Days and hours for current month
    days_in_month = get_days_in_month(now.year, now.month)
    days = [{'label': str(d), 'value': d} for d in range(1, days_in_month + 1)]
    hours = [{'label': f"{h:02d}:00", 'value': h} for h in range(24)]
    
    # Default values (current time)
    default_values = {
        'year': now.year,
        'month': now.month,
        'day': now.day,
        'hour': now.hour
    }
    
    # Maximum allowed values (10 days from now)
    max_values = {
        'year': max_date.year,
        'month': max_date.month,
        'day': max_date.day,
        'hour': max_date.hour
    }
    
    return years, months, days, hours, default_values, max_values

@app.callback(
    [Output('start-year', 'options'),
     Output('start-month', 'options'),
     Output('start-day', 'options'),
     Output('start-hour', 'options'),
     Output('end-year', 'options'),
     Output('end-month', 'options'),
     Output('end-day', 'options'),
     Output('end-hour', 'options'),
     Output('start-year', 'value'),
     Output('start-month', 'value'),
     Output('start-day', 'value'),
     Output('start-hour', 'value'),
     Output('end-year', 'value'),
     Output('end-month', 'value'),
     Output('end-day', 'value'),
     Output('end-hour', 'value')],
    [Input('tabs', 'value')]
)
def update_dropdown_options(tab):
    if tab != 'data':
        raise dash.exceptions.PreventUpdate
        
    years, months, days, hours, default_values, max_values = generate_time_options()
    now = datetime.now()
    
    # Set default start time to current time
    start_values = [
        now.year,
        now.month,
        now.day,
        now.hour
    ]
    
    # Set default end time to current time + 1 hour (capped at 10 days from now)
    end_time = min(now + timedelta(hours=1), now + timedelta(days=10))
    end_values = [
        end_time.year,
        end_time.month,
        end_time.day,
        end_time.hour
    ]
    
    # Generate day options for both start and end months
    start_days = [{'label': str(d), 'value': d} for d in range(1, get_days_in_month(start_values[0], start_values[1]) + 1)]
    end_days = [{'label': str(d), 'value': d} for d in range(1, get_days_in_month(end_values[0], end_values[1]) + 1)]
    
    return [
        years,  # start year options
        months,  # start month options
        start_days,  # start day options
        hours,  # start hour options
        years,  # end year options
        months,  # end month options
        end_days,  # end day options
        hours,  # end hour options
        *start_values,  # start values (year, month, day, hour)
        *end_values  # end values (year, month, day, hour)
    ]

# Callback for handling forecasted wind data fetching and display
@app.callback(
    [Output('wind-data-table-forecasted', 'children'),
     Output('time-range-status', 'children')],
    [Input('get-wind-data', 'n_clicks')],
    [State('wind-lat-input', 'value'),
     State('wind-lon-input', 'value'),
     State('start-year', 'value'),
     State('start-month', 'value'),
     State('start-day', 'value'),
     State('start-hour', 'value'),
     State('end-year', 'value'),
     State('end-month', 'value'),
     State('end-day', 'value'),
     State('end-hour', 'value')],
    prevent_initial_call=True
)
def update_wind_data(n_clicks, lat, lon, 
                   start_year, start_month, start_day, start_hour,
                   end_year, end_month, end_day, end_hour):
    
    # Create datetime objects from dropdown values
    start_time = None
    end_time = None
    
    if all([start_year, start_month, start_day, start_hour is not None]):
        try:
            start_time = datetime(
                year=start_year,
                month=start_month,
                day=start_day,
                hour=start_hour
            )
        except (ValueError, TypeError):
            return "Invalid start time specified", ""
    
    if all([end_year, end_month, end_day, end_hour is not None]):
        try:
            end_time = datetime(
                year=end_year,
                month=end_month,
                day=end_day,
                hour=end_hour
            )
        except (ValueError, TypeError):
            return "Invalid end time specified", ""
    
    # Validate time range
    if start_time and end_time and start_time > end_time:
        return "Start time must be before end time", ""
    if n_clicks is None or lat is None or lon is None:
        return dash.no_update, dash.no_update
    
    # Parse time inputs
    start_time = pd.to_datetime(start_time) if start_time else None
    end_time = pd.to_datetime(end_time) if end_time else None
    
    # Get wind data using the refactored function
    wind_result = get_wind_data(lat, lon, start_time=start_time, end_time=end_time)
    
    if wind_result['status'] == 'error':
        return html.Div([
            html.H4('Error', style={'color': 'red'}),
            html.P(wind_result['message'])
        ]), ""
    
    # Get summary using the utility function
    summary_data = get_wind_summary(wind_result)
    
    # Cache the location and time parameters
    update_simulation_parameters(
        latitude=lat,
        longitude=lon,
        burn_start_time=start_time,
        burn_end_time=end_time
    )
    
    # Create wind dataframe and cache it
    df = create_wind_dataframe(wind_result)
    cache_wind_data(
        forecasted=df,
        retrieved_at=summary_data['retrieved_at']
    )
    
    # Create summary display
    summary = html.Div([
        html.H4('Current Wind Conditions', style={'marginBottom': '15px', 'color': STANFORD_RED, 'fontSize': '1.2em'}),
        html.Div([
            html.Div([
                html.Div('Wind Speed', style={'fontWeight': 'bold', 'fontSize': '1.05em'}),
                html.Div(f"{summary_data['speed_mps']:.1f} m/s ({summary_data['speed_knots']:.1f} kts)", style={'fontSize': '1.05em'})
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Div('Wind Direction', style={'fontWeight': 'bold', 'fontSize': '1.05em'}),
                html.Div(f"{summary_data['direction_deg']:.0f}° {summary_data['direction_cardinal']}", style={'fontSize': '1.05em'})
            ])
        ]),
        html.Hr(),
        html.Div([
            html.Div([
                html.Div('Location', style={'fontWeight': 'bold', 'fontSize': '1.05em'}),
                html.Div(f"Lat: {lat:.4f}, Lon: {lon:.4f}", style={'fontSize': '1.05em'})
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Div('Forecast Updated', style={'fontWeight': 'bold', 'fontSize': '1.05em'}),
                html.Div(summary_data['retrieved_at'], style={'fontSize': '1.05em'})
            ])
        ])
    ])
    
    # Round speed columns to nearest 0.1
    if 'speed_mps' in df.columns:
        df['speed_mps'] = df['speed_mps'].round(1)
    if 'speed_knots' in df.columns:
        df['speed_knots'] = df['speed_knots'].round(1)
    
    table = dash_table.DataTable(
        id='wind-data-datatable',
        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '8px',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_header={
            'backgroundColor': STANFORD_RED,
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    # Create status message
    status_msg = []
    if start_time:
        status_msg.append(f"From: {start_time.strftime('%Y-%m-%d %H:00')}")
    if end_time:
        status_msg.append(f"To: {end_time.strftime('%Y-%m-%d %H:00')}")
    status_msg = " | ".join(status_msg) if status_msg else "Using default time range (most recent data)"
    
    return [summary, html.Hr(), table], status_msg

# Callback for handling historic wind data fetching and display
@app.callback(
    [Output('wind-data-table-historic', 'children'),
     Output('wind-rose-plot-historic', 'children'),
     Output('wind-rose-plot-vonmises', 'children')],
    [Input('get-wind-data', 'n_clicks')],
    [State('wind-lat-input', 'value'),
     State('wind-lon-input', 'value')],
    prevent_initial_call=True
)
def update_historic_wind_data(n_clicks, lat, lon):
    """Fetch historic wind data from ASOS stations and display with wind rose."""
    if n_clicks is None or lat is None or lon is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    try:
        # Get nearest ASOS stations
        points = get_points(lat, lon)
        candidates = get_stations_by_distance(points, lat, lon, limit=12)
        
        # Keep only ASOS stations; if none, fall back to ICAO-like codes
        asos_candidates = [(d, sid, nm) for (d, sid, nm) in candidates if is_asos_station(sid)]
        if asos_candidates:
            candidates = asos_candidates
        else:
            icao_like = [(d, sid, nm) for (d, sid, nm) in candidates if len(str(sid)) == 4 and str(sid).isalpha()]
            if icao_like:
                candidates = icao_like
        
        # Fetch data from stations
        from datetime import timezone as tz_module
        end = datetime.now(tz_module.utc)
        start = end - timedelta(days=7)
        
        df = pd.DataFrame()
        station_name = "Unknown"
        
        for _, sid, name in candidates:
            feats = fetch_station_observations(sid, start, end)
            cur = obs_features_to_df(feats, sid)
            if cur.empty:
                continue
            # Keep station only if it has any non-zero wind
            if cur["wind_speed_mps"].fillna(0).gt(0).any():
                df = cur
                station_name = name
                break
        
        # If still empty, fall back to nearest candidate even if zero
        if df.empty and candidates:
            _, sid, name = candidates[0]
            feats = fetch_station_observations(sid, start, end)
            df = obs_features_to_df(feats, sid)
            station_name = name
        
        if df.empty:
            return (
                html.Div("No historic wind data available for this location", style={'color': 'red'}),
                html.Div("No data found", style={'color': 'red'}),
                html.Div("No data available", style={'color': 'red'})
            )
        
        # Filter and process data
        df["wind_speed_mps"] = pd.to_numeric(df["wind_speed_mps"], errors="coerce")
        df_filtered = df.loc[df["wind_speed_mps"] > 0].copy()
        
        # Create summary
        if not df_filtered.empty:
            speed_mean = float(df_filtered["wind_speed_mps"].mean())
            speed_var = float(df_filtered["wind_speed_mps"].var(ddof=1)) if len(df_filtered) > 1 else np.nan
            dir_series = pd.to_numeric(df_filtered["wind_dir_deg"], errors="coerce").dropna()
            if not dir_series.empty:
                dir_mean_deg, dir_circ_var, dir_circ_std, R = circ_mean_var_deg(dir_series.values)
            else:
                dir_mean_deg = dir_circ_var = dir_circ_std = R = np.nan
        else:
            speed_mean = speed_var = dir_mean_deg = dir_circ_var = dir_circ_std = R = np.nan
        
        summary = html.Div([
            html.H4('Historic Wind Conditions (Last 7 Days)', style={'marginBottom': '15px', 'color': STANFORD_RED, 'fontSize': '1.2em'}),
            html.Div([
                html.Div([
                    html.Div('Mean Wind Speed', style={'fontWeight': 'bold', 'fontSize': '1.05em'}),
                    html.Div(f"{speed_mean:.1f} m/s" if not np.isnan(speed_mean) else "N/A", style={'fontSize': '1.05em'})
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Div('Mean Wind Direction', style={'fontWeight': 'bold', 'fontSize': '1.05em'}),
                    html.Div(f"{dir_mean_deg:.0f}°" if not np.isnan(dir_mean_deg) else "N/A", style={'fontSize': '1.05em'})
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Div('Circular Variance', style={'fontWeight': 'bold', 'fontSize': '1.05em'}),
                    html.Div(f"{dir_circ_var:.3f}" if not np.isnan(dir_circ_var) else "N/A", style={'fontSize': '1.05em'})
                ])
            ]),
            html.Hr(),
            html.Div([
                html.Div([
                    html.Div('Station', style={'fontWeight': 'bold', 'fontSize': '1.05em'}),
                    html.Div(station_name, style={'fontSize': '1.05em'})
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Div('Location', style={'fontWeight': 'bold', 'fontSize': '1.05em'}),
                    html.Div(f"Lat: {lat:.4f}, Lon: {lon:.4f}", style={'fontSize': '1.05em'})
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Div('Data Points', style={'fontWeight': 'bold', 'fontSize': '1.05em'}),
                    html.Div(f"{len(df_filtered)} observations", style={'fontSize': '1.05em'})
                ])
            ])
        ])
        
        # Create data table
        df_display = df_filtered.copy()
        df_display["time_utc"] = df_display["time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Round speed columns to nearest 0.1
        if 'wind_speed_mps' in df_display.columns:
            df_display['wind_speed_mps'] = df_display['wind_speed_mps'].round(1)
        if 'wind_speed_knots' in df_display.columns:
            df_display['wind_speed_knots'] = df_display['wind_speed_knots'].round(1)
        
        table = dash_table.DataTable(
            id='wind-data-datatable',
            columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in df_display.columns],
            data=df_display.to_dict('records'),
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '8px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': STANFORD_RED,
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
        
        # Create wind roses
        # Historic wind rose (without von Mises overlay)
        wind_rose_historic_fig = create_wind_rose_figure(df_filtered, title="Historic Wind Rose (Last 7 Days)", overlay_vonmises=False, show_observed=True)
        wind_rose_historic_component = dcc.Graph(figure=wind_rose_historic_fig) if wind_rose_historic_fig else html.Div("No wind data available for wind rose")
        
        # Von Mises wind rose (von Mises distribution only, no observed data)
        wind_rose_vonmises_fig = create_wind_rose_figure(df_filtered, title="Von Mises Fit (Last 7 Days)", overlay_vonmises=True, show_observed=False)
        wind_rose_vonmises_component = dcc.Graph(figure=wind_rose_vonmises_fig) if wind_rose_vonmises_fig else html.Div("No wind data available for von Mises fit")
        
        return [summary, html.Hr(), table], wind_rose_historic_component, wind_rose_vonmises_component
        
    except Exception as e:
        error_msg = f"Error fetching historic wind data: {str(e)}"
        return (
            html.Div(error_msg, style={'color': 'red'}),
            html.Div(error_msg, style={'color': 'red'}),
            html.Div(error_msg, style={'color': 'red'})
        )

# Callback for updating the map
@app.callback(
    Output('map-plot', 'figure'),
    [Input('simulation-toggle', 'value'),
     Input('burn-area-toggle', 'value')]
)
def update_map(visible_simulations, visible_areas):
    fig = go.Figure()
    
    # Add burn areas
    for i, area_df in enumerate(burn_areas, 1):
        area_id = f'area{i}'
        if area_id in visible_areas and not area_df.empty and hasattr(area_df, 'geometry'):
            for geom in area_df.geometry:
                if geom is not None and not geom.is_empty:
                    x, y = get_geometry_coordinates(geom)
                    if x and y:  # Only add if we have valid coordinates
                        fig.add_trace(go.Scattermapbox(
                            lon=x,
                            lat=y,
                            mode='lines+markers' if len(x) == 1 else 'lines',
                            fill='toself' if len(x) > 2 else None,
                            name=f'Burn Area {i}',
                            line=dict(width=LINE_WIDTH, color=STANFORD_RED),
                            fillcolor=f'rgba(140, 21, 21, {AREA_OPACITY})',
                            hoverinfo='text',
                            text=f'Burn Area {i}'
                        ))
    
    # Add sensor nodes
    colors = ['#0066CC', '#00CC66', '#FF9900']  # Blue, Green, Orange for better differentiation
    for i, (snode_df, color) in enumerate(zip(snodes, colors), 1):
        sim_id = f'sim{i}'
        if sim_id in visible_simulations and not snode_df.empty and hasattr(snode_df, 'geometry'):
            # Convert to Web Mercator for better visualization if needed
            snode_df = snode_df.to_crs(epsg=4326) if snode_df.crs else snode_df
            
            # Add points for this simulation
            fig.add_trace(go.Scattermapbox(
                lon=snode_df.geometry.x,
                lat=snode_df.geometry.y,
                mode='markers',
                marker=dict(
                    size=NODE_SIZE,
                    color=color,
                    opacity=NODE_OPACITY
                ),
                name=f'Sim {i} Nodes',
                text=snode_df.get('label', f'Node {i}'),
                hoverinfo='text'
            ))
    
    # Calculate center point for the map
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Calculate appropriate zoom level to fit burn area bounds
    # Rough approximation: zoom level based on bounds width
    lon_range = bounds[2] - bounds[0]
    zoom_level = max(13, 16 - (lon_range * 4))  # Adjust zoom to fit bounds, closer view
    
    # Update layout with better defaults
    fig.update_layout(
        mapbox_style=MAPBOX_STYLE,
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_level,
            style=MAPBOX_STYLE
        ),
        margin={"r":0, "t":0, "l":0, "b":0},
        showlegend=True,
        legend=dict(
            bgcolor='white',
            bordercolor=STANFORD_DARK_RED,
            borderwidth=1,
            x=0.01,
            y=0.99,
            yanchor='top'
        ),
        hovermode='closest'
    )
    
    return fig

# Callback to populate simulation parameters options and values when tab is clicked
@app.callback(
    [Output('sim-start-year', 'options'),
     Output('sim-start-year', 'value'),
     Output('sim-start-month', 'options'),
     Output('sim-start-month', 'value'),
     Output('sim-start-day', 'options'),
     Output('sim-start-day', 'value'),
     Output('sim-start-hour', 'options'),
     Output('sim-start-hour', 'value'),
     Output('sim-end-year', 'options'),
     Output('sim-end-year', 'value'),
     Output('sim-end-month', 'options'),
     Output('sim-end-month', 'value'),
     Output('sim-end-day', 'options'),
     Output('sim-end-day', 'value'),
     Output('sim-end-hour', 'options'),
     Output('sim-end-hour', 'value'),
     Output('timestep-duration', 'value'),
     Output('atmospheric-stability', 'value')],
    [Input('tabs', 'value')],
    prevent_initial_call=True
)
def update_sim_params_on_tab_click(tab):
    """Update simulation parameters options and values when the tab is clicked."""
    if tab != 'sim-params':
        raise dash.exceptions.PreventUpdate
    
    # Get cached simulation parameters
    sim_params = get_simulation_parameters()
    
    # Extract values or use defaults
    start_year = sim_params.burn_start_time.year if sim_params.burn_start_time else None
    start_month = sim_params.burn_start_time.month if sim_params.burn_start_time else None
    start_day = sim_params.burn_start_time.day if sim_params.burn_start_time else None
    start_hour = sim_params.burn_start_time.hour if sim_params.burn_start_time else None
    
    end_year = sim_params.burn_end_time.year if sim_params.burn_end_time else None
    end_month = sim_params.burn_end_time.month if sim_params.burn_end_time else None
    end_day = sim_params.burn_end_time.day if sim_params.burn_end_time else None
    end_hour = sim_params.burn_end_time.hour if sim_params.burn_end_time else None
    
    # Generate year options (current year ± 5 years)
    current_year = datetime.now().year
    year_options = [{'label': str(y), 'value': y} for y in range(current_year - 5, current_year + 6)]
    
    # Generate month options
    month_options = [{'label': f'{m:02d}', 'value': m} for m in range(1, 13)]
    
    # Generate day options
    day_options = [{'label': f'{d:02d}', 'value': d} for d in range(1, 32)]
    
    # Generate hour options
    hour_options = [{'label': f'{h:02d}:00', 'value': h} for h in range(0, 24)]
    
    return (year_options, start_year,
            month_options, start_month,
            day_options, start_day,
            hour_options, start_hour,
            year_options, end_year,
            month_options, end_month,
            day_options, end_day,
            hour_options, end_hour,
            sim_params.timestep_duration,
            sim_params.atmospheric_stability)

# Wind Drift Tab Callbacks
@app.callback(
    [Output('wd-time-slider', 'max'),
     Output('wd-time-slider', 'marks')],
    Input('wd-wind-drift-graph', 'id'),
    prevent_initial_call=False
)
def initialize_wd_time_slider(_):
    """Initialize time slider for wind drift tab."""
    num_files = len(wind_drift_dashboard.burn_area_files)
    if num_files == 0:
        return 0, {}
    
    # 4 steps per burn area + 24 steps for 6 hours of smoldering
    max_time_step = (num_files - 1) * 4 + 3 + 24
    
    # Create marks for every 4 steps (one per burn area)
    marks = {}
    for area_idx in range(num_files):
        time_step = area_idx * 4
        marks[time_step] = str(area_idx + 1)
    
    # Add smoldering mark at the end
    smoldering_start = num_files * 4
    marks[smoldering_start] = 'Smoldering'
    
    return max_time_step, marks


@app.callback(
    Output('wd-reseed-store', 'data'),
    Input('wd-reseed-button', 'n_clicks'),
    prevent_initial_call=True
)
def handle_wd_reseed_click(n_clicks):
    """Handle reseed button click for wind drift tab."""
    return n_clicks


@app.callback(
    [Output('wd-wind-drift-graph', 'figure'),
     Output('wd-particles-display', 'children'),
     Output('wd-direction-display', 'children'),
     Output('wd-influence-display', 'children'),
     Output('wd-time-display', 'children')],
    [Input('wd-particles-slider', 'value'),
     Input('wd-direction-slider', 'value'),
     Input('wd-influence-slider', 'value'),
     Input('wd-time-slider', 'value'),
     Input('wd-reseed-store', 'data')]
)
def update_wd_visualization(num_particles, wind_direction, terrain_influence, time_step, reseed_trigger):
    """Update wind drift visualization based on slider values."""
    num_files = len(wind_drift_dashboard.burn_area_files)
    smoldering_start_step = num_files * 4
    
    # Determine if in smoldering phase
    if time_step >= smoldering_start_step:
        burn_area_index = num_files - 1
        is_smoldering = True
        phase = "Smoldering"
    else:
        burn_area_index = time_step // 4
        burn_area_index = max(0, min(burn_area_index, num_files - 1))
        is_smoldering = False
        phase = "Active"
    
    # Calculate minutes elapsed
    minutes_elapsed = time_step * 15
    hours = minutes_elapsed // 60
    minutes = minutes_elapsed % 60
    
    fig = wind_drift_dashboard.create_figure(num_particles, wind_direction, terrain_influence, burn_area_index, is_smoldering=is_smoldering)
    
    time_str = f'T+{hours}h {minutes}m' if hours > 0 else f'T+{minutes}m'
    
    return (
        fig,
        f'Trajectories: {num_particles}',
        f'Direction: {wind_direction}°',
        f'Influence: {terrain_influence:.2f}',
        f'Time: {time_str} ({phase})'
    )

if __name__ == '__main__':
    app.run_server(debug=DEBUG, port=PORT)