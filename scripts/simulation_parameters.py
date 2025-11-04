"""
Simulation parameters cache and management.

This module manages the caching and retrieval of simulation parameters,
including wind data, burn times, and simulation settings.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class SimulationParameters:
    """Container for all simulation parameters."""
    
    # Burn location
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Forecasted burn time
    burn_start_time: Optional[datetime] = None
    burn_end_time: Optional[datetime] = None
    
    # Simulation settings
    timestep_duration: str = "30min"  # 15min, 30min, 1hr, 2hr
    atmospheric_stability: str = "D"  # A-F
    
    # Wind data (cached)
    forecasted_wind_data: Optional[pd.DataFrame] = None
    historic_wind_data: Optional[pd.DataFrame] = None
    
    # Metadata
    last_updated: Optional[datetime] = None
    wind_data_retrieved_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary, handling DataFrames and datetimes."""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'burn_start_time': self.burn_start_time.isoformat() if self.burn_start_time else None,
            'burn_end_time': self.burn_end_time.isoformat() if self.burn_end_time else None,
            'timestep_duration': self.timestep_duration,
            'atmospheric_stability': self.atmospheric_stability,
            'forecasted_wind_data': self.forecasted_wind_data.to_dict('records') if self.forecasted_wind_data is not None else None,
            'historic_wind_data': self.historic_wind_data.to_dict('records') if self.historic_wind_data is not None else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'wind_data_retrieved_at': self.wind_data_retrieved_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationParameters':
        """Create SimulationParameters from dictionary."""
        params = cls()
        params.latitude = data.get('latitude')
        params.longitude = data.get('longitude')
        
        if data.get('burn_start_time'):
            params.burn_start_time = datetime.fromisoformat(data['burn_start_time'])
        if data.get('burn_end_time'):
            params.burn_end_time = datetime.fromisoformat(data['burn_end_time'])
        
        params.timestep_duration = data.get('timestep_duration', '30min')
        params.atmospheric_stability = data.get('atmospheric_stability', 'D')
        
        if data.get('forecasted_wind_data'):
            params.forecasted_wind_data = pd.DataFrame(data['forecasted_wind_data'])
        if data.get('historic_wind_data'):
            params.historic_wind_data = pd.DataFrame(data['historic_wind_data'])
        
        if data.get('last_updated'):
            params.last_updated = datetime.fromisoformat(data['last_updated'])
        params.wind_data_retrieved_at = data.get('wind_data_retrieved_at')
        
        return params


# Global cache instance
_simulation_cache = SimulationParameters()


def get_simulation_parameters() -> SimulationParameters:
    """Get the current simulation parameters."""
    return _simulation_cache


def update_simulation_parameters(**kwargs) -> SimulationParameters:
    """Update simulation parameters and return the updated object."""
    global _simulation_cache
    
    for key, value in kwargs.items():
        if hasattr(_simulation_cache, key):
            setattr(_simulation_cache, key, value)
    
    _simulation_cache.last_updated = datetime.now()
    return _simulation_cache


def cache_wind_data(forecasted: Optional[pd.DataFrame] = None, 
                   historic: Optional[pd.DataFrame] = None,
                   retrieved_at: Optional[str] = None) -> SimulationParameters:
    """Cache wind data and return updated parameters."""
    global _simulation_cache
    
    if forecasted is not None:
        _simulation_cache.forecasted_wind_data = forecasted
    if historic is not None:
        _simulation_cache.historic_wind_data = historic
    if retrieved_at is not None:
        _simulation_cache.wind_data_retrieved_at = retrieved_at
    
    _simulation_cache.last_updated = datetime.now()
    return _simulation_cache


def clear_cache() -> None:
    """Clear all cached simulation parameters."""
    global _simulation_cache
    _simulation_cache = SimulationParameters()


def get_cached_wind_data() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Get cached wind data (forecasted, historic)."""
    return _simulation_cache.forecasted_wind_data, _simulation_cache.historic_wind_data


def has_wind_data() -> bool:
    """Check if wind data is cached."""
    return (_simulation_cache.forecasted_wind_data is not None or 
            _simulation_cache.historic_wind_data is not None)
