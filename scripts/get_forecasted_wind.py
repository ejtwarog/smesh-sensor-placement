#!/usr/bin/env python3
import sys
import csv
import re
import requests
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtp
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

def parse_valid_time(valid_time: str):
    """
    Parses strings like:
      "2025-10-14T10:00:00+00:00/PT3H"
      "2025-10-14T13:00:00+00:00/PT1H"
    Returns (start_dt_utc, duration_hours:int)
    """
    if "/" in valid_time:
        start_str, dur_str = valid_time.split("/", 1)
        start = dtp.isoparse(start_str).astimezone(timezone.utc)
        # Parse duration like PT1H, PT3H, PT30M
        m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", dur_str)
        if not m:
            raise ValueError(f"Unhandled ISO8601 duration: {dur_str}")
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        total_hours = hours + (1 if minutes >= 30 else 0)  # round 30m to 1h
        total_hours = max(1, total_hours)  # ensure at least 1 hour
    else:
        # Sometimes it's a single timestamp; treat as 1 hour
        start = dtp.isoparse(valid_time).astimezone(timezone.utc)
        total_hours = 1
    return start, total_hours

def to_mps(value, uom: str):
    """
    Convert value to m/s based on Weather.gov uom strings like:
      "wmoUnit:m_s-1", "wmoUnit:km_h-1", "wmoUnit:kn"
    """
    if value is None:
        return None
    uom = (uom or "").lower()
    if "m_s-1" in uom:
        return value
    if "km_h-1" in uom or "km/h" in uom:
        return value / 3.6
    if "kn" in uom or "kt" in uom or "knot" in uom:
        return value * 0.514444
    if "m_h-1" in uom:  # mph
        return value * 0.44704
    # Fallback: assume already m/s
    return value

def mps_to_knots(v):
    return None if v is None else v / 0.514444

def main():
    if len(sys.argv) < 3:
        print("Usage: get_wind.py <lat> <lon> [out.csv]", file=sys.stderr)
        sys.exit(1)
    lat, lon = float(sys.argv[1]), float(sys.argv[2])
    out_path = sys.argv[3] if len(sys.argv) > 3 else "wind_hourly.csv"

    headers = {
        # NWS asks for a real UA with contact:
        "User-Agent": "evan-wind-fetch/1.0 (evantwarog@example.com)",
        "Accept": "application/geo+json"
    }

    # 1) Point metadata
    pt = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=headers, timeout=30).json()
    grid_url = pt["properties"]["forecastGridData"]

    # 2) Grid data (contains windSpeed & windDirection with uom + values)
    grid = requests.get(grid_url, headers=headers, timeout=30).json()

    sp = grid["properties"]["windSpeed"]
    dr = grid["properties"]["windDirection"]

    speed_uom = sp.get("uom")
    dir_uom = dr.get("uom")

    speed_values = sp.get("values", [])
    dir_values = dr.get("values", [])

    # 3) Expand each series to hourly dicts keyed by UTC hour
    def expand_series(values, is_speed: bool):
        out = {}
        uom = speed_uom if is_speed else dir_uom
        for item in values:
            vt = item.get("validTime")
            val = item.get("value")
            if vt is None:
                continue
            start, hours = parse_valid_time(vt)
            # Fill each hour in interval with the same value (stepwise-constant)
            for h in range(hours):
                t = (start + timedelta(hours=h)).replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
                if is_speed:
                    out[t] = to_mps(val, uom)  # store in m/s
                else:
                    out[t] = val  # degrees (direction units are usually degrees)
        return out

    speed_by_hour = expand_series(speed_values, is_speed=True)
    dir_by_hour = expand_series(dir_values, is_speed=False)

    # 4) Merge hours present in either series; sort
    all_hours = sorted(set(speed_by_hour.keys()) | set(dir_by_hour.keys()))

    # 5) Write CSV with multiple unit options
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time_utc",
            "wind_speed_mps",
            "wind_speed_knots",
            "wind_speed_mph",
            "wind_dir_deg",
            "speed_uom_source",
            "dir_uom_source"
        ])
        for t in all_hours:
            v_mps = speed_by_hour.get(t)
            v_kn = mps_to_knots(v_mps) if v_mps is not None else None
            v_mph = (v_mps * 2.23694) if v_mps is not None else None
            deg = dir_by_hour.get(t)
            w.writerow([
                t.isoformat().replace("+00:00", "Z"),
                f"{v_mps:.3f}" if v_mps is not None else "",
                f"{v_kn:.1f}" if v_kn is not None else "",
                f"{v_mph:.1f}" if v_mph is not None else "",
                f"{deg:.0f}" if deg is not None else "",
                speed_uom or "",
                dir_uom or "",
            ])

    print(f"✓ Wrote {out_path}")
    print(f"Speed units reported by API: {speed_uom}")
    print(f"Direction units reported by API: {dir_uom}")

def _degrees_to_cardinal(degrees: Optional[float]) -> str:
    """Convert degrees to cardinal direction."""
    if degrees is None:
        return "N/A"
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    ix = int((degrees + 11.25)/22.5 - 0.02)
    return dirs[ix % 16]

def get_wind_data(lat: float, lon: float, start_time: datetime = None, end_time: datetime = None) -> Dict:
    """
    Fetch wind data from NWS API for given coordinates and time range.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        start_time: Optional start time for filtering results (inclusive)
        end_time: Optional end time for filtering results (inclusive)
        
    Returns:
        Dictionary containing wind data and metadata
    """
    headers = {
        "User-Agent": "smoke-sim-dashboard/1.0",
        "Accept": "application/geo+json"
    }
    
    try:
        # 1) Get point metadata
        pt_url = f"https://api.weather.gov/points/{lat},{lon}"
        pt_response = requests.get(pt_url, headers=headers, timeout=30)
        pt_response.raise_for_status()
        pt_data = pt_response.json()
        
        # 2) Get grid data with wind information
        grid_url = pt_data["properties"]["forecastGridData"]
        grid_response = requests.get(grid_url, headers=headers, timeout=30)
        grid_response.raise_for_status()
        grid_data = grid_response.json()
        
        # 3) Extract wind speed and direction
        wind_speed = grid_data["properties"].get("windSpeed", {})
        wind_dir = grid_data["properties"].get("windDirection", {})
        
        # 4) Process wind speed values
        speed_uom = wind_speed.get("uom", "")
        speed_values = wind_speed.get("values", [])
        
        # 5) Process wind direction values
        dir_uom = wind_dir.get("uom", "")
        dir_values = wind_dir.get("values", [])
        
        # 6) Combine and process the data
        wind_data = []
        for speed, direction in zip(speed_values, dir_values):
            # Extract the timestamp part before the /PT
            timestamp_str = speed["validTime"].split('/')[0] if '/' in speed["validTime"] else speed["validTime"]
            timestamp = dtp.isoparse(timestamp_str).replace(tzinfo=timezone.utc)
            
            # Skip if outside time range
            if start_time and timestamp < start_time.replace(tzinfo=timezone.utc):
                continue
            if end_time and timestamp > end_time.replace(tzinfo=timezone.utc):
                continue
                
            speed_mps = to_mps(speed["value"], speed_uom)
            wind_data.append({
                "timestamp": timestamp_str,
                "speed_mps": speed_mps,
                "speed_knots": mps_to_knots(speed_mps) if speed_mps is not None else None,
                "direction_deg": direction["value"],
                "direction_cardinal": _degrees_to_cardinal(direction["value"]) if direction["value"] is not None else None
            })
        
        return {
            "status": "success",
            "data": wind_data,
            "units": {
                "speed": "m/s",
                "direction": "degrees"
            },
            "location": {
                "lat": lat,
                "lon": lon
            },
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "message": f"Failed to fetch wind data: {str(e)}"
        }

def get_wind_summary(wind_data: Dict) -> Dict:
    """Extract summary information from wind data."""
    if wind_data.get('status') != 'success' or not wind_data.get('data'):
        return {}
    
    latest = wind_data['data'][0]  # Most recent forecast
    
    return {
        'speed_mps': latest.get('speed_mps'),
        'speed_knots': latest.get('speed_knots'),
        'direction_deg': latest.get('direction_deg'),
        'direction_cardinal': latest.get('direction_cardinal'),
        'location': wind_data.get('location', {}),
        'retrieved_at': wind_data.get('retrieved_at')
    }

def create_wind_dataframe(wind_data: Dict) -> pd.DataFrame:
    """Convert wind data to a pandas DataFrame."""
    if wind_data.get('status') != 'success' or not wind_data.get('data'):
        return pd.DataFrame()
    
    df = pd.DataFrame(wind_data['data'])
    if 'timestamp' in df.columns:
        # Convert to datetime and format
        df['time'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        df = df[['time', 'speed_mps', 'speed_knots', 'direction_deg', 'direction_cardinal']]
    return df

if __name__ == "__main__":
    main()
