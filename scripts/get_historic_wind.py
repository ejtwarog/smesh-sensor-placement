#!/usr/bin/env python3
import sys, math, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs
from scipy.stats import vonmises

# Optional dependencies
try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    GaussianMixture = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

UA = "evan-observed-wind/1.0 (evantwarog@example.com)"
HDRS = {"User-Agent": UA, "Accept": "application/geo+json"}

# --- helpers ---------------------------------------------------------------

def utcnow():
    return datetime.now(timezone.utc)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def to_mps(value, unit):
    if value is None:
        return None
    u = (unit or "").lower()
    if "m_s-1" in u:                 # m/s
        return value
    if "km_h-1" in u or "km/h" in u: # km/h
        return value / 3.6
    if "kn" in u or "kt" in u:       # knots
        return value * 0.514444
    if "mi_h-1" in u or "mph" in u or "m_h-1" in u: # mph
        return value * 0.44704
    return value  # fallback

def circ_mean_var_deg(deg, weights=None):
    """Circular mean/variance for angles in degrees.
       Returns (mean_dir_deg, circular_variance, circular_std, R)."""
    th = np.deg2rad(np.asarray(deg, dtype=float))
    if weights is None:
        weights = np.ones_like(th)
    else:
        weights = np.asarray(weights, dtype=float)
    C = np.sum(weights * np.cos(th))
    S = np.sum(weights * np.sin(th))
    W = np.sum(weights)
    if W == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean_dir = (np.degrees(np.arctan2(S, C)) % 360)
    R = np.hypot(C, S) / W
    circ_var = 1.0 - R
    circ_std = np.sqrt(-2.0 * np.log(R)) if R > 0 else np.inf
    return mean_dir, circ_var, circ_std, R

# --- API wrappers ----------------------------------------------------------

def get_points(lat, lon):
    r = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=HDRS, timeout=30)
    r.raise_for_status()
    return r.json()

def get_stations_by_distance(points_json, lat, lon, limit=12):
    url = points_json["properties"]["observationStations"]
    r = requests.get(url, headers=HDRS, timeout=30)
    r.raise_for_status()
    out = []
    for feat in r.json().get("features", []):
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates")  # [lon, lat]
        if not coords or len(coords) < 2:
            continue
        st_lon, st_lat = coords[0], coords[1]
        out.append((
            haversine_km(lat, lon, st_lat, st_lon),
            props.get("stationIdentifier"),
            props.get("name") or props.get("stationIdentifier"),
        ))
    out.sort(key=lambda x: x[0])
    return out[:limit]  # closest first

def is_asos_station(station_id: str) -> bool:
    """Return True if the station is ASOS (based on station metadata)."""
    try:
        r = requests.get(f"https://api.weather.gov/stations/{station_id}", headers=HDRS, timeout=20)
        r.raise_for_status()
        props = (r.json().get("properties") or {})
        stype = (props.get("stationType") or "").upper()
        if stype == "ASOS":
            return True
        name = (props.get("name") or "").upper()
        return "ASOS" in name
    except requests.RequestException:
        return False

def fetch_station_observations(station_id, start, end):
    """Pull observations with pagination; returns list of GeoJSON features."""
    base = f"https://api.weather.gov/stations/{station_id}/observations"
    params = {
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "limit": 500,
    }
    out, cursor = [], None
    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        resp = requests.get(base, headers=HDRS, params=p, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        out.extend(j.get("features", []))
        next_url = None
        if isinstance(j.get("pagination"), dict):
            next_url = j["pagination"].get("next")
        if not next_url and isinstance(j.get("links"), dict):
            next_url = j["links"].get("next")
        if not next_url:
            break
        qs = parse_qs(urlparse(next_url).query)
        next_cursor = qs.get("cursor", [None])[0]
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    return out

def obs_features_to_df(features, station_id):
    rows = []
    for f in features:
        props = f.get("properties", {}) or {}
        ts = props.get("timestamp")
        if not ts:
            continue
        ws = (props.get("windSpeed") or {})
        wg = (props.get("windGust") or {})
        wd = (props.get("windDirection") or {})
        rows.append({
            "time_utc": pd.to_datetime(ts, utc=True),
            "station": station_id,
            "wind_speed_mps": to_mps(ws.get("value"), ws.get("unitCode")),
            "wind_gust_mps": to_mps(wg.get("value"), wg.get("unitCode")) if wg.get("value") is not None else None,
            "wind_dir_deg": wd.get("value"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["time_utc"]).sort_values("time_utc")
    return df

# --- main ------------------------------------------------------------------

def plot_wind_rose(df: pd.DataFrame, title: str, out_png: str, overlay_vonmises: bool = True):
    """Create a wind rose (directional frequency) with optional von Mises overlay and save to PNG.
    Uses 16 sectors (22.5°). If matplotlib is unavailable, prints a notice.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'wind_speed_mps' and 'wind_dir_deg' columns
    title : str
        Title for the plot
    out_png : str
        Output PNG file path
    overlay_vonmises : bool
        If True, overlay the fitted von Mises distribution (default: True)
    """
    if plt is None:
        print(f"[plot] matplotlib not available; skipping wind rose. Install matplotlib to enable.", file=sys.stderr)
        return

    # Ensure numeric and valid values
    spd = pd.to_numeric(df.get("wind_speed_mps"), errors="coerce")
    direc = pd.to_numeric(df.get("wind_dir_deg"), errors="coerce")
    m = spd.notna() & direc.notna()
    spd = spd[m]
    direc = (direc[m] % 360).values

    if len(spd) == 0:
        print("[plot] no valid wind speed/direction pairs to plot", file=sys.stderr)
        return

    # Direction sectors (degrees)
    sector = 22.5
    bins_deg = np.arange(0, 360 + sector, sector)
    counts, _ = np.histogram(direc, bins=bins_deg)
    total = counts.sum()
    if total == 0:
        print("[plot] all observations filtered out; nothing to plot", file=sys.stderr)
        return

    theta = np.deg2rad((bins_deg[:-1] + bins_deg[1:]) / 2.0)
    width = np.deg2rad(sector)
    frac = counts / total

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # Plot histogram bars
    ax.bar(theta, frac, width=width, bottom=0.0, align='center', color='#1f77b4', 
           edgecolor='white', alpha=0.7, label='Observed')
    
    # Overlay von Mises distribution if requested
    if overlay_vonmises:
        try:
            # Fit von Mises to the data
            vm_result = fit_vonmises_regression(df, speed_col='wind_speed_mps', dir_col='wind_dir_deg', n_components=1)
            mu = vm_result['mu']
            kappa = vm_result['kappa']
            
            # Generate smooth curve for von Mises
            theta_smooth = np.linspace(0, 2*np.pi, 360)
            vm_model = vonmises(kappa, loc=mu)
            pdf_values = vm_model.pdf(theta_smooth)
            
            # Normalize PDF to match histogram scale
            pdf_normalized = pdf_values / (2 * np.pi) * len(direc)
            
            # Plot von Mises curve
            ax.plot(theta_smooth, pdf_normalized, 'r-', linewidth=2.5, label=f'von Mises (κ={kappa:.2f})')
            
        except Exception as e:
            print(f"[plot] warning: could not fit von Mises distribution: {e}", file=sys.stderr)
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(title, va='bottom', pad=20)
    ax.set_rlabel_position(225)
    ax.set_ylim(0, max(frac) * 1.15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    try:
        plt.savefig(out_png, dpi=200, bbox_inches='tight')
        print(f"[plot] saved wind rose with von Mises overlay -> {out_png}", file=sys.stderr)
    finally:
        plt.close(fig)

def main():
    if len(sys.argv) < 3:
        print("Usage: get_historic_wind.py <lat> <lon> [out.csv]", file=sys.stderr)
        sys.exit(1)

    lat, lon = float(sys.argv[1]), float(sys.argv[2])
    out_csv = sys.argv[3] if len(sys.argv) > 3 else None

    end = utcnow()
    start = end - timedelta(days=7)

    points = get_points(lat, lon)
    candidates = get_stations_by_distance(points, lat, lon, limit=12)

    # Keep only ASOS stations; if none, fall back to ICAO-like codes (4 letters)
    asos_candidates = [(d, sid, nm) for (d, sid, nm) in candidates if is_asos_station(sid)]
    if asos_candidates:
        candidates = asos_candidates
    else:
        icao_like = [(d, sid, nm) for (d, sid, nm) in candidates if len(str(sid)) == 4 and str(sid).isalpha()]
        if icao_like:
            candidates = icao_like

    df = pd.DataFrame()
    for _, sid, _ in candidates:
        feats = fetch_station_observations(sid, start, end)
        cur = obs_features_to_df(feats, sid)
        if cur.empty:
            continue
        # Keep station only if it has any non-zero wind
        if cur["wind_speed_mps"].fillna(0).gt(0).any():
            df = cur
            break

    # If still empty, fall back to nearest candidate even if zero
    if df.empty and candidates:
        _, sid, _ = candidates[0]
        feats = fetch_station_observations(sid, start, end)
        df = obs_features_to_df(feats, sid)

    # cast, filter, then copy to avoid SettingWithCopyWarning
    df["wind_speed_mps"] = pd.to_numeric(df["wind_speed_mps"], errors="coerce")
    df = df.loc[df["wind_speed_mps"] > 0].copy()

    # --- stats ---------------------------------------------------------------
    # Linear stats for wind speed (sample variance by default)
    if not df.empty:
        speed_mean = float(df["wind_speed_mps"].mean())
        speed_var  = float(df["wind_speed_mps"].var(ddof=1)) if len(df) > 1 else np.nan
        # Circular stats for direction (unweighted)
        dir_series = pd.to_numeric(df["wind_dir_deg"], errors="coerce").dropna()
        if not dir_series.empty:
            dir_mean_deg, dir_circ_var, dir_circ_std, R = circ_mean_var_deg(dir_series.values)
        else:
            dir_mean_deg = dir_circ_var = dir_circ_std = R = np.nan
        # Print to stderr (keeps CSV clean)
        print(f"[stats] speed_mean_mps={speed_mean:.3f}  speed_var_mps2={np.nan if np.isnan(speed_var) else round(speed_var,3)}", file=sys.stderr)
        print(f"[stats] dir_mean_deg={np.nan if np.isnan(dir_mean_deg) else round(dir_mean_deg,1)}  "
              f"circ_var={np.nan if np.isnan(dir_circ_var) else round(dir_circ_var,3)}  "
              f"circ_std={'inf' if np.isinf(dir_circ_std) else (np.nan if np.isnan(dir_circ_std) else round(dir_circ_std,3))}  "
              f"R={np.nan if np.isnan(R) else round(R,3)}", file=sys.stderr)
    else:
        print("[stats] no rows after filtering wind > 0 m/s", file=sys.stderr)

    # format timestamps without chained assignment
    if not df.empty:
        df.loc[:, "time_utc"] = df["time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Raw CSV output (no resampling)
    if out_csv:
        df.to_csv(out_csv, index=False)
    else:
        print(df.to_csv(index=False, header=True), end="")

    # Wind rose plot
    if not df.empty:
        station = str(df["station"].iloc[0]) if "station" in df.columns and len(df) else "unknown"
        title = f"Wind Rose — {station} — {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} UTC"
        if out_csv:
            # save alongside CSV with suffix
            if out_csv.lower().endswith('.csv'):
                out_png = out_csv[:-4] + "_wind_rose.png"
            else:
                out_png = out_csv + "_wind_rose.png"
        else:
            out_png = "wind_rose.png"
        plot_wind_rose(df, title, out_png)

def fit_vonmises_regression(df, speed_col='wind_speed_mps', dir_col='wind_dir_deg',
                            n_components=1, kappa_init=1.0):
    """
    Fit a von Mises or mixture of von Mises model to wind direction data
    conditional on wind speed.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing wind speed and wind direction columns.
    speed_col : str
        Name of the wind speed column (default: 'wind_speed_mps').
    dir_col : str
        Name of the wind direction column in degrees, clockwise from north
        (default: 'wind_dir_deg').
    n_components : int
        Number of von Mises mixture components to fit (default 1).
    kappa_init : float
        Initial concentration parameter for von Mises (used when n_components=1).

    Returns
    -------
    dict
        If n_components == 1:
            {'mu': mean_direction (radians), 'kappa': concentration, 'model': vonmises_frozen}
        If n_components > 1:
            {'weights': π_k, 'mu': μ_k (radians), 'kappa': κ_k}
    """
    # Remove NaN values
    df_clean = df[[speed_col, dir_col]].dropna()
    
    if len(df_clean) == 0:
        raise ValueError(f"No valid data in columns {speed_col}, {dir_col}")
    
    # Convert direction to radians
    theta = np.deg2rad(df_clean[dir_col].values)
    speed = df_clean[speed_col].values

    if n_components == 1:
        # Fit a single von Mises using circular mean direction
        mu = np.angle(np.sum(np.exp(1j * theta)))  # mean direction in radians
        R = np.abs(np.sum(np.exp(1j * theta))) / len(theta)  # mean resultant length
        kappa = max(kappa_init, (R * (2 - R**2)) / (1 - R**2))  # Mardia-Jupp approximation
        model = vonmises(kappa, loc=mu)
        return {'mu': mu, 'kappa': kappa, 'model': model, 'n_samples': len(df_clean)}

    else:
        # Fit a mixture model in 2D (cos, sin representation)
        if GaussianMixture is None:
            raise ImportError("scikit-learn is required for mixture models. Install with: pip install scikit-learn")
        
        X = np.column_stack((np.cos(theta), np.sin(theta)))
        gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=42)
        gmm.fit(X)
        mu = np.arctan2(gmm.means_[:, 1], gmm.means_[:, 0])
        kappa = 1 / (gmm.covariances_ + 1e-6)  # approximate conversion, avoid division by zero
        return {'weights': gmm.weights_, 'mu': mu, 'kappa': kappa, 'n_samples': len(df_clean)}


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 10)
    main()
