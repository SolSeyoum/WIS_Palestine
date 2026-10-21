"""
WIS_dashboard.py
================
Climate-Water Balance and Hydrological Stress Dashboard.

Work at River Basin or Administrative Unit level. Data is read from a fixed
folder layout, indicators (CWB, AI, ESI, SPEI, TWSA) are computed in
`water_indicators.py`, and shown as status cards, trend charts, a choropleth
map, anomaly heatmaps and exports.

Expected data folder layout
---------------------------
    data/river_basins/boundary/*.shp   (or .geojson / .json / .zip)
    data/river_basins/monthly/<unit name>.csv      columns: time;PCP;AETI;RET;TWS
    data/river_basins/annual.csv                   columns: year;River_basins;PCP;AETI;RET;TWS;area_sq_km

    data/admin_units/boundary/*.shp
    data/admin_units/monthly/<unit name>.csv
    data/admin_units/annual.csv

Boundary unit-name column: `BASIN_NAME` (River Basins), `Admin_level1`
(Administrative Units).

Run:
    streamlit run WIS_dashboard.py
"""

import io
import math
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import geopandas as gpd

    _HAS_GEO = True
except ImportError:  # pragma: no cover
    _HAS_GEO = False

from water_indicators import (
    METHOD_PACKAGE,
    IndicatorConfig,
    compute_all_indicators,
    latest_snapshot,
    period_average_snapshot,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

REGION_NAME = "Palestine"
DATA_SOURCE = "WaPOR data"

st.set_page_config(
    page_title="Climate-Water Balance & Hydrological Stress Dashboard",
    page_icon="💧",
    layout="wide",
)


def _version_tuple(v: str):
    out = []
    for part in v.split(".")[:2]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


# Streamlit replaced `use_container_width=True` with `width="stretch"`.
STRETCH = (
    {"width": "stretch"}
    if _version_tuple(st.__version__) >= (1, 50)
    else {"use_container_width": True}
)

STATUS_COLORS = {
    "Extremely Wet": "#08519c",
    "Very Wet": "#3182bd",
    "Moderately Wet": "#6baed6",
    "Near Normal": "#31a354",
    "Moderately Dry": "#fec44f",
    "Severely Dry": "#e6550d",
    "Extremely Dry": "#a50f15",
    "No Data": "#bdbdbd",
}

ARIDITY_COLORS = {
    "Humid": "#08519c",
    "Dry Sub-Humid": "#3182bd",
    "Semi-Arid": "#fec44f",
    "Arid": "#e6550d",
    "Hyper-Arid": "#a50f15",
    "No Data": "#bdbdbd",
}

# Standardized indices are shown on a fixed -3..3 range for comparability.
STD_INDEX_RANGE = (-3, 3)

# BrBG: brown = dry / deficit, teal-blue = wet / surplus.
WATER_COLORSCALE = "BrBG"

BOUNDARY_ID_COLUMN = {
    "River Basins": "BASIN_NAME",
    "Administrative Units": "Admin_level1",
}

# Wet/dry bands for SPEI and TWSA (positive = wetter).
SPEI_ZONES = [
    (-3, -2, "#B2182B"),
    (-2, -1.5, "#EF8A62"),
    (-1.5, -1, "#FDCC8A"),
    (-1, 1, "#E0E0E0"),
    (1, 1.5, "#B3CDE3"),
    (1.5, 2, "#67A9CF"),
    (2, 3, "#2166AC"),
]
# ESI is the opposite: positive = more stress, so mirror the bands and the
# brown (dry/stressed) colours sit at the high end.
ESI_ZONES = [(-y1, -y0, color) for y0, y1, color in SPEI_ZONES]
BAND_OPACITY = 0.35


def status_badge(label: str) -> str:
    color = STATUS_COLORS.get(label, "#bdbdbd")
    return (
        f"<span style='background-color:{color};color:white;padding:4px 10px;"
        f"border-radius:12px;font-size:0.85rem;font-weight:600;'>{label}</span>"
    )


def add_bands(fig, zones):
    for y0, y1, color in zones:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=BAND_OPACITY, line_width=0)


# --------------------------------------------------------------------------
# Data loading (cached; keyed on file signatures so edits invalidate caches)
# --------------------------------------------------------------------------


def folder_signature(path: str):
    """Cheap fingerprint (name, size, mtime) of a file or folder's files."""
    if os.path.isfile(path):
        st_ = os.stat(path)
        return ((os.path.basename(path), st_.st_size, st_.st_mtime_ns),)
    if not os.path.isdir(path):
        return ()
    sig = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isfile(full):
            st_ = os.stat(full)
            sig.append((name, st_.st_size, st_.st_mtime_ns))
    return tuple(sig)


def find_boundary_file(folder: str):
    """First usable boundary file: .shp, then .geojson/.json, then .zip."""
    if not folder or not os.path.isdir(folder):
        return None
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return None
    for ext in (".shp", ".geojson", ".json", ".zip"):
        matches = [f for f in entries if f.lower().endswith(ext)]
        if matches:
            return os.path.join(folder, matches[0])
    return None


def load_boundary_from_path(path: str):
    """Load .shp / .geojson / .json / .zip with geopandas, in EPSG:4326."""
    if path.lower().endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmpdir)
            found = []
            for root, _dirs, files in os.walk(tmpdir):
                found += [os.path.join(root, f) for f in files]
            shp = [f for f in found if f.lower().endswith(".shp")]
            geo = [f for f in found if f.lower().endswith((".geojson", ".json"))]
            if shp:
                gdf = gpd.read_file(shp[0])
            elif geo:
                gdf = gpd.read_file(geo[0])
            else:
                raise ValueError(f"No .shp or .geojson file found inside {path}.")
    else:
        gdf = gpd.read_file(path)
    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


@st.cache_data(show_spinner="Reading boundary file…")
def load_boundary(path: str, signature, id_col: str):
    """Returns (slim_gdf[unit, geometry], geojson dict, bounds, error)."""
    try:
        gdf = load_boundary_from_path(path)
    except Exception as e:
        return None, None, None, f"Could not read boundary file `{path}`: {e}"
    if id_col not in gdf.columns:
        cols = ", ".join(gdf.columns.drop("geometry").tolist())
        return (
            None,
            None,
            None,
            f"Boundary file `{os.path.basename(path)}` has no `{id_col}` column (found: {cols}).",
        )
    slim = gdf[[id_col, "geometry"]].rename(columns={id_col: "unit"})
    slim["unit"] = slim["unit"].astype(str).str.strip()
    slim = slim.reset_index(drop=True)
    geojson = slim.__geo_interface__
    # plotly matches on properties.unit
    for feat in geojson["features"]:
        feat["properties"] = {"unit": feat["properties"]["unit"]}
    return slim, geojson, tuple(slim.total_bounds), None


def _read_csv_flex_path(path: str) -> pd.DataFrame:
    """Try ';'-delimited first, fall back to a standard comma CSV."""
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] == 1:
            raise ValueError("single column - wrong delimiter")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


_VALUE_ALIASES = {
    "pcp": "PCP",
    "ret": "RET",
    "tws": "TWS",
    "aeti": "ET",
    "et": "ET",
    "aet": "ET",
}


@st.cache_data(show_spinner="Reading monthly CSV files…")
def read_monthly_csvs(folder: str, signature) -> pd.DataFrame:
    """One CSV per unit (time;PCP;AETI;RET;TWS); unit = filename."""
    empty = pd.DataFrame(columns=["unit", "date", "PCP", "ET", "RET", "TWS"])
    if not folder or not os.path.isdir(folder):
        return empty
    frames = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".csv"):
            continue
        df = _read_csv_flex_path(os.path.join(folder, fname))
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "time":
                rename_map[c] = "date"
            elif cl in _VALUE_ALIASES:
                rename_map[c] = _VALUE_ALIASES[cl]
        df = df.rename(columns=rename_map)
        df["unit"] = os.path.splitext(fname)[0]
        frames.append(df)
    if not frames:
        return empty
    out = pd.concat(frames, ignore_index=True)
    out["unit"] = out["unit"].astype(str).str.strip()
    return out


@st.cache_data(show_spinner="Reading annual CSV…")
def read_annual_csv(path: str, signature) -> pd.DataFrame:
    df = _read_csv_flex_path(path)
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in (
            "river_basins",
            "riverbasin",
            "basin",
            "basin_name",
            "adm1_name",
            "admin_unit",
            "unit",
        ):
            rename_map[c] = "unit"
        elif cl == "year":
            rename_map[c] = "year"
        elif cl in _VALUE_ALIASES:
            rename_map[c] = _VALUE_ALIASES[cl]
        elif cl == "area_sq_km":
            rename_map[c] = "area_sq_km"
    return df.rename(columns=rename_map)


@st.cache_data(show_spinner="Computing indicators…")
def compute_indicators_cached(
    monthly_dir: str, signature, spei_scale: int
) -> pd.DataFrame:
    """Load + compute once per (data on disk, SPEI window). Widget changes
    that don't touch these inputs (date range, units, map metric…) reuse it."""
    monthly = read_monthly_csvs(monthly_dir, signature)
    out = compute_all_indicators(
        monthly, unit_col="unit", config=IndicatorConfig(spei_scale=spei_scale)
    )
    out["date"] = pd.to_datetime(out["date"])
    return out


def ensure_unit_column(df: pd.DataFrame, label: str, fallback_col: str) -> pd.DataFrame:
    if "unit" in df.columns or df.empty:
        return df
    if fallback_col in df.columns:
        return df.rename(columns={fallback_col: "unit"})
    st.sidebar.error(
        f"{label} has no `unit` or `{fallback_col}` column (found: "
        f"{', '.join(df.columns.tolist())})."
    )
    return df


def zoom_from_bounds(bounds, map_width_px=1000, map_height_px=600, pad=0.05) -> float:
    """Tightest web-map zoom (Web-Mercator, 512 px tiles) that still fits the
    whole boundary extent in the map frame."""
    minx, miny, maxx, maxy = bounds

    def merc_y(lat):
        lat = max(min(lat, 85.0), -85.0)
        return math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))

    lon_span = max(maxx - minx, 1e-6)
    lat_span = max(merc_y(maxy) - merc_y(miny), 1e-6)
    lon_zoom = math.log2(360 * (map_width_px / 512) / lon_span)
    lat_zoom = math.log2(360 * (map_height_px / 512) / lat_span)
    return float(min(max(min(lon_zoom, lat_zoom) - pad, 0.5), 18))


MAP_ENGINES = {
    # "Tile map, no basemap (WebGL)": "white-bg",
    "Light basemap": "carto-positron",
    "OpenStreetMap": "open-street-map",
    "Flat": "flat",
}


def _polygon_rings(geometry):
    """Yield exterior/interior rings of a GeoJSON Polygon/MultiPolygon as (lons, lats)."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        polys = [coords]
    elif gtype == "MultiPolygon":
        polys = coords
    else:
        return
    for poly in polys:
        for ring in poly:
            yield [p[0] for p in ring], [p[1] for p in ring]


def flat_choropleth(
    df,
    geojson,
    color,
    title,
    color_discrete_map=None,
    color_continuous_scale=None,
    range_color=None,
    hover_data=None,
    **_ignored,
):
    """Choropleth drawn with plain 2-D filled shapes (go.Scatter, fill='toself').

    Uses no WebGL, map tiles or CDN data, so it renders with every Plotly /
    Streamlit combination (the tile-based map traces need a recent Plotly.js).
    """
    from plotly.colors import sample_colorscale

    rows = df.set_index("unit")
    categorical = color_discrete_map is not None
    if not categorical:
        lo, hi = range_color if range_color else (rows[color].min(), rows[color].max())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = -1.0, 1.0

    hover_cols = [k for k, v in (hover_data or {}).items() if v is not False]
    fig = go.Figure()
    shown = set()
    lat_all = []
    for feat in geojson["features"]:
        unit = feat["properties"]["unit"]
        value = rows[color].get(unit) if unit in rows.index else None
        missing = value is None or (not categorical and pd.isna(value))
        if categorical:
            label = "No Data" if missing else value
            fill = color_discrete_map.get(label, "#bdbdbd")
        else:
            label = None
            fill = (
                "#bdbdbd"
                if missing
                else sample_colorscale(
                    color_continuous_scale, [(float(value) - lo) / (hi - lo)]
                )[0]
            )
        lines = [f"<b>{unit}</b>"]
        if categorical:
            lines.append(f"{color}: {label}")
        else:
            lines.append(f"{color}: n/a" if missing else f"{color}: {float(value):.2f}")
        for col in hover_cols:
            if col in rows.columns and unit in rows.index:
                v = rows[col].get(unit)
                if pd.isna(v):
                    lines.append(f"{col}: n/a")
                elif isinstance(v, (int, float, np.floating, np.integer)):
                    lines.append(f"{col}: {v:.2f}")
                else:
                    lines.append(f"{col}: {v}")
        hover = "<br>".join(lines)

        xs, ys = [], []
        for lons, lats in _polygon_rings(feat["geometry"]):
            xs += lons + [None]
            ys += lats + [None]
            lat_all += lats
        first = categorical and label not in shown
        if categorical:
            shown.add(label)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=fill,
                line=dict(color="#555555", width=0.8),
                text=hover,
                hoverinfo="text",
                hoveron="fills",
                name=str(label) if categorical else unit,
                legendgroup=str(label) if categorical else None,
                showlegend=bool(first),
            )
        )

    if not categorical:
        # Invisible marker trace that only provides the colour bar.
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                showlegend=False,
                hoverinfo="skip",
                marker=dict(
                    colorscale=color_continuous_scale,
                    cmin=lo,
                    cmax=hi,
                    color=[lo],
                    showscale=True,
                    colorbar=dict(title=color),
                ),
            )
        )

    # Equirectangular correction so shapes are not stretched at this latitude.
    mid_lat = float(np.mean(lat_all)) if lat_all else 0.0
    fig.update_xaxes(visible=False)
    fig.update_yaxes(
        visible=False,
        scaleanchor="x",
        scaleratio=1 / max(math.cos(math.radians(mid_lat)), 0.1),
    )
    fig.update_layout(title=title, height=600, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def make_choropleth(df, geojson, center, zoom, basemap=None, **kwargs):
    """Choropleth of `df` over `geojson` using the engine chosen in MAP_ENGINES.

    'Flat polygons' works everywhere. The tile maps need a Plotly.js that knows
    the `choroplethmap` trace (recent Streamlit); on older ones they render as an
    empty chart, and online basemaps additionally need internet access.
    """
    engine = MAP_ENGINES.get(basemap, "flat")
    if engine == "flat":
        return flat_choropleth(df, geojson, **kwargs)

    common = dict(
        geojson=geojson,
        locations="unit",
        featureidkey="properties.unit",
        hover_name="unit",
        height=600,
        center=center,
        zoom=zoom,
        opacity=0.85,
        **kwargs,
    )
    if hasattr(px, "choropleth_map"):
        fig = px.choropleth_map(df, map_style=engine, **common)
    else:  # pragma: no cover
        fig = px.choropleth_mapbox(df, mapbox_style=engine, **common)
    fig.update_traces(marker_line_width=0.8, marker_line_color="#555555")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


# --------------------------------------------------------------------------
# Sidebar — spatial unit + data folder
# --------------------------------------------------------------------------

st.sidebar.title("💧 Data & Settings")

if not _HAS_GEO:
    st.sidebar.error(
        "`geopandas` / `shapely` are not installed. Run "
        "`pip install geopandas shapely` to enable boundary loading and the map view."
    )

spatial_unit = st.sidebar.radio(
    "Spatial unit", ["River Basins", "Administrative Units"]
)
unit_label = "basin" if spatial_unit == "River Basins" else "administrative unit"

data_root = os.path.join(
    "data", "river_basins" if spatial_unit == "River Basins" else "admin_units"
)
boundary_id_col = BOUNDARY_ID_COLUMN[spatial_unit]
boundary_dir = os.path.join(data_root, "boundary")
monthly_dir = os.path.join(data_root, "monthly")
annual_path = os.path.join(data_root, "annual.csv")

# ---- Boundary ----
gdf = geojson = bounds = None
if _HAS_GEO:
    boundary_path = find_boundary_file(boundary_dir)
    if boundary_path is not None:
        gdf, geojson, bounds, b_err = load_boundary(
            boundary_path, folder_signature(boundary_dir), boundary_id_col
        )
        if b_err:
            st.sidebar.error(b_err)
    else:
        st.sidebar.info(f"No boundary file found in `{boundary_dir}`.")

# ---- Monthly / annual ----
monthly_sig = folder_signature(monthly_dir)
monthly_df = read_monthly_csvs(monthly_dir, monthly_sig)

annual_df = None
if os.path.isfile(annual_path):
    annual_df = read_annual_csv(annual_path, folder_signature(annual_path))
    annual_df = ensure_unit_column(
        annual_df, "annual.csv", fallback_col=boundary_id_col
    )
    if "unit" in annual_df.columns:
        annual_df = annual_df.copy()
        annual_df["unit"] = annual_df["unit"].astype(str).str.strip()

if monthly_df.empty:
    st.title("💧 Climate-Water Balance & Hydrological Stress Dashboard")
    st.info(
        f"No monthly data found in `{monthly_dir}`. Select **{spatial_unit}** and place "
        "data on disk following the folder layout in the README, then reload."
    )
    with st.expander("Column reference"):
        st.markdown(
            f"""
**Monthly CSV** — one file per unit in `{monthly_dir}/`, filename = unit name

| Column | Meaning | Units |
|---|---|---|
| `time` | Monthly date | YYYY-MM-DD |
| `PCP` | Precipitation | mm |
| `AETI` | Actual evapotranspiration and interception | mm |
| `RET` | Reference evapotranspiration | mm |
| `TWS` | Terrestrial water storage | mm equivalent |

**Annual CSV** — `{annual_path}`: `year`, `River_basins` (unit name), `PCP`, `AETI`, `RET`, `TWS`, `area_sq_km`

**Boundary file** — auto-detected in `{boundary_dir}/` (.shp, .geojson/.json or .zip).
The unit-name column must be **`{boundary_id_col}`** for {spatial_unit}.
            """
        )
    st.stop()

# Reserve sidebar slots so the visual order stays fixed.
st.sidebar.markdown("---")
ph_date_range = st.sidebar.container()
ph_status_mode = st.sidebar.container()
ph_indicator_settings = st.sidebar.container()
ph_units = st.sidebar.container()

# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

spei_scale = ph_indicator_settings.slider("SPEI accumulation window (months)", 1, 12, 3)

try:
    indicators = compute_indicators_cached(monthly_dir, monthly_sig, spei_scale)
except ValueError as e:
    st.error(str(e))
    st.stop()

spei_methods = set(indicators["SPEI_method"].dropna().unique())
spei_fallback_used = bool(spei_methods - {METHOD_PACKAGE})

# Unit filter: default to a readable subset for trend charts. The map always
# shows every unit.
all_units = sorted(indicators["unit"].dropna().unique().tolist())
default_units = all_units[: min(5, len(all_units))]
if len(all_units) > 15:
    ph_units.caption(
        f"{len(all_units)} {unit_label}s available — defaulting to {len(default_units)} "
        "for the trend charts to keep them readable. Add more as needed."
    )
selected_units = ph_units.multiselect(
    f"{spatial_unit} (Trends & Anomaly tabs)", all_units, default=default_units
)
if not selected_units:
    ph_units.warning(f"Select at least one {unit_label} to see trend charts.")

min_date, max_date = indicators["date"].min(), indicators["date"].max()
date_range = ph_date_range.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
indicators_all = indicators
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    indicators_all = indicators_all[
        (indicators_all["date"] >= pd.Timestamp(start_d))
        & (indicators_all["date"] <= pd.Timestamp(end_d))
    ]

# indicators_all -> every unit, date-filtered (map + alerts)
# indicators     -> selected units, date-filtered (Trends / Anomaly)
indicators = indicators_all[indicators_all["unit"].isin(selected_units)]

if indicators_all.empty:
    st.warning("No data in the selected date range.")
    st.stop()

status_mode = ph_status_mode.radio(
    "Status basis (cards / map / comparisons)",
    ["Latest month in period", "Period average", "Select a month"],
    help="What 'status' means for the summary cards, the Spatial Map and the "
    "cross-unit comparison chart: the most recent month in the selected date "
    "range, an average over that range, or one specific month you pick.",
)

selected_month_ts = None
if status_mode == "Select a month":
    month_choices = sorted(indicators_all["date"].dropna().unique(), reverse=True)
    month_label_map = {pd.Timestamp(d).strftime("%B %Y"): d for d in month_choices}
    selected_month_label = ph_status_mode.selectbox(
        "Month to display", list(month_label_map.keys()), key="status_month_select"
    )
    selected_month_ts = month_label_map[selected_month_label]

# --------------------------------------------------------------------------
# Header + status cards
# --------------------------------------------------------------------------

st.markdown(
    f"""
    <p style="font-size: 2.46rem; font-weight: 700; margin-bottom: 0.2rem;">
        {REGION_NAME} Climate-Water Balance & Hydrological Stress Dashboard
    </p>
    <p style="font-size: 1.6rem; font-weight: 500; color: #08519c; margin-top: 0;">
        Using {DATA_SOURCE}
    </p>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Spatial unit: **{spatial_unit}**. Indicators: Standardized Precipitation-"
    "Evapotranspiration Index (SPEI), Climatic Water Balance (CWB), Evaporative "
    "Stress Index (ESI), Terrestrial Water Storage Anomaly (TWSA) and Aridity Index (AI)."
)
if spei_fallback_used:
    st.warning(
        "SPEI could not be fitted with the log-logistic `spei` package for some or all "
        "units; a simpler per-calendar-month normal standardization was used instead "
        f"({', '.join(sorted(spei_methods - {METHOD_PACKAGE}))}). "
        "Install/upgrade the `spei` package or check for very short or constant records."
    )

period_start, period_end = indicators_all["date"].min(), indicators_all["date"].max()
if status_mode == "Period average":
    snapshot_all = period_average_snapshot(indicators_all, location_col="unit")
    if period_start.strftime("%b %Y") == period_end.strftime("%b %Y"):
        status_period_label = f"average, {period_start.strftime('%B %Y')}"
    else:
        status_period_label = f"average, {period_start.strftime('%b %Y')} – {period_end.strftime('%b %Y')}"
elif status_mode == "Select a month":
    snapshot_all = indicators_all[indicators_all["date"] == selected_month_ts].copy()
    status_period_label = pd.Timestamp(selected_month_ts).strftime("%B %Y")
else:
    snapshot_all = latest_snapshot(indicators_all, location_col="unit")
    status_period_label = snapshot_all["date"].max().strftime("%B %Y")
st.subheader(f"Status — {status_period_label}")

if snapshot_all.empty:
    st.warning("No data available for the chosen status basis.")
    st.stop()

snapshot = (
    snapshot_all[snapshot_all["unit"].isin(selected_units)]
    if selected_units
    else snapshot_all.iloc[0:0]
)
if snapshot.empty:
    st.caption(
        "No units selected — pick some in the sidebar to see status cards, or check the map for all units."
    )
else:
    n_cols = min(len(snapshot), 6)
    cols = st.columns(max(n_cols, 1))
    for i, (_, row) in enumerate(snapshot.sort_values("unit").iterrows()):
        with cols[i % n_cols]:
            st.markdown(f"**{row['unit']}**")
            st.markdown(status_badge(row["SPEI_status"]), unsafe_allow_html=True)
            st.metric("SPEI", f"{row['SPEI']:.2f}" if pd.notna(row["SPEI"]) else "n/a")
            cwb_txt = f"{row['CWB']:.0f} mm" if pd.notna(row["CWB"]) else "n/a"
            st.caption(f"CWB {cwb_txt} · Aridity: {row['Aridity_class']}")

alert_rows = snapshot_all[
    snapshot_all["SPEI_status"].isin(["Severely Dry", "Extremely Dry"])
]
if not alert_rows.empty:
    names = alert_rows["unit"].tolist()
    st.error(
        f"⚠️ Water stress alert: **{', '.join(names)}** for the selected period classified as "
        f"{'/'.join(sorted(alert_rows['SPEI_status'].unique()))}."
    )

st.markdown("---")

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tab_map, tab_trends, tab_anomaly, tab_data = st.tabs(
    ["🗺️ Spatial Map", "📈 Trends", "🔥 Anomaly Analysis", "📋 Data & Exports"]
)

# ---- Spatial Map ----
with tab_map:
    if not _HAS_GEO:
        st.info("Install `geopandas` and `shapely` to enable the spatial map view.")
    elif gdf is None or gdf.empty:
        st.info(
            f"No {unit_label} boundary available — place a .shp/.geojson/.zip in "
            f"`{boundary_dir}` (unit-name column `{boundary_id_col}`) to enable the map view."
        )
    else:
        map_cols = ["unit", "CWB", "SPEI", "SPEI_status", "AI", "Aridity_class"]
        map_cols = [c for c in map_cols if c in snapshot_all.columns]
        # Left-join onto all polygons so units without data still render (grey).
        map_df = pd.DataFrame({"unit": gdf["unit"].unique()}).merge(
            snapshot_all[map_cols], on="unit", how="left"
        )
        for col in ("SPEI_status", "Aridity_class"):
            if col in map_df.columns:
                map_df[col] = map_df[col].fillna("No Data")

        map_metric = st.selectbox(
            "Map metric", ["CWB (continuous)", "SPEI (standardized)", "Aridity class"]
        )
        center = {
            "lat": (bounds[1] + bounds[3]) / 2,
            "lon": (bounds[0] + bounds[2]) / 2,
        }
        zoom = zoom_from_bounds(bounds)
        basemap = st.radio(
            "Map style",
            list(MAP_ENGINES.keys()),
            index=list(MAP_ENGINES.values()).index("carto-positron"),
            horizontal=True,
            help="' Both basemaps "
            "need a recent Streamlit (Plotly.js >= 2.35) "
            "internet access; otherwise they show an empty chart. "
            "'Flat' works with every Streamlit/Plotly version.",
        )

        if map_metric == "Aridity class":
            fig_map = make_choropleth(
                map_df,
                geojson,
                center,
                zoom,
                basemap=basemap,
                color="Aridity_class",
                color_discrete_map=ARIDITY_COLORS,
                hover_data={"AI": ":.2f", "unit": False},
                title=f"Aridity classification by {unit_label} — {status_period_label}",
            )
        elif map_metric == "SPEI (standardized)":
            fig_map = make_choropleth(
                map_df,
                geojson,
                center,
                zoom,
                basemap=basemap,
                color="SPEI",
                color_continuous_scale=WATER_COLORSCALE,
                range_color=STD_INDEX_RANGE,
                hover_data={"SPEI": ":.2f", "SPEI_status": True, "unit": False},
                title=f"SPEI by {unit_label} — {status_period_label}",
            )
        else:
            cwb_bound = (
                float(np.nanmax(np.abs(map_df["CWB"])))
                if map_df["CWB"].notna().any()
                else 1.0
            )
            fig_map = make_choropleth(
                map_df,
                geojson,
                center,
                zoom,
                basemap=basemap,
                color="CWB",
                color_continuous_scale=WATER_COLORSCALE,
                range_color=(-cwb_bound, cwb_bound),
                hover_data={"CWB": ":.0f", "Aridity_class": True, "unit": False},
                title=f"Climatic Water Balance (CWB, mm) by {unit_label} — {status_period_label}",
            )
        st.plotly_chart(fig_map, **STRETCH)

        no_geom = set(snapshot_all["unit"]) - set(gdf["unit"])
        if no_geom:
            st.warning(
                "These units from the CSV data have no matching polygon in the boundary file "
                f"(check name spelling): {', '.join(sorted(no_geom))}"
            )

# ---- Trends ----
with tab_trends:
    fig_spei = px.line(
        indicators,
        x="date",
        y="SPEI",
        color="unit",
        title=f"Standardized Precipitation-Evapotranspiration Index (SPEI-{spei_scale})",
    )
    fig_spei.add_hline(y=0, line_dash="dash", line_color="gray")
    add_bands(fig_spei, SPEI_ZONES)
    fig_spei.update_layout(yaxis=dict(range=list(STD_INDEX_RANGE)))
    st.plotly_chart(fig_spei, **STRETCH)

    fig_esi = px.line(
        indicators,
        x="date",
        y="ESI",
        color="unit",
        title="Evaporative Stress Index (higher = more stress)",
    )
    fig_esi.add_hline(y=0, line_dash="dash", line_color="gray")
    add_bands(fig_esi, ESI_ZONES)
    fig_esi.update_layout(yaxis=dict(range=list(STD_INDEX_RANGE)))
    st.plotly_chart(fig_esi, **STRETCH)

    fig_cwb = px.bar(
        indicators,
        x="date",
        y="CWB",
        color="unit",
        title="Climatic Water Balance (PCP − RET, mm)",
        barmode="group",
    )
    fig_cwb.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_cwb, **STRETCH)

    fig_twsa = px.line(
        indicators,
        x="date",
        y="TWSA",
        color="unit",
        title="Terrestrial Water Storage Anomaly (standardized)",
    )
    fig_twsa.add_hline(y=0, line_dash="dash", line_color="gray")
    add_bands(fig_twsa, SPEI_ZONES)
    fig_twsa.update_layout(yaxis=dict(range=list(STD_INDEX_RANGE)))
    st.plotly_chart(fig_twsa, **STRETCH)

    if annual_df is not None and not annual_df.empty and "unit" in annual_df.columns:
        st.markdown("#### Annual totals")
        adf = annual_df[annual_df["unit"].isin(selected_units)].copy()
        metric_options = [c for c in ["PCP", "ET", "RET", "TWS"] if c in adf.columns]
        if metric_options and not adf.empty:
            metric = st.selectbox("Annual metric", metric_options)
            fig_annual = px.bar(
                adf.sort_values("year"),
                x="year",
                y=metric,
                color="unit",
                barmode="group",
                title=f"Annual {metric} by {unit_label}",
            )
            st.plotly_chart(fig_annual, **STRETCH)

# ---- Anomaly Analysis ----
with tab_anomaly:
    if not selected_units:
        st.info(
            f"Select at least one {unit_label} in the sidebar to see anomaly analysis."
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            heat_metric = st.selectbox("Metric", ["SPEI", "TWSA", "ESI"], index=0)
        with c2:
            unit_for_heat = st.selectbox("Unit", selected_units)

        hdf = indicators[indicators["unit"] == unit_for_heat].copy()
        hdf["Year"] = hdf["date"].dt.year
        hdf["Month"] = hdf["date"].dt.strftime("%b")
        month_order = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        pivot = hdf.pivot_table(
            index="Year", columns="Month", values=heat_metric, aggfunc="mean"
        ).reindex(columns=month_order)

        # ESI: high = stress, so flip the scale so brown still means "bad".
        heat_scale = WATER_COLORSCALE + ("_r" if heat_metric == "ESI" else "")
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale=heat_scale,
                zmid=0,
                zmin=STD_INDEX_RANGE[0],
                zmax=STD_INDEX_RANGE[1],
                colorbar=dict(title=heat_metric),
            )
        )
        fig_heat.update_layout(
            title=f"{heat_metric} by month/year — {unit_for_heat}", height=450
        )
        st.plotly_chart(fig_heat, **STRETCH)
        if heat_metric == "ESI":
            st.caption(
                "Brown = higher evaporative stress, teal/blue = lower stress "
                "(relative to that calendar month's own history)."
            )
        else:
            st.caption(
                "Teal/blue = wetter / higher availability, brown = drier / lower availability "
                "(relative to that calendar month's own history)."
            )

        st.markdown(f"#### Cross-unit comparison — {status_period_label} (all units)")
        comp_metric = st.selectbox(
            "Comparison metric", ["SPEI", "TWSA", "ESI", "CWB"], key="comp_metric"
        )
        fig_comp = px.bar(
            snapshot_all.sort_values(comp_metric),
            x="unit",
            y=comp_metric,
            color="SPEI_status",
            color_discrete_map=STATUS_COLORS,
            title=f"{comp_metric} by {unit_label} — {status_period_label}",
        )
        fig_comp.update_layout(xaxis_title=None)
        if comp_metric in ("SPEI", "TWSA", "ESI"):
            fig_comp.update_layout(yaxis=dict(range=list(STD_INDEX_RANGE)))
        st.plotly_chart(fig_comp, **STRETCH)

# ---- Data & Exports ----
with tab_data:
    export_scope = st.radio(
        "Scope",
        [
            f"All {len(all_units)} {unit_label}s",
            f"Selected {len(selected_units)} (Trends tab)",
        ],
        horizontal=True,
    )
    data_scope_df = indicators_all if export_scope.startswith("All") else indicators

    st.markdown("**Computed monthly indicators**")
    display_cols = [
        "date",
        "unit",
        "PCP",
        "ET",
        "RET",
        "TWS",
        "CWB",
        "AI",
        "ESI",
        "SPEI",
        "TWSA",
        "Aridity_class",
    ]
    display_cols = [c for c in display_cols if c in data_scope_df.columns]
    st.dataframe(
        data_scope_df[display_cols].sort_values(["unit", "date"]),
        height=400,
        **STRETCH,
    )

    csv_buf = io.StringIO()
    data_scope_df[display_cols].to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download monthly indicators as CSV",
        data=csv_buf.getvalue(),
        file_name="water_availability_indicators_monthly.csv",
        mime="text/csv",
    )

    if annual_df is not None and not annual_df.empty:
        st.markdown("**Annual data**")
        scope_units = all_units if export_scope.startswith("All") else selected_units
        adf = (
            annual_df[annual_df["unit"].isin(scope_units)]
            if "unit" in annual_df.columns
            else annual_df
        )
        st.dataframe(adf, height=300, **STRETCH)
        csv_buf2 = io.StringIO()
        adf.to_csv(csv_buf2, index=False)
        st.download_button(
            "⬇️ Download annual data as CSV",
            data=csv_buf2.getvalue(),
            file_name="water_availability_annual.csv",
            mime="text/csv",
        )

    if _HAS_GEO and gdf is not None and not gdf.empty:
        snapshot_export = snapshot_all.copy()
        if "date" in snapshot_export.columns:
            snapshot_export["date"] = snapshot_export["date"].astype(str)
        merged_export = gdf.merge(snapshot_export, on="unit", how="left")
        st.download_button(
            f"⬇️ Download {unit_label} boundaries + status ({status_period_label}, all units) as GeoJSON",
            data=merged_export.to_json().encode("utf-8"),
            file_name="units_with_status.geojson",
            mime="application/geo+json",
        )

    with st.expander("Method notes"):
        st.markdown(
            f"""
- **SPEI**: the climatic water balance (PCP − RET) is accumulated over the chosen window and
  standardized per calendar month with the `spei` package (log-logistic fit; Vicente-Serrano
  et al., 2010). If the package is unavailable or a fit fails, a per-calendar-month normal
  standardization is used instead; the dashboard warns when that happens and the method is
  recorded per unit in the `SPEI_method` field.
- **TWSA**: standardized anomaly of terrestrial water storage per calendar month, analogous to
  GRACE/GRACE-FO anomaly products. TWS is taken from GLDAS for the area of interest.
- **ESI**: standardized anomaly of the evaporative fraction (ET / RET), sign-flipped so
  positive values mean more evaporative stress. Its colour bands and heatmap scale are
  therefore mirrored relative to SPEI/TWSA (brown = stress).
- Standardized indices use the McKee et al. (1993) category thresholds and a fixed −3..3 range.
- **Aridity Index** = PCP / RET, classified with the UNEP scheme (Humid ≥ 0.65, Dry Sub-Humid
  ≥ 0.5, Semi-Arid ≥ 0.2, Arid ≥ 0.05, else Hyper-Arid). In *Period average* mode AI is
  mean PCP / mean RET. Note that in single-month modes the class reflects that month only,
  not the long-term climatology UNEP intends.
- Standardization is computed independently per unit and calendar month.
- Data is read from `{data_root}/` (boundary/, monthly/, annual.csv) and cached; changing
  files on disk invalidates the cache automatically.
- **Status basis** applies to the cards, alerts, the Spatial Map and the cross-unit
  comparison: latest month in the range, the period average, or a pinned month.
            """
        )
