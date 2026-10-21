"""
water_indicators.py
====================
Climate / water-balance / hydrological-stress indicators for the
Climate-Water Balance Dashboard.

Input: a long-format monthly time series, one row per (unit, date):

    unit  - basin or administrative-unit name (grouping key)
    date  - monthly timestamp
    PCP   - precipitation (mm)
    ET    - actual evapotranspiration (mm)          [AETI in source CSVs]
    RET   - reference / potential evapotranspiration (mm)
    TWS   - terrestrial water storage (consistent unit)

Indicators:
    CWB   - Climatic Water Balance = PCP - RET (mm)
    AI    - Aridity Index (UNEP) = PCP / RET
    ESI   - Evaporative Stress Index: standardized anomaly of -(ET/RET);
            positive = more evaporative stress
    SPEI  - Standardized Precipitation-Evapotranspiration Index. CWB is
            accumulated over `spei_scale` months and standardized with the
            `spei` package (log-logistic fit per calendar month; Vicente-
            Serrano et al., 2010). If the package is missing or a fit fails,
            a per-calendar-month normal standardization is used instead and
            the `SPEI_method` column records which method produced the values.
    TWSA  - standardized anomaly of TWS, per calendar month

    SPEI_status, Aridity_class - categorical labels for display.

This module is the single source of truth for classification thresholds
(aridity, SPEI) and for the period-average snapshot.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import spei as spei_pkg

    _HAS_SPEI_PKG = True
except ImportError:  # pragma: no cover
    _HAS_SPEI_PKG = False

REQUIRED_COLUMNS = ["date", "PCP", "ET", "RET", "TWS"]

METHOD_PACKAGE = "log-logistic (spei package)"
METHOD_FALLBACK = "normal fallback"
METHOD_MIXED = "mixed (partial normal fallback)"

# Ordered high -> low; first threshold the value meets or exceeds wins.
SPEI_THRESHOLDS = [
    (2.0, "Extremely Wet", False),  # SPEI > 2
    (1.5, "Very Wet", False),  # 1.5 < SPEI <= 2
    (1.0, "Moderately Wet", False),  # 1 < SPEI <= 1.5
    (-1.0, "Near Normal", True),  # -1 <= SPEI <= 1
    (-1.5, "Moderately Dry", True),  # -1.5 <= SPEI < -1
    (-2.0, "Severely Dry", True),  # -2 <= SPEI < -1.5
]  # anything below -2 -> "Extremely Dry"

# UNEP aridity classes (PCP / RET). Humid starts at 0.65.
ARIDITY_THRESHOLDS = [
    (0.65, "Humid"),
    (0.5, "Dry Sub-Humid"),
    (0.2, "Semi-Arid"),
    (0.05, "Arid"),
]  # below 0.05 -> "Hyper-Arid"


@dataclass
class IndicatorConfig:
    spei_scale: int = 3
    twsa_scale: int = 1


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def classify_spei(value) -> str:
    """Classify a SPEI value (McKee et al. 1993 style categories)."""
    if pd.isna(value):
        return "No Data"
    for threshold, label, inclusive in SPEI_THRESHOLDS:
        if value > threshold or (inclusive and value == threshold):
            return label
    return "Extremely Dry"


def classify_aridity(ai) -> str:
    """UNEP aridity class from the Aridity Index (PCP / RET)."""
    if pd.isna(ai):
        return "No Data"
    for threshold, label in ARIDITY_THRESHOLDS:
        if ai >= threshold:
            return label
    return "Hyper-Arid"


# --------------------------------------------------------------------------
# Standardization helpers
# --------------------------------------------------------------------------


def _rolling_accum(s: pd.Series, scale: int) -> pd.Series:
    if scale <= 1:
        return s
    return s.rolling(window=scale, min_periods=scale).sum()


def _normal_standardize_by_month(values: pd.Series, months: pd.Series) -> pd.Series:
    """Per-calendar-month z-score (mean/std of that month's own history)."""
    df = pd.DataFrame({"v": values.values, "m": months.values}, index=values.index)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for _, grp in df.groupby("m"):
        valid = grp["v"].dropna()
        if len(valid) < 3:
            continue
        mu, sigma = valid.mean(), valid.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            out.loc[grp.index] = 0.0
        else:
            out.loc[grp.index] = (grp["v"] - mu) / sigma
    return out


def _spei_via_package(cwb: pd.Series, dates: pd.Series, timescale: int):
    """SPEI from a CWB series using the `spei` package.

    Returns (values, method). Values the package cannot produce fall back to
    the normal standardization of the rolled CWB; the method label reports
    whether the fallback was used for none, some, or all values.
    """
    months = dates.dt.month
    fallback = _normal_standardize_by_month(_rolling_accum(cwb, timescale), months)

    if not _HAS_SPEI_PKG:
        return fallback, METHOD_FALLBACK

    series = pd.Series(cwb.values, index=pd.DatetimeIndex(dates.values))
    try:
        result = spei_pkg.spei(series, timescale=timescale)
    except Exception:
        return fallback, METHOD_FALLBACK

    # The package drops the first (timescale - 1) periods; reindex them back.
    result = result.reindex(series.index)
    out = pd.Series(result.values, index=cwb.index, dtype=float)
    bad = ~np.isfinite(out)
    replaced = bad & fallback.notna()
    out.loc[bad] = fallback.loc[bad]
    return out, (METHOD_MIXED if replaced.any() else METHOD_PACKAGE)


# --------------------------------------------------------------------------
# Indicator computation
# --------------------------------------------------------------------------


def _compute_unit(g: pd.DataFrame, config: IndicatorConfig) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    month = g["date"].dt.month

    g["CWB"] = g["PCP"] - g["RET"]
    g["AI"] = g["PCP"] / g["RET"].replace(0, np.nan)
    g["Aridity_class"] = g["AI"].apply(classify_aridity)

    evap_fraction = g["ET"] / g["RET"].replace(0, np.nan)
    g["ESI"] = -_normal_standardize_by_month(evap_fraction, month)

    g["SPEI"], method = _spei_via_package(g["CWB"], g["date"], config.spei_scale)
    g["SPEI_method"] = method
    g["SPEI_status"] = g["SPEI"].apply(classify_spei)

    tws_accum = _rolling_accum(g["TWS"], config.twsa_scale)
    g["TWSA"] = _normal_standardize_by_month(tws_accum, month)
    return g


def compute_all_indicators(
    df: pd.DataFrame, unit_col=None, config: IndicatorConfig = None
) -> pd.DataFrame:
    """Compute all indicators, grouped by `unit_col` if given (standardization
    is done independently per unit and calendar month)."""
    config = config or IndicatorConfig()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required column(s): {', '.join(missing)}"
        )

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])

    if unit_col and unit_col in work.columns:
        pieces = [_compute_unit(g, config) for _, g in work.groupby(unit_col)]
        return pd.concat(pieces, ignore_index=True)
    return _compute_unit(work, config)


# --------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------


def latest_snapshot(df: pd.DataFrame, location_col=None) -> pd.DataFrame:
    """Most recent row per unit (or overall, if no location_col)."""
    if location_col and location_col in df.columns:
        idx = df.groupby(location_col)["date"].idxmax()
        return df.loc[idx].reset_index(drop=True)
    latest_date = df["date"].max()
    return df[df["date"] == latest_date].reset_index(drop=True)


def period_average_snapshot(df: pd.DataFrame, location_col: str = "unit") -> pd.DataFrame:
    """Average indicators over the (already date-filtered) period per unit.

    AI is recomputed as mean PCP / mean RET (not the mean of monthly ratios),
    and the SPEI / aridity labels are re-derived from the averaged values.
    """
    cols = [
        c
        for c in ["PCP", "ET", "RET", "TWS", "CWB", "ESI", "SPEI", "TWSA"]
        if c in df.columns
    ]
    agg = df.groupby(location_col, as_index=False)[cols].mean(numeric_only=True)

    if {"PCP", "RET"} <= set(agg.columns):
        agg["AI"] = agg["PCP"] / agg["RET"].replace(0, np.nan)
    else:
        agg["AI"] = np.nan
    agg["Aridity_class"] = agg["AI"].apply(classify_aridity)
    agg["SPEI_status"] = (
        agg["SPEI"].apply(classify_spei) if "SPEI" in agg.columns else "No Data"
    )
    # Period end, kept so downstream display/export code can read a date.
    agg["date"] = df["date"].max() if "date" in df.columns and not df.empty else pd.NaT
    return agg
