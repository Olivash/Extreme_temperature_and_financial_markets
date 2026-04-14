#!/usr/bin/env python -u
"""
Six independent ΔEHD projection maps for 2040 using IFS reforecast slopes.

Combines IFS-derived risk-ratio slopes (RR per °C) with different ΔT sources
and two direct EHD projection methods to produce six estimates of ΔEHD in 2040.

Variants:
  1. ERA5-Land local trend     × IFS local (clima) slopes
  2. ERA5-Land 200km trend     × IFS 200km (inter) slopes
  3. GCM ECS-weighted local ΔT × IFS local (clima) slopes
  4. GCM ECS-weighted 200km ΔT × IFS 200km (inter) slopes
  5. Direct ERA5-Land EHD linear trend extrapolation
  6. Direct CIL GDPCIR EHD (GCM ECS-weighted ensemble)

Each is converted to GDP impact via β = −0.0555 (Approach 2), then aggregated
to global GDP-weighted and settlement-weighted means.

Outputs:
  projections/output/map_slopes_6variant_ehd_2040.png
  projections/output/slopes_6variant_comparison.csv

Usage:
  cd ~/projects/macro/extreme_heat/biodiversity_interactions
  /turbo/mgoldklang/pyenvs/peg_nov_24/bin/python projections/scripts/project_slopes_ehd_maps.py
"""

import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJ_DIR     = Path(__file__).resolve().parent.parent          # projections/
ROOT_DIR     = PROJ_DIR.parent                                  # biodiversity_interactions/
PARENT_DIR   = ROOT_DIR.parent                                  # extreme_heat/
OUT_DIR      = PROJ_DIR / "output"

# IFS slopes
SLOPES_DIR = Path("/home/mgoldklang/code/Extreme_temperature_and_financial_markets/data/slopes")
SLOPE_LOCAL  = SLOPES_DIR / "global_summer_clima_rr_per_degC_2001_2020_025deg.nc"
SLOPE_200KM  = SLOPES_DIR / "global_summer_inter_rr_per_degC_200km_era5clim_2001_2020_025deg.nc"

# ERA5
ERA5_TMAX_DIR = Path("/NImounts/NiClimateDev/data/historical/era5_land/summer/tmax")
EHD_DIR_ERA5  = ERA5_TMAX_DIR / "global"
ERA5_T2M_DIR  = Path("/NImounts/NiClimateDev/data/reanalysis/era5/2m_temperature")

# NEX-GDDP CMIP6
TASMAX_DIR = Path("/NImounts/NiNoSnapUsers/mgoldklang/climate/projections/tasmax")

# CIL GDPCIR EHD
CIL_EHD_DIR = Path("/NImounts/NiNoSnapUsers/mgoldklang/climate/projections/ehd")

# Approach 2 panel + regression
REGR_FILE  = PARENT_DIR / "output" / "panel_regression_gdp_heat_results_admin1level.csv"
PANEL_FILE = PARENT_DIR / "output" / "panel_mixed_admin1level.parquet"

# Mixed rasters
CACHE_DIR      = PARENT_DIR / "cache"
ADMIN2_RASTER  = CACHE_DIR / "admin2_raster_01deg.npy"
ADMIN2_LOOKUP  = CACHE_DIR / "admin2_lookup_01deg.csv"
ADMIN1_RASTER  = CACHE_DIR / "admin1_raster_01deg.nc"
ADMIN1_LOOKUP  = CACHE_DIR / "admin1_lookup.csv"
SETTLE_FILE    = Path("/home/mgoldklang/projects/climate_extremes/conus_equities/cache/settlement_01deg_2025.nc")

# ─── Config ───────────────────────────────────────────────────────────────────
SENS_YEARS     = list(range(2000, 2025))
CLIM_START     = 1990
CLIM_END       = 2020
CLIM_SSP       = "ssp245"
TARGET_YEAR    = 2040
PROJ_WINDOW    = (2035, 2044)  # 10-year window around 2040

# Gaussian σ for 200km smoothing at 0.25° (≈27.8 km/cell at equator)
SIGMA_025      = 200.0 / 27.8   # ≈ 7.2 cells

# ERA5-Land 0.1° grid
ERA5_LATS = np.round(np.arange(90.0, -90.1, -0.1), 1)
ERA5_LONS = np.round(np.arange(-180.0, 180.0, 0.1), 1)

# ─── ECS values (Zelinka et al. 2020) — used for GCM weighting ──────────────
ECS_VALUES = {
    "ACCESS-CM2":       4.66,   "ACCESS-ESM1-5":    3.88,
    "BCC-CSM2-MR":      3.02,   "CanESM5":          5.64,
    "CMCC-CM2-SR5":     3.55,   "CMCC-ESM2":        3.58,
    "EC-Earth3":        4.26,   "EC-Earth3-Veg-LR": 4.23,
    "GFDL-CM4":         3.89,   "GFDL-ESM4":        2.65,
    "INM-CM4-8":        1.83,   "INM-CM5-0":        1.92,
    "IPSL-CM6A-LR":     3.09,   "KACE-1-0-G":       4.48,
    "KIOST-ESM":        3.36,   "MIROC6":           2.60,
    "MPI-ESM1-2-HR":    2.98,   "MPI-ESM1-2-LR":    3.00,
    "MRI-ESM2-0":       3.13,   "NESM3":            4.72,
    "NorESM2-LM":       2.56,   "NorESM2-MM":       2.49,
    "TaiESM1":          4.36,
}

# CIL models (subset with hist + ssp245 EHD files)
CIL_ECS = {
    "ACCESS-CM2":       4.66,   "ACCESS-ESM1-5":    3.88,
    "BCC-CSM2-MR":      3.02,   "CanESM5":          5.64,
    "CMCC-CM2-SR5":     3.55,   "CMCC-ESM2":        3.58,
    "EC-Earth3":        4.26,   "EC-Earth3-Veg-LR": 4.23,
    "FGOALS-g3":        2.87,   "GFDL-CM4":         3.89,
    "GFDL-ESM4":        2.65,   "HadGEM3-GC31-LL":  5.55,
    "INM-CM4-8":        1.83,   "INM-CM5-0":        1.92,
    "MIROC6":           2.60,   "NorESM2-LM":       2.56,
    "NorESM2-MM":       2.49,
}


def compute_ecs_weights(models, ecs_dict, ecs_center=3.0, ecs_sigma=0.75):
    """Gaussian kernel weights centred on AR6 best estimate."""
    raw = {}
    for m in models:
        ecs = ecs_dict[m]
        raw[m] = np.exp(-0.5 * ((ecs - ecs_center) / ecs_sigma) ** 2)
    total = sum(raw.values())
    return {m: w / total for m, w in raw.items()}


# ─── Step 2: Load IFS slopes ─────────────────────────────────────────────────

def load_ifs_slopes():
    """Load both IFS slope fields at native 0.25°. Clip extreme values."""
    print("Loading IFS slopes...")

    ds_local = xr.open_dataset(str(SLOPE_LOCAL))
    rr_local = ds_local["rr_per_degC"].values.copy()
    ifs_lats = ds_local["latitude"].values   # descending: 83.5 → -90
    ifs_lons = ds_local["longitude"].values  # -180 → 179.75
    ds_local.close()

    ds_200km = xr.open_dataset(str(SLOPE_200KM))
    rr_200km = ds_200km["rr_per_degC"].values.copy()
    ds_200km.close()

    # Clip extremes: |RR| > 50 → NaN
    bad_local = np.abs(rr_local) > 50
    bad_200km = ~np.isfinite(rr_200km) | (np.abs(rr_200km) > 50)
    rr_local[bad_local] = np.nan
    rr_200km[bad_200km] = np.nan

    print(f"  Local:  {np.isfinite(rr_local).sum():,} valid, "
          f"clipped {bad_local.sum():,} extremes, "
          f"median={np.nanmedian(rr_local):.3f}")
    print(f"  200km:  {np.isfinite(rr_200km).sum():,} valid, "
          f"clipped {bad_200km.sum():,} extremes, "
          f"median={np.nanmedian(rr_200km):.3f}")

    return rr_local.astype(np.float32), rr_200km.astype(np.float32), ifs_lats, ifs_lons


# ─── Step 3: Load & coarsen EHD baseline to 0.25° ────────────────────────────

def load_ehd_year_01(year):
    """Load annual EHD at 0.1°, lon already -180..180."""
    p = EHD_DIR_ERA5 / f"exceedance_frequency_{year}.nc"
    if not p.exists():
        return None
    with xr.open_dataset(str(p)) as ds:
        return ds["t2m"].load().values.astype(np.float32)


def coarsen_01_to_025(arr_01, ifs_lats, ifs_lons):
    """
    Coarsen 0.1° (1801×3600) → IFS 0.25° grid (666×1440)
    via xarray interpolation onto IFS target coords.
    """
    da = xr.DataArray(
        arr_01, dims=["lat", "lon"],
        coords={"lat": ERA5_LATS, "lon": ERA5_LONS},
    )
    # Block-mean coarsen to ~0.25° first, then interp to exact IFS coords
    # ERA5 is 1801×3600, coarsen by 2-3 isn't clean. Just interpolate directly.
    return da.interp(lat=ifs_lats, lon=ifs_lons, method="linear").values.astype(np.float32)


def load_ehd_baseline_025(ifs_lats, ifs_lons):
    """Load mean EHD 2000-2024 at 0.25° (IFS grid)."""
    print("Loading EHD baseline (2000-2024) → coarsen to 0.25°...")
    parts = []
    for yr in SENS_YEARS:
        arr = load_ehd_year_01(yr)
        if arr is not None:
            parts.append(arr)
        if yr % 5 == 0:
            print(f"    {yr} ({len(parts)} years loaded)")
    stack = np.stack(parts, axis=0)
    baseline_01 = np.nanmean(stack, axis=0).astype(np.float32)

    baseline_025 = coarsen_01_to_025(baseline_01, ifs_lats, ifs_lons)
    print(f"  EHD baseline 0.25°: mean={np.nanmean(baseline_025):.4f}, "
          f"median={np.nanmedian(baseline_025):.4f}")
    return baseline_025, stack  # return stack for variant 5


# ─── Step 4: Compute ΔT fields ───────────────────────────────────────────────

def load_era5_summer_t2m_025(year):
    """
    Load ERA5 reanalysis summer-mean T2m at native 0.25° for one year.
    NH=JJA, SH=DJF. Returns (lats_desc, lons_0360, combined_2d) or None.
    Uses monthly mean GRIB files (~2MB each) — much faster than ERA5-Land daily.
    """
    def _load_month(yr, mo):
        p = ERA5_T2M_DIR / str(yr) / f"era5_t2m_{yr}{mo:02d}.grib"
        if not p.exists():
            return None
        ds = xr.open_dataset(str(p), engine="cfgrib")
        arr = ds["t2m"].values.astype(np.float32)
        lats = ds.latitude.values
        lons = ds.longitude.values
        ds.close()
        return lats, lons, arr

    # JJA
    jja_parts = []
    for m in [6, 7, 8]:
        r = _load_month(year, m)
        if r is None:
            return None
        jja_parts.append(r[2])
    jja_mean = np.nanmean(np.stack(jja_parts, axis=0), axis=0)

    # DJF (Dec from year-1, Jan+Feb from year)
    djf_parts = []
    r_dec = _load_month(year - 1, 12)
    r_jan = _load_month(year, 1)
    r_feb = _load_month(year, 2)
    if r_jan is None or r_feb is None:
        return None
    djf_parts = [r_jan[2], r_feb[2]]
    if r_dec is not None:
        djf_parts.append(r_dec[2])
    djf_mean = np.nanmean(np.stack(djf_parts, axis=0), axis=0)

    lats, lons = r[0], r[1]  # from JJA load
    nh_mask = lats >= 0
    combined = np.full_like(jja_mean, np.nan)
    combined[nh_mask, :] = jja_mean[nh_mask, :]
    combined[~nh_mask, :] = djf_mean[~nh_mask, :]

    return lats, lons, combined


def ols_slope_vectorized(X, Y):
    """Vectorized OLS slope: Y ~ β*X + α across spatial dims."""
    X_dm = X - np.nanmean(X, axis=0, keepdims=True)
    Y_dm = Y - np.nanmean(Y, axis=0, keepdims=True)
    num = np.nansum(X_dm * Y_dm, axis=0)
    den = np.nansum(X_dm ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(den > 0, num / den, np.nan)
    return beta.astype(np.float32)


def apply_gaussian_filter(arr_2d, sigma):
    """Apply Gaussian filter to a 2D field, handling NaN."""
    mask = np.isnan(arr_2d)
    filled = np.where(mask, 0.0, arr_2d.astype(np.float64))
    smoothed = gaussian_filter(filled, sigma=sigma, mode="wrap")
    weight = gaussian_filter((~mask).astype(np.float64), sigma=sigma, mode="wrap")
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(weight > 0.01, smoothed / weight, np.nan)
    return result.astype(np.float32)


def regrid_era5_to_ifs(data_025, lats_src, lons_src_360, ifs_lats, ifs_lons):
    """Regrid ERA5 reanalysis (0.25°, lon 0-360) to IFS grid (lon -180..180)."""
    lons_180 = lons_src_360.copy()
    lons_180[lons_180 >= 180] -= 360.0
    sort_idx = np.argsort(lons_180)
    da = xr.DataArray(
        data_025[:, sort_idx], dims=["lat", "lon"],
        coords={"lat": lats_src, "lon": lons_180[sort_idx]},
    )
    return da.interp(lat=ifs_lats, lon=ifs_lons, method="linear").values.astype(np.float32)


def _load_era5land_tmax_summer_year(year):
    """
    Load ERA5-Land daily Tmax at 0.1° for one year, compute summer mean.
    NH=JJA, SH=DJF.  Returns (year, 1801×3600 float32 array) or (year, None).
    Lon is 0-360 in source files; shifted to -180..180 in output.
    """
    tmax_dir = ERA5_TMAX_DIR

    def _load_month(yr, mo):
        p = tmax_dir / f"era5_tmax_{yr}_{mo:02d}.nc"
        if not p.exists():
            return None
        with xr.open_dataset(str(p)) as ds:
            # daily tmax -> monthly mean tmax  (valid_time, lat, lon)
            return ds["t2m"].values.astype(np.float32).mean(axis=0)  # (lat, lon)

    # JJA
    jja_parts = []
    for m in [6, 7, 8]:
        arr = _load_month(year, m)
        if arr is None:
            return (year, None)
        jja_parts.append(arr)
    jja_mean = np.nanmean(np.stack(jja_parts, axis=0), axis=0)

    # DJF (Dec from year-1, Jan+Feb from year)
    djf_parts = [_load_month(year, 1), _load_month(year, 2)]
    if djf_parts[0] is None or djf_parts[1] is None:
        return (year, None)
    dec = _load_month(year - 1, 12)
    if dec is not None:
        djf_parts.append(dec)
    djf_mean = np.nanmean(np.stack(djf_parts, axis=0), axis=0)

    # Combine: NH=JJA, SH=DJF  (lat descending: 90 → -90)
    n_lat = jja_mean.shape[0]
    nh_rows = n_lat // 2 + 1  # rows with lat >= 0
    combined = np.empty_like(jja_mean)
    combined[:nh_rows, :] = jja_mean[:nh_rows, :]
    combined[nh_rows:, :] = djf_mean[nh_rows:, :]

    # Shift lon from 0..360 to -180..180
    n_lon = combined.shape[1]
    half = n_lon // 2
    combined = np.roll(combined, half, axis=1)

    return (year, combined)


def compute_era5_dt_025(ifs_lats, ifs_lons):
    """
    Compute ERA5-Land Tmax ΔT for 2040 at IFS 0.25° grid.
    Uses ERA5-Land daily Tmax at 0.1° (summer mean JJA/DJF), loaded in parallel.
    Returns dt_local_025, dt_200km_025.
    """
    print("Computing ERA5-Land Tmax ΔT fields (parallel loading)...")
    n_yr = len(SENS_YEARS)
    n_lat, n_lon = len(ifs_lats), len(ifs_lons)

    # Parallel load of summer-mean tmax for each year
    with Pool(min(8, n_yr)) as pool:
        results = pool.map(_load_era5land_tmax_summer_year, SENS_YEARS)

    # Coarsen each year from 0.1° to IFS 0.25° grid
    tmax_025 = np.full((n_yr, n_lat, n_lon), np.nan, dtype=np.float32)
    for i, (yr, arr_01) in enumerate(sorted(results, key=lambda x: x[0])):
        if arr_01 is not None:
            tmax_025[i] = coarsen_01_to_025(arr_01, ifs_lats, ifs_lons)
        if yr % 5 == 0:
            print(f"    ERA5-Land Tmax {yr}")

    # OLS trend per pixel: Tmax(t) = α + slope × year
    print("  Fitting per-pixel OLS trends...")
    years_arr = np.array(SENS_YEARS, dtype=np.float32)
    years_3d = np.broadcast_to(years_arr[:, None, None], tmax_025.shape)
    trend_slope = ols_slope_vectorized(years_3d, tmax_025)

    # ΔT_2040 = slope × (2040 − midpoint)
    midpoint = np.mean(SENS_YEARS)
    dt_local_025 = (trend_slope * (TARGET_YEAR - midpoint)).astype(np.float32)

    # 200km smoothed variant
    dt_200km_025 = apply_gaussian_filter(dt_local_025, sigma=SIGMA_025)

    print(f"  ERA5-Land ΔTmax local:  mean={np.nanmean(dt_local_025):.3f}°C, "
          f"median={np.nanmedian(dt_local_025):.3f}°C")
    print(f"  ERA5-Land ΔTmax 200km:  mean={np.nanmean(dt_200km_025):.3f}°C, "
          f"median={np.nanmedian(dt_200km_025):.3f}°C")

    return dt_local_025, dt_200km_025


def cmip6_summer_mean_year(model, scenario, year):
    """Compute summer-mean Tmax from NEX-GDDP (0.25°, lon 0-360)."""
    p = TASMAX_DIR / model / scenario / f"{year}.nc"
    if not p.exists():
        return None

    with xr.open_dataset(str(p)) as ds:
        da = ds["tasmax"]
        jja = da.sel(time=da.time.dt.month.isin([6, 7, 8])).mean("time").load()
        jf = da.sel(time=da.time.dt.month.isin([1, 2])).mean("time").load()

    # December from year-1
    p_prev = TASMAX_DIR / model / scenario / f"{year - 1}.nc"
    if not p_prev.exists():
        p_prev = TASMAX_DIR / model / "historical" / f"{year - 1}.nc"
    if p_prev.exists():
        with xr.open_dataset(str(p_prev)) as ds_prev:
            dec = ds_prev["tasmax"].sel(
                time=ds_prev.time.dt.month == 12
            ).mean("time").load()
        djf = (dec + jf) / 2.0
    else:
        djf = jf

    lats = jja.lat.values
    lons = jja.lon.values
    nh_mask = lats >= 0
    combined = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
    combined[nh_mask, :] = jja.values[nh_mask, :].astype(np.float32)
    combined[~nh_mask, :] = djf.values[~nh_mask, :].astype(np.float32)

    return lats, lons, combined


def regrid_gcm_to_ifs(data_025, lats_src, lons_src_360, ifs_lats, ifs_lons):
    """Regrid GCM data (lon 0-360) to IFS grid (lon -180..180)."""
    lons_180 = lons_src_360.copy()
    lons_180[lons_180 >= 180] -= 360.0
    sort_idx = np.argsort(lons_180)
    lons_sorted = lons_180[sort_idx]
    data_sorted = data_025[:, sort_idx]

    da = xr.DataArray(
        data_sorted, dims=["lat", "lon"],
        coords={"lat": lats_src, "lon": lons_sorted},
    )
    return da.interp(lat=ifs_lats, lon=ifs_lons, method="linear").values.astype(np.float32)


GCM_DT_CACHE = PROJ_DIR / "output" / "gcm_dt_ecs_weighted_025.nc"

def compute_gcm_dt_025(ifs_lats, ifs_lons):
    """
    Compute ECS-weighted multi-model ΔT at IFS 0.25° grid.
    Uses a reduced baseline (5 years around 2005) and projection (5 years around 2040)
    to limit I/O. Results are cached.
    Returns dt_local_025, dt_200km_025.
    """
    # Check cache
    if GCM_DT_CACHE.exists():
        print(f"Loading cached GCM ΔT from {GCM_DT_CACHE.name}")
        ds_cache = xr.open_dataset(str(GCM_DT_CACHE))
        dt_local_025 = ds_cache["dt_local"].values.astype(np.float32)
        dt_200km_025 = ds_cache["dt_200km"].values.astype(np.float32)
        ds_cache.close()
        print(f"  GCM ΔT local:  mean={np.nanmean(dt_local_025):.3f}°C, "
              f"median={np.nanmedian(dt_local_025):.3f}°C")
        print(f"  GCM ΔT 200km:  mean={np.nanmean(dt_200km_025):.3f}°C, "
              f"median={np.nanmedian(dt_200km_025):.3f}°C")
        return dt_local_025, dt_200km_025

    print("Computing GCM ΔT fields (ECS-weighted multi-model mean)...")
    models = sorted(ECS_VALUES.keys())

    # Check which models have data; keep only top ECS-weighted models for speed
    all_available = []
    for m in models:
        p_test = TASMAX_DIR / m / "ssp245"
        if p_test.exists():
            all_available.append(m)

    # Keep top models by ECS weight (covering ~80% of total weight)
    weights_all = compute_ecs_weights(all_available, ECS_VALUES)
    ranked = sorted(all_available, key=lambda m: weights_all[m], reverse=True)
    MAX_MODELS = 8
    available = ranked[:MAX_MODELS]
    cum_weight = sum(weights_all[m] for m in available)
    print(f"  Using top {MAX_MODELS} of {len(all_available)} models "
          f"(cumulative ECS weight: {cum_weight:.1%})")

    # Reduced year sets: 3 baseline + 3 projection years (halves NFS I/O)
    baseline_years = [(yr, "historical" if yr <= 2014 else CLIM_SSP)
                      for yr in [2004, 2005, 2006]]
    proj_years = [(yr, "ssp245") for yr in [2039, 2040, 2041]]
    print(f"  Baseline years: {[y for y,_ in baseline_years]}")
    print(f"  Projection years: {[y for y,_ in proj_years]}")

    weights = compute_ecs_weights(available, ECS_VALUES)
    n_lat, n_lon = len(ifs_lats), len(ifs_lons)
    dt_wsum = np.zeros((n_lat, n_lon), dtype=np.float64)
    w_sum = np.zeros((n_lat, n_lon), dtype=np.float64)

    for mi, model in enumerate(available):
        t0 = time.time()
        print(f"  [{mi+1}/{len(available)}] {model} (w={weights[model]:.4f})...",
              end="", flush=True)

        # Baseline
        base_parts = []
        for yr, scen in baseline_years:
            r = cmip6_summer_mean_year(model, scen, yr)
            if r is not None:
                base_parts.append(r[2])
        if not base_parts:
            print(" no baseline — skip")
            continue
        lats_src, lons_src, _ = r
        baseline = np.nanmean(np.stack(base_parts, axis=0), axis=0)

        # Projection
        proj_parts = []
        for yr, scen in proj_years:
            r = cmip6_summer_mean_year(model, scen, yr)
            if r is not None:
                proj_parts.append(r[2])
        if not proj_parts:
            print(" no projection years — skip")
            continue

        proj_mean = np.nanmean(np.stack(proj_parts, axis=0), axis=0)
        dt_model = (proj_mean - baseline).astype(np.float32)

        # Regrid to IFS grid
        dt_ifs = regrid_gcm_to_ifs(dt_model, lats_src, lons_src, ifs_lats, ifs_lons)

        w = weights[model]
        valid = np.isfinite(dt_ifs)
        dt_wsum[valid] += w * dt_ifs[valid]
        w_sum[valid] += w

        print(f" ΔT mean={np.nanmean(dt_ifs):.2f}°C  ({time.time()-t0:.0f}s)")

    with np.errstate(divide="ignore", invalid="ignore"):
        dt_local_025 = np.where(w_sum > 0, dt_wsum / w_sum, np.nan).astype(np.float32)

    dt_200km_025 = apply_gaussian_filter(dt_local_025, sigma=SIGMA_025)

    # Cache
    ds_out = xr.Dataset({
        "dt_local": xr.DataArray(dt_local_025, dims=["lat", "lon"],
                                  coords={"lat": ifs_lats, "lon": ifs_lons}),
        "dt_200km": xr.DataArray(dt_200km_025, dims=["lat", "lon"],
                                  coords={"lat": ifs_lats, "lon": ifs_lons}),
    })
    ds_out.to_netcdf(str(GCM_DT_CACHE))
    print(f"  Cached → {GCM_DT_CACHE.name}")

    print(f"  GCM ΔT local:  mean={np.nanmean(dt_local_025):.3f}°C, "
          f"median={np.nanmedian(dt_local_025):.3f}°C")
    print(f"  GCM ΔT 200km:  mean={np.nanmean(dt_200km_025):.3f}°C, "
          f"median={np.nanmedian(dt_200km_025):.3f}°C")

    return dt_local_025, dt_200km_025


# ─── Step 5-6: Compute ΔEHD variants ─────────────────────────────────────────

def compute_slope_dehd(ehd_base, rr, dt):
    """ΔEHD = EHD_base × (RR^ΔT − 1), clipped so EHD stays in [0,1]."""
    with np.errstate(invalid="ignore", over="ignore"):
        rr_powered = np.power(rr, dt)
        dehd = ehd_base * (rr_powered - 1.0)
        # Clip: new EHD = base + ΔEHD must be in [0, 1]
        new_ehd = ehd_base + dehd
        new_ehd = np.clip(new_ehd, 0.0, 1.0)
        dehd = new_ehd - ehd_base
    return dehd.astype(np.float32)


def compute_era5_ehd_trend_025(ehd_stack_01, ifs_lats, ifs_lons):
    """
    Variant 5: Linear trend extrapolation of ERA5-Land EHD.
    ehd_stack_01: (n_years, 1801, 3600) at 0.1°
    Returns ΔEHD at 0.25°.
    """
    print("Computing ERA5-Land EHD trend extrapolation (variant 5)...")
    n_yr = ehd_stack_01.shape[0]
    n_lat_025, n_lon_025 = len(ifs_lats), len(ifs_lons)

    # Coarsen each year to 0.25°
    ehd_025 = np.full((n_yr, n_lat_025, n_lon_025), np.nan, dtype=np.float32)
    for i in range(n_yr):
        ehd_025[i] = coarsen_01_to_025(ehd_stack_01[i], ifs_lats, ifs_lons)

    # OLS trend per pixel
    years_arr = np.array(SENS_YEARS, dtype=np.float32)
    years_3d = np.broadcast_to(years_arr[:, None, None], ehd_025.shape)
    trend_slope = ols_slope_vectorized(years_3d, ehd_025)
    intercept = np.nanmean(ehd_025, axis=0) - trend_slope * np.mean(SENS_YEARS)

    ehd_2040 = intercept + trend_slope * TARGET_YEAR
    ehd_baseline = np.nanmean(ehd_025, axis=0)
    dehd = (ehd_2040 - ehd_baseline).astype(np.float32)

    # Clip
    new_ehd = np.clip(ehd_baseline + dehd, 0.0, 1.0)
    dehd = (new_ehd - ehd_baseline).astype(np.float32)

    print(f"  EHD trend slope: mean={np.nanmean(trend_slope):.6f}/yr, "
          f"median={np.nanmedian(trend_slope):.6f}/yr")
    print(f"  ΔEHD: mean={np.nanmean(dehd):.4f}, median={np.nanmedian(dehd):.4f}")
    return dehd


def compute_cil_dehd_025(ifs_lats, ifs_lons):
    """
    Variant 6: CIL GDPCIR EHD — ECS-weighted multi-model ΔEHD.
    Returns ΔEHD at IFS 0.25° grid.
    """
    print("Computing CIL GDPCIR ΔEHD (variant 6)...")
    models = sorted(CIL_ECS.keys())

    # Check which models have data
    available = []
    for m in models:
        hist = CIL_EHD_DIR / m / "ehd_historical.nc"
        ssp = CIL_EHD_DIR / m / "ehd_ssp245.nc"
        if hist.exists() and ssp.exists():
            available.append(m)
    print(f"  {len(available)}/{len(models)} CIL models available")

    weights = compute_ecs_weights(available, CIL_ECS)
    n_lat, n_lon = len(ifs_lats), len(ifs_lons)
    dehd_wsum = np.zeros((n_lat, n_lon), dtype=np.float64)
    w_sum = np.zeros((n_lat, n_lon), dtype=np.float64)

    for mi, model in enumerate(available):
        t0 = time.time()
        print(f"  [{mi+1}/{len(available)}] {model} (w={weights[model]:.4f})...",
              end="", flush=True)

        # Load historical baseline (2000-2014)
        ds_hist = xr.open_dataset(str(CIL_EHD_DIR / model / "ehd_historical.nc"))
        lats_src = ds_hist.lat.values
        lons_src = ds_hist.lon.values  # 0-360

        yrs_h = ds_hist.year.values
        mask_h = (yrs_h >= 2000) & (yrs_h <= 2014)
        hist_mean = ds_hist["ehd"].values[mask_h].mean(axis=0) if mask_h.any() else None
        ds_hist.close()

        # Load ssp245: baseline extension (2015-2024) + projection (2035-2044)
        ds_ssp = xr.open_dataset(str(CIL_EHD_DIR / model / "ehd_ssp245.nc"))
        yrs_s = ds_ssp.year.values
        ehd_ssp = ds_ssp["ehd"].values

        mask_bl = (yrs_s >= 2015) & (yrs_s <= 2024)
        bl_parts = []
        if hist_mean is not None:
            bl_parts.append(hist_mean)
        if mask_bl.any():
            bl_parts.append(ehd_ssp[mask_bl].mean(axis=0))
        if not bl_parts:
            ds_ssp.close()
            print(" no baseline — skip")
            continue
        baseline = np.nanmean(np.stack(bl_parts, axis=0), axis=0)

        # Projection: 10-year mean around 2040 (clip to available years)
        mask_proj = (yrs_s >= PROJ_WINDOW[0]) & (yrs_s <= min(PROJ_WINDOW[1], int(yrs_s.max())))
        if not mask_proj.any():
            ds_ssp.close()
            print(" no projection years — skip")
            continue
        proj_mean = ehd_ssp[mask_proj].mean(axis=0)
        ds_ssp.close()

        dehd_model = (proj_mean - baseline).astype(np.float32)

        # Regrid to IFS grid
        dehd_ifs = regrid_gcm_to_ifs(dehd_model, lats_src, lons_src, ifs_lats, ifs_lons)

        w = weights[model]
        valid = np.isfinite(dehd_ifs)
        dehd_wsum[valid] += w * dehd_ifs[valid]
        w_sum[valid] += w

        print(f" ΔEHD mean={np.nanmean(dehd_ifs):.4f}  ({time.time()-t0:.0f}s)")

    with np.errstate(divide="ignore", invalid="ignore"):
        dehd = np.where(w_sum > 0, dehd_wsum / w_sum, np.nan).astype(np.float32)

    print(f"  CIL ΔEHD: mean={np.nanmean(dehd):.4f}, median={np.nanmedian(dehd):.4f}")
    return dehd


# ─── Step 7-8: Interpolate to 0.1° and aggregate to entities ─────────────────

def interp_025_to_01(arr_025, ifs_lats, ifs_lons):
    """Bilinearly interpolate 0.25° → ERA5 0.1°."""
    da = xr.DataArray(
        arr_025, dims=["lat", "lon"],
        coords={"lat": ifs_lats, "lon": ifs_lons},
    )
    return da.interp(lat=ERA5_LATS, lon=ERA5_LONS, method="linear").values.astype(np.float32)


def load_mixed_raster():
    """Build combined raster mapping each 0.1° pixel to Approach-2 entity."""
    panel = pd.read_parquet(str(PANEL_FILE),
                            columns=["GID_2", "iso3", "is_admin1"])
    entities = (panel[["GID_2", "iso3", "is_admin1"]]
                .drop_duplicates("GID_2")
                .reset_index(drop=True))
    entities["entity_idx"] = np.arange(len(entities))
    n_entities = len(entities)

    # Admin2 raster
    admin2_grid = np.load(str(ADMIN2_RASTER))
    lk2 = pd.read_csv(str(ADMIN2_LOOKUP))
    adm2_entities = entities[~entities["is_admin1"]]
    rid2eidx = dict(
        zip(
            lk2.set_index("GID_2")["raster_id"].reindex(adm2_entities["GID_2"]).values,
            adm2_entities["entity_idx"].values,
        )
    )
    max_rid = int(admin2_grid.max()) + 1
    adm2_map = np.full(max_rid, -1, dtype=np.int32)
    for rid, eidx in rid2eidx.items():
        if not np.isnan(rid):
            adm2_map[int(rid)] = int(eidx)

    combined = np.where(admin2_grid >= 0,
                        adm2_map[np.clip(admin2_grid, 0, max_rid - 1)],
                        -1).astype(np.int32)

    # Admin1 overlay
    ds1 = xr.open_dataset(str(ADMIN1_RASTER))
    admin1_grid = ds1["admin1_id"].values.astype(np.int32)
    ds1.close()
    lk1 = pd.read_csv(str(ADMIN1_LOOKUP))

    adm1_entities = entities[entities["is_admin1"]].copy()
    adm1_entities["GID_nmbr"] = (adm1_entities["GID_2"]
                                 .str.replace("ADM1_", "")
                                 .astype(float))
    lk1_sub = lk1.merge(
        adm1_entities[["GID_nmbr", "entity_idx"]], on="GID_nmbr", how="inner"
    )
    max_rid1 = int(admin1_grid.max()) + 1
    adm1_map = np.full(max_rid1, -1, dtype=np.int32)
    for _, row in lk1_sub.iterrows():
        rid = int(row["raster_id"])
        if 0 <= rid < max_rid1:
            adm1_map[rid] = int(row["entity_idx"])

    adm1_eidx = np.where(admin1_grid >= 0,
                         adm1_map[np.clip(admin1_grid, 0, max_rid1 - 1)],
                         -1).astype(np.int32)
    mask_adm1 = adm1_eidx >= 0
    combined[mask_adm1] = adm1_eidx[mask_adm1]

    return combined, entities, n_entities


def load_settlement():
    ds = xr.open_dataset(str(SETTLE_FILE))
    arr = ds["built_surface"].values.astype(np.float32)
    ds.close()
    return arr


def agg_to_entities(heat_2d, combined_raster, settlement, n_entities):
    """Settlement-weighted mean of heat_2d → float32 array (n_entities,)."""
    valid = (combined_raster >= 0) & np.isfinite(heat_2d) & (settlement > 0)
    ids_v = combined_raster[valid]
    heat_v = heat_2d[valid].astype(np.float64)
    sett_v = settlement[valid].astype(np.float64)

    wsum = np.bincount(ids_v, weights=heat_v * sett_v, minlength=n_entities)
    wsm = np.bincount(ids_v, weights=sett_v, minlength=n_entities)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(wsm > 0, wsum / wsm, np.nan).astype(np.float32)


# ─── Step 9: GDP impacts and global weighted means ───────────────────────────

def compute_global_stats(dehd_entity, panel_entities, beta):
    """Compute GDP-weighted and settlement-weighted global means."""
    panel = pd.read_parquet(str(PANEL_FILE))

    # Get mean GDP per capita per entity
    gdp_mean = panel.groupby("GID_2")["gdp_per_capita"].mean()

    # Match to entities
    ent = panel_entities.copy()
    ent["dehd"] = dehd_entity
    ent["gdp_pc"] = ent["GID_2"].map(gdp_mean).astype(np.float64)
    ent["gdp_impact"] = beta * ent["dehd"]

    valid = ent.dropna(subset=["dehd", "gdp_pc"])

    # GDP-weighted mean
    gdp_w = valid["gdp_pc"].values
    gdp_weighted_impact = np.average(valid["gdp_impact"].values, weights=gdp_w)
    gdp_weighted_dehd = np.average(valid["dehd"].values, weights=gdp_w)

    # Simple (settlement-weighted) mean — already settlement-weighted in aggregation
    sett_weighted_impact = valid["gdp_impact"].mean()
    sett_weighted_dehd = valid["dehd"].mean()

    dehd_vals = valid["dehd"].values
    return {
        "gdp_weighted_dehd": gdp_weighted_dehd,
        "gdp_weighted_impact": gdp_weighted_impact,
        "sett_weighted_dehd": sett_weighted_dehd,
        "sett_weighted_impact": sett_weighted_impact,
        "median_dehd": float(np.nanmedian(dehd_vals)),
        "p5_dehd": float(np.nanpercentile(dehd_vals, 5)),
        "p95_dehd": float(np.nanpercentile(dehd_vals, 95)),
        "n_entities": len(valid),
    }


# ─── Step 10: Maps ───────────────────────────────────────────────────────────

def plot_6panel(dehd_grids, ifs_lats, ifs_lons, variant_names, out_path):
    """2×3 panel figure of ΔEHD maps at 0.25°."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.ravel()

    bounds = [-0.005, 0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.10, 0.15]
    cmap = plt.cm.get_cmap("plasma", len(bounds) - 1)
    norm = BoundaryNorm(bounds, cmap.N)

    lon_mesh, lat_mesh = np.meshgrid(ifs_lons, ifs_lats)

    for i, (ax, grid, name) in enumerate(zip(axes, dehd_grids, variant_names)):
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_facecolor("#e8e8e8")

        pcm = ax.pcolormesh(lon_mesh, lat_mesh, grid,
                            norm=norm, cmap=cmap, shading="auto",
                            rasterized=True)
        ax.set_title(name, fontsize=11, fontweight="bold")
        if i >= 3:
            ax.set_xlabel("Longitude", fontsize=9)
        if i % 3 == 0:
            ax.set_ylabel("Latitude", fontsize=9)
        ax.tick_params(labelsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="horizontal",
                        fraction=0.025, pad=0.06, aspect=50, ticks=bounds)
    cbar.set_label(u"ΔEHD (fraction, 2040 vs baseline)", fontsize=12)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(u"Six independent ΔEHD projections for 2040 (SSP2-4.5)\n"
                 u"IFS reforecast slopes × ΔT sources + direct methods",
                 fontsize=15, fontweight="bold", y=1.02)

    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    wall_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Six ΔEHD projection maps using IFS slopes")
    print("=" * 65)

    # ── Load regression beta ──────────────────────────────────────────────
    regr = pd.read_csv(str(REGR_FILE))
    row = regr[
        (regr["spec"] == "linear_ctrl_baseline") &
        (regr["variable"] == "heat_freq_weighted") &
        (regr["se_type"] == "conley_500km")
    ]
    beta = float(row.iloc[0]["beta"])
    print(f"  β (Approach 2, Conley 500km) = {beta:.6f}")

    # ── Step 2: Load IFS slopes ───────────────────────────────────────────
    rr_local, rr_200km, ifs_lats, ifs_lons = load_ifs_slopes()

    # ── Step 3: Load EHD baseline ─────────────────────────────────────────
    ehd_base_025, ehd_stack_01 = load_ehd_baseline_025(ifs_lats, ifs_lons)

    # ── Step 4a: ERA5-Land ΔT ─────────────────────────────────────────────
    dt_era5_local, dt_era5_200km = compute_era5_dt_025(ifs_lats, ifs_lons)

    # ── Step 4b: GCM ΔT ──────────────────────────────────────────────────
    dt_gcm_local, dt_gcm_200km = compute_gcm_dt_025(ifs_lats, ifs_lons)

    # ── Step 5: Compute slope-based ΔEHD (variants 1-4) ───────────────────
    print("\nComputing ΔEHD variants 1-4 (slope-based)...")
    dehd_v1 = compute_slope_dehd(ehd_base_025, rr_local, dt_era5_local)
    dehd_v2 = compute_slope_dehd(ehd_base_025, rr_200km, dt_era5_200km)
    dehd_v3 = compute_slope_dehd(ehd_base_025, rr_local, dt_gcm_local)
    dehd_v4 = compute_slope_dehd(ehd_base_025, rr_200km, dt_gcm_200km)

    for label, arr in [("V1 ERA5×local", dehd_v1), ("V2 ERA5×200km", dehd_v2),
                        ("V3 GCM×local", dehd_v3), ("V4 GCM×200km", dehd_v4)]:
        print(f"  {label}: mean={np.nanmean(arr):.4f}, "
              f"median={np.nanmedian(arr):.4f}, "
              f"p95={np.nanpercentile(arr[np.isfinite(arr)], 95):.4f}")

    # ── Step 6a: Variant 5 — ERA5 EHD trend ───────────────────────────────
    dehd_v5 = compute_era5_ehd_trend_025(ehd_stack_01, ifs_lats, ifs_lons)
    del ehd_stack_01  # free memory

    # ── Step 6b: Variant 6 — CIL GDPCIR ──────────────────────────────────
    dehd_v6 = compute_cil_dehd_025(ifs_lats, ifs_lons)

    dehd_grids = [dehd_v1, dehd_v2, dehd_v3, dehd_v4, dehd_v5, dehd_v6]
    variant_names = [
        "V1: ERA5 local trend × IFS local slope",
        "V2: ERA5 200km trend × IFS 200km slope",
        "V3: GCM local ΔT × IFS local slope",
        "V4: GCM 200km ΔT × IFS 200km slope",
        "V5: ERA5-Land EHD direct trend",
        "V6: CIL GDPCIR EHD (ECS-weighted)",
    ]
    variant_short = [
        "era5_local_slope", "era5_200km_slope",
        "gcm_local_slope", "gcm_200km_slope",
        "era5_ehd_trend", "cil_gdpcir",
    ]

    # ── Step 7-8: Aggregate to entities ───────────────────────────────────
    print("\nAggregating to Approach 2 entities...")
    t0 = time.time()
    combined_raster, entities, n_entities = load_mixed_raster()
    settlement = load_settlement()
    print(f"  {n_entities:,} entities loaded  ({time.time()-t0:.0f}s)")

    dehd_entities = []
    for i, (grid, name) in enumerate(zip(dehd_grids, variant_short)):
        print(f"  Interpolating {name} → 0.1° and aggregating...", end="", flush=True)
        t0 = time.time()
        grid_01 = interp_025_to_01(grid, ifs_lats, ifs_lons)
        ent_vals = agg_to_entities(grid_01, combined_raster, settlement, n_entities)
        dehd_entities.append(ent_vals)
        n_valid = np.isfinite(ent_vals).sum()
        print(f" {n_valid:,} valid, mean={np.nanmean(ent_vals):.4f}  ({time.time()-t0:.0f}s)")

    # ── Step 9: GDP impacts ───────────────────────────────────────────────
    print("\nComputing GDP impacts and global weighted means...")
    rows = []
    for i, (name, short) in enumerate(zip(variant_names, variant_short)):
        stats = compute_global_stats(dehd_entities[i], entities, beta)
        rows.append({
            "variant": short,
            "variant_label": name,
            "n_entities": stats["n_entities"],
            "global_gdp_weighted_dehd": stats["gdp_weighted_dehd"],
            "global_gdp_weighted_impact": stats["gdp_weighted_impact"],
            "global_sett_weighted_dehd": stats["sett_weighted_dehd"],
            "global_sett_weighted_impact": stats["sett_weighted_impact"],
            "median_dehd": stats["median_dehd"],
            "p5_dehd": stats["p5_dehd"],
            "p95_dehd": stats["p95_dehd"],
        })

    df_comp = pd.DataFrame(rows)

    # ── Step 11: Save comparison CSV ──────────────────────────────────────
    csv_path = OUT_DIR / "slopes_6variant_comparison.csv"
    df_comp.to_csv(str(csv_path), index=False, float_format="%.6f")
    print(f"\nSaved comparison: {csv_path}")

    # Print formatted table
    print("\n" + "=" * 90)
    print("Six-variant ΔEHD comparison (2040, SSP2-4.5)")
    print("=" * 90)
    print(f"{'Variant':<28} {'GDP-wt ΔEHD':>12} {'GDP-wt impact':>14} "
          f"{'Sett-wt ΔEHD':>13} {'Sett-wt impact':>15} {'Median':>8}")
    print("-" * 90)
    for _, r in df_comp.iterrows():
        print(f"{r['variant']:<28} {r['global_gdp_weighted_dehd']:>12.4f} "
              f"{r['global_gdp_weighted_impact']:>14.6f} "
              f"{r['global_sett_weighted_dehd']:>13.4f} "
              f"{r['global_sett_weighted_impact']:>15.6f} "
              f"{r['median_dehd']:>8.4f}")

    # ── Step 10: Generate maps ────────────────────────────────────────────
    print("\nGenerating 6-panel map...")
    map_path = OUT_DIR / "map_slopes_6variant_ehd_2040.png"
    plot_6panel(dehd_grids, ifs_lats, ifs_lons, variant_names, map_path)

    elapsed = time.time() - wall_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
