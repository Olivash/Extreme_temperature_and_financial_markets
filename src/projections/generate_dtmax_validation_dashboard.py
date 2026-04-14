#!/usr/bin/env python -u
"""
ΔTmax-centred validation dashboard.

Computes ERA5 and GCM summer ΔTmax anomalies (2013-2024 vs 2000-2012) at
local and 200km (regional) scales, applies IFS reforecast slopes to produce
ΔEHD, compares against observed ΔEHD, and computes GDP impacts. Generates
an interactive Leaflet HTML dashboard.

Layers:
  - ΔTmax: ERA5 local, ERA5 200km, GCM local, GCM 200km
  - ΔEHD:  V1 (ERA5 local), V2 (ERA5 200km), V3 (GCM local), V4 (GCM 200km),
           V5 (ERA5 EHD trend), V6 (CIL direct), Observed
  - Residuals: each predicted − observed
  - GDP impact: β × ΔEHD per country

Output: projections/output/dtmax_validation_explorer.html

Usage:
  cd ~/projects/macro/extreme_heat/biodiversity_interactions
  /turbo/mgoldklang/pyenvs/peg_nov_24/bin/python projections/scripts/generate_dtmax_validation_dashboard.py
"""

import sys
import os
import io
import json
import time
import base64
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy.stats import pearsonr, spearmanr
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_slopes_ehd_maps import (
    load_ifs_slopes, compute_slope_dehd, compute_ecs_weights,
    cmip6_summer_mean_year, regrid_gcm_to_ifs, load_settlement,
    agg_to_entities, interp_025_to_01, coarsen_01_to_025,
    load_ehd_year_01, ols_slope_vectorized, apply_gaussian_filter,
    _load_era5land_tmax_summer_year, compute_era5_ehd_trend_025,
    load_mixed_raster, compute_global_stats,
    ECS_VALUES, CIL_ECS, CIL_EHD_DIR, TASMAX_DIR, SIGMA_025,
    ERA5_LATS, ERA5_LONS, PANEL_FILE, REGR_FILE,
)
from bootstrap_impact_comparison import load_admin1_raster

PROJ_DIR   = Path(__file__).resolve().parent.parent
ROOT_DIR   = PROJ_DIR.parent
PARENT_DIR = ROOT_DIR.parent
OUT_DIR    = PROJ_DIR / "output"

TRAIN_YEARS = list(range(2000, 2013))
VAL_YEARS   = list(range(2013, 2025))
ALL_YEARS   = list(range(2000, 2025))
TRAIN_MID   = np.mean(TRAIN_YEARS)
VAL_MID     = np.mean(VAL_YEARS)
CLIM_SSP    = "ssp245"

ADMIN1_GPKG = PARENT_DIR / "polyg_adm1_gdp_perCapita_1990_2022 (1).gpkg"

# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1: Compute ΔTmax fields
# ═══════════════════════════════════════════════════════════════════════════════

def compute_era5_dtmax(ifs_lats, ifs_lons):
    """
    ERA5-Land summer Tmax trend fitted on TRAIN_YEARS, extrapolated to VAL_MID.
    Returns (dt_local, dt_200km) at IFS 0.25°.
    """
    print("\n" + "="*70)
    print("ERA5-Land ΔTmax (OLS trend 2000-2012 → extrapolate to 2018.5)")
    print("="*70)
    n_yr = len(TRAIN_YEARS)
    n_lat, n_lon = len(ifs_lats), len(ifs_lons)

    with Pool(min(8, n_yr)) as pool:
        results = pool.map(_load_era5land_tmax_summer_year, TRAIN_YEARS)

    tmax_025 = np.full((n_yr, n_lat, n_lon), np.nan, dtype=np.float32)
    for i, (yr, arr_01) in enumerate(sorted(results, key=lambda x: x[0])):
        if arr_01 is not None:
            tmax_025[i] = coarsen_01_to_025(arr_01, ifs_lats, ifs_lons)
        if yr % 5 == 0:
            print(f"    {yr}")

    years_arr = np.array(TRAIN_YEARS, dtype=np.float32)
    years_3d = np.broadcast_to(years_arr[:, None, None], tmax_025.shape)
    trend_slope = ols_slope_vectorized(years_3d, tmax_025)

    dt_local = (trend_slope * (VAL_MID - TRAIN_MID)).astype(np.float32)
    dt_200km = apply_gaussian_filter(dt_local, sigma=SIGMA_025)

    n_valid = np.isfinite(dt_local).sum()
    print(f"\n  ERA5 ΔTmax local:  mean={np.nanmean(dt_local):.4f}°C, "
          f"median={np.nanmedian(dt_local):.4f}°C, valid={n_valid:,}")
    print(f"  ERA5 ΔTmax 200km:  mean={np.nanmean(dt_200km):.4f}°C, "
          f"median={np.nanmedian(dt_200km):.4f}°C")
    return dt_local, dt_200km


def compute_gcm_dtmax(ifs_lats, ifs_lons):
    """
    GCM ECS-weighted ΔTmax: mean(tasmax 2013-2024) − mean(tasmax 2000-2012).
    Returns (dt_local, dt_200km) at IFS 0.25°.
    """
    print("\n" + "="*70)
    print("GCM ΔTmax (ECS-weighted, 2013-2024 vs 2000-2012)")
    print("="*70)
    models = sorted(ECS_VALUES.keys())
    available = [m for m in models if (TASMAX_DIR / m / "ssp245").exists()]
    weights_all = compute_ecs_weights(available, ECS_VALUES)
    ranked = sorted(available, key=lambda m: weights_all[m], reverse=True)
    available = ranked[:8]
    weights = compute_ecs_weights(available, ECS_VALUES)
    print(f"  Using top 8 models (of {len(ranked)} available)")

    n_lat, n_lon = len(ifs_lats), len(ifs_lons)
    dt_wsum = np.zeros((n_lat, n_lon), dtype=np.float64)
    w_sum = np.zeros((n_lat, n_lon), dtype=np.float64)

    for mi, model in enumerate(available):
        t0 = time.time()
        print(f"  [{mi+1}/{len(available)}] {model} (w={weights[model]:.4f})...",
              end="", flush=True)

        # Baseline: mean tasmax over TRAIN_YEARS
        base_parts = []
        for yr in TRAIN_YEARS:
            scen = "historical" if yr <= 2014 else CLIM_SSP
            r = cmip6_summer_mean_year(model, scen, yr)
            if r is not None:
                base_parts.append(r[2])
        if not base_parts:
            print(" skip (no baseline)")
            continue
        lats_src, lons_src, _ = r
        baseline = np.nanmean(np.stack(base_parts, axis=0), axis=0)

        # Validation: mean tasmax over VAL_YEARS
        val_parts = []
        for yr in VAL_YEARS:
            scen = "historical" if yr <= 2014 else CLIM_SSP
            r = cmip6_summer_mean_year(model, scen, yr)
            if r is not None:
                val_parts.append(r[2])
        if not val_parts:
            print(" skip (no val)")
            continue
        val_mean = np.nanmean(np.stack(val_parts, axis=0), axis=0)
        dt_model = (val_mean - baseline).astype(np.float32)

        dt_ifs = regrid_gcm_to_ifs(dt_model, lats_src, lons_src, ifs_lats, ifs_lons)
        w = weights[model]
        valid = np.isfinite(dt_ifs)
        dt_wsum[valid] += w * dt_ifs[valid]
        w_sum[valid] += w
        print(f" ΔT={np.nanmean(dt_ifs):.3f}°C, "
              f"valid={valid.sum():,} ({time.time()-t0:.0f}s)")

    with np.errstate(divide="ignore", invalid="ignore"):
        dt_local = np.where(w_sum > 0, dt_wsum / w_sum, np.nan).astype(np.float32)
    dt_200km = apply_gaussian_filter(dt_local, sigma=SIGMA_025)

    n_valid = np.isfinite(dt_local).sum()
    print(f"\n  GCM ΔTmax local:  mean={np.nanmean(dt_local):.4f}°C, "
          f"median={np.nanmedian(dt_local):.4f}°C, valid={n_valid:,}")
    print(f"  GCM ΔTmax 200km:  mean={np.nanmean(dt_200km):.4f}°C, "
          f"median={np.nanmedian(dt_200km):.4f}°C")
    return dt_local, dt_200km


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2: Compute ΔEHD from ΔTmax via IFS slopes, plus direct methods
# ═══════════════════════════════════════════════════════════════════════════════

def compute_observed_dehd(ifs_lats, ifs_lons):
    """Load observed ΔEHD = mean(EHD 2013-2024) − mean(EHD 2000-2012) at 0.25°."""
    print("\nLoading observed EHD...")
    train_parts, val_parts = [], []
    for yr in TRAIN_YEARS:
        arr = load_ehd_year_01(yr)
        if arr is not None:
            train_parts.append(arr)
    for yr in VAL_YEARS:
        arr = load_ehd_year_01(yr)
        if arr is not None:
            val_parts.append(arr)

    ehd_train_01 = np.nanmean(np.stack(train_parts, axis=0), axis=0).astype(np.float32)
    ehd_val_01 = np.nanmean(np.stack(val_parts, axis=0), axis=0).astype(np.float32)
    ehd_train_025 = coarsen_01_to_025(ehd_train_01, ifs_lats, ifs_lons)
    ehd_val_025 = coarsen_01_to_025(ehd_val_01, ifs_lats, ifs_lons)

    observed = (ehd_val_025 - ehd_train_025).astype(np.float32)
    print(f"  Observed ΔEHD: mean={np.nanmean(observed):.4f}, "
          f"median={np.nanmedian(observed):.4f}")
    return observed, ehd_train_025, np.stack(train_parts, axis=0)


def compute_v5_dehd(ehd_train_stack_01, ifs_lats, ifs_lons):
    """V5: Direct ERA5 EHD trend extrapolation."""
    print("\nComputing V5: ERA5 EHD trend...")
    n_yr = len(TRAIN_YEARS)
    n_lat, n_lon = len(ifs_lats), len(ifs_lons)
    stack_025 = np.full((n_yr, n_lat, n_lon), np.nan, dtype=np.float32)
    for i in range(n_yr):
        stack_025[i] = coarsen_01_to_025(ehd_train_stack_01[i], ifs_lats, ifs_lons)

    years_arr = np.array(TRAIN_YEARS, dtype=np.float32)
    years_3d = np.broadcast_to(years_arr[:, None, None], stack_025.shape)
    slope = ols_slope_vectorized(years_3d, stack_025)
    ehd_base = np.nanmean(stack_025, axis=0)

    dehd = (slope * (VAL_MID - TRAIN_MID)).astype(np.float32)
    new_ehd = np.clip(ehd_base + dehd, 0.0, 1.0)
    dehd = (new_ehd - ehd_base).astype(np.float32)
    print(f"  V5 ΔEHD: mean={np.nanmean(dehd):.4f}")
    return dehd


def compute_v6_dehd(ifs_lats, ifs_lons):
    """V6: CIL GDPCIR ECS-weighted ΔEHD for validation period."""
    print("\nComputing V6: CIL GDPCIR ΔEHD...")
    cil_models = sorted(CIL_ECS.keys())
    available = [m for m in cil_models
                 if (CIL_EHD_DIR / m / "ehd_historical.nc").exists()
                 and (CIL_EHD_DIR / m / "ehd_ssp245.nc").exists()]
    weights = compute_ecs_weights(available, CIL_ECS)
    print(f"  {len(available)} CIL models available")

    n_lat, n_lon = len(ifs_lats), len(ifs_lons)
    dehd_wsum = np.zeros((n_lat, n_lon), dtype=np.float64)
    w_sum = np.zeros((n_lat, n_lon), dtype=np.float64)

    for model in available:
        ds_hist = xr.open_dataset(str(CIL_EHD_DIR / model / "ehd_historical.nc"))
        lats_s, lons_s = ds_hist.lat.values, ds_hist.lon.values
        yrs_h = ds_hist.year.values
        ehd_hist = ds_hist["ehd"].values
        ds_hist.close()
        ds_ssp = xr.open_dataset(str(CIL_EHD_DIR / model / "ehd_ssp245.nc"))
        yrs_s = ds_ssp.year.values
        ehd_ssp = ds_ssp["ehd"].values
        ds_ssp.close()

        mask_bl = (yrs_h >= 2000) & (yrs_h <= 2012)
        if not mask_bl.any():
            continue
        bl = ehd_hist[mask_bl].mean(axis=0)

        val_p = []
        mask_vh = (yrs_h >= 2013) & (yrs_h <= 2014)
        if mask_vh.any():
            val_p.append(ehd_hist[mask_vh])
        mask_vs = (yrs_s >= 2015) & (yrs_s <= 2024)
        if mask_vs.any():
            val_p.append(ehd_ssp[mask_vs])
        if not val_p:
            continue

        vm = np.concatenate(val_p, axis=0).mean(axis=0)
        dehd_m = (vm - bl).astype(np.float32)
        dehd_ifs = regrid_gcm_to_ifs(dehd_m, lats_s, lons_s, ifs_lats, ifs_lons)
        w = weights[model]
        v = np.isfinite(dehd_ifs)
        dehd_wsum[v] += w * dehd_ifs[v]
        w_sum[v] += w

    with np.errstate(divide="ignore", invalid="ignore"):
        dehd = np.where(w_sum > 0, dehd_wsum / w_sum, np.nan).astype(np.float32)
    print(f"  V6 ΔEHD: mean={np.nanmean(dehd):.4f}")
    return dehd


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred, obs):
    mask = np.isfinite(pred) & np.isfinite(obs)
    p, o = pred[mask], obs[mask]
    if len(p) < 10:
        return {"rmse": np.nan, "bias": np.nan, "r": np.nan, "rho": np.nan, "n": len(p)}
    resid = p - o
    return {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "bias": float(np.mean(resid)),
        "r": float(np.corrcoef(p, o)[0, 1]),
        "rho": float(spearmanr(p, o).correlation),
        "n": len(p),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_bin_colors_hex(bounds, cmap_name):
    cmap = plt.cm.get_cmap(cmap_name, len(bounds) - 1)
    return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            for r, g, b, _ in [cmap(i) for i in range(len(bounds) - 1)]]


def render_colorbar_png(bounds, cmap_name, label, width_in=10, height_in=1.0):
    cmap = plt.cm.get_cmap(cmap_name, len(bounds) - 1)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal", ticks=bounds)
    cbar.set_label(label, fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def to_b64(png_bytes):
    return base64.b64encode(png_bytes).decode("ascii")


# ═══════════════════════════════════════════════════════════════════════════════
#  Build HTML
# ═══════════════════════════════════════════════════════════════════════════════

# Color bounds
DTMAX_BOUNDS = [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6, 1.0]
DEHD_BOUNDS  = [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04, 0.06]
RESID_BOUNDS = [-0.04, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04]
GDP_BOUNDS   = [-0.004, -0.003, -0.002, -0.001, 0.0, 0.0005]


def build_html(geojson_str, colorbars_b64, color_scales, bounds_js,
               stats_js, table_html, dtmax_summary):

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>\u0394Tmax Validation Explorer — ERA5 vs GCM (2013-2024 vs 2000-2012)</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.min.css"/>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/leaflet.sync@0.2.4/L.Map.Sync.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f5; color: #333; }}
  #header {{ background: #1a1a2e; color: #fff; padding: 12px 20px;
             display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
  #header h1 {{ font-size: 16px; font-weight: 600; white-space: nowrap; }}
  #header select {{ font-size: 14px; padding: 4px 8px; border-radius: 4px;
                    border: 1px solid #555; background: #2a2a4e; color: #fff; }}
  #stats-bar {{ background: #222244; color: #ccc; padding: 8px 20px;
                font-size: 12px; display: flex; gap: 30px; flex-wrap: wrap; }}
  .stat-group {{ }}
  .stat-header {{ color: #8888bb; font-size: 10px; text-transform: uppercase;
                  letter-spacing: 0.5px; margin-bottom: 2px; }}
  .stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .stat-item {{ display: flex; gap: 4px; }}
  .stat-label {{ color: #888; }}
  .stat-val {{ color: #4fc3f7; font-weight: 600; }}
  #map-container {{ display: flex; height: calc(100vh - 300px); min-height: 400px; }}
  .map-col {{ flex: 1; position: relative; border-right: 1px solid #ccc; }}
  .map-col:last-child {{ border-right: none; }}
  .map-col .map-label {{ position: absolute; top: 6px; left: 50%; transform: translateX(-50%);
                         z-index: 1000; background: rgba(0,0,0,0.7); color: #fff;
                         padding: 3px 10px; border-radius: 4px; font-size: 12px;
                         font-weight: 600; pointer-events: none; white-space: nowrap; }}
  .leaflet-map {{ width: 100%; height: 100%; background: #e8e8e8; }}
  #legends {{ display: flex; justify-content: center; gap: 30px;
              padding: 8px 20px; background: #fff; flex-wrap: wrap; }}
  #legends img {{ height: 55px; }}
  #table-section {{ padding: 10px 20px 20px; }}
  #table-section h2 {{ font-size: 14px; margin-bottom: 6px; }}
  .metrics-table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 16px; }}
  .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
  .metrics-table th {{ background: #eee; font-weight: 600; text-align: center; }}
  .metrics-table td:first-child {{ text-align: left; font-weight: 500; }}
  .best {{ background: #e8f5e9; font-weight: 700; }}
  .hover-tooltip {{ font-size: 12px; line-height: 1.5; }}
  .hover-tooltip b {{ font-size: 13px; }}
  .summary-box {{ background: #f0f4ff; border: 1px solid #c0d0ff; border-radius: 6px;
                  padding: 12px 16px; margin-bottom: 16px; font-size: 13px; }}
  .summary-box h3 {{ margin: 0 0 8px; font-size: 14px; }}
  .summary-box table {{ border-collapse: collapse; font-size: 12px; }}
  .summary-box td, .summary-box th {{ padding: 3px 10px; text-align: right; }}
  .summary-box th {{ text-align: left; font-weight: 600; }}
</style>
</head>
<body>

<div id="header">
  <h1>\u0394Tmax Validation — ERA5 vs GCM \u00d7 IFS Slopes \u2192 \u0394EHD \u2192 GDP</h1>
  <label style="font-size:13px;">View:
    <select id="variant-select">
      <optgroup label="\u0394Tmax anomaly maps">
        <option value="dt_local">\u0394Tmax: ERA5 local vs GCM local</option>
        <option value="dt_200km">\u0394Tmax: ERA5 200km vs GCM 200km</option>
      </optgroup>
      <optgroup label="\u0394EHD (IFS slopes applied to \u0394Tmax)">
        <option value="dehd_local" selected>\u0394EHD: V1 (ERA5 local) vs Observed</option>
        <option value="dehd_200km">\u0394EHD: V2 (ERA5 200km) vs Observed</option>
        <option value="dehd_gcm_local">\u0394EHD: V3 (GCM local) vs Observed</option>
        <option value="dehd_gcm_200km">\u0394EHD: V4 (GCM 200km) vs Observed</option>
        <option value="dehd_v5">\u0394EHD: V5 (ERA5 EHD trend) vs Observed</option>
        <option value="dehd_v6">\u0394EHD: V6 (CIL direct) vs Observed</option>
      </optgroup>
      <optgroup label="ERA5 vs GCM \u0394EHD">
        <option value="dehd_era5_gcm_local">\u0394EHD: V1 (ERA5) vs V3 (GCM) — local</option>
        <option value="dehd_era5_gcm_200km">\u0394EHD: V2 (ERA5) vs V4 (GCM) — 200km</option>
      </optgroup>
      <optgroup label="GDP impact (\u03b2 \u00d7 \u0394EHD)">
        <option value="gdp_local">GDP impact: V1 (ERA5 local) vs V3 (GCM local)</option>
        <option value="gdp_200km">GDP impact: V2 (ERA5 200km) vs V4 (GCM 200km)</option>
      </optgroup>
    </select>
  </label>
</div>

<div id="stats-bar">
  <div class="stat-group">
    <div class="stat-header">Left panel metrics (vs observed \u0394EHD)</div>
    <div class="stat-row">
      <div class="stat-item"><span class="stat-label">RMSE:</span>
        <span class="stat-val" id="stat-rmse">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">Bias:</span>
        <span class="stat-val" id="stat-bias">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">r:</span>
        <span class="stat-val" id="stat-r">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">\u03c1:</span>
        <span class="stat-val" id="stat-rho">\u2014</span></div>
    </div>
  </div>
  <div class="stat-group">
    <div class="stat-header">Right panel metrics (vs observed \u0394EHD)</div>
    <div class="stat-row">
      <div class="stat-item"><span class="stat-label">RMSE:</span>
        <span class="stat-val" id="stat-rmse2">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">Bias:</span>
        <span class="stat-val" id="stat-bias2">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">r:</span>
        <span class="stat-val" id="stat-r2">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">\u03c1:</span>
        <span class="stat-val" id="stat-rho2">\u2014</span></div>
    </div>
  </div>
</div>

<div id="map-container">
  <div class="map-col">
    <div class="map-label" id="label-left">Left</div>
    <div id="map-left" class="leaflet-map"></div>
  </div>
  <div class="map-col">
    <div class="map-label" id="label-center">Center</div>
    <div id="map-center" class="leaflet-map"></div>
  </div>
  <div class="map-col">
    <div class="map-label" id="label-right">Difference</div>
    <div id="map-right" class="leaflet-map"></div>
  </div>
</div>

<div id="legends">
  <div><strong style="font-size:11px;">\u0394Tmax (\u00b0C)</strong><br>
    <img src="data:image/png;base64,{colorbars_b64['dtmax']}" alt="\u0394Tmax"></div>
  <div><strong style="font-size:11px;">\u0394EHD</strong><br>
    <img src="data:image/png;base64,{colorbars_b64['dehd']}" alt="\u0394EHD"></div>
  <div><strong style="font-size:11px;">Residual / Diff</strong><br>
    <img src="data:image/png;base64,{colorbars_b64['resid']}" alt="Residual"></div>
  <div><strong style="font-size:11px;">GDP impact (pp)</strong><br>
    <img src="data:image/png;base64,{colorbars_b64['gdp']}" alt="GDP"></div>
</div>

<div id="table-section">
  {dtmax_summary}
  {table_html}
</div>

<script>
    var geodata = {geojson_str};

    // Color scales and bounds for each layer type
    var scales = {{
      dtmax:  {{ bounds: {bounds_js['dtmax']}, colors: {color_scales['dtmax']} }},
      dehd:   {{ bounds: {bounds_js['dehd']},  colors: {color_scales['dehd']} }},
      resid:  {{ bounds: {bounds_js['resid']}, colors: {color_scales['resid']} }},
      gdp:    {{ bounds: {bounds_js['gdp']},   colors: {color_scales['gdp']} }}
    }};

    // View definitions: left, center, right layers + scale type + labels
    var choices = {{
      "dt_local":  {{ left: "dt_era5_local", center: "dt_gcm_local", diff: "dd_dt_local",
                      scaleL: "dtmax", scaleC: "dtmax", scaleR: "resid",
                      labelL: "ERA5 \u0394Tmax (local)", labelC: "GCM \u0394Tmax (local)",
                      labelR: "ERA5 \u2212 GCM", statsL: null, statsR: null }},
      "dt_200km":  {{ left: "dt_era5_200km", center: "dt_gcm_200km", diff: "dd_dt_200km",
                      scaleL: "dtmax", scaleC: "dtmax", scaleR: "resid",
                      labelL: "ERA5 \u0394Tmax (200km)", labelC: "GCM \u0394Tmax (200km)",
                      labelR: "ERA5 \u2212 GCM", statsL: null, statsR: null }},
      "dehd_local":     {{ left: "v1", center: "obs", diff: "r1",
                           scaleL: "dehd", scaleC: "dehd", scaleR: "resid",
                           labelL: "V1: ERA5 local \u00d7 IFS", labelC: "Observed \u0394EHD",
                           labelR: "V1 \u2212 Observed", statsL: "v1", statsR: null }},
      "dehd_200km":     {{ left: "v2", center: "obs", diff: "r2",
                           scaleL: "dehd", scaleC: "dehd", scaleR: "resid",
                           labelL: "V2: ERA5 200km \u00d7 IFS", labelC: "Observed \u0394EHD",
                           labelR: "V2 \u2212 Observed", statsL: "v2", statsR: null }},
      "dehd_gcm_local": {{ left: "v3", center: "obs", diff: "r3",
                           scaleL: "dehd", scaleC: "dehd", scaleR: "resid",
                           labelL: "V3: GCM local \u00d7 IFS", labelC: "Observed \u0394EHD",
                           labelR: "V3 \u2212 Observed", statsL: "v3", statsR: null }},
      "dehd_gcm_200km": {{ left: "v4", center: "obs", diff: "r4",
                           scaleL: "dehd", scaleC: "dehd", scaleR: "resid",
                           labelL: "V4: GCM 200km \u00d7 IFS", labelC: "Observed \u0394EHD",
                           labelR: "V4 \u2212 Observed", statsL: "v4", statsR: null }},
      "dehd_v5":        {{ left: "v5", center: "obs", diff: "r5",
                           scaleL: "dehd", scaleC: "dehd", scaleR: "resid",
                           labelL: "V5: ERA5 EHD trend", labelC: "Observed \u0394EHD",
                           labelR: "V5 \u2212 Observed", statsL: "v5", statsR: null }},
      "dehd_v6":        {{ left: "v6", center: "obs", diff: "r6",
                           scaleL: "dehd", scaleC: "dehd", scaleR: "resid",
                           labelL: "V6: CIL GDPCIR", labelC: "Observed \u0394EHD",
                           labelR: "V6 \u2212 Observed", statsL: "v6", statsR: null }},
      "dehd_era5_gcm_local": {{ left: "v1", center: "v3", diff: "d_v1v3",
                                scaleL: "dehd", scaleC: "dehd", scaleR: "resid",
                                labelL: "V1: ERA5 local", labelC: "V3: GCM local",
                                labelR: "V1 \u2212 V3", statsL: "v1", statsR: "v3" }},
      "dehd_era5_gcm_200km": {{ left: "v2", center: "v4", diff: "d_v2v4",
                                scaleL: "dehd", scaleC: "dehd", scaleR: "resid",
                                labelL: "V2: ERA5 200km", labelC: "V4: GCM 200km",
                                labelR: "V2 \u2212 V4", statsL: "v2", statsR: "v4" }},
      "gdp_local":  {{ left: "gdp_v1", center: "gdp_v3", diff: "gdp_d13",
                        scaleL: "gdp", scaleC: "gdp", scaleR: "resid",
                        labelL: "GDP impact: V1 (ERA5)", labelC: "GDP impact: V3 (GCM)",
                        labelR: "V1 \u2212 V3 (pp)", statsL: "v1", statsR: "v3" }},
      "gdp_200km": {{ left: "gdp_v2", center: "gdp_v4", diff: "gdp_d24",
                       scaleL: "gdp", scaleC: "gdp", scaleR: "resid",
                       labelL: "GDP impact: V2 (ERA5)", labelC: "GDP impact: V4 (GCM)",
                       labelR: "V2 \u2212 V4 (pp)", statsL: "v2", statsR: "v4" }}
    }};

    {stats_js}

    function getColor(val, bounds, colors) {{
      if (val === null || val === undefined || isNaN(val)) return '#cccccc';
      for (var i = 0; i < bounds.length - 1; i++) {{
        if (val < bounds[i+1]) return colors[i];
      }}
      return colors[colors.length - 1];
    }}

    var layerLeft = null, layerCenter = null, layerRight = null;

    function makeMap(divId) {{
      return L.map(divId, {{
        center: [20, 0], zoom: 2, minZoom: 1, maxZoom: 6,
        worldCopyJump: true, attributionControl: false
      }});
    }}

    var mapLeft   = makeMap('map-left');
    var mapCenter = makeMap('map-center');
    var mapRight  = makeMap('map-right');

    mapLeft.sync(mapCenter); mapLeft.sync(mapRight);
    mapCenter.sync(mapLeft); mapCenter.sync(mapRight);
    mapRight.sync(mapLeft);  mapRight.sync(mapCenter);

    function makeTooltip(props, varKey) {{
      var val = props[varKey];
      var name = props.Country || props.iso3;
      if (val === null || val === undefined || isNaN(val)) {{
        return '<div class="hover-tooltip"><b>' + name + '</b><br>No data</div>';
      }}
      var lines = ['<b>' + name + '</b> (' + props.iso3 + ')'];
      lines.push(varKey + ': ' + val.toFixed(5));
      if (props.obs !== undefined && !isNaN(props.obs))
        lines.push('Observed \u0394EHD: ' + props.obs.toFixed(5));
      if (props.dt_era5_local !== undefined && !isNaN(props.dt_era5_local))
        lines.push('ERA5 \u0394T local: ' + props.dt_era5_local.toFixed(3) + '\u00b0C');
      if (props.dt_gcm_local !== undefined && !isNaN(props.dt_gcm_local))
        lines.push('GCM \u0394T local: ' + props.dt_gcm_local.toFixed(3) + '\u00b0C');
      if (props.gdp_v1 !== undefined && !isNaN(props.gdp_v1))
        lines.push('GDP V1: ' + (props.gdp_v1*100).toFixed(3) + ' pp');
      if (props.gdp_v3 !== undefined && !isNaN(props.gdp_v3))
        lines.push('GDP V3: ' + (props.gdp_v3*100).toFixed(3) + ' pp');
      return '<div class="hover-tooltip">' + lines.join('<br>') + '</div>';
    }}

    function createLayer(map, varKey, scaleName) {{
      var sc = scales[scaleName];
      return L.geoJSON(geodata, {{
        style: function(feature) {{
          var val = feature.properties[varKey];
          return {{
            fillColor: getColor(val, sc.bounds, sc.colors),
            weight: 0.5, color: '#444', fillOpacity: 0.85
          }};
        }},
        onEachFeature: function(feature, layer) {{
          layer.on('mouseover', function() {{
            this.setStyle({{ weight: 2, color: '#000' }}); this.bringToFront();
          }});
          layer.on('mouseout', function() {{
            this.setStyle({{ weight: 0.5, color: '#444' }});
          }});
          layer.bindTooltip(function() {{
            return makeTooltip(feature.properties, varKey);
          }}, {{ sticky: true }});
        }}
      }}).addTo(map);
    }}

    function updateStats(key, prefix) {{
      var s = allStats[key];
      if (!s) {{
        document.getElementById('stat-' + prefix + 'rmse').textContent = '\u2014';
        document.getElementById('stat-' + prefix + 'bias').textContent = '\u2014';
        document.getElementById('stat-' + prefix + 'r').textContent = '\u2014';
        document.getElementById('stat-' + prefix + 'rho').textContent = '\u2014';
        return;
      }}
      document.getElementById('stat-' + prefix + 'rmse').textContent = s.rmse;
      document.getElementById('stat-' + prefix + 'bias').textContent = s.bias;
      document.getElementById('stat-' + prefix + 'r').textContent = s.r;
      document.getElementById('stat-' + prefix + 'rho').textContent = s.rho;
    }}

    function updateMaps(choice) {{
      var c = choices[choice];
      if (!c) return;

      if (layerLeft) mapLeft.removeLayer(layerLeft);
      if (layerCenter) mapCenter.removeLayer(layerCenter);
      if (layerRight) mapRight.removeLayer(layerRight);

      layerLeft = createLayer(mapLeft, c.left, c.scaleL);
      layerCenter = createLayer(mapCenter, c.center, c.scaleC);
      layerRight = createLayer(mapRight, c.diff, c.scaleR);

      document.getElementById('label-left').textContent = c.labelL;
      document.getElementById('label-center').textContent = c.labelC;
      document.getElementById('label-right').textContent = c.labelR;

      updateStats(c.statsL, '');
      updateStats(c.statsR, 'rmse2'.replace('rmse2','').length ? '' : '');
      // Update right-side stats
      var s2 = allStats[c.statsR];
      if (s2) {{
        document.getElementById('stat-rmse2').textContent = s2.rmse;
        document.getElementById('stat-bias2').textContent = s2.bias;
        document.getElementById('stat-r2').textContent = s2.r;
        document.getElementById('stat-rho2').textContent = s2.rho;
      }} else {{
        document.getElementById('stat-rmse2').textContent = '\u2014';
        document.getElementById('stat-bias2').textContent = '\u2014';
        document.getElementById('stat-r2').textContent = '\u2014';
        document.getElementById('stat-rho2').textContent = '\u2014';
      }}
    }}

    document.getElementById('variant-select').addEventListener('change', function() {{
      updateMaps(this.value);
    }});

    updateMaps('dehd_local');

    setTimeout(function() {{
      mapLeft.invalidateSize(); mapCenter.invalidateSize(); mapRight.invalidateSize();
    }}, 200);
</script>
</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    wall_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ΔTmax Validation Dashboard")
    print(f"  Train: {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}")
    print(f"  Validate: {VAL_YEARS[0]}-{VAL_YEARS[-1]}")
    print("=" * 70)

    # ── Load IFS slopes ──
    rr_local, rr_200km, ifs_lats, ifs_lons = load_ifs_slopes()

    # ── Compute ΔTmax fields ──
    dt_era5_local, dt_era5_200km = compute_era5_dtmax(ifs_lats, ifs_lons)
    dt_gcm_local, dt_gcm_200km = compute_gcm_dtmax(ifs_lats, ifs_lons)

    # ΔTmax cross-correlation
    print("\n" + "="*70)
    print("ERA5 vs GCM ΔTmax correlation")
    print("="*70)
    for label, e, g in [("local", dt_era5_local, dt_gcm_local),
                         ("200km", dt_era5_200km, dt_gcm_200km)]:
        mask = np.isfinite(e) & np.isfinite(g)
        r_p = pearsonr(e[mask], g[mask])[0]
        r_s = spearmanr(e[mask], g[mask]).correlation
        print(f"  {label}: n={mask.sum():,}, r={r_p:.3f}, ρ={r_s:.3f}, "
              f"ERA5 mean={np.mean(e[mask]):.3f}, GCM mean={np.mean(g[mask]):.3f}")

    # ── Compute ΔEHD from ΔTmax × IFS slopes ──
    observed, ehd_train_025, ehd_train_stack_01 = compute_observed_dehd(ifs_lats, ifs_lons)

    print("\nApplying IFS slopes to ΔTmax...")
    dehd_v1 = compute_slope_dehd(ehd_train_025, rr_local, dt_era5_local)
    dehd_v2 = compute_slope_dehd(ehd_train_025, rr_200km, dt_era5_200km)
    dehd_v3 = compute_slope_dehd(ehd_train_025, rr_local, dt_gcm_local)
    dehd_v4 = compute_slope_dehd(ehd_train_025, rr_200km, dt_gcm_200km)
    for label, d in [("V1 ERA5×local", dehd_v1), ("V2 ERA5×200km", dehd_v2),
                      ("V3 GCM×local", dehd_v3), ("V4 GCM×200km", dehd_v4)]:
        print(f"  {label}: mean ΔEHD={np.nanmean(d):.4f}")

    # V5/V6 direct methods
    dehd_v5 = compute_v5_dehd(ehd_train_stack_01, ifs_lats, ifs_lons)
    dehd_v6 = compute_v6_dehd(ifs_lats, ifs_lons)

    # ── Pixel-level validation metrics ──
    print("\n" + "="*70)
    print("Pixel-level validation (predicted ΔEHD vs observed)")
    print("="*70)
    variant_keys = ["v1", "v2", "v3", "v4", "v5", "v6"]
    variant_labels = {
        "v1": "V1: ERA5 local × IFS local",
        "v2": "V2: ERA5 200km × IFS 200km",
        "v3": "V3: GCM local × IFS local",
        "v4": "V4: GCM 200km × IFS 200km",
        "v5": "V5: ERA5 EHD trend",
        "v6": "V6: CIL GDPCIR (ECS-wt)",
    }
    dehd_grids = {"v1": dehd_v1, "v2": dehd_v2, "v3": dehd_v3,
                  "v4": dehd_v4, "v5": dehd_v5, "v6": dehd_v6}

    pixel_metrics = {}
    for k in variant_keys:
        pixel_metrics[k] = compute_metrics(dehd_grids[k], observed)
        m = pixel_metrics[k]
        print(f"  {variant_labels[k]}: RMSE={m['rmse']:.5f}, bias={m['bias']:.5f}, "
              f"r={m['r']:.3f}, ρ={m['rho']:.3f}, n={m['n']:,}")

    # ── Load regression beta ──
    regr = pd.read_csv(str(REGR_FILE))
    row = regr[
        (regr["spec"] == "linear_ctrl_baseline") &
        (regr["variable"] == "heat_freq_weighted") &
        (regr["se_type"] == "conley_500km")
    ]
    beta = float(row.iloc[0]["beta"])
    print(f"\n  β (Conley 500km) = {beta:.6f}")

    # ── Aggregate to admin1 entities for choropleth ──
    print("\nAggregating to admin1 entities...")
    settlement = load_settlement()
    combined_raster, entities, n_entities = load_admin1_raster()

    # Interpolate all grids to 0.1° and aggregate
    grid_keys = (["obs"] + variant_keys +
                 ["dt_era5_local", "dt_era5_200km", "dt_gcm_local", "dt_gcm_200km"])
    grid_data = {"obs": observed}
    grid_data.update(dehd_grids)
    grid_data["dt_era5_local"] = dt_era5_local
    grid_data["dt_era5_200km"] = dt_era5_200km
    grid_data["dt_gcm_local"] = dt_gcm_local
    grid_data["dt_gcm_200km"] = dt_gcm_200km

    entity_vals = {}
    for k in grid_keys:
        g01 = interp_025_to_01(grid_data[k], ifs_lats, ifs_lons)
        entity_vals[k] = agg_to_entities(g01, combined_raster, settlement, n_entities)
        nv = np.isfinite(entity_vals[k]).sum()
        print(f"  {k}: {nv:,} valid entities")

    # Residuals (predicted - observed)
    for i, k in enumerate(variant_keys, 1):
        entity_vals[f"r{i}"] = entity_vals[k] - entity_vals["obs"]

    # Inter-method diffs
    entity_vals["d_v1v3"] = entity_vals["v1"] - entity_vals["v3"]
    entity_vals["d_v2v4"] = entity_vals["v2"] - entity_vals["v4"]
    entity_vals["dd_dt_local"] = entity_vals["dt_era5_local"] - entity_vals["dt_gcm_local"]
    entity_vals["dd_dt_200km"] = entity_vals["dt_era5_200km"] - entity_vals["dt_gcm_200km"]

    # GDP impact = beta × ΔEHD
    for k in variant_keys:
        entity_vals[f"gdp_{k}"] = beta * entity_vals[k]
    entity_vals["gdp_d13"] = entity_vals["gdp_v1"] - entity_vals["gdp_v3"]
    entity_vals["gdp_d24"] = entity_vals["gdp_v2"] - entity_vals["gdp_v4"]

    # ── Build country-level GeoJSON ──
    print("\nBuilding country-level choropleth...")
    admin1_gdf = gpd.read_file(str(ADMIN1_GPKG))

    ent_df = entities[["GID_nmbr", "entity_idx"]].copy()
    all_val_keys = list(entity_vals.keys())
    for k in all_val_keys:
        ent_df[k] = entity_vals[k]

    admin1_merged = admin1_gdf.merge(ent_df, on="GID_nmbr", how="inner")
    print(f"  Joined: {len(admin1_merged):,} admin1 polygons")

    country_vals = admin1_merged.groupby("iso3")[all_val_keys].mean()
    country_geom = admin1_merged.dissolve(by="iso3").geometry
    choropleth_gdf = gpd.GeoDataFrame(
        country_vals.join(country_geom), geometry="geometry"
    ).reset_index()
    print(f"  {len(choropleth_gdf):,} countries")

    # Country-level metrics
    print("\nCountry-level validation metrics...")
    country_metrics = {}
    for k in variant_keys:
        pred = choropleth_gdf[k].values
        obs_c = choropleth_gdf["obs"].values
        country_metrics[k] = compute_metrics(pred, obs_c)
        m = country_metrics[k]
        print(f"  {variant_labels[k]}: RMSE={m['rmse']:.5f}, r={m['r']:.3f}")

    # Simplify + add names
    choropleth_gdf["geometry"] = choropleth_gdf.geometry.simplify(0.1, preserve_topology=True)
    country_names = admin1_gdf.drop_duplicates("iso3").set_index("iso3")["Country"]
    choropleth_gdf["Country"] = choropleth_gdf["iso3"].map(country_names)

    for col in all_val_keys:
        choropleth_gdf[col] = choropleth_gdf[col].round(6)

    keep_cols = ["iso3", "Country", "geometry"] + all_val_keys
    geojson_str = choropleth_gdf[keep_cols].to_json()
    print(f"  GeoJSON: {len(geojson_str)/1024/1024:.1f} MB")

    # ── Stats JS ──
    all_stats = {}
    for k in variant_keys:
        m = pixel_metrics[k]
        all_stats[k] = {
            "rmse": f"{m['rmse']:.5f}", "bias": f"{m['bias']:.5f}",
            "r": f"{m['r']:.3f}", "rho": f"{m['rho']:.3f}",
        }
    stats_js = f"var allStats = {json.dumps(all_stats)};"

    # ── ΔTmax summary box ──
    dtmax_summary = '<div class="summary-box">'
    dtmax_summary += '<h3>ΔTmax Summary (validation period anomaly)</h3>'
    dtmax_summary += '<table>'
    dtmax_summary += '<tr><th>Source</th><th>Scale</th><th>Mean ΔT (°C)</th><th>Median</th><th>Valid pixels</th></tr>'
    for label, scale, arr in [("ERA5", "local", dt_era5_local), ("ERA5", "200km", dt_era5_200km),
                               ("GCM", "local", dt_gcm_local), ("GCM", "200km", dt_gcm_200km)]:
        nv = np.isfinite(arr).sum()
        dtmax_summary += (f'<tr><td style="text-align:left">{label}</td>'
                         f'<td style="text-align:left">{scale}</td>'
                         f'<td>{np.nanmean(arr):.4f}</td>'
                         f'<td>{np.nanmedian(arr):.4f}</td>'
                         f'<td>{nv:,}</td></tr>')
    # Fix the label bug
    dtmax_summary = '<div class="summary-box">'
    dtmax_summary += '<h3>\u0394Tmax Summary (2013-2024 vs 2000-2012 anomaly)</h3>'
    dtmax_summary += '<table>'
    dtmax_summary += '<tr><th>Source</th><th>Scale</th><th>Mean \u0394T (\u00b0C)</th><th>Median</th><th>Valid pixels</th></tr>'
    for label, scale, arr in [("ERA5", "local", dt_era5_local), ("ERA5", "200km", dt_era5_200km),
                               ("GCM", "local", dt_gcm_local), ("GCM", "200km", dt_gcm_200km)]:
        nv = np.isfinite(arr).sum()
        dtmax_summary += (f'<tr><td style="text-align:left">{label}</td>'
                         f'<td style="text-align:left">{scale}</td>'
                         f'<td>{np.nanmean(arr):.4f}</td>'
                         f'<td>{np.nanmedian(arr):.4f}</td>'
                         f'<td>{nv:,}</td></tr>')
    dtmax_summary += '</table>'
    dtmax_summary += (f'<p style="font-size:11px;color:#666;margin-top:6px;">'
                     f'ERA5: OLS trend on 2000-2012, extrapolated {VAL_MID-TRAIN_MID:.1f} years to {VAL_MID}. '
                     f'GCM: ECS-weighted mean of top 8 CMIP6 models. '
                     f'200km = Gaussian smoothing (\u03c3\u2248{SIGMA_025:.1f} cells). '
                     f'\u03b2 = {beta:.4f} (Conley 500km).</p>')
    dtmax_summary += '</div>'

    # ── Metrics table ──
    rows = ['<thead><tr>'
            '<th rowspan="2">Method</th>'
            '<th colspan="4">Pixel-level (0.25\u00b0)</th>'
            '<th colspan="3">Country-level</th>'
            '<th>GDP impact</th>'
            '</tr><tr>'
            '<th>RMSE</th><th>Bias</th><th>r</th><th>\u03c1</th>'
            '<th>RMSE</th><th>Bias</th><th>r</th>'
            '<th>GDP-wt \u0394EHD</th>'
            '</tr></thead><tbody>']

    best_px = min(variant_keys, key=lambda k: pixel_metrics[k]["rmse"])
    best_co = min(variant_keys, key=lambda k: country_metrics[k]["rmse"])

    # GDP-weighted ΔEHD per variant (using entity-level panel data)
    panel = pd.read_parquet(str(PANEL_FILE))
    gdp_mean = panel.groupby("GID_2")["gdp_per_capita"].mean()
    gdp_arr = entities["GID_2"].map(gdp_mean).values.astype(np.float64)

    for k in variant_keys:
        pm = pixel_metrics[k]
        cm = country_metrics[k]
        pcls = ' class="best"' if k == best_px else ''
        ccls = ' class="best"' if k == best_co else ''

        # GDP-weighted ΔEHD
        ev = entity_vals[k]
        mask = np.isfinite(ev) & np.isfinite(gdp_arr) & (gdp_arr > 0)
        gdp_wt_dehd = float(np.average(ev[mask], weights=gdp_arr[mask])) if mask.sum() > 0 else np.nan
        gdp_impact = beta * gdp_wt_dehd

        rows.append(
            f'<tr><td>{variant_labels[k]}</td>'
            f'<td{pcls}>{pm["rmse"]:.5f}</td>'
            f'<td>{pm["bias"]:.5f}</td>'
            f'<td>{pm["r"]:.3f}</td>'
            f'<td>{pm["rho"]:.3f}</td>'
            f'<td{ccls}>{cm["rmse"]:.5f}</td>'
            f'<td>{cm["bias"]:.5f}</td>'
            f'<td>{cm["r"]:.3f}</td>'
            f'<td>{gdp_wt_dehd:.4f} ({gdp_impact*100:.3f} pp)</td>'
            f'</tr>')

    rows.append('</tbody>')
    table_html = '<table class="metrics-table">' + "\n".join(rows) + "</table>"
    table_html += ('<p style="font-size:11px;color:#666;margin-top:4px;">'
                   'Green = lowest RMSE. GDP impact = \u03b2 \u00d7 GDP-weighted \u0394EHD. '
                   'Observed \u0394EHD from ERA5-Land. '
                   'IFS slopes from ECMWF 2001-2020 reforecasts.</p>')

    # ── Render colorbars ──
    colorbars_b64 = {
        'dtmax': to_b64(render_colorbar_png(DTMAX_BOUNDS, "RdYlBu_r", "\u0394Tmax (\u00b0C)")),
        'dehd':  to_b64(render_colorbar_png(DEHD_BOUNDS, "RdYlBu_r", "\u0394EHD (fraction)")),
        'resid': to_b64(render_colorbar_png(RESID_BOUNDS, "RdBu_r", "Residual / Difference")),
        'gdp':   to_b64(render_colorbar_png(GDP_BOUNDS, "RdYlGn_r", "GDP impact (fraction)")),
    }
    color_scales = {
        'dtmax': json.dumps(get_bin_colors_hex(DTMAX_BOUNDS, "RdYlBu_r")),
        'dehd':  json.dumps(get_bin_colors_hex(DEHD_BOUNDS, "RdYlBu_r")),
        'resid': json.dumps(get_bin_colors_hex(RESID_BOUNDS, "RdBu_r")),
        'gdp':   json.dumps(get_bin_colors_hex(GDP_BOUNDS, "RdYlGn_r")),
    }
    bounds_js = {
        'dtmax': json.dumps(DTMAX_BOUNDS),
        'dehd':  json.dumps(DEHD_BOUNDS),
        'resid': json.dumps(RESID_BOUNDS),
        'gdp':   json.dumps(GDP_BOUNDS),
    }

    # ── Build HTML ──
    print("\nBuilding HTML...")
    html = build_html(geojson_str, colorbars_b64, color_scales, bounds_js,
                      stats_js, table_html, dtmax_summary)

    out_html = OUT_DIR / "dtmax_validation_explorer.html"
    with open(str(out_html), "w") as f:
        f.write(html)

    size_mb = os.path.getsize(str(out_html)) / (1024 * 1024)
    elapsed = time.time() - wall_start
    print(f"\nSaved: {out_html}  ({size_mb:.1f} MB)")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Also save the ΔTmax and ΔEHD grids comparison PNG ──
    print("\nGenerating ΔTmax comparison figure...")
    from matplotlib.colors import TwoSlopeNorm
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    lon_mesh, lat_mesh = np.meshgrid(ifs_lons, ifs_lats)
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=1.5)

    for ax, grid, title in zip(axes.ravel(),
            [dt_era5_local, dt_era5_200km, dt_gcm_local, dt_gcm_200km],
            ["ERA5 ΔTmax (local)", "ERA5 ΔTmax (200km)",
             "GCM ΔTmax (local)", "GCM ΔTmax (200km)"]):
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
        ax.set_facecolor("#e8e8e8")
        nv = np.isfinite(grid).sum()
        pcm = ax.pcolormesh(lon_mesh, lat_mesh, grid,
                            norm=norm, cmap="RdYlBu_r", shading="auto", rasterized=True)
        ax.set_title(f"{title}\nmean={np.nanmean(grid):.3f}°C, valid={nv:,}", fontsize=11)

    fig.colorbar(pcm, ax=axes.ravel().tolist(), orientation="horizontal",
                 fraction=0.025, pad=0.06, aspect=50, label="ΔTmax (°C)")
    fig.suptitle("ΔTmax anomaly: ERA5 vs GCM (2013-2024 vs 2000-2012)\n"
                 "Applied to IFS reforecast slopes → ΔEHD",
                 fontsize=14, fontweight="bold")
    fig.savefig(str(OUT_DIR / "dtmax_comparison.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'dtmax_comparison.png'}")


if __name__ == "__main__":
    main()
