#!/usr/bin/env python -u
"""
Generate an interactive HTML dashboard for ΔEHD validation results.

Shows observed vs predicted ΔEHD anomaly (2013-2024 mean − 2000-2012 mean)
as country-level Leaflet choropleths with dropdown to switch between methods
and view residuals. Includes validation metrics and comparison tables.

Output: projections/output/validation_explorer.html

Usage:
  cd ~/projects/macro/extreme_heat/biodiversity_interactions
  /turbo/mgoldklang/pyenvs/peg_nov_24/bin/python projections/scripts/generate_validation_dashboard.py
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
    _load_era5land_tmax_summer_year,
    ECS_VALUES, CIL_ECS, CIL_EHD_DIR, TASMAX_DIR, SIGMA_025,
    ERA5_LATS, ERA5_LONS, PANEL_FILE, OUT_DIR,
)
from bootstrap_impact_comparison import load_admin1_raster

PROJ_DIR   = Path(__file__).resolve().parent.parent
ROOT_DIR   = PROJ_DIR.parent
PARENT_DIR = ROOT_DIR.parent

TRAIN_YEARS = list(range(2000, 2013))
VAL_YEARS   = list(range(2013, 2025))
TRAIN_MID   = np.mean(TRAIN_YEARS)
VAL_MID     = np.mean(VAL_YEARS)
CLIM_SSP    = "ssp245"

ADMIN1_GPKG = PARENT_DIR / "polyg_adm1_gdp_perCapita_1990_2022 (1).gpkg"

# Color bounds for ΔEHD anomaly and residuals
DEHD_BOUNDS = [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04, 0.06]
RESID_BOUNDS = [-0.04, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04]


# ═══════════════════════════════════════════════════════════════════════════════
#  Compute all ΔEHD fields (same as validate_ehd_methods.py, condensed)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_dehd(ifs_lats, ifs_lons, rr_local, rr_200km):
    """Compute observed and 6 predicted ΔEHD at 0.25°. Returns dict."""
    n_lat, n_lon = len(ifs_lats), len(ifs_lons)

    # ── Observed EHD ──
    print("\nLoading EHD for train and validation periods...")
    train_parts, val_parts = [], []
    for yr in TRAIN_YEARS:
        arr = load_ehd_year_01(yr)
        if arr is not None:
            train_parts.append(arr)
    for yr in VAL_YEARS:
        arr = load_ehd_year_01(yr)
        if arr is not None:
            val_parts.append(arr)

    train_stack_01 = np.stack(train_parts, axis=0)
    train_mean_01 = np.nanmean(train_stack_01, axis=0).astype(np.float32)
    val_mean_01 = np.nanmean(np.stack(val_parts, axis=0), axis=0).astype(np.float32)

    ehd_train_025 = coarsen_01_to_025(train_mean_01, ifs_lats, ifs_lons)
    ehd_val_025 = coarsen_01_to_025(val_mean_01, ifs_lats, ifs_lons)
    observed = (ehd_val_025 - ehd_train_025).astype(np.float32)
    print(f"  Observed ΔEHD: mean={np.nanmean(observed):.4f}")

    # Coarsen train stack to 0.25° for V5
    ehd_train_stack_025 = np.full((len(TRAIN_YEARS), n_lat, n_lon), np.nan, dtype=np.float32)
    for i in range(len(TRAIN_YEARS)):
        ehd_train_stack_025[i] = coarsen_01_to_025(train_stack_01[i], ifs_lats, ifs_lons)

    # ── V1/V2: ERA5-Land Tmax trend × IFS slopes ──
    print("\nComputing ERA5-Land Tmax ΔT (train 2000-2012 → 2018.5)...")
    with Pool(min(8, len(TRAIN_YEARS))) as pool:
        results = pool.map(_load_era5land_tmax_summer_year, TRAIN_YEARS)

    tmax_025 = np.full((len(TRAIN_YEARS), n_lat, n_lon), np.nan, dtype=np.float32)
    for i, (yr, arr_01) in enumerate(sorted(results, key=lambda x: x[0])):
        if arr_01 is not None:
            tmax_025[i] = coarsen_01_to_025(arr_01, ifs_lats, ifs_lons)

    years_arr = np.array(TRAIN_YEARS, dtype=np.float32)
    years_3d = np.broadcast_to(years_arr[:, None, None], tmax_025.shape)
    era5_tmax_slope = ols_slope_vectorized(years_3d, tmax_025)

    dt_era5_local = (era5_tmax_slope * (VAL_MID - TRAIN_MID)).astype(np.float32)
    dt_era5_200km = apply_gaussian_filter(dt_era5_local, sigma=SIGMA_025)
    print(f"  ERA5 ΔT local: mean={np.nanmean(dt_era5_local):.3f}°C")

    dehd_v1 = compute_slope_dehd(ehd_train_025, rr_local, dt_era5_local)
    dehd_v2 = compute_slope_dehd(ehd_train_025, rr_200km, dt_era5_200km)

    # ── V3/V4: GCM ΔT × IFS slopes ──
    print("\nComputing GCM ΔT for validation period...")
    models = sorted(ECS_VALUES.keys())
    available = [m for m in models if (TASMAX_DIR / m / "ssp245").exists()]
    weights_all = compute_ecs_weights(available, ECS_VALUES)
    ranked = sorted(available, key=lambda m: weights_all[m], reverse=True)
    available = ranked[:8]
    weights = compute_ecs_weights(available, ECS_VALUES)

    dt_wsum = np.zeros((n_lat, n_lon), dtype=np.float64)
    w_sum = np.zeros((n_lat, n_lon), dtype=np.float64)

    for mi, model in enumerate(available):
        t0 = time.time()
        print(f"  [{mi+1}/{len(available)}] {model}...", end="", flush=True)
        base_parts, val_gcm_parts = [], []
        for yr in TRAIN_YEARS:
            scen = "historical" if yr <= 2014 else CLIM_SSP
            r = cmip6_summer_mean_year(model, scen, yr)
            if r is not None:
                base_parts.append(r[2])
        if not base_parts:
            print(" skip")
            continue
        lats_src, lons_src, _ = r
        baseline = np.nanmean(np.stack(base_parts, axis=0), axis=0)
        for yr in VAL_YEARS:
            scen = "historical" if yr <= 2014 else CLIM_SSP
            r = cmip6_summer_mean_year(model, scen, yr)
            if r is not None:
                val_gcm_parts.append(r[2])
        if not val_gcm_parts:
            print(" skip")
            continue
        val_mean = np.nanmean(np.stack(val_gcm_parts, axis=0), axis=0)
        dt_model = (val_mean - baseline).astype(np.float32)
        dt_ifs = regrid_gcm_to_ifs(dt_model, lats_src, lons_src, ifs_lats, ifs_lons)
        w = weights[model]
        valid = np.isfinite(dt_ifs)
        dt_wsum[valid] += w * dt_ifs[valid]
        w_sum[valid] += w
        print(f" ΔT={np.nanmean(dt_ifs):.2f}°C ({time.time()-t0:.0f}s)")

    with np.errstate(divide="ignore", invalid="ignore"):
        dt_gcm_local = np.where(w_sum > 0, dt_wsum / w_sum, np.nan).astype(np.float32)
    dt_gcm_200km = apply_gaussian_filter(dt_gcm_local, sigma=SIGMA_025)

    dehd_v3 = compute_slope_dehd(ehd_train_025, rr_local, dt_gcm_local)
    dehd_v4 = compute_slope_dehd(ehd_train_025, rr_200km, dt_gcm_200km)

    # ── V5: ERA5 EHD trend ──
    print("\nComputing ERA5 EHD trend (V5)...")
    years_3d_t = np.broadcast_to(years_arr[:, None, None], ehd_train_stack_025.shape)
    ehd_slope = ols_slope_vectorized(years_3d_t, ehd_train_stack_025)
    ehd_base_train = np.nanmean(ehd_train_stack_025, axis=0)
    dehd_v5 = (ehd_slope * (VAL_MID - TRAIN_MID)).astype(np.float32)
    new_ehd = np.clip(ehd_base_train + dehd_v5, 0.0, 1.0)
    dehd_v5 = (new_ehd - ehd_base_train).astype(np.float32)
    print(f"  V5 ΔEHD: mean={np.nanmean(dehd_v5):.4f}")

    # ── V6: CIL GDPCIR ──
    print("\nComputing CIL GDPCIR ΔEHD (V6)...")
    cil_models = sorted(CIL_ECS.keys())
    cil_available = [m for m in cil_models
                     if (CIL_EHD_DIR / m / "ehd_historical.nc").exists()
                     and (CIL_EHD_DIR / m / "ehd_ssp245.nc").exists()]
    cil_weights = compute_ecs_weights(cil_available, CIL_ECS)

    dehd_wsum = np.zeros((n_lat, n_lon), dtype=np.float64)
    cil_w = np.zeros((n_lat, n_lon), dtype=np.float64)

    for model in cil_available:
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
        w = cil_weights[model]
        v = np.isfinite(dehd_ifs)
        dehd_wsum[v] += w * dehd_ifs[v]
        cil_w[v] += w

    with np.errstate(divide="ignore", invalid="ignore"):
        dehd_v6 = np.where(cil_w > 0, dehd_wsum / cil_w, np.nan).astype(np.float32)
    print(f"  V6 ΔEHD: mean={np.nanmean(dehd_v6):.4f}")

    return {
        "observed": observed,
        "v1": dehd_v1, "v2": dehd_v2,
        "v3": dehd_v3, "v4": dehd_v4,
        "v5": dehd_v5, "v6": dehd_v6,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred, obs):
    """RMSE, bias, Pearson r, Spearman ρ for arrays."""
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

def build_html(geojson_str, cbar_dehd_b64, cbar_resid_b64,
               dehd_colors_js, resid_colors_js, dehd_bounds_js, resid_bounds_js,
               stats_js, metrics_table_html):

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ΔEHD Validation Explorer — Historical Out-of-Sample (2013-2024 vs 2000-2012)</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.min.css"/>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/leaflet.sync@0.2.4/L.Map.Sync.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f5; color: #333; }}
  #header {{ background: #1a2e1a; color: #fff; padding: 12px 20px;
             display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
  #header h1 {{ font-size: 16px; font-weight: 600; white-space: nowrap; }}
  #header select {{ font-size: 14px; padding: 4px 8px; border-radius: 4px;
                    border: 1px solid #555; background: #2a4e2a; color: #fff; }}
  #stats-bar {{ background: #224422; color: #ccc; padding: 6px 20px;
                font-size: 12px; display: flex; gap: 30px; flex-wrap: wrap; }}
  .stat-group {{ }}
  .stat-header {{ color: #88bb88; font-size: 10px; text-transform: uppercase;
                  letter-spacing: 0.5px; margin-bottom: 2px; }}
  .stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .stat-item {{ display: flex; gap: 4px; }}
  .stat-label {{ color: #888; }}
  .stat-val {{ color: #4fc37f; font-weight: 600; }}
  #map-container {{ display: flex; height: calc(100vh - 280px); min-height: 400px; }}
  .map-col {{ flex: 1; position: relative; border-right: 1px solid #ccc; }}
  .map-col:last-child {{ border-right: none; }}
  .map-col .map-label {{ position: absolute; top: 6px; left: 50%; transform: translateX(-50%);
                         z-index: 1000; background: rgba(0,0,0,0.7); color: #fff;
                         padding: 3px 10px; border-radius: 4px; font-size: 12px;
                         font-weight: 600; pointer-events: none; white-space: nowrap; }}
  .leaflet-map {{ width: 100%; height: 100%; background: #e8e8e8; }}
  #legends {{ display: flex; justify-content: center; gap: 40px;
              padding: 8px 20px; background: #fff; }}
  #legends img {{ height: 60px; }}
  #table-section {{ padding: 10px 20px 20px; }}
  #table-section h2 {{ font-size: 14px; margin-bottom: 6px; }}
  .metrics-table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
  .metrics-table th {{ background: #eee; font-weight: 600; text-align: center; }}
  .metrics-table td:first-child {{ text-align: left; font-weight: 500; }}
  .best {{ background: #e8f5e9; font-weight: 700; }}
  .hover-tooltip {{ font-size: 12px; line-height: 1.5; }}
  .hover-tooltip b {{ font-size: 13px; }}
</style>
</head>
<body>

<div id="header">
  <h1>\u0394EHD Validation Explorer — Historical Out-of-Sample</h1>
  <label style="font-size:13px;">Comparison:
    <select id="variant-select">
      <optgroup label="Predicted vs Observed">
        <option value="v1_obs">V1 (ERA5 local \u00d7 IFS local) vs Observed</option>
        <option value="v2_obs">V2 (ERA5 200km \u00d7 IFS 200km) vs Observed</option>
        <option value="v3_obs">V3 (GCM local \u00d7 IFS local) vs Observed</option>
        <option value="v4_obs">V4 (GCM 200km \u00d7 IFS 200km) vs Observed</option>
        <option value="v5_obs">V5 (ERA5 EHD trend) vs Observed</option>
        <option value="v6_obs" selected>V6 (CIL GDPCIR) vs Observed</option>
      </optgroup>
      <optgroup label="Method Comparisons">
        <option value="v1_v5">V1 (ERA5 slope) vs V5 (ERA5 trend)</option>
        <option value="v3_v6">V3 (GCM slope) vs V6 (CIL direct)</option>
        <option value="v1_v3">V1 (ERA5 ΔT) vs V3 (GCM ΔT) — local</option>
        <option value="v2_v4">V2 (ERA5 ΔT) vs V4 (GCM ΔT) — 200km</option>
      </optgroup>
    </select>
  </label>
</div>

<div id="stats-bar">
  <div class="stat-group">
    <div class="stat-header">Pixel-level metrics (0.25\u00b0)</div>
    <div class="stat-row">
      <div class="stat-item"><span class="stat-label">RMSE:</span>
        <span class="stat-val" id="stat-rmse">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">Bias:</span>
        <span class="stat-val" id="stat-bias">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">r:</span>
        <span class="stat-val" id="stat-r">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">\u03c1:</span>
        <span class="stat-val" id="stat-rho">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">n:</span>
        <span class="stat-val" id="stat-n">\u2014</span></div>
    </div>
  </div>
  <div class="stat-group">
    <div class="stat-header">Country-level metrics</div>
    <div class="stat-row">
      <div class="stat-item"><span class="stat-label">RMSE:</span>
        <span class="stat-val" id="stat-c-rmse">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">Bias:</span>
        <span class="stat-val" id="stat-c-bias">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">r:</span>
        <span class="stat-val" id="stat-c-r">\u2014</span></div>
    </div>
  </div>
</div>

<div id="map-container">
  <div class="map-col">
    <div class="map-label" id="label-left">Predicted</div>
    <div id="map-left" class="leaflet-map"></div>
  </div>
  <div class="map-col">
    <div class="map-label" id="label-center">Observed / Reference</div>
    <div id="map-center" class="leaflet-map"></div>
  </div>
  <div class="map-col">
    <div class="map-label" id="label-right">Residual (Predicted \u2212 Observed)</div>
    <div id="map-right" class="leaflet-map"></div>
  </div>
</div>

<div id="legends">
  <div><strong style="font-size:11px;">\u0394EHD anomaly</strong><br>
    <img src="data:image/png;base64,{cbar_dehd_b64}" alt="\u0394EHD colorbar"></div>
  <div><strong style="font-size:11px;">Residual</strong><br>
    <img src="data:image/png;base64,{cbar_resid_b64}" alt="Residual colorbar"></div>
</div>

<div id="table-section">
  <h2>Validation Metrics: 6 Methods \u00d7 3 Spatial Scales</h2>
  {metrics_table_html}
</div>

<script>
    var geodata = {geojson_str};

    var choices = {{
      "v1_obs": {{left: "v1", center: "obs", diff: "r1", label: "V1: ERA5 local \u00d7 IFS local", centerLabel: "Observed (ERA5-Land)"}},
      "v2_obs": {{left: "v2", center: "obs", diff: "r2", label: "V2: ERA5 200km \u00d7 IFS 200km", centerLabel: "Observed (ERA5-Land)"}},
      "v3_obs": {{left: "v3", center: "obs", diff: "r3", label: "V3: GCM local \u00d7 IFS local", centerLabel: "Observed (ERA5-Land)"}},
      "v4_obs": {{left: "v4", center: "obs", diff: "r4", label: "V4: GCM 200km \u00d7 IFS 200km", centerLabel: "Observed (ERA5-Land)"}},
      "v5_obs": {{left: "v5", center: "obs", diff: "r5", label: "V5: ERA5 EHD trend", centerLabel: "Observed (ERA5-Land)"}},
      "v6_obs": {{left: "v6", center: "obs", diff: "r6", label: "V6: CIL GDPCIR (ECS-wt)", centerLabel: "Observed (ERA5-Land)"}},
      "v1_v5": {{left: "v1", center: "v5", diff: "d_v1v5", label: "V1: ERA5 slope", centerLabel: "V5: ERA5 trend"}},
      "v3_v6": {{left: "v3", center: "v6", diff: "d_v3v6", label: "V3: GCM slope", centerLabel: "V6: CIL direct"}},
      "v1_v3": {{left: "v1", center: "v3", diff: "d_v1v3", label: "V1: ERA5 \u0394T local", centerLabel: "V3: GCM \u0394T local"}},
      "v2_v4": {{left: "v2", center: "v4", diff: "d_v2v4", label: "V2: ERA5 \u0394T 200km", centerLabel: "V4: GCM \u0394T 200km"}}
    }};

    {stats_js}

    var dehdBounds = {dehd_bounds_js};
    var residBounds = {resid_bounds_js};
    var dehdColors = {dehd_colors_js};
    var residColors = {resid_colors_js};

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

    function makeTooltip(props, varKey, isDiff) {{
      var val = props[varKey];
      var name = props.Country || props.iso3;
      if (val === null || val === undefined || isNaN(val)) {{
        return '<div class="hover-tooltip"><b>' + name + '</b><br>No data</div>';
      }}
      var obsVal = props.obs;
      var obsStr = (obsVal !== null && !isNaN(obsVal)) ? obsVal.toFixed(4) : 'n/a';
      if (isDiff) {{
        return '<div class="hover-tooltip"><b>' + name + '</b> (' + props.iso3 + ')' +
          '<br>Residual: ' + val.toFixed(4) +
          '<br>Observed \u0394EHD: ' + obsStr + '</div>';
      }}
      return '<div class="hover-tooltip"><b>' + name + '</b> (' + props.iso3 + ')' +
        '<br>\u0394EHD: ' + val.toFixed(4) +
        '<br>Observed \u0394EHD: ' + obsStr + '</div>';
    }}

    function createLayer(map, varKey, bounds, colors, isDiff) {{
      return L.geoJSON(geodata, {{
        style: function(feature) {{
          var val = feature.properties[varKey];
          return {{
            fillColor: getColor(val, bounds, colors),
            weight: 0.5, color: '#444', fillOpacity: 0.85
          }};
        }},
        onEachFeature: function(feature, layer) {{
          layer.on('mouseover', function() {{
            this.setStyle({{ weight: 2, color: '#000' }});
            this.bringToFront();
          }});
          layer.on('mouseout', function() {{
            this.setStyle({{ weight: 0.5, color: '#444' }});
          }});
          layer.bindTooltip(function() {{
            return makeTooltip(feature.properties, varKey, isDiff);
          }}, {{ sticky: true }});
        }}
      }}).addTo(map);
    }}

    function updateMaps(choice) {{
      var c = choices[choice];
      if (!c) return;

      if (layerLeft) mapLeft.removeLayer(layerLeft);
      if (layerCenter) mapCenter.removeLayer(layerCenter);
      if (layerRight) mapRight.removeLayer(layerRight);

      layerLeft = createLayer(mapLeft, c.left, dehdBounds, dehdColors, false);
      layerCenter = createLayer(mapCenter, c.center, dehdBounds, dehdColors, false);
      layerRight = createLayer(mapRight, c.diff, residBounds, residColors, true);

      document.getElementById('label-left').textContent = c.label;
      document.getElementById('label-center').textContent = c.centerLabel;
      document.getElementById('label-right').textContent =
        c.diff.startsWith('r') ? 'Residual (Predicted \u2212 Observed)' : 'Difference (Left \u2212 Right)';

      var s = pixelStats[c.left] || {{}};
      document.getElementById('stat-rmse').textContent = s.rmse || '\u2014';
      document.getElementById('stat-bias').textContent = s.bias || '\u2014';
      document.getElementById('stat-r').textContent    = s.r    || '\u2014';
      document.getElementById('stat-rho').textContent  = s.rho  || '\u2014';
      document.getElementById('stat-n').textContent    = s.n    || '\u2014';

      var sc = countryStats[c.left] || {{}};
      document.getElementById('stat-c-rmse').textContent = sc.rmse || '\u2014';
      document.getElementById('stat-c-bias').textContent = sc.bias || '\u2014';
      document.getElementById('stat-c-r').textContent    = sc.r    || '\u2014';
    }}

    document.getElementById('variant-select').addEventListener('change', function() {{
      updateMaps(this.value);
    }});

    updateMaps('v6_obs');

    setTimeout(function() {{
      mapLeft.invalidateSize();
      mapCenter.invalidateSize();
      mapRight.invalidateSize();
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
    print("Generating ΔEHD Validation Explorer Dashboard")
    print("=" * 70)

    # ── Step 1: Compute all ΔEHD grids ────────────────────────────────────
    rr_local, rr_200km, ifs_lats, ifs_lons = load_ifs_slopes()
    grids = compute_all_dehd(ifs_lats, ifs_lons, rr_local, rr_200km)

    variant_keys = ["v1", "v2", "v3", "v4", "v5", "v6"]
    variant_labels = {
        "v1": "V1: ERA5 local × IFS local",
        "v2": "V2: ERA5 200km × IFS 200km",
        "v3": "V3: GCM local × IFS local",
        "v4": "V4: GCM 200km × IFS 200km",
        "v5": "V5: ERA5 EHD trend",
        "v6": "V6: CIL GDPCIR (ECS-wt)",
    }

    # ── Step 2: Pixel-level metrics ───────────────────────────────────────
    print("\nComputing pixel-level metrics...")
    pixel_metrics = {}
    for k in variant_keys:
        pixel_metrics[k] = compute_metrics(grids[k], grids["observed"])
        m = pixel_metrics[k]
        print(f"  {variant_labels[k]}: RMSE={m['rmse']:.5f}, bias={m['bias']:.5f}, "
              f"r={m['r']:.3f}, ρ={m['rho']:.3f}")

    # ── Step 3: Interpolate to 0.1° and aggregate to admin1 entities ──────
    print("\nAggregating to admin1 entities...")
    settlement = load_settlement()
    combined_raster, entities, n_entities = load_admin1_raster()

    grids_01 = {}
    for k in ["observed"] + variant_keys:
        grids_01[k] = interp_025_to_01(grids[k], ifs_lats, ifs_lons)

    entity_vals = {}
    for k in ["observed"] + variant_keys:
        entity_vals[k] = agg_to_entities(
            grids_01[k], combined_raster, settlement, n_entities)
        nv = np.isfinite(entity_vals[k]).sum()
        print(f"  {k}: {nv:,} valid entities")

    # Residuals at entity level
    entity_resids = {}
    for k in variant_keys:
        entity_resids[f"r{k[-1]}"] = entity_vals[k] - entity_vals["observed"]

    # Method-vs-method diffs at entity level
    entity_diffs = {
        "d_v1v5": entity_vals["v1"] - entity_vals["v5"],
        "d_v3v6": entity_vals["v3"] - entity_vals["v6"],
        "d_v1v3": entity_vals["v1"] - entity_vals["v3"],
        "d_v2v4": entity_vals["v2"] - entity_vals["v4"],
    }

    # ── Step 4: Build country-level GeoJSON ───────────────────────────────
    print("\nBuilding country-level choropleth...")
    admin1_gdf = gpd.read_file(str(ADMIN1_GPKG))

    ent_df = entities[["GID_nmbr", "entity_idx"]].copy()
    all_keys = ["observed"] + variant_keys
    resid_keys = list(entity_resids.keys())
    diff_keys = list(entity_diffs.keys())

    for k in all_keys:
        ent_df[k] = entity_vals[k]
    for k in resid_keys:
        ent_df[k] = entity_resids[k]
    for k in diff_keys:
        ent_df[k] = entity_diffs[k]

    # Rename observed for shorter property name
    ent_df = ent_df.rename(columns={"observed": "obs"})
    all_keys[0] = "obs"

    val_cols = all_keys + resid_keys + diff_keys

    admin1_merged = admin1_gdf.merge(ent_df, on="GID_nmbr", how="inner")
    print(f"  Joined: {len(admin1_merged):,} admin1 polygons")

    # Dissolve to country — mean per iso3
    country_vals = admin1_merged.groupby("iso3")[val_cols].mean()
    country_geom = admin1_merged.dissolve(by="iso3").geometry
    choropleth_gdf = gpd.GeoDataFrame(
        country_vals.join(country_geom), geometry="geometry"
    ).reset_index()
    print(f"  {len(choropleth_gdf):,} countries")

    # Country-level metrics
    print("\nCountry-level metrics...")
    country_metrics = {}
    for k in variant_keys:
        pred = choropleth_gdf[k].values
        obs = choropleth_gdf["obs"].values
        country_metrics[k] = compute_metrics(pred, obs)
        m = country_metrics[k]
        print(f"  {variant_labels[k]}: RMSE={m['rmse']:.5f}, bias={m['bias']:.5f}, "
              f"r={m['r']:.3f}, n={m['n']}")

    # Simplify geometry
    choropleth_gdf["geometry"] = choropleth_gdf.geometry.simplify(0.1, preserve_topology=True)

    # Add country name
    country_names = admin1_gdf.drop_duplicates("iso3").set_index("iso3")["Country"]
    choropleth_gdf["Country"] = choropleth_gdf["iso3"].map(country_names)

    # Round for smaller JSON
    for col in val_cols:
        choropleth_gdf[col] = choropleth_gdf[col].round(6)

    keep_cols = ["iso3", "Country", "geometry"] + val_cols
    geojson_str = choropleth_gdf[keep_cols].to_json()
    print(f"  GeoJSON: {len(geojson_str)/1024/1024:.1f} MB")

    # ── Step 5: Build stats JS ────────────────────────────────────────────
    pixel_stats_js = {}
    for k in variant_keys:
        m = pixel_metrics[k]
        pixel_stats_js[k] = {
            "rmse": f"{m['rmse']:.5f}",
            "bias": f"{m['bias']:.5f}",
            "r": f"{m['r']:.3f}",
            "rho": f"{m['rho']:.3f}",
            "n": f"{m['n']:,}",
        }
    country_stats_js = {}
    for k in variant_keys:
        m = country_metrics[k]
        country_stats_js[k] = {
            "rmse": f"{m['rmse']:.5f}",
            "bias": f"{m['bias']:.5f}",
            "r": f"{m['r']:.3f}",
        }
    stats_js = f"var pixelStats = {json.dumps(pixel_stats_js)};\n"
    stats_js += f"    var countryStats = {json.dumps(country_stats_js)};"

    # ── Step 6: Build metrics table ───────────────────────────────────────
    rows = []
    rows.append(
        "<thead><tr>"
        "<th>Method</th>"
        "<th colspan='4'>Pixel-level (0.25°)</th>"
        "<th colspan='3'>Country-level</th>"
        "</tr><tr>"
        "<th></th>"
        "<th>RMSE</th><th>Bias</th><th>r</th><th>ρ</th>"
        "<th>RMSE</th><th>Bias</th><th>r</th>"
        "</tr></thead><tbody>"
    )

    # Find best (lowest RMSE) for highlighting
    best_pixel = min(variant_keys, key=lambda k: pixel_metrics[k]["rmse"])
    best_country = min(variant_keys, key=lambda k: country_metrics[k]["rmse"])

    for k in variant_keys:
        pm = pixel_metrics[k]
        cm = country_metrics[k]
        pcls = ' class="best"' if k == best_pixel else ''
        ccls = ' class="best"' if k == best_country else ''
        rows.append(
            f"<tr>"
            f"<td>{variant_labels[k]}</td>"
            f"<td{pcls}>{pm['rmse']:.5f}</td>"
            f"<td>{pm['bias']:.5f}</td>"
            f"<td>{pm['r']:.3f}</td>"
            f"<td>{pm['rho']:.3f}</td>"
            f"<td{ccls}>{cm['rmse']:.5f}</td>"
            f"<td>{cm['bias']:.5f}</td>"
            f"<td>{cm['r']:.3f}</td>"
            f"</tr>"
        )
    rows.append("</tbody>")
    metrics_table_html = '<table class="metrics-table">' + "\n".join(rows) + "</table>"
    metrics_table_html += (
        '<p style="font-size:11px;color:#666;margin-top:4px;">'
        'Validation period: 2013-2024 vs 2000-2012 baseline. '
        'Green highlight = lowest RMSE (best). '
        'Observed \u0394EHD from ERA5-Land exceedance frequency grids.</p>'
    )

    # ── Step 7: Render colorbars ──────────────────────────────────────────
    dehd_colors = get_bin_colors_hex(DEHD_BOUNDS, "RdYlBu_r")
    resid_colors = get_bin_colors_hex(RESID_BOUNDS, "RdBu_r")

    cbar_dehd_b64 = to_b64(render_colorbar_png(
        DEHD_BOUNDS, "RdYlBu_r", "ΔEHD anomaly (fraction)"))
    cbar_resid_b64 = to_b64(render_colorbar_png(
        RESID_BOUNDS, "RdBu_r", "Residual (predicted − observed)"))

    # ── Step 8: Build HTML ────────────────────────────────────────────────
    print("\nBuilding HTML...")
    html = build_html(
        geojson_str, cbar_dehd_b64, cbar_resid_b64,
        json.dumps(dehd_colors), json.dumps(resid_colors),
        json.dumps(DEHD_BOUNDS), json.dumps(RESID_BOUNDS),
        stats_js, metrics_table_html,
    )

    out_html = OUT_DIR / "validation_explorer.html"
    with open(str(out_html), "w") as f:
        f.write(html)

    size_mb = os.path.getsize(str(out_html)) / (1024 * 1024)
    elapsed = time.time() - wall_start
    print(f"\nSaved: {out_html}  ({size_mb:.1f} MB)")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
