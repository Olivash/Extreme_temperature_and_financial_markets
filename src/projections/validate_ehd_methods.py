#!/usr/bin/env python -u
"""
Historical out-of-sample validation of ΔEHD projection methods.

Train each method's parameters on 2000-2012, predict ΔEHD for 2013-2024,
compare against observed ERA5-Land EHD. Produces pixel-level, entity-level,
and country-level validation metrics plus diagnostic figures.

Period-mean validation:
  Observed ΔEHD = mean(EHD 2013-2024) − mean(EHD 2000-2012) at 0.25° IFS grid.

Year-by-year supplementary:
  Predict EHD(t) for each t ∈ 2013-2024 and compare to observed.

Outputs:
  projections/output/validation_results.csv
  projections/output/validation_summary.csv
  projections/output/validation_maps.png
  projections/output/validation_residual_maps.png
  projections/output/validation_calibration.png
  projections/output/validation_variance_decomposition.png
  projections/output/validation_yearly_rmse.png
  projections/output/validation_yearly_bias.png
  projections/output/validation_yearly_timeseries.png

Usage:
  cd ~/projects/macro/extreme_heat/biodiversity_interactions
  /turbo/mgoldklang/pyenvs/peg_nov_24/bin/python projections/scripts/validate_ehd_methods.py
"""

import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# ─── Import reusable functions from projection script ─────────────────────────
from project_slopes_ehd_maps import (
    load_ifs_slopes, compute_slope_dehd, compute_ecs_weights,
    cmip6_summer_mean_year, regrid_gcm_to_ifs, load_settlement,
    agg_to_entities, interp_025_to_01, coarsen_01_to_025,
    load_ehd_year_01, ols_slope_vectorized, apply_gaussian_filter,
    _load_era5land_tmax_summer_year, regrid_era5_to_ifs,
    load_mixed_raster,
    SLOPES_DIR, ERA5_TMAX_DIR, EHD_DIR_ERA5, TASMAX_DIR, CIL_EHD_DIR,
    CIL_ECS, ECS_VALUES, ERA5_LATS, ERA5_LONS, SIGMA_025,
    PANEL_FILE,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJ_DIR   = Path(__file__).resolve().parent.parent          # projections/
ROOT_DIR   = PROJ_DIR.parent                                  # biodiversity_interactions/
PARENT_DIR = ROOT_DIR.parent                                  # extreme_heat/
OUT_DIR    = PROJ_DIR / "output"

# ─── Validation periods ──────────────────────────────────────────────────────
TRAIN_YEARS = list(range(2000, 2013))   # 2000-2012 (13 years)
VAL_YEARS   = list(range(2013, 2025))   # 2013-2024 (12 years)
TRAIN_MID   = np.mean(TRAIN_YEARS)      # 2006.0
VAL_MID     = np.mean(VAL_YEARS)        # 2018.5

CLIM_SSP = "ssp245"


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_ehd_years_025(years, ifs_lats, ifs_lons):
    """
    Load EHD for specified years at 0.1°, return stack at 0.1° and mean at 0.25°.
    Returns (ehd_mean_025, ehd_stack_01, years_loaded).
    """
    parts_01 = []
    years_loaded = []
    for yr in years:
        arr = load_ehd_year_01(yr)
        if arr is not None:
            parts_01.append(arr)
            years_loaded.append(yr)
    stack_01 = np.stack(parts_01, axis=0)
    mean_01 = np.nanmean(stack_01, axis=0).astype(np.float32)
    mean_025 = coarsen_01_to_025(mean_01, ifs_lats, ifs_lons)
    return mean_025, stack_01, years_loaded


def coarsen_stack_to_025(stack_01, ifs_lats, ifs_lons):
    """Coarsen a (n_years, lat, lon) stack from 0.1° to 0.25°."""
    n_yr = stack_01.shape[0]
    n_lat, n_lon = len(ifs_lats), len(ifs_lons)
    stack_025 = np.full((n_yr, n_lat, n_lon), np.nan, dtype=np.float32)
    for i in range(n_yr):
        stack_025[i] = coarsen_01_to_025(stack_01[i], ifs_lats, ifs_lons)
    return stack_025


# ═══════════════════════════════════════════════════════════════════════════════
#  Predicted ΔEHD for each method (period-mean)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_era5_dt_train(ifs_lats, ifs_lons):
    """
    ERA5-Land Tmax trend fitted on TRAIN_YEARS, extrapolated to VAL_MID.
    Returns (dt_local_025, dt_200km_025) — ΔT at 0.25°.
    """
    print("Computing ERA5-Land Tmax ΔT (train 2000-2012 → 2018.5)...")
    n_yr = len(TRAIN_YEARS)
    n_lat, n_lon = len(ifs_lats), len(ifs_lons)

    with Pool(min(8, n_yr)) as pool:
        results = pool.map(_load_era5land_tmax_summer_year, TRAIN_YEARS)

    tmax_025 = np.full((n_yr, n_lat, n_lon), np.nan, dtype=np.float32)
    for i, (yr, arr_01) in enumerate(sorted(results, key=lambda x: x[0])):
        if arr_01 is not None:
            tmax_025[i] = coarsen_01_to_025(arr_01, ifs_lats, ifs_lons)
        if yr % 5 == 0:
            print(f"    ERA5-Land Tmax {yr}")

    # OLS trend per pixel
    years_arr = np.array(TRAIN_YEARS, dtype=np.float32)
    years_3d = np.broadcast_to(years_arr[:, None, None], tmax_025.shape)
    trend_slope = ols_slope_vectorized(years_3d, tmax_025)

    # ΔT = slope × (VAL_MID − TRAIN_MID)
    dt_local_025 = (trend_slope * (VAL_MID - TRAIN_MID)).astype(np.float32)
    dt_200km_025 = apply_gaussian_filter(dt_local_025, sigma=SIGMA_025)

    print(f"  ERA5 ΔT local:  mean={np.nanmean(dt_local_025):.3f}°C")
    print(f"  ERA5 ΔT 200km:  mean={np.nanmean(dt_200km_025):.3f}°C")
    return dt_local_025, dt_200km_025, trend_slope


def compute_gcm_dt_validation(ifs_lats, ifs_lons):
    """
    GCM ECS-weighted ΔT: mean(tasmax 2013-2024) − mean(tasmax 2000-2012).
    Returns (dt_local_025, dt_200km_025).
    """
    print("Computing GCM ΔT for validation period (2013-2024 vs 2000-2012)...")
    models = sorted(ECS_VALUES.keys())

    available = []
    for m in models:
        p_test = TASMAX_DIR / m / "ssp245"
        if p_test.exists():
            available.append(m)

    # Top models by ECS weight
    weights_all = compute_ecs_weights(available, ECS_VALUES)
    ranked = sorted(available, key=lambda m: weights_all[m], reverse=True)
    MAX_MODELS = 8
    available = ranked[:MAX_MODELS]
    weights = compute_ecs_weights(available, ECS_VALUES)
    print(f"  Using top {MAX_MODELS} models")

    n_lat, n_lon = len(ifs_lats), len(ifs_lons)
    dt_wsum = np.zeros((n_lat, n_lon), dtype=np.float64)
    w_sum = np.zeros((n_lat, n_lon), dtype=np.float64)

    for mi, model in enumerate(available):
        t0 = time.time()
        print(f"  [{mi+1}/{len(available)}] {model} (w={weights[model]:.4f})...",
              end="", flush=True)

        # Baseline: mean(2000-2012), historical for ≤2014
        base_parts = []
        for yr in TRAIN_YEARS:
            scen = "historical" if yr <= 2014 else CLIM_SSP
            r = cmip6_summer_mean_year(model, scen, yr)
            if r is not None:
                base_parts.append(r[2])
        if not base_parts:
            print(" no baseline — skip")
            continue
        lats_src, lons_src, _ = r
        baseline = np.nanmean(np.stack(base_parts, axis=0), axis=0)

        # Validation: mean(2013-2024), historical for 2013-2014, ssp245 for 2015+
        val_parts = []
        for yr in VAL_YEARS:
            scen = "historical" if yr <= 2014 else CLIM_SSP
            r = cmip6_summer_mean_year(model, scen, yr)
            if r is not None:
                val_parts.append(r[2])
        if not val_parts:
            print(" no val years — skip")
            continue

        val_mean = np.nanmean(np.stack(val_parts, axis=0), axis=0)
        dt_model = (val_mean - baseline).astype(np.float32)
        dt_ifs = regrid_gcm_to_ifs(dt_model, lats_src, lons_src, ifs_lats, ifs_lons)

        w = weights[model]
        valid = np.isfinite(dt_ifs)
        dt_wsum[valid] += w * dt_ifs[valid]
        w_sum[valid] += w

        print(f" ΔT mean={np.nanmean(dt_ifs):.2f}°C  ({time.time()-t0:.0f}s)")

    with np.errstate(divide="ignore", invalid="ignore"):
        dt_local_025 = np.where(w_sum > 0, dt_wsum / w_sum, np.nan).astype(np.float32)
    dt_200km_025 = apply_gaussian_filter(dt_local_025, sigma=SIGMA_025)

    print(f"  GCM ΔT local:  mean={np.nanmean(dt_local_025):.3f}°C")
    print(f"  GCM ΔT 200km:  mean={np.nanmean(dt_200km_025):.3f}°C")
    return dt_local_025, dt_200km_025


def compute_gcm_dt_yearly(ifs_lats, ifs_lons):
    """
    Per-year GCM ΔT: for each year t in VAL_YEARS, compute
    tasmax(t) − mean(tasmax TRAIN_YEARS), ECS-weighted.
    Returns dict {year: dt_local_025}.
    """
    print("Computing per-year GCM ΔT for validation years...")
    models = sorted(ECS_VALUES.keys())
    available = []
    for m in models:
        if (TASMAX_DIR / m / "ssp245").exists():
            available.append(m)
    weights_all = compute_ecs_weights(available, ECS_VALUES)
    ranked = sorted(available, key=lambda m: weights_all[m], reverse=True)
    available = ranked[:8]
    weights = compute_ecs_weights(available, ECS_VALUES)

    n_lat, n_lon = len(ifs_lats), len(ifs_lons)

    # Pre-compute baseline per model
    baselines = {}
    for model in available:
        parts = []
        for yr in TRAIN_YEARS:
            scen = "historical" if yr <= 2014 else CLIM_SSP
            r = cmip6_summer_mean_year(model, scen, yr)
            if r is not None:
                parts.append(r[2])
        if parts:
            baselines[model] = (np.nanmean(np.stack(parts, axis=0), axis=0),
                                r[0], r[1])

    dt_yearly = {}
    for yr in VAL_YEARS:
        dt_wsum = np.zeros((n_lat, n_lon), dtype=np.float64)
        w_sum = np.zeros((n_lat, n_lon), dtype=np.float64)
        scen = "historical" if yr <= 2014 else CLIM_SSP

        for model in available:
            if model not in baselines:
                continue
            bl, lats_src, lons_src = baselines[model]
            r = cmip6_summer_mean_year(model, scen, yr)
            if r is None:
                continue
            dt_model = (r[2] - bl).astype(np.float32)
            dt_ifs = regrid_gcm_to_ifs(dt_model, lats_src, lons_src, ifs_lats, ifs_lons)
            w = weights[model]
            valid = np.isfinite(dt_ifs)
            dt_wsum[valid] += w * dt_ifs[valid]
            w_sum[valid] += w

        with np.errstate(divide="ignore", invalid="ignore"):
            dt_yearly[yr] = np.where(w_sum > 0, dt_wsum / w_sum, np.nan).astype(np.float32)
        print(f"    GCM ΔT {yr}: mean={np.nanmean(dt_yearly[yr]):.3f}°C")

    return dt_yearly


def compute_era5_ehd_trend_train(ehd_train_stack_025, ifs_lats, ifs_lons):
    """
    V5: Linear trend of EHD fitted on TRAIN_YEARS, extrapolated to VAL_MID.
    Returns ΔEHD_025 (period-mean) and trend_slope for yearly prediction.
    """
    print("Computing ERA5-Land EHD trend (train 2000-2012)...")
    years_arr = np.array(TRAIN_YEARS, dtype=np.float32)
    years_3d = np.broadcast_to(years_arr[:, None, None], ehd_train_stack_025.shape)
    trend_slope = ols_slope_vectorized(years_3d, ehd_train_stack_025)

    ehd_baseline_train = np.nanmean(ehd_train_stack_025, axis=0)

    # Period-mean: predicted EHD at VAL_MID − baseline_train mean
    dehd = (trend_slope * (VAL_MID - TRAIN_MID)).astype(np.float32)

    # Clip
    new_ehd = np.clip(ehd_baseline_train + dehd, 0.0, 1.0)
    dehd = (new_ehd - ehd_baseline_train).astype(np.float32)

    print(f"  EHD trend slope: mean={np.nanmean(trend_slope):.6f}/yr")
    print(f"  ΔEHD (train→val): mean={np.nanmean(dehd):.4f}")
    return dehd, trend_slope


def compute_cil_dehd_validation(ifs_lats, ifs_lons):
    """
    V6: CIL GDPCIR EHD — ECS-weighted ΔEHD for validation period.
    Baseline = mean(hist 2000-2012), Validation = mean(hist 2013-2014 + ssp245 2015-2024).
    Returns ΔEHD at IFS 0.25° and per-year CIL EHD dict.
    """
    print("Computing CIL GDPCIR ΔEHD for validation period...")
    models = sorted(CIL_ECS.keys())
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

    # Also collect per-year for yearly validation
    yearly_wsum = {yr: np.zeros((n_lat, n_lon), dtype=np.float64) for yr in VAL_YEARS}
    yearly_w = {yr: np.zeros((n_lat, n_lon), dtype=np.float64) for yr in VAL_YEARS}

    for mi, model in enumerate(available):
        t0 = time.time()
        print(f"  [{mi+1}/{len(available)}] {model} (w={weights[model]:.4f})...",
              end="", flush=True)

        ds_hist = xr.open_dataset(str(CIL_EHD_DIR / model / "ehd_historical.nc"))
        lats_src = ds_hist.lat.values
        lons_src = ds_hist.lon.values
        yrs_h = ds_hist.year.values
        ehd_hist = ds_hist["ehd"].values
        ds_hist.close()

        ds_ssp = xr.open_dataset(str(CIL_EHD_DIR / model / "ehd_ssp245.nc"))
        yrs_s = ds_ssp.year.values
        ehd_ssp = ds_ssp["ehd"].values
        ds_ssp.close()

        # Baseline: mean(hist 2000-2012)
        mask_bl_h = (yrs_h >= 2000) & (yrs_h <= 2012)
        if not mask_bl_h.any():
            print(" no baseline — skip")
            continue
        baseline = ehd_hist[mask_bl_h].mean(axis=0)

        # Validation mean: hist 2013-2014 + ssp 2015-2024
        val_parts = []
        mask_val_h = (yrs_h >= 2013) & (yrs_h <= 2014)
        if mask_val_h.any():
            val_parts.append(ehd_hist[mask_val_h])
        mask_val_s = (yrs_s >= 2015) & (yrs_s <= 2024)
        if mask_val_s.any():
            val_parts.append(ehd_ssp[mask_val_s])
        if not val_parts:
            print(" no val years — skip")
            continue
        val_all = np.concatenate(val_parts, axis=0)
        val_mean = val_all.mean(axis=0)

        dehd_model = (val_mean - baseline).astype(np.float32)
        dehd_ifs = regrid_gcm_to_ifs(dehd_model, lats_src, lons_src, ifs_lats, ifs_lons)

        w = weights[model]
        valid = np.isfinite(dehd_ifs)
        dehd_wsum[valid] += w * dehd_ifs[valid]
        w_sum[valid] += w

        # Per-year EHD for yearly validation (absolute, not delta)
        for yr in VAL_YEARS:
            idx_h = np.where(yrs_h == yr)[0]
            idx_s = np.where(yrs_s == yr)[0]
            if len(idx_h) > 0:
                ehd_yr = ehd_hist[idx_h[0]]
            elif len(idx_s) > 0:
                ehd_yr = ehd_ssp[idx_s[0]]
            else:
                continue
            ehd_yr_ifs = regrid_gcm_to_ifs(ehd_yr.astype(np.float32), lats_src, lons_src,
                                           ifs_lats, ifs_lons)
            valid_yr = np.isfinite(ehd_yr_ifs)
            yearly_wsum[yr][valid_yr] += w * ehd_yr_ifs[valid_yr]
            yearly_w[yr][valid_yr] += w

        print(f" ΔEHD mean={np.nanmean(dehd_ifs):.4f}  ({time.time()-t0:.0f}s)")

    with np.errstate(divide="ignore", invalid="ignore"):
        dehd = np.where(w_sum > 0, dehd_wsum / w_sum, np.nan).astype(np.float32)

    cil_yearly = {}
    for yr in VAL_YEARS:
        with np.errstate(divide="ignore", invalid="ignore"):
            cil_yearly[yr] = np.where(yearly_w[yr] > 0,
                                      yearly_wsum[yr] / yearly_w[yr],
                                      np.nan).astype(np.float32)

    print(f"  CIL ΔEHD: mean={np.nanmean(dehd):.4f}")
    return dehd, cil_yearly


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(predicted, observed, mask=None):
    """Compute RMSE, bias, Pearson r, Spearman ρ between predicted and observed."""
    if mask is None:
        mask = np.isfinite(predicted) & np.isfinite(observed)
    p = predicted[mask]
    o = observed[mask]
    if len(p) < 10:
        return {"rmse": np.nan, "bias": np.nan, "pearson_r": np.nan,
                "spearman_rho": np.nan, "n": len(p)}

    resid = p - o
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    bias = float(np.mean(resid))
    pearson_r = float(np.corrcoef(p, o)[0, 1])
    spearman_rho = float(spearmanr(p, o).correlation)

    return {
        "rmse": rmse,
        "bias": bias,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "n": len(p),
    }


def compute_gdp_weighted_rmse(predicted_entity, observed_entity, gdp_weights):
    """GDP-weighted RMSE at entity level."""
    mask = (np.isfinite(predicted_entity) & np.isfinite(observed_entity)
            & np.isfinite(gdp_weights) & (gdp_weights > 0))
    if mask.sum() < 10:
        return np.nan
    p = predicted_entity[mask]
    o = observed_entity[mask]
    w = gdp_weights[mask]
    w = w / w.sum()
    return float(np.sqrt(np.sum(w * (p - o) ** 2)))


# ═══════════════════════════════════════════════════════════════════════════════
#  Entity and country aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def get_entity_gdp_weights():
    """Get GDP per capita per entity from the panel file."""
    panel = pd.read_parquet(str(PANEL_FILE))
    return panel.groupby("GID_2")["gdp_per_capita"].mean()


def aggregate_to_country(entity_values, entities_df, gdp_mean):
    """
    Aggregate entity-level values to country-level (GDP-weighted within country).
    Returns dict {iso3: value}.
    """
    df = entities_df.copy()
    df["val"] = entity_values
    df["gdp_pc"] = df["GID_2"].map(gdp_mean).astype(np.float64)
    df = df.dropna(subset=["val", "gdp_pc"])

    result = {}
    for iso3, grp in df.groupby("iso3"):
        w = grp["gdp_pc"].values
        v = grp["val"].values
        if w.sum() > 0:
            result[iso3] = float(np.average(v, weights=w))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════════════════════

def plot_validation_maps(observed_025, predicted_grids, ifs_lats, ifs_lons,
                         variant_names, out_path):
    """7-panel figure: observed + 6 predicted ΔEHD."""
    fig, axes = plt.subplots(3, 3, figsize=(27, 18))
    axes = axes.ravel()

    bounds = [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04, 0.06]
    cmap = plt.cm.get_cmap("RdYlBu_r", len(bounds) - 1)
    norm = BoundaryNorm(bounds, cmap.N)
    lon_mesh, lat_mesh = np.meshgrid(ifs_lons, ifs_lats)

    # Observed
    ax = axes[0]
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_facecolor("#e8e8e8")
    ax.pcolormesh(lon_mesh, lat_mesh, observed_025, norm=norm, cmap=cmap,
                  shading="auto", rasterized=True)
    ax.set_title("Observed ΔEHD (ERA5-Land)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=9)

    # 6 predicted
    for i, (grid, name) in enumerate(zip(predicted_grids, variant_names)):
        ax = axes[i + 1]
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
        ax.set_facecolor("#e8e8e8")
        ax.pcolormesh(lon_mesh, lat_mesh, grid, norm=norm, cmap=cmap,
                      shading="auto", rasterized=True)
        ax.set_title(name, fontsize=10, fontweight="bold")
        if (i + 1) % 3 == 0:
            pass
        if (i + 1) >= 6:
            ax.set_xlabel("Longitude", fontsize=9)
        if (i + 1) % 3 == 0:
            pass
        if i % 3 == 2:
            pass

    # Hide unused panels
    for j in range(7, 9):
        axes[j].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="horizontal",
                        fraction=0.02, pad=0.04, aspect=50, ticks=bounds)
    cbar.set_label("ΔEHD (fraction)", fontsize=12)

    fig.suptitle("Historical validation: Observed vs Predicted ΔEHD\n"
                 "(2013-2024 mean − 2000-2012 mean)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_residual_maps(residual_grids, ifs_lats, ifs_lons, variant_names, out_path):
    """6-panel residual maps (predicted − observed)."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.ravel()

    bounds = [-0.04, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04]
    cmap = plt.cm.get_cmap("RdBu_r", len(bounds) - 1)
    norm = BoundaryNorm(bounds, cmap.N)
    lon_mesh, lat_mesh = np.meshgrid(ifs_lons, ifs_lats)

    for i, (ax, grid, name) in enumerate(zip(axes, residual_grids, variant_names)):
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
        ax.set_facecolor("#e8e8e8")
        ax.pcolormesh(lon_mesh, lat_mesh, grid, norm=norm, cmap=cmap,
                      shading="auto", rasterized=True)
        ax.set_title(f"{name}\n(predicted − observed)", fontsize=10, fontweight="bold")
        if i >= 3:
            ax.set_xlabel("Longitude", fontsize=9)
        if i % 3 == 0:
            ax.set_ylabel("Latitude", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="horizontal",
                        fraction=0.025, pad=0.06, aspect=50, ticks=bounds)
    cbar.set_label("Residual ΔEHD (predicted − observed)", fontsize=12)

    fig.suptitle("Residual maps: Prediction errors by method",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_calibration(observed_025, predicted_grids, variant_names, out_path):
    """Calibration plot: binned predicted vs observed ΔEHD (20 quantile bins)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:6]
    mask_obs = np.isfinite(observed_025)

    for i, (grid, name) in enumerate(zip(predicted_grids, variant_names)):
        mask = mask_obs & np.isfinite(grid)
        obs_flat = observed_025[mask].ravel()
        pred_flat = grid[mask].ravel()

        # 20 quantile bins based on predicted values
        try:
            bin_edges = np.nanpercentile(pred_flat, np.linspace(0, 100, 21))
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 3:
                continue
            bin_idx = np.digitize(pred_flat, bin_edges[1:-1])
            bin_means_pred = []
            bin_means_obs = []
            for b in range(len(bin_edges) - 1):
                sel = bin_idx == b
                if sel.sum() > 0:
                    bin_means_pred.append(np.mean(pred_flat[sel]))
                    bin_means_obs.append(np.mean(obs_flat[sel]))
            ax.plot(bin_means_pred, bin_means_obs, "o-", color=colors[i],
                    label=name, markersize=4, linewidth=1.5, alpha=0.8)
        except Exception:
            continue

    # 1:1 line
    lims = ax.get_xlim()
    rng = [min(lims[0], ax.get_ylim()[0]), max(lims[1], ax.get_ylim()[1])]
    ax.plot(rng, rng, "k--", alpha=0.4, label="1:1 line")
    ax.set_xlim(rng); ax.set_ylim(rng)

    ax.set_xlabel("Predicted ΔEHD (binned mean)", fontsize=11)
    ax.set_ylabel("Observed ΔEHD (binned mean)", fontsize=11)
    ax.set_title("Calibration: Predicted vs Observed ΔEHD\n(20 quantile bins)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_variance_decomposition(predicted_grids, variant_names, out_path):
    """
    Decompose inter-method variance into 3 axes:
    - ΔT source (ERA5 vs GCM): V1 vs V3, V2 vs V4
    - Spatial scale (local vs 200km): V1 vs V2, V3 vs V4
    - Functional form (slope-based vs direct): V3/V4 vs V6, V1/V2 vs V5
    """
    # Stack valid pixels
    stacked = np.stack(predicted_grids, axis=0)  # (6, lat, lon)
    all_valid = np.all(np.isfinite(stacked), axis=0)
    vals = stacked[:, all_valid]  # (6, N)

    total_var = np.var(vals, axis=0).mean()

    # ΔT source: ERA5 vs GCM (V1 vs V3, V2 vs V4)
    dt_diff_1 = (vals[0] - vals[2])  # V1 - V3
    dt_diff_2 = (vals[1] - vals[3])  # V2 - V4
    dt_var = (np.var(dt_diff_1) + np.var(dt_diff_2)) / 2

    # Spatial scale: local vs 200km (V1 vs V2, V3 vs V4)
    scale_diff_1 = (vals[0] - vals[1])  # V1 - V2
    scale_diff_2 = (vals[2] - vals[3])  # V3 - V4
    scale_var = (np.var(scale_diff_1) + np.var(scale_diff_2)) / 2

    # Functional form: slope-based vs direct
    slope_mean = (vals[0] + vals[1] + vals[2] + vals[3]) / 4
    direct_mean = (vals[4] + vals[5]) / 2
    form_diff = slope_mean - direct_mean
    form_var = np.var(form_diff)

    vars_dict = {
        "ΔT source\n(ERA5 trend vs GCM)": dt_var,
        "Spatial scale\n(local vs 200km)": scale_var,
        "Functional form\n(slope-based vs direct)": form_var,
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    labels = list(vars_dict.keys())
    values = np.array(list(vars_dict.values()))
    pcts = values / total_var * 100

    bars = ax.bar(labels, pcts, color=["#2196F3", "#FF9800", "#4CAF50"],
                  edgecolor="white", linewidth=1.5)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    ax.set_ylabel("% of total inter-method variance", fontsize=11)
    ax.set_title("Variance decomposition across 6 ΔEHD methods\n"
                 "(historical validation period: 2013-2024 vs 2000-2012)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(pcts) * 1.3)

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_yearly_rmse(yearly_metrics, out_path):
    """Line plot of RMSE(t) for each method over 2013-2024."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:6]

    for i, (name, metrics_by_year) in enumerate(yearly_metrics.items()):
        years = sorted(metrics_by_year.keys())
        rmses = [metrics_by_year[yr]["rmse"] for yr in years]
        ax.plot(years, rmses, "o-", color=colors[i], label=name,
                markersize=4, linewidth=1.5)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("RMSE (EHD fraction)", fontsize=11)
    ax.set_title("Year-by-year pixel-level RMSE: Predicted vs Observed EHD",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_yearly_bias(yearly_metrics, out_path):
    """Line plot of bias(t) for each method over 2013-2024."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:6]

    for i, (name, metrics_by_year) in enumerate(yearly_metrics.items()):
        years = sorted(metrics_by_year.keys())
        biases = [metrics_by_year[yr]["bias"] for yr in years]
        ax.plot(years, biases, "o-", color=colors[i], label=name,
                markersize=4, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Bias (predicted − observed)", fontsize=11)
    ax.set_title("Year-by-year pixel-level bias: Predicted vs Observed EHD",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_yearly_timeseries(observed_yearly_global, predicted_yearly_global,
                           variant_names, out_path):
    """Global GDP-weighted mean EHD: observed vs 6 predicted trajectories."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:6]

    years = sorted(observed_yearly_global.keys())
    obs_vals = [observed_yearly_global[yr] for yr in years]
    ax.plot(years, obs_vals, "ks-", label="Observed (ERA5-Land)", linewidth=2.5,
            markersize=6, zorder=10)

    for i, name in enumerate(variant_names):
        pred_vals = [predicted_yearly_global[name].get(yr, np.nan) for yr in years]
        ax.plot(years, pred_vals, "o--", color=colors[i], label=name,
                markersize=4, linewidth=1.2, alpha=0.8)

    # Also show training period baseline
    ax.axvspan(2000, 2012.5, alpha=0.08, color="gray", label="Training period")

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Global GDP-weighted mean EHD", fontsize=11)
    ax.set_title("Global GDP-weighted EHD trajectory:\nObserved vs Predicted",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    wall_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Historical out-of-sample validation of ΔEHD projection methods")
    print(f"  Train: {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]} ({len(TRAIN_YEARS)} years)")
    print(f"  Validate: {VAL_YEARS[0]}-{VAL_YEARS[-1]} ({len(VAL_YEARS)} years)")
    print("=" * 70)

    # ── Step 1: Load IFS slopes ───────────────────────────────────────────
    rr_local, rr_200km, ifs_lats, ifs_lons = load_ifs_slopes()

    # ── Step 2: Load observed EHD ─────────────────────────────────────────
    print("\nLoading observed EHD for training and validation periods...")
    ehd_train_025, ehd_train_stack_01, train_yrs = load_ehd_years_025(
        TRAIN_YEARS, ifs_lats, ifs_lons)
    ehd_val_025, ehd_val_stack_01, val_yrs = load_ehd_years_025(
        VAL_YEARS, ifs_lats, ifs_lons)
    print(f"  Train: {len(train_yrs)} years, mean EHD={np.nanmean(ehd_train_025):.4f}")
    print(f"  Val:   {len(val_yrs)} years, mean EHD={np.nanmean(ehd_val_025):.4f}")

    # Observed ΔEHD at 0.25°
    observed_dehd = (ehd_val_025 - ehd_train_025).astype(np.float32)
    print(f"  Observed ΔEHD: mean={np.nanmean(observed_dehd):.4f}, "
          f"median={np.nanmedian(observed_dehd):.4f}")

    # Also coarsen train stack to 0.25° for V5
    ehd_train_stack_025 = coarsen_stack_to_025(ehd_train_stack_01, ifs_lats, ifs_lons)

    # ── Step 3: Compute predicted ΔEHD for each method ────────────────────

    # V1/V2: ERA5-Land Tmax trend × IFS slopes
    dt_era5_local, dt_era5_200km, era5_tmax_slope = compute_era5_dt_train(
        ifs_lats, ifs_lons)
    dehd_v1 = compute_slope_dehd(ehd_train_025, rr_local, dt_era5_local)
    dehd_v2 = compute_slope_dehd(ehd_train_025, rr_200km, dt_era5_200km)

    # V3/V4: GCM ΔT × IFS slopes
    dt_gcm_local, dt_gcm_200km = compute_gcm_dt_validation(ifs_lats, ifs_lons)
    dehd_v3 = compute_slope_dehd(ehd_train_025, rr_local, dt_gcm_local)
    dehd_v4 = compute_slope_dehd(ehd_train_025, rr_200km, dt_gcm_200km)

    # V5: ERA5-Land EHD trend
    dehd_v5, ehd_trend_slope = compute_era5_ehd_trend_train(
        ehd_train_stack_025, ifs_lats, ifs_lons)

    # V6: CIL GDPCIR ΔEHD
    dehd_v6, cil_yearly = compute_cil_dehd_validation(ifs_lats, ifs_lons)

    predicted_grids = [dehd_v1, dehd_v2, dehd_v3, dehd_v4, dehd_v5, dehd_v6]
    variant_names = [
        "V1: ERA5 local × IFS local",
        "V2: ERA5 200km × IFS 200km",
        "V3: GCM local × IFS local",
        "V4: GCM 200km × IFS 200km",
        "V5: ERA5 EHD trend",
        "V6: CIL GDPCIR (ECS-wt)",
    ]
    variant_short = [
        "V1_era5_local", "V2_era5_200km",
        "V3_gcm_local", "V4_gcm_200km",
        "V5_ehd_trend", "V6_cil_gdpcir",
    ]

    for name, grid in zip(variant_names, predicted_grids):
        print(f"  {name}: mean ΔEHD={np.nanmean(grid):.4f}, "
              f"median={np.nanmedian(grid):.4f}")

    # ── Step 4: Pixel-level metrics ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Pixel-level validation metrics (0.25° IFS grid)")
    print("=" * 70)

    pixel_results = []
    for name, short, grid in zip(variant_names, variant_short, predicted_grids):
        m = compute_metrics(grid, observed_dehd)
        m["variant"] = short
        m["variant_label"] = name
        m["scale"] = "pixel_025"
        pixel_results.append(m)
        print(f"  {name}: RMSE={m['rmse']:.5f}, bias={m['bias']:.5f}, "
              f"r={m['pearson_r']:.3f}, ρ={m['spearman_rho']:.3f}, n={m['n']:,}")

    # ── Step 5: Entity-level metrics ──────────────────────────────────────
    print("\nAggregating to entities...")
    t0 = time.time()
    combined_raster, entities, n_entities = load_mixed_raster()
    settlement = load_settlement()
    print(f"  {n_entities:,} entities loaded  ({time.time()-t0:.0f}s)")

    # Observed ΔEHD at entity level
    obs_dehd_01 = interp_025_to_01(observed_dehd, ifs_lats, ifs_lons)
    obs_entity = agg_to_entities(obs_dehd_01, combined_raster, settlement, n_entities)

    # Predicted ΔEHD at entity level
    pred_entities = []
    for grid, name in zip(predicted_grids, variant_short):
        grid_01 = interp_025_to_01(grid, ifs_lats, ifs_lons)
        ent_vals = agg_to_entities(grid_01, combined_raster, settlement, n_entities)
        pred_entities.append(ent_vals)

    # GDP weights
    gdp_mean = get_entity_gdp_weights()
    gdp_arr = entities["GID_2"].map(gdp_mean).values.astype(np.float64)

    print("\n" + "=" * 70)
    print("Entity-level validation metrics")
    print("=" * 70)

    entity_results = []
    for name, short, ent_pred in zip(variant_names, variant_short, pred_entities):
        m = compute_metrics(ent_pred, obs_entity)
        m["gdp_weighted_rmse"] = compute_gdp_weighted_rmse(ent_pred, obs_entity, gdp_arr)
        m["variant"] = short
        m["variant_label"] = name
        m["scale"] = "entity"
        entity_results.append(m)
        print(f"  {name}: RMSE={m['rmse']:.5f}, bias={m['bias']:.5f}, "
              f"r={m['pearson_r']:.3f}, GDP-wt RMSE={m['gdp_weighted_rmse']:.5f}")

    # ── Step 6: Country-level metrics ─────────────────────────────────────
    print("\nAggregating to country level...")
    obs_country = aggregate_to_country(obs_entity, entities, gdp_mean)

    country_results = []
    for name, short, ent_pred in zip(variant_names, variant_short, pred_entities):
        pred_country = aggregate_to_country(ent_pred, entities, gdp_mean)
        # Match countries
        common = sorted(set(obs_country) & set(pred_country))
        obs_arr = np.array([obs_country[c] for c in common], dtype=np.float32)
        pred_arr = np.array([pred_country[c] for c in common], dtype=np.float32)
        m = compute_metrics(pred_arr, obs_arr)
        m["variant"] = short
        m["variant_label"] = name
        m["scale"] = "country"
        country_results.append(m)
        print(f"  {name}: RMSE={m['rmse']:.5f}, bias={m['bias']:.5f}, "
              f"r={m['pearson_r']:.3f}, n_countries={m['n']}")

    # ── Step 7: Save results ──────────────────────────────────────────────
    all_results = pixel_results + entity_results + country_results
    df_all = pd.DataFrame(all_results)
    csv_path = OUT_DIR / "validation_results.csv"
    df_all.to_csv(str(csv_path), index=False, float_format="%.6f")
    print(f"\nSaved: {csv_path}")

    # Compact summary (pixel-level only, key metrics)
    df_summary = pd.DataFrame(pixel_results)[
        ["variant", "variant_label", "rmse", "bias", "pearson_r", "spearman_rho", "n"]
    ]
    # Add entity and country metrics as columns
    for er in entity_results:
        idx = df_summary[df_summary["variant"] == er["variant"]].index[0]
        df_summary.loc[idx, "entity_rmse"] = er["rmse"]
        df_summary.loc[idx, "entity_r"] = er["pearson_r"]
        df_summary.loc[idx, "entity_gdp_wt_rmse"] = er.get("gdp_weighted_rmse", np.nan)
    for cr in country_results:
        idx = df_summary[df_summary["variant"] == cr["variant"]].index[0]
        df_summary.loc[idx, "country_rmse"] = cr["rmse"]
        df_summary.loc[idx, "country_r"] = cr["pearson_r"]

    summary_path = OUT_DIR / "validation_summary.csv"
    df_summary.to_csv(str(summary_path), index=False, float_format="%.6f")
    print(f"Saved: {summary_path}")

    # ── Step 8: Figures ───────────────────────────────────────────────────
    print("\nGenerating figures...")

    # Validation maps
    plot_validation_maps(observed_dehd, predicted_grids, ifs_lats, ifs_lons,
                         variant_names, OUT_DIR / "validation_maps.png")

    # Residual maps
    residual_grids = [(p - observed_dehd).astype(np.float32) for p in predicted_grids]
    plot_residual_maps(residual_grids, ifs_lats, ifs_lons, variant_names,
                       OUT_DIR / "validation_residual_maps.png")

    # Calibration
    plot_calibration(observed_dehd, predicted_grids, variant_names,
                     OUT_DIR / "validation_calibration.png")

    # Variance decomposition
    plot_variance_decomposition(predicted_grids, variant_names,
                                OUT_DIR / "validation_variance_decomposition.png")

    # ── Step 9: Year-by-year validation ───────────────────────────────────
    print("\n" + "=" * 70)
    print("Year-by-year validation")
    print("=" * 70)

    # Load observed EHD per year at 0.25°
    print("Loading per-year observed EHD at 0.25°...")
    obs_yearly_025 = {}
    for yr in VAL_YEARS:
        arr_01 = load_ehd_year_01(yr)
        if arr_01 is not None:
            obs_yearly_025[yr] = coarsen_01_to_025(arr_01, ifs_lats, ifs_lons)
            print(f"    {yr}: mean={np.nanmean(obs_yearly_025[yr]):.4f}")

    # Per-year GCM ΔT (for V3/V4 yearly)
    gcm_dt_yearly = compute_gcm_dt_yearly(ifs_lats, ifs_lons)

    # Predict EHD(t) for each method and year
    yearly_metrics = {name: {} for name in variant_names}
    predicted_yearly_global = {name: {} for name in variant_names}
    observed_yearly_global = {}

    # Entity-level observed yearly (for GDP-weighted global mean)
    for yr in sorted(obs_yearly_025.keys()):
        obs_01 = interp_025_to_01(obs_yearly_025[yr], ifs_lats, ifs_lons)
        obs_ent = agg_to_entities(obs_01, combined_raster, settlement, n_entities)
        mask_valid = np.isfinite(obs_ent) & np.isfinite(gdp_arr) & (gdp_arr > 0)
        if mask_valid.sum() > 0:
            observed_yearly_global[yr] = float(
                np.average(obs_ent[mask_valid], weights=gdp_arr[mask_valid]))

    for yr in sorted(obs_yearly_025.keys()):
        obs = obs_yearly_025[yr]
        dt_yr_from_train = float(yr) - TRAIN_MID  # years from training midpoint

        # V1: EHD_base_train + EHD_base_train × (RR^(tmax_slope × dt_yr) − 1)
        dt_v1 = (era5_tmax_slope * dt_yr_from_train).astype(np.float32)
        ehd_pred_v1 = ehd_train_025 + compute_slope_dehd(ehd_train_025, rr_local, dt_v1)

        # V2: same with 200km
        dt_v2 = apply_gaussian_filter(dt_v1, sigma=SIGMA_025)
        ehd_pred_v2 = ehd_train_025 + compute_slope_dehd(ehd_train_025, rr_200km, dt_v2)

        # V3: GCM ΔT(yr) × local slope
        if yr in gcm_dt_yearly:
            dt_gcm_yr = gcm_dt_yearly[yr]
            ehd_pred_v3 = ehd_train_025 + compute_slope_dehd(
                ehd_train_025, rr_local, dt_gcm_yr)
            dt_gcm_yr_200 = apply_gaussian_filter(dt_gcm_yr, sigma=SIGMA_025)
            ehd_pred_v4 = ehd_train_025 + compute_slope_dehd(
                ehd_train_025, rr_200km, dt_gcm_yr_200)
        else:
            ehd_pred_v3 = np.full_like(obs, np.nan)
            ehd_pred_v4 = np.full_like(obs, np.nan)

        # V5: EHD_base_train + ehd_slope × dt
        ehd_pred_v5 = ehd_train_025 + (ehd_trend_slope * dt_yr_from_train).astype(np.float32)
        ehd_pred_v5 = np.clip(ehd_pred_v5, 0.0, 1.0)

        # V6: CIL EHD(yr) directly
        if yr in cil_yearly:
            ehd_pred_v6 = cil_yearly[yr]
        else:
            ehd_pred_v6 = np.full_like(obs, np.nan)

        preds = [ehd_pred_v1, ehd_pred_v2, ehd_pred_v3, ehd_pred_v4,
                 ehd_pred_v5, ehd_pred_v6]

        for name, pred in zip(variant_names, preds):
            m = compute_metrics(pred, obs)
            yearly_metrics[name][yr] = m

            # GDP-weighted global mean for time series
            pred_01 = interp_025_to_01(pred, ifs_lats, ifs_lons)
            pred_ent = agg_to_entities(pred_01, combined_raster, settlement, n_entities)
            mask_valid = np.isfinite(pred_ent) & np.isfinite(gdp_arr) & (gdp_arr > 0)
            if mask_valid.sum() > 0:
                predicted_yearly_global[name][yr] = float(
                    np.average(pred_ent[mask_valid], weights=gdp_arr[mask_valid]))

        print(f"  {yr}: V1 RMSE={yearly_metrics[variant_names[0]][yr]['rmse']:.5f}, "
              f"V5 RMSE={yearly_metrics[variant_names[4]][yr]['rmse']:.5f}, "
              f"V6 RMSE={yearly_metrics[variant_names[5]][yr]['rmse']:.5f}")

    # Year-by-year figures
    plot_yearly_rmse(yearly_metrics, OUT_DIR / "validation_yearly_rmse.png")
    plot_yearly_bias(yearly_metrics, OUT_DIR / "validation_yearly_bias.png")
    plot_yearly_timeseries(observed_yearly_global, predicted_yearly_global,
                           variant_names, OUT_DIR / "validation_yearly_timeseries.png")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - wall_start
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY (pixel-level, 0.25°)")
    print("=" * 70)
    print(f"{'Method':<32} {'RMSE':>8} {'Bias':>8} {'r':>6} {'ρ':>6}")
    print("-" * 62)
    for r in pixel_results:
        print(f"{r['variant_label']:<32} {r['rmse']:>8.5f} {r['bias']:>8.5f} "
              f"{r['pearson_r']:>6.3f} {r['spearman_rho']:>6.3f}")
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
