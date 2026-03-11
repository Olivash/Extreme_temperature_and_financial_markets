#!/usr/bin/env python -u
"""
Aggregate ERA5 annual mean temperature and total precipitation to admin2 polygons.

Reads monthly ERA5 GRIB files (0.25° resolution), computes:
  - Annual mean 2m temperature (°C)
  - Annual total precipitation (mm)

Rasterises GADM admin2 polygons at 0.25° and aggregates via area-weighted mean.

Inputs (set paths in src/config.py or via environment variables):
  <ERA5_DIR>/2m_temperature/{YYYY}/era5_t2m_{YYYY}{MM:02d}.grib
  <ERA5_DIR>/total_precipitation/{YYYY}/era5_tp_{YYYY}{MM:02d}.grib
  <ADMIN2_GPKG> — polyg_adm2_gdp_perCapita_1990_2022.gpkg

  See data/raw/climate/README.md and data/raw/gdp/README.md for download
  instructions.

Output:
  <PANEL_DIR>/admin2_era5_annual_controls.parquet
    Columns: GID_2, iso3, year, annual_temp_c, annual_precip_mm, n_cells_025

Runtime: ~30–60 min (600 GRIB files × ~3s each).
"""

import sys
import time
import calendar
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path
from rasterio.transform import from_bounds
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ERA5_DIR, ADMIN2_GPKG, PANEL_DIR, CACHE_DIR

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ──────────────────────────────────────────────────────────────────
T2M_DIR    = ERA5_DIR / "2m_temperature"
TP_DIR     = ERA5_DIR / "total_precipitation"
OUTPUT_DIR = PANEL_DIR

# ─── Grid parameters (must match ERA5 0.25° grid) ───────────────────────────
N_LAT, N_LON = 721, 1440
OUT_LATS = np.round(np.arange(90.0, -90.1, -0.25), 2)[:N_LAT]
OUT_LONS = np.round(np.arange(-180.0, 180.0, 0.25), 2)[:N_LON]
assert len(OUT_LATS) == N_LAT and len(OUT_LONS) == N_LON

YEARS = range(2000, 2025)


# ─── GRIB loading ───────────────────────────────────────────────────────────
def load_era5_month(path):
    """Load one ERA5 monthly GRIB file, return 2D array in native 0-360 convention."""
    ds = xr.open_dataset(str(path), engine="cfgrib")
    var = list(ds.data_vars)[0]
    data = ds[var].values
    if data.ndim == 3:
        data = data[0]  # squeeze time/step dim
    ds.close()
    return data.astype(np.float32)


def convert_lon_360_to_180(data):
    """Roll data array from 0-360 to -180..180 longitude convention."""
    return np.roll(data, N_LON // 2, axis=-1)


# ─── Annual climate variables ───────────────────────────────────────────────
def compute_annual_temp(year):
    """Annual mean 2m temperature (°C), simple mean of 12 monthly means."""
    monthly = []
    for m in range(1, 13):
        path = T2M_DIR / str(year) / f"era5_t2m_{year}{m:02d}.grib"
        monthly.append(load_era5_month(path))
    annual_k = np.nanmean(monthly, axis=0)
    return convert_lon_360_to_180(annual_k) - 273.15   # K → °C


def compute_annual_precip(year):
    """Annual total precipitation (mm).
    ERA5 monthly tp is the average daily rate (m/day), so
    monthly total = rate × days_in_month; annual = sum of monthly totals.
    """
    total = np.zeros((N_LAT, N_LON), dtype=np.float64)
    for m in range(1, 13):
        path = TP_DIR / str(year) / f"era5_tp_{year}{m:02d}.grib"
        data = load_era5_month(path).astype(np.float64)
        days = calendar.monthrange(year, m)[1]
        total += data * days      # avg rate × days = monthly total (metres)
    return convert_lon_360_to_180(total) * 1000   # m → mm


# ─── Admin2 rasterisation at 0.25° ─────────────────────────────────────────
def rasterize_admin2():
    """Burn admin2 polygon IDs onto 0.25° grid. Returns (id_grid, lookup_df)."""
    cache_raster = CACHE_DIR / "admin2_raster_025deg.npy"
    cache_lookup = CACHE_DIR / "admin2_lookup_025deg.csv"

    if cache_raster.exists() and cache_lookup.exists():
        print("  Admin2 raster (0.25°): cached")
        return np.load(cache_raster), pd.read_csv(cache_lookup)

    print("  Loading admin2 polygons ...", flush=True)
    t0 = time.time()
    gdf = gpd.read_file(ADMIN2_GPKG)
    print(f"    {len(gdf)} polygons ({time.time() - t0:.0f}s)")

    lookup = gdf[["GID_2", "adm2ID", "iso3", "NAME_2"]].copy()
    lookup["raster_id"] = np.arange(len(gdf))
    lookup.to_csv(cache_lookup, index=False)

    shapes = [(geom, rid) for geom, rid in zip(gdf.geometry, lookup["raster_id"])]
    transform = from_bounds(-180.0, -90.0, 180.0, 90.0, N_LON, N_LAT)

    print("  Rasterising ...", flush=True)
    id_grid = rasterize(
        shapes,
        out_shape=(N_LAT, N_LON),
        transform=transform,
        fill=-1,
        dtype=np.int32,
        all_touched=True,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_raster, id_grid)

    n_assigned = np.sum(id_grid >= 0)
    print(f"    {n_assigned:,} / {N_LAT * N_LON:,} cells assigned "
          f"({100 * n_assigned / (N_LAT * N_LON):.1f}%)")
    print(f"  Done in {time.time() - t0:.0f}s")
    return id_grid, lookup


# ─── Area-weighted aggregation ──────────────────────────────────────────────
def aggregate_to_admin2(data, admin2_grid, n_regions):
    """Simple area-weighted mean of gridded data within each admin2 polygon."""
    valid = (admin2_grid >= 0) & np.isfinite(data)
    ids  = admin2_grid[valid]
    vals = data[valid].astype(np.float64)

    val_sum = np.bincount(ids, weights=vals, minlength=n_regions)
    n_cells = np.bincount(ids, minlength=n_regions)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_val = np.where(n_cells > 0, val_sum / n_cells, np.nan)
    return mean_val.astype(np.float32), n_cells.astype(np.int32)


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wall_start = time.time()

    print(f"ERA5 input directory : {ERA5_DIR}")
    print(f"Output directory     : {OUTPUT_DIR}")

    # Step 1: Rasterise admin2
    print("=" * 60)
    print("Step 1: Rasterise admin2 at 0.25°")
    print("=" * 60)
    admin2_grid, lookup = rasterize_admin2()
    n_regions = len(lookup)
    print(f"  {n_regions:,} admin2 regions")

    # Step 2: Year-by-year aggregation
    print(f"\n{'=' * 60}")
    print(f"Step 2: Aggregate ERA5 T and P ({min(YEARS)}–{max(YEARS)})")
    print("=" * 60)

    rows = []
    for year in YEARS:
        t0 = time.time()

        temp = compute_annual_temp(year)
        temp_mean, temp_cells = aggregate_to_admin2(temp, admin2_grid, n_regions)

        precip = compute_annual_precip(year)
        precip_mean, _ = aggregate_to_admin2(precip, admin2_grid, n_regions)

        for rid in range(n_regions):
            if temp_cells[rid] > 0:
                rows.append({
                    "raster_id":       rid,
                    "year":            year,
                    "annual_temp_c":   temp_mean[rid],
                    "annual_precip_mm":precip_mean[rid],
                    "n_cells_025":     temp_cells[rid],
                })

        n_with = np.sum(temp_cells > 0)
        print(f"  {year}  regions={n_with:,}/{n_regions:,}  "
              f"mean_T={np.nanmean(temp_mean):.1f}°C  "
              f"mean_P={np.nanmean(precip_mean):.0f}mm  "
              f"{time.time() - t0:.1f}s")

    # Step 3: Build output
    print(f"\n{'=' * 60}")
    print("Step 3: Build output")
    print("=" * 60)

    panel = pd.DataFrame(rows)
    panel = panel.merge(lookup[["raster_id", "GID_2", "iso3"]], on="raster_id", how="left")
    col_order = ["GID_2", "iso3", "year", "annual_temp_c", "annual_precip_mm", "n_cells_025"]
    panel = panel[[c for c in col_order if c in panel.columns]]
    panel = panel.sort_values(["iso3", "GID_2", "year"]).reset_index(drop=True)

    out_path = OUTPUT_DIR / "admin2_era5_annual_controls.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"  {out_path}  ({len(panel):,} rows)")

    print(f"\n  T range: [{panel['annual_temp_c'].min():.1f}, {panel['annual_temp_c'].max():.1f}] °C")
    print(f"  P range: [{panel['annual_precip_mm'].min():.0f}, {panel['annual_precip_mm'].max():.0f}] mm")

    annual = panel.groupby("year")[["annual_temp_c", "annual_precip_mm"]].mean()
    print("\n  Global mean by year:")
    for y, row in annual.iterrows():
        print(f"    {y}  T={row['annual_temp_c']:.2f}°C  P={row['annual_precip_mm']:.0f}mm")

    elapsed = time.time() - wall_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
