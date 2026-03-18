#!/usr/bin/env python -u
"""
Aggregate ERA5 annual mean temperature and total precipitation to admin1 polygons.

Uses the admin1 raster cached by aggregate_heat_to_admin1.py.

Inputs (configure via src/config.py or environment variables):
  ERA5_DIR    — root of ERA5 monthly GRIB files
  ADMIN1_GPKG — GADM admin1 GeoPackage

Output:
  <PANEL_DIR>/admin1_era5land_annual_controls.parquet
    Columns: GID_nmbr, iso3, year, annual_temp_c, annual_precip_mm, n_cells_025
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

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (ERA5_DIR, ADMIN1_GPKG, PANEL_DIR, CACHE_DIR,
                    CONTROLS_ADM1_PATH)

T2M_DIR = ERA5_DIR / "2m_temperature"
TP_DIR  = ERA5_DIR / "total_precipitation"

ADMIN1_LOOKUP_CACHE = CACHE_DIR / "admin1_lookup.csv"

# ─── Grid parameters (ERA5 0.25°) ────────────────────────────────────────────
N_LAT, N_LON = 721, 1440
OUT_LATS = np.round(np.arange(90.0, -90.1, -0.25), 2)[:N_LAT]
OUT_LONS = np.round(np.arange(-180.0, 180.0, 0.25), 2)[:N_LON]

YEARS = range(2000, 2025)


def rasterize_admin1_025():
    cache_raster = CACHE_DIR / "admin1_raster_025deg.npy"
    cache_lookup = CACHE_DIR / "admin1_lookup_025deg.csv"

    if cache_raster.exists() and cache_lookup.exists():
        print("  Admin1 raster (0.25°): cached")
        return np.load(str(cache_raster)), pd.read_csv(str(cache_lookup))

    print("  Loading admin1 GPKG for 0.25° raster...")
    gdf = gpd.read_file(str(ADMIN1_GPKG))
    id_col  = "GID_nmbr" if "GID_nmbr" in gdf.columns else gdf.columns[0]
    iso_col = "iso3"     if "iso3"     in gdf.columns else None

    lookup = gdf[[c for c in [id_col, iso_col] if c]].copy()
    lookup.columns = [c for c in ["GID_nmbr", "iso3"][:len(lookup.columns)]]
    lookup["raster_id"] = np.arange(len(gdf))
    lookup.to_csv(str(cache_lookup), index=False)

    shapes    = [(geom, rid) for geom, rid in zip(gdf.geometry, lookup["raster_id"])]
    transform = from_bounds(-180.0, -90.0, 180.0, 90.0, N_LON, N_LAT)
    id_grid   = rasterize(shapes, out_shape=(N_LAT, N_LON), transform=transform,
                          fill=-1, dtype=np.int32, all_touched=True)
    np.save(str(cache_raster), id_grid)
    n_assigned = np.sum(id_grid >= 0)
    print(f"    {n_assigned:,} / {N_LAT*N_LON:,} cells assigned "
          f"({100*n_assigned/(N_LAT*N_LON):.1f}%)")
    return id_grid, lookup


def load_era5_month(path):
    ds   = xr.open_dataset(str(path), engine="cfgrib")
    var  = list(ds.data_vars)[0]
    data = ds[var].values
    if data.ndim == 3:
        data = data[0]
    ds.close()
    return data.astype(np.float32)

def convert_lon_360_to_180(data):
    return np.roll(data, N_LON // 2, axis=-1)

def compute_annual_temp(year):
    monthly = [load_era5_month(T2M_DIR / str(year) / f"era5_t2m_{year}{m:02d}.grib")
               for m in range(1, 13)]
    return convert_lon_360_to_180(np.nanmean(monthly, axis=0)) - 273.15

def compute_annual_precip(year):
    total = np.zeros((N_LAT, N_LON), dtype=np.float64)
    for m in range(1, 13):
        data  = load_era5_month(TP_DIR / str(year) / f"era5_tp_{year}{m:02d}.grib").astype(np.float64)
        total += data * calendar.monthrange(year, m)[1]
    return convert_lon_360_to_180(total) * 1000  # m → mm


def aggregate_to_admin1(data, admin1_grid, n_regions):
    valid   = (admin1_grid >= 0) & np.isfinite(data)
    ids     = admin1_grid[valid]
    vals    = data[valid].astype(np.float64)
    val_sum = np.bincount(ids, weights=vals, minlength=n_regions)
    n_cells = np.bincount(ids,               minlength=n_regions)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_val = np.where(n_cells > 0, val_sum / n_cells, np.nan)
    return mean_val.astype(np.float32), n_cells.astype(np.int32)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    wall_start = time.time()

    print("=" * 60)
    print("Step 1: Rasterise admin1 at 0.25°")
    print("=" * 60)
    admin1_grid, lookup = rasterize_admin1_025()
    n_regions = len(lookup)
    print(f"  {n_regions:,} admin1 regions")

    print(f"\n{'=' * 60}")
    print(f"Step 2: Aggregate ERA5 T and P ({min(YEARS)}–{max(YEARS)})")
    print("=" * 60)

    rows = []
    for year in YEARS:
        t0 = time.time()

        temp,   t_cells = aggregate_to_admin1(compute_annual_temp(year),   admin1_grid, n_regions)
        precip, _       = aggregate_to_admin1(compute_annual_precip(year), admin1_grid, n_regions)

        for rid in range(n_regions):
            if t_cells[rid] > 0:
                rows.append({
                    "raster_id":        rid,
                    "year":             year,
                    "annual_temp_c":    float(temp[rid]),
                    "annual_precip_mm": float(precip[rid]),
                    "n_cells_025":      int(t_cells[rid]),
                })

        n_with = np.sum(t_cells > 0)
        print(f"  {year}  raster={n_with:,}/{n_regions:,}  "
              f"mean_T={np.nanmean(temp):.1f}°C  {time.time()-t0:.1f}s")

    print(f"\n{'=' * 60}")
    print("Step 3: Build output")
    print("=" * 60)

    panel = pd.DataFrame(rows)
    panel = panel.merge(lookup[["raster_id", "GID_nmbr", "iso3"]], on="raster_id", how="left")
    col_order = ["GID_nmbr", "iso3", "year", "annual_temp_c", "annual_precip_mm", "n_cells_025"]
    panel = panel[[c for c in col_order if c in panel.columns]]
    panel = panel.sort_values(["iso3", "GID_nmbr", "year"]).reset_index(drop=True)

    panel.to_parquet(str(CONTROLS_ADM1_PATH), index=False)
    print(f"  {CONTROLS_ADM1_PATH}  ({len(panel):,} rows)")
    print(f"  T range: [{panel['annual_temp_c'].min():.1f}, {panel['annual_temp_c'].max():.1f}] °C")
    print(f"  P range: [{panel['annual_precip_mm'].min():.0f}, {panel['annual_precip_mm'].max():.0f}] mm")

    elapsed = time.time() - wall_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
