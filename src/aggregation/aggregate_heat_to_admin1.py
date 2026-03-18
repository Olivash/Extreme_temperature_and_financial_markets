#!/usr/bin/env python -u
"""
Aggregate settlement-weighted extreme-heat exceedance frequency to admin1 polygons.

Rasterises admin1 polygons at 0.1°, reuses settlement epoch logic and settlement
raster caches from the admin2 pipeline.

Inputs (configure via src/config.py or environment variables):
  EHD_GLOBAL_DIR  — per-year exceedance_frequency_{year}.nc files
  SETTLEMENT_DIR  — GHS built-up surface TIFs (6 epochs: 2000-2025)
  ADMIN1_GPKG     — GADM admin1 GeoPackage with GDP per capita 1990-2022

Output:
  <PANEL_DIR>/admin1_heat_settlement_weighted.parquet
    Columns: GID_nmbr, iso3, year, heat_freq_weighted, total_settlement_m2, n_cells
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (EHD_GLOBAL_DIR, SETTLEMENT_DIR, ADMIN1_GPKG,
                    PANEL_DIR, CACHE_DIR, HEAT_ADM1_PANEL_PATH)

# ─── Grid parameters ─────────────────────────────────────────────────────────
N_LAT, N_LON = 1801, 3600
OUT_LATS = np.round(np.arange(90.0, -90.1, -0.1), 1)
OUT_LONS = np.round(np.arange(-180.0, 180.0, 0.1), 1)

ADMIN1_RASTER_CACHE = CACHE_DIR / "admin1_raster_01deg.nc"
ADMIN1_LOOKUP_CACHE = CACHE_DIR / "admin1_lookup.csv"

# ─── Settlement epoch mapping ─────────────────────────────────────────────────
SETTLEMENT_EPOCHS = [2000, 2005, 2010, 2015, 2020, 2025]

def epoch_for_year(year):
    if year <= 2002:   return 2000
    elif year <= 2007: return 2005
    elif year <= 2012: return 2010
    elif year <= 2017: return 2015
    elif year <= 2022: return 2020
    else:              return 2025


def settle_cache_path(epoch):
    return CACHE_DIR / f"settlement_01deg_{epoch}.nc"

def settle_tif_path(epoch):
    return SETTLEMENT_DIR / f"GHS_BUILT_S_E{epoch}_GLOBE_R2023A_54009_100_V1_0.tif"

def reproject_settlement(epoch):
    cache_path = settle_cache_path(epoch)
    if cache_path.exists():
        print(f"  Settlement {epoch}: cached ({cache_path.name})")
        return
    print(f"  Settlement {epoch}: reprojecting {settle_tif_path(epoch).name} ...", flush=True)
    t0 = time.time()
    global_arr = np.zeros((N_LAT, N_LON), dtype=np.float32)
    band_height_cells = int(round(10.0 / 0.1))
    n_bands = (N_LAT + band_height_cells - 1) // band_height_cells
    with rasterio.open(str(settle_tif_path(epoch))) as src:
        for bi in range(n_bands):
            row_start = bi * band_height_cells
            row_end   = min(row_start + band_height_cells, N_LAT)
            n_rows    = row_end - row_start
            lat_top   = min(90.0 - row_start * 0.1, 90.0)
            lat_bot   = max(90.0 - row_end   * 0.1, -90.0)
            if lat_top - lat_bot < 0.05:
                continue
            dst_transform = from_bounds(-180.0, lat_bot, 180.0, lat_top, N_LON, n_rows)
            dst_band = np.zeros((n_rows, N_LON), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_band,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.average,
                src_nodata=src.nodata,
                dst_nodata=0.0,
            )
            global_arr[row_start:row_end, :] = dst_band
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"built_surface": (["latitude", "longitude"], global_arr)},
        coords={"latitude": OUT_LATS, "longitude": OUT_LONS},
    )
    ds.attrs.update({"units": "m2 built-up per grid cell", "source_epoch": epoch})
    ds.to_netcdf(str(cache_path))
    print(f"  Settlement {epoch}: done in {time.time()-t0:.0f}s")

def load_settlement(epoch):
    ds  = xr.open_dataset(str(settle_cache_path(epoch)))
    arr = ds["built_surface"].values
    ds.close()
    return arr


def rasterize_admin1():
    if ADMIN1_RASTER_CACHE.exists() and ADMIN1_LOOKUP_CACHE.exists():
        print("  Admin1 raster: cached")
        ds      = xr.open_dataset(str(ADMIN1_RASTER_CACHE))
        id_grid = ds["admin1_id"].values
        ds.close()
        lookup  = pd.read_csv(str(ADMIN1_LOOKUP_CACHE))
        return id_grid, lookup

    print("  Admin1 raster: loading polygons ...", flush=True)
    t0  = time.time()
    gdf = gpd.read_file(str(ADMIN1_GPKG))
    print(f"    {len(gdf):,} polygons ({time.time()-t0:.0f}s)")

    id_col   = "GID_nmbr" if "GID_nmbr" in gdf.columns else gdf.columns[0]
    iso_col  = "iso3"     if "iso3"     in gdf.columns else None
    name_col = "Subnat"   if "Subnat"   in gdf.columns else None

    def safe_centroid(geom):
        try:
            if geom is None or geom.is_empty: return (np.nan, np.nan)
            rp = geom.representative_point()
            return (np.nan, np.nan) if rp.is_empty else (rp.y, rp.x)
        except Exception:
            return (np.nan, np.nan)

    coords = [safe_centroid(g) for g in gdf.geometry]

    lookup = gdf[[c for c in [id_col, iso_col, name_col] if c]].copy()
    lookup.columns = [c for c in ["GID_nmbr", "iso3", "Subnat"][:len(lookup.columns)]]
    lookup["raster_id"]    = np.arange(len(gdf))
    lookup["centroid_lat"] = [c[0] for c in coords]
    lookup["centroid_lon"] = [c[1] for c in coords]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(str(ADMIN1_LOOKUP_CACHE), index=False)

    shapes    = [(geom, rid) for geom, rid in zip(gdf.geometry, lookup["raster_id"])]
    transform = from_bounds(-180.0, -90.0, 180.0, 90.0, N_LON, N_LAT)

    print("  Admin1 raster: burning ...", flush=True)
    id_grid = rasterize(
        shapes,
        out_shape=(N_LAT, N_LON),
        transform=transform,
        fill=-1,
        dtype=np.int32,
        all_touched=True,
    )

    n_assigned = np.sum(id_grid >= 0)
    print(f"    {n_assigned:,} / {N_LAT*N_LON:,} cells assigned "
          f"({100*n_assigned/(N_LAT*N_LON):.1f}%)")

    ds = xr.Dataset(
        {"admin1_id": (["latitude", "longitude"], id_grid)},
        coords={"latitude": OUT_LATS, "longitude": OUT_LONS},
    )
    ds.to_netcdf(str(ADMIN1_RASTER_CACHE))
    print(f"  Admin1 raster: done in {time.time()-t0:.0f}s")
    return id_grid, lookup


def load_heat(year):
    path = EHD_GLOBAL_DIR / f"exceedance_frequency_{year}.nc"
    ds   = xr.open_dataset(str(path))
    arr  = ds["t2m"].values.astype(np.float32)
    ds.close()
    return arr


def aggregate_year(year, admin1_grid, settlement, n_regions):
    heat = load_heat(year)

    valid  = (admin1_grid >= 0) & np.isfinite(heat) & (settlement > 0)
    ids_v  = admin1_grid[valid]
    heat_v = heat[valid]
    sett_v = settlement[valid].astype(np.float64)

    weighted_sum = np.bincount(ids_v, weights=heat_v * sett_v, minlength=n_regions)
    weight_sum   = np.bincount(ids_v, weights=sett_v,           minlength=n_regions)
    n_cells      = np.bincount(ids_v,                            minlength=n_regions)

    with np.errstate(divide="ignore", invalid="ignore"):
        heat_weighted = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)

    return {
        "heat_freq_weighted":  heat_weighted.astype(np.float32),
        "total_settlement_m2": weight_sum.astype(np.float64),
        "n_cells":             n_cells.astype(np.int32),
    }


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    wall_start = time.time()
    years = list(range(2000, 2025))

    print("=" * 60)
    print("Step 1: Ensure settlement rasters are cached")
    print("=" * 60)
    needed_epochs = sorted(set(epoch_for_year(y) for y in years))
    for epoch in needed_epochs:
        reproject_settlement(epoch)

    print(f"\n{'=' * 60}")
    print("Step 2: Rasterise admin1 polygons")
    print("=" * 60)
    admin1_grid, lookup = rasterize_admin1()
    n_regions = len(lookup)
    print(f"  {n_regions:,} admin1 regions")

    print(f"\n{'=' * 60}")
    print("Step 3: Settlement-weighted heat aggregation (2000-2024)")
    print("=" * 60)

    settle_cache = {}
    for epoch in needed_epochs:
        settle_cache[epoch] = load_settlement(epoch)
        print(f"  Loaded settlement {epoch}: max={settle_cache[epoch].max():.0f}  "
              f"nonzero={np.count_nonzero(settle_cache[epoch]):,}")

    rows = []
    for year in years:
        t0     = time.time()
        epoch  = epoch_for_year(year)
        result = aggregate_year(year, admin1_grid, settle_cache[epoch], n_regions)

        n_with_data = np.sum(result["n_cells"] > 0)
        for rid in range(n_regions):
            if result["n_cells"][rid] > 0:
                rows.append({
                    "raster_id":            rid,
                    "year":                 year,
                    "heat_freq_weighted":   float(result["heat_freq_weighted"][rid]),
                    "total_settlement_m2":  float(result["total_settlement_m2"][rid]),
                    "n_cells":              int(result["n_cells"][rid]),
                })

        print(f"  {year}  epoch={epoch}  raster={n_with_data:,}/{n_regions:,}  "
              f"{time.time()-t0:.1f}s")

    print(f"\n{'=' * 60}")
    print("Step 4: Build output panel")
    print("=" * 60)

    panel = pd.DataFrame(rows)
    panel = panel.merge(lookup, on="raster_id", how="left")

    col_order = ["GID_nmbr", "iso3", "year",
                 "heat_freq_weighted", "total_settlement_m2", "n_cells"]
    panel = panel[[c for c in col_order if c in panel.columns]]
    panel = panel.sort_values(["iso3", "GID_nmbr", "year"]).reset_index(drop=True)

    panel.to_parquet(str(HEAT_ADM1_PANEL_PATH), index=False)
    print(f"  Saved: {HEAT_ADM1_PANEL_PATH}  ({len(panel):,} rows, "
          f"{panel['GID_nmbr'].nunique():,} regions)")

    elapsed = time.time() - wall_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
