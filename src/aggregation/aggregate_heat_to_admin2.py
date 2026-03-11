#!/usr/bin/env python -u
"""
Aggregate global EHD exceedance frequency to admin2 polygons.

Weights by GHS Built-up Surface so the result reflects heat exposure
where people actually live and work.

Pipeline:
  1. Reproject GHS settlement rasters (Mollweide 100m) → WGS84 0.1°
     (processed in latitude bands to keep memory < 2 GB per band).
  2. Rasterise GADM admin2 polygons onto the same 0.1° grid.
  3. For each year: compute settlement-weighted mean EHD per admin2.
  4. Merge with GDP per capita from the GeoPackage; write outputs.

Inputs (set paths in src/config.py or via environment variables):
  <EHD_GLOBAL_DIR>/exceedance_frequency_{year}.nc          — from compute_ehd_global.py
  <SETTLEMENT_DIR>/GHS_BUILT_S_E{epoch}_GLOBE_R2023A_54009_100_V1_0.tif
  <ADMIN2_GPKG>    — polyg_adm2_gdp_perCapita_1990_2022.gpkg

  See data/raw/shapes/README.md and data/raw/gdp/README.md for data sources.

Outputs:
  <PANEL_DIR>/admin2_heat_settlement_weighted.parquet
  <PANEL_DIR>/admin2_heat_settlement_weighted.gpkg
  <CACHE_DIR>/admin2_raster_01deg.nc    — rasterised admin2 grid (cached)
  <CACHE_DIR>/settlement_01deg_{epoch}.nc  — reprojected settlement rasters
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EHD_GLOBAL_DIR, SETTLEMENT_DIR, ADMIN2_GPKG,
    CACHE_DIR, PANEL_DIR,
)

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Paths ──────────────────────────────────────────────────────────────────
HEAT_DIR   = EHD_GLOBAL_DIR
OUTPUT_DIR = PANEL_DIR

# ─── Grid parameters (must match heat NetCDFs) ─────────────────────────────
N_LAT, N_LON = 1801, 3600
OUT_LATS = np.round(np.arange(90.0, -90.1, -0.1), 1)
OUT_LONS = np.round(np.arange(-180.0, 180.0, 0.1), 1)
assert len(OUT_LATS) == N_LAT and len(OUT_LONS) == N_LON

# ─── Settlement epoch mapping ──────────────────────────────────────────────
SETTLEMENT_EPOCHS = [2000, 2005, 2010, 2015, 2020, 2025]


def epoch_for_year(year):
    """Map a heat year (2000–2024) to the nearest 5-year settlement epoch."""
    if year <= 2002:
        return 2000
    elif year <= 2007:
        return 2005
    elif year <= 2012:
        return 2010
    elif year <= 2017:
        return 2015
    elif year <= 2022:
        return 2020
    else:
        return 2025


# ─── Step 1: Reproject settlement raster to 0.1° ──────────────────────────
def settle_tif_path(epoch):
    return SETTLEMENT_DIR / f"GHS_BUILT_S_E{epoch}_GLOBE_R2023A_54009_100_V1_0.tif"


def settle_cache_path(epoch):
    return CACHE_DIR / f"settlement_01deg_{epoch}.nc"


def reproject_settlement(epoch, band_height_deg=10):
    """
    Reproject one GHS settlement TIF (Mollweide 100m) → WGS84 0.1°.
    Processes in latitude bands to limit memory usage.
    """
    cache_path = settle_cache_path(epoch)
    if cache_path.exists():
        print(f"  Settlement {epoch}: cached ({cache_path.name})")
        return

    src_path = settle_tif_path(epoch)
    print(f"  Settlement {epoch}: reprojecting {src_path.name} ...", flush=True)
    t0 = time.time()

    global_arr = np.zeros((N_LAT, N_LON), dtype=np.float32)
    band_height_cells = int(round(band_height_deg / 0.1))
    n_bands = (N_LAT + band_height_cells - 1) // band_height_cells

    with rasterio.open(src_path) as src:
        for bi in range(n_bands):
            row_start = bi * band_height_cells
            row_end = min(row_start + band_height_cells, N_LAT)
            n_rows = row_end - row_start

            lat_top = 90.0 - row_start * 0.1
            lat_bot = 90.0 - row_end * 0.1
            lat_top = min(lat_top, 90.0)
            lat_bot = max(lat_bot, -90.0)

            if lat_top - lat_bot < 0.05:
                print(f"    band {bi+1}/{n_bands}  "
                      f"lat [{lat_top:+.1f}, {lat_bot:+.1f}]  "
                      f"skipped (degenerate)", flush=True)
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
            print(f"    band {bi+1}/{n_bands}  "
                  f"lat [{lat_top:+.1f}, {lat_bot:+.1f}]  "
                  f"max={dst_band.max():.0f}", flush=True)

    ds = xr.Dataset(
        {"built_surface": (["latitude", "longitude"], global_arr)},
        coords={"latitude": OUT_LATS, "longitude": OUT_LONS},
    )
    ds.attrs["units"] = "m2 built-up per grid cell"
    ds.attrs["source_epoch"] = epoch
    ds.to_netcdf(cache_path)
    print(f"  Settlement {epoch}: done in {time.time() - t0:.0f}s → {cache_path.name}")


def load_settlement(epoch):
    """Load cached 0.1° settlement grid."""
    ds = xr.open_dataset(settle_cache_path(epoch))
    arr = ds["built_surface"].values
    ds.close()
    return arr


# ─── Step 2: Rasterise admin2 polygons to 0.1° ────────────────────────────
ADMIN2_RASTER_CACHE = CACHE_DIR / "admin2_raster_01deg.nc"
ADMIN2_LOOKUP_CACHE = CACHE_DIR / "admin2_lookup.csv"


def rasterize_admin2():
    """
    Burn admin2 polygon IDs into a (1801, 3600) grid.
    Returns (id_grid, lookup_df).
    """
    if ADMIN2_RASTER_CACHE.exists() and ADMIN2_LOOKUP_CACHE.exists():
        print("  Admin2 raster: cached")
        ds = xr.open_dataset(ADMIN2_RASTER_CACHE)
        id_grid = ds["admin2_id"].values
        ds.close()
        lookup = pd.read_csv(ADMIN2_LOOKUP_CACHE)
        return id_grid, lookup

    print("  Admin2 raster: loading polygons ...", flush=True)
    t0 = time.time()
    gdf = gpd.read_file(ADMIN2_GPKG)
    print(f"    {len(gdf)} polygons loaded ({time.time()-t0:.0f}s)")

    lookup = gdf[["GID_2", "adm2ID", "iso3", "NAME_2"]].copy()
    lookup["raster_id"] = np.arange(len(gdf))
    lookup.to_csv(ADMIN2_LOOKUP_CACHE, index=False)

    shapes = [(geom, rid) for geom, rid in zip(gdf.geometry, lookup["raster_id"])]
    transform = from_bounds(-180.0, -90.0, 180.0, 90.0, N_LON, N_LAT)

    print("  Admin2 raster: burning ...", flush=True)
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
        {"admin2_id": (["latitude", "longitude"], id_grid)},
        coords={"latitude": OUT_LATS, "longitude": OUT_LONS},
    )
    ds.to_netcdf(ADMIN2_RASTER_CACHE)
    print(f"  Admin2 raster: done in {time.time()-t0:.0f}s")
    return id_grid, lookup


# ─── Step 3: Load heat frequency ──────────────────────────────────────────
def load_heat(year):
    """Load exceedance_frequency_{year}.nc and return (1801, 3600) array."""
    path = HEAT_DIR / f"exceedance_frequency_{year}.nc"
    ds = xr.open_dataset(path)
    arr = ds["t2m"].values.astype(np.float32)
    ds.close()
    return arr


# ─── Step 4: Settlement-weighted aggregation ──────────────────────────────
def aggregate_year(year, admin2_grid, settlement, n_regions):
    """
    Settlement-weighted mean heat frequency per admin2 region.
    Returns dict with arrays of length n_regions.
    """
    heat = load_heat(year)

    valid = (admin2_grid >= 0) & np.isfinite(heat) & (settlement > 0)
    ids_v  = admin2_grid[valid]
    heat_v = heat[valid]
    sett_v = settlement[valid].astype(np.float64)

    weighted_sum = np.bincount(ids_v, weights=heat_v * sett_v, minlength=n_regions)
    weight_sum   = np.bincount(ids_v, weights=sett_v, minlength=n_regions)
    n_cells      = np.bincount(ids_v, minlength=n_regions)

    with np.errstate(divide="ignore", invalid="ignore"):
        heat_weighted = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)

    heat_sum_uw = np.bincount(ids_v, weights=heat_v, minlength=n_regions)
    n_cells_any = np.bincount(admin2_grid[valid], minlength=n_regions)
    with np.errstate(divide="ignore", invalid="ignore"):
        heat_unweighted = np.where(n_cells_any > 0, heat_sum_uw / n_cells_any, np.nan)

    return {
        "heat_freq_weighted":   heat_weighted.astype(np.float32),
        "heat_freq_unweighted": heat_unweighted.astype(np.float32),
        "total_settlement_m2":  weight_sum.astype(np.float64),
        "n_cells":              n_cells.astype(np.int32),
    }


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wall_start = time.time()

    years = list(range(2000, 2025))

    # ── Step 1: Reproject settlements ─────────────────────────────────────
    print("=" * 60)
    print("Step 1: Reproject settlement rasters to 0.1°")
    print("=" * 60)
    needed_epochs = sorted(set(epoch_for_year(y) for y in years))
    for epoch in needed_epochs:
        reproject_settlement(epoch)

    # ── Step 2: Rasterise admin2 polygons ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 2: Rasterise admin2 polygons")
    print("=" * 60)
    admin2_grid, lookup = rasterize_admin2()
    n_regions = len(lookup)
    print(f"  {n_regions} admin2 regions")

    # ── Step 3: Year-by-year aggregation ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 3: Settlement-weighted aggregation (2000–2024)")
    print("=" * 60)

    settle_cache = {}
    for epoch in needed_epochs:
        settle_cache[epoch] = load_settlement(epoch)
        print(f"  Loaded settlement {epoch}: "
              f"max={settle_cache[epoch].max():.0f}, "
              f"nonzero={np.count_nonzero(settle_cache[epoch]):,}")

    rows = []
    for year in years:
        t0 = time.time()
        epoch = epoch_for_year(year)
        settlement = settle_cache[epoch]

        result = aggregate_year(year, admin2_grid, settlement, n_regions)

        for rid in range(n_regions):
            if result["n_cells"][rid] > 0:
                rows.append({
                    "raster_id":           rid,
                    "year":                year,
                    "heat_freq_weighted":  result["heat_freq_weighted"][rid],
                    "heat_freq_unweighted":result["heat_freq_unweighted"][rid],
                    "total_settlement_m2": result["total_settlement_m2"][rid],
                    "n_cells":             result["n_cells"][rid],
                })

        n_with_data = np.sum(result["n_cells"] > 0)
        mean_heat   = np.nanmean(result["heat_freq_weighted"])
        print(f"  {year}  epoch={epoch}  "
              f"regions={n_with_data:,}/{n_regions:,}  "
              f"mean_heat={mean_heat:.4f}  "
              f"{time.time()-t0:.1f}s")

    # ── Step 4: Build panel and merge with admin2 attributes ──────────────
    print(f"\n{'=' * 60}")
    print("Step 4: Build output panel")
    print("=" * 60)

    panel = pd.DataFrame(rows)
    print(f"  Raw panel: {len(panel):,} rows ({panel['raster_id'].nunique():,} regions × "
          f"{panel['year'].nunique()} years)")

    panel = panel.merge(lookup, on="raster_id", how="left")

    # Merge with GDP per capita columns from the GeoPackage
    print("  Loading GDP data ...", flush=True)
    gdf_full = gpd.read_file(ADMIN2_GPKG)
    gdp_cols = [c for c in gdf_full.columns if c.isdigit()]
    gdp_wide = gdf_full[["GID_2"] + gdp_cols].copy()

    gdp_long = gdp_wide.melt(
        id_vars=["GID_2"],
        value_vars=gdp_cols,
        var_name="year",
        value_name="gdp_per_capita",
    )
    gdp_long["year"] = gdp_long["year"].astype(int)
    panel = panel.merge(gdp_long, on=["GID_2", "year"], how="left")

    col_order = [
        "GID_2", "adm2ID", "iso3", "NAME_2", "year",
        "heat_freq_weighted", "heat_freq_unweighted",
        "total_settlement_m2", "n_cells",
        "gdp_per_capita", "raster_id",
    ]
    panel = panel[[c for c in col_order if c in panel.columns]]
    panel = panel.sort_values(["iso3", "GID_2", "year"]).reset_index(drop=True)

    # ── Save outputs ──────────────────────────────────────────────────────
    parquet_path = OUTPUT_DIR / "admin2_heat_settlement_weighted.parquet"
    panel.to_parquet(parquet_path, index=False)
    print(f"  Parquet: {parquet_path}  ({len(panel):,} rows)")

    print("  Building GeoPackage ...", flush=True)
    geom_lookup = gdf_full[["GID_2", "geometry"]].copy()
    panel_geo = panel.merge(geom_lookup, on="GID_2", how="left")
    panel_geo = gpd.GeoDataFrame(panel_geo, geometry="geometry", crs="EPSG:4326")
    gpkg_path = OUTPUT_DIR / "admin2_heat_settlement_weighted.gpkg"
    panel_geo.to_file(gpkg_path, driver="GPKG")
    print(f"  GeoPackage: {gpkg_path}")

    # ── Verification ──────────────────────────────────────────────────────
    annual_mean = panel.groupby("year")["heat_freq_weighted"].mean()
    print("\n  Global mean heat frequency by year:")
    for y, v in annual_mean.items():
        bar = "#" * int(v * 200)
        print(f"    {y}  {v:.4f}  {bar}")

    vmin = panel["heat_freq_weighted"].min()
    vmax = panel["heat_freq_weighted"].max()
    print(f"\n  Value range: [{vmin:.4f}, {vmax:.4f}]")
    if vmin < 0 or vmax > 1:
        print("  WARNING: values outside [0, 1] — investigate!")

    elapsed = time.time() - wall_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
