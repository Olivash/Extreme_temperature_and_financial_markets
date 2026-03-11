#!/usr/bin/env python -u
"""
Compute and save the global Northern Hemisphere JJA 95th-percentile
climatology as a single NetCDF file.

Processes each NH tile (lat ≥ 0) in turn, computes the calendar-day
95th percentile using the 1990–2020 baseline, then index-assigns into
a global output array and saves once at the end.

Analogous to compute_climatology_sh_djf.py but for the full NH domain.
The existing CONUS file (tmax_95th_pctl_doy_nh_jja_1990_2020.nc) covers
only 25°N–50°N, 125°W–65°W.  This script produces the full 0°–90°N
global coverage.

JJA definition (NH summer):
  Summer months : June, July, August
  Buffer months : May, September  (for ±7-day window at Jun 1 / Aug 31)
  Year convention: straightforward — all months within the same calendar year

Input: ERA5-Land daily Tmax NetCDFs (one file per year-month)
  <ERA5_LAND_DIR>/era5_tmax_{YYYY}_{MM:02d}.nc
  Variable: t2m (Kelvin), 0.1° global grid, latitude descending, lon 0-360°

Output:
  <CLIMATOLOGY_DIR>/tmax_95th_pctl_doy_nh_jja_global_1990_2020.nc
    dims  : (dayofyear, latitude, longitude)
    lat   : NH only, 90.0 → 0.0 (descending)
    lon   : -180.0 → 179.9 (ascending)
    values: 95th-percentile Tmax threshold in °C

Runtime estimate: ~1–3 hours depending on available cores and I/O speed.

Usage:
    python src/climate/compute_climatology_nh_jja_global.py
"""

import sys
import time
import numpy as np
import xarray as xr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ERA5_LAND_DIR, CLIMATOLOGY_DIR

sys.stdout.reconfigure(line_buffering=True)

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_DIR = ERA5_LAND_DIR

CLIM_START, CLIM_END = 1990, 2020
WINDOW, PERCENTILE   = 15, 95
TILE_LAT, TILE_LON   = 30, 60   # degrees per tile

# JJA season
SUMMER_MONTHS = [6, 7, 8]
BUFFER_MONTHS = [5, 9]
ALL_MONTHS    = sorted(SUMMER_MONTHS + BUFFER_MONTHS)   # [5, 6, 7, 8, 9]

# ERA5-Land 0.1° native grid
FILE_LATS = np.round(np.arange(90.0, -90.1, -0.1), 1)   # descending, 90→-90
FILE_LONS = np.round(np.arange(0.0, 360.0, 0.1), 1)     # ascending, 0-360

# Output grid (-180 to +180 lon, 90 to 0 lat for NH)
NH_LATS  = FILE_LATS[FILE_LATS >= 0.0]                  # 90.0 → 0.0 (descending)
OUT_LONS = np.round(np.arange(-180.0, 180.0, 0.1), 1)   # -180 → 179.9

N_LAT_NH = len(NH_LATS)    # 901
N_LON    = len(OUT_LONS)   # 3600


# ─── Fast vectorised nanpercentile ──────────────────────────────────────────
def fast_nanpercentile(data, pct, axis=0):
    """
    ~10–20x faster than np.nanpercentile for large arrays.
    Push NaN → +inf, sort, index by percentile rank.
    """
    filled = np.where(np.isnan(data), np.inf, data)
    s = np.sort(filled, axis=axis)
    n_valid = np.sum(np.isfinite(s), axis=axis)
    idx = np.clip(
        np.round(pct / 100.0 * (n_valid - 1)).astype(np.intp),
        0, data.shape[axis] - 1,
    )
    result = np.take_along_axis(s, np.expand_dims(idx, axis), axis).squeeze(axis)
    return np.where(n_valid > 0, result, np.nan).astype(np.float32)


# ─── Data loading ────────────────────────────────────────────────────────────
def load_month(year, month, lat_sl, lon_sl):
    """Load one monthly file, subset, K → °C. Keeps native 0-360° lon."""
    path = INPUT_DIR / f"era5_tmax_{year}_{month:02d}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        da = ds["t2m"].sel(latitude=lat_sl, longitude=lon_sl).load()
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    return da - 273.15


def load_season(year, lat_sl, lon_sl):
    """
    Load one JJA season (with buffer months) for the given lat/lon slice.
    All months are within the same calendar year.
    """
    parts = []
    for m in ALL_MONTHS:
        try:
            parts.append(load_month(year, m, lat_sl, lon_sl))
        except FileNotFoundError:
            pass
    if not parts:
        return None
    return xr.concat(parts, dim="time")


# ─── Tile generation ────────────────────────────────────────────────────────
def make_nh_tiles():
    """Generate 30°×60° tiles covering the Northern Hemisphere (lat ≥ 0)."""
    nh_lats = FILE_LATS[FILE_LATS >= 0.0]   # 90.0 → 0.0 (descending)

    n_lat = int(round(TILE_LAT / 0.1))
    n_lon = int(round(TILE_LON / 0.1))

    lat_chunks = [nh_lats[i : i + n_lat] for i in range(0, len(nh_lats), n_lat)]
    lon_chunks = [FILE_LONS[i : i + n_lon] for i in range(0, len(FILE_LONS), n_lon)]

    tiles = []
    for lats in lat_chunks:
        for lons in lon_chunks:
            # lats is descending: lats[0] is max, lats[-1] is min
            lat_sl = slice(float(lats[0]) + 0.05, float(lats[-1]) - 0.05)
            lon_sl = slice(float(lons[0]) - 0.05, float(lons[-1]) + 0.05)
            label = (f"lat{lats[-1]:+06.1f}to{lats[0]:+06.1f}"
                     f"_lon{lons[0]:05.1f}to{lons[-1]:05.1f}")
            tiles.append(dict(lat_sl=lat_sl, lon_sl=lon_sl,
                              lats=lats, lons=lons, label=label))
    return tiles


# ─── Tile climatology computation ────────────────────────────────────────────
def compute_tile_climatology(tile):
    """
    For one NH tile, load 1990–2020 JJA seasons, compute per-DOY 95th
    percentile. Returns xr.DataArray (dayofyear, lat, lon) or None if
    the tile is all ocean / data missing.
    """
    lat_sl, lon_sl = tile["lat_sl"], tile["lon_sl"]

    # Quick existence check
    try:
        sample = load_month(2000, 7, lat_sl, lon_sl)
        if sample.size == 0 or np.all(np.isnan(sample.values)):
            return None
    except FileNotFoundError:
        return None

    # Load climatology seasons
    seasons = []
    for yr in range(CLIM_START, CLIM_END + 1):
        s = load_season(yr, lat_sl, lon_sl)
        if s is not None:
            seasons.append(s)
    if not seasons:
        return None

    combined = xr.concat(seasons, dim="time")
    data     = np.ascontiguousarray(combined.values, dtype=np.float32)
    all_doys = combined.time.dt.dayofyear.values

    # Target DOYs = only core JJA days
    target_doys = np.sort(np.unique(
        all_doys[combined.time.dt.month.isin(SUMMER_MONTHS).values]
    ))
    if len(target_doys) == 0:
        return None

    # Calendar-day percentile with ±7-day window
    half     = WINDOW // 2
    n_doy    = len(target_doys)
    n_lat    = data.shape[1]
    n_lon    = data.shape[2]
    pctl_arr = np.empty((n_doy, n_lat, n_lon), dtype=np.float32)

    for i, doy in enumerate(target_doys):
        diff = np.abs(all_doys - doy)
        diff = np.minimum(diff, 366 - diff)
        pctl_arr[i] = fast_nanpercentile(data[diff <= half], PERCENTILE, axis=0)

    return xr.DataArray(
        pctl_arr,
        dims=["dayofyear", "latitude", "longitude"],
        coords=dict(
            dayofyear=target_doys,
            latitude=combined.latitude,
            longitude=combined.longitude,
        ),
    )


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    CLIMATOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLIMATOLOGY_DIR / f"tmax_{PERCENTILE}th_pctl_doy_nh_jja_global_{CLIM_START}_{CLIM_END}.nc"

    if out_path.exists():
        print(f"Output already exists: {out_path}")
        print("Delete it to recompute.  Exiting.")
        return

    wall_start = time.time()
    tiles = make_nh_tiles()
    n_tiles = len(tiles)
    print(f"Input directory : {INPUT_DIR}")
    print(f"Output          : {out_path}")
    print(f"NH tiles        : {n_tiles}")
    print(f"Lat range       : {NH_LATS[0]:.1f} → {NH_LATS[-1]:.1f}")
    print(f"Output grid     : {N_LAT_NH} lat × {N_LON} lon × DOYs")
    print()

    # Determine JJA DOYs from a sample tile
    jja_doys_approx = []
    for yr in [2000]:
        try:
            s = load_season(yr, tiles[0]["lat_sl"], tiles[0]["lon_sl"])
            if s is not None:
                jja_doys_approx = sorted(np.unique(
                    s.time.dt.dayofyear.values[
                        s.time.dt.month.isin(SUMMER_MONTHS).values
                    ]
                ))
                break
        except Exception:
            pass

    global_pctl = {}   # doy → (N_LAT_NH, N_LON) float32 array

    print(f"Approximate JJA DOYs: {jja_doys_approx[:5]} ... {jja_doys_approx[-5:]}"
          f" (n={len(jja_doys_approx)})")

    # Process tiles
    for ti, tile in enumerate(tiles):
        label = tile["label"]
        t0 = time.time()
        print(f"\n[{ti+1:2d}/{n_tiles}] {label}")

        pctl = compute_tile_climatology(tile)
        if pctl is None:
            print(f"  → ocean/empty, skipped ({time.time()-t0:.0f}s)")
            continue

        tile_lats = pctl.latitude.values
        tile_lons_native = pctl.longitude.values   # 0-360
        tile_doys = pctl.dayofyear.values

        # Global lat indices: NH_LATS[0] = 90.0 at index 0
        lat_idx = np.round((90.0 - tile_lats) / 0.1).astype(int)
        lat_idx = np.clip(lat_idx, 0, N_LAT_NH - 1)

        # Convert 0-360 → -180..180 for output lon index
        lons_conv = tile_lons_native.copy()
        lons_conv[lons_conv >= 180] -= 360
        lon_idx = np.round((lons_conv + 180.0) / 0.1).astype(int)
        lon_idx = np.clip(lon_idx, 0, N_LON - 1)

        for i, doy in enumerate(tile_doys):
            if doy not in global_pctl:
                global_pctl[doy] = np.full((N_LAT_NH, N_LON), np.nan, dtype=np.float32)
            global_pctl[doy][np.ix_(lat_idx, lon_idx)] = pctl.values[i]

        n_doy   = len(tile_doys)
        n_cells = np.sum(np.isfinite(pctl.values[0]))
        print(f"  → {pctl.sizes['latitude']}×{pctl.sizes['longitude']} "
              f"({n_cells:,} land cells, {n_doy} DOYs)  {time.time()-t0:.0f}s")

    if not global_pctl:
        print("\nERROR: no tile data found — check ERA5_LAND_DIR.")
        sys.exit(1)

    # ── Assemble and save ─────────────────────────────────────────────────────
    all_doys  = sorted(global_pctl.keys())
    pctl_cube = np.stack([global_pctl[d] for d in all_doys], axis=0)

    da = xr.DataArray(
        pctl_cube,
        dims=["dayofyear", "latitude", "longitude"],
        coords={
            "dayofyear": all_doys,
            "latitude":  NH_LATS,
            "longitude": OUT_LONS,
        },
    )
    da.to_netcdf(out_path)

    elapsed = time.time() - wall_start
    n_valid = int(np.sum(np.isfinite(pctl_cube)))
    file_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved: {out_path}")
    print(f"  Shape: {pctl_cube.shape}  ({n_valid:,} finite values)")
    print(f"  File size: {file_mb:.1f} MB")
    print(f"  Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
