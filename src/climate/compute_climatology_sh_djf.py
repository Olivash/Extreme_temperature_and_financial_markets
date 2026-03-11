#!/usr/bin/env python -u
"""
Compute and save the global Southern Hemisphere DJF 95th-percentile
climatology as a single NetCDF file.

Processes each SH tile (lat < 0) in turn, computes the calendar-day
95th percentile using the 1990–2020 baseline, then index-assigns into
a global output array and saves once at the end.

Analogous to the NH JJA CONUS file (tmax_95th_pctl_doy_nh_jja_1990_2020.nc)
produced by compute_ehd_conus.py, but covering the full SH domain.

DJF definition (SH summer):
  Summer months : December, January, February
  Buffer months : November, March  (for ±7-day window at Dec 1 / Feb 28)
  Year convention: "DJF of year Y" = {Nov, Dec of Y-1} + {Jan, Feb, Mar of Y}

Input: ERA5-Land daily Tmax NetCDFs (one file per year-month)
  <ERA5_LAND_DIR>/era5_tmax_{YYYY}_{MM:02d}.nc
  Variable: t2m (Kelvin), 0.1° global grid, latitude descending, lon 0-360°

Output:
  <CLIMATOLOGY_DIR>/tmax_95th_pctl_doy_sh_djf_1990_2020.nc
    dims  : (dayofyear, latitude, longitude)
    lat   : SH only, 0.0 → -90.0 (descending)
    lon   : -180.0 → 179.9 (ascending)
    values: 95th-percentile Tmax threshold in °C

Runtime estimate: ~1–3 hours depending on available cores and I/O speed.

Usage:
    python src/climate/compute_climatology_sh_djf.py
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

# DJF season
SUMMER_MONTHS = [12, 1, 2]
BUFFER_MONTHS = [11, 3]
ALL_MONTHS    = sorted(SUMMER_MONTHS + BUFFER_MONTHS)   # [1, 2, 3, 11, 12]

# ERA5-Land 0.1° native grid
FILE_LATS = np.round(np.arange(90.0, -90.1, -0.1), 1)   # descending, 90→-90
FILE_LONS = np.round(np.arange(0.0, 360.0, 0.1), 1)     # ascending, 0-360

# Output grid (-180 to +180 lon, 0 to -90 lat for SH)
SH_LATS = FILE_LATS[FILE_LATS <= 0.0]                   # 0.0 → -90.0
OUT_LONS = np.round(np.arange(-180.0, 180.0, 0.1), 1)   # -180 → 179.9

N_LAT_SH  = len(SH_LATS)    # 901
N_LON     = len(OUT_LONS)    # 3600


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
    Load one DJF season (with buffer months) for the given lat/lon slice.
    DJF of year Y = {Nov, Dec of Y-1} + {Jan, Feb, Mar of Y}
    """
    parts = []
    for m in ALL_MONTHS:
        file_year = year - 1 if m >= 11 else year
        try:
            parts.append(load_month(file_year, m, lat_sl, lon_sl))
        except FileNotFoundError:
            pass
    if not parts:
        return None
    return xr.concat(parts, dim="time")


# ─── Tile generation ────────────────────────────────────────────────────────
def make_sh_tiles():
    """Generate 30°×60° tiles covering the Southern Hemisphere (lat ≤ 0)."""
    sh_lats = FILE_LATS[FILE_LATS <= 0.0]   # 0.0 → -90.0

    n_lat = int(round(TILE_LAT / 0.1))
    n_lon = int(round(TILE_LON / 0.1))

    lat_chunks = [sh_lats[i : i + n_lat] for i in range(0, len(sh_lats), n_lat)]
    lon_chunks = [FILE_LONS[i : i + n_lon] for i in range(0, len(FILE_LONS), n_lon)]

    tiles = []
    for lats in lat_chunks:
        for lons in lon_chunks:
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
    For one SH tile, load 1990–2020 DJF seasons, compute per-DOY 95th
    percentile. Returns xr.DataArray (dayofyear, lat, lon) or None if
    the tile is all ocean / data missing.
    """
    lat_sl, lon_sl = tile["lat_sl"], tile["lon_sl"]

    # Quick existence check
    try:
        sample = load_month(2000, 1, lat_sl, lon_sl)
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

    # Target DOYs = only core DJF days
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
    out_path = CLIMATOLOGY_DIR / f"tmax_{PERCENTILE}th_pctl_doy_sh_djf_{CLIM_START}_{CLIM_END}.nc"

    if out_path.exists():
        print(f"Output already exists: {out_path}")
        print("Delete it to recompute.  Exiting.")
        return

    wall_start = time.time()
    tiles = make_sh_tiles()
    n_tiles = len(tiles)
    print(f"Input directory : {INPUT_DIR}")
    print(f"Output          : {out_path}")
    print(f"SH tiles        : {n_tiles}")
    print(f"Lat range       : {SH_LATS[0]:.1f} → {SH_LATS[-1]:.1f}")
    print(f"Output grid     : {N_LAT_SH} lat × {N_LON} lon × DOYs")
    print()

    # Collect all unique target DOYs first (DJF: roughly DOY 335–365 + 1–59)
    # We'll determine this from a sample tile.  Approximate set:
    djf_doys_approx = []
    for yr in [2000]:
        try:
            s = load_season(yr, tiles[0]["lat_sl"], tiles[0]["lon_sl"])
            if s is not None:
                djf_doys_approx = sorted(np.unique(
                    s.time.dt.dayofyear.values[
                        s.time.dt.month.isin(SUMMER_MONTHS).values
                    ]
                ))
                break
        except Exception:
            pass

    # Unique DJF DOYs (Dec: ~335–365, Jan: 1–31, Feb: 32–59)
    # Build global output array using a dict keyed by doy
    doy_to_idx = {}
    global_pctl = {}   # doy → (N_LAT_SH, N_LON) float32 array, starts as NaN

    print(f"Approximate DJF DOYs: {djf_doys_approx[:5]} ... {djf_doys_approx[-5:]}"
          f" (n={len(djf_doys_approx)})")

    # Checkpoint directory — one .npz file per completed tile
    ckpt_dir = CLIMATOLOGY_DIR / ".sh_djf_tile_cache"
    ckpt_dir.mkdir(exist_ok=True)

    # Process tiles
    for ti, tile in enumerate(tiles):
        label = tile["label"]
        ckpt_path = ckpt_dir / f"{label}.npz"
        t0 = time.time()
        print(f"\n[{ti+1:2d}/{n_tiles}] {label}")

        # Resume: load checkpoint if it exists
        if ckpt_path.exists():
            ck = np.load(ckpt_path)
            tile_doys        = ck["doys"]
            tile_lats        = ck["lats"]
            tile_lons_native = ck["lons"]
            pctl_values      = ck["pctl"]
            if len(tile_doys) == 0:
                print(f"  → ocean/empty, skipped (cached)  0s")
                continue
            n_cells = int(np.sum(np.isfinite(pctl_values[0])))
            print(f"  → cached ({n_cells:,} land cells, {len(tile_doys)} DOYs)  0s")
        else:
            pctl = compute_tile_climatology(tile)
            if pctl is None:
                print(f"  → ocean/empty, skipped ({time.time()-t0:.0f}s)")
                # Save a sentinel so we don't recompute ocean tiles either
                np.savez(ckpt_path, doys=np.array([]), lats=np.array([]),
                         lons=np.array([]), pctl=np.array([]))
                continue

            tile_doys        = pctl.dayofyear.values
            tile_lats        = pctl.latitude.values
            tile_lons_native = pctl.longitude.values
            pctl_values      = pctl.values
            n_cells = int(np.sum(np.isfinite(pctl_values[0])))
            print(f"  → {pctl.sizes['latitude']}×{pctl.sizes['longitude']} "
                  f"({n_cells:,} land cells, {len(tile_doys)} DOYs)  {time.time()-t0:.0f}s")
            np.savez(ckpt_path, doys=tile_doys, lats=tile_lats,
                     lons=tile_lons_native, pctl=pctl_values)

        if len(tile_doys) == 0:
            continue

        # Global lat indices for this tile
        lat_idx = np.round((0.0 - tile_lats) / 0.1).astype(int)   # 0.0 is top of SH
        lat_idx = np.clip(lat_idx, 0, N_LAT_SH - 1)

        # Convert 0-360 → -180..180 for output lon index
        lons_conv = tile_lons_native.copy()
        lons_conv[lons_conv >= 180] -= 360
        lon_idx = np.round((lons_conv + 180.0) / 0.1).astype(int)
        lon_idx = np.clip(lon_idx, 0, N_LON - 1)

        for i, doy in enumerate(tile_doys):
            if doy not in global_pctl:
                global_pctl[doy] = np.full((N_LAT_SH, N_LON), np.nan, dtype=np.float32)
            global_pctl[doy][np.ix_(lat_idx, lon_idx)] = pctl_values[i]

    if not global_pctl:
        print("\nERROR: no tile data found — check ERA5_LAND_DIR.")
        sys.exit(1)

    # ── Assemble and save ─────────────────────────────────────────────────────
    # Save as an unnamed DataArray — same format as the NH JJA CONUS file
    # (variable name: __xarray_dataarray_variable__, loadable via xr.open_dataarray)
    all_doys = sorted(global_pctl.keys())
    pctl_cube = np.stack([global_pctl[d] for d in all_doys], axis=0)   # (doy, lat, lon)

    da = xr.DataArray(
        pctl_cube,
        dims=["dayofyear", "latitude", "longitude"],
        coords={
            "dayofyear": all_doys,
            "latitude":  SH_LATS,
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

    # Clean up tile cache now that the final file is written
    import shutil
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print("  Tile cache cleaned up.")


if __name__ == "__main__":
    main()
