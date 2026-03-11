#!/usr/bin/env python -u
"""
Global summer extreme-heat exceedance frequency (EHD).

Tiles the globe into 30°lat × 60°lon chunks, computes the calendar-day
95th-percentile threshold using a vectorised numpy sort (~10–20× faster
than np.nanpercentile), then merges into global per-year NetCDF files.

  NH summer = JJA  (buffer months: May, Sep)
  SH summer = DJF  (buffer months: Nov, Mar)

Climatology period : 1990–2020  (configurable via CLIM_START / CLIM_END)
Frequency period   : 2000–2024  (configurable via FREQ_START / FREQ_END)

Input files (ERA5-Land 0.1°, daily Tmax):
    <ERA5_LAND_DIR>/era5_tmax_{YYYY}_{MM:02d}.nc
    Variable: t2m  (Kelvin)

    Set ERA5_LAND_DIR via environment variable or edit src/config.py.
    See data/raw/climate/README.md for download instructions.

Outputs:
    <EHD_GLOBAL_DIR>/exceedance_frequency_{year}.nc      — per-year grids
    <EHD_GLOBAL_DIR>/mean_exceedance_frequency_YYYY_YYYY.nc — time-mean
    <EHD_GLOBAL_DIR>/tiles/                              — tile cache
"""

import shutil
import sys
import time
import numpy as np
import xarray as xr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ERA5_LAND_DIR, EHD_GLOBAL_DIR, CACHE_DIR

sys.stdout.reconfigure(line_buffering=True)

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_DIR  = ERA5_LAND_DIR
OUTPUT_DIR = EHD_GLOBAL_DIR

CLIM_START, CLIM_END = 1990, 2020
FREQ_START, FREQ_END = 2000, 2024
WINDOW, PERCENTILE   = 15, 95
TILE_LAT, TILE_LON   = 30, 60   # degrees per tile

# File grid coordinates (ERA5-Land 0.1° resolution)
FILE_LATS = np.round(np.arange(90.0, -90.1, -0.1), 1)   # descending
FILE_LONS = np.round(np.arange(0.0, 360.0, 0.1), 1)     # ascending, 0-360

# Global output grid: -180 to 180 lon, 90 to -90 lat
OUT_LATS = FILE_LATS
OUT_LONS = np.round(np.arange(-180.0, 180.0, 0.1), 1)

HEMISPHERES = {
    "NH": dict(
        lat_bounds=(0, 90),       # includes equator
        summer=[6, 7, 8],
        buffer=[5, 9],
        is_djf=False,
        tag="jja",
    ),
    "SH": dict(
        lat_bounds=(-90, -0.1),   # excludes equator (NH has it)
        summer=[12, 1, 2],
        buffer=[11, 3],
        is_djf=True,
        tag="djf",
    ),
}


# ─── Fast vectorised nanpercentile ──────────────────────────────────────────
def fast_nanpercentile(data, pct, axis=0):
    """
    ~10–20× faster than np.nanpercentile.

    np.nanpercentile internally uses apply_along_axis (Python loop per
    grid point). This replaces it with: push NaN → +inf, sort, index.
    Fully vectorised in C.
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


# ─── Data loading (keeps native 0-360 lon) ──────────────────────────────────
def load_month(year, month, lat_sl, lon_sl):
    """Load one file, spatial-subset, K → °C.  Keeps native 0-360° lon."""
    path = INPUT_DIR / f"era5_tmax_{year}_{month:02d}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        da = ds["t2m"].sel(latitude=lat_sl, longitude=lon_sl).load()
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    return da - 273.15


def load_season(year, months, is_djf, lat_sl, lon_sl):
    """Load and concat months for one season (handles DJF year-crossing)."""
    parts = []
    for m in months:
        file_year = year - 1 if (is_djf and m >= 11) else year
        parts.append(load_month(file_year, m, lat_sl, lon_sl))
    return xr.concat(parts, dim="time")


# ─── Tile generation ────────────────────────────────────────────────────────
def make_tiles(lat_lo, lat_hi):
    """
    Generate non-overlapping tiles by partitioning the actual grid
    coordinate arrays, then building slices with ±0.05° buffers.
    """
    hemi_lats = FILE_LATS[(FILE_LATS >= lat_lo) & (FILE_LATS <= lat_hi)]
    n_lat = int(round(TILE_LAT / 0.1))
    n_lon = int(round(TILE_LON / 0.1))

    lat_chunks = [hemi_lats[i : i + n_lat] for i in range(0, len(hemi_lats), n_lat)]
    lon_chunks = [FILE_LONS[i : i + n_lon] for i in range(0, len(FILE_LONS), n_lon)]

    tiles = []
    for lats in lat_chunks:
        for lons in lon_chunks:
            lat_sl = slice(float(lats[0]) + 0.05, float(lats[-1]) - 0.05)
            lon_sl = slice(float(lons[0]) - 0.05, float(lons[-1]) + 0.05)
            label = (
                f"lat{lats[-1]:+06.1f}to{lats[0]:+06.1f}"
                f"_lon{lons[0]:05.1f}to{lons[-1]:05.1f}"
            )
            tiles.append(dict(lat_sl=lat_sl, lon_sl=lon_sl, label=label))
    return tiles


# ─── Tile processing ────────────────────────────────────────────────────────
def process_tile(tile, cfg):
    """
    Process one tile: compute calendar-day percentile, then yearly
    exceedance frequency. Returns (pctl_da, {year: freq_da}) or (None, None).
    """
    lat_sl, lon_sl = tile["lat_sl"], tile["lon_sl"]
    summer = cfg["summer"]
    buffer = cfg["buffer"]
    is_djf = cfg["is_djf"]
    all_months = sorted(summer + buffer)

    # Quick check: skip all-ocean tiles
    try:
        sample = load_month(2000, summer[1], lat_sl, lon_sl)
        if sample.size == 0 or np.all(np.isnan(sample.values)):
            return None, None
    except FileNotFoundError:
        return None, None

    # ── Load climatology (with buffer months) ──
    seasons = []
    for yr in range(CLIM_START, CLIM_END + 1):
        try:
            seasons.append(load_season(yr, all_months, is_djf, lat_sl, lon_sl))
        except FileNotFoundError:
            pass
    if not seasons:
        return None, None

    combined = xr.concat(seasons, dim="time")
    data = np.ascontiguousarray(combined.values, dtype=np.float32)
    all_doys = combined.time.dt.dayofyear.values
    target_doys = np.sort(
        np.unique(all_doys[combined.time.dt.month.isin(summer).values])
    )

    # ── Calendar-day percentile ──
    half = WINDOW // 2
    pctl_arr = np.empty(
        (len(target_doys), data.shape[1], data.shape[2]), dtype=np.float32
    )
    for i, doy in enumerate(target_doys):
        diff = np.abs(all_doys - doy)
        diff = np.minimum(diff, 366 - diff)
        pctl_arr[i] = fast_nanpercentile(data[diff <= half], PERCENTILE, axis=0)

    pctl = xr.DataArray(
        pctl_arr,
        dims=["dayofyear", "latitude", "longitude"],
        coords=dict(
            dayofyear=target_doys,
            latitude=combined.latitude,
            longitude=combined.longitude,
        ),
    )
    del combined, data, seasons

    # ── Exceedance frequency per year ──
    freqs = {}
    for yr in range(FREQ_START, FREQ_END + 1):
        try:
            season = load_season(yr, summer, is_djf, lat_sl, lon_sl)
        except FileNotFoundError:
            continue
        doys = season.time.dt.dayofyear.values
        thresh = (
            pctl.sel(dayofyear=doys)
            .assign_coords(dayofyear=season.time.values)
            .rename({"dayofyear": "time"})
        )
        freqs[yr] = (season > thresh).sum("time") / float(len(season.time))

    return pctl, freqs


# ─── Tile checkpointing ─────────────────────────────────────────────────────
TILE_DIR = OUTPUT_DIR / "tiles"


def tile_freq_path(label, yr):
    return TILE_DIR / f"freq_{label}_{yr}.nc"


def tile_pctl_path(label, tag):
    return TILE_DIR / f"pctl_{label}_{tag}.nc"


def tile_done(label, tag):
    """Check if a tile has already been fully processed."""
    if not tile_pctl_path(label, tag).exists():
        return False
    return tile_freq_path(label, FREQ_START).exists()


def save_tile(freqs, label, tag):
    """Save tile frequency results to individual files."""
    for yr, freq_da in freqs.items():
        freq_da.to_dataset(name="t2m").to_netcdf(tile_freq_path(label, yr))
    # Write a marker to flag completion
    tile_pctl_path(label, tag).touch()


# ─── Index-based global merge ────────────────────────────────────────────────
def merge_year_to_global(labels_tags, yr):
    """
    Merge all tile freq files for one year into a global grid using
    direct index assignment. Outputs -180..180 lon, 90..-90 lat.
    """
    n_lat, n_lon = len(OUT_LATS), len(OUT_LONS)
    global_data = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

    for label, tag in labels_tags:
        p = tile_freq_path(label, yr)
        if not p.exists():
            continue
        tile = xr.open_dataset(p).load()["t2m"]

        tile_lats = tile.latitude.values
        tile_lons_native = tile.longitude.values  # 0-360 convention

        lat_idx = np.round((90.0 - tile_lats) / 0.1).astype(int)
        lons_converted = tile_lons_native.copy()
        lons_converted[lons_converted >= 180] -= 360
        lon_idx = np.round((lons_converted + 180.0) / 0.1).astype(int)

        lat_idx = np.clip(lat_idx, 0, n_lat - 1)
        lon_idx = np.clip(lon_idx, 0, n_lon - 1)

        global_data[np.ix_(lat_idx, lon_idx)] = tile.values

    return xr.DataArray(
        global_data,
        dims=["latitude", "longitude"],
        coords={"latitude": OUT_LATS, "longitude": OUT_LONS},
        name="t2m",
    )


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    wall_start = time.time()

    print(f"ERA5-Land input directory : {INPUT_DIR}")
    print(f"Output directory          : {OUTPUT_DIR}")

    all_tile_labels = []   # (label, tag) tuples for all processed tiles

    for hname, cfg in HEMISPHERES.items():
        lat_lo, lat_hi = cfg["lat_bounds"]
        tag = cfg["tag"]

        print(f"\n{'=' * 60}")
        print(f"{hname}  ({tag.upper()})")
        print(f"{'=' * 60}")

        tiles = make_tiles(lat_lo, lat_hi)
        print(f"  {len(tiles)} tiles\n")

        for ti, tile in enumerate(tiles):
            label = tile["label"]
            t0 = time.time()
            print(f"  [{ti + 1:2d}/{len(tiles)}] {label}  ", end="", flush=True)

            if tile_done(label, tag):
                all_tile_labels.append((label, tag))
                p = tile_freq_path(label, FREQ_START)
                if p.exists():
                    ds = xr.open_dataset(p)
                    print(f"{ds.dims['latitude']}×{ds.dims['longitude']}  cached")
                    ds.close()
                else:
                    print("cached")
                continue

            pctl, freqs = process_tile(tile, cfg)

            if pctl is None:
                print("— ocean/empty, skipped")
                continue

            save_tile(freqs, label, tag)
            all_tile_labels.append((label, tag))
            nlat = pctl.sizes["latitude"]
            nlon = pctl.sizes["longitude"]
            print(f"{nlat}×{nlon}  {time.time() - t0:.0f}s")

    # ── Merge into global per-year files ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Merging into global per-year files")
    print(f"{'=' * 60}")

    freq_list = []
    for yr in range(FREQ_START, FREQ_END + 1):
        merged = merge_year_to_global(all_tile_labels, yr)
        n_valid = int(np.sum(np.isfinite(merged.values)))
        if n_valid == 0:
            print(f"  {yr} — no data")
            continue
        p = OUTPUT_DIR / f"exceedance_frequency_{yr}.nc"
        merged.to_dataset(name="t2m").to_netcdf(p)
        freq_list.append(merged)
        print(f"  {yr}  ({n_valid:,} land cells)")

    if freq_list:
        mean_freq = sum(freq_list) / len(freq_list)
        p = OUTPUT_DIR / f"mean_exceedance_frequency_{FREQ_START}_{FREQ_END}.nc"
        mean_freq.to_dataset(name="t2m").to_netcdf(p)
        print(f"\n  Mean → {p}")

    elapsed = time.time() - wall_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
