#!/usr/bin/env python
"""
Fast computation of summer extreme-heat exceedance frequency — CONUS domain.

Replaces the global tiled version with a CONUS spatial subset, giving
~21× fewer grid points and enabling faster iteration.

Algorithm:
  1. Subset ERA5-Land to CONUS bounding box on load.
  2. Build calendar-day 95th-percentile thresholds from the 1990–2020
     climatology using a ±7-day centred window.
  3. For each year 2000–2024, count the fraction of summer (JJA) days
     exceeding the per-DOY threshold.

Input files (ERA5-Land 0.1°, daily Tmax):
    <ERA5_LAND_DIR>/era5_tmax_{YYYY}_{MM:02d}.nc
    Variable: t2m  (Kelvin)

    Set ERA5_LAND_DIR via environment variable or edit src/config.py.
    See data/raw/climate/README.md for download instructions.

Outputs:
    <CLIMATOLOGY_DIR>/tmax_95th_pctl_doy_nh_jja_{CLIM_START}_{CLIM_END}.nc
        Calendar-day percentile thresholds (saved for reuse).
    <EHD_GLOBAL_DIR>/exceedance_frequency_{year}.nc
        Per-year exceedance frequency grids (CONUS domain).
    <EHD_GLOBAL_DIR>/mean_exceedance_frequency_{FREQ_START}_{FREQ_END}.nc

Output dimensions: 251 lat × 601 lon
    (50.0°N – 25.0°N,  125.0°W – 65.0°W)
"""

import time
import numpy as np
import xarray as xr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ERA5_LAND_DIR, EHD_GLOBAL_DIR, CLIMATOLOGY_DIR

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_DIR      = ERA5_LAND_DIR
OUTPUT_DIR     = EHD_GLOBAL_DIR
PCTL_SAVE_DIR  = CLIMATOLOGY_DIR

CLIM_START = 1990
CLIM_END   = 2020
FREQ_START = 2000
FREQ_END   = 2024

SUMMER_MONTHS    = [6, 7, 8]
BUFFER_MONTHS    = [5, 9]            # for full ±7-day window at Jun 1 / Aug 31
ALL_LOAD_MONTHS  = sorted(BUFFER_MONTHS + SUMMER_MONTHS)  # [5, 6, 7, 8, 9]

WINDOW     = 15   # centred window in days (±7)
PERCENTILE = 95

# CONUS bounding box in the file's native coordinate conventions:
#   Latitude  : descending (90 → -90), so slice high → low
#   Longitude : 0–360°, so 125°W = 235°E, 65°W = 295°E
# Small buffer (0.05°) avoids floating-point boundary exclusion.
# Output dimensions: 251 lat × 601 lon
LAT_SLICE = slice(50.05, 24.95)
LON_SLICE = slice(234.95, 295.05)

N_WORKERS = 8


# ─── Data loading ────────────────────────────────────────────────────────────
def load_month(year, month):
    """Load one monthly file, subset to CONUS, convert K → °C."""
    path = INPUT_DIR / f"era5_tmax_{year}_{month:02d}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        da = ds["t2m"].sel(latitude=LAT_SLICE, longitude=LON_SLICE).load()
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    # Convert lon 0-360 → -180..180  (lat is already descending)
    da = da.assign_coords(longitude=da.longitude.values - 360)
    return da - 273.15


def load_months(year, months):
    """Load and concatenate several months for one year, CONUS only."""
    return xr.concat([load_month(year, m) for m in months], dim="time")


# ─── Calendar-day percentile ─────────────────────────────────────────────────
def _percentile_chunk(args):
    """Worker: compute percentile for a chunk of target DOYs."""
    target_doys, all_doys, data, half, pct = args
    out = np.empty((len(target_doys), data.shape[1], data.shape[2]),
                   dtype=np.float32)
    for i, doy in enumerate(target_doys):
        diff = np.abs(all_doys - doy)
        diff = np.minimum(diff, 366 - diff)
        out[i] = np.nanpercentile(data[diff <= half], pct, axis=0)
    return out


def calendar_day_percentile(combined, target_doys):
    """
    95th-percentile threshold for each calendar DOY using a centred
    ±7-day window pooled across all climatology years.

    Uses numpy + ThreadPoolExecutor (numpy releases the GIL).
    """
    half = WINDOW // 2
    data = np.ascontiguousarray(combined.values, dtype=np.float32)
    all_doys = combined.time.dt.dayofyear.values
    unique_doys = np.sort(np.unique(target_doys))

    chunks = [c for c in np.array_split(unique_doys, N_WORKERS) if len(c)]
    args = [(ch, all_doys, data, half, PERCENTILE) for ch in chunks]

    print(f"  {len(unique_doys)} DOYs across {len(chunks)} threads ...")
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_percentile_chunk, args))

    return xr.DataArray(
        np.concatenate(results, axis=0),
        dims=["dayofyear", "latitude", "longitude"],
        coords={
            "dayofyear": np.concatenate(chunks),
            "latitude":  combined.latitude,
            "longitude": combined.longitude,
        },
    )


# ─── Exceedance counting ────────────────────────────────────────────────────
def count_exceedance(season, pctl):
    """Count days exceeding the DOY-specific threshold, return frequency."""
    doys = season.time.dt.dayofyear.values
    thresh = (pctl.sel(dayofyear=doys)
              .assign_coords(dayofyear=season.time.values)
              .rename({"dayofyear": "time"}))
    exceed = (season > thresh).sum(dim="time")
    n_days = float(len(season.time))
    return exceed / n_days


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PCTL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    wall_start = time.time()

    print(f"ERA5-Land input directory : {INPUT_DIR}")
    print(f"Output directory          : {OUTPUT_DIR}")
    print(f"Climatology save directory: {PCTL_SAVE_DIR}")

    pctl_path = (PCTL_SAVE_DIR /
                 f"tmax_{PERCENTILE}th_pctl_doy_nh_jja"
                 f"_{CLIM_START}_{CLIM_END}.nc")

    # ── 1. Percentile thresholds ─────────────────────────────────────────────
    pctl = None
    if pctl_path.exists():
        try:
            candidate = xr.open_dataarray(pctl_path).load()
            if (candidate.sizes.get("latitude", 0) == 251
                    and candidate.sizes.get("longitude", 0) == 601):
                pctl = candidate
                print(f"Loaded cached percentile: {pctl_path}")
        except Exception:
            pass

    if pctl is None:
        print(f"Loading climatology {CLIM_START}–{CLIM_END} "
              f"(months {ALL_LOAD_MONTHS}, CONUS) ...")
        seasons = []
        for yr in range(CLIM_START, CLIM_END + 1):
            try:
                seasons.append(load_months(yr, ALL_LOAD_MONTHS))
                print(f"  {yr}")
            except FileNotFoundError as e:
                print(f"  {yr} — missing: {e}")

        combined = xr.concat(seasons, dim="time")
        print(f"  Shape: {dict(combined.sizes)}  "
              f"({combined.nbytes / 1e9:.2f} GB)")

        # Core-summer DOYs only (Jun/Jul/Aug)
        summer_doys = combined.time.dt.dayofyear.values[
            combined.time.dt.month.isin(SUMMER_MONTHS).values
        ]

        print(f"\nComputing {PERCENTILE}th percentile "
              f"(window={WINDOW}, threads={N_WORKERS}) ...")
        t0 = time.time()
        pctl = calendar_day_percentile(combined, summer_doys)
        print(f"  Done in {time.time() - t0:.1f} s")

        pctl.to_netcdf(pctl_path)
        print(f"  Saved → {pctl_path}")

        del combined, seasons   # free ~3 GB

    # ── 2. Yearly exceedance frequency ───────────────────────────────────────
    print(f"\nExceedance frequency {FREQ_START}–{FREQ_END} ...")
    freq_list = []
    for yr in range(FREQ_START, FREQ_END + 1):
        try:
            season = load_months(yr, SUMMER_MONTHS)
        except FileNotFoundError as e:
            print(f"  {yr} — missing: {e}")
            continue
        freq = count_exceedance(season, pctl)
        freq.to_netcdf(OUTPUT_DIR / f"exceedance_frequency_{yr}.nc")
        freq_list.append(freq)
        print(f"  {yr}  ({len(season.time)} days)")

    # ── 3. Mean frequency ────────────────────────────────────────────────────
    if freq_list:
        mean_freq = sum(freq_list) / len(freq_list)
        out = OUTPUT_DIR / f"mean_exceedance_frequency_{FREQ_START}_{FREQ_END}.nc"
        mean_freq.to_netcdf(out)
        print(f"\nMean frequency → {out}")

    print(f"\nTotal: {time.time() - wall_start:.0f} s")


if __name__ == "__main__":
    main()
