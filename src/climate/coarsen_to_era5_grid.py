#!/usr/bin/env python -u
"""
Coarsen ERA5-Land 0.1° climatology / EHD grids to the native ERA5 0.25° grid.

For each 0.25° ERA5 cell, all ERA5-Land 0.1° cells whose centres fall within
the cell (nearest-0.25° assignment) are averaged.  Since 0.25/0.1 = 2.5 is
not an integer, each ERA5 cell receives 2–3 contributing 0.1° cells, which
alternates in a fixed pattern across the grid.

ERA5 target grid
    Latitude  : 90.0 → -90.0 (descending), 0.25° step, 721 points
    Longitude : -180.0 → 179.75 (ascending), 0.25° step, 1440 points
    Convention: same -180..+180 as the ERA5-Land climatology outputs

Works with any NetCDF that has dimensions (dayofyear, latitude, longitude)
or (latitude, longitude).  Handles float32 values and NaN masking.

Outputs are named with a `_025deg` suffix and are loadable with
xr.open_dataarray() (same variable name convention as input).

Usage examples
--------------
# Coarsen both climatologies (run from repo root):
    python src/climate/coarsen_to_era5_grid.py \\
        data/processed/climatologies/tmax_95th_pctl_doy_nh_jja_1990_2020.nc \\
        data/processed/climatologies/tmax_95th_pctl_doy_sh_djf_1990_2020.nc

# Coarsen all per-year EHD grids:
    python src/climate/coarsen_to_era5_grid.py \\
        data/processed/exceedance_frequency/exceedance_frequency_*.nc

# Or call from Python:
    from coarsen_to_era5_grid import coarsen_file
    coarsen_file("data/processed/climatologies/tmax_95th_pctl_doy_sh_djf_1990_2020.nc")
"""

import sys
import time
import numpy as np
import xarray as xr
from pathlib import Path

# ─── ERA5 0.25° target grid ──────────────────────────────────────────────────
ERA5_LATS = np.round(np.arange(90.0, -90.1, -0.25), 2)   # 721 points, descending
ERA5_LONS = np.round(np.arange(-180.0, 180.0, 0.25), 2)  # 1440 points, ascending
N_LAT_ERA5 = len(ERA5_LATS)   # 721
N_LON_ERA5 = len(ERA5_LONS)   # 1440


# ─── Build ERA5-Land → ERA5 index mapping ────────────────────────────────────
def build_index_map(src_lats: np.ndarray, src_lons: np.ndarray):
    """
    Map each ERA5-Land 0.1° (lat, lon) cell to its nearest ERA5 0.25° cell.

    Strategy: round each 0.1° coordinate to the nearest 0.25°, then compute
    the integer index into the ERA5 grid arrays.

    Returns
    -------
    lat_idx : int array, shape (N_SRC_LAT,)
    lon_idx : int array, shape (N_SRC_LON,)
    """
    # Nearest 0.25° for each 0.1° latitude
    lat_025 = np.round(src_lats / 0.25) * 0.25
    lat_idx  = np.round((90.0 - lat_025) / 0.25).astype(int)
    lat_idx  = np.clip(lat_idx, 0, N_LAT_ERA5 - 1)

    # Nearest 0.25° for each 0.1° longitude
    lon_025 = np.round(src_lons / 0.25) * 0.25
    lon_idx  = np.round((lon_025 + 180.0) / 0.25).astype(int)
    lon_idx  = np.clip(lon_idx, 0, N_LON_ERA5 - 1)

    return lat_idx, lon_idx


def coarsen_2d(arr_2d: np.ndarray, lat_idx: np.ndarray, lon_idx: np.ndarray) -> np.ndarray:
    """
    Coarsen a (N_SRC_LAT, N_SRC_LON) float32 array to ERA5 0.25° by
    averaging all valid (non-NaN) source cells assigned to each target cell.

    Parameters
    ----------
    arr_2d  : (N_SRC_LAT, N_SRC_LON) float32 array
    lat_idx : (N_SRC_LAT,) int  — ERA5 lat index for each source row
    lon_idx : (N_SRC_LON,) int  — ERA5 lon index for each source column

    Returns
    -------
    out : (N_LAT_ERA5, N_LON_ERA5) float32 array, NaN where no data
    """
    N_src_lat = arr_2d.shape[0]
    N_src_lon = arr_2d.shape[1]
    n_era5    = N_LAT_ERA5 * N_LON_ERA5

    # Flat ERA5 cell index for every source cell
    src_lat_full = np.repeat(lat_idx, N_src_lon)                       # (N,)
    src_lon_full = np.tile(lon_idx, N_src_lat)                         # (N,)
    flat_idx     = src_lat_full * N_LON_ERA5 + src_lon_full            # (N,)

    vals   = arr_2d.ravel().astype(np.float64)
    valid  = np.isfinite(vals)

    val_sum = np.bincount(flat_idx[valid], weights=vals[valid],
                          minlength=n_era5)
    n_count = np.bincount(flat_idx[valid], minlength=n_era5)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(n_count > 0, val_sum / n_count, np.nan)

    return mean.reshape(N_LAT_ERA5, N_LON_ERA5).astype(np.float32)


def coarsen_array(da: xr.DataArray) -> xr.DataArray:
    """
    Coarsen an xr.DataArray with dims (dayofyear|time, latitude, longitude)
    or (latitude, longitude) to the ERA5 0.25° grid.

    All non-lat/lon dimensions are looped over; only the spatial dims are
    aggregated.
    """
    src_lats = da.latitude.values
    src_lons = da.longitude.values
    lat_idx, lon_idx = build_index_map(src_lats, src_lons)

    era5_coords = {"latitude": ERA5_LATS, "longitude": ERA5_LONS}

    if da.dims == ("latitude", "longitude"):
        out_data = coarsen_2d(da.values, lat_idx, lon_idx)
        return xr.DataArray(
            out_data,
            dims=["latitude", "longitude"],
            coords=era5_coords,
        )

    # Has a leading dimension (dayofyear, time, etc.)
    lead_dim  = da.dims[0]
    lead_vals = da[lead_dim].values
    n_lead    = len(lead_vals)
    out_cube  = np.empty((n_lead, N_LAT_ERA5, N_LON_ERA5), dtype=np.float32)

    for i in range(n_lead):
        out_cube[i] = coarsen_2d(da.values[i], lat_idx, lon_idx)
        if (i + 1) % 10 == 0 or i == 0 or i == n_lead - 1:
            print(f"  slice {i+1}/{n_lead}", end="\r", flush=True)

    print()
    return xr.DataArray(
        out_cube,
        dims=[lead_dim, "latitude", "longitude"],
        coords={lead_dim: lead_vals, **era5_coords},
    )


def coarsen_file(src_path, dst_path=None):  # src_path: str|Path, dst_path: str|Path|None
    """
    Coarsen one NetCDF file to the ERA5 0.25° grid and save.

    The output is saved as an unnamed DataArray (same convention as the
    NH JJA / SH DJF climatology files).

    Parameters
    ----------
    src_path : path to input NetCDF
    dst_path : output path; if None, appends `_025deg` before the .nc suffix

    Returns
    -------
    Path of the written output file
    """
    src_path = Path(src_path)
    if dst_path is None:
        dst_path = src_path.parent / (src_path.stem + "_025deg.nc")
    dst_path = Path(dst_path)

    if dst_path.exists():
        print(f"  Already exists, skipping: {dst_path.name}")
        return dst_path

    t0 = time.time()
    print(f"Input : {src_path.name}")
    print(f"Output: {dst_path.name}")

    # Try to open as DataArray first (climatology files), fall back to Dataset
    try:
        da = xr.open_dataarray(src_path).load()
    except ValueError:
        # File is a Dataset — pick the first variable
        ds = xr.open_dataset(src_path).load()
        var = list(ds.data_vars)[0]
        da = ds[var]

    print(f"  Source dims : {dict(da.sizes)}")
    print(f"  Lat range   : {float(da.latitude.min()):.2f} → {float(da.latitude.max()):.2f}")
    print(f"  Lon range   : {float(da.longitude.min()):.2f} → {float(da.longitude.max()):.2f}")

    da_coarse = coarsen_array(da)

    n_valid_src = int(np.sum(np.isfinite(da.values)))
    n_valid_dst = int(np.sum(np.isfinite(da_coarse.values)))
    print(f"  Output dims : {dict(da_coarse.sizes)}")
    print(f"  Valid cells : {n_valid_src:,} → {n_valid_dst:,}")

    # Save as unnamed DataArray (same convention as NH/SH climatology files)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    da_coarse.to_netcdf(dst_path)

    elapsed = time.time() - t0
    size_mb = dst_path.stat().st_size / 1e6
    print(f"  Written {size_mb:.1f} MB in {elapsed:.1f}s → {dst_path}")
    return dst_path


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python coarsen_to_era5_grid.py <file1.nc> [file2.nc ...]")
        sys.exit(1)

    files = sys.argv[1:]
    print(f"ERA5 target grid: {N_LAT_ERA5} lat × {N_LON_ERA5} lon (0.25°)")
    print(f"Files to process: {len(files)}\n")

    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"WARNING: not found, skipping — {p}")
            continue
        coarsen_file(p)
        print()


if __name__ == "__main__":
    main()
