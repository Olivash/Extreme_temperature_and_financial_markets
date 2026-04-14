#!/usr/bin/env python -u
"""
Compute EHD sensitivity slopes from CIL GDPCIR QDM historical data.

Follows the IFS reforecast methodology (Global_Summer.ipynb):

  LOCAL slope:
    - summer-mean Tmax anomaly (subtract mean over years) per model per year
    - regress log(EHD) on T_local_anom across years → log(RR) per K
    - exp(slope) → RR per K

  200km (interannual) slope:
    - compute summer-mean Tmax per year per model
    - apply latitude-varying Gaussian smooth (R=200km):
        sigma_lat  = (200 / 110.574) / grid_deg            (same all rows)
        sigma_lon[i] = min((200 / (111.320 * cos(lat))) / grid_deg,
                           max(3*sigma_lat, 200))           (per row)
        pass 1: gaussian_filter1d(arr, sigma_lat, axis=0, mode='nearest')
        pass 2: gaussian_filter1d(row, sigma_lon[i], mode='wrap')  per row
    - subtract climatology (mean over years) → T_200km_anom
    - regress log(EHD) on T_200km_anom across years → log(RR) per K
    - exp(slope) → RR per K

  ECS-weighted ensemble of per-model slope maps.

Parallelisation:
  - Model level:  ProcessPoolExecutor(N_WORKERS=4) — independent processes
  - Year level:   ThreadPoolExecutor(YEAR_THREADS=4) per worker
                  Each year opens its own zarr connection (thread-safe).
  - SAS token fetched once in main, passed to all workers.
  - Skip-if-exists cache: already-completed models load from disk instantly.

Outputs:
  projections/output/cil_ehd_slopes_local.nc           — ensemble RR per K (local)
  projections/output/cil_ehd_slopes_200km.nc            — ensemble RR per K (200km)
  projections/output/cil_ehd_slopes_comparison.png      — 4-panel vs IFS slopes
  projections/output/cil_ehd_slopes_per_model.csv       — per-model summary
  projections/output/cil_slopes/{model}_local.nc        — per-model RR/K local
  projections/output/cil_slopes/{model}_200km.nc        — per-model RR/K 200km
  projections/output/cil_tmax_hist/{model}_local.npy    — mean hist summer Tmax (K)
  projections/output/cil_tmax_hist/{model}_200km.npy    — mean hist 200km-smoothed Tmax
  projections/output/cil_tmax_hist/{model}_lats.npy     — lat coordinate
  projections/output/cil_tmax_hist/{model}_lons.npy     — lon coordinate

Environment: /turbo/mgoldklang/pyenvs/nov25/bin/python
Required:
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
  REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

Usage:
  cd ~/projects/macro/extreme_heat/biodiversity_interactions
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \\
  REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \\
  nohup /turbo/mgoldklang/pyenvs/nov25/bin/python \\
      projections/scripts/compute_cil_ehd_slopes.py \\
      > projections/output/cil_ehd_slopes.log 2>&1 &
"""

import os
import sys
import time
import traceback
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)

_CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"
os.environ["SSL_CERT_FILE"]      = _CA_BUNDLE
os.environ["REQUESTS_CA_BUNDLE"] = _CA_BUNDLE

# Patch certifi so pystac_client / requests use the internal-CA-aware bundle
import certifi, certifi.core
certifi.core.where = lambda: _CA_BUNDLE
certifi.where      = certifi.core.where

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJ_DIR         = Path(__file__).resolve().parent.parent
OUT_DIR          = PROJ_DIR / "output"
PER_MODEL_SLOPES = OUT_DIR / "cil_slopes"
TMAX_HIST_DIR    = OUT_DIR / "cil_tmax_hist"
for _d in [OUT_DIR, PER_MODEL_SLOPES, TMAX_HIST_DIR]:
    _d.mkdir(exist_ok=True)

CLIM_DIR   = Path("/users/is/mgoldklang/code/Extreme_temperature_and_financial_markets"
                  "/data/processed/climatologies")
SLOPES_DIR = Path("/home/mgoldklang/code/Extreme_temperature_and_financial_markets/data/slopes")
SLOPE_LOCAL = SLOPES_DIR / "global_summer_clima_rr_per_degC_2001_2020_025deg.nc"
SLOPE_200KM = SLOPES_DIR / "global_summer_inter_rr_per_degC_200km_era5clim_2001_2020_025deg.nc"

# ─── Config ───────────────────────────────────────────────────────────────────
HIST_YEARS    = list(range(1950, 2015))    # 65 years — full CMIP6 historical period
R_KM          = 200.0
EHD_FLOOR     = 1.0 / 92.0

N_WORKERS     = 4    # model-level parallel processes
YEAR_THREADS  = 4    # per-model year download threads (within each process)
                     # max concurrent zarr reads = N_WORKERS × YEAR_THREADS = 16

# ─── ECS values ───────────────────────────────────────────────────────────────
ECS_VALUES = {
    "ACCESS-CM2":       4.66,   "ACCESS-ESM1-5":    3.88,
    "BCC-CSM2-MR":      3.02,   "CanESM5":          5.64,
    "CMCC-CM2-SR5":     3.55,   "CMCC-ESM2":        3.58,
    "EC-Earth3":        4.26,   "EC-Earth3-Veg-LR": 4.23,
    "FGOALS-g3":        2.87,   "GFDL-CM4":         3.89,
    "GFDL-ESM4":        2.65,   "HadGEM3-GC31-LL":  5.55,
    "INM-CM4-8":        1.83,   "INM-CM5-0":        1.92,
    "IPSL-CM6A-LR":     3.09,   "KACE-1-0-G":       4.48,
    "KIOST-ESM":        3.36,   "MIROC6":           2.60,
    "MPI-ESM1-2-HR":    2.98,   "MPI-ESM1-2-LR":    3.00,
    "MRI-ESM2-0":       3.13,   "NorESM2-LM":       2.56,
    "NorESM2-MM":       2.49,
}

MODEL_INSTITUTION = {
    "ACCESS-CM2":       "CSIRO-ARCCSS",   "ACCESS-ESM1-5":    "CSIRO",
    "BCC-CSM2-MR":      "BCC",            "CanESM5":          "CCCma",
    "CMCC-CM2-SR5":     "CMCC",           "CMCC-ESM2":        "CMCC",
    "EC-Earth3":        "EC-Earth-Consortium",
    "EC-Earth3-Veg-LR": "EC-Earth-Consortium",
    "FGOALS-g3":        "CAS",            "GFDL-CM4":         "NOAA-GFDL",
    "GFDL-ESM4":        "NOAA-GFDL",      "HadGEM3-GC31-LL":  "MOHC",
    "INM-CM4-8":        "INM",            "INM-CM5-0":        "INM",
    "IPSL-CM6A-LR":     "IPSL",           "KACE-1-0-G":       "NIMS-KMA",
    "KIOST-ESM":        "KIOST",          "MIROC6":           "MIROC",
    "MPI-ESM1-2-HR":    "MPI-M",          "MPI-ESM1-2-LR":    "MPI-M",
    "MRI-ESM2-0":       "MRI",            "NorESM2-LM":       "NCC",
    "NorESM2-MM":       "NCC",
}

MODEL_ENSEMBLE = {"HadGEM3-GC31-LL": "r1i1p1f3"}


def compute_ecs_weights(models, ecs_center=3.0, ecs_sigma=0.75):
    raw   = {m: np.exp(-0.5 * ((ECS_VALUES[m] - ecs_center) / ecs_sigma) ** 2) for m in models}
    total = sum(raw.values())
    return {m: w / total for m, w in raw.items()}


# ─── Planetary Computer access ────────────────────────────────────────────────

def get_storage_options():
    import planetary_computer as pc, pystac_client
    cat   = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                      modifier=pc.sign_inplace)
    items = list(cat.search(collections=["cil-gdpcir-cc0"], max_items=1).items())
    if not items:
        raise RuntimeError("PC STAC: no items in cil-gdpcir-cc0")
    return pc.sign(items[0]).assets["tasmax"].extra_fields["xarray:open_kwargs"]["storage_options"]


def open_cil_zarr(institution, model, storage_options, scenario="historical"):
    import adlfs
    ens  = MODEL_ENSEMBLE.get(model, "r1i1p1f1")
    blob = f"cil-gdpcir/CMIP/{institution}/{model}/{scenario}/{ens}/day/tasmax/v1.1.zarr"
    fs   = adlfs.AzureBlobFileSystem(**storage_options)
    return xr.open_zarr(fs.get_mapper(blob), consolidated=False, chunks=None)


# ─── Climatology threshold preparation ────────────────────────────────────────

def prepare_thresholds_for_zarr_grid(z_lats_raw, z_lons_raw, nh_idx_native, sh_idx_native):
    """
    Load the ERA5-Land 95th-percentile daily Tmax climatology files (0.25°) and
    remap them to the zarr native grid using nearest-neighbour lookup.

    Returns:
      thresh_nh  (93, n_nh, n_lon)  float32 K  — JJA thresholds for NH rows
      thresh_sh  (92, n_sh, n_lon)  float32 K  — DJF thresholds for SH rows
      nh_doys    (93,)  int  — dayofyear for each NH slice
      sh_doys    (92,)  int  — dayofyear for each SH slice
    """
    # Convert zarr lons to -180..180 (handles 0-360 zarrs via roll-equivalent mapping)
    z_lons_180 = z_lons_raw.copy().astype(np.float64)
    z_lons_180[z_lons_180 > 180.0]  -= 360.0   # e.g. 181 → -179
    z_lons_180[z_lons_180 <= -180.0] += 360.0  # clamp to (-180, 180]
    # Sanity check
    assert z_lons_180.min() >= -180.0 and z_lons_180.max() <= 180.0, \
        f"Lon normalisation failed: {z_lons_180.min():.2f}..{z_lons_180.max():.2f}"

    def _remap_threshold(zarr_path, lat_idx, lon_idx_180):
        # Threshold zarr: (n_doy, 721, 1440); lat 90→-90 descending, lon -180→179.75
        # var='thresh' in Celsius; coords: lat, lon, dayofyear
        ds = xr.open_zarr(str(zarr_path))
        thresh = ds["thresh"].values.astype(np.float32)   # (n_doy, 721, 1440) Celsius
        doys   = ds["dayofyear"].values
        ds.close()
        # Map zarr lat row indices → nearest threshold lat index
        # Threshold lat: descending from 90→-90 at 0.25° → idx = (90 - lat) / 0.25
        lat_vals = z_lats_raw[lat_idx]
        t_lat_idx = np.clip(np.round((90.0 - lat_vals) / 0.25).astype(int), 0, 720)
        # Map zarr lon (-180..180) → threshold lon index (step 0.25, start -180)
        t_lon_idx = np.clip(np.round((lon_idx_180 + 180.0) / 0.25).astype(int), 0, 1439)
        # Fancy-index: (n_doy, n_lat_subset, n_lon_native)
        # lat_idx and lon_idx broadcast to select the right threshold cell for each zarr cell
        out = thresh[:, t_lat_idx[:, None], t_lon_idx[None, :]]
        out = out + 273.15    # Celsius → Kelvin to match zarr tasmax (K)
        return out, doys

    thresh_nh, nh_doys = _remap_threshold(
        CLIM_DIR / "tmax_95th_pctl_doy_nh_jja_global_1990_2020_025deg.zarr",
        nh_idx_native, z_lons_180,
    )
    thresh_sh, sh_doys = _remap_threshold(
        CLIM_DIR / "tmax_95th_pctl_doy_sh_djf_1990_2020_025deg.zarr",
        sh_idx_native, z_lons_180,
    )
    return thresh_nh, thresh_sh, nh_doys, sh_doys


# ─── Per-year parallel download: Tmax + EHD ───────────────────────────────────

def _fetch_year(yr, institution, model, storage_options,
                nh_idx_native, sh_idx_native, lat_sort, lon_sort,
                thresh_nh, thresh_sh, nh_doys, sh_doys):
    """
    Download daily summer Tmax for a single year.  For each day compare against
    the ERA5-Land 95th-percentile climatology threshold → EHD fraction.
    Also returns summer-mean Tmax.

    NH summer = JJA (doys 152-244); SH summer = DJF (Dec yr-1 + Jan/Feb yr).
    Both outputs sorted to ascending-lat, -180..180 grid.
    Returns (yr, mean_tmax, ehd_frac) or (yr, None, None) if no data.
    """
    # Apply certifi patch in thread context (threads share process env, so this is redundant
    # but harmless; the real patch happens at process start via _CA_BUNDLE at module level)
    institution_local = MODEL_INSTITUTION[model]
    ds  = open_cil_zarr(institution_local, model, storage_options)
    yr_ = ds.time.dt.year.values
    mo_ = ds.time.dt.month.values
    doy_ = ds.time.dt.dayofyear.values
    n_lat, n_lon = len(ds.lat), len(ds.lon)
    result_tmax = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    result_ehd  = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

    # Build DOY → threshold-slice lookup
    nh_doy_to_idx = {int(d): i for i, d in enumerate(nh_doys)}
    sh_doy_to_idx = {int(d): i for i, d in enumerate(sh_doys)}

    def _ehd_mean(days_arr, thresh_arr):
        """
        Fraction of summer days exceeding threshold.
        Ocean cells (NaN threshold) → NaN EHD (excluded from regression).
        """
        with np.errstate(invalid="ignore"):
            exceed = np.where(
                np.isfinite(thresh_arr),
                (days_arr > thresh_arr).astype(np.float32),
                np.nan,
            )
        with np.errstate(all="ignore"):
            frac = np.nanmean(exceed, axis=0).astype(np.float32)
        return frac   # NaN where all days had NaN threshold (ocean)

    if len(nh_idx_native):
        tidx = np.where((yr_ == yr) & np.isin(mo_, [6, 7, 8]))[0]
        if len(tidx):
            days = ds["tasmax"].isel(time=tidx, lat=nh_idx_native).load().values.astype(np.float32)
            doys_yr = doy_[tidx]
            thresh_stack = np.stack(
                [thresh_nh[nh_doy_to_idx[int(d)]] for d in doys_yr], axis=0
            )
            result_tmax[nh_idx_native] = days.mean(axis=0)
            result_ehd[nh_idx_native]  = _ehd_mean(days, thresh_stack)

    if len(sh_idx_native):
        t_dec = np.where((yr_ == yr - 1) & (mo_ == 12))[0]
        t_jf  = np.where((yr_ == yr)     & np.isin(mo_, [1, 2]))[0]
        t_all = np.concatenate([t_dec, t_jf]) if len(t_dec) and len(t_jf) else (
                t_dec if len(t_dec) else t_jf)
        if len(t_all):
            days = ds["tasmax"].isel(time=t_all, lat=sh_idx_native).load().values.astype(np.float32)
            doys_yr = doy_[t_all]
            # Some SH DJF days (e.g., doy=366 in leap year) may not be in threshold
            valid_mask = np.array([int(d) in sh_doy_to_idx for d in doys_yr])
            if valid_mask.any():
                thresh_stack = np.stack(
                    [thresh_sh[sh_doy_to_idx[int(d)]] for d in doys_yr[valid_mask]], axis=0
                )
                days_valid = days[valid_mask]
                result_tmax[sh_idx_native] = days_valid.mean(axis=0)
                result_ehd[sh_idx_native]  = _ehd_mean(days_valid, thresh_stack)

    ds.close()

    if not np.isfinite(result_tmax).any():
        return yr, None, None

    return (yr,
            result_tmax[lat_sort, :][:, lon_sort],
            result_ehd[lat_sort,  :][:, lon_sort])


def download_tmax_years(institution, model, storage_options,
                        years, nh_idx_native, sh_idx_native,
                        lat_sort, lon_sort,
                        thresh_nh, thresh_sh, nh_doys, sh_doys):
    """
    Download summer daily Tmax for all requested years in parallel.
    Computes mean Tmax AND EHD fraction (days > ERA5 95th pctile threshold) per year.
    Returns (years_ok list, {yr: tmax_array}, {yr: ehd_array}).
    """
    tmax_data = {}
    ehd_data  = {}
    with ThreadPoolExecutor(max_workers=YEAR_THREADS) as pool:
        futs = {
            pool.submit(_fetch_year, yr, institution, model, storage_options,
                        nh_idx_native, sh_idx_native, lat_sort, lon_sort,
                        thresh_nh, thresh_sh, nh_doys, sh_doys): yr
            for yr in years
        }
        for fut in as_completed(futs):
            yr, tmax, ehd = fut.result()
            if tmax is not None:
                tmax_data[yr] = tmax
                ehd_data[yr]  = ehd
    years_ok = sorted(tmax_data)
    return years_ok, tmax_data, ehd_data


# ─── Summer-mean Tmax (kept for reference / single-year use) ─────────────────

def summer_tmax_year(ds, year, nh_lat_idx, sh_lat_idx):
    yr = ds.time.dt.year.values
    mo = ds.time.dt.month.values
    n_lat, n_lon = len(ds.lat), len(ds.lon)
    result = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    if len(nh_lat_idx):
        tidx = np.where((yr == year) & np.isin(mo, [6, 7, 8]))[0]
        if len(tidx):
            chunk = ds["tasmax"].isel(time=tidx, lat=nh_lat_idx).load().values
            result[nh_lat_idx] = chunk.mean(axis=0).astype(np.float32)
    if len(sh_lat_idx):
        t_dec = np.where((yr == year - 1) & (mo == 12))[0]
        t_jf  = np.where((yr == year) & np.isin(mo, [1, 2]))[0]
        parts = []
        if len(t_dec):
            parts.append(ds["tasmax"].isel(time=t_dec, lat=sh_lat_idx).load().values)
        if len(t_jf):
            parts.append(ds["tasmax"].isel(time=t_jf,  lat=sh_lat_idx).load().values)
        if parts:
            result[sh_lat_idx] = np.concatenate(parts, axis=0).mean(axis=0).astype(np.float32)
    return result if np.isfinite(result).any() else None


# ─── Latitude-varying Gaussian smoothing (from notebook cell 12) ─────────────

def lat_varying_gaussian(arr_2d, lat_1d, r_km=200.0, grid_deg=0.25):
    sigma_lat = (r_km / 110.574) / grid_deg
    coslat    = np.clip(np.cos(np.deg2rad(lat_1d)), 0.05, 1.0)
    sigma_lon = np.minimum(
        (r_km / (111.320 * coslat)) / grid_deg,
        max(3.0 * sigma_lat, 200.0)
    )
    sm_lat = gaussian_filter1d(arr_2d.astype(np.float64), sigma=sigma_lat,
                               axis=0, mode="nearest")
    smoothed = np.empty_like(sm_lat)
    for i in range(len(lat_1d)):
        smoothed[i, :] = gaussian_filter1d(sm_lat[i, :], sigma=float(sigma_lon[i]),
                                           mode="wrap")
    return smoothed.astype(np.float32)


# ─── OLS slope ────────────────────────────────────────────────────────────────

def ols_slope(X, Y):
    """Vectorised OLS: Y ~ β*X + α.  X, Y: (n_years, lat, lon). Returns β (lat, lon)."""
    X_dm = X - np.nanmean(X, axis=0, keepdims=True)
    Y_dm = Y - np.nanmean(Y, axis=0, keepdims=True)
    num  = np.nansum(X_dm * Y_dm, axis=0)
    den  = np.nansum(X_dm ** 2,   axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(den > 0, num / den, np.nan)
    return beta.astype(np.float32)


# ─── Per-model processing ─────────────────────────────────────────────────────

def process_model(model, storage_options):
    """
    Compute local and 200km EHD sensitivity slopes for one model.

    For each historical year, download daily summer Tmax from CIL zarr and
    count the fraction of days exceeding the ERA5-Land 95th-pctile threshold
    (loaded from pre-computed climatology files at 0.25°).  This gives a
    continuous EHD fraction [0,1] per year per grid cell.

    Regresses log(EHD) on summer-mean Tmax anomaly → RR per K.

    Returns dict with lats, lons, rr_local, rr_200km, n_years.
    Saves per-model .nc (slopes) and .npy (Tmax baselines) to disk.
    """
    t0 = time.time()
    institution = MODEL_INSTITUTION[model]

    # ── Skip if all outputs already exist ─────────────────────────────────────
    p_loc = PER_MODEL_SLOPES / f"{model}_local.nc"
    p_200 = PER_MODEL_SLOPES / f"{model}_200km.nc"
    p_tl  = TMAX_HIST_DIR    / f"{model}_local.npy"
    p_t2  = TMAX_HIST_DIR    / f"{model}_200km.npy"
    if all(p.exists() for p in [p_loc, p_200, p_tl, p_t2]):
        print(f"  {model}: all outputs cached — loading from disk", flush=True)
        ds_l = xr.open_dataset(str(p_loc), engine="netcdf4")
        ds_2 = xr.open_dataset(str(p_200), engine="netcdf4")
        lats = ds_l["lat"].values
        lons = ds_l["lon"].values
        rr_l = ds_l["rr_per_degC"].values
        rr_2 = ds_2["rr_per_degC"].values
        ds_l.close(); ds_2.close()
        return {"lats": lats, "lons": lons, "rr_local": rr_l,
                "rr_200km": rr_2, "n_years": -1}

    # ── Open zarr once for grid metadata, then close ──────────────────────────
    print(f"  {model}: reading zarr metadata...", flush=True)
    ds_meta    = open_cil_zarr(institution, model, storage_options)
    z_lats_raw = ds_meta.lat.values
    z_lons_raw = ds_meta.lon.values
    n_time     = len(ds_meta.time)
    ds_meta.close()

    # Sort indices: ascending lat, -180..180 lon
    z_lons_180 = z_lons_raw.copy()
    z_lons_180[z_lons_180 >= 180] -= 360.0
    lon_sort_idx  = np.argsort(z_lons_180)
    lat_sort_idx  = np.argsort(z_lats_raw)
    z_lons_sorted = z_lons_180[lon_sort_idx]
    z_lats_sorted = z_lats_raw[lat_sort_idx]
    nh_idx_native = np.where(z_lats_raw >= 0)[0]
    sh_idx_native = np.where(z_lats_raw  < 0)[0]
    grid_deg = float(np.abs(z_lats_sorted[1] - z_lats_sorted[0]))

    print(f"  {model}: zarr {len(z_lats_raw)}×{len(z_lons_raw)} "
          f"grid_deg={grid_deg:.3f}  time={n_time}d", flush=True)

    # ── Prepare climatology thresholds on zarr native grid ────────────────────
    thresh_nh, thresh_sh, nh_doys, sh_doys = prepare_thresholds_for_zarr_grid(
        z_lats_raw, z_lons_raw, nh_idx_native, sh_idx_native
    )
    print(f"  {model}: thresholds prepared  "
          f"NH={thresh_nh.shape}  SH={thresh_sh.shape}", flush=True)

    # ── Parallel download: daily Tmax → EHD fraction + mean Tmax ──────────────
    print(f"  {model}: downloading {len(HIST_YEARS)} years daily Tmax "
          f"({YEAR_THREADS} threads)...", flush=True)

    years_ok, tmax_data, ehd_data = download_tmax_years(
        institution, model, storage_options,
        HIST_YEARS, nh_idx_native, sh_idx_native,
        lat_sort_idx, lon_sort_idx,
        thresh_nh, thresh_sh, nh_doys, sh_doys,
    )

    if len(years_ok) < 5:
        print(f"  {model}: only {len(years_ok)} years — skip", flush=True)
        return None

    print(f"  {model}: {len(years_ok)} years downloaded  "
          f"{time.time()-t0:.0f}s", flush=True)

    tmax_list = [tmax_data[yr] for yr in years_ok]
    ehd_list  = [ehd_data[yr]  for yr in years_ok]
    n_yr = len(years_ok)
    print(f"  {model}: {n_yr} years paired  {time.time()-t0:.0f}s", flush=True)

    # ── Stack arrays ──────────────────────────────────────────────────────────
    tmax_arr = np.stack(tmax_list, axis=0)   # (n_yr, lat, lon) K
    ehd_arr  = np.stack(ehd_list,  axis=0)   # (n_yr, lat, lon) fraction
    log_ehd  = np.log(np.maximum(ehd_arr, EHD_FLOOR))

    # ── LOCAL slope ───────────────────────────────────────────────────────────
    t_anom_local     = tmax_arr - np.nanmean(tmax_arr, axis=0, keepdims=True)
    slope_local_log  = ols_slope(t_anom_local, log_ehd)
    rr_local         = np.exp(slope_local_log)
    rr_local[~np.isfinite(rr_local) | (rr_local < 0.5) | (rr_local > 50)] = np.nan

    # ── 200km slope ───────────────────────────────────────────────────────────
    tmax_smooth  = np.stack(
        [lat_varying_gaussian(tmax_arr[i], z_lats_sorted, r_km=R_KM, grid_deg=grid_deg)
         for i in range(n_yr)],
        axis=0,
    )
    clim_smooth  = np.nanmean(tmax_smooth, axis=0, keepdims=True)
    t_anom_200km = tmax_smooth - clim_smooth
    slope_200_log = ols_slope(t_anom_200km, log_ehd)
    rr_200km = np.exp(slope_200_log)
    rr_200km[~np.isfinite(rr_200km) | (rr_200km < 0.5) | (rr_200km > 50)] = np.nan

    print(f"  {model}: RR_local med={np.nanmedian(rr_local):.3f}  "
          f"RR_200km med={np.nanmedian(rr_200km):.3f}  "
          f"total {time.time()-t0:.0f}s", flush=True)

    # ── Save per-model slopes ─────────────────────────────────────────────────
    def _save_slope(arr, path):
        da = xr.DataArray(arr, dims=["lat", "lon"],
                          coords={"lat": z_lats_sorted, "lon": z_lons_sorted})
        xr.Dataset({"rr_per_degC": da}).to_netcdf(str(path))

    _save_slope(rr_local, p_loc)
    _save_slope(rr_200km, p_200)

    # ── Save Tmax historical baselines ────────────────────────────────────────
    np.save(str(p_tl),                        np.nanmean(tmax_arr, axis=0).astype(np.float32))
    np.save(str(p_t2),                        clim_smooth.squeeze(axis=0))
    np.save(str(TMAX_HIST_DIR / f"{model}_lats.npy"), z_lats_sorted)
    np.save(str(TMAX_HIST_DIR / f"{model}_lons.npy"), z_lons_sorted)
    print(f"  {model}: slopes + Tmax baselines saved", flush=True)

    return {"lats": z_lats_sorted, "lons": z_lons_sorted,
            "rr_local": rr_local,  "rr_200km": rr_200km, "n_years": n_yr}


# ─── Worker wrapper (top-level for pickling) ─────────────────────────────────

def process_model_worker(args):
    """Top-level wrapper for ProcessPoolExecutor (must be picklable)."""
    # Re-apply certifi patch in each subprocess (processes don't inherit monkey-patches)
    import certifi, certifi.core as _cc
    _cc.where = lambda: _CA_BUNDLE
    certifi.where = _cc.where
    os.environ["SSL_CERT_FILE"]      = _CA_BUNDLE
    os.environ["REQUESTS_CA_BUNDLE"] = _CA_BUNDLE

    model, _storage_options_unused = args
    # Fetch a fresh SAS token per worker so long-running jobs don't hit the 24h expiry
    try:
        storage_options = get_storage_options()
    except Exception as e:
        print(f"  [{model}] failed to refresh SAS token: {e}", flush=True)
        return model, None
    try:
        r = process_model(model, storage_options)
        if r is None:
            return model, None
        return model, {
            "n_years":      r["n_years"],
            "rr_local_med": float(np.nanmedian(r["rr_local"])),
            "rr_200km_med": float(np.nanmedian(r["rr_200km"])),
        }
    except Exception as e:
        print(f"  [{model}] FAILED: {e}", flush=True)
        traceback.print_exc()
        return model, None


# ─── ECS-weighted ensemble ────────────────────────────────────────────────────

def weighted_ensemble(results, weights, ref_lats, ref_lons):
    sum_local = np.zeros((len(ref_lats), len(ref_lons)), np.float64)
    sum_200km = np.zeros_like(sum_local)
    sum_w     = np.zeros_like(sum_local)

    for m, r in results.items():
        w  = weights[m]
        rl = r["rr_local"]
        r2 = r["rr_200km"]
        if not (np.array_equal(r["lats"], ref_lats) and np.array_equal(r["lons"], ref_lons)):
            da_l = xr.DataArray(rl, dims=["lat","lon"],
                                coords={"lat": r["lats"], "lon": r["lons"]})
            da_2 = xr.DataArray(r2, dims=["lat","lon"],
                                coords={"lat": r["lats"], "lon": r["lons"]})
            rl = da_l.interp(lat=ref_lats, lon=ref_lons, method="linear").values.astype(np.float32)
            r2 = da_2.interp(lat=ref_lats, lon=ref_lons, method="linear").values.astype(np.float32)
        valid = np.isfinite(rl) & np.isfinite(r2)
        sum_local[valid] += w * rl[valid]
        sum_200km[valid] += w * r2[valid]
        sum_w[valid]     += w

    with np.errstate(invalid="ignore", divide="ignore"):
        ens_local = np.where(sum_w > 0, sum_local / sum_w, np.nan).astype(np.float32)
        ens_200km = np.where(sum_w > 0, sum_200km / sum_w, np.nan).astype(np.float32)
    return ens_local, ens_200km


# ─── Comparison figure ────────────────────────────────────────────────────────

def make_comparison_figure(cil_local, cil_200km, ref_lats, ref_lons):
    """4-panel comparison figure — CIL vs IFS slopes (all RR per K)."""
    try:
        ds_l      = xr.open_dataset(str(SLOPE_LOCAL))
        ifs_local = ds_l["rr_per_degC"].values.copy()
        ifs_lats  = ds_l["latitude"].values
        ifs_lons  = ds_l["longitude"].values
        ds_l.close()
        ds_2      = xr.open_dataset(str(SLOPE_200KM))
        ifs_200km = ds_2["rr_per_degC"].values.copy()
        ds_2.close()
        for s in [ifs_local, ifs_200km]:
            s[~np.isfinite(s) | (s > 50)] = np.nan
        if ifs_lats[0] > ifs_lats[-1]:
            ifs_local = ifs_local[::-1, :]
            ifs_200km = ifs_200km[::-1, :]
            ifs_lats  = ifs_lats[::-1]
        have_ifs = True
    except Exception as e:
        print(f"  Could not load IFS slopes: {e}")
        have_ifs = False

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(
        f"EHD Sensitivity: CIL GDPCIR empirical (log-EHD ~ T_anom, {int(R_KM)}km) "
        "vs IFS reforecast\nRisk Ratio per K (log scale)",
        fontsize=12
    )
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=0.8, vmax=5)
    cmap = "YlOrRd"

    panels = [
        (axes[0, 0], cil_local,  ref_lats, ref_lons, "CIL — local RR/K"),
        (axes[0, 1], cil_200km,  ref_lats, ref_lons, f"CIL — {int(R_KM)}km RR/K"),
    ]
    if have_ifs:
        panels += [
            (axes[1, 0], ifs_local, ifs_lats, ifs_lons, "IFS reforecast — local (clima)"),
            (axes[1, 1], ifs_200km, ifs_lats, ifs_lons, f"IFS reforecast — {int(R_KM)}km (inter)"),
        ]
    else:
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)

    for ax, data, lats, lons, title in panels:
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                           shading="auto", rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="RR / K")

    plt.tight_layout()
    out = OUT_DIR / "cil_ehd_slopes_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CIL GDPCIR EHD Sensitivity Slopes  [PARALLEL]")
    print(f"  Years: {HIST_YEARS[0]}–{HIST_YEARS[-1]}  |  R_km={R_KM}  |  EHD_floor={EHD_FLOOR:.4f}")
    print(f"  Parallelism: {N_WORKERS} model workers × {YEAR_THREADS} year threads "
          f"= {N_WORKERS * YEAR_THREADS} max concurrent zarr reads")
    print("  EHD: CIL zarr daily Tmax > ERA5-Land 0.25° 95th-pctile threshold (continuous fraction)")
    print("  Method: OLS( log(EHD_CIL), T_anom_CIL )  →  exp(slope) = RR per K")
    print("=" * 70)

    available = [
        m for m in sorted(MODEL_INSTITUTION)
        if m in ECS_VALUES
    ]
    weights = compute_ecs_weights(available)

    print(f"\nModels: {len(available)}")
    cached = [m for m in available
              if all((PER_MODEL_SLOPES / f"{m}_{s}.nc").exists()
                     for s in ["local", "200km"])]
    pending = [m for m in available if m not in cached]
    print(f"  Cached: {len(cached)}  |  Pending: {len(pending)}")
    for m in available:
        tag = "[cached]" if m in cached else "[pending]"
        print(f"    {tag} {m}")

    # ── Fetch SAS token once ───────────────────────────────────────────────────
    print("\nFetching Planetary Computer SAS token...", flush=True)
    storage_options = get_storage_options()
    print("  Token obtained.", flush=True)

    # ── Parallel model processing ─────────────────────────────────────────────
    print(f"\nProcessing {len(available)} models ({N_WORKERS} workers)...", flush=True)
    t_start = time.time()
    summaries = {}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        args    = [(m, storage_options) for m in available]
        futures = {pool.submit(process_model_worker, a): a[0] for a in args}
        done    = 0
        for fut in as_completed(futures):
            m, summary = fut.result()
            summaries[m] = summary
            done += 1
            tag = "OK" if summary is not None else "FAILED/SKIP"
            print(f"  [{done}/{len(available)}] {m}  {tag}  "
                  f"({(time.time()-t_start)/60:.1f} min elapsed)", flush=True)

    print(f"\nAll models done in {(time.time()-t_start)/60:.1f} min", flush=True)

    # ── Load full arrays from disk for ensemble ────────────────────────────────
    # Workers only returned summaries to avoid pickling large arrays.
    # Reload per-model RR grids from the saved .nc files.
    results = {}
    import pandas as pd
    summary_rows = []

    for m in available:
        p_loc = PER_MODEL_SLOPES / f"{m}_local.nc"
        p_200 = PER_MODEL_SLOPES / f"{m}_200km.nc"
        if not p_loc.exists() or not p_200.exists():
            print(f"  {m}: output files missing — skip ensemble")
            continue
        ds_l = xr.open_dataset(str(p_loc), engine="netcdf4")
        ds_2 = xr.open_dataset(str(p_200), engine="netcdf4")
        rr_l = ds_l["rr_per_degC"].values
        rr_2 = ds_2["rr_per_degC"].values
        lats = ds_l["lat"].values
        lons = ds_l["lon"].values
        ds_l.close(); ds_2.close()
        results[m] = {"lats": lats, "lons": lons, "rr_local": rr_l, "rr_200km": rr_2,
                      "n_years": (summaries.get(m) or {}).get("n_years", -1)}
        summary_rows.append({
            "model":        m,
            "ecs":          ECS_VALUES[m],
            "weight":       weights[m],
            "n_years":      results[m]["n_years"],
            "rr_local_med": float(np.nanmedian(rr_l)),
            "rr_local_p90": float(np.nanpercentile(rr_l[np.isfinite(rr_l)], 90)),
            "rr_200km_med": float(np.nanmedian(rr_2)),
            "rr_200km_p90": float(np.nanpercentile(rr_2[np.isfinite(rr_2)], 90)),
        })

    # ── Per-model CSV ──────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    csv_path = OUT_DIR / "cil_ehd_slopes_per_model.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\nPer-model summary → {csv_path}")
    if not df.empty:
        print(df[["model", "ecs", "weight", "n_years", "rr_local_med", "rr_200km_med"]]
              .to_string(float_format="{:.4f}".format))
    else:
        print("  (no models completed successfully)")

    # ── ECS-weighted ensemble ─────────────────────────────────────────────────
    if not results:
        print("ERROR: no valid results"); return

    ref_model = next(iter(results))
    ref_lats  = results[ref_model]["lats"]
    ref_lons  = results[ref_model]["lons"]

    vw = {m: weights[m] for m in results}
    ws = sum(vw.values())
    vw = {m: w / ws for m, w in vw.items()}

    print(f"\nBuilding ECS-weighted ensemble ({len(results)} models)...")
    ens_local, ens_200km = weighted_ensemble(results, vw, ref_lats, ref_lons)
    print(f"  RR_local:  median={np.nanmedian(ens_local):.3f}  "
          f"valid={np.isfinite(ens_local).sum():,}")
    print(f"  RR_200km:  median={np.nanmedian(ens_200km):.3f}  "
          f"valid={np.isfinite(ens_200km).sum():,}")

    # ── Save ensemble NetCDF ──────────────────────────────────────────────────
    def _save(arr, path, desc):
        da = xr.DataArray(arr, dims=["lat", "lon"],
                          coords={"lat": ref_lats, "lon": ref_lons})
        ds = xr.Dataset({"rr_per_degC": da})
        ds["rr_per_degC"].attrs = {
            "long_name": desc, "units": "risk_ratio_per_K",
            "source": "CIL GDPCIR QDM historical 2000-2014",
            "method": "OLS(log EHD, T_anom), ECS-weighted ensemble",
        }
        ds.to_netcdf(str(path))
        print(f"  Saved {path}")

    _save(ens_local, OUT_DIR / "cil_ehd_slopes_local.nc",
          "CIL empirical EHD RR per K — local T anomaly")
    _save(ens_200km, OUT_DIR / "cil_ehd_slopes_200km.nc",
          f"CIL empirical EHD RR per K — {int(R_KM)}km Gaussian T anomaly")

    # ── Comparison figure ──────────────────────────────────────────────────────
    print("\nGenerating comparison figure...")
    make_comparison_figure(ens_local, ens_200km, ref_lats, ref_lons)

    print("\n" + "=" * 70)
    print("DONE")
    for f in ["cil_ehd_slopes_local.nc", "cil_ehd_slopes_200km.nc",
              "cil_ehd_slopes_per_model.csv", "cil_ehd_slopes_comparison.png"]:
        p = OUT_DIR / f
        if p.exists():
            print(f"  {p}  ({p.stat().st_size/1e6:.1f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
