#!/usr/bin/env python -u
"""
Compute CIL GDPCIR ssp245 EHD projections to 2050 using three methods:

  V_raw:   direct EHD from pre-computed CIL ssp245 local files
  V_local: EHD_base × RR_local^ΔTmax_local(t)   (per-model local slope)
  V_200km: EHD_base × RR_200km^ΔTmax_200km(t)   (per-model 200km slope)

EHD_base = settlement-weighted admin1 mean EHD 2000-2014 per entity.
ΔTmax(t)  = summer-mean Tmax year t (from CIL ssp245 zarr) − hist mean.

Runs 19 models in parallel via ProcessPoolExecutor(max_workers=4).
Ssp245 Tmax downloaded from Planetary Computer CIL GDPCIR zarr (I/O bottleneck).
All other data (EHD, slopes, Tmax hist baselines) loaded from local files.

Outputs:
  projections/output/cil_projections/{model}_projections.parquet
  projections/output/cil_ensemble_ehd_ssp245.parquet      — ECS-wt ensemble
  projections/output/cil_ensemble_ehd_ssp245_figure.png   — timeseries

Usage:
  cd ~/projects/macro/extreme_heat/biodiversity_interactions
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \\
  REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \\
  nohup /turbo/mgoldklang/pyenvs/nov25/bin/python \\
      projections/scripts/compute_cil_ehd_projections.py \\
      > projections/output/cil_projections.log 2>&1 &
"""

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("SSL_CERT_FILE",      "/etc/ssl/certs/ca-certificates.crt")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJ_DIR         = Path(__file__).resolve().parent.parent
OUT_DIR          = PROJ_DIR / "output"
PROJ_OUTPUT_DIR  = OUT_DIR / "cil_projections"
CIL_SLOPES_DIR   = OUT_DIR / "cil_slopes"
TMAX_HIST_DIR    = OUT_DIR / "cil_tmax_hist"

CIL_EHD_DIR      = Path("/NImounts/NiNoSnapUsers/mgoldklang/climate/projections/ehd_cil")

# Admin1 mixed entities (from parent project)
HEAT_DIR         = PROJ_DIR.parent
CACHE_DIR        = HEAT_DIR / "cache"
ADMIN2_RASTER    = CACHE_DIR / "admin2_raster_01deg.npy"
ADMIN2_LOOKUP    = CACHE_DIR / "admin2_lookup_01deg.csv"
ADMIN1_RASTER    = CACHE_DIR / "admin1_raster_01deg.nc"
ADMIN1_LOOKUP    = CACHE_DIR / "admin1_lookup.csv"
PANEL_FILE       = HEAT_DIR / "output" / "panel_mixed_admin1level.parquet"

SETTLE_DIR       = Path("/home/mgoldklang/projects/climate_extremes/conus_equities/cache")
SETTLE_EPOCH     = 2020     # latest available epoch

for _d in [OUT_DIR, PROJ_OUTPUT_DIR]:
    _d.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
HIST_YEARS     = list(range(2000, 2015))   # EHD baseline
SSP245_YEARS   = list(range(2025, 2051))   # 26 years
ROLLING_WINDOW = 10                         # centred rolling mean
N_WORKERS      = 4    # model-level parallel workers (ProcessPoolExecutor)
YEAR_THREADS   = 4    # per-model year download threads (ThreadPoolExecutor)
                      # total concurrent zarr reads = N_WORKERS × YEAR_THREADS
R_KM           = 200.0
EHD_FLOOR      = 1.0 / 92.0

# 0.1° ERA5-resolution grid (descending lat — matches settlement/admin rasters)
ERA5_LATS = np.round(np.arange(90.0, -90.1, -0.1), 1)   # 1801 values
ERA5_LONS = np.round(np.arange(-180.0, 180.0, 0.1), 1)  # 3600 values

# ─── ECS weights ──────────────────────────────────────────────────────────────
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


# ─── Planetary Computer / zarr access ────────────────────────────────────────

def get_storage_options():
    import planetary_computer as pc, pystac_client
    cat   = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                      modifier=pc.sign_inplace)
    items = list(cat.search(collections=["cil-gdpcir-cc0"], max_items=1).items())
    if not items:
        raise RuntimeError("PC STAC: no items in cil-gdpcir-cc0")
    return pc.sign(items[0]).assets["tasmax"].extra_fields["xarray:open_kwargs"]["storage_options"]


def open_cil_zarr(institution, model, scenario, storage_options):
    import adlfs
    ens  = MODEL_ENSEMBLE.get(model, "r1i1p1f1")
    blob = f"cil-gdpcir/CMIP/{institution}/{model}/{scenario}/{ens}/day/tasmax/v1.1.zarr"
    fs   = adlfs.AzureBlobFileSystem(**storage_options)
    return xr.open_zarr(fs.get_mapper(blob), consolidated=False, chunks=None)


# ─── Parallel per-year Tmax download ─────────────────────────────────────────

def _fetch_year(yr, institution, model, scenario, storage_options,
                lat_sort, lon_sort):
    """
    Download summer-mean Tmax for one year by opening a dedicated zarr connection.
    Called in a thread — each thread gets its own AzureBlobFileSystem + zarr handle,
    which avoids zarr store contention and is fully thread-safe.
    """
    import os
    os.environ.setdefault("SSL_CERT_FILE",      "/etc/ssl/certs/ca-certificates.crt")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")
    ds   = open_cil_zarr(institution, model, scenario, storage_options)
    lats = ds.lat.values
    nh   = np.where(lats >= 0)[0]
    sh   = np.where(lats  < 0)[0]
    tmax = summer_tmax_year(ds, yr, nh, sh)
    ds.close()
    if tmax is None:
        return yr, None
    return yr, tmax[lat_sort, :][:, lon_sort]


def download_tmax_parallel(institution, model, scenario, storage_options,
                           years, lat_sort, lon_sort, n_threads=YEAR_THREADS):
    """
    Download summer-mean Tmax for all requested years in parallel.
    Returns (years_ok, tmax_stack) where tmax_stack is (n_ok, nlat, nlon).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as tac
    results = {}
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futs = {
            pool.submit(_fetch_year, yr, institution, model, scenario,
                        storage_options, lat_sort, lon_sort): yr
            for yr in years
        }
        for fut in tac(futs):
            yr, data = fut.result()
            if data is not None:
                results[yr] = data

    years_ok = sorted(results)
    if not years_ok:
        return [], None
    return years_ok, np.stack([results[y] for y in years_ok], axis=0)


# ─── Summer-mean Tmax (copied from compute_cil_ehd_slopes.py) ─────────────────

def summer_tmax_year(ds, year, nh_lat_idx, sh_lat_idx):
    """Summer-mean Tmax (K) for one year. NH=JJA, SH=DJF."""
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


# ─── Lat-varying Gaussian (copied from compute_cil_ehd_slopes.py) ─────────────

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


# ─── Grid helpers ─────────────────────────────────────────────────────────────

def regrid_to_zarr_grid(ehd_arr, ehd_lats, ehd_lons, zarr_lats, zarr_lons):
    """Bilinearly interpolate (lat, lon) EHD from fixed 0.25° grid onto zarr model grid."""
    da = xr.DataArray(ehd_arr.astype(np.float32), dims=["lat", "lon"],
                      coords={"lat": ehd_lats, "lon": ehd_lons})
    return (da.interp(lat=zarr_lats, lon=zarr_lons, method="linear")
              .values.astype(np.float32))


def regrid_cube_to_era5(cube, zarr_lats, zarr_lons):
    """
    Bilinearly regrid (n_year, lat, lon) array from zarr model grid to ERA5 0.1° grid.
    zarr_lats: ascending; zarr_lons: -180..180.
    Returns (n_year, 1801, 3600) float32 on ERA5_LATS (desc) × ERA5_LONS.
    """
    n_yr = cube.shape[0]
    src = xr.DataArray(
        cube,
        dims=["year", "lat", "lon"],
        coords={"year": np.arange(n_yr), "lat": zarr_lats, "lon": zarr_lons},
    )
    out = src.interp(lat=ERA5_LATS, lon=ERA5_LONS, method="linear").values.astype(np.float32)
    # Mask latitudes outside model coverage
    lat_min, lat_max = float(zarr_lats.min()), float(zarr_lats.max())
    out[:, ERA5_LATS < lat_min, :] = np.nan
    out[:, ERA5_LATS > lat_max, :] = np.nan
    return out


# ─── Admin entity aggregation ─────────────────────────────────────────────────

def load_mixed_raster():
    """
    Build combined (1801×3600) raster mapping 0.1° pixel → entity index (0..N-1).
    Returns: combined_raster (int32), entity_lookup (DataFrame), n_entities (int).
    """
    panel    = pd.read_parquet(str(PANEL_FILE), columns=["GID_2", "iso3", "is_admin1"])
    entities = (panel[["GID_2", "iso3", "is_admin1"]]
                .drop_duplicates("GID_2")
                .reset_index(drop=True))
    entities["entity_idx"] = np.arange(len(entities))
    n_entities = len(entities)

    # Admin2 raster
    admin2_grid = np.load(str(ADMIN2_RASTER))
    lk2 = pd.read_csv(str(ADMIN2_LOOKUP))
    adm2_ent = entities[~entities["is_admin1"]]
    rid2eidx = dict(
        zip(
            lk2.set_index("GID_2")["raster_id"].reindex(adm2_ent["GID_2"]).values,
            adm2_ent["entity_idx"].values,
        )
    )
    max_rid2 = int(admin2_grid.max()) + 1
    adm2_map = np.full(max_rid2, -1, dtype=np.int32)
    for rid, eidx in rid2eidx.items():
        if not (isinstance(rid, float) and np.isnan(rid)):
            adm2_map[int(rid)] = int(eidx)
    combined = np.where(admin2_grid >= 0,
                        adm2_map[np.clip(admin2_grid, 0, max_rid2 - 1)],
                        -1).astype(np.int32)

    # Admin1 raster (overlaid on top)
    ds1 = xr.open_dataset(str(ADMIN1_RASTER))
    admin1_grid = ds1["admin1_id"].values.astype(np.int32)
    ds1.close()
    lk1 = pd.read_csv(str(ADMIN1_LOOKUP))
    adm1_ent = entities[entities["is_admin1"]].copy()
    adm1_ent["GID_nmbr"] = adm1_ent["GID_2"].str.replace("ADM1_", "").astype(float)
    lk1_sub = lk1.merge(adm1_ent[["GID_nmbr", "entity_idx"]], on="GID_nmbr", how="inner")
    max_rid1 = int(admin1_grid.max()) + 1
    adm1_map = np.full(max_rid1, -1, dtype=np.int32)
    for _, row in lk1_sub.iterrows():
        rid = int(row["raster_id"])
        if 0 <= rid < max_rid1:
            adm1_map[rid] = int(row["entity_idx"])
    adm1_eidx = np.where(admin1_grid >= 0,
                         adm1_map[np.clip(admin1_grid, 0, max_rid1 - 1)],
                         -1).astype(np.int32)
    mask_adm1 = adm1_eidx >= 0
    combined[mask_adm1] = adm1_eidx[mask_adm1]

    return combined, entities, n_entities


def load_settlement():
    path = SETTLE_DIR / f"settlement_01deg_{SETTLE_EPOCH}.nc"
    ds   = xr.open_dataset(str(path))
    # Dimension may be 'latitude'/'longitude' or 'lat'/'lon'
    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"
    arr = ds["built_surface"].values.astype(np.float32)
    ds.close()
    return arr   # (1801, 3600), descending lat


def agg_to_entities(heat_2d, combined_raster, settlement, n_entities):
    """Settlement-weighted mean of (1801×3600) heat field → float32 (n_entities,)."""
    valid  = (combined_raster >= 0) & np.isfinite(heat_2d) & (settlement > 0)
    ids_v  = combined_raster[valid]
    heat_v = heat_2d[valid].astype(np.float64)
    sett_v = settlement[valid].astype(np.float64)
    wsum   = np.bincount(ids_v, weights=heat_v * sett_v, minlength=n_entities)
    wsm    = np.bincount(ids_v, weights=sett_v,          minlength=n_entities)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(wsm > 0, wsum / wsm, np.nan).astype(np.float32)


# ─── Per-model worker ─────────────────────────────────────────────────────────

def process_model(model, storage_options):
    """
    Process one model:
      1. Load slopes + Tmax hist baseline from disk.
      2. Compute EHD baseline (2000-2014 mean on zarr grid → regrid → entities).
      3. Load EHD ssp245 (local, 2025-2050 → regrid → entities) → V_raw.
      4. Download ssp245 Tmax from CIL zarr (2025-2050 summer means).
      5. Compute V_local and V_200km at zarr resolution → regrid → entities.
      6. Return long-format DataFrame.
    """
    t0 = time.time()

    # ── Check cache ────────────────────────────────────────────────────────────
    cache_path = PROJ_OUTPUT_DIR / f"{model}_projections.parquet"
    if cache_path.exists():
        print(f"  {model}: cached — loading from disk", flush=True)
        return pd.read_parquet(str(cache_path))

    # ── Load fixtures (fast local I/O) ─────────────────────────────────────────
    p_sl = CIL_SLOPES_DIR  / f"{model}_local.nc"
    p_s2 = CIL_SLOPES_DIR  / f"{model}_200km.nc"
    p_tl = TMAX_HIST_DIR   / f"{model}_local.npy"
    p_t2 = TMAX_HIST_DIR   / f"{model}_200km.npy"
    p_la = TMAX_HIST_DIR   / f"{model}_lats.npy"
    p_lo = TMAX_HIST_DIR   / f"{model}_lons.npy"
    for p in [p_sl, p_s2, p_tl, p_t2, p_la, p_lo]:
        if not p.exists():
            print(f"  {model}: missing {p.name} — skipping", flush=True)
            return None

    ds_sl    = xr.open_dataset(str(p_sl), engine="scipy")
    ds_s2    = xr.open_dataset(str(p_s2), engine="scipy")
    rr_local = ds_sl["rr_per_degC"].values.astype(np.float32)
    rr_200km = ds_s2["rr_per_degC"].values.astype(np.float32)
    ds_sl.close(); ds_s2.close()

    tmax_hist_local  = np.load(str(p_tl))    # (nlat, nlon) K on zarr grid
    tmax_hist_200km  = np.load(str(p_t2))    # (nlat, nlon) K, smoothed
    zarr_lats        = np.load(str(p_la))    # ascending
    zarr_lons        = np.load(str(p_lo))    # -180..180
    grid_deg         = float(np.abs(zarr_lats[1] - zarr_lats[0]))

    # ── EHD historical → baseline on zarr grid ─────────────────────────────────
    ehd_hist_path = CIL_EHD_DIR / model / "ehd_historical.nc"
    if not ehd_hist_path.exists():
        print(f"  {model}: no ehd_historical.nc — skipping", flush=True)
        return None
    ds_eh     = xr.open_dataset(str(ehd_hist_path), engine="scipy")
    ehd_lats  = ds_eh["lat"].values
    ehd_lons  = ds_eh["lon"].values
    hist_yrs  = ds_eh["year"].values
    hist_mask = np.isin(hist_yrs, HIST_YEARS)
    ehd_hist_cube = ds_eh["ehd"].values[hist_mask]   # (n, 720, 1440)
    ds_eh.close()

    # Subset EHD lats to zarr coverage, then interpolate to zarr grid
    lat_min, lat_max = zarr_lats.min(), zarr_lats.max()
    lat_mask  = (ehd_lats >= lat_min - 0.3) & (ehd_lats <= lat_max + 0.3)
    ehd_sub   = ehd_hist_cube[:, lat_mask, :]
    ehd_lsub  = ehd_lats[lat_mask]

    # Per-year interpolation to zarr grid, then mean
    frames = []
    for i in range(ehd_sub.shape[0]):
        frames.append(regrid_to_zarr_grid(ehd_sub[i], ehd_lsub, ehd_lons,
                                          zarr_lats, zarr_lons))
    ehd_base_zarr = np.nanmean(np.stack(frames, axis=0), axis=0).astype(np.float32)
    del ehd_hist_cube, ehd_sub, frames

    # ── EHD ssp245 → V_raw on zarr grid ──────────────────────────────────────
    ehd_ssp_path = CIL_EHD_DIR / model / "ehd_ssp245.nc"
    if not ehd_ssp_path.exists():
        print(f"  {model}: no ehd_ssp245.nc — skipping", flush=True)
        return None
    ds_sp    = xr.open_dataset(str(ehd_ssp_path), engine="scipy")
    ssp_yrs  = ds_sp["year"].values
    ssp_mask = np.isin(ssp_yrs, SSP245_YEARS)
    ehd_ssp  = ds_sp["ehd"].values[ssp_mask]   # (n_yr, 720, 1440)
    years_ok = ssp_yrs[ssp_mask].tolist()
    ds_sp.close()

    n_yr = len(years_ok)
    if n_yr == 0:
        print(f"  {model}: no ssp245 years in {SSP245_YEARS} — skipping", flush=True)
        return None

    # Regrid V_raw EHD to zarr grid
    v_raw_zarr = np.stack(
        [regrid_to_zarr_grid(ehd_ssp[i, lat_mask, :], ehd_lsub, ehd_lons,
                             zarr_lats, zarr_lons)
         for i in range(n_yr)],
        axis=0,
    )   # (n_yr, nlat_z, nlon_z)
    del ehd_ssp

    print(f"  {model}: local files loaded in {time.time()-t0:.0f}s  "
          f"[{n_yr} ssp245 years, zarr {len(zarr_lats)}×{len(zarr_lons)}]", flush=True)

    # ── Download ssp245 Tmax from CIL zarr (parallel across years) ───────────
    institution = MODEL_INSTITUTION[model]
    print(f"  {model}: downloading ssp245 Tmax ({len(years_ok)} years, "
          f"{YEAR_THREADS} threads)...", flush=True)

    # Need sort indices — open zarr briefly for metadata only
    ds_meta    = open_cil_zarr(institution, model, "ssp245", storage_options)
    z_lats_raw = ds_meta.lat.values
    z_lons_raw = ds_meta.lon.values
    ds_meta.close()

    z_lons_180 = z_lons_raw.copy()
    z_lons_180[z_lons_180 >= 180] -= 360.0
    lon_sort   = np.argsort(z_lons_180)
    lat_sort   = np.argsort(z_lats_raw)

    years_tmax, tmax_ssp = download_tmax_parallel(
        institution, model, "ssp245", storage_options,
        years_ok, lat_sort, lon_sort,
        n_threads=YEAR_THREADS,
    )

    if not years_tmax:
        print(f"  {model}: no ssp245 Tmax downloaded — skipping", flush=True)
        return None

    # Align year lists (V_raw and Tmax may differ if some years had no zarr data)
    common_yrs = sorted(set(years_ok) & set(years_tmax))
    raw_idx  = [years_ok.index(y)   for y in common_yrs]
    tmax_idx = [years_tmax.index(y) for y in common_yrs]
    v_raw_zarr = v_raw_zarr[raw_idx]
    tmax_ssp   = tmax_ssp[tmax_idx]
    n_yr = len(common_yrs)

    print(f"  {model}: zarr Tmax downloaded in {time.time()-t0:.0f}s  "
          f"[{n_yr} common years]", flush=True)

    # ── V_local: EHD_base × RR_local^ΔTmax_local ──────────────────────────────
    dt_local = tmax_ssp - tmax_hist_local[np.newaxis, :]          # (n_yr, nlat, nlon)
    # Clip extreme ΔTmax to avoid overflow in exponentiation
    dt_local  = np.clip(dt_local, -15.0, 15.0)
    log_rr_l  = np.log(np.maximum(rr_local, 0.5))[np.newaxis, :]  # (1, nlat, nlon)
    v_local_zarr = ehd_base_zarr[np.newaxis, :] * np.exp(log_rr_l * dt_local)
    v_local_zarr = np.clip(v_local_zarr, 0.0, 1.0).astype(np.float32)

    # ── V_200km: smooth ssp245 Tmax, then apply 200km slope ───────────────────
    tmax_ssp_smooth = np.stack(
        [lat_varying_gaussian(tmax_ssp[i], zarr_lats, r_km=R_KM, grid_deg=grid_deg)
         for i in range(n_yr)],
        axis=0,
    )   # (n_yr, nlat, nlon)
    dt_200km = tmax_ssp_smooth - tmax_hist_200km[np.newaxis, :]
    dt_200km  = np.clip(dt_200km, -15.0, 15.0)
    log_rr_2  = np.log(np.maximum(rr_200km, 0.5))[np.newaxis, :]
    v_200km_zarr = ehd_base_zarr[np.newaxis, :] * np.exp(log_rr_2 * dt_200km)
    v_200km_zarr = np.clip(v_200km_zarr, 0.0, 1.0).astype(np.float32)
    del tmax_ssp, tmax_ssp_smooth, dt_local, dt_200km

    print(f"  {model}: projections computed in {time.time()-t0:.0f}s", flush=True)

    # ── Regrid all variants to ERA5 0.1° ──────────────────────────────────────
    raw_01   = regrid_cube_to_era5(v_raw_zarr,   zarr_lats, zarr_lons)
    local_01 = regrid_cube_to_era5(v_local_zarr, zarr_lats, zarr_lons)
    k200_01  = regrid_cube_to_era5(v_200km_zarr, zarr_lats, zarr_lons)
    base_01  = regrid_cube_to_era5(ehd_base_zarr[np.newaxis, :], zarr_lats, zarr_lons)[0]
    del v_raw_zarr, v_local_zarr, v_200km_zarr

    # ── Load admin fixtures inside worker (avoids pickling large arrays) ───────
    combined_raster, entity_lookup, n_entities = load_mixed_raster()
    settlement = load_settlement()

    # ── Settlement-weighted aggregation to entities ────────────────────────────
    ehd_base_ent = agg_to_entities(base_01, combined_raster, settlement, n_entities)
    v_raw_ent   = np.stack([agg_to_entities(raw_01[i],   combined_raster, settlement, n_entities)
                             for i in range(n_yr)], axis=0)
    v_local_ent = np.stack([agg_to_entities(local_01[i], combined_raster, settlement, n_entities)
                             for i in range(n_yr)], axis=0)
    v_200km_ent = np.stack([agg_to_entities(k200_01[i],  combined_raster, settlement, n_entities)
                             for i in range(n_yr)], axis=0)
    del raw_01, local_01, k200_01

    print(f"  {model}: entity aggregation done in {time.time()-t0:.0f}s", flush=True)

    # ── Build DataFrame ────────────────────────────────────────────────────────
    # Long format: (n_yr × n_entities) rows
    year_arr   = np.repeat(common_yrs, n_entities)
    entity_arr = np.tile(np.arange(n_entities), n_yr)
    df = pd.DataFrame({
        "year":      year_arr.astype(np.int16),
        "entity_idx": entity_arr.astype(np.int32),
        "ehd_base":  np.tile(ehd_base_ent, n_yr).astype(np.float32),
        "v_raw":     v_raw_ent.ravel().astype(np.float32),
        "v_local":   v_local_ent.ravel().astype(np.float32),
        "v_200km":   v_200km_ent.ravel().astype(np.float32),
    })

    df.to_parquet(str(cache_path), index=False)
    size_mb = cache_path.stat().st_size / 1e6
    print(f"  {model}: saved → {cache_path.name}  ({size_mb:.1f} MB)  "
          f"total {time.time()-t0:.0f}s", flush=True)
    return df


def process_model_worker(args):
    """Picklable top-level wrapper for ProcessPoolExecutor."""
    model, storage_options = args
    try:
        return model, process_model(model, storage_options)
    except Exception as e:
        print(f"  [{model}] FAILED: {e}", flush=True)
        traceback.print_exc()
        return model, None


# ─── ECS-weighted ensemble ────────────────────────────────────────────────────

def ecs_ensemble(results, weights, variants=("v_raw", "v_local", "v_200km")):
    """
    Given {model: DataFrame}, compute ECS-weighted ensemble stats per
    (year, entity_idx) for each variant.
    Returns a DataFrame with columns: year, entity_idx, {variant}_mean,
    {variant}_p5, {variant}_p95, and ehd_base (from first valid model).
    """
    models = [m for m in results if results[m] is not None]
    if not models:
        raise RuntimeError("No valid model results")

    w_total = sum(weights[m] for m in models)
    wt      = {m: weights[m] / w_total for m in models}

    ref_df = results[models[0]][["year", "entity_idx", "ehd_base"]].drop_duplicates(
        ["year", "entity_idx"]).sort_values(["year", "entity_idx"]).reset_index(drop=True)

    out = ref_df.copy()
    for var in variants:
        # Stack model arrays: (n_models, n_year×n_entity)
        stacked = np.stack(
            [results[m].sort_values(["year", "entity_idx"])[var].values for m in models],
            axis=0,
        )   # (n_models, N)
        w_arr = np.array([wt[m] for m in models])[:, np.newaxis]
        ens_mean = np.nansum(stacked * w_arr, axis=0)   # weighted mean
        # Percentiles (unweighted across models for simplicity)
        ens_p5   = np.nanpercentile(stacked, 5,  axis=0)
        ens_p95  = np.nanpercentile(stacked, 95, axis=0)

        out[f"{var}_mean"] = ens_mean.astype(np.float32)
        out[f"{var}_p5"]   = ens_p5.astype(np.float32)
        out[f"{var}_p95"]  = ens_p95.astype(np.float32)

    return out


def add_rolling_mean(df, window=ROLLING_WINDOW,
                     variants=("v_raw", "v_local", "v_200km")):
    """Add {var}_mean_roll (centre=True rolling) columns grouped by entity_idx."""
    df = df.sort_values(["entity_idx", "year"]).copy()
    for var in variants:
        col = f"{var}_mean"
        roll = (df.groupby("entity_idx")[col]
                  .transform(lambda s: s.rolling(window, center=True, min_periods=1).mean()))
        df[f"{var}_roll"] = roll.astype(np.float32)
    return df


# ─── Figures ──────────────────────────────────────────────────────────────────

def make_timeseries_figure(ens_df, entity_lookup, out_path):
    """
    Global settlement-weighted mean EHD for each variant + ensemble spread.
    Three panels: V_raw, V_local, V_200km.
    """
    # Aggregate across all entities (simple mean for global signal)
    gb = ens_df.groupby("year")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    labels = {"v_raw": "V_raw (direct EHD)",
              "v_local": "V_local (local slope × ΔTmax)",
              "v_200km": "V_200km (200km slope × ΔTmax)"}

    for ax, var in zip(axes, ("v_raw", "v_local", "v_200km")):
        mn  = gb[f"{var}_mean"].mean()
        p5  = gb[f"{var}_p5"].mean()
        p95 = gb[f"{var}_p95"].mean()
        years = mn.index.values

        ax.fill_between(years, p5, p95, alpha=0.25, color="steelblue", label="Model p5–p95")
        ax.plot(years, mn,  color="steelblue",  lw=2, label="ECS-wt mean")

        # Add 10-yr rolling mean if available
        if f"{var}_roll" in ens_df.columns:
            roll = gb[f"{var}_roll"].mean()
            ax.plot(roll.index.values, roll.values, color="darkred",
                    lw=2.5, ls="--", label=f"{ROLLING_WINDOW}-yr rolling mean")

        ax.set_title(labels[var], fontsize=10)
        ax.set_xlabel("Year"); ax.set_ylabel("EHD fraction")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "CIL GDPCIR ssp245 EHD Projections 2025–2050 — Global Settlement-weighted Mean",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CIL GDPCIR ssp245 EHD Projections")
    print(f"  Variants: V_raw | V_local | V_200km")
    print(f"  Years: {SSP245_YEARS[0]}–{SSP245_YEARS[-1]}  |  N_WORKERS={N_WORKERS}")
    print("=" * 70)

    # Models with slopes already computed
    available = sorted(
        m for m in MODEL_INSTITUTION
        if (CIL_SLOPES_DIR / f"{m}_local.nc").exists()
        and (CIL_EHD_DIR / m / "ehd_ssp245.nc").exists()
        and m in ECS_VALUES
    )
    print(f"\nModels with slopes + ssp245 EHD: {len(available)}")
    for m in available:
        cached = (PROJ_OUTPUT_DIR / f"{m}_projections.parquet").exists()
        print(f"  {'[cached]' if cached else '        '} {m}")

    # Fetch SAS token once (valid ~24h; passed to all workers)
    print("\nFetching Planetary Computer SAS token...", flush=True)
    storage_options = get_storage_options()
    print("  Token obtained.", flush=True)

    # ── Parallel processing ───────────────────────────────────────────────────
    print(f"\nProcessing {len(available)} models with {N_WORKERS} workers...", flush=True)
    t_start = time.time()

    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        args = [(m, storage_options) for m in available]
        futures = {pool.submit(process_model_worker, a): a[0] for a in args}
        for fut in as_completed(futures):
            model = futures[fut]
            try:
                _, df = fut.result()
                results[model] = df
                done = sum(1 for v in results.values() if v is not None)
                print(f"\n  [{done}/{len(available)}] {model} complete", flush=True)
            except Exception as e:
                print(f"\n  [{model}] worker exception: {e}", flush=True)
                results[model] = None

    elapsed = time.time() - t_start
    print(f"\nAll models done in {elapsed/60:.1f} min", flush=True)

    valid_models = [m for m in results if results[m] is not None]
    print(f"  Valid: {len(valid_models)} / {len(available)}")

    if not valid_models:
        print("ERROR: no valid results — exiting")
        return

    # ── ECS-weighted ensemble ─────────────────────────────────────────────────
    print("\nBuilding ECS-weighted ensemble...", flush=True)
    weights = compute_ecs_weights(valid_models)
    _, entity_lookup, _ = load_mixed_raster()

    ens_df = ecs_ensemble({m: results[m] for m in valid_models}, weights)
    ens_df = add_rolling_mean(ens_df)

    ens_path = OUT_DIR / "cil_ensemble_ehd_ssp245.parquet"
    ens_df.to_parquet(str(ens_path), index=False)
    print(f"  Ensemble saved → {ens_path}  ({ens_path.stat().st_size/1e6:.1f} MB)")
    print(f"  Shape: {ens_df.shape}  |  years: {sorted(ens_df['year'].unique())}")
    for var in ("v_raw", "v_local", "v_200km"):
        col = f"{var}_mean"
        print(f"  {var}: global mean EHD {ens_df[col].mean():.4f}  "
              f"(base {ens_df['ehd_base'].mean():.4f})")

    # ── Figure ────────────────────────────────────────────────────────────────
    print("\nGenerating timeseries figure...", flush=True)
    fig_path = OUT_DIR / "cil_ensemble_ehd_ssp245_figure.png"
    make_timeseries_figure(ens_df, entity_lookup, fig_path)

    print("\n" + "=" * 70)
    print("DONE")
    for p in [ens_path, fig_path]:
        if p.exists():
            print(f"  {p}  ({p.stat().st_size/1e6:.1f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
