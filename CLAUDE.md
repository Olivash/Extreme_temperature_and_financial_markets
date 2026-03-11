# CLAUDE.md — Project Context for Claude Code

Academic repo: econometric analysis of extreme heat and financial/economic outcomes.
Ported from internal Man Group research (`~/projects/macro/extreme_heat`).
No Man-specific code, data, or alpha logic. Pure academic pipeline.

---

## Repo Structure

```
src/
  config.py                              # All paths; override via env vars
  climate/
    compute_ehd_conus.py                 # CONUS EHD + saves JJA CONUS climatology
    compute_ehd_global.py                # Global tiled EHD (NH JJA + SH DJF)
    compute_climatology_sh_djf.py        # Global SH DJF 95th-pctl climatology
    compute_climatology_nh_jja_global.py # Global NH JJA 95th-pctl climatology
    coarsen_to_era5_grid.py              # Coarsen 0.1° → ERA5 0.25° grid
  aggregation/
    aggregate_heat_to_admin2.py          # Settlement-weighted EHD → admin2
    aggregate_era5_controls_to_admin2.py # Annual T and P → admin2
  econometrics/
    panel_regression_gdp_heat.py         # BHM-style panel regression
data/
  raw/
    climate/README.md      # ERA5-Land + ERA5 CDS download instructions
    gdp/README.md          # GADM/GDL GDP growth parquet construction
    shapes/README.md       # GHS settlement raster downloads
  processed/
    climatologies/         # 95th-pctl threshold NetCDFs (see README there)
    panel/                 # Ready-to-use parquets + regression results CSV
```

---

## Processed Data Files

### Already present

| File | Size | Notes |
|------|------|-------|
| `data/processed/climatologies/tmax_95th_pctl_doy_nh_jja_1990_2020.nc` | 54 MB | CONUS only (25–50°N, 125–65°W), JJA DOYs 152–244 |
| `data/processed/climatologies/tmax_95th_pctl_doy_nh_jja_1990_2020_025deg.nc` | 369 MB | CONUS coarsened to full ERA5 global 0.25° grid |
| `data/processed/panel/admin2_heat_settlement_weighted.parquet` | 13 MB | EHD panel, admin2, settlement-weighted |
| `data/processed/panel/admin2_era5land_annual_controls.parquet` | 11 MB | Annual T + P at admin2 |
| `data/processed/panel/panel_regression_gdp_heat_results.csv` | 13 KB | Regression results (β = −0.036, Conley 500km t = −4.58) |

### Being generated (may still be running)

| File | Script | Log |
|------|--------|-----|
| `data/processed/climatologies/tmax_95th_pctl_doy_sh_djf_1990_2020.nc` | `compute_climatology_sh_djf.py` | `data/processed/climatologies/compute_sh_djf.log` |
| `data/processed/climatologies/tmax_95th_pctl_doy_sh_djf_1990_2020_025deg.nc` | `coarsen_to_era5_grid.py` (auto-watcher) | `data/processed/climatologies/coarsen_sh_djf.log` |
| `data/processed/climatologies/tmax_95th_pctl_doy_nh_jja_global_1990_2020.nc` | `compute_climatology_nh_jja_global.py` | `data/processed/climatologies/compute_nh_jja_global.log` |
| `data/processed/climatologies/tmax_95th_pctl_doy_nh_jja_global_1990_2020_025deg.nc` | `coarsen_to_era5_grid.py` (auto-watcher) | `data/processed/climatologies/coarsen_nh_jja_global.log` |

---

## Checking / Restarting Jobs

### Check if still running

```bash
pgrep -fa "compute_climatology"
tail -5 data/processed/climatologies/compute_sh_djf.log
tail -5 data/processed/climatologies/compute_nh_jja_global.log
```

### Check if output files landed

```bash
ls -lh data/processed/climatologies/*.nc
```

### Re-run SH DJF if interrupted

```bash
ERA5_LAND_DIR=/NImounts/NiClimateDev/data/historical/era5_land/summer/tmax \
  nohup /turbo/mgoldklang/pyenvs/peg_nov_24/bin/python \
  src/climate/compute_climatology_sh_djf.py \
  > data/processed/climatologies/compute_sh_djf.log 2>&1 &
```

### Re-run NH JJA global if interrupted

```bash
ERA5_LAND_DIR=/NImounts/NiClimateDev/data/historical/era5_land/summer/tmax \
  nohup /turbo/mgoldklang/pyenvs/peg_nov_24/bin/python \
  src/climate/compute_climatology_nh_jja_global.py \
  > data/processed/climatologies/compute_nh_jja_global.log 2>&1 &
```

### Coarsen once both files exist

```bash
/turbo/mgoldklang/pyenvs/peg_nov_24/bin/python src/climate/coarsen_to_era5_grid.py \
  data/processed/climatologies/tmax_95th_pctl_doy_sh_djf_1990_2020.nc \
  data/processed/climatologies/tmax_95th_pctl_doy_nh_jja_global_1990_2020.nc
```

---

## Key Technical Decisions

- **All paths** live in `src/config.py`; every path has an env var override.
- **ERA5-Land source**: `/NImounts/NiClimateDev/data/historical/era5_land/summer/tmax/` — files named `era5_tmax_{YYYY}_{MM:02d}.nc`, variable `t2m` in Kelvin, lon 0–360°.
- **Python env**: `/turbo/mgoldklang/pyenvs/peg_nov_24/bin/python` (Python 3.8 — use `Union[X, Y]` not `X | Y` type hints).
- **Output format**: all climatology files are **unnamed `xr.DataArray`** (variable `__xarray_dataarray_variable__`), loadable with `xr.open_dataarray()`. Do NOT save as `xr.Dataset`.
- **Lon convention**: ERA5-Land files use 0–360°; all output files use −180 → +179.9°.
- **Coarsening**: nearest-0.25° bincount averaging (0.25/0.1 = 2.5 is non-integer; each ERA5 cell gets 2–3 contributing 0.1° cells). Output is full global ERA5 grid (721 lat × 1440 lon), NaN outside the source domain.
- **Tile format**: the global compute scripts use `np.ix_(lat_idx, lon_idx)` to assign tile results into a pre-allocated global array.

---

## Climatology File Conventions

| Hemisphere | Season | DOY range | Lat range | Output lat top |
|------------|--------|-----------|-----------|----------------|
| NH (CONUS) | JJA | 152–244 | 25–50°N | 50°N → 25°N descending |
| NH (global) | JJA | 153–244 | 0–90°N | 90°N → 0° descending |
| SH (global) | DJF | 335–365, 1–59 | 0–90°S | 0° → −90° descending |

DJF year convention: "DJF of year Y" = {Nov, Dec of Y−1} + {Jan, Feb, Mar of Y}.

---

## Planned Future Work

- IFS reforecast sensitivities: `d log P(exceed 95th) / d T_regional` as a function of regional temperature change
- Comparison against downscaled CMIP6 models
- Use the `_025deg` climatology files (compatible with ERA5 control variables and CMIP6 output)
