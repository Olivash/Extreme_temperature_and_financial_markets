# Processed Climatologies

Calendar-day 95th-percentile thresholds for daily maximum 2m temperature
(ERA5-Land, 1990–2020 baseline).  Used to define "exceedance" events and
to compute the EHD (extreme heat day frequency) panel.

All files are loadable with `xr.open_dataarray(path)`.

---

## tmax_95th_pctl_doy_nh_jja_1990_2020.nc

| Property | Value |
|----------|-------|
| Domain   | CONUS (25°N–50°N, 125°W–65°W) |
| Season   | Northern Hemisphere summer: JJA (June–August) |
| Resolution | 0.1° (251 lat × 601 lon) |
| Window   | ±7 calendar days centred on each target DOY (15-day pool) |
| Percentile | 95th |
| Variable | unnamed DataArray (°C) |
| Dimensions | `(dayofyear, latitude, longitude)` |
| DOY range | JJA days only (DOY 152–244) |
| File size | ~54 MB |

**Produced by:** `src/climate/compute_ehd_conus.py`

```python
import xarray as xr
pctl = xr.open_dataarray("data/processed/climatologies/tmax_95th_pctl_doy_nh_jja_1990_2020.nc")
# pctl.sel(dayofyear=182)  → threshold map for 1 July
```

---

## tmax_95th_pctl_doy_nh_jja_global_1990_2020.nc

| Property | Value |
|----------|-------|
| Domain   | Full Northern Hemisphere (0°–90°N, global) |
| Season   | Northern Hemisphere summer: JJA (June–August) |
| Resolution | 0.1° (901 lat × 3600 lon) |
| Window   | ±7 calendar days centred on each target DOY (15-day pool) |
| Percentile | 95th |
| Variable | unnamed DataArray (°C) |
| Dimensions | `(dayofyear, latitude, longitude)` |
| DOY range | JJA days only (DOY 153–244) |

**Produced by:** `src/climate/compute_climatology_nh_jja_global.py`

```python
import xarray as xr
pctl = xr.open_dataarray("data/processed/climatologies/tmax_95th_pctl_doy_nh_jja_global_1990_2020.nc")
# pctl.sel(dayofyear=182)  → threshold map for 1 July (global NH)
```

---

## tmax_95th_pctl_doy_sh_djf_1990_2020.nc

| Property | Value |
|----------|-------|
| Domain   | Full Southern Hemisphere (0°–90°S) |
| Season   | Southern Hemisphere summer: DJF (December–January–February) |
| Resolution | 0.1° (901 lat × 3600 lon) |
| Window   | ±7 calendar days centred on each target DOY (15-day pool) |
| Percentile | 95th |
| Variable | unnamed DataArray (°C) |
| Dimensions | `(dayofyear, latitude, longitude)` |
| DOY range | DJF days only (DOY ≈ 335–365 + 1–59) |

**Produced by:** `src/climate/compute_climatology_sh_djf.py`

```bash
ERA5_LAND_DIR=/path/to/era5_land_tmax python src/climate/compute_climatology_sh_djf.py
```

```python
import xarray as xr
pctl = xr.open_dataarray("data/processed/climatologies/tmax_95th_pctl_doy_sh_djf_1990_2020.nc")
# pctl.sel(dayofyear=15)   → threshold map for 15 January
# pctl.sel(dayofyear=355)  → threshold map for 21 December
```

---

## _025deg variants

Coarsened versions of the above files on the native ERA5 0.25° grid
(721 lat × 1440 lon).  These are compatible with the ERA5 climate
control variables (`admin2_era5land_annual_controls.parquet`) and with
CMIP6 output which is typically at ≥0.25° resolution.

| File | Source |
|------|--------|
| `tmax_95th_pctl_doy_nh_jja_1990_2020_025deg.nc` | NH JJA CONUS at 0.25° |
| `tmax_95th_pctl_doy_nh_jja_global_1990_2020_025deg.nc` | NH JJA global at 0.25° |
| `tmax_95th_pctl_doy_sh_djf_1990_2020_025deg.nc` | SH DJF global at 0.25° |

**Produced by:** `src/climate/coarsen_to_era5_grid.py`

```bash
python src/climate/coarsen_to_era5_grid.py \
    data/processed/climatologies/tmax_95th_pctl_doy_nh_jja_1990_2020.nc \
    data/processed/climatologies/tmax_95th_pctl_doy_sh_djf_1990_2020.nc
```

**Coarsening method:** For each ERA5 0.25° cell, all ERA5-Land 0.1° cells
whose centres round to the same 0.25° grid point are averaged (mean of
valid/non-NaN values).  Since 0.25/0.1 = 2.5 is non-integer, each ERA5 cell
receives 2–3 contributing 0.1° cells in an alternating pattern.

---

## Notes on DJF year convention

For the SH DJF file, "DJF of year Y" means:
- December and November of year Y−1
- January, February, and March of year Y

For example, "DJF 2000" uses November–December 1999 and January–March 2000.
The DOY axis uses standard calendar DOYs: December days are DOY 335–365,
January days are DOY 1–31, February days are DOY 32–59.
