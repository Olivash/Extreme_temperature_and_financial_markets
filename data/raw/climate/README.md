# Raw Climate Data

## ERA5-Land Daily Maximum Temperature (0.1°)

**Used by:** `src/climate/compute_ehd_global.py`, `src/climate/compute_ehd_conus.py`

### Files expected

```
data/raw/climate/era5_land_tmax/
    era5_tmax_{YYYY}_{MM:02d}.nc      # one file per year-month
```

**Years required:**
- Climatology baseline: 1990–2020 (months 5–9 for NH JJA, months 11–3 for SH DJF)
- Frequency period:     2000–2024 (months 6–8 for NH, 12–2 for SH)

**Variable name:** `t2m` (Kelvin)
**Grid:** 0.1° resolution, latitude descending (90 → −90), longitude 0–360°
**Dimensions:** `(time, latitude, longitude)`

### How to download

1. Register at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).
2. Install the CDS API client:
   ```bash
   pip install cdsapi
   ```
3. Configure your API key in `~/.cdsapirc`.
4. Download using the script below (adapt years/months as needed):

```python
import cdsapi
c = cdsapi.Client()

for year in range(1990, 2025):
    for month in range(1, 13):
        c.retrieve(
            "reanalysis-era5-land",
            {
                "variable": "2m_temperature",
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": "12:00",          # daily maximum proxy; for true daily max
                                           # use hourly and resample
                "format": "netcdf",
            },
            f"data/raw/climate/era5_land_tmax/era5_tmax_{year}_{month:02d}.nc",
        )
```

> **Note on daily Tmax:** ERA5-Land provides hourly data. The scripts here
> use monthly files where each daily record represents the daily maximum 2m
> temperature. If your download contains hourly data, resample to daily max
> before placing files in this directory.

### Storage estimate

~150 GB for the full 1990–2024 period at 0.1° global resolution.

---

## ERA5 Monthly Mean Temperature and Precipitation (0.25°)

**Used by:** `src/aggregation/aggregate_era5_controls_to_admin2.py`

### Files expected

```
data/raw/climate/era5_025deg/
    2m_temperature/
        {YYYY}/
            era5_t2m_{YYYY}{MM:02d}.grib
    total_precipitation/
        {YYYY}/
            era5_tp_{YYYY}{MM:02d}.grib
```

**Years required:** 2000–2024

### How to download

```python
import cdsapi
c = cdsapi.Client()

for year in range(2000, 2025):
    for month in range(1, 13):
        # 2m temperature (monthly mean)
        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": "2m_temperature",
                "year": str(year),
                "month": f"{month:02d}",
                "time": "00:00",
                "format": "grib",
            },
            f"data/raw/climate/era5_025deg/2m_temperature/{year}/era5_t2m_{year}{month:02d}.grib",
        )
        # Total precipitation (monthly mean)
        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": "total_precipitation",
                "year": str(year),
                "month": f"{month:02d}",
                "time": "00:00",
                "format": "grib",
            },
            f"data/raw/climate/era5_025deg/total_precipitation/{year}/era5_tp_{year}{month:02d}.grib",
        )
```

### Storage estimate

~15 GB for 2000–2024.
