# EHD Projection Scripts

Scripts for projecting future extreme heat day (EHD) frequency using CMIP6 GCMs and IFS reforecast sensitivity slopes.

## Pipeline

### 1. `compute_cil_ehd_slopes.py`
Computes empirical EHD sensitivity slopes from CIL GDPCIR QDM CMIP6 historical simulations (1950–2014).

- Downloads daily summer Tmax from 23 CMIP6 models via Planetary Computer Azure Blob Storage
- Computes EHD fraction (days > ERA5-Land 95th-percentile climatology threshold at 0.25°)
- Regresses log(EHD) on summer-mean Tmax anomaly → RR per K (local and 200km smoothed)
- ECS-weighted ensemble across models

**Environment**: `/turbo/mgoldklang/pyenvs/nov25/bin/python`

### 2. `compute_cil_ehd_projections.py`
Projects future EHD using GCM ΔT and the IFS/CIL sensitivity slopes.

### 3. `project_slopes_ehd_maps.py`
Main projection pipeline: produces EHD maps for multiple methods (V1–V6) at 0.25°,
aggregated to admin1/country level, with GDP-weighted damage estimates.

### 4. `validate_ehd_methods.py`
Historical out-of-sample validation: trains each method on 2000–2012, predicts 2013–2024,
compares against observed ERA5-Land EHD.

### 5. `plot_slope_comparison.py`
Generates comparison figures of IFS vs CIL empirical slopes.

### 6. `generate_html_maps.py`
Interactive HTML maps of projected EHD changes and GDP damages.

## Key Data Sources

| Data | Path | Notes |
|------|------|-------|
| CIL GDPCIR daily Tmax | Azure Blob via Planetary Computer | 23 CMIP6 models, 1950–2050 |
| ERA5-Land 95th pctile thresholds | `data/processed/climatologies/` | 0.25°, JJA (NH) and DJF (SH) |
| IFS reforecast slopes | `data/slopes/` | 2001–2020, local and 200km |
| ERA5-Land EHD (observed) | `/NImounts/NiClimateDev/...` | 0.1°, 2000–2024 |
| NEX-GDDP CMIP6 Tmax | `/NImounts/NiNoSnapUsers/.../tasmax/` | 0.25°, historical + SSP245 |
