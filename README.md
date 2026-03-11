# Extreme Temperature and Financial Markets

**Do extreme heat events damage economic output? Evidence from a global admin2 panel.**

This repository provides the full replication code and processed panel data for
a study of extreme summer heat and sub-national GDP growth across ~35,000
administrative-level-2 (admin2) units globally, covering 2000–2024.

The econometric design follows [Burke, Hsiang & Miguel (2015)](https://doi.org/10.1038/nature15725)
with an updated climate index: instead of annual mean temperature, we use
*exceedance frequency* — the fraction of summer days on which local daily
maximum temperature exceeds the calendar-day 95th percentile of the 1990–2020
climatology.  This index is population-weighted to the built-up surface area
(GHS) within each admin2 unit, ensuring the signal reflects heat where people
live and work.

**Key result:** A 10 percentage-point increase in summer exceedance frequency
reduces admin2 GDP growth by **0.36 pp** (β = −0.0361, Conley 500 km SE =
0.0079, t = −4.58) after controlling for annual mean temperature, precipitation,
entity fixed effects, year fixed effects, and country-specific quadratic time
trends.

---

## Future Extensions

This codebase will be extended to include IFS ensemble reforecast sensitivities
of the log-probability of exceeding the local 95th-percentile threshold as a
function of regional temperature change.  These sensitivity fields will be
compared against downscaled CMIP6 projections to improve the calibration of
climate damage functions for physical risk analysis.

---

## Repository Structure

```
Extreme_temperature_and_financial_markets/
│
├── README.md                          ← this file
│
├── src/                               ← all source code
│   ├── config.py                      ← centralised path configuration
│   │
│   ├── climate/
│   │   ├── compute_ehd_global.py      ← global EHD grids (NH JJA + SH DJF)
│   │   └── compute_ehd_conus.py       ← CONUS-only, faster, saves climatology
│   │
│   ├── aggregation/
│   │   ├── aggregate_heat_to_admin2.py          ← settlement-weighted EHD → admin2
│   │   └── aggregate_era5_controls_to_admin2.py ← annual T and P → admin2
│   │
│   └── econometrics/
│       └── panel_regression_gdp_heat.py         ← BHM-style panel regressions
│
└── data/
    ├── raw/                           ← inputs (not committed; see READMEs)
    │   ├── climate/README.md          ← ERA5-Land and ERA5 download instructions
    │   ├── gdp/README.md              ← GADM + GDL GDP data instructions
    │   └── shapes/README.md           ← GHS settlement raster instructions
    │
    └── processed/                     ← outputs (committed where feasible)
        ├── climatologies/
        │   ├── tmax_95th_pctl_doy_nh_jja_1990_2020.nc   ← CONUS JJA thresholds
        │   └── README.md
        └── panel/
            ├── admin2_heat_settlement_weighted.parquet   ← EHD panel (25 yr)
            ├── admin2_era5land_annual_controls.parquet   ← annual T and P panel
            ├── panel_regression_gdp_heat_results.csv     ← full regression table
            └── README.md
```

---

## Data Sources

| Dataset | Description | Source |
|---------|-------------|--------|
| ERA5-Land daily Tmax | 0.1°, 1990–2024, global | [Copernicus CDS](https://cds.climate.copernicus.eu/) |
| ERA5 monthly T and P | 0.25°, 2000–2024, global | [Copernicus CDS](https://cds.climate.copernicus.eu/) |
| GADM v4.1 | Admin2 polygons, ~48,000 units | [gadm.org](https://gadm.org/) |
| GDL Subnational HDI | GDP per capita 1990–2022 by admin2 | [globaldatalab.org](https://globaldatalab.org/shdi/) |
| GHS-BUILT-S R2023A | Built-up surface, 100 m, 6 epochs | [GHSL](https://ghsl.jrc.ec.europa.eu/ghs_buS2023.php) |

---

## Methodology

### 1. Extreme Heat Days (EHD)

For each 0.1° grid cell, we compute the fraction of summer days on which
daily maximum 2m temperature (Tmax) exceeds the calendar-day 95th percentile
of the 1990–2020 baseline.

**Percentile construction:** For each target day-of-year (DOY), we pool all
Tmax values within a centred 15-day window (±7 days) across all 31 baseline
years, yielding ~450 observations per DOY per grid cell.  The 95th percentile
of this distribution is the exceedance threshold.

**Hemispheric seasons:**
- Northern Hemisphere (lat ≥ 0): June–July–August (JJA).
- Southern Hemisphere (lat < 0): December–January–February (DJF).

**Settlement weighting:** Grid-cell EHD values are aggregated to admin2
polygons using the GHS Built-up Surface raster as weights, matched to the
nearest 5-year epoch (2000, 2005, …, 2025).  This concentrates the signal
on areas of human habitation rather than uninhabited land.

### 2. Climate Controls

Annual mean temperature (°C) and annual total precipitation (mm) are
aggregated from ERA5 0.25° monthly reanalysis data to admin2 via
area-weighted mean.  These serve as controls (T, P) in the main
specification, and (T, T², P) in the quadratic control variant.

### 3. Panel Regression

The primary regression is:

```
ΔlnGDPpc_it = β₁ · EHD_it + γ₁ · T_it + γ₂ · P_it
              + α_i + δ_t + f_c(t) + ε_it
```

where:
- `i` indexes admin2 units, `t` indexes years, `c(i)` the country of unit i
- `α_i` = admin2 entity fixed effect
- `δ_t` = year fixed effect
- `f_c(t)` = country-specific quadratic time trend (absorbs country-level
  structural change), implemented via Frisch-Waugh-Lovell (FWL) pre-regression

**Standard errors:** Conley (1999) spatial HAC with a Bartlett kernel at
250 km, 500 km, and 1,000 km cutoffs, implemented from scratch using a
`scipy.spatial.cKDTree` for efficiency.  Also reported: entity-clustered,
time-clustered, two-way clustered, and Driscoll-Kraay.

### 4. Robustness

| Specification | Description |
|---------------|-------------|
| `baseline` | Linear EHD, T + P controls |
| `quad_ctrl` | Linear EHD, T + T² + P controls |
| `spline_4df` | Natural cubic spline (4 df) on EHD |
| `quadratic` | EHD + EHD² with turning point |
| `quintile_bins` | EHD quintile dummies (Q1 = reference) |
| `residual_dv` | DV = country-cycle-residualised growth |
| `income_low/mid/high` | Subsample by GDP per capita tercile |
| `lag_lead` | Heat in year t → GDP in year t+1 |

---

## Reproducing the Results

### Prerequisites

Python 3.11+ with the following packages:

```
numpy pandas geopandas xarray rasterio
linearmodels statsmodels scipy patsy cfgrib eccodes
```

Install with pip:

```bash
pip install numpy pandas geopandas xarray rasterio linearmodels statsmodels scipy patsy cfgrib eccodes
```

### Option A — Use the pre-built panel (fastest)

The processed panel files are already in `data/processed/panel/`.  To
reproduce the regression results you need only supply the GDP growth parquet
(constructed from the publicly available GADM/GDL data; see
`data/raw/gdp/README.md`) and the GADM GeoPackage:

```bash
export GDP_GROWTH_PATH=/path/to/adm2_gdp_growth_residuals_long.parquet
export ADMIN2_GPKG=/path/to/polyg_adm2_gdp_perCapita_1990_2022.gpkg
python src/econometrics/panel_regression_gdp_heat.py
```

Results are written to `data/processed/panel/panel_regression_gdp_heat_results.csv`.

### Option B — Rebuild the full panel from ERA5

1. **Download ERA5-Land** daily Tmax and ERA5 monthly T/P.
   See `data/raw/climate/README.md` for instructions.

2. **Compute EHD grids** (global, ~6–12 hours on a multi-core machine):
   ```bash
   python src/climate/compute_ehd_global.py
   ```
   Or the faster CONUS-only version (~40 min):
   ```bash
   python src/climate/compute_ehd_conus.py
   ```

3. **Aggregate EHD to admin2** (~2–3 hours; requires GHS settlement rasters):
   ```bash
   python src/aggregation/aggregate_heat_to_admin2.py
   ```

4. **Aggregate ERA5 controls to admin2** (~1 hour):
   ```bash
   python src/aggregation/aggregate_era5_controls_to_admin2.py
   ```

5. **Run regressions** (~30–90 min depending on Conley cutoffs):
   ```bash
   python src/econometrics/panel_regression_gdp_heat.py
   ```

### Path configuration

All data paths are centralised in `src/config.py`.  Override any path via
environment variables (e.g., `ERA5_LAND_DIR`, `ADMIN2_GPKG`) without editing
the config file.

---

## Results Summary

### Baseline (Table 1)

| SE estimator | β | SE | t | p |
|---|---|---|---|---|
| Entity-clustered | −0.0361 | 0.00617 | −5.85 | <0.001 |
| Conley 250 km | −0.0361 | 0.00712 | −5.07 | <0.001 |
| **Conley 500 km** | **−0.0361** | **0.00787** | **−4.58** | **<0.001** |
| Conley 1000 km | −0.0361 | 0.00891 | −4.05 | <0.001 |
| Two-way clustered | −0.0361 | 0.00834 | −4.33 | <0.001 |
| Driscoll-Kraay | −0.0361 | 0.00769 | −4.69 | <0.001 |

### Income Heterogeneity

The effect is concentrated in lower-income admin2 units, consistent with
greater adaptive capacity in wealthier regions.

### Lag Structure

No significant lead effect (heat in year t → GDP in t+1), consistent with
contemporaneous damage rather than persistent damage channels.

---

## Citation

If you use this code or data, please cite:

> Goldklang, M. (2025). *Extreme Temperature and Financial Markets.*
> Working paper. https://github.com/Olivash/Extreme_temperature_and_financial_markets

---

## References

- Burke, M., Hsiang, S. M., & Miguel, E. (2015). Global non-linear effect of
  temperature on economic production. *Nature*, 527, 235–239.
  https://doi.org/10.1038/nature15725

- Conley, T. G. (1999). GMM estimation with cross sectional dependence.
  *Journal of Econometrics*, 92(1), 1–45.

- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal
  of the Royal Meteorological Society*, 146(730), 1999–2049.

- Schiavina, M., et al. (2023). GHS-BUILT-S R2023A — GHS built-up surface
  grid, derived from Sentinel2 composite and Landsat, multitemporal
  (1975–2030). European Commission, Joint Research Centre (JRC).

- Smits, J., & Permanyer, I. (2019). The Subnational Human Development
  Database. *Scientific Data*, 6, 190038.
