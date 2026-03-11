# Processed Panel Data

These files constitute the analysis-ready panel used in the regressions.
They can be used directly to reproduce all results in
`src/econometrics/panel_regression_gdp_heat.py` without re-running the
upstream climate and aggregation pipeline.

---

## admin2_heat_settlement_weighted.parquet

Settlement-weighted extreme-heat exceedance frequency per admin2 unit per year.

**Produced by:** `src/aggregation/aggregate_heat_to_admin2.py`

| Column | Type | Description |
|--------|------|-------------|
| `GID_2` | str | GADM admin2 unique ID |
| `adm2ID` | int | Numeric admin2 ID |
| `iso3` | str | ISO 3166-1 alpha-3 country code |
| `NAME_2` | str | Admin2 name |
| `year` | int | Calendar year (2000–2024) |
| `heat_freq_weighted` | float32 | Fraction of JJA (NH) / DJF (SH) days exceeding the per-DOY 95th percentile, weighted by GHS built-up surface m² |
| `heat_freq_unweighted` | float32 | Same, but simple area-weighted mean (no settlement weighting) |
| `total_settlement_m2` | float64 | Total GHS built-up surface in admin2 (m²) |
| `n_cells` | int32 | Number of 0.1° grid cells contributing |
| `gdp_per_capita` | float | GDP per capita (USD, PPP) from GADM/GDL |
| `raster_id` | int | Internal rasterisation index |

**Coverage:** ~35,000–48,000 admin2 regions × 25 years = ~850,000 rows.
**Period:** 2000–2024 (heat); matched GDP available 2000–2022.

---

## admin2_era5land_annual_controls.parquet

Annual mean temperature and total precipitation per admin2 unit.
Used as climate controls (T, T², P) in the panel regressions.

**Produced by:** `src/aggregation/aggregate_era5_controls_to_admin2.py`

| Column | Type | Description |
|--------|------|-------------|
| `GID_2` | str | GADM admin2 unique ID |
| `iso3` | str | ISO 3166-1 alpha-3 country code |
| `year` | int | Calendar year (2000–2024) |
| `annual_temp_c` | float32 | Annual mean 2m temperature (°C), area-weighted |
| `annual_precip_mm` | float32 | Annual total precipitation (mm), area-weighted |
| `n_cells_025` | int32 | Number of 0.25° ERA5 grid cells contributing |

---

## panel_regression_gdp_heat_results.csv

Full regression output table from all specifications.

**Produced by:** `src/econometrics/panel_regression_gdp_heat.py`

| Column | Description |
|--------|-------------|
| `spec` | Specification label (e.g., `linear_ctrl_baseline`) |
| `depvar` | Dependent variable (`admin2_growth` or `residual`) |
| `variable` | Regressor name |
| `se_type` | Standard error estimator (see below) |
| `beta` | Coefficient estimate |
| `se` | Standard error |
| `tstat` | t-statistic |
| `pval` | Two-sided p-value |
| `r2_within` | Within R² |
| `n_obs` | Observation count |
| `n_entities` | Entity count |

**SE types included:**
- `entity_clustered` — clustered by admin2
- `time_clustered` — clustered by year
- `twoway_clustered` — two-way clustered
- `driscoll_kraay` — Driscoll-Kraay kernel HAC
- `conley_250km`, `conley_500km`, `conley_1000km` — Conley spatial HAC

**Key result:**
```
Specification : linear_ctrl_baseline
Variable      : heat_freq_weighted
β             = −0.0361
Conley 500km SE = 0.00787
t             = −4.58   (p < 0.001)
```

Interpretation: a 10 percentage point increase in the fraction of summer
days exceeding the local 95th-percentile temperature threshold is associated
with a 0.36% reduction in admin2 GDP growth, controlling for annual mean
temperature, total precipitation, entity fixed effects, year fixed effects,
and country-specific quadratic time trends.
