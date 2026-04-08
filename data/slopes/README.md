---

## Data

### Reforecast Data
- **Source**: ECMWF medium range seasonal reforecasts (MRA001_AYIM_REFORCST)
- **Resolution**: 0.25° global
- **Ensemble**: 11 members (1 control + 10 perturbed)
- **Hindcast years**: 2001–2020
- **Seasons**: JJA (June–August), DJF (December–February)
- **Variable**: 2m temperature (t2m), day-12 lead time forecast, daily maximum

### Climatological Thresholds
ERA5-based 95th percentile thresholds computed over 1990–2020, varying by 
day-of-year and grid point:
- `tmax_95th_pctl_doy_nh_jja_global_1990_2020_025deg.zarr` — JJA threshold (NH)
- `tmax_95th_pctl_doy_sh_djf_1990_2020_025deg.zarr` — DJF threshold (SH)

---

## Methodology

### 1. Exceedance Probability
For each grid point and hindcast year, we count how many of the 275 ensemble 
members (25 initialisation dates × 11 members) exceed the ERA5 climatological 
95th percentile threshold for that day of year. The probability of exceedance 
is the fraction of members exceeding the threshold per year.

A floor of 1/n_members is applied before taking the log to avoid log(0).

### 2. Spatial Smoothing
Regional temperature anomalies are computed by applying a Gaussian spatial 
filter at multiple radii (50, 100, 200, 500, 1000, 1500, 2000 km). The 
smoothing radius is latitude-dependent in longitude to account for convergence 
of meridians. Anomalies are computed relative to the 2001–2020 mean.


### 3. Interannual  Slope Regression
For each grid point, we regress log(P(exceedance)) against the 200km 
Gaussian-smoothed regional T2m anomaly across the 20 hindcast years. The slope 
gives the rate of change of log-probability per °C of regional warming.

### 4. Upper Tails 
For each grid point, we pool all the data together, years, ensemble,days, then we find howthe top 5 percent is changing with increasing temperature magnitude. Assuming the tail shape is constant (only the location shifts with temperature), 


### 4. Global Summer
A global summer field is constructed by stitching:
- **Northern Hemisphere (lat ≥ 0)**: JJA slopes
- **Southern Hemisphere (lat < 0)**: DJF slopes

### 5. Threshold Sensitivity
Two versions of the global summer risk ratio are computed:
- **ERA5 climatology**: external threshold from ERA5 1990–2020
- **Internal climatology**: threshold from the reforecast's own distribution

The difference map highlights where the choice of climatology matters.

---

## Output Files (`data/slopes/`)

| File | Description |
|------|-------------|
| `jja_upper_tail_slopes.nc` | JJA raw log-prob slopes per temperature magnitudes per grid point,Upper Tails  |
| `jja_rr_per_degC_200km_era5clim_2001_2020_025deg.nc` | JJA risk ratio per °C, 200km smoothing, ERA5 threshold |
| `djf_upper_tail_slopes.nc` | DJF raw log-prob slopes per temperature magnitudes per grid point Upper Tails |
| `djf_rr_per_degC_200km_era5clim_2001_2020_025deg.nc` | DJF risk ratio per °C, 200km smoothing, ERA5 threshold |
| `djf_exceedance_count_95pctl_era5clim_2001_2020_025deg.nc` | DJF exceedance counts per year per grid point |
| `global_summer_rr_per_degC_200km_era5clim_2001_2020_025deg.pdf` | Global summer risk ratio map (ERA5 threshold) |
| `global_summer_clima_rr_per_degC_2001_2020_025deg.nc` | Global summer risk ratio using internal climatology, Upper Tails |
| `global_summer_inter_rr_per_degC_200km_era5clim_2001_2020_025deg.nc` | Global summer risk ratio using ERA5 external climatology |
| `global_summer_rr_diff_internal_vs_era5clim_200km_2001_2020_025deg.nc` | Difference: internal minus ERA5 climatology risk ratio. 
---

## Key Results

- DJF slopes show strongest sensitivity in Southern Hemisphere land regions 
  (southern Africa, Australia, South America)
- ERA5 vs internal climatology difference is small over most land areas, 
  suggesting the reforecast is well-calibrated in the upper tail

---


