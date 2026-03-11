"""
Central path configuration for the extreme-temperature panel study.

All scripts import from this module.  Override any path by setting the
corresponding environment variable before running a script, e.g.:

    ERA5_LAND_DIR=/my/tmax/data python src/climate/compute_ehd_global.py

Default values assume a self-contained repo layout:

    data/
      raw/
        climate/        ← ERA5-Land daily Tmax NetCDFs (era5_tmax_YYYY_MM.nc)
        gdp/            ← GDP growth parquet + GADM GeoPackage
        shapes/         ← GHS settlement TIF rasters
      processed/
        climatologies/  ← 95th-percentile thresholds (output of compute_ehd_*)
        panel/          ← aggregated heat + controls + regression results
"""

import os
from pathlib import Path

# ── Repo root (one level above this file) ────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Raw climate data ──────────────────────────────────────────────────────────
# ERA5-Land daily Tmax at 0.1° resolution.
# Files named: era5_tmax_{YYYY}_{MM:02d}.nc
# Variable: t2m (Kelvin)
# Download: https://cds.climate.copernicus.eu/  (dataset: reanalysis-era5-land)
ERA5_LAND_DIR = Path(
    os.environ.get("ERA5_LAND_DIR",
                   str(REPO_ROOT / "data" / "raw" / "climate" / "era5_land_tmax"))
)

# ERA5 (0.25°) monthly GRIBs — used for annual T and P controls.
# Sub-directories: 2m_temperature/{YYYY}/era5_t2m_{YYYY}{MM:02d}.grib
#                  total_precipitation/{YYYY}/era5_tp_{YYYY}{MM:02d}.grib
# Download: https://cds.climate.copernicus.eu/  (dataset: reanalysis-era5-single-levels)
ERA5_DIR = Path(
    os.environ.get("ERA5_DIR",
                   str(REPO_ROOT / "data" / "raw" / "climate" / "era5_025deg"))
)

# ── Raw spatial / GDP data ────────────────────────────────────────────────────
# GADM admin2 GeoPackage with GDP per capita 1990-2022.
# Source: Global Data Lab (https://globaldatalab.org/) area-level income data
#         merged with GADM v4.1 polygons (https://gadm.org/).
# File:   polyg_adm2_gdp_perCapita_1990_2022.gpkg
ADMIN2_GPKG = Path(
    os.environ.get("ADMIN2_GPKG",
                   str(REPO_ROOT / "data" / "raw" / "gdp"
                       / "polyg_adm2_gdp_perCapita_1990_2022.gpkg"))
)

# Admin2 GDP growth parquet (long format: GID_2, year, admin2_growth, residual).
# Derived from the GeoPackage above: log-difference of GDP per capita,
# residualized against country-year means.
# See data/raw/gdp/README.md for construction details.
GDP_GROWTH_PATH = Path(
    os.environ.get("GDP_GROWTH_PATH",
                   str(REPO_ROOT / "data" / "raw" / "gdp"
                       / "adm2_gdp_growth_residuals_long.parquet"))
)

# GHS Built-up Surface rasters (Mollweide 100m, 6 epochs: 2000-2025).
# Source: https://ghsl.jrc.ec.europa.eu/ghs_buS2023.php
#   GHS_BUILT_S_E{epoch}_GLOBE_R2023A_54009_100_V1_0.tif
SETTLEMENT_DIR = Path(
    os.environ.get("SETTLEMENT_DIR",
                   str(REPO_ROOT / "data" / "raw" / "shapes" / "ghs_settlement"))
)

# ── Processed / intermediate outputs ─────────────────────────────────────────
PROCESSED_DIR        = REPO_ROOT / "data" / "processed"
CLIMATOLOGY_DIR      = PROCESSED_DIR / "climatologies"
PANEL_DIR            = PROCESSED_DIR / "panel"
EHD_GLOBAL_DIR       = PROCESSED_DIR / "exceedance_frequency"   # per-year global grids

# EHD panel — output of aggregate_heat_to_admin2.py
HEAT_PANEL_PATH      = PANEL_DIR / "admin2_heat_settlement_weighted.parquet"

# Annual climate controls — output of aggregate_era5_controls_to_admin2.py
CONTROLS_PATH        = PANEL_DIR / "admin2_era5land_annual_controls.parquet"

# Regression results — output of panel_regression_gdp_heat.py
RESULTS_CSV          = PANEL_DIR / "panel_regression_gdp_heat_results.csv"

# Cache directory (rasterised admin2 grids, reprojected settlement rasters)
CACHE_DIR            = PROCESSED_DIR / "cache"
