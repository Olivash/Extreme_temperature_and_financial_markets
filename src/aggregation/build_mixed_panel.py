#!/usr/bin/env python -u
"""
Build mixed admin2/admin1 panel.

Rule: for each admin1 unit globally —
  - If ALL its admin2 are raster-covered (has_missing=False) → keep admin2 entities
  - If ANY admin2 is centroid-sampled (has_missing=True)    → collapse the ENTIRE
    admin1 to one entity; use admin1 GDP and admin1 climate

No double-counting: each original admin2 appears exactly once.

Inputs (configure via src/config.py or environment variables):
  HEAT_PANEL_PATH      — admin2_heat_settlement_weighted.parquet
  CONTROLS_PATH        — admin2_era5land_annual_controls.parquet
  GDP_GROWTH_PATH      — adm2_gdp_growth_residuals_long.parquet
  ADMIN2_GPKG          — admin2 GeoPackage (for centroids)
  HEAT_ADM1_PANEL_PATH — admin1_heat_settlement_weighted.parquet
  CONTROLS_ADM1_PATH   — admin1_era5land_annual_controls.parquet
  GDP_ADM1_PATH        — admin1_gdp_growth_long.parquet
  CROSSWALK_PATH       — admin2_admin1_crosswalk.parquet
  ADMIN1_GPKG          — admin1 GeoPackage (for centroids)

Output:
  <PANEL_DIR>/panel_mixed_admin1level.parquet
    Columns: GID_2, iso3, year_heat, year_gdp, heat_freq_weighted,
             admin2_growth, annual_temp_c, annual_precip_mm, gdp_per_capita,
             income_tercile, centroid_lat, centroid_lon, is_admin1
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ADMIN1_GPKG, ADMIN2_GPKG,
    HEAT_PANEL_PATH, CONTROLS_PATH, GDP_GROWTH_PATH,
    HEAT_ADM1_PANEL_PATH, CONTROLS_ADM1_PATH, GDP_ADM1_PATH,
    CROSSWALK_PATH, PANEL_DIR, MIXED_PANEL_PATH,
)


def main():
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Step 1: Load crosswalk ────────────────────────────────────────────
    print("Loading crosswalk...")
    xwalk = pd.read_parquet(str(CROSSWALK_PATH),
                            columns=["GID_2", "iso3", "GID_nmbr", "has_missing"])
    adm2_to_collapse = xwalk.set_index("GID_2")["has_missing"].to_dict()
    adm2_to_adm1     = xwalk.set_index("GID_2")["GID_nmbr"].to_dict()

    n_collapse = sum(bool(v) for v in adm2_to_collapse.values() if v is not None)
    n_keep     = sum(not bool(v) for v in adm2_to_collapse.values() if v is not None)
    print(f"  Admin2 to KEEP as individual entities  : {n_keep:,}")
    print(f"  Admin2 to COLLAPSE into admin1         : {n_collapse:,}")

    # ── Step 2: Admin2 portion (fully-covered admin1s) ────────────────────
    print("\nBuilding admin2 portion...")
    heat2 = pd.read_parquet(str(HEAT_PANEL_PATH))
    keep_gids     = {gid for gid, flag in adm2_to_collapse.items() if not flag}
    heat2_kept    = heat2[heat2["GID_2"].isin(keep_gids)].copy()
    heat2_kept    = heat2_kept.rename(columns={"year": "year_heat"})
    heat2_kept["year_gdp"] = heat2_kept["year_heat"]
    del heat2

    gdp2       = pd.read_parquet(str(GDP_GROWTH_PATH))
    gdp2       = gdp2.rename(columns={"year": "year_gdp"})
    adm2_panel = heat2_kept.merge(
        gdp2[["GID_2", "year_gdp", "admin2_growth", "residual"]],
        on=["GID_2", "year_gdp"], how="inner"
    )
    adm2_panel = adm2_panel.dropna(subset=["admin2_growth", "heat_freq_weighted"])

    if CONTROLS_PATH.exists():
        ctrl2 = pd.read_parquet(str(CONTROLS_PATH))
        ctrl2 = ctrl2.rename(columns={"year": "year_heat"})
        adm2_panel = adm2_panel.merge(
            ctrl2[["GID_2", "year_heat", "annual_temp_c", "annual_precip_mm"]],
            on=["GID_2", "year_heat"], how="left"
        )

    print("  Loading admin2 GPKG for centroids...")
    ignore_cols = [str(y) for y in range(1990, 2023)] + ["slope", "id", "adm2ID"]
    gdf2 = gpd.read_file(str(ADMIN2_GPKG), ignore_fields=ignore_cols)

    def safe_rp(geom):
        try:
            if geom is None or geom.is_empty: return (np.nan, np.nan)
            rp = geom.representative_point()
            return (np.nan, np.nan) if rp.is_empty else (rp.y, rp.x)
        except Exception:
            return (np.nan, np.nan)

    coords = [safe_rp(g) for g in gdf2.geometry]
    gdf2["centroid_lat"] = [c[0] for c in coords]
    gdf2["centroid_lon"] = [c[1] for c in coords]
    adm2_panel = adm2_panel.merge(
        gdf2[["GID_2", "centroid_lat", "centroid_lon"]], on="GID_2", how="left"
    )
    adm2_panel["is_admin1"]     = False
    adm2_panel["gdp_per_capita"] = adm2_panel.get("gdp_per_capita", np.nan)
    adm2_panel["residual"]       = adm2_panel.get("residual", np.nan)
    print(f"  Admin2 panel: {len(adm2_panel):,} rows, "
          f"{adm2_panel['GID_2'].nunique():,} entities")

    # ── Step 3: Admin1 portion (admin1s with any missing admin2) ─────────
    print("\nBuilding admin1 portion...")
    collapse_adm1_ids = set(
        xwalk.loc[xwalk["has_missing"].fillna(False), "GID_nmbr"].dropna().unique()
    )
    print(f"  Admin1 units to include: {len(collapse_adm1_ids):,}")

    heat1        = pd.read_parquet(str(HEAT_ADM1_PANEL_PATH))
    heat1_needed = heat1[heat1["GID_nmbr"].isin(collapse_adm1_ids)].copy()
    heat1_needed = heat1_needed.rename(columns={"year": "year_heat"})
    heat1_needed["year_gdp"] = heat1_needed["year_heat"]
    del heat1

    gdp1       = pd.read_parquet(str(GDP_ADM1_PATH))
    gdp1       = gdp1.rename(columns={"year": "year_gdp"})
    adm1_panel = heat1_needed.merge(
        gdp1[["GID_nmbr", "year_gdp", "gdp_pc", "admin1_growth"]],
        on=["GID_nmbr", "year_gdp"], how="inner"
    )
    adm1_panel = adm1_panel.dropna(subset=["admin1_growth", "heat_freq_weighted"])
    adm1_panel = adm1_panel.rename(columns={"admin1_growth": "admin2_growth",
                                             "gdp_pc":        "gdp_per_capita"})

    if CONTROLS_ADM1_PATH.exists():
        ctrl1 = pd.read_parquet(str(CONTROLS_ADM1_PATH))
        ctrl1 = ctrl1.rename(columns={"year": "year_heat"})
        adm1_panel = adm1_panel.merge(
            ctrl1[["GID_nmbr", "year_heat", "annual_temp_c", "annual_precip_mm"]],
            on=["GID_nmbr", "year_heat"], how="left"
        )

    print("  Loading admin1 GPKG for centroids...")
    gdf1 = gpd.read_file(str(ADMIN1_GPKG))
    id_col = "GID_nmbr" if "GID_nmbr" in gdf1.columns else gdf1.columns[0]
    gdf1["GID_nmbr"]     = pd.to_numeric(gdf1[id_col], errors="coerce")
    gdf1["centroid_lat"] = gdf1.geometry.centroid.y
    gdf1["centroid_lon"] = gdf1.geometry.centroid.x
    adm1_panel = adm1_panel.merge(
        gdf1[["GID_nmbr", "centroid_lat", "centroid_lon"]], on="GID_nmbr", how="left"
    )
    adm1_panel["GID_2"]      = "ADM1_" + adm1_panel["GID_nmbr"].astype(str)
    adm1_panel["is_admin1"]  = True
    adm1_panel["residual"]   = np.nan
    print(f"  Admin1 panel: {len(adm1_panel):,} rows, "
          f"{adm1_panel['GID_2'].nunique():,} entities")

    # ── Step 4: Stack and save ────────────────────────────────────────────
    print("\nStacking panels...")
    shared_cols = [
        "GID_2", "iso3", "year_heat", "year_gdp",
        "heat_freq_weighted", "admin2_growth", "residual",
        "annual_temp_c", "annual_precip_mm",
        "gdp_per_capita", "centroid_lat", "centroid_lon", "is_admin1",
    ]
    mixed = pd.concat([
        adm2_panel[[c for c in shared_cols if c in adm2_panel.columns]],
        adm1_panel[[c for c in shared_cols if c in adm1_panel.columns]],
    ], ignore_index=True)

    # Income tercile
    median_gdp = mixed.groupby("GID_2")["gdp_per_capita"].median()
    valid_med  = median_gdp.dropna()
    if len(valid_med) > 0:
        try:
            tercile_labels      = pd.qcut(valid_med, 3, labels=["low", "mid", "high"])
            mixed["income_tercile"] = mixed["GID_2"].map(tercile_labels)
        except Exception as e:
            print(f"  Income tercile warning: {e}")
            mixed["income_tercile"] = np.nan

    mixed = mixed.sort_values(["iso3", "GID_2", "year_heat"]).reset_index(drop=True)
    mixed.to_parquet(str(MIXED_PANEL_PATH), index=False)

    n_adm1 = mixed["is_admin1"].sum()
    n_adm2 = (~mixed["is_admin1"]).sum()
    print(f"\n  Admin2 rows : {n_adm2:,}")
    print(f"  Admin1 rows : {n_adm1:,}")
    print(f"  Total rows  : {len(mixed):,}")
    print(f"  Entities    : {mixed['GID_2'].nunique():,}")
    print(f"  Countries   : {mixed['iso3'].nunique()}")
    print(f"\nSaved: {MIXED_PANEL_PATH}")
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
