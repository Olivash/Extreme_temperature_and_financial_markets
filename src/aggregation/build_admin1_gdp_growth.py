#!/usr/bin/env python -u
"""
Build admin1 GDP growth panel from the admin1 GDP per capita GeoPackage.

Loads the admin1 GPKG (≈2,855 regions, year columns 1990-2022), melts to long
format, then computes log-difference GDP per capita growth year-over-year.

Input (set ADMIN1_GPKG in src/config.py or via env var):
  polyg_adm1_gdp_perCapita_1990_2022.gpkg
  Source: Global Data Lab (https://globaldatalab.org/) area-level income data
          merged with GADM v4.1 polygons (https://gadm.org/).

Output:
  <PANEL_DIR>/admin1_gdp_growth_long.parquet
    Columns: GID_nmbr, iso3, Subnat, year, gdp_pc, admin1_growth
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
from config import ADMIN1_GPKG, GDP_ADM1_PATH, PANEL_DIR


def main():
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading admin1 GPKG...")
    gdf = gpd.read_file(str(ADMIN1_GPKG))
    print(f"  {len(gdf):,} admin1 regions loaded  ({time.time()-t0:.0f}s)")
    print(f"  Columns: {list(gdf.columns[:10])} ...")

    # Identify year columns (string digits between 1990 and 2022)
    year_cols = [c for c in gdf.columns if c.isdigit() and 1990 <= int(c) <= 2022]
    year_cols.sort()
    print(f"  Year columns: {year_cols[0]}–{year_cols[-1]} ({len(year_cols)} years)")

    # Identify key columns — handle possible naming variations
    id_col   = "GID_nmbr" if "GID_nmbr" in gdf.columns else gdf.columns[0]
    iso_col  = "iso3"     if "iso3"     in gdf.columns else None
    name_col = "Subnat"   if "Subnat"   in gdf.columns else None

    if iso_col is None:
        for c in gdf.columns:
            if c.lower() in ("iso3", "iso_a3", "country_iso"):
                iso_col = c; break
    if name_col is None:
        for c in gdf.columns:
            if c.lower() in ("subnat", "name_1", "adm1name", "region"):
                name_col = c; break

    print(f"  ID col: {id_col}, ISO col: {iso_col}, Name col: {name_col}")

    # Keep useful columns + year cols
    keep = [c for c in [id_col, iso_col, name_col] if c] + year_cols
    df = gdf[keep].drop(columns=["geometry"], errors="ignore").copy()
    rename = {id_col: "GID_nmbr"}
    if iso_col:  rename[iso_col]  = "iso3"
    if name_col: rename[name_col] = "Subnat"
    df = df.rename(columns=rename)
    df["GID_nmbr"] = pd.to_numeric(df["GID_nmbr"], errors="coerce")
    df = df.dropna(subset=["GID_nmbr"])

    # Melt to long format
    print("Melting to long format...")
    long = df.melt(
        id_vars=[c for c in ["GID_nmbr", "iso3", "Subnat"] if c in df.columns],
        value_vars=year_cols,
        var_name="year",
        value_name="gdp_pc",
    )
    long["year"]   = long["year"].astype(int)
    long["gdp_pc"] = pd.to_numeric(long["gdp_pc"], errors="coerce")
    long = long.sort_values(["GID_nmbr", "year"]).reset_index(drop=True)

    # Log-difference growth
    print("Computing log-difference growth rates...")
    long["log_gdp"] = np.log(long["gdp_pc"].replace(0, np.nan))
    long["admin1_growth"] = (
        long.groupby("GID_nmbr")["log_gdp"]
        .transform(lambda s: s.diff())
    )

    long = long.drop(columns=["log_gdp"])
    long = long[long["year"] >= 1991]  # first difference loses one year

    n_valid = long["admin1_growth"].notna().sum()
    print(f"  {len(long):,} rows, {long['GID_nmbr'].nunique():,} regions, "
          f"{n_valid:,} valid growth obs")

    col_order = [c for c in ["GID_nmbr", "iso3", "Subnat", "year", "gdp_pc", "admin1_growth"]
                 if c in long.columns]
    long = long[col_order]
    long.to_parquet(str(GDP_ADM1_PATH), index=False)
    print(f"Saved: {GDP_ADM1_PATH}  ({len(long):,} rows)")
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
