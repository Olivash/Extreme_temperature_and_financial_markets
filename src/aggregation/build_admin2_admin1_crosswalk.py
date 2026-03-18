#!/usr/bin/env python -u
"""
Build admin2 → admin1 crosswalk.

For each admin2 region, assigns the parent admin1 (GID_nmbr) via spatial join
of admin2 representative points into admin1 polygons. Also flags which admin1s
contain at least one centroid-sampled (too-small) admin2.

Inputs (configure via src/config.py or environment variables):
  ADMIN1_GPKG      — GADM admin1 GeoPackage with GDP per capita 1990-2022
  HEAT_PANEL_PATH  — admin2_heat_settlement_weighted.parquet
                     (must contain centroid_sampled column, output of
                      aggregate_heat_to_admin2.py)
  ADMIN2_LOOKUP_CSV — admin2_lookup.csv with centroid_lat, centroid_lon columns
                      (also output of aggregate_heat_to_admin2.py)

Output:
  <PANEL_DIR>/admin2_admin1_crosswalk.parquet
    Columns: GID_2, iso3, GID_nmbr, adm1_name, n_admin2_total,
             n_admin2_missing, has_missing
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point  # noqa: F401

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (ADMIN1_GPKG, HEAT_PANEL_PATH, PANEL_DIR, CACHE_DIR,
                    ADMIN2_LOOKUP_CSV, CROSSWALK_PATH)


def main():
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Step 1: Load admin2 lookup (centroids) ────────────────────────────
    if not ADMIN2_LOOKUP_CSV.exists():
        raise FileNotFoundError(
            f"admin2_lookup.csv not found at {ADMIN2_LOOKUP_CSV}.\n"
            "Run src/aggregation/aggregate_heat_to_admin2.py first."
        )
    print(f"Loading admin2 lookup from {ADMIN2_LOOKUP_CSV.name} ...")
    lookup = pd.read_csv(str(ADMIN2_LOOKUP_CSV))
    print(f"  {len(lookup):,} admin2 regions")

    # ── Step 2: Load heat parquet → per-GID_2 centroid_sampled flag ───────
    print(f"\nLoading heat parquet to get centroid_sampled flags...")
    heat = pd.read_parquet(str(HEAT_PANEL_PATH), columns=["GID_2", "centroid_sampled"])
    missing_gids = set(heat.loc[heat["centroid_sampled"], "GID_2"].unique())
    print(f"  Admin2 regions with any centroid-sampled year: {len(missing_gids):,}")
    del heat

    # ── Step 3: Load admin1 polygons ──────────────────────────────────────
    print(f"\nLoading admin1 GPKG...")
    gdf1 = gpd.read_file(str(ADMIN1_GPKG))
    print(f"  {len(gdf1):,} admin1 regions")

    id_col   = "GID_nmbr" if "GID_nmbr" in gdf1.columns else gdf1.columns[0]
    name_col = "Subnat"   if "Subnat"   in gdf1.columns else None
    if name_col is None:
        for c in gdf1.columns:
            if c.lower() in ("subnat", "name_1", "adm1name"):
                name_col = c; break

    gdf1 = (gdf1[[id_col, name_col, "geometry"]].copy() if name_col
            else gdf1[[id_col, "geometry"]].copy())
    rename = {id_col: "GID_nmbr"}
    if name_col:
        rename[name_col] = "adm1_name"
    gdf1 = gdf1.rename(columns=rename)
    gdf1["GID_nmbr"] = pd.to_numeric(gdf1["GID_nmbr"], errors="coerce")

    # ── Step 4: Build admin2 centroid GeoDataFrame ────────────────────────
    print(f"\nBuilding admin2 centroid points for spatial join...")
    lookup2 = lookup.dropna(subset=["centroid_lat", "centroid_lon"]).copy()
    geometry = [Point(lon, lat)
                for lat, lon in zip(lookup2["centroid_lat"], lookup2["centroid_lon"])]
    gdf2 = gpd.GeoDataFrame(
        lookup2[["GID_2", "iso3"]].copy(),
        geometry=geometry,
        crs="EPSG:4326",
    )

    if gdf1.crs is None:
        gdf1 = gdf1.set_crs("EPSG:4326")
    elif gdf1.crs.to_epsg() != 4326:
        gdf1 = gdf1.to_crs("EPSG:4326")

    # ── Step 5: Spatial join ──────────────────────────────────────────────
    print(f"\nSpatial join: admin2 centroids into admin1 polygons...")
    t1 = time.time()
    joined = gpd.sjoin(
        gdf2,
        gdf1[["GID_nmbr", "adm1_name", "geometry"]] if "adm1_name" in gdf1.columns
        else gdf1[["GID_nmbr", "geometry"]],
        how="left",
        op="within",
    )
    joined = joined.drop(columns=["geometry", "index_right"])
    print(f"  Join done in {time.time()-t1:.0f}s")

    n_matched = joined["GID_nmbr"].notna().sum()
    n_total   = len(joined)
    print(f"  Matched: {n_matched:,} / {n_total:,} ({100*n_matched/n_total:.1f}%)")

    # Unmatched → nearest-polygon fallback
    unmatched = joined[joined["GID_nmbr"].isna()].copy()
    if len(unmatched) > 0:
        print(f"  {len(unmatched):,} unmatched — nearest-polygon fallback...")
        gdf1_valid = gdf1[gdf1.geometry.notna()].copy()
        try:
            if hasattr(gpd, "sjoin_nearest"):
                near = gpd.sjoin_nearest(
                    unmatched[["GID_2", "iso3", "geometry"]],
                    gdf1_valid[["GID_nmbr"] + (["adm1_name"] if "adm1_name" in gdf1_valid.columns else []) + ["geometry"]],
                    how="left",
                ).drop(columns=["geometry", "index_right"])
            else:
                adm1_cx = gdf1_valid.geometry.centroid.x.values
                adm1_cy = gdf1_valid.geometry.centroid.y.values
                near_rows = []
                for _, row in unmatched.iterrows():
                    dists = (adm1_cx - row.geometry.x)**2 + (adm1_cy - row.geometry.y)**2
                    best  = np.argmin(dists)
                    r = {"GID_2": row["GID_2"], "GID_nmbr": gdf1_valid.iloc[best]["GID_nmbr"]}
                    if "adm1_name" in gdf1_valid.columns:
                        r["adm1_name"] = gdf1_valid.iloc[best]["adm1_name"]
                    near_rows.append(r)
                near = pd.DataFrame(near_rows)
            joined = joined.set_index("GID_2")
            near   = near.set_index("GID_2")
            for gid in near.index:
                if pd.isna(joined.loc[gid, "GID_nmbr"]):
                    joined.loc[gid, "GID_nmbr"] = near.loc[gid, "GID_nmbr"]
                    if "adm1_name" in near.columns:
                        joined.loc[gid, "adm1_name"] = near.loc[gid, "adm1_name"]
            joined = joined.reset_index()
            print(f"  After fallback: {joined['GID_nmbr'].notna().sum():,} / {n_total:,} matched")
        except Exception as e:
            print(f"  Nearest fallback failed: {e}")

    # ── Step 6: Flag missing admin2 per admin1 ────────────────────────────
    joined["is_missing"] = joined["GID_2"].isin(missing_gids)

    admin1_stats = (
        joined.groupby("GID_nmbr")
        .agg(
            n_admin2_total=("GID_2", "count"),
            n_admin2_missing=("is_missing", "sum"),
        )
        .reset_index()
    )
    admin1_stats["has_missing"] = admin1_stats["n_admin2_missing"] > 0

    crosswalk = joined.merge(admin1_stats, on="GID_nmbr", how="left")
    crosswalk = crosswalk.drop(columns=["is_missing"])

    n_with_missing = admin1_stats["has_missing"].sum()
    print(f"\n  Admin1 units with ≥1 missing admin2: "
          f"{n_with_missing:,} / {len(admin1_stats):,} "
          f"({100*n_with_missing/len(admin1_stats):.1f}%)")

    col_order = ["GID_2", "iso3", "GID_nmbr", "adm1_name",
                 "n_admin2_total", "n_admin2_missing", "has_missing"]
    crosswalk = crosswalk[[c for c in col_order if c in crosswalk.columns]]
    crosswalk = crosswalk.sort_values(["iso3", "GID_2"]).reset_index(drop=True)

    crosswalk.to_parquet(str(CROSSWALK_PATH), index=False)
    print(f"\nSaved: {CROSSWALK_PATH}  ({len(crosswalk):,} rows)")
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
