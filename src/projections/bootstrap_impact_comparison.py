#!/usr/bin/env python -u
"""
Bootstrap GDP impact comparison across ΔEHD variants and panel specifications.

Runs two panel specs:
  A) admin1level (mixed admin1+admin2, 11K entities, β = −0.055)
  B) admin1pure  (admin1 only, 2.7K entities, β = −0.039)

For each: samples beta 1000 times from N(beta_hat, SE^2), applies to entity-level
ΔEHD (settlement-weighted from gridded data), computes GDP-weighted global impacts.
Tests V2 vs V6 for significant difference.

Usage:
  cd ~/projects/macro/extreme_heat/biodiversity_interactions
  /turbo/mgoldklang/pyenvs/peg_nov_24/bin/python projections/scripts/bootstrap_impact_comparison.py
"""

import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from project_slopes_ehd_maps import (
    load_ifs_slopes,
    load_ehd_baseline_025,
    compute_era5_dt_025,
    compute_gcm_dt_025,
    compute_slope_dehd,
    compute_era5_ehd_trend_025,
    compute_cil_dehd_025,
    load_mixed_raster,
    load_settlement,
    agg_to_entities,
    interp_025_to_01,
    ERA5_LATS,
    ERA5_LONS,
    REGR_FILE,
    PANEL_FILE,
    OUT_DIR,
)

PROJ_DIR = Path(__file__).resolve().parent.parent           # projections/
ROOT_DIR = PROJ_DIR.parent                                  # biodiversity_interactions/
PARENT_DIR = ROOT_DIR.parent                                # extreme_heat/
CACHE_DIR = PARENT_DIR / "cache"

N_DRAWS = 1000
np.random.seed(42)

# ─── Panel specs ─────────────────────────────────────────────────────────────
PANELS = {
    "admin1level": {
        "regr_file": PARENT_DIR / "output" / "panel_regression_gdp_heat_results_admin1level.csv",
        "panel_file": PARENT_DIR / "output" / "panel_mixed_admin1level.parquet",
        "raster_type": "mixed",  # uses load_mixed_raster
    },
    "admin1pure": {
        "regr_file": PARENT_DIR / "output" / "panel_regression_gdp_heat_results_admin1pure.csv",
        "panel_file": PARENT_DIR / "output" / "panel_admin1_pure.parquet",
        "raster_type": "admin1",  # admin1 raster only
    },
}


def load_admin1_raster():
    """Build raster + entity table for admin1-pure panel."""
    panel = pd.read_parquet(
        str(PANELS["admin1pure"]["panel_file"]),
        columns=["GID_2", "GID_nmbr", "iso3"],
    )
    entities = (panel[["GID_2", "GID_nmbr", "iso3"]]
                .drop_duplicates("GID_2")
                .reset_index(drop=True))
    entities["entity_idx"] = np.arange(len(entities))
    n_entities = len(entities)

    # Admin1 raster
    ds = xr.open_dataset(str(CACHE_DIR / "admin1_raster_01deg.nc"))
    admin1_grid = ds["admin1_id"].values.astype(np.int32)
    ds.close()

    lk = pd.read_csv(str(CACHE_DIR / "admin1_lookup.csv"))

    # Map raster_id → entity_idx via GID_nmbr
    merged = lk.merge(
        entities[["GID_nmbr", "entity_idx"]], on="GID_nmbr", how="inner"
    )
    max_rid = int(admin1_grid.max()) + 1
    rid_map = np.full(max_rid, -1, dtype=np.int32)
    for _, row in merged.iterrows():
        rid = int(row["raster_id"])
        if 0 <= rid < max_rid:
            rid_map[rid] = int(row["entity_idx"])

    combined = np.where(
        admin1_grid >= 0,
        rid_map[np.clip(admin1_grid, 0, max_rid - 1)],
        -1,
    ).astype(np.int32)

    return combined, entities, n_entities


def load_beta_and_sample(regr_file, n_draws):
    """Load beta and SE, return (beta_hat, beta_se, sampled_betas)."""
    regr = pd.read_csv(str(regr_file))
    row = regr[
        (regr["spec"] == "linear_ctrl_baseline") &
        (regr["variable"] == "heat_freq_weighted") &
        (regr["se_type"] == "conley_500km")
    ].iloc[0]
    beta_hat = float(row["beta"])
    beta_se = float(row["se"])
    betas = np.random.normal(beta_hat, beta_se, size=n_draws)
    return beta_hat, beta_se, betas


def run_bootstrap_for_panel(panel_name, dehd_grids_01, variant_names, settlement):
    """Run bootstrap for one panel spec. Returns results dict."""
    spec = PANELS[panel_name]
    print(f"\n{'='*65}")
    print(f"Panel: {panel_name}")
    print(f"{'='*65}")

    # Load beta
    beta_hat, beta_se, betas = load_beta_and_sample(spec["regr_file"], N_DRAWS)
    print(f"  beta = {beta_hat:.6f}, SE = {beta_se:.6f}")

    # Load raster + entities
    if spec["raster_type"] == "mixed":
        combined_raster, entities, n_entities = load_mixed_raster()
    else:
        combined_raster, entities, n_entities = load_admin1_raster()
    print(f"  {n_entities:,} entities")

    # GDP weights
    panel = pd.read_parquet(str(spec["panel_file"]))
    gdp_mean = panel.groupby("GID_2")["gdp_per_capita"].mean()
    ent = entities.copy()
    ent["gdp_pc"] = ent["GID_2"].map(gdp_mean).astype(np.float64)

    # Aggregate ΔEHD grids to entities
    dehd_entity = {}
    for name in variant_names:
        grid_01 = dehd_grids_01[name]
        ent_vals = agg_to_entities(grid_01, combined_raster, settlement, n_entities)
        dehd_entity[name] = ent_vals
        n_valid = np.isfinite(ent_vals).sum()
        print(f"  {name}: {n_valid:,} valid, mean ΔEHD={np.nanmean(ent_vals):.4f}")

    # Bootstrap impacts
    results = {}
    for name, ent_dehd in dehd_entity.items():
        gdp_w = ent["gdp_pc"].values.copy()
        valid = np.isfinite(ent_dehd) & np.isfinite(gdp_w) & (gdp_w > 0)
        dehd_v = ent_dehd[valid]
        gdp_v = gdp_w[valid]
        gdp_v_norm = gdp_v / gdp_v.sum()

        gdp_wt_dehd = np.sum(dehd_v * gdp_v_norm)
        impacts = betas * gdp_wt_dehd

        results[name] = {
            "gdp_wt_dehd": gdp_wt_dehd,
            "impacts": impacts,
            "mean": np.mean(impacts),
            "median": np.median(impacts),
            "p5": np.percentile(impacts, 5),
            "p95": np.percentile(impacts, 95),
            "sd": np.std(impacts),
        }

    # Print table
    print(f"\n  {'Variant':<34} {'GDP-wt ΔEHD':>11} {'Mean':>10} "
          f"{'Median':>10} {'P5':>10} {'P95':>10}")
    print(f"  {'-'*85}")
    for name, r in results.items():
        print(f"  {name:<34} {r['gdp_wt_dehd']:>11.4f} {r['mean']:>10.6f} "
              f"{r['median']:>10.6f} {r['p5']:>10.6f} {r['p95']:>10.6f}")

    # V2 vs V6 significance test
    v2_key = "V2: ERA5 Tmax 200km slope"
    v6_key = "V6: CIL GDPCIR"
    print(f"\n  --- V2 vs V6 significance ({panel_name}) ---")

    ent_v2 = dehd_entity[v2_key]
    ent_v6 = dehd_entity[v6_key]
    both_valid = np.isfinite(ent_v2) & np.isfinite(ent_v6)
    d_v2 = ent_v2[both_valid]
    d_v6 = ent_v6[both_valid]
    gdp_w_valid = ent["gdp_pc"].values[both_valid]
    gdp_w_valid = np.where(np.isfinite(gdp_w_valid), gdp_w_valid, 0)
    gdp_w_norm = gdp_w_valid / gdp_w_valid.sum()

    obs_diff = np.sum((d_v2 - d_v6) * gdp_w_norm)

    # Permutation test
    n_perm = 10000
    perm_diffs = np.empty(n_perm)
    pair_diff = d_v2 - d_v6
    for i in range(n_perm):
        signs = np.random.choice([-1, 1], size=len(pair_diff))
        perm_diffs[i] = np.sum(signs * pair_diff * gdp_w_norm)
    p_two_sided = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

    # GDP-weighted t-test
    from scipy import stats
    wt_mean = np.sum(pair_diff * gdp_w_norm)
    wt_var = np.sum(gdp_w_norm * (pair_diff - wt_mean)**2)
    n_eff = 1.0 / np.sum(gdp_w_norm**2)
    wt_se = np.sqrt(wt_var / n_eff)
    wt_t = wt_mean / wt_se
    wt_p = 2 * stats.t.sf(np.abs(wt_t), df=n_eff - 1)

    # Impact difference under sampled betas
    impact_diff_mean = np.mean(betas) * obs_diff
    impact_diff_p5 = np.percentile(betas, 95) * obs_diff  # note sign flip
    impact_diff_p95 = np.percentile(betas, 5) * obs_diff

    print(f"  N entities (paired): {both_valid.sum():,}")
    print(f"  GDP-wt ΔEHD diff (V2 − V6): {obs_diff:.6f}")
    print(f"  Impact diff (V2 − V6): mean={impact_diff_mean:.6f}, "
          f"[P5={impact_diff_p5:.6f}, P95={impact_diff_p95:.6f}]")
    print(f"  Permutation p-value: {p_two_sided:.4f}")
    print(f"  GDP-weighted t-test: t={wt_t:.2f}, p={wt_p:.2e}, N_eff={n_eff:.0f}")
    sig = "***" if wt_p < 0.001 else "**" if wt_p < 0.01 else "*" if wt_p < 0.05 else "n.s."
    print(f"  → {sig}")

    return results, betas


def main():
    wall_start = time.time()

    print("=" * 65)
    print("Bootstrap GDP impact — admin1level + admin1pure")
    print("=" * 65)

    # ── Compute all 6 ΔEHD grids (shared across panels) ─────────────────
    rr_local, rr_200km, ifs_lats, ifs_lons = load_ifs_slopes()
    ehd_base_025, ehd_stack_01 = load_ehd_baseline_025(ifs_lats, ifs_lons)
    dt_era5_local, dt_era5_200km = compute_era5_dt_025(ifs_lats, ifs_lons)
    dt_gcm_local, dt_gcm_200km = compute_gcm_dt_025(ifs_lats, ifs_lons)

    print("\nComputing ΔEHD variants...")
    dehd_v1 = compute_slope_dehd(ehd_base_025, rr_local, dt_era5_local)
    dehd_v2 = compute_slope_dehd(ehd_base_025, rr_200km, dt_era5_200km)
    dehd_v3 = compute_slope_dehd(ehd_base_025, rr_local, dt_gcm_local)
    dehd_v4 = compute_slope_dehd(ehd_base_025, rr_200km, dt_gcm_200km)
    dehd_v5 = compute_era5_ehd_trend_025(ehd_stack_01, ifs_lats, ifs_lons)
    del ehd_stack_01
    dehd_v6 = compute_cil_dehd_025(ifs_lats, ifs_lons)

    variant_names = [
        "V1: ERA5 Tmax local slope",
        "V2: ERA5 Tmax 200km slope",
        "V3: GCM Tmax local slope",
        "V4: GCM Tmax 200km slope",
        "V5: ERA5 EHD trend",
        "V6: CIL GDPCIR",
    ]
    grids_025 = {
        variant_names[0]: dehd_v1, variant_names[1]: dehd_v2,
        variant_names[2]: dehd_v3, variant_names[3]: dehd_v4,
        variant_names[4]: dehd_v5, variant_names[5]: dehd_v6,
    }

    # Pre-interpolate all grids to 0.1° (shared)
    print("\nInterpolating all variants to 0.1°...")
    dehd_grids_01 = {}
    for name, g in grids_025.items():
        dehd_grids_01[name] = interp_025_to_01(g, ifs_lats, ifs_lons)

    settlement = load_settlement()

    # ── Run both panels ──────────────────────────────────────────────────
    all_results = {}
    for panel_name in ["admin1level", "admin1pure"]:
        results, betas = run_bootstrap_for_panel(
            panel_name, dehd_grids_01, variant_names, settlement
        )
        all_results[panel_name] = results

    # ── Cross-panel comparison table ─────────────────────────────────────
    print(f"\n{'='*95}")
    print("Cross-panel summary: GDP growth rate impact (pp)")
    print(f"{'='*95}")
    print(f"  {'Variant':<34} {'admin1level':>30} {'admin1pure':>30}")
    print(f"  {'':<34} {'Mean [P5, P95]':>30} {'Mean [P5, P95]':>30}")
    print(f"  {'-'*94}")
    for name in variant_names:
        parts = []
        for pn in ["admin1level", "admin1pure"]:
            r = all_results[pn][name]
            parts.append(f"{r['mean']:>8.4f} [{r['p5']:>7.4f}, {r['p95']:>7.4f}]")
        print(f"  {name:<34} {parts[0]:>30} {parts[1]:>30}")

    elapsed = time.time() - wall_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
