"""
Panel regression: Extreme heat → GDP growth (mixed admin1/admin2 panel).

Uses panel_mixed_admin1level.parquet where any admin1 containing at least one
centroid-sampled admin2 is collapsed to a single admin1 entity; all other
admin2s remain as individual entities.

Identical spec to panel_regression_gdp_heat.py; differences:
  - Reads pre-merged panel_mixed_admin1level.parquet
  - Centroid lat/lon read from panel columns (not GPKG)
  - Lead spec skipped (pre-merged panel; no heat-only parquet)

Output: <PANEL_DIR>/panel_regression_gdp_heat_results_mixed.csv
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse
from linearmodels.panel import PanelOLS
from linearmodels.panel.data import PanelData
import statsmodels.api as sm

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MIXED_PANEL_PATH, PANEL_DIR, RESULTS_MIXED_CSV

MIN_OBS       = 1000
MIN_ENTITIES  = 50
CONLEY_CUTOFFS_KM = [250, 500, 1000]
EARTH_RADIUS_KM   = 6371.0
CONTROL_COLS = ["annual_temp_c", "annual_precip_mm"]


# ── 2. Data Loading ──────────────────────────────────────────────────────────

def load_and_merge_panel():
    """Load the pre-merged mixed panel. Already contains all needed columns."""
    print(f"Loading mixed panel from {MIXED_PANEL_PATH.name}...")
    panel = pd.read_parquet(str(MIXED_PANEL_PATH))

    # Standardise column names expected by downstream code
    if "year_heat" not in panel.columns and "year" in panel.columns:
        panel = panel.rename(columns={"year": "year_heat"})
    panel = panel.dropna(subset=["admin2_growth", "heat_freq_weighted"])

    # Static income tercile from entity-median GDP per capita
    if "income_tercile" not in panel.columns:
        median_gdp     = panel.groupby("GID_2")["gdp_per_capita"].median()
        valid_med      = median_gdp.dropna()
        if len(valid_med) > 0:
            tercile_labels = pd.qcut(valid_med, 3, labels=["low", "mid", "high"])
            panel["income_tercile"] = panel["GID_2"].map(tercile_labels)
        else:
            panel["income_tercile"] = np.nan

    if "annual_temp_c_sq" not in panel.columns and "annual_temp_c" in panel.columns:
        panel["annual_temp_c_sq"] = panel["annual_temp_c"] ** 2

    # Drop rows missing controls
    ctrl_present = [c for c in CONTROL_COLS if c in panel.columns]
    if ctrl_present:
        panel = panel.dropna(subset=ctrl_present)

    n_obs      = len(panel)
    n_ent      = panel["GID_2"].nunique()
    n_years    = panel["year_heat"].nunique()
    n_countries = panel["iso3"].nunique()
    n_adm1     = panel.get("is_admin1", pd.Series([False]*n_obs)).sum()
    print(f"Panel: {n_obs:,} obs, {n_ent:,} entities, {n_countries} countries, "
          f"{n_years} years  (admin1 rows: {n_adm1:,})")
    assert n_obs >= MIN_OBS
    assert n_ent >= MIN_ENTITIES
    return panel


# ── 3. Centroid Extraction ───────────────────────────────────────────────────

def load_centroids(panel):
    """Extract centroids from panel lat/lon columns (pre-computed in panel build)."""
    centroids = (
        panel[["GID_2", "centroid_lat", "centroid_lon"]]
        .drop_duplicates("GID_2")
        .rename(columns={"centroid_lat": "lat", "centroid_lon": "lon"})
        .dropna(subset=["lat", "lon"])
    )
    print(f"Centroids: {len(centroids):,} entities with valid lat/lon")
    return centroids


# ── 4. Country-Specific Trend Removal (FWL) ──────────────────────────────────

def detrend_country(panel, cols, year_col="year_heat"):
    result    = panel.copy().reset_index(drop=True)
    base_year = result[year_col].min()
    for col in cols:
        resid = np.full(len(result), np.nan)
        for _, grp in result.groupby("iso3"):
            idx   = grp.index
            y     = grp[col].values.astype(float)
            t     = (grp[year_col].values - base_year).astype(float)
            valid = np.isfinite(y)
            if valid.sum() <= 3:
                resid[idx] = y; continue
            X_v  = np.column_stack([np.ones(valid.sum()), t[valid], t[valid] ** 2])
            beta, _, _, _ = np.linalg.lstsq(X_v, y[valid], rcond=None)
            X_all = np.column_stack([np.ones(len(t)), t, t ** 2])
            resid[idx] = y - X_all @ beta
        result[col] = resid
    return result


# ── 5. Two-Way Demeaning ─────────────────────────────────────────────────────

def twoway_demean(panel, ycol, xcols):
    cols = [ycol] + list(xcols)
    sub  = panel.set_index(["GID_2", "year_heat"])[cols].copy()
    pdata   = PanelData(sub)
    demeaned = pdata.demean("both")
    y_dm = demeaned.values2d[:, 0]
    X_dm = demeaned.values2d[:, 1:]
    index = demeaned.index
    return pd.Series(y_dm, index=index, name=ycol), \
           pd.DataFrame(X_dm, index=index, columns=xcols)


# ── 6. Conley Spatial HAC ────────────────────────────────────────────────────

def _latlon_to_xyz(lat, lon):
    lr, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(lr)*np.cos(lo), np.cos(lr)*np.sin(lo), np.sin(lr)])

def _chord_distance_km(km):
    return 2.0 * np.sin((km / EARTH_RADIUS_KM) / 2.0)

def compute_conley_se(panel, ycol, xcols, centroids_df, cutoff_km=500, country_trends=True):
    xcols = list(xcols)
    if country_trends and "iso3" in panel.columns:
        panel = detrend_country(panel, [ycol] + xcols)
    y_dm, X_dm = twoway_demean(panel, ycol, xcols)

    common_idx = y_dm.index.intersection(X_dm.index)
    y_dm = y_dm.loc[common_idx].values
    X_dm = X_dm.loc[common_idx].values
    gids = common_idx.get_level_values(0)

    beta, _, _, _ = np.linalg.lstsq(X_dm, y_dm, rcond=None)
    XtX     = X_dm.T @ X_dm
    XtX_inv = np.linalg.pinv(XtX)
    resid   = y_dm - X_dm @ beta

    tss = np.sum(y_dm ** 2)
    rss = np.sum(resid ** 2)
    r2_within = 1.0 - rss / tss if tss > 0 else np.nan

    K  = len(xcols)
    N  = gids.nunique() if hasattr(gids, "nunique") else len(set(gids))
    n_obs = len(y_dm)

    unique_gids = np.array(sorted(set(gids)))
    gid_to_idx  = {g: i for i, g in enumerate(unique_gids)}
    entity_idx  = np.array([gid_to_idx[g] for g in gids])

    scores = np.zeros((len(unique_gids), K))
    xe     = X_dm * resid[:, None]
    np.add.at(scores, entity_idx, xe)

    cent         = centroids_df.set_index("GID_2")
    matched_lats = np.array([cent.loc[g, "lat"] if g in cent.index else np.nan for g in unique_gids])
    matched_lons = np.array([cent.loc[g, "lon"] if g in cent.index else np.nan for g in unique_gids])
    valid        = ~(np.isnan(matched_lats) | np.isnan(matched_lons))
    if valid.sum() < N * 0.9:
        print(f"  Warning: only {valid.sum()}/{N} entities have centroids")

    scores_v = scores[valid]
    lats_v   = matched_lats[valid]
    lons_v   = matched_lons[valid]
    N_v      = len(scores_v)

    xyz          = _latlon_to_xyz(lats_v, lons_v)
    tree         = cKDTree(xyz)
    chord_cutoff = _chord_distance_km(cutoff_km)
    pairs        = tree.query_pairs(chord_cutoff, output_type="ndarray")

    if len(pairs) > 0:
        d      = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
        gc_km  = 2.0 * EARTH_RADIUS_KM * np.arcsin(d / 2.0)
        weights = np.clip(1.0 - gc_km / cutoff_km, 0, 1)
        row  = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(N_v)])
        col  = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(N_v)])
        data = np.concatenate([weights, weights, np.ones(N_v)])
        W    = sparse.csr_matrix((data, (row, col)), shape=(N_v, N_v))
    else:
        W = sparse.eye(N_v, format="csr")

    WS   = W @ scores_v
    meat = scores_v.T @ WS
    V    = XtX_inv @ meat @ XtX_inv * (N_v / (N_v - 1.0))

    diag_V    = np.where(np.diag(V) > 0, np.diag(V), np.nan)
    conley_se = np.sqrt(diag_V)
    tstat     = beta / conley_se
    from scipy.stats import t as tdist
    pval = np.where(np.isfinite(tstat),
                    2.0 * tdist.sf(np.abs(tstat), df=n_obs - N - K), np.nan)
    return {"beta": beta, "se": conley_se, "tstat": tstat, "pval": pval,
            "r2_within": r2_within, "n_obs": n_obs, "n_entities": N,
            "n_pairs": len(pairs) if len(pairs) > 0 else 0}


# ── 7. PanelOLS ──────────────────────────────────────────────────────────────

def _drop_rank_deficient(X_df):
    from scipy.linalg import qr as scipy_qr
    vals = X_df.values
    if vals.shape[1] <= 1: return X_df, []
    _, R, piv = scipy_qr(vals, pivoting=True, mode="economic")
    tol  = max(vals.shape) * abs(R[0, 0]) * np.finfo(float).eps * 100
    rank = np.sum(np.abs(np.diag(R)) > tol)
    if rank >= vals.shape[1]: return X_df, []
    keep    = sorted(piv[:rank])
    dropped = [X_df.columns[i] for i in range(len(X_df.columns)) if i not in keep]
    return X_df.iloc[:, keep], dropped

def run_panelols(panel, ycol, xcols, cov_type="clustered", cluster_entity=True,
                 cluster_time=False, country_trends=True):
    if country_trends and "iso3" in panel.columns:
        panel = detrend_country(panel, [ycol] + list(xcols))
    sub = panel.set_index(["GID_2", "year_heat"])[[ycol] + list(xcols)].dropna()
    y = sub[ycol]; X = sub[xcols]
    X, dropped = _drop_rank_deficient(X)
    if dropped: print(f"  Dropped: {dropped}")
    model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    if cov_type == "clustered":
        res = model.fit(cov_type="clustered", cluster_entity=cluster_entity,
                        cluster_time=cluster_time)
    elif cov_type == "kernel":
        res = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=5)
    else:
        res = model.fit(cov_type=cov_type)
    return res


# ── 8. Regression Specs ───────────────────────────────────────────────────────

def _stars(pval):
    if pval < 0.001: return "***"
    elif pval < 0.01: return "**"
    elif pval < 0.05: return "*"
    elif pval < 0.1:  return "+"
    return ""

def _print_result(row, compact=False):
    beta, se, tstat, pval = row["beta"], row["se"], row["tstat"], row["pval"]
    sig = _stars(pval) if not np.isnan(pval) else ""
    if compact:
        print(f"  {row['spec']:25s} | {row['variable']:20s} | {row['se_type']:20s} | "
              f"β={beta:+.6f} SE={se:.6f} t={tstat:+.3f} p={pval:.4f}{sig}")
    else:
        print(f"  β = {beta:+.6f}  SE={se:.6f}  t={tstat:+.3f}  p={pval:.4f} {sig}")
        print(f"  R²w={row.get('r2_within', np.nan):.6f}  "
              f"N={row.get('n_obs','?'):,}  entities={row.get('n_entities','?'):,}")
        print()


def run_baseline(panel, centroids_df):
    print("\n" + "=" * 70)
    print("BASELINE: heat_t → ΔlnGDP_t  (entity+time FE, country quadratic trends)")
    print("=" * 70)
    xcols  = ["heat_freq_weighted"] + CONTROL_COLS
    ycol   = "admin2_growth"
    xvar   = xcols[0]
    rows   = []
    res    = run_panelols(panel, ycol, xcols)
    base   = {"spec": "baseline", "depvar": ycol, "variable": xvar,
              "beta": res.params[xvar], "r2_within": res.rsquared_within,
              "n_obs": int(res.nobs), "n_entities": int(res.entity_info["total"])}
    rows.append({**base, "se_type": "entity_clustered",
                 "se": res.std_errors[xvar], "tstat": res.tstats[xvar],
                 "pval": res.pvalues[xvar]})
    _print_result(rows[-1])
    print("Computing Conley 500km SEs...")
    conley = compute_conley_se(panel, ycol, xcols, centroids_df, cutoff_km=500)
    rows.append({**base, "se_type": "conley_500km",
                 "se": conley["se"][0], "tstat": conley["tstat"][0],
                 "pval": conley["pval"][0]})
    _print_result(rows[-1])
    print(f"  Conley neighbor pairs: {conley['n_pairs']:,}")
    return rows


def run_se_comparison(panel, centroids_df):
    print("\n" + "=" * 70)
    print("SE COMPARISON: 7 standard error types")
    print("=" * 70)
    xcols = ["heat_freq_weighted"] + CONTROL_COLS
    ycol  = "admin2_growth"; xvar = xcols[0]; rows = []
    res_ec = run_panelols(panel, ycol, xcols, cluster_entity=True, cluster_time=False)
    base = {"spec": "se_comparison", "depvar": ycol, "variable": xvar,
            "beta": res_ec.params[xvar], "r2_within": res_ec.rsquared_within,
            "n_obs": int(res_ec.nobs), "n_entities": int(res_ec.entity_info["total"])}
    rows.append({**base, "se_type": "entity_clustered",
                 "se": res_ec.std_errors[xvar], "tstat": res_ec.tstats[xvar],
                 "pval": res_ec.pvalues[xvar]})
    res_tc = run_panelols(panel, ycol, xcols, cluster_entity=False, cluster_time=True)
    rows.append({**base, "se_type": "time_clustered",
                 "se": res_tc.std_errors[xvar], "tstat": res_tc.tstats[xvar],
                 "pval": res_tc.pvalues[xvar]})
    res_tw = run_panelols(panel, ycol, xcols, cluster_entity=True, cluster_time=True)
    rows.append({**base, "se_type": "twoway_clustered",
                 "se": res_tw.std_errors[xvar], "tstat": res_tw.tstats[xvar],
                 "pval": res_tw.pvalues[xvar]})
    res_dk = run_panelols(panel, ycol, xcols, cov_type="kernel")
    rows.append({**base, "se_type": "driscoll_kraay",
                 "se": res_dk.std_errors[xvar], "tstat": res_dk.tstats[xvar],
                 "pval": res_dk.pvalues[xvar]})
    for cutoff in CONLEY_CUTOFFS_KM:
        print(f"  Conley {cutoff}km...")
        conley = compute_conley_se(panel, ycol, xcols, centroids_df, cutoff_km=cutoff)
        rows.append({**base, "se_type": f"conley_{cutoff}km",
                     "se": conley["se"][0], "tstat": conley["tstat"][0],
                     "pval": conley["pval"][0]})
    for r in rows: _print_result(r, compact=True)
    return rows


def run_quadratic_spec(panel, centroids_df):
    print("\n" + "=" * 70)
    print("QUADRATIC: heat + heat²")
    print("=" * 70)
    panel["heat_sq"] = panel["heat_freq_weighted"] ** 2
    xcols = ["heat_freq_weighted", "heat_sq"] + CONTROL_COLS; rows = []
    res   = run_panelols(panel, "admin2_growth", xcols)
    base  = {"spec": "quadratic", "depvar": "admin2_growth",
             "r2_within": res.rsquared_within, "n_obs": int(res.nobs),
             "n_entities": int(res.entity_info["total"])}
    for col in xcols:
        rows.append({**base, "variable": col, "se_type": "entity_clustered",
                     "beta": res.params[col], "se": res.std_errors[col],
                     "tstat": res.tstats[col], "pval": res.pvalues[col]})
    b1, b2 = res.params["heat_freq_weighted"], res.params["heat_sq"]
    if b2 != 0:
        turning = -b1 / (2 * b2)
        print(f"  Turning point: {turning:.4f}")
        rows.append({**base, "variable": "turning_point", "se_type": "entity_clustered",
                     "beta": turning, "se": np.nan, "tstat": np.nan, "pval": np.nan})
    print("  Conley 500km...")
    conley = compute_conley_se(panel, "admin2_growth", xcols, centroids_df, 500)
    for i, col in enumerate(xcols):
        rows.append({**base, "variable": col, "se_type": "conley_500km",
                     "beta": conley["beta"][i], "se": conley["se"][i],
                     "tstat": conley["tstat"][i], "pval": conley["pval"][i]})
    for r in rows: _print_result(r, compact=True)
    panel.drop(columns=["heat_sq"], inplace=True)
    return rows


def run_bin_spec(panel, centroids_df):
    print("\n" + "=" * 70)
    print("QUINTILE BINS: Heat quintile dummies (Q1=ref)")
    print("=" * 70)
    panel["heat_q"] = pd.qcut(panel["heat_freq_weighted"], 5, labels=False,
                               duplicates="drop") + 1
    n_bins = panel["heat_q"].nunique(); bin_cols = []
    for q in range(2, n_bins + 1):
        col = f"heat_Q{q}"; panel[col] = (panel["heat_q"] == q).astype(float)
        bin_cols.append(col)
    rows = []; res = run_panelols(panel, "admin2_growth", bin_cols + CONTROL_COLS)
    base = {"spec": "quintile_bins", "depvar": "admin2_growth",
            "r2_within": res.rsquared_within, "n_obs": int(res.nobs),
            "n_entities": int(res.entity_info["total"])}
    for col in bin_cols:
        rows.append({**base, "variable": col, "se_type": "entity_clustered",
                     "beta": res.params[col], "se": res.std_errors[col],
                     "tstat": res.tstats[col], "pval": res.pvalues[col]})
    print("  Conley 500km...")
    conley = compute_conley_se(panel, "admin2_growth", bin_cols + CONTROL_COLS, centroids_df, 500)
    for i, col in enumerate(bin_cols):
        rows.append({**base, "variable": col, "se_type": "conley_500km",
                     "beta": conley["beta"][i], "se": conley["se"][i],
                     "tstat": conley["tstat"][i], "pval": conley["pval"][i]})
    for r in rows: _print_result(r, compact=True)
    panel.drop(columns=bin_cols + ["heat_q"], inplace=True)
    return rows


def run_income_heterogeneity(panel, centroids_df):
    print("\n" + "=" * 70)
    print("INCOME HETEROGENEITY: by GDP per capita tercile")
    print("=" * 70)
    xcols = ["heat_freq_weighted"] + CONTROL_COLS
    ycol  = "admin2_growth"; xvar = xcols[0]; rows = []
    for tercile in ["low", "mid", "high"]:
        sub = panel[panel["income_tercile"] == tercile].copy()
        n_sub = len(sub); n_ent = sub["GID_2"].nunique()
        print(f"\n  Tercile={tercile}: {n_sub:,} obs, {n_ent:,} entities")
        if n_sub < MIN_OBS or n_ent < MIN_ENTITIES:
            print("  Skipping"); continue
        res  = run_panelols(sub, ycol, xcols)
        base = {"spec": f"income_{tercile}", "depvar": ycol, "variable": xvar,
                "r2_within": res.rsquared_within, "n_obs": int(res.nobs),
                "n_entities": int(res.entity_info["total"])}
        rows.append({**base, "se_type": "entity_clustered",
                     "beta": res.params[xvar], "se": res.std_errors[xvar],
                     "tstat": res.tstats[xvar], "pval": res.pvalues[xvar]})
        print(f"  Conley 500km ({tercile})...")
        conley = compute_conley_se(sub, ycol, xcols, centroids_df, 500)
        rows.append({**base, "se_type": "conley_500km",
                     "beta": conley["beta"][0], "se": conley["se"][0],
                     "tstat": conley["tstat"][0], "pval": conley["pval"][0]})
        for r in rows[-2:]: _print_result(r, compact=True)
    return rows


def run_spline_spec(panel, centroids_df):
    from patsy import cr
    print("\n" + "=" * 70)
    print("SPLINE: cr(heat_freq_weighted, df=4)")
    print("=" * 70)
    heat = panel["heat_freq_weighted"].values
    spline_basis = cr(heat, df=4)
    spline_cols  = [f"heat_sp{i+1}" for i in range(spline_basis.shape[1])]
    for i, col in enumerate(spline_cols):
        panel[col] = spline_basis[:, i]
    rows = []
    res  = run_panelols(panel, "admin2_growth", spline_cols + CONTROL_COLS)
    base = {"spec": "spline_4df", "depvar": "admin2_growth",
            "r2_within": res.rsquared_within, "n_obs": int(res.nobs),
            "n_entities": int(res.entity_info["total"])}
    for col in spline_cols:
        if col in res.params.index:
            rows.append({**base, "variable": col, "se_type": "entity_clustered",
                         "beta": res.params[col], "se": res.std_errors[col],
                         "tstat": res.tstats[col], "pval": res.pvalues[col]})
    print("  Conley 500km on spline basis...")
    conley = compute_conley_se(panel, "admin2_growth", spline_cols + CONTROL_COLS,
                               centroids_df, 500)
    for i, col in enumerate(spline_cols):
        rows.append({**base, "variable": col, "se_type": "conley_500km",
                     "beta": conley["beta"][i], "se": conley["se"][i],
                     "tstat": conley["tstat"][i], "pval": conley["pval"][i]})
    for r in rows: _print_result(r, compact=True)
    panel.drop(columns=spline_cols, inplace=True)
    return rows


# ── 9. Main ───────────────────────────────────────────────────────────────────

def main():
    global CONTROL_COLS
    t0 = time.time()
    PANEL_DIR.mkdir(exist_ok=True)

    panel       = load_and_merge_panel()
    centroids_df = load_centroids(panel)

    control_configs = [
        ("linear_ctrl", ["annual_temp_c", "annual_precip_mm"]),
        ("quad_ctrl",   ["annual_temp_c", "annual_temp_c_sq", "annual_precip_mm"]),
    ]

    all_rows = []
    for ctrl_label, ctrl_cols in control_configs:
        if "annual_temp_c_sq" not in panel.columns and "annual_temp_c_sq" in ctrl_cols:
            panel["annual_temp_c_sq"] = panel["annual_temp_c"] ** 2
        print(f"\n{'#' * 70}")
        print(f"# CONTROL SET: {ctrl_label}  {ctrl_cols}")
        print(f"{'#' * 70}")
        CONTROL_COLS = ctrl_cols
        rows = []
        rows.extend(run_baseline(panel, centroids_df))
        rows.extend(run_se_comparison(panel, centroids_df))
        rows.extend(run_spline_spec(panel, centroids_df))
        rows.extend(run_quadratic_spec(panel, centroids_df))
        rows.extend(run_bin_spec(panel, centroids_df))
        rows.extend(run_income_heterogeneity(panel, centroids_df))
        for r in rows:
            r["spec"] = f"{ctrl_label}_{r['spec']}"
        all_rows.extend(rows)

    results   = pd.DataFrame(all_rows)
    col_order = ["spec", "depvar", "variable", "se_type", "beta", "se",
                 "tstat", "pval", "r2_within", "n_obs", "n_entities"]
    results   = results[[c for c in col_order if c in results.columns]]
    results.to_csv(str(RESULTS_MIXED_CSV), index=False, float_format="%.8f")
    print(f"\nResults saved to {RESULTS_MIXED_CSV}")
    print(f"Total rows: {len(results)}")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
