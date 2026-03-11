"""
Panel regression: Extreme heat → GDP growth (admin2, global)

Two-way fixed effects (entity + time) with country-specific quadratic time
trends (BHM 2015-style) and Conley spatial HAC standard errors.

Main specification: contemporaneous (heat_t → GDP_t).
Robustness: spline, quadratic, quintile bins, GDP-residual DV,
            income tercile heterogeneity, lag/lead structure.

Inputs (set paths in src/config.py or via environment variables):
  HEAT_PANEL_PATH  — admin2_heat_settlement_weighted.parquet
  GDP_GROWTH_PATH  — adm2_gdp_growth_residuals_long.parquet
  CONTROLS_PATH    — admin2_era5land_annual_controls.parquet
  ADMIN2_GPKG      — polyg_adm2_gdp_perCapita_1990_2022.gpkg  (for centroids)

  See data/raw/gdp/README.md for the GDP data construction.

Output:
  <PANEL_DIR>/panel_regression_gdp_heat_results.csv
    One row per specification × SE type × variable.

Key result:
  Baseline β (heat_freq_weighted → admin2_growth) = −0.0361
  Conley 500 km SE = 0.00787, t = −4.58

Reference: Burke, Hsiang & Miguel (2015). "Global non-linear effect of
temperature on economic production." Nature 527, 235–239.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import HEAT_PANEL_PATH, GDP_GROWTH_PATH, CONTROLS_PATH, ADMIN2_GPKG, PANEL_DIR

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── 1. Constants & Paths ─────────────────────────────────────────────────────

HEAT_PATH     = HEAT_PANEL_PATH
OUTPUT_DIR    = PANEL_DIR
OUTPUT_CSV    = OUTPUT_DIR / "panel_regression_gdp_heat_results.csv"
GPKG_PATH     = ADMIN2_GPKG

MIN_OBS        = 1000
MIN_ENTITIES   = 50
CONLEY_CUTOFFS_KM = [250, 500, 1000]
EARTH_RADIUS_KM   = 6371.0

CONTROL_COLS = ["annual_temp_c", "annual_precip_mm"]


# ── 2. Data Loading ──────────────────────────────────────────────────────────

def load_and_merge_panel():
    """Load heat + GDP growth parquets, merge on (GID_2, year) [contemporaneous]."""
    print("Loading heat parquet...")
    heat = pd.read_parquet(HEAT_PATH)
    heat = heat.rename(columns={"year": "year_heat"})
    heat["year_gdp"] = heat["year_heat"]   # contemporaneous: heat_t → GDP_t

    print("Loading GDP growth parquet...")
    gdp = pd.read_parquet(GDP_GROWTH_PATH)
    gdp = gdp.rename(columns={"year": "year_gdp"})

    print("Merging...")
    panel = heat.merge(gdp[["GID_2", "year_gdp", "admin2_growth", "residual"]],
                       on=["GID_2", "year_gdp"], how="inner")
    panel = panel.dropna(subset=["admin2_growth", "heat_freq_weighted"])

    # Static income tercile from entity-median GDP per capita
    median_gdp = panel.groupby("GID_2")["gdp_per_capita"].median()
    tercile_labels = pd.qcut(median_gdp, 3, labels=["low", "mid", "high"])
    panel["income_tercile"] = panel["GID_2"].map(tercile_labels)

    # Merge ERA5 annual climate controls (T, precip)
    base_ctrl = ["annual_temp_c", "annual_precip_mm"]
    if CONTROLS_PATH.exists():
        controls = pd.read_parquet(CONTROLS_PATH)
        controls = controls.rename(columns={"year": "year_heat"})
        n_before = len(panel)
        panel = panel.merge(
            controls[["GID_2", "year_heat"] + base_ctrl],
            on=["GID_2", "year_heat"], how="left",
        )
        panel = panel.dropna(subset=base_ctrl)
        panel["annual_temp_c_sq"] = panel["annual_temp_c"] ** 2
        print(f"ERA5 controls (T, P) merged: {n_before:,} → {len(panel):,} obs "
              f"({n_before - len(panel):,} dropped for missing controls)")
    else:
        print(f"WARNING: {CONTROLS_PATH} not found — running without T/P controls")
        for col in base_ctrl + ["annual_temp_c_sq"]:
            panel[col] = np.nan

    n_obs      = len(panel)
    n_ent      = panel["GID_2"].nunique()
    n_years    = panel["year_heat"].nunique()
    n_countries= panel["iso3"].nunique()
    print(f"Panel: {n_obs:,} obs, {n_ent:,} entities, {n_countries} countries, "
          f"{n_years} years "
          f"(heat {panel['year_heat'].min()}–{panel['year_heat'].max()}, "
          f"GDP {panel['year_gdp'].min()}–{panel['year_gdp'].max()})")
    print("Specification: contemporaneous (heat_t → GDP_t), country quadratic trends (FWL)")
    assert n_obs >= MIN_OBS,      f"Too few observations: {n_obs}"
    assert n_ent >= MIN_ENTITIES, f"Too few entities: {n_ent}"
    return panel


# ── 3. Centroid Extraction ───────────────────────────────────────────────────

def load_centroids(panel_gids):
    """Read admin2 GPKG, compute centroids, return (GID_2, lat, lon)."""
    import geopandas as gpd
    print("Loading admin2 GPKG for centroids...")
    gdf = gpd.read_file(str(GPKG_PATH), ignore_fields=[
        str(y) for y in range(1990, 2023)] + ["slope", "id", "adm2ID"])
    gdf = gdf[gdf["GID_2"].isin(panel_gids)].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf["lat"] = gdf.geometry.centroid.y
        gdf["lon"] = gdf.geometry.centroid.x
    centroids = gdf[["GID_2", "lat", "lon"]].copy()
    print(f"Centroids loaded: {len(centroids):,} entities")
    return centroids


# ── 4. Country-Specific Trend Removal (FWL) ──────────────────────────────────

def detrend_country(panel, cols, year_col="year_heat"):
    """Residualise columns against country-specific quadratic time trends.

    For each country (iso3), fits OLS: col_it = a_c + b_c·t + d_c·t² + ε.
    Equivalent to including iso3×t and iso3×t² as regressors (Frisch-Waugh-Lovell).
    """
    result   = panel.copy().reset_index(drop=True)
    panel_r  = result
    base_year = panel_r[year_col].min()

    for col in cols:
        resid = np.full(len(panel_r), np.nan)
        for _, grp in panel_r.groupby("iso3"):
            idx = grp.index
            y   = grp[col].values.astype(float)
            t   = (grp[year_col].values - base_year).astype(float)
            valid = np.isfinite(y)
            if valid.sum() <= 3:
                resid[idx] = y
                continue
            X_v  = np.column_stack([np.ones(valid.sum()), t[valid], t[valid] ** 2])
            beta, _, _, _ = np.linalg.lstsq(X_v, y[valid], rcond=None)
            X_all = np.column_stack([np.ones(len(t)), t, t ** 2])
            resid[idx] = y - X_all @ beta
        result[col] = resid

    return result


# ── 5. Two-Way Demeaning ─────────────────────────────────────────────────────

def twoway_demean(panel, ycol, xcols):
    """Demean y and X by entity and time using linearmodels PanelData.

    Returns demeaned y (Series) and X (DataFrame), aligned.
    """
    cols = [ycol] + list(xcols)
    sub  = panel.set_index(["GID_2", "year_heat"])[cols].copy()
    pdata = PanelData(sub)
    demeaned = pdata.demean("both")
    y_dm = demeaned.values2d[:, 0]
    X_dm = demeaned.values2d[:, 1:]
    index = demeaned.index
    y_out = pd.Series(y_dm, index=index, name=ycol)
    X_out = pd.DataFrame(X_dm, index=index, columns=xcols)
    return y_out, X_out


# ── 6. Conley Spatial HAC ────────────────────────────────────────────────────

def _latlon_to_xyz(lat, lon):
    """Convert lat/lon (degrees) to 3D unit vectors on sphere."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])


def _chord_distance_km(km):
    """Convert great-circle distance (km) to Euclidean chord distance on unit sphere."""
    angle = km / EARTH_RADIUS_KM
    return 2.0 * np.sin(angle / 2.0)


def compute_conley_se(panel, ycol, xcols, centroids_df, cutoff_km=500, country_trends=True):
    """Compute Conley spatial HAC standard errors with Bartlett kernel.

    Steps:
      0. (Optional) Residualise against country-specific quadratic trends (FWL).
      1. Two-way demean y and X.
      2. OLS on demeaned data → beta, residuals.
      3. Entity-level scores S_i = sum_t (X̃_it · ê_it).
      4. Build spatial weight matrix W using cKDTree + Bartlett kernel.
      5. Sandwich: V = (X'X)^{-1} (S'WS) (X'X)^{-1} · N/(N-1).

    Returns dict: beta, se, tstat, pval, r2_within, n_obs, n_entities.
    """
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

    K = len(xcols)
    N = gids.nunique() if hasattr(gids, "nunique") else len(set(gids))
    n_obs = len(y_dm)

    unique_gids = np.array(sorted(set(gids)))
    gid_to_idx  = {g: i for i, g in enumerate(unique_gids)}
    entity_idx  = np.array([gid_to_idx[g] for g in gids])

    scores = np.zeros((len(unique_gids), K))
    xe = X_dm * resid[:, None]
    np.add.at(scores, entity_idx, xe)

    cent = centroids_df.set_index("GID_2")
    matched_lats = np.array([cent.loc[g, "lat"] if g in cent.index else np.nan
                             for g in unique_gids])
    matched_lons = np.array([cent.loc[g, "lon"] if g in cent.index else np.nan
                             for g in unique_gids])
    valid = ~(np.isnan(matched_lats) | np.isnan(matched_lons))
    if valid.sum() < N * 0.9:
        print(f"  Warning: only {valid.sum()}/{N} entities have centroids")

    scores_v = scores[valid]
    lats_v   = matched_lats[valid]
    lons_v   = matched_lons[valid]
    N_v      = len(scores_v)

    xyz = _latlon_to_xyz(lats_v, lons_v)
    tree = cKDTree(xyz)
    chord_cutoff = _chord_distance_km(cutoff_km)
    pairs = tree.query_pairs(chord_cutoff, output_type="ndarray")

    if len(pairs) > 0:
        d      = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
        gc_km  = 2.0 * EARTH_RADIUS_KM * np.arcsin(d / 2.0)
        weights = 1.0 - gc_km / cutoff_km
        weights = np.clip(weights, 0, 1)

        row  = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(N_v)])
        col  = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(N_v)])
        data = np.concatenate([weights, weights, np.ones(N_v)])
        W    = sparse.csr_matrix((data, (row, col)), shape=(N_v, N_v))
    else:
        W = sparse.eye(N_v, format="csr")

    WS   = W @ scores_v
    meat = scores_v.T @ WS
    V    = XtX_inv @ meat @ XtX_inv * (N_v / (N_v - 1.0))

    diag_V   = np.diag(V)
    diag_V   = np.where(diag_V > 0, diag_V, np.nan)
    conley_se = np.sqrt(diag_V)
    tstat     = beta / conley_se
    from scipy.stats import t as tdist
    pval = np.where(np.isfinite(tstat),
                    2.0 * tdist.sf(np.abs(tstat), df=n_obs - N - K),
                    np.nan)

    return {
        "beta":       beta,
        "se":         conley_se,
        "tstat":      tstat,
        "pval":       pval,
        "r2_within":  r2_within,
        "n_obs":      n_obs,
        "n_entities": N,
        "n_pairs":    len(pairs) if len(pairs) > 0 else 0,
    }


# ── 7. Baseline Regression ───────────────────────────────────────────────────

def _drop_rank_deficient(X_df):
    """Drop linearly dependent columns from X using pivoted QR."""
    from scipy.linalg import qr as scipy_qr
    vals = X_df.values
    if vals.shape[1] <= 1:
        return X_df, []
    _, R, piv = scipy_qr(vals, pivoting=True, mode="economic")
    tol  = max(vals.shape) * abs(R[0, 0]) * np.finfo(float).eps * 100
    rank = np.sum(np.abs(np.diag(R)) > tol)
    if rank >= vals.shape[1]:
        return X_df, []
    keep    = sorted(piv[:rank])
    dropped = [X_df.columns[i] for i in range(len(X_df.columns)) if i not in keep]
    return X_df.iloc[:, keep], dropped


def run_panelols(panel, ycol, xcols, cov_type="clustered", cluster_entity=True,
                 cluster_time=False, clusters=None, country_trends=True):
    """Run PanelOLS with entity+time FE (+ country quadratic trends via FWL)."""
    if country_trends and "iso3" in panel.columns:
        panel = detrend_country(panel, [ycol] + list(xcols))
    sub = panel.set_index(["GID_2", "year_heat"])[[ycol] + list(xcols)].dropna()
    y = sub[ycol]
    X = sub[xcols]

    X, dropped = _drop_rank_deficient(X)
    if dropped:
        print(f"  Dropped rank-deficient columns: {dropped}")
        xcols = list(X.columns)

    model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    if cov_type == "clustered":
        res = model.fit(cov_type="clustered",
                        cluster_entity=cluster_entity,
                        cluster_time=cluster_time,
                        clusters=clusters)
    elif cov_type == "kernel":
        res = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=5)
    else:
        res = model.fit(cov_type=cov_type)
    return res


def run_baseline(panel, centroids_df):
    """Baseline: heat_freq_weighted → admin2_growth, entity+time FE + country trends."""
    print("\n" + "=" * 70)
    print("BASELINE: heat_t → ΔlnGDP_t (entity+time FE, country quadratic trends, T/P controls)")
    print("=" * 70)

    xcols = ["heat_freq_weighted"] + CONTROL_COLS
    ycol  = "admin2_growth"
    rows  = []

    res  = run_panelols(panel, ycol, xcols)
    xvar = xcols[0]
    row_base = {
        "spec":       "baseline",
        "depvar":     ycol,
        "variable":   xvar,
        "beta":       res.params[xvar],
        "r2_within":  res.rsquared_within,
        "n_obs":      int(res.nobs),
        "n_entities": int(res.entity_info["total"]),
    }

    row_ec = {**row_base, "se_type": "entity_clustered",
              "se": res.std_errors[xvar], "tstat": res.tstats[xvar],
              "pval": res.pvalues[xvar]}
    rows.append(row_ec)
    _print_result(row_ec)

    print("\nComputing Conley 500km SEs...")
    conley = compute_conley_se(panel, ycol, xcols, centroids_df, cutoff_km=500)
    row_conley = {**row_base, "se_type": "conley_500km",
                  "se":    conley["se"][0],
                  "tstat": conley["tstat"][0],
                  "pval":  conley["pval"][0]}
    rows.append(row_conley)
    _print_result(row_conley)
    print(f"  Conley neighbor pairs: {conley['n_pairs']:,}")

    return rows


# ── 8. Robustness Specifications ─────────────────────────────────────────────

def run_se_comparison(panel, centroids_df):
    """Same baseline model with 7 SE types."""
    print("\n" + "=" * 70)
    print("SE COMPARISON: 7 standard error types (with T/P controls)")
    print("=" * 70)

    xcols = ["heat_freq_weighted"] + CONTROL_COLS
    ycol  = "admin2_growth"
    xvar  = xcols[0]
    rows  = []

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

    for r in rows:
        _print_result(r, compact=True)
    return rows


def run_spline_spec(panel, centroids_df):
    """Natural cubic spline basis (4 df) on heat_freq_weighted."""
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
            "r2_within": res.rsquared_within,
            "n_obs": int(res.nobs), "n_entities": int(res.entity_info["total"])}

    for col in spline_cols:
        if col in res.params.index:
            rows.append({**base, "variable": col, "se_type": "entity_clustered",
                         "beta": res.params[col], "se": res.std_errors[col],
                         "tstat": res.tstats[col], "pval": res.pvalues[col]})
        else:
            print(f"  Warning: {col} absorbed/dropped")

    # Joint Wald test
    kept_cols = [c for c in spline_cols if c in res.params.index]
    K_wald    = len(kept_cols)
    beta_vec  = res.params[kept_cols].values
    vcov      = res.cov[kept_cols].loc[kept_cols].values
    try:
        vcov_inv  = np.linalg.pinv(vcov)
        wald_stat = float(beta_vec @ vcov_inv @ beta_vec) / K_wald
        from scipy.stats import f as fdist
        wald_pval = fdist.sf(wald_stat, K_wald, int(res.nobs) - K_wald)
        if wald_stat < 0:
            wald_stat, wald_pval = np.nan, np.nan
    except np.linalg.LinAlgError:
        wald_stat, wald_pval = np.nan, np.nan
    rows.append({**base, "variable": "joint_wald", "se_type": "entity_clustered",
                 "beta": np.nan, "se": np.nan,
                 "tstat": wald_stat, "pval": wald_pval})
    print(f"  Joint Wald F-stat: {wald_stat:.3f}, p={wald_pval:.4f}")

    print("  Conley 500km on spline basis...")
    conley = compute_conley_se(panel, "admin2_growth",
                               spline_cols + CONTROL_COLS, centroids_df, 500)
    for i, col in enumerate(spline_cols):
        rows.append({**base, "variable": col, "se_type": "conley_500km",
                     "beta": conley["beta"][i], "se": conley["se"][i],
                     "tstat": conley["tstat"][i], "pval": conley["pval"][i]})

    for r in rows:
        _print_result(r, compact=True)
    panel.drop(columns=spline_cols, inplace=True)
    return rows


def run_quadratic_spec(panel, centroids_df):
    """heat + heat² with turning point."""
    print("\n" + "=" * 70)
    print("QUADRATIC: heat + heat²")
    print("=" * 70)

    panel["heat_sq"] = panel["heat_freq_weighted"] ** 2
    xcols = ["heat_freq_weighted", "heat_sq"] + CONTROL_COLS
    rows  = []

    res  = run_panelols(panel, "admin2_growth", xcols)
    base = {"spec": "quadratic", "depvar": "admin2_growth",
            "r2_within": res.rsquared_within,
            "n_obs": int(res.nobs), "n_entities": int(res.entity_info["total"])}

    for col in xcols:
        rows.append({**base, "variable": col, "se_type": "entity_clustered",
                     "beta": res.params[col], "se": res.std_errors[col],
                     "tstat": res.tstats[col], "pval": res.pvalues[col]})

    b1, b2 = res.params["heat_freq_weighted"], res.params["heat_sq"]
    if b2 != 0:
        turning = -b1 / (2 * b2)
        print(f"  Turning point: heat_freq = {turning:.4f}")
        rows.append({**base, "variable": "turning_point", "se_type": "entity_clustered",
                     "beta": turning, "se": np.nan, "tstat": np.nan, "pval": np.nan})

    print("  Conley 500km...")
    conley = compute_conley_se(panel, "admin2_growth", xcols, centroids_df, 500)
    for i, col in enumerate(xcols):
        rows.append({**base, "variable": col, "se_type": "conley_500km",
                     "beta": conley["beta"][i], "se": conley["se"][i],
                     "tstat": conley["tstat"][i], "pval": conley["pval"][i]})

    for r in rows:
        _print_result(r, compact=True)
    panel.drop(columns=["heat_sq"], inplace=True)
    return rows


def run_bin_spec(panel, centroids_df):
    """Heat quintile dummies (Q1 = reference)."""
    print("\n" + "=" * 70)
    print("QUINTILE BINS: Heat quintile dummies (Q1 = ref)")
    print("=" * 70)

    panel["heat_q"] = pd.qcut(panel["heat_freq_weighted"], 5,
                               labels=False, duplicates="drop") + 1
    n_bins   = panel["heat_q"].nunique()
    bin_cols = []
    for q in range(2, n_bins + 1):
        col = f"heat_Q{q}"
        panel[col] = (panel["heat_q"] == q).astype(float)
        bin_cols.append(col)

    rows = []
    res  = run_panelols(panel, "admin2_growth", bin_cols + CONTROL_COLS)
    base = {"spec": "quintile_bins", "depvar": "admin2_growth",
            "r2_within": res.rsquared_within,
            "n_obs": int(res.nobs), "n_entities": int(res.entity_info["total"])}

    for col in bin_cols:
        rows.append({**base, "variable": col, "se_type": "entity_clustered",
                     "beta": res.params[col], "se": res.std_errors[col],
                     "tstat": res.tstats[col], "pval": res.pvalues[col]})

    betas    = [res.params[col] for col in bin_cols]
    monotone = (all(betas[i] <= betas[i+1] for i in range(len(betas)-1)) or
                all(betas[i] >= betas[i+1] for i in range(len(betas)-1)))
    print(f"  Bin coefficients: {['%.5f' % b for b in betas]}")
    print(f"  Monotonic: {monotone}")

    print("  Conley 500km...")
    conley = compute_conley_se(panel, "admin2_growth",
                               bin_cols + CONTROL_COLS, centroids_df, 500)
    for i, col in enumerate(bin_cols):
        rows.append({**base, "variable": col, "se_type": "conley_500km",
                     "beta": conley["beta"][i], "se": conley["se"][i],
                     "tstat": conley["tstat"][i], "pval": conley["pval"][i]})

    for r in rows:
        _print_result(r, compact=True)
    panel.drop(columns=bin_cols + ["heat_q"], inplace=True)
    return rows


def run_residual_dv_spec(panel, centroids_df):
    """DV = idiosyncratic GDP growth (country-cycle residual)."""
    print("\n" + "=" * 70)
    print("RESIDUAL DV: heat → GDP growth residual (net of country cycle)")
    print("=" * 70)

    sub   = panel.dropna(subset=["residual"]).copy()
    xcols = ["heat_freq_weighted"] + CONTROL_COLS
    ycol  = "residual"
    rows  = []

    res  = run_panelols(sub, ycol, xcols)
    xvar = xcols[0]
    base = {"spec": "residual_dv", "depvar": ycol, "variable": xvar,
            "r2_within": res.rsquared_within,
            "n_obs": int(res.nobs), "n_entities": int(res.entity_info["total"])}

    rows.append({**base, "se_type": "entity_clustered",
                 "beta": res.params[xvar], "se": res.std_errors[xvar],
                 "tstat": res.tstats[xvar], "pval": res.pvalues[xvar]})

    print("  Conley 500km...")
    conley = compute_conley_se(sub, ycol, xcols, centroids_df, 500)
    rows.append({**base, "se_type": "conley_500km",
                 "beta": conley["beta"][0], "se": conley["se"][0],
                 "tstat": conley["tstat"][0], "pval": conley["pval"][0]})

    for r in rows:
        _print_result(r)
    return rows


def run_income_heterogeneity(panel, centroids_df):
    """Split by GDP per capita tercile."""
    print("\n" + "=" * 70)
    print("INCOME HETEROGENEITY: by GDP per capita tercile (with T/P controls)")
    print("=" * 70)

    xcols = ["heat_freq_weighted"] + CONTROL_COLS
    ycol  = "admin2_growth"
    xvar  = xcols[0]
    rows  = []

    for tercile in ["low", "mid", "high"]:
        sub   = panel[panel["income_tercile"] == tercile].copy()
        n_sub = len(sub)
        n_ent = sub["GID_2"].nunique()
        print(f"\n  Tercile={tercile}: {n_sub:,} obs, {n_ent:,} entities")

        if n_sub < MIN_OBS or n_ent < MIN_ENTITIES:
            print("  Skipping (too few obs/entities)")
            continue

        res  = run_panelols(sub, ycol, xcols)
        base = {"spec": f"income_{tercile}", "depvar": ycol, "variable": xvar,
                "r2_within": res.rsquared_within,
                "n_obs": int(res.nobs), "n_entities": int(res.entity_info["total"])}

        rows.append({**base, "se_type": "entity_clustered",
                     "beta": res.params[xvar], "se": res.std_errors[xvar],
                     "tstat": res.tstats[xvar], "pval": res.pvalues[xvar]})

        print(f"  Conley 500km ({tercile})...")
        conley = compute_conley_se(sub, ycol, xcols, centroids_df, 500)
        rows.append({**base, "se_type": "conley_500km",
                     "beta": conley["beta"][0], "se": conley["se"][0],
                     "tstat": conley["tstat"][0], "pval": conley["pval"][0]})

        for r in rows[-2:]:
            _print_result(r, compact=True)

    return rows


def run_lag_structure(panel, centroids_df):
    """Contemporaneous (baseline, year t) vs lead (year t+1) specification."""
    print("\n" + "=" * 70)
    print("LAG STRUCTURE: contemporaneous (baseline) vs lead (t→t+1, with T/P controls)")
    print("=" * 70)

    xcols = ["heat_freq_weighted"] + CONTROL_COLS
    ycol  = "admin2_growth"
    xvar  = xcols[0]
    rows  = []

    res_contemp = run_panelols(panel, ycol, xcols)
    base_contemp = {"spec": "lag_contemporaneous", "depvar": ycol, "variable": xvar,
                    "r2_within": res_contemp.rsquared_within,
                    "n_obs": int(res_contemp.nobs),
                    "n_entities": int(res_contemp.entity_info["total"])}
    rows.append({**base_contemp, "se_type": "entity_clustered",
                 "beta": res_contemp.params[xvar], "se": res_contemp.std_errors[xvar],
                 "tstat": res_contemp.tstats[xvar], "pval": res_contemp.pvalues[xvar]})

    # Lead: heat in year t → GDP growth in year t+1
    print("  Building lead panel (t→t+1)...")
    heat = pd.read_parquet(HEAT_PATH, columns=["GID_2", "iso3", "year", "heat_freq_weighted"])
    heat = heat.rename(columns={"year": "year_heat"})
    heat["year_gdp"] = heat["year_heat"] + 1

    gdp = pd.read_parquet(GDP_GROWTH_PATH, columns=["GID_2", "year", "admin2_growth"])
    gdp = gdp.rename(columns={"year": "year_gdp"})

    lead = heat.merge(gdp, on=["GID_2", "year_gdp"], how="inner")
    lead = lead.dropna(subset=["admin2_growth", "heat_freq_weighted"])

    if CONTROLS_PATH.exists():
        base_ctrl = ["annual_temp_c", "annual_precip_mm"]
        controls  = pd.read_parquet(CONTROLS_PATH)
        controls  = controls.rename(columns={"year": "year_heat"})
        lead = lead.merge(
            controls[["GID_2", "year_heat"] + base_ctrl],
            on=["GID_2", "year_heat"], how="left",
        )
        lead = lead.dropna(subset=base_ctrl)
        lead["annual_temp_c_sq"] = lead["annual_temp_c"] ** 2

    res_lead = run_panelols(lead, ycol, xcols)
    base_lead = {"spec": "lag_lead", "depvar": ycol, "variable": xvar,
                 "r2_within": res_lead.rsquared_within,
                 "n_obs": int(res_lead.nobs),
                 "n_entities": int(res_lead.entity_info["total"])}
    rows.append({**base_lead, "se_type": "entity_clustered",
                 "beta": res_lead.params[xvar], "se": res_lead.std_errors[xvar],
                 "tstat": res_lead.tstats[xvar], "pval": res_lead.pvalues[xvar]})

    print("  Conley 500km (contemporaneous)...")
    conley_contemp = compute_conley_se(panel, ycol, xcols, centroids_df, 500)
    rows.append({**base_contemp, "se_type": "conley_500km",
                 "beta": conley_contemp["beta"][0], "se": conley_contemp["se"][0],
                 "tstat": conley_contemp["tstat"][0], "pval": conley_contemp["pval"][0]})

    print("  Conley 500km (lead)...")
    conley_lead = compute_conley_se(lead, ycol, xcols, centroids_df, 500)
    rows.append({**base_lead, "se_type": "conley_500km",
                 "beta": conley_lead["beta"][0], "se": conley_lead["se"][0],
                 "tstat": conley_lead["tstat"][0], "pval": conley_lead["pval"][0]})

    for r in rows:
        _print_result(r, compact=True)
    return rows


# ── 9. Output Formatting ─────────────────────────────────────────────────────

def _stars(pval):
    if pval < 0.001: return "***"
    elif pval < 0.01: return "**"
    elif pval < 0.05: return "*"
    elif pval < 0.1:  return "+"
    return ""


def _print_result(row, compact=False):
    beta  = row["beta"]
    se    = row["se"]
    tstat = row["tstat"]
    pval  = row["pval"]
    sig   = _stars(pval) if not np.isnan(pval) else ""

    if compact:
        print(f"  {row['spec']:25s} | {row['variable']:20s} | {row['se_type']:20s} | "
              f"β={beta:+.6f} SE={se:.6f} t={tstat:+.3f} p={pval:.4f}{sig}")
    else:
        print(f"  Spec:     {row['spec']}")
        print(f"  Variable: {row['variable']}")
        print(f"  SE type:  {row['se_type']}")
        print(f"  β = {beta:+.6f}  ({se:.6f})")
        print(f"  t = {tstat:+.3f}  p = {pval:.4f} {sig}")
        print(f"  R² within = {row.get('r2_within', np.nan):.6f}")
        print(f"  N = {row.get('n_obs', '?'):,}  entities = {row.get('n_entities', '?'):,}")
        print()


# ── 10. Main ─────────────────────────────────────────────────────────────────

def main():
    global CONTROL_COLS
    t0 = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    panel = load_and_merge_panel()

    print("\n--- Summary Statistics ---")
    for col in ["heat_freq_weighted", "admin2_growth", "residual", "gdp_per_capita",
                "annual_temp_c", "annual_temp_c_sq", "annual_precip_mm"]:
        if col in panel.columns:
            s = panel[col].dropna()
            print(f"  {col:25s}: mean={s.mean():.6f} sd={s.std():.6f} "
                  f"min={s.min():.6f} max={s.max():.6f} N={len(s):,}")

    centroids_df = load_centroids(set(panel["GID_2"].unique()))

    control_configs = [
        ("linear_ctrl", ["annual_temp_c", "annual_precip_mm"]),
        ("quad_ctrl",   ["annual_temp_c", "annual_temp_c_sq", "annual_precip_mm"]),
    ]

    all_rows = []
    for ctrl_label, ctrl_cols in control_configs:
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
        rows.extend(run_residual_dv_spec(panel, centroids_df))
        rows.extend(run_income_heterogeneity(panel, centroids_df))
        rows.extend(run_lag_structure(panel, centroids_df))

        for r in rows:
            r["spec"] = f"{ctrl_label}_{r['spec']}"
        all_rows.extend(rows)

    results   = pd.DataFrame(all_rows)
    col_order = ["spec", "depvar", "variable", "se_type", "beta", "se",
                 "tstat", "pval", "r2_within", "n_obs", "n_entities"]
    results   = results[[c for c in col_order if c in results.columns]]
    results.to_csv(OUTPUT_CSV, index=False, float_format="%.8f")
    print(f"\nResults saved to {OUTPUT_CSV}")
    print(f"Total rows: {len(results)}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
