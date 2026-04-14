#!/usr/bin/env python -u
"""
Two-panel comparison figure: IFS reforecast slopes vs CIL GDPCIR ensemble slopes.

Panel 1 (left):  Local slope    — IFS clima  (top) | CIL ensemble local  (bottom)
Panel 2 (right): 200km slope    — IFS inter  (top) | CIL ensemble 200km  (bottom)

All maps in Risk Ratio per K (log colour scale, 0.8–5).
Both CIL and IFS slopes are on the same units (log RR per °C from the same
log-EHD ~ T_anom regression methodology).

Output:
  projections/output/slope_ifs_vs_cil_comparison.png

Usage:
  /turbo/mgoldklang/pyenvs/peg_nov_24/bin/python \\
      projections/scripts/plot_slope_comparison.py
"""

import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PROJ_DIR    = Path(__file__).resolve().parent.parent
OUT_DIR     = PROJ_DIR / "output"

SLOPES_DIR  = Path("/home/mgoldklang/code/Extreme_temperature_and_financial_markets/data/slopes")
SLOPE_LOCAL = SLOPES_DIR / "global_summer_clima_rr_per_degC_2001_2020_025deg.nc"
SLOPE_200KM = SLOPES_DIR / "global_summer_inter_rr_per_degC_200km_era5clim_2001_2020_025deg.nc"

CIL_LOCAL   = OUT_DIR / "cil_ehd_slopes_local.nc"
CIL_200KM   = OUT_DIR / "cil_ehd_slopes_200km.nc"


def load_ifs():
    ds_l = xr.open_dataset(str(SLOPE_LOCAL))
    rr_l = ds_l["rr_per_degC"].values.copy()
    lats  = ds_l["latitude"].values
    lons  = ds_l["longitude"].values
    ds_l.close()

    ds_2 = xr.open_dataset(str(SLOPE_200KM))
    rr_2 = ds_2["rr_per_degC"].values.copy()
    ds_2.close()

    # Clip
    for arr in [rr_l, rr_2]:
        arr[~np.isfinite(arr) | (arr > 50)] = np.nan

    # Sort to ascending lat
    if lats[0] > lats[-1]:
        rr_l  = rr_l[::-1, :]
        rr_2  = rr_2[::-1, :]
        lats  = lats[::-1]

    return rr_l, rr_2, lats, lons


def load_cil():
    if not CIL_LOCAL.exists() or not CIL_200KM.exists():
        return None, None, None, None
    ds_l = xr.open_dataset(str(CIL_LOCAL))
    rr_l = ds_l["rr_per_degC"].values.copy()
    lats  = ds_l["lat"].values
    lons  = ds_l["lon"].values
    ds_l.close()
    ds_2 = xr.open_dataset(str(CIL_200KM))
    rr_2 = ds_2["rr_per_degC"].values.copy()
    ds_2.close()
    return rr_l, rr_2, lats, lons


def draw_map(ax, data, lats, lons, title, norm, cmap, show_cbar=False, pending=False):
    if pending or data is None:
        ax.set_facecolor("#f0f0f0")
        ax.text(0.5, 0.5, "Pending\n(slopes job still running)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#888888", style="italic")
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        return

    im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                       shading="auto", rasterized=True)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    if show_cbar:
        cb = plt.colorbar(im, ax=ax, orientation="vertical",
                          fraction=0.025, pad=0.03,
                          ticks=[0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
        cb.ax.set_yticklabels(["0.8", "1.0", "1.5", "2.0", "3.0", "5.0"], fontsize=8)
        cb.set_label("RR / K", fontsize=9)
    return im


def regrid_to(source, src_lats, src_lons, tgt_lats, tgt_lons):
    """Bilinearly interpolate source onto target lat/lon grid."""
    da = xr.DataArray(source, dims=["lat", "lon"],
                      coords={"lat": src_lats, "lon": src_lons})
    return da.interp(lat=tgt_lats, lon=tgt_lons, method="linear").values.astype(np.float32)


def make_difference_figure(cil_local, cil_200km, cil_lats, cil_lons,
                            ifs_local, ifs_200km, ifs_lats, ifs_lons):
    """
    Two-panel figure: CIL ensemble − IFS (on CIL grid) for local and 200km slopes.
    Diverging colourmap centred at 0, units = RR/K difference.
    """
    # Regrid IFS onto CIL grid
    ifs_loc_rg = regrid_to(ifs_local, ifs_lats, ifs_lons, cil_lats, cil_lons)
    ifs_200_rg = regrid_to(ifs_200km, ifs_lats, ifs_lons, cil_lats, cil_lons)

    diff_local = cil_local - ifs_loc_rg
    diff_200km = cil_200km - ifs_200_rg

    # Symmetric colour range: 95th percentile of absolute differences
    vmax = float(np.nanpercentile(
        np.abs(np.concatenate([diff_local[np.isfinite(diff_local)],
                               diff_200km[np.isfinite(diff_200km)]])),
        95
    ))
    vmax = round(vmax * 2) / 2   # round to nearest 0.5 for clean ticks
    vmax = max(vmax, 0.5)        # floor at ±0.5

    print(f"  Difference range:  ±{vmax:.2f} RR/K  (95th pctile abs)")

    cmap = "RdBu_r"   # red = CIL > IFS (more sensitive), blue = CIL < IFS
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5),
                             gridspec_kw={"wspace": 0.08})

    panels = [
        (axes[0], diff_local, "CIL local − IFS local  (RR/K)"),
        (axes[1], diff_200km, "CIL 200 km − IFS 200 km  (RR/K)"),
    ]
    for ax, data, title in panels:
        im = ax.pcolormesh(cil_lons, cil_lats, data, cmap=cmap, norm=norm,
                           shading="auto", rasterized=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)
        cb = plt.colorbar(im, ax=ax, orientation="vertical",
                          fraction=0.025, pad=0.03)
        cb.set_label("ΔRISK RATIO / K\n(CIL − IFS)", fontsize=8)
        cb.ax.tick_params(labelsize=8)

        # Annotate medians
        med = np.nanmedian(data)
        ax.text(0.02, 0.03, f"median = {med:+.3f}", transform=ax.transAxes,
                fontsize=8, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    fig.suptitle(
        "Difference in EHD Sensitivity Slope: CIL GDPCIR Ensemble − IFS Reforecast\n"
        "Red = CIL more sensitive than IFS  |  Blue = CIL less sensitive",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.text(
        0.5, -0.02,
        "CIL: log EHD ~ T_anom (2000–2014, QDM-corrected, ECS-weighted)  |  "
        "IFS: log P(exceedance) ~ T_anom (2001–2020, reforecast ensemble)\n"
        "IFS regridded onto CIL 0.25° grid before differencing.",
        ha="center", fontsize=8, color="#555555"
    )

    out = OUT_DIR / "slope_ifs_vs_cil_difference.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}  ({out.stat().st_size/1e6:.1f} MB)")


def main():
    print("Loading IFS slopes...", flush=True)
    ifs_local, ifs_200km, ifs_lats, ifs_lons = load_ifs()
    print(f"  IFS local:  {np.isfinite(ifs_local).sum():,} valid  "
          f"median={np.nanmedian(ifs_local):.3f}")
    print(f"  IFS 200km:  {np.isfinite(ifs_200km).sum():,} valid  "
          f"median={np.nanmedian(ifs_200km):.3f}")

    print("Loading CIL ensemble slopes...", flush=True)
    cil_local, cil_200km, cil_lats, cil_lons = load_cil()
    if cil_local is not None:
        print(f"  CIL local:  {np.isfinite(cil_local).sum():,} valid  "
              f"median={np.nanmedian(cil_local):.3f}")
        print(f"  CIL 200km:  {np.isfinite(cil_200km).sum():,} valid  "
              f"median={np.nanmedian(cil_200km):.3f}")
    else:
        print("  CIL slopes not yet available — will show pending placeholder")

    # ── Figure layout: 2 rows × 2 cols ────────────────────────────────────────
    # Row 0: IFS local | IFS 200km
    # Row 1: CIL local | CIL 200km
    # Two clearly labelled comparison panels, one column per scale.
    norm = mcolors.LogNorm(vmin=0.8, vmax=5.0)
    cmap = "YlOrRd"

    fig, axes = plt.subplots(
        2, 2,
        figsize=(18, 9),
        gridspec_kw={"hspace": 0.12, "wspace": 0.08}
    )

    # ── Column labels (scale) ─────────────────────────────────────────────────
    for col, label in enumerate(["Local (no smoothing)", "200 km smoothed"]):
        axes[0, col].set_title(
            f"{'─'*12}  {label}  {'─'*12}",
            fontsize=11, fontweight="bold", pad=8
        )

    # Draw the four maps
    draw_map(axes[0, 0], ifs_local, ifs_lats, ifs_lons,
             "IFS Reforecast — local  (clima, 2001–2020)",
             norm, cmap)
    draw_map(axes[0, 1], ifs_200km, ifs_lats, ifs_lons,
             "IFS Reforecast — 200 km  (inter, 2001–2020)",
             norm, cmap, show_cbar=True)
    draw_map(axes[1, 0], cil_local, cil_lats, cil_lons,
             "CIL GDPCIR ensemble — local  (2000–2014)",
             norm, cmap,
             pending=(cil_local is None))
    draw_map(axes[1, 1], cil_200km, cil_lats, cil_lons,
             "CIL GDPCIR ensemble — 200 km  (2000–2014)",
             norm, cmap, show_cbar=True,
             pending=(cil_200km is None))

    # ── Row labels ────────────────────────────────────────────────────────────
    for row, label in enumerate(["IFS Reforecast", "CIL GDPCIR\n(ECS-weighted ensemble)"]):
        axes[row, 0].annotate(
            label, xy=(-0.12, 0.5), xycoords="axes fraction",
            rotation=90, va="center", ha="center",
            fontsize=10, fontweight="bold"
        )

    # ── Shared colour bar annotation ──────────────────────────────────────────
    fig.text(
        0.5, 1.01,
        "EHD Sensitivity Slope: Risk Ratio per K  "
        "[ exp( d log(EHD) / d T_anom ) ]",
        ha="center", va="bottom", fontsize=12, fontweight="bold"
    )
    fig.text(
        0.5, -0.01,
        "IFS: log P(exceedance) ~ pooled-member T anomaly (2001–2020 JJA/DJF reforecasts)  |  "
        "CIL: log EHD ~ summer-mean T anomaly (2000–2014 CIL GDPCIR QDM historical)",
        ha="center", va="top", fontsize=8, color="#555555"
    )

    out = OUT_DIR / "slope_ifs_vs_cil_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out}  ({out.stat().st_size/1e6:.1f} MB)")

    # ── Difference figure (only once CIL slopes are available) ────────────────
    if cil_local is not None:
        print("\nGenerating difference maps (CIL − IFS)...")
        make_difference_figure(cil_local, cil_200km, cil_lats, cil_lons,
                               ifs_local, ifs_200km, ifs_lats, ifs_lons)
    else:
        print("\nSkipping difference maps — CIL slopes not yet available.")
        print("Re-run this script after compute_cil_ehd_slopes.py finishes.")


if __name__ == "__main__":
    main()
