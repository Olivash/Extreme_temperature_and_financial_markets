#!/usr/bin/env python -u
"""
Generate an interactive HTML ΔEHD explorer with Leaflet.

Recomputes all 6 ΔEHD variant grids (reusing functions from project_slopes_ehd_maps.py),
renders each as a transparent PNG overlay, then embeds everything into a single
self-contained HTML file with synced Leaflet map panes and a dropdown selector.

Output: projections/output/slopes_explorer.html

Usage:
  cd ~/projects/macro/extreme_heat/biodiversity_interactions
  /turbo/mgoldklang/pyenvs/peg_nov_24/bin/python projections/scripts/generate_html_maps.py
"""

import sys
import os
import io
import json
import time
import base64
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# Import computation functions from the existing script
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
    compute_global_stats,
    regrid_gcm_to_ifs,
    REGR_FILE,
    PANEL_FILE,
    OUT_DIR,
    ERA5_LATS,
    ERA5_LONS,
    CIL_ECS,
    CIL_EHD_DIR,
    PROJ_WINDOW,
    TASMAX_DIR,
    compute_ecs_weights,
    cmip6_summer_mean_year,
)
from bootstrap_impact_comparison import load_admin1_raster

PROJ_DIR  = Path(__file__).resolve().parent.parent
ROOT_DIR  = PROJ_DIR.parent
PARENT_DIR = ROOT_DIR.parent  # extreme_heat/

PANELS = {
    "admin1level": {
        "label": "Admin1-level mixed",
        "regr_file": PARENT_DIR / "output" / "panel_regression_gdp_heat_results_admin1level.csv",
        "panel_file": PARENT_DIR / "output" / "panel_mixed_admin1level.parquet",
        "raster_loader": "mixed",
    },
    "admin1pure": {
        "label": "Admin1 pure",
        "regr_file": PARENT_DIR / "output" / "panel_regression_gdp_heat_results_admin1pure.csv",
        "panel_file": PARENT_DIR / "output" / "panel_admin1_pure.parquet",
        "raster_loader": "admin1",
    },
}

# ─── Config ──────────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -60.0, 85.0   # crop to visible land
IMG_DPI = 150
IMG_W_IN, IMG_H_IN = 18, 6       # ~1800×600 px at 100 dpi

DEHD_BOUNDS = [-0.005, 0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.10, 0.15]
DIFF_BOUNDS = [-0.10, -0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06, 0.10]

ADMIN1_GPKG = PARENT_DIR / "polyg_adm1_gdp_perCapita_1990_2022 (1).gpkg"


# ─── Color helpers ───────────────────────────────────────────────────────────

def get_bin_colors_hex(bounds, cmap_name):
    """Return list of hex colors for each bin defined by bounds."""
    cmap = plt.cm.get_cmap(cmap_name, len(bounds) - 1)
    colors = []
    for i in range(len(bounds) - 1):
        r, g, b, _ = cmap(i)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def render_colorbar_png(bounds, cmap_name, label, width_in=10, height_in=1.0):
    """Render a horizontal colorbar as a PNG (bytes)."""
    cmap = plt.cm.get_cmap(cmap_name, len(bounds) - 1)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal", ticks=bounds)
    cbar.set_label(label, fontsize=14)
    cbar.ax.tick_params(labelsize=11)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=IMG_DPI, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def to_b64(png_bytes):
    return base64.b64encode(png_bytes).decode("ascii")


# ─── HTML template ───────────────────────────────────────────────────────────

def build_html(geojson_str, cbar_dehd_b64, cbar_diff_b64,
               dehd_colors_js, diff_colors_js, dehd_bounds_js, diff_bounds_js,
               betas_js, panel_stats, csv_table_html):
    """Build self-contained HTML with GeoJSON choropleth layers and hover tooltips."""

    # Build stats JS object — one sub-object per panel
    stats_js_parts = []
    for panel_name, rows in panel_stats.items():
        panel_parts = []
        for row in rows:
            k = row["key"]
            panel_parts.append(
                f'      "{k}": {{'
                f'beta: "{row["beta"]}", '
                f'gdpDehd: "{row["gdp_dehd"]}", '
                f'gdpImpact: "{row["gdp_impact"]}", '
                f'settDehd: "{row["sett_dehd"]}", '
                f'settImpact: "{row["sett_impact"]}", '
                f'p5: "{row["p5"]}", '
                f'p95: "{row["p95"]}"}}'
            )
        stats_js_parts.append(
            f'    "{panel_name}": {{\n' + ",\n".join(panel_parts) + "\n    }"
        )
    stats_js = "var stats = {\n" + ",\n".join(stats_js_parts) + "\n    };"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ΔEHD Explorer — 6-Variant Projections (2040, SSP2-4.5)</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.min.css"/>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/leaflet.sync@0.2.4/L.Map.Sync.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f5; color: #333; }}
  #header {{ background: #1a1a2e; color: #fff; padding: 12px 20px;
             display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
  #header h1 {{ font-size: 16px; font-weight: 600; white-space: nowrap; }}
  #header select {{ font-size: 14px; padding: 4px 8px; border-radius: 4px;
                    border: 1px solid #555; background: #2a2a4e; color: #fff; }}
  #header select option {{ background: #2a2a4e; }}
  #stats-bar {{ background: #222244; color: #ccc; padding: 6px 20px;
                font-size: 12px; display: flex; gap: 30px; flex-wrap: wrap; }}
  .stat-group {{ }}
  .stat-header {{ color: #8888bb; font-size: 10px; text-transform: uppercase;
                  letter-spacing: 0.5px; margin-bottom: 2px; }}
  .stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .stat-item {{ display: flex; gap: 4px; }}
  .stat-label {{ color: #888; }}
  .stat-val {{ color: #4fc3f7; font-weight: 600; }}
  #map-container {{ display: flex; height: calc(100vh - 280px); min-height: 400px; }}
  .map-col {{ flex: 1; position: relative; border-right: 1px solid #ccc; }}
  .map-col:last-child {{ border-right: none; }}
  .map-col .map-label {{ position: absolute; top: 6px; left: 50%; transform: translateX(-50%);
                         z-index: 1000; background: rgba(0,0,0,0.7); color: #fff;
                         padding: 3px 10px; border-radius: 4px; font-size: 12px;
                         font-weight: 600; pointer-events: none; white-space: nowrap; }}
  .leaflet-map {{ width: 100%; height: 100%; background: #e8e8e8; }}
  #legends {{ display: flex; justify-content: center; gap: 40px;
              padding: 8px 20px; background: #fff; }}
  #legends img {{ height: 60px; }}
  #table-section {{ padding: 10px 20px 20px; }}
  #table-section h2 {{ font-size: 14px; margin-bottom: 6px; }}
  #comp-table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  #comp-table th, #comp-table td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
  #comp-table th {{ background: #eee; font-weight: 600; text-align: center; }}
  #comp-table td:first-child {{ text-align: left; font-weight: 500; }}
  #comp-table2 {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  #comp-table2 th, #comp-table2 td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
  #comp-table2 th {{ background: #eee; font-weight: 600; text-align: center; }}
  #comp-table2 td:first-child {{ text-align: left; font-weight: 500; }}
  .hover-tooltip {{ font-size: 12px; line-height: 1.5; }}
  .hover-tooltip b {{ font-size: 13px; }}
</style>
</head>
<body>

<div id="header">
  <h1>ΔEHD Explorer — 2040 Projections (SSP2-4.5)</h1>
  <label style="font-size:13px;">ΔT Source:
    <select id="variant-select">
      <optgroup label="ΔT source comparison (ERA5 vs GCM)">
        <option value="era5_local">ERA5 Tmax local × IFS local slope</option>
        <option value="era5_200km">ERA5 Tmax 200km × IFS 200km slope</option>
        <option value="gcm_local">GCM Tmax local × IFS local slope</option>
        <option value="gcm_200km">GCM Tmax 200km × IFS 200km slope</option>
      </optgroup>
      <optgroup label="Direct methods">
        <option value="era5_ehd">ERA5 EHD trend vs CIL GDPCIR</option>
        <option value="cil_gdpcir">CIL GDPCIR (self)</option>
      </optgroup>
      <optgroup label="EHD trend vs applied slopes">
        <option value="v5_vs_v1">EHD trend vs ERA5 Tmax local slope</option>
        <option value="v5_vs_v2">EHD trend vs ERA5 Tmax 200km slope</option>
        <option value="v5_vs_v3">EHD trend vs GCM Tmax local slope</option>
        <option value="v5_vs_v4">EHD trend vs GCM Tmax 200km slope</option>
      </optgroup>
      <optgroup label="GCM raw EHD vs applied slopes">
        <option value="v6_vs_v3">CIL GDPCIR vs GCM Tmax local slope</option>
        <option value="v6_vs_v4">CIL GDPCIR vs GCM Tmax 200km slope</option>
      </optgroup>
      <optgroup label="Key comparison">
        <option value="v2_vs_v6">V2 (ERA5 Tmax 200km) vs V6 (CIL GDPCIR)</option>
      </optgroup>
    </select>
  </label>
</div>

<div id="stats-bar">
  <div class="stat-group">
    <div class="stat-header">Admin1-level mixed (11K ent)</div>
    <div class="stat-row">
      <div class="stat-item"><span class="stat-label">&beta;:</span>
        <span class="stat-val" id="stat-a1l-beta">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">GDP-wt \u0394EHD:</span>
        <span class="stat-val" id="stat-a1l-dehd">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">Impact:</span>
        <span class="stat-val" id="stat-a1l-impact">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">[P5, P95]:</span>
        <span class="stat-val" id="stat-a1l-ci">\u2014</span></div>
    </div>
  </div>
  <div class="stat-group">
    <div class="stat-header">Admin1 pure (2.7K ent)</div>
    <div class="stat-row">
      <div class="stat-item"><span class="stat-label">&beta;:</span>
        <span class="stat-val" id="stat-a1p-beta">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">GDP-wt \u0394EHD:</span>
        <span class="stat-val" id="stat-a1p-dehd">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">Impact:</span>
        <span class="stat-val" id="stat-a1p-impact">\u2014</span></div>
      <div class="stat-item"><span class="stat-label">[P5, P95]:</span>
        <span class="stat-val" id="stat-a1p-ci">\u2014</span></div>
    </div>
  </div>
</div>

<div id="map-container">
  <div class="map-col">
    <div class="map-label" id="label-left">Selected Variant</div>
    <div id="map-left" class="leaflet-map"></div>
  </div>
  <div class="map-col">
    <div class="map-label" id="label-center">GCM Reference</div>
    <div id="map-center" class="leaflet-map"></div>
  </div>
  <div class="map-col">
    <div class="map-label" id="label-right">Difference (Selected \u2212 GCM)</div>
    <div id="map-right" class="leaflet-map"></div>
  </div>
</div>

<div id="legends">
  <div><strong style="font-size:11px;">\u0394EHD</strong><br>
    <img src="data:image/png;base64,{cbar_dehd_b64}" alt="\u0394EHD colorbar"></div>
  <div><strong style="font-size:11px;">Difference</strong><br>
    <img src="data:image/png;base64,{cbar_diff_b64}" alt="Diff colorbar"></div>
</div>

<div id="table-section">
  <h2>Six-Variant Comparison</h2>
  {csv_table_html}
</div>

<script>
    var geodata = {geojson_str};

    var choices = {{
      "era5_local":  {{left: "v1", center: "v3", diff: "d_local",
                      label: "V1: ERA5 Tmax local slope", centerLabel: "V3: GCM local slope"}},
      "era5_200km":  {{left: "v2", center: "v4", diff: "d_200km",
                      label: "V2: ERA5 Tmax 200km slope", centerLabel: "V4: GCM 200km slope"}},
      "gcm_local":   {{left: "v3", center: "v3", diff: null,
                      label: "V3: GCM local slope", centerLabel: "V3: GCM local slope"}},
      "gcm_200km":   {{left: "v4", center: "v4", diff: null,
                      label: "V4: GCM 200km slope", centerLabel: "V4: GCM 200km slope"}},
      "era5_ehd":    {{left: "v5", center: "v6", diff: "d_direct",
                      label: "V5: ERA5 EHD trend", centerLabel: "V6: CIL GDPCIR"}},
      "cil_gdpcir":  {{left: "v6", center: "v6", diff: null,
                      label: "V6: CIL GDPCIR", centerLabel: "V6: CIL GDPCIR"}},
      "v5_vs_v1":    {{left: "v5", center: "v1", diff: "d_v5v1",
                      label: "V5: ERA5 EHD trend", centerLabel: "V1: ERA5 Tmax local slope"}},
      "v5_vs_v2":    {{left: "v5", center: "v2", diff: "d_v5v2",
                      label: "V5: ERA5 EHD trend", centerLabel: "V2: ERA5 Tmax 200km slope"}},
      "v5_vs_v3":    {{left: "v5", center: "v3", diff: "d_v5v3",
                      label: "V5: ERA5 EHD trend", centerLabel: "V3: GCM local slope"}},
      "v5_vs_v4":    {{left: "v5", center: "v4", diff: "d_v5v4",
                      label: "V5: ERA5 EHD trend", centerLabel: "V4: GCM 200km slope"}},
      "v6_vs_v3":    {{left: "v6", center: "v3", diff: "d_v6v3",
                      label: "V6: CIL GDPCIR (raw GCM)", centerLabel: "V3: GCM Tmax local slope"}},
      "v6_vs_v4":    {{left: "v6", center: "v4", diff: "d_v6v4",
                      label: "V6: CIL GDPCIR (raw GCM)", centerLabel: "V4: GCM Tmax 200km slope"}},
      "v2_vs_v6":    {{left: "v2", center: "v6", diff: "d_v2v6",
                      label: "V2: ERA5 Tmax 200km slope", centerLabel: "V6: CIL GDPCIR"}}
    }};

    {stats_js}

    // Beta values per panel for computing hover impact
    {betas_js}

    // Color scales
    var dehdBounds = {dehd_bounds_js};
    var diffBounds = {diff_bounds_js};
    var dehdColors = {dehd_colors_js};
    var diffColors = {diff_colors_js};

    function getColor(val, bounds, colors) {{
      if (val === null || val === undefined || isNaN(val)) return '#cccccc';
      for (var i = 0; i < bounds.length - 1; i++) {{
        if (val < bounds[i+1]) return colors[i];
      }}
      return colors[colors.length - 1];
    }}

    // Current state
    var currentChoice = 'era5_local';
    var layerLeft = null, layerCenter = null, layerRight = null;

    function makeMap(divId) {{
      var map = L.map(divId, {{
        center: [20, 0], zoom: 2,
        minZoom: 1, maxZoom: 6,
        worldCopyJump: true,
        attributionControl: false
      }});
      return map;
    }}

    var mapLeft   = makeMap('map-left');
    var mapCenter = makeMap('map-center');
    var mapRight  = makeMap('map-right');

    mapLeft.sync(mapCenter);
    mapLeft.sync(mapRight);
    mapCenter.sync(mapLeft);
    mapCenter.sync(mapRight);
    mapRight.sync(mapLeft);
    mapRight.sync(mapCenter);

    // GDP pc projections to 2040 with and without heat drag.
    // ΔEHD ramps linearly 0 → dehd over T years (2025–2040).
    // Without heat: GDP_t = GDP_0 × (1+g)^T
    // With heat:    GDP_t = GDP_0 × Π(1 + g + β·(t/T)·dehd)
    // Returns {{ pctChange, withHeat, noHeat }} where pctChange = (with/without - 1)
    var T_YEARS = 16; // 2025–2040
    function gdpProjection(beta, dehd, g) {{
      if (g === null || g === undefined || isNaN(g)) g = 0;
      var noHeat = 1.0, withHeat = 1.0;
      for (var t = 1; t <= T_YEARS; t++) {{
        noHeat   *= (1.0 + g);
        withHeat *= (1.0 + g + beta * (t / T_YEARS) * dehd);
      }}
      return {{ pctChange: (withHeat / noHeat - 1.0), withHeat: withHeat, noHeat: noHeat }};
    }}

    function makeTooltipContent(props, varKey, panelKey, isDiff, leftKey, centerKey) {{
      var val = props[varKey];
      var beta = betas[panelKey];
      var g = props.g;
      var name = props.Country || props.iso3;
      var gPct = (g !== null && !isNaN(g)) ? (g * 100).toFixed(1) : 'n/a';
      if (val === null || val === undefined || isNaN(val)) {{
        return '<div class="hover-tooltip"><b>' + name + '</b> (' + props.iso3 + ')<br>No data</div>';
      }}
      if (isDiff) {{
        var leftVal = props[leftKey];
        var centerVal = props[centerKey];
        var leftP  = (leftVal != null && !isNaN(leftVal)) ? gdpProjection(beta, leftVal, g) : null;
        var centerP = (centerVal != null && !isNaN(centerVal)) ? gdpProjection(beta, centerVal, g) : null;
        var leftPct  = leftP  ? (leftP.pctChange * 100).toFixed(2) : 'n/a';
        var centerPct = centerP ? (centerP.pctChange * 100).toFixed(2) : 'n/a';
        var diffPct = (leftP && centerP) ? ((leftP.pctChange - centerP.pctChange) * 100).toFixed(2) : 'n/a';
        return '<div class="hover-tooltip">' +
          '<b>' + name + '</b> (' + props.iso3 + ', baseline g=' + gPct + '%)' +
          '<br>\u0394EHD diff: ' + val.toFixed(4) +
          '<br>GDP pc \u0394 vs no-heat (left): ' + leftPct + '%' +
          '<br>GDP pc \u0394 vs no-heat (right): ' + centerPct + '%' +
          '<br><b>Difference: ' + diffPct + '%</b>' +
          '</div>';
      }}
      var proj = gdpProjection(beta, val, g);
      var pctStr = (proj.pctChange * 100).toFixed(2);
      var noHeatPct = ((proj.noHeat - 1) * 100).toFixed(1);
      var withHeatPct = ((proj.withHeat - 1) * 100).toFixed(1);
      return '<div class="hover-tooltip">' +
        '<b>' + name + '</b> (' + props.iso3 + ', baseline g=' + gPct + '%)' +
        '<br>\u0394EHD (2040): ' + val.toFixed(4) +
        '<br>GDP pc growth w/o heat: +' + noHeatPct + '%' +
        '<br>GDP pc growth w/ heat: +' + withHeatPct + '%' +
        '<br><b>Heat impact on GDP pc: ' + pctStr + '%</b>' +
        '</div>';
    }}

    function createLayer(map, varKey, bounds, colors, panelKey, isDiff, leftKey, centerKey) {{
      return L.geoJSON(geodata, {{
        style: function(feature) {{
          var val = feature.properties[varKey];
          return {{
            fillColor: getColor(val, bounds, colors),
            weight: 0.5,
            color: '#444',
            fillOpacity: 0.85
          }};
        }},
        onEachFeature: function(feature, layer) {{
          layer.on('mouseover', function(e) {{
            this.setStyle({{ weight: 2, color: '#000' }});
            this.bringToFront();
          }});
          layer.on('mouseout', function(e) {{
            this.setStyle({{ weight: 0.5, color: '#444' }});
          }});
          layer.bindTooltip(function() {{
            return makeTooltipContent(feature.properties, varKey, panelKey, isDiff, leftKey, centerKey);
          }}, {{ sticky: true }});
        }}
      }}).addTo(map);
    }}

    function updateMaps(choice) {{
      var c = choices[choice];
      if (!c) return;
      currentChoice = choice;

      if (layerLeft) mapLeft.removeLayer(layerLeft);
      if (layerCenter) mapCenter.removeLayer(layerCenter);
      if (layerRight) mapRight.removeLayer(layerRight);

      layerLeft = createLayer(mapLeft, c.left, dehdBounds, dehdColors, 'admin1pure', false);
      layerCenter = createLayer(mapCenter, c.center, dehdBounds, dehdColors, 'admin1pure', false);

      if (c.diff) {{
        layerRight = createLayer(mapRight, c.diff, diffBounds, diffColors, 'admin1pure', true, c.left, c.center);
      }} else {{
        // No difference — show empty grey layer
        layerRight = L.geoJSON(geodata, {{
          style: function() {{ return {{ fillColor: '#cccccc', weight: 0.5, color: '#444', fillOpacity: 0.4 }}; }},
          onEachFeature: function(feature, layer) {{
            layer.bindTooltip(function() {{
              return '<div class="hover-tooltip"><b>' +
                (feature.properties.Country || feature.properties.iso3) +
                '</b><br>Identical (no difference)</div>';
            }}, {{ sticky: true }});
          }}
        }}).addTo(mapRight);
      }}

      document.getElementById('label-left').textContent = c.label;
      document.getElementById('label-center').textContent = c.centerLabel;
      document.getElementById('label-right').textContent =
        c.diff ? 'Difference (Selected \u2212 Reference)' : 'Difference (identical)';

      var leftKey = c.left;
      var s1 = stats["admin1level"][leftKey];
      if (s1) {{
        document.getElementById('stat-a1l-beta').textContent   = s1.beta;
        document.getElementById('stat-a1l-dehd').textContent   = s1.gdpDehd;
        document.getElementById('stat-a1l-impact').textContent = s1.gdpImpact;
        document.getElementById('stat-a1l-ci').textContent     = '[' + s1.p5 + ', ' + s1.p95 + ']';
      }}
      var s2 = stats["admin1pure"][leftKey];
      if (s2) {{
        document.getElementById('stat-a1p-beta').textContent   = s2.beta;
        document.getElementById('stat-a1p-dehd').textContent   = s2.gdpDehd;
        document.getElementById('stat-a1p-impact').textContent = s2.gdpImpact;
        document.getElementById('stat-a1p-ci').textContent     = '[' + s2.p5 + ', ' + s2.p95 + ']';
      }}
    }}

    document.getElementById('variant-select').addEventListener('change', function() {{
      updateMaps(this.value);
    }});

    updateMaps('era5_local');

    setTimeout(function() {{
      mapLeft.invalidateSize();
      mapCenter.invalidateSize();
      mapRight.invalidateSize();
    }}, 200);
</script>
</body>
</html>"""
    return html


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    wall_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Generating interactive ΔEHD explorer HTML")
    print("=" * 65)

    # ── Compute all 6 ΔEHD grids ────────────────────────────────────────
    rr_local, rr_200km, ifs_lats, ifs_lons = load_ifs_slopes()
    ehd_base_025, ehd_stack_01 = load_ehd_baseline_025(ifs_lats, ifs_lons)
    dt_era5_local, dt_era5_200km = compute_era5_dt_025(ifs_lats, ifs_lons)
    dt_gcm_local, dt_gcm_200km = compute_gcm_dt_025(ifs_lats, ifs_lons)

    print("\nComputing ΔEHD variants 1-4 (slope-based)...")
    dehd_v1 = compute_slope_dehd(ehd_base_025, rr_local, dt_era5_local)
    dehd_v2 = compute_slope_dehd(ehd_base_025, rr_200km, dt_era5_200km)
    dehd_v3 = compute_slope_dehd(ehd_base_025, rr_local, dt_gcm_local)
    dehd_v4 = compute_slope_dehd(ehd_base_025, rr_200km, dt_gcm_200km)

    dehd_v5 = compute_era5_ehd_trend_025(ehd_stack_01, ifs_lats, ifs_lons)
    del ehd_stack_01
    dehd_v6 = compute_cil_dehd_025(ifs_lats, ifs_lons)

    grids = {"v1": dehd_v1, "v2": dehd_v2, "v3": dehd_v3,
             "v4": dehd_v4, "v5": dehd_v5, "v6": dehd_v6}

    # ── Difference grids ─────────────────────────────────────────────────
    print("\nComputing difference grids...")
    diff_local  = (dehd_v1 - dehd_v3).astype(np.float32)   # V1 - V3
    diff_200km  = (dehd_v2 - dehd_v4).astype(np.float32)   # V2 - V4
    diff_direct = (dehd_v5 - dehd_v6).astype(np.float32)   # V5 - V6
    # V5 (EHD trend) vs slope-based variants
    diff_v5_v1  = (dehd_v5 - dehd_v1).astype(np.float32)   # V5 - V1
    diff_v5_v2  = (dehd_v5 - dehd_v2).astype(np.float32)   # V5 - V2
    diff_v5_v3  = (dehd_v5 - dehd_v3).astype(np.float32)   # V5 - V3
    diff_v5_v4  = (dehd_v5 - dehd_v4).astype(np.float32)   # V5 - V4
    # V6 (CIL raw GCM EHD) vs slope-applied GCM variants
    diff_v6_v3  = (dehd_v6 - dehd_v3).astype(np.float32)   # V6 - V3
    diff_v6_v4  = (dehd_v6 - dehd_v4).astype(np.float32)   # V6 - V4
    # V2 vs V6 (key comparison)
    diff_v2_v6  = (dehd_v2 - dehd_v6).astype(np.float32)   # V2 - V6
    diffs = {
        "d_local": diff_local, "d_200km": diff_200km, "d_direct": diff_direct,
        "d_v5v1": diff_v5_v1, "d_v5v2": diff_v5_v2,
        "d_v5v3": diff_v5_v3, "d_v5v4": diff_v5_v4,
        "d_v6v3": diff_v6_v3, "d_v6v4": diff_v6_v4,
        "d_v2v6": diff_v2_v6,
    }

    for k, d in diffs.items():
        print(f"  {k}: mean={np.nanmean(d):.5f}, "
              f"min={np.nanmin(d[np.isfinite(d)]):.5f}, "
              f"max={np.nanmax(d[np.isfinite(d)]):.5f}")

    # ── Aggregate ALL grids to admin1 entities for choropleth maps ────────
    settlement = load_settlement()
    variant_keys = ["v1", "v2", "v3", "v4", "v5", "v6"]

    # Pre-interpolate all grids (variants + diffs) to 0.1°
    print("\nInterpolating to 0.1°...")
    grids_01 = {}
    for k in variant_keys:
        grids_01[k] = interp_025_to_01(grids[k], ifs_lats, ifs_lons)
    diffs_01 = {}
    for k, d in diffs.items():
        diffs_01[k] = interp_025_to_01(d, ifs_lats, ifs_lons)

    # Load admin1 polygons and raster for aggregation
    print("\nLoading admin1 polygons for choropleth...")
    admin1_gdf = gpd.read_file(str(ADMIN1_GPKG))
    combined_raster, entities, n_entities = load_admin1_raster()

    # Aggregate variant grids to admin1 entities
    print("Aggregating variant grids to admin1 entities...")
    entity_vals = {}
    for k in variant_keys:
        entity_vals[k] = agg_to_entities(
            grids_01[k], combined_raster, settlement, n_entities
        )
        nv = np.isfinite(entity_vals[k]).sum()
        print(f"  {k}: {nv:,} valid, mean={np.nanmean(entity_vals[k]):.4f}")

    # Aggregate difference grids to admin1 entities
    print("Aggregating difference grids to admin1 entities...")
    entity_diffs = {}
    for k in diffs_01:
        entity_diffs[k] = agg_to_entities(
            diffs_01[k], combined_raster, settlement, n_entities
        )
        nv = np.isfinite(entity_diffs[k]).sum()
        print(f"  {k}: {nv:,} valid, mean={np.nanmean(entity_diffs[k]):.4f}")

    # Join entity values to admin1 polygons, then dissolve to country level
    print("Joining entity values to admin1 polygons...")
    ent_df = entities[["GID_nmbr", "entity_idx"]].copy()
    val_cols = list(variant_keys) + list(entity_diffs.keys())
    for k in variant_keys:
        ent_df[k] = entity_vals[k]
    for k in entity_diffs:
        ent_df[k] = entity_diffs[k]

    admin1_merged = admin1_gdf.merge(ent_df, on="GID_nmbr", how="inner")
    print(f"  Joined: {len(admin1_merged):,} admin1 polygons with data")

    # Dissolve to country level — settlement-weighted mean per country
    print("Dissolving to country level (mean per iso3)...")
    country_vals = admin1_merged.groupby("iso3")[val_cols].mean()
    country_geom = admin1_merged.dissolve(by="iso3").geometry
    choropleth_gdf = gpd.GeoDataFrame(
        country_vals.join(country_geom), geometry="geometry"
    ).reset_index()
    print(f"  {len(choropleth_gdf):,} countries with data")

    # ── Build simplified GeoJSON with all variant values as properties ────
    print("\nSimplifying geometries for GeoJSON...")
    choropleth_gdf = choropleth_gdf.copy()
    choropleth_gdf["geometry"] = choropleth_gdf.geometry.simplify(
        0.1, preserve_topology=True
    )

    # Add Country name from admin1 gpkg
    country_names = admin1_gdf.drop_duplicates("iso3").set_index("iso3")["Country"]
    choropleth_gdf["Country"] = choropleth_gdf["iso3"].map(country_names)

    # Compute country-level baseline GDP pc annual growth rate (2010-2022)
    print("Computing country-level baseline GDP pc growth rates (2010-2022)...")
    gdp_years = [str(y) for y in range(2010, 2023)]
    gdp_years = [y for y in gdp_years if y in admin1_gdf.columns]
    country_gdp = admin1_gdf.groupby("iso3")[gdp_years].mean()
    g0 = country_gdp[gdp_years[0]].values.astype(float)
    g1 = country_gdp[gdp_years[-1]].values.astype(float)
    n_yrs = len(gdp_years) - 1
    valid_g = (g0 > 0) & (g1 > 0) & np.isfinite(g0) & np.isfinite(g1)
    annual_g = pd.Series(np.nan, index=country_gdp.index)
    annual_g[valid_g] = (g1[valid_g] / g0[valid_g]) ** (1.0 / n_yrs) - 1.0
    choropleth_gdf["g"] = choropleth_gdf["iso3"].map(annual_g).round(6)
    print(f"  {annual_g.notna().sum()} countries, "
          f"mean={annual_g.mean()*100:.2f}%, median={annual_g.median()*100:.2f}%")

    # Round numeric properties for smaller JSON
    for col in val_cols:
        choropleth_gdf[col] = choropleth_gdf[col].round(6)

    # Keep only needed columns
    keep_cols = ["iso3", "Country", "g", "geometry"] + val_cols
    geojson_str = choropleth_gdf[keep_cols].to_json()
    print(f"  GeoJSON: {len(geojson_str)/1024/1024:.1f} MB, "
          f"{len(choropleth_gdf)} countries")

    # Precompute colormap hex values
    dehd_colors = get_bin_colors_hex(DEHD_BOUNDS, "plasma")
    diff_colors = get_bin_colors_hex(DIFF_BOUNDS, "RdBu_r")

    print("  Rendering colorbars...")
    cbar_dehd_b64 = to_b64(render_colorbar_png(
        DEHD_BOUNDS, "plasma", u"ΔEHD (fraction)"))
    cbar_diff_b64 = to_b64(render_colorbar_png(
        DIFF_BOUNDS, "RdBu_r", u"Difference (ERA5 − GCM)"))

    N_BOOT = 1000
    np.random.seed(42)
    panel_stats = {}

    for panel_name, pspec in PANELS.items():
        print(f"\nComputing stats for {panel_name}...")

        # Load beta + SE
        regr = pd.read_csv(str(pspec["regr_file"]))
        row = regr[
            (regr["spec"] == "linear_ctrl_baseline") &
            (regr["variable"] == "heat_freq_weighted") &
            (regr["se_type"] == "conley_500km")
        ].iloc[0]
        beta_hat = float(row["beta"])
        beta_se = float(row["se"])
        betas = np.random.normal(beta_hat, beta_se, size=N_BOOT)
        print(f"  beta={beta_hat:.6f}, SE={beta_se:.6f}")

        # Load raster + entities (reuse admin1pure from choropleth step)
        if pspec["raster_loader"] == "mixed":
            p_raster, p_entities, p_n_entities = load_mixed_raster()
        else:
            p_raster, p_entities, p_n_entities = combined_raster, entities, n_entities

        # GDP weights
        panel_df = pd.read_parquet(str(pspec["panel_file"]))
        gdp_mean = panel_df.groupby("GID_2")["gdp_per_capita"].mean()
        ent = p_entities.copy()
        ent["gdp_pc"] = ent["GID_2"].map(gdp_mean).astype(np.float64)

        rows = []
        for k in variant_keys:
            # Reuse admin1pure entity vals; recompute for mixed
            if pspec["raster_loader"] == "mixed":
                ent_v = agg_to_entities(grids_01[k], p_raster,
                                        settlement, p_n_entities)
            else:
                ent_v = entity_vals[k]

            # GDP-weighted ΔEHD
            gdp_w = ent["gdp_pc"].values.copy()
            valid = np.isfinite(ent_v) & np.isfinite(gdp_w) & (gdp_w > 0)
            dehd_v = ent_v[valid]
            gdp_v = gdp_w[valid]
            gdp_v_norm = gdp_v / gdp_v.sum()
            gdp_wt_dehd = float(np.sum(dehd_v * gdp_v_norm))

            # Bootstrap impact distribution
            impacts = betas * gdp_wt_dehd

            rows.append({
                "key": k,
                "beta": f"{beta_hat:.4f}",
                "gdp_dehd":   f"{gdp_wt_dehd:.4f}",
                "gdp_impact": f"{np.mean(impacts):.6f}",
                "sett_dehd":  f"{float(np.nanmean(ent_v)):.4f}",
                "sett_impact": f"{beta_hat * float(np.nanmean(ent_v)):.6f}",
                "p5":  f"{np.percentile(impacts, 5):.6f}",
                "p95": f"{np.percentile(impacts, 95):.6f}",
            })
            print(f"  {k}: GDP-wt ΔEHD={gdp_wt_dehd:.4f}, "
                  f"impact={np.mean(impacts):.6f} "
                  f"[{np.percentile(impacts,5):.6f}, {np.percentile(impacts,95):.6f}]")

        panel_stats[panel_name] = rows

    # ── Compute GDP-weighted compounded impact per variant per panel ──────
    # Uses country-level ΔEHD and growth rates from choropleth_gdf
    T_COMPOUND = 16  # 2025-2040
    print("\nComputing compounded GDP pc impacts (2025-2040)...")

    # Get country-level GDP pc for weighting (mean across admin1 within country)
    country_gdp_pc = admin1_merged.groupby("iso3")[gdp_years[-1]].mean()

    compound_stats = {}
    for panel_name, pspec in PANELS.items():
        regr = pd.read_csv(str(pspec["regr_file"]))
        row = regr[
            (regr["spec"] == "linear_ctrl_baseline") &
            (regr["variable"] == "heat_freq_weighted") &
            (regr["se_type"] == "conley_500km")
        ].iloc[0]
        beta_hat = float(row["beta"])

        compound_stats[panel_name] = {}
        for k in variant_keys:
            cdf = choropleth_gdf[["iso3", k, "g"]].copy()
            cdf["gdp_pc"] = cdf["iso3"].map(country_gdp_pc)
            cdf = cdf.dropna(subset=[k, "g", "gdp_pc"])
            cdf = cdf[cdf["gdp_pc"] > 0]

            # Per-country compounded impact: Π(1+g+β·(t/T)·ΔEHD) / Π(1+g) − 1
            dehd_arr = cdf[k].values
            g_arr = cdf["g"].values
            gdp_arr = cdf["gdp_pc"].values

            pct_changes = np.zeros(len(cdf))
            for i in range(len(cdf)):
                no_heat = 1.0
                with_heat = 1.0
                for t in range(1, T_COMPOUND + 1):
                    no_heat *= (1.0 + g_arr[i])
                    with_heat *= (1.0 + g_arr[i] + beta_hat * (t / T_COMPOUND) * dehd_arr[i])
                pct_changes[i] = with_heat / no_heat - 1.0

            # GDP-weighted mean
            gdp_norm = gdp_arr / gdp_arr.sum()
            wt_compound = float(np.sum(pct_changes * gdp_norm))
            compound_stats[panel_name][k] = wt_compound
            print(f"  {panel_name} {k}: GDP-wt compounded impact = {wt_compound*100:.3f}%")

    # Add compound impact to panel_stats rows
    for panel_name in panel_stats:
        for row in panel_stats[panel_name]:
            k = row["key"]
            row["compound"] = f"{compound_stats[panel_name][k]*100:.3f}"

    # ── Build comparison table from both panels ──────────────────────────
    variant_labels = {
        "v1": "V1: ERA5 Tmax local slope",
        "v2": "V2: ERA5 Tmax 200km slope",
        "v3": "V3: GCM Tmax local slope",
        "v4": "V4: GCM Tmax 200km slope",
        "v5": "V5: ERA5 EHD trend",
        "v6": "V6: CIL GDPCIR",
    }
    table_rows = []
    table_rows.append(
        "<thead><tr>"
        "<th>Variant</th>"
        "<th colspan='4' style='background:#e8e0f0;'>Admin1-level mixed (&beta;=&minus;0.055)</th>"
        "<th colspan='4' style='background:#e0e8f0;'>Admin1 pure (&beta;=&minus;0.039)</th>"
        "</tr><tr>"
        "<th></th>"
        "<th style='background:#e8e0f0;'>GDP-wt \u0394EHD</th>"
        "<th style='background:#e8e0f0;'>Annual (pp)</th>"
        "<th style='background:#e8e0f0;'>[P5, P95]</th>"
        "<th style='background:#e8e0f0;'>Cum. 2040 (%)</th>"
        "<th style='background:#e0e8f0;'>GDP-wt \u0394EHD</th>"
        "<th style='background:#e0e8f0;'>Annual (pp)</th>"
        "<th style='background:#e0e8f0;'>[P5, P95]</th>"
        "<th style='background:#e0e8f0;'>Cum. 2040 (%)</th>"
        "</tr></thead>"
    )
    table_rows.append("<tbody>")
    for k in variant_keys:
        r1 = next(r for r in panel_stats["admin1level"] if r["key"] == k)
        r2 = next(r for r in panel_stats["admin1pure"] if r["key"] == k)
        table_rows.append(
            f"<tr>"
            f"<td>{variant_labels[k]}</td>"
            f"<td>{r1['gdp_dehd']}</td>"
            f"<td>{r1['gdp_impact']}</td>"
            f"<td>[{r1['p5']}, {r1['p95']}]</td>"
            f"<td>{r1['compound']}</td>"
            f"<td>{r2['gdp_dehd']}</td>"
            f"<td>{r2['gdp_impact']}</td>"
            f"<td>[{r2['p5']}, {r2['p95']}]</td>"
            f"<td>{r2['compound']}</td>"
            f"</tr>"
        )
    table_rows.append("</tbody>")
    csv_table_html = '<table id="comp-table">' + "\n".join(table_rows) + "</table>"

    # ── Build second table: raw ΔEHD × cumulative impact across all models ──
    print("\nBuilding raw ΔEHD × cumulative impact table...")
    # For each variant, compute per-country ΔEHD and compound impact for both panels,
    # then show global GDP-weighted stats
    t2_rows = []
    t2_rows.append(
        "<thead><tr>"
        "<th rowspan='2'>Variant</th>"
        "<th colspan='2'>Raw \u0394EHD</th>"
        "<th colspan='2'>Cum. GDP pc impact by 2040</th>"
        "</tr><tr>"
        "<th>GDP-wt mean</th>"
        "<th>Unwt. mean</th>"
        "<th style='background:#e8e0f0;'>Mixed (\u03b2=\u22120.055)</th>"
        "<th style='background:#e0e8f0;'>Pure (\u03b2=\u22120.039)</th>"
        "</tr></thead>"
    )
    t2_rows.append("<tbody>")

    # Unweighted country mean ΔEHD
    for k in variant_keys:
        cdf = choropleth_gdf[["iso3", k]].dropna(subset=[k])
        unwt_mean = float(cdf[k].mean())
        r1 = next(r for r in panel_stats["admin1level"] if r["key"] == k)
        r2 = next(r for r in panel_stats["admin1pure"] if r["key"] == k)
        t2_rows.append(
            f"<tr>"
            f"<td>{variant_labels[k]}</td>"
            f"<td>{r1['gdp_dehd']}</td>"
            f"<td>{unwt_mean:.4f}</td>"
            f"<td style='background:#f8f4fc;'>{r1['compound']}%</td>"
            f"<td style='background:#f4f8fc;'>{r2['compound']}%</td>"
            f"</tr>"
        )
    t2_rows.append("</tbody>")
    csv_table_html += (
        '\n<h2 style="margin-top:16px;">'
        'Raw \u0394EHD &amp; Cumulative GDP pc Impact by 2040 (linear ramp, compounded)</h2>\n'
        '<table id="comp-table2" style="border-collapse:collapse;width:100%;font-size:12px;">'
        + "\n".join(t2_rows) + "</table>"
        + '<p style="font-size:11px;color:#666;margin-top:4px;">'
        '\u0394EHD ramps linearly from 0 (2025) to projected value (2040). '
        'Each year: growth = baseline_g + \u03b2\u00b7(\u0394EHD\u2099). '
        'Cum. impact = \u03a0(1+g+\u03b2\u00b7\u0394EHD\u2099) / \u03a0(1+g) \u2212 1, '
        'GDP-weighted across countries.</p>'
    )

    # ── Per-GCM ΔEHD bar chart: raw EHD from each model → compounded impact ──
    print("\nComputing per-GCM ΔEHD and compounded impacts...")
    models = sorted(CIL_ECS.keys())
    available_models = [m for m in models
                        if (CIL_EHD_DIR / m / "ehd_historical.nc").exists()
                        and (CIL_EHD_DIR / m / "ehd_ssp245.nc").exists()]
    ecs_weights = compute_ecs_weights(available_models, CIL_ECS)

    # Get country growth rates and GDP weights from choropleth_gdf
    cdf_base = choropleth_gdf[["iso3", "g"]].copy()
    cdf_base["gdp_pc"] = cdf_base["iso3"].map(country_gdp_pc)

    # Get beta for both panels
    panel_betas = {}
    for panel_name, pspec in PANELS.items():
        regr = pd.read_csv(str(pspec["regr_file"]))
        row = regr[
            (regr["spec"] == "linear_ctrl_baseline") &
            (regr["variable"] == "heat_freq_weighted") &
            (regr["se_type"] == "conley_500km")
        ].iloc[0]
        panel_betas[panel_name] = float(row["beta"])

    gcm_results = []
    for mi, model in enumerate(available_models):
        print(f"  [{mi+1}/{len(available_models)}] {model}...", end="", flush=True)
        t0 = time.time()

        # Load and compute per-model ΔEHD (same logic as compute_cil_dehd_025)
        ds_hist = xr.open_dataset(str(CIL_EHD_DIR / model / "ehd_historical.nc"))
        lats_src = ds_hist.lat.values
        lons_src = ds_hist.lon.values
        yrs_h = ds_hist.year.values
        mask_h = (yrs_h >= 2000) & (yrs_h <= 2014)
        hist_mean = ds_hist["ehd"].values[mask_h].mean(axis=0) if mask_h.any() else None
        ds_hist.close()

        ds_ssp = xr.open_dataset(str(CIL_EHD_DIR / model / "ehd_ssp245.nc"))
        yrs_s = ds_ssp.year.values
        ehd_ssp = ds_ssp["ehd"].values
        mask_bl = (yrs_s >= 2015) & (yrs_s <= 2024)
        bl_parts = []
        if hist_mean is not None:
            bl_parts.append(hist_mean)
        if mask_bl.any():
            bl_parts.append(ehd_ssp[mask_bl].mean(axis=0))
        baseline = np.nanmean(np.stack(bl_parts, axis=0), axis=0)

        mask_proj = (yrs_s >= PROJ_WINDOW[0]) & (yrs_s <= min(PROJ_WINDOW[1], int(yrs_s.max())))
        proj_mean = ehd_ssp[mask_proj].mean(axis=0)
        ds_ssp.close()

        dehd_model = (proj_mean - baseline).astype(np.float32)
        dehd_ifs = regrid_gcm_to_ifs(dehd_model, lats_src, lons_src, ifs_lats, ifs_lons)

        # Compute per-GCM ΔTmax (global area-weighted mean)
        bl_years = [2003, 2004, 2005, 2006, 2007]
        proj_years_t = [2038, 2039, 2040, 2041, 2042]
        dt_model = np.nan
        try:
            bl_tmax, proj_tmax = [], []
            for yr in bl_years:
                res = cmip6_summer_mean_year(model, "historical", yr)
                if res is not None:
                    bl_tmax.append(res[2])
                    tmax_lats = res[0]
            for yr in proj_years_t:
                res = cmip6_summer_mean_year(model, "ssp245", yr)
                if res is not None:
                    proj_tmax.append(res[2])
            if bl_tmax and proj_tmax:
                bl_mean = np.nanmean(np.stack(bl_tmax), axis=0)
                proj_mean_t = np.nanmean(np.stack(proj_tmax), axis=0)
                dt_grid = proj_mean_t - bl_mean
                # Area-weighted global mean (cos(lat) weighting)
                cos_w = np.cos(np.deg2rad(tmax_lats))[:, None]
                cos_w = np.broadcast_to(cos_w, dt_grid.shape)
                valid_dt = np.isfinite(dt_grid)
                dt_model = float(np.sum(dt_grid[valid_dt] * cos_w[valid_dt]) /
                                 np.sum(cos_w[valid_dt]))
        except Exception:
            pass

        # Interpolate to 0.1° and aggregate to admin1 → country
        dehd_01 = interp_025_to_01(dehd_ifs, ifs_lats, ifs_lons)
        ent_dehd = agg_to_entities(dehd_01, combined_raster, settlement, n_entities)
        ent_df_m = entities[["GID_nmbr", "entity_idx"]].copy()
        ent_df_m["dehd"] = ent_dehd
        adm1_m = admin1_gdf[["GID_nmbr", "iso3"]].merge(ent_df_m, on="GID_nmbr", how="inner")
        country_dehd = adm1_m.groupby("iso3")["dehd"].mean()

        # Merge with growth rates and GDP weights
        cdf_m = cdf_base.copy()
        cdf_m["dehd"] = cdf_m["iso3"].map(country_dehd)
        cdf_m = cdf_m.dropna(subset=["dehd", "g", "gdp_pc"])
        cdf_m = cdf_m[cdf_m["gdp_pc"] > 0]

        # GDP-weighted mean ΔEHD
        gdp_norm = cdf_m["gdp_pc"].values / cdf_m["gdp_pc"].values.sum()
        gdp_wt_dehd = float(np.sum(cdf_m["dehd"].values * gdp_norm))

        # Compounded impacts per panel
        row_result = {"model": model, "ecs": CIL_ECS[model],
                      "weight": ecs_weights[model], "gdp_wt_dehd": gdp_wt_dehd,
                      "dt_tmax": dt_model}
        for pname, beta in panel_betas.items():
            pct_changes = np.zeros(len(cdf_m))
            dehd_arr = cdf_m["dehd"].values
            g_arr = cdf_m["g"].values
            for i in range(len(cdf_m)):
                no_heat = with_heat = 1.0
                for t in range(1, T_COMPOUND + 1):
                    no_heat *= (1.0 + g_arr[i])
                    with_heat *= (1.0 + g_arr[i] + beta * (t / T_COMPOUND) * dehd_arr[i])
                pct_changes[i] = with_heat / no_heat - 1.0
            wt_compound = float(np.sum(pct_changes * gdp_norm))
            row_result[f"compound_{pname}"] = wt_compound

        gcm_results.append(row_result)
        print(f" ΔEHD={gdp_wt_dehd:.4f}, "
              f"cum_mixed={row_result['compound_admin1level']*100:.2f}%, "
              f"cum_pure={row_result['compound_admin1pure']*100:.2f}%  "
              f"({time.time()-t0:.0f}s)")

    # Sort by compounded impact (most negative first)
    gcm_results.sort(key=lambda r: r["compound_admin1pure"])

    # Render bar chart
    print("  Rendering per-GCM bar chart...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor("white")

    model_names = [r["model"] for r in gcm_results]
    y_pos = np.arange(len(model_names))

    for ax_i, (pname, plabel) in enumerate([
        ("admin1level", u"Mixed (\u03b2=\u22120.055)"),
        ("admin1pure",  u"Pure (\u03b2=\u22120.039)")
    ]):
        vals = [r[f"compound_{pname}"] * 100 for r in gcm_results]
        colors = ["#d32f2f" if v < 0 else "#388e3c" for v in vals]
        bars = axes[ax_i].barh(y_pos, vals, color=colors, edgecolor="#333", linewidth=0.5, height=0.7)
        axes[ax_i].set_xlabel("Cumulative GDP pc impact by 2040 (%)", fontsize=10)
        axes[ax_i].set_title(plabel, fontsize=11, fontweight="bold")
        axes[ax_i].axvline(0, color="#333", linewidth=0.8)
        axes[ax_i].set_yticks(y_pos)
        if ax_i == 0:
            labels = [f"{r['model']} (ECS={r['ecs']:.1f}, w={r['weight']:.2f})" for r in gcm_results]
            axes[ax_i].set_yticklabels(labels, fontsize=8)
        axes[ax_i].tick_params(axis="x", labelsize=9)
        axes[ax_i].invert_yaxis()

        # Add value labels
        for i, v in enumerate(vals):
            axes[ax_i].text(v - 0.15 if v < 0 else v + 0.05, i, f"{v:.2f}%",
                            va="center", fontsize=7, color="#333")

        # Add ensemble weighted mean line
        ens_val = compound_stats[pname]["v6"] * 100
        axes[ax_i].axvline(ens_val, color="#1565c0", linewidth=2, linestyle="--", alpha=0.8)
        axes[ax_i].text(ens_val, len(model_names) - 0.5,
                        f"  Ensemble: {ens_val:.2f}%", color="#1565c0",
                        fontsize=8, fontweight="bold", va="top")

    fig.suptitle(u"Per-GCM Compounded GDP pc Impact by 2040 (SSP2-4.5, linear \u0394EHD ramp)",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    gcm_chart_b64 = to_b64(buf.read())
    print(f"  Chart: {len(buf.getvalue())/1024:.0f} KB")

    # Render ΔTmax and ΔEHD distribution histograms across GCMs
    print("  Rendering ΔTmax / ΔEHD distribution histograms...")
    dt_vals = [r["dt_tmax"] for r in gcm_results if np.isfinite(r["dt_tmax"])]
    dehd_vals_gcm = [r["gdp_wt_dehd"] for r in gcm_results]
    dt_labels = [r["model"] for r in gcm_results if np.isfinite(r["dt_tmax"])]
    dehd_labels = [r["model"] for r in gcm_results]

    # Sort both by value for cleaner bar charts
    dt_order = np.argsort(dt_vals)
    dehd_order = np.argsort(dehd_vals_gcm)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.patch.set_facecolor("white")

    # ΔTmax bar chart
    dt_sorted = [dt_vals[i] for i in dt_order]
    dt_lbl_sorted = [dt_labels[i] for i in dt_order]
    ecs_sorted = [next(r["ecs"] for r in gcm_results if r["model"] == m) for m in dt_lbl_sorted]
    bar_colors_dt = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(dt_sorted)))
    ax1.barh(range(len(dt_sorted)), dt_sorted, color=bar_colors_dt,
             edgecolor="#333", linewidth=0.5, height=0.7)
    ax1.set_yticks(range(len(dt_sorted)))
    ax1.set_yticklabels([f"{m} (ECS={e:.1f})" for m, e in zip(dt_lbl_sorted, ecs_sorted)],
                        fontsize=8)
    ax1.set_xlabel(u"Global mean \u0394Tmax (\u00b0C, 2040 vs baseline)", fontsize=10)
    ax1.set_title(u"Summer \u0394Tmax by GCM", fontsize=11, fontweight="bold")
    for i, v in enumerate(dt_sorted):
        ax1.text(v + 0.02, i, f"{v:.2f}\u00b0C", va="center", fontsize=7)
    ax1.invert_yaxis()

    # ΔEHD bar chart
    dehd_sorted = [dehd_vals_gcm[i] for i in dehd_order]
    dehd_lbl_sorted = [dehd_labels[i] for i in dehd_order]
    bar_colors_ehd = plt.cm.plasma(np.linspace(0.15, 0.85, len(dehd_sorted)))
    ax2.barh(range(len(dehd_sorted)), dehd_sorted, color=bar_colors_ehd,
             edgecolor="#333", linewidth=0.5, height=0.7)
    ax2.set_yticks(range(len(dehd_sorted)))
    ax2.set_yticklabels(dehd_lbl_sorted, fontsize=8)
    ax2.set_xlabel(u"GDP-weighted mean \u0394EHD (fraction)", fontsize=10)
    ax2.set_title(u"\u0394EHD by GCM (raw, SSP2-4.5 2040)", fontsize=11, fontweight="bold")
    for i, v in enumerate(dehd_sorted):
        ax2.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=7)
    ax2.invert_yaxis()

    fig2.suptitle(u"GCM Spread: \u0394Tmax and \u0394EHD (SSP2-4.5, 2040 vs baseline)",
                  fontsize=12, fontweight="bold", y=0.98)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    buf2.seek(0)
    dist_chart_b64 = to_b64(buf2.read())
    print(f"  Distribution chart: {len(buf2.getvalue())/1024:.0f} KB")

    # Append both charts to table HTML
    csv_table_html += (
        '\n<h2 style="margin-top:20px;">GCM Spread: \u0394Tmax and \u0394EHD</h2>\n'
        f'<img src="data:image/png;base64,{dist_chart_b64}" '
        f'style="max-width:100%;height:auto;" alt="GCM distribution chart">'
        '\n<h2 style="margin-top:20px;">Per-GCM Compounded GDP pc Impact by 2040</h2>\n'
        f'<img src="data:image/png;base64,{gcm_chart_b64}" '
        f'style="max-width:100%;height:auto;" alt="Per-GCM bar chart">'
    )

    # ── Collect beta values for JS hover computation ─────────────────────
    betas_dict = {}
    for panel_name, pspec in PANELS.items():
        regr = pd.read_csv(str(pspec["regr_file"]))
        row = regr[
            (regr["spec"] == "linear_ctrl_baseline") &
            (regr["variable"] == "heat_freq_weighted") &
            (regr["se_type"] == "conley_500km")
        ].iloc[0]
        betas_dict[panel_name] = float(row["beta"])
    betas_js = "var betas = " + json.dumps(betas_dict) + ";"

    dehd_colors_js = json.dumps(dehd_colors)
    diff_colors_js = json.dumps(diff_colors)
    dehd_bounds_js = json.dumps(DEHD_BOUNDS)
    diff_bounds_js = json.dumps(DIFF_BOUNDS)

    # ── Build HTML ───────────────────────────────────────────────────────
    print("\nBuilding HTML...")
    html = build_html(geojson_str, cbar_dehd_b64, cbar_diff_b64,
                      dehd_colors_js, diff_colors_js,
                      dehd_bounds_js, diff_bounds_js,
                      betas_js, panel_stats, csv_table_html)

    out_html = OUT_DIR / "slopes_explorer.html"
    with open(str(out_html), "w") as f:
        f.write(html)

    size_mb = os.path.getsize(str(out_html)) / (1024 * 1024)
    print(f"\nSaved: {out_html}  ({size_mb:.1f} MB)")

    elapsed = time.time() - wall_start
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
