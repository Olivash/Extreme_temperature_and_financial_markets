# Raw GDP and Admin2 Polygon Data

## 1. GADM Admin2 GeoPackage with GDP per Capita

**File:** `polyg_adm2_gdp_perCapita_1990_2022.gpkg`
**Used by:** `src/aggregation/aggregate_heat_to_admin2.py`,
`src/aggregation/aggregate_era5_controls_to_admin2.py`,
`src/econometrics/panel_regression_gdp_heat.py`

### Contents

A GeoPackage containing ~48,000 admin2 (sub-national) polygons sourced from
[GADM v4.1](https://gadm.org/) with GDP per capita columns merged from the
[Global Data Lab (GDL) Subnational Human Development Database](https://globaldatalab.org/shdi/).

| Column | Description |
|--------|-------------|
| `GID_2` | GADM admin2 unique identifier (e.g., `AFG.1.1_1`) |
| `adm2ID` | Numeric admin2 ID |
| `iso3` | ISO 3166-1 alpha-3 country code |
| `NAME_2` | Admin2 name |
| `geometry` | Polygon geometry (EPSG:4326) |
| `1990` … `2022` | GDP per capita (USD, PPP) for each year |

### How to obtain

1. Download GADM v4.1 polygons at the admin2 level from https://gadm.org/download_world.html.
2. Download subnational GDP per capita data from the
   [GDL Area Database](https://globaldatalab.org/areadata/table/gdppc/).
3. Merge using the `iso_code` + region name fields, then export to GeoPackage.

Alternatively, contact the authors for the merged file used in this study.

---

## 2. Admin2 GDP Growth Parquet

**File:** `adm2_gdp_growth_residuals_long.parquet`
**Used by:** `src/econometrics/panel_regression_gdp_heat.py`

### Contents

A long-format panel of admin2 annual GDP growth rates, derived from the
GeoPackage above.

| Column | Description |
|--------|-------------|
| `GID_2` | GADM admin2 unique identifier |
| `iso3` | ISO 3166-1 alpha-3 country code |
| `year` | Calendar year (2000–2022) |
| `admin2_growth` | Log-difference of GDP per capita (i.e., ln(GDPpc_t / GDPpc_{t-1})) |
| `residual` | `admin2_growth` residualized against country-year means (removes country business cycle) |
| `gdp_per_capita` | Level of GDP per capita (USD, PPP) |

### Construction

```python
import pandas as pd
import geopandas as gpd
import numpy as np

gdf = gpd.read_file("polyg_adm2_gdp_perCapita_1990_2022.gpkg")
gdp_cols = [c for c in gdf.columns if c.isdigit()]

# Melt to long format
long = gdf[["GID_2", "adm2ID", "iso3"] + gdp_cols].melt(
    id_vars=["GID_2", "adm2ID", "iso3"],
    var_name="year", value_name="gdp_per_capita"
)
long["year"] = long["year"].astype(int)
long = long.sort_values(["GID_2", "year"])

# Log-differenced growth rate
long["admin2_growth"] = (
    long.groupby("GID_2")["gdp_per_capita"]
    .transform(lambda x: np.log(x).diff())
)

# Residual: remove country-year mean (country business cycle)
country_year_mean = long.groupby(["iso3", "year"])["admin2_growth"].transform("mean")
long["residual"] = long["admin2_growth"] - country_year_mean

long = long.dropna(subset=["admin2_growth"])
long.to_parquet("adm2_gdp_growth_residuals_long.parquet", index=False)
```
