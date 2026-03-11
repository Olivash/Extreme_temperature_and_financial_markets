# Raw Spatial Data — GHS Settlement Rasters

**Used by:** `src/aggregation/aggregate_heat_to_admin2.py`

## GHS Built-up Surface (GHS-BUILT-S)

Settlement-weighted aggregation uses the Global Human Settlement (GHS)
Built-up Surface rasters (R2023A release) to weight each 0.1° grid cell by
the amount of built-up area (m²) present.  This ensures that EHD values
reflect heat exposure where people live and work, rather than a simple
geographic average.

### Files expected

```
data/raw/shapes/ghs_settlement/
    GHS_BUILT_S_E2000_GLOBE_R2023A_54009_100_V1_0.tif
    GHS_BUILT_S_E2005_GLOBE_R2023A_54009_100_V1_0.tif
    GHS_BUILT_S_E2010_GLOBE_R2023A_54009_100_V1_0.tif
    GHS_BUILT_S_E2015_GLOBE_R2023A_54009_100_V1_0.tif
    GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0.tif
    GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100_V1_0.tif
```

### Specifications

| Property | Value |
|----------|-------|
| Source   | [GHSL Data Package 2023](https://ghsl.jrc.ec.europa.eu/ghs_buS2023.php) |
| Epochs   | 2000, 2005, 2010, 2015, 2020, 2025 |
| Resolution | 100 m |
| Projection | Mollweide (EPSG:54009) |
| Unit | m² of built-up surface per 100 m cell |
| Variable | Total built-up surface (residential + non-residential) |

### How to download

1. Visit https://ghsl.jrc.ec.europa.eu/ghs_buS2023.php
2. Download the **GLOBE** tiles at **100 m** resolution for each epoch.
3. Place files in `data/raw/shapes/ghs_settlement/`.

Direct download links (as of 2024):
```
https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/
    GHS_BUILT_S_E2000_GLOBE_R2023A_54009_100_V1_0/
        V1-0/GHS_BUILT_S_E2000_GLOBE_R2023A_54009_100_V1_0.tif
```
(Replace `E2000` with `E2005`, `E2010`, `E2015`, `E2020`, `E2025`.)

### Storage estimate

~8 GB total for all 6 epochs.

---

## Epoch Matching

Heat years are matched to the nearest available settlement epoch:

| Heat years | Settlement epoch |
|-----------|-----------------|
| 2000–2002 | 2000 |
| 2003–2007 | 2005 |
| 2008–2012 | 2010 |
| 2013–2017 | 2015 |
| 2018–2022 | 2020 |
| 2023–2024 | 2025 |
