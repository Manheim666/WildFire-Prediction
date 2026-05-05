# MANHEIM — Wildfire Prediction & Risk Intelligence for Azerbaijan

> End-to-end machine learning pipeline that predicts wildfire risk across 16 Azerbaijani cities using satellite fire detections, weather reanalysis data, and multi-model ensemble learning. Produces daily (30-day) and hourly (168-hour) risk forecasts with an interactive web dashboard for real-time monitoring.

**Built for presentation to the Ministry of Ecology and Natural Resources of Azerbaijan.**

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Notebooks](#notebooks)
5. [Project Structure](#project-structure)
6. [Models & Methodology](#models--methodology)
7. [Dashboard](#dashboard)
8. [Cities Covered](#cities-covered)
9. [Data Sources](#data-sources)
10. [Setup & Installation](#setup--installation)
11. [Glossary](#glossary)
12. [License](#license)

---

## Overview

**MANHEIM** ingests multi-source environmental data — Open-Meteo weather reanalysis (ERA5/ERA5-Land), NASA FIRMS satellite fire detections (MODIS + VIIRS), terrain elevation, vegetation indices, and land-cover statistics — then engineers 200+ predictive features and trains recall-optimized classifiers to forecast wildfire ignition probability for each of 16 cities across Azerbaijan.

The system operates at two temporal resolutions:

- **Daily pipeline** — 30-day rolling forecast with CatBoost + XGBoost ensembles, Optuna-tuned hyperparameters, isotonic probability calibration, and SHAP explainability
- **Hourly pipeline** — 168-hour (7-day) forecast with daytime-masked labels for clean signal separation

Risk levels are classified into four tiers: **Low** (< 15%), **Moderate** (15–35%), **High** (35–60%), **Extreme** (> 60%). The threshold is tuned to **maximize recall** — in wildfire early warning, missing a real fire is far more dangerous than a false alarm.

---

## Key Features

- **Multi-source data fusion** — weather, satellite fire, terrain, vegetation, land cover, population
- **200+ engineered features** — FWI family (FFMC, DMC, DC, ISI, BUI), VPD, dew point, heat index, drought proxy, dry-spell tracking, lag/rolling aggregates (1–30 day), Prophet seasonal residuals, cyclical time encodings, historical fire rates
- **Ensemble model selection** — 8+ classifiers compared: XGBoost, CatBoost, LightGBM, RandomForest, ExtraTrees, HistGradientBoosting, BalancedRF, LogisticRegression; best selected by composite recall/F1 score
- **Bayesian hyperparameter optimization** — Optuna with 50+ trials, precision-constrained search, early pruning
- **Recall-first threshold tuning** — operational threshold optimized on validation set to catch fires, not minimize false alarms
- **Isotonic probability calibration** — predicted probabilities match observed fire frequencies
- **SHAP explainability** — every prediction can be decomposed into feature contributions
- **3-way temporal split** — train (< 2024), validation (2024), test (≥ 2025) — test data never seen during training
- **Interactive web dashboard** — Leaflet risk map, daily/hourly toggle, forecast strip, detail panel, Plotly charts, filterable table
- **Folium + Plotly visualizations** — date-selectable HTML maps, animated dashboards, climate trend figures
- **Colab + local compatible** — runs identically on Google Colab and local JupyterLab/VS Code

---

## Pipeline Architecture

```
 ┌────────────────────────────────────────────────────────────────┐
 │              MANHEIM Pipeline — Run NB1 → NB6                  │
 │                                                                │
 │  NB1  Data Ingestion                                           │
 │  ├── Open-Meteo Archive API (ERA5 + ERA5-Land, 2012–present)   │
 │  ├── NASA FIRMS (MODIS C6.1 + VIIRS C2, 3 sensors)            │
 │  ├── Open-Elevation + GEE vegetation indices                   │
 │  └── → master_daily.parquet, master_hourly.parquet             │
 │          │                                                     │
 │          ▼                                                     │
 │  NB2  EDA & Feature Engineering                                │
 │  ├── 200+ features: FWI, VPD, lags, rolling, Prophet residuals │
 │  ├── Outlier detection, fire-weather hypothesis tests          │
 │  └── → engineered_daily.parquet, engineered_hourly.parquet     │
 │          │                                                     │
 │     ┌────┴──────────────────┐                                  │
 │     ▼                       ▼                                  │
 │  NB3  Weather Forecast   NB4  Wildfire Detection               │
 │  ├── Prophet + XGBoost   ├── 8+ models + Optuna               │
 │  ├── 30-day + 168-hour   ├── SHAP + calibration               │
 │  └── stacking ensemble   └── recall-optimized threshold        │
 │     │                       │                                  │
 │     └───────────┬───────────┘                                  │
 │                 ▼                                              │
 │  NB5  Risk Prediction & Dashboard                              │
 │  ├── 30-day + 168-hour wildfire risk per city                  │
 │  ├── Folium maps, Plotly animated dashboards                   │
 │  └── JSON/CSV export → web dashboard                           │
 │                 │                                              │
 │                 ▼                                              │
 │  NB6  Climate Report                                           │
 │  └── Trend analysis, forecast vs history, risk outlook         │
 │                                                                │
 │  src/  Shared Python Module                                    │
 │  └── config · features · modeling · evaluation · visualization │
 └────────────────────────────────────────────────────────────────┘
```

---

## Notebooks

| # | Notebook | Purpose | Runtime |
|---|----------|---------|---------|
| 1 | `01_Data_Ingestion.ipynb` | Ingest weather, fire, terrain, vegetation data for 16 cities | ~5–15 min |
| 2 | `02_EDA_FeatureEngineering.ipynb` | EDA, hypothesis tests, 200+ feature engineering (daily + hourly) | ~3–5 min |
| 3 | `03_Weather_TimeSeries.ipynb` | Prophet + XGBoost stacking forecasts (30-day daily + 168h hourly) | ~60–120 min |
| 4 | `04_Wildfire_Detection.ipynb` | Multi-model classification, Optuna tuning, SHAP, calibration | ~10–20 min |
| 5 | `05_Risk_Prediction_Dashboard.ipynb` | Risk scoring, Folium/Plotly maps, dashboard JSON export | ~2–5 min |
| 6 | `06_Climate_Report.ipynb` | Climate trend analysis, forecast vs history, risk outlook | ~1–2 min |

**Run in order: NB1 → NB2 → NB3 → NB4 → NB5 → NB6.** Each notebook auto-detects the project root.

---

## Project Structure

```
WildFire-Prediction/
├── notebooks/                        Run sequentially: NB1 → NB6
│   ├── 01_Data_Ingestion.ipynb
│   ├── 02_EDA_FeatureEngineering.ipynb
│   ├── 03_Weather_TimeSeries.ipynb
│   ├── 04_Wildfire_Detection.ipynb
│   ├── 05_Risk_Prediction_Dashboard.ipynb
│   └── 06_Climate_Report.ipynb
│
├── src/                              Shared Python module
│   ├── config.py                     Paths, constants, city coordinates
│   ├── features.py                   FWI, VPD, lags, rolling, anomaly features
│   ├── modeling.py                   Model factory functions
│   ├── evaluation.py                 Metrics, threshold tuning, leaderboards
│   ├── visualization.py              Plotting helpers (confusion matrices, PR, SHAP)
│   ├── utils.py                      Data I/O utilities
│   └── prediction_pipeline.py        End-to-end scoring pipeline
│
├── data/
│   ├── raw/                          Open-Meteo cache, FIRMS archives, legacy CSVs
│   ├── processed/                    Engineered parquet files (master, engineered, fires)
│   └── reference/                    Static geography, city coordinates
│
├── models/
│   ├── wildfire/                     Fire models, manifests, feature lists
│   ├── weather/                      Weather forecast model bundles
│   └── prophet_cache/                Cached Prophet models per city/variable
│
├── outputs/                          Pipeline artefacts (forecasts, risk scores)
├── reports/
│   ├── figures/                      Publication-quality figures
│   ├── maps/                         Interactive Folium/Plotly HTML maps
│   └── metrics/                      CSV leaderboards and summaries
│
├── dashboard/                        Standalone web dashboard (HTML/JS/CSS)
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   └── data/                         JSON data files for the dashboard
│
├── docs/                             GitHub Pages deployment
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Models & Methodology

### Wildfire Detection (NB4)

**Objective:** Maximize recall while maintaining reasonable precision. Missing a wildfire is far more costly than a false alarm.

| Stage | Detail |
|-------|--------|
| **Data split** | Train (< 2024), Validation (2024), Test (≥ 2025) — strict temporal separation |
| **Feature pruning** | Remove near-zero-variance + highly correlated (r > 0.95) features |
| **Base models** | XGBoost, CatBoost, LightGBM, RandomForest, ExtraTrees, HistGBC, BalancedRF, LogisticRegression |
| **Class weighting** | Cost-sensitive learning with `scale_pos_weight` proportional to class imbalance |
| **Oversampling** | Conservative SMOTE (ratio 0.2–0.3) on gradient boosters |
| **Hyperparameter search** | Optuna Bayesian optimization, 50+ trials, precision floor constraint |
| **Threshold tuning** | Recall-optimized on validation set; composite objective: `0.6 × Recall + 0.4 × F1` |
| **Calibration** | Isotonic regression on validation set → meaningful probability scores |
| **Explainability** | SHAP TreeExplainer with top-25 feature importance |
| **Overfitting guard** | Train-vs-val F1 gap < 15%; flagged otherwise |

### Hourly Pipeline (NB4 Part B)

- Daytime label masking: relabels nighttime fire-hours to 0 (eliminates noisy labels broadcast from daily resolution)
- Separate Optuna-tuned CatBoost + XGBoost models
- 68 hourly-specific features including hourly lag/rolling with `h` suffix

### Weather Forecasting (NB3)

- Prophet (yearly/weekly/daily seasonality) + XGBoost (recursive multi-step) stacking per city per variable
- 8-model comparison: Ridge, ElasticNet, RF, ExtraTrees, HistGBR, XGB, LGB, CatBoost
- 128+ model bundles (16 cities × 8 weather variables)

### Evaluation Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Recall** | ≥ 0.70 | Must catch real fires |
| **Precision** | ≥ 0.15 | Floor to prevent degenerate models |
| **PR-AUC** | ≥ 0.20 | Primary ranking metric for imbalanced data |
| **Overfit gap** | < 0.15 | Train-vs-val F1 difference |

---

## Dashboard

The interactive web dashboard provides real-time wildfire risk monitoring:

- **Daily / Hourly toggle** — switch between 30-day and 168-hour forecasts
- **Leaflet risk map** — colour-coded markers for all 16 cities
- **Forecast strip** — scrollable daily/hourly risk cards
- **Detail panel** — city-specific weather conditions and risk breakdown
- **Plotly charts** — risk trend and weather condition plots
- **Filterable table** — sort by date, city, risk level

**Live dashboard:** Deploy via GitHub Pages from the `docs/` folder, or serve locally:

```bash
cd dashboard && python3 -m http.server 8765
```

---

## Cities Covered

16 cities across Azerbaijan's diverse climate zones:

| City | Lat | Lon | Climate Zone |
|------|----:|----:|-------------|
| Baku | 40.41 | 49.87 | Semi-arid coastal; highest fire rate |
| Shabran | 41.21 | 48.99 | Northeastern; major fire events 2021–22 |
| Ganja | 40.68 | 46.36 | Western highland |
| Mingachevir | 40.76 | 47.06 | Central lowland (Kura River) |
| Shirvan | 39.93 | 48.93 | Kura-Araz lowland; dry climate |
| Lankaran | 38.75 | 48.85 | Southern subtropical coastal |
| Shaki | 41.20 | 47.17 | Northern foothill; forested |
| Nakhchivan | 39.21 | 45.41 | Exclave; arid continental |
| Yevlakh | 40.62 | 47.15 | Central plains; dry lowland |
| Quba | 41.36 | 48.53 | Northern mountains |
| Khachmaz | 41.46 | 48.81 | Northeastern Caspian coast |
| Gabala | 41.00 | 47.85 | High elevation; dense forest |
| Shamakhi | 40.63 | 48.64 | Mountain plateau |
| Jalilabad | 39.21 | 48.30 | Southern lowland; agricultural |
| Barda | 40.37 | 47.13 | Karabakh region; central lowland |
| Zaqatala | 41.63 | 46.64 | Northwestern mountain-foothill; lowest fire rate |

Fire labels are aggregated daily within a **20 km radius** of each city centroid using NASA FIRMS detections.

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| **Open-Meteo Archive** | Hourly weather reanalysis (ERA5 + ERA5-Land, 2012–present) | Free, no API key |
| **Open-Meteo Forecast** | 16-day ahead hourly weather | Free, no API key |
| **NASA FIRMS** | Active fire detections (MODIS C6.1 + VIIRS C2) | Free archive CSVs |
| **Open-Elevation** | Terrain elevation + derived slope | Free, no API key |
| **Google Earth Engine** | MODIS burned area, Sentinel-2 NDVI/NDBI | Free (GEE account) |
| **Reference CSVs** | Land cover, urban density, population, roads | Static local files |

### Important Notes

- Fire risk = probability of a FIRMS-detected hotspot, **not** burn area or severity
- Probabilities are isotonically calibrated — use the threshold from `model_manifest.json`
- Weather forecast accuracy degrades beyond day 7; days 1–7 are most reliable
- Evaluate with **PR-AUC and recall**, not accuracy (class imbalance ~8–10% fire-day prevalence)

---

## Setup & Installation

### Local

```bash
git clone https://github.com/your-repo/WildFire-Prediction.git
cd WildFire-Prediction
pip install -r requirements.txt
jupyter lab notebooks/
```

### Google Colab

1. Upload the project folder to Google Drive
2. Open any notebook — the first cell auto-mounts Drive and detects the project root
3. Run notebooks in order: **NB1 → NB2 → NB3 → NB4 → NB5 → NB6**

### Environment Variable (optional)

```bash
export MANHEIM_ROOT=/absolute/path/to/project
```

### Git LFS

Large data files (`.parquet`, `.csv`, `.pkl`) are tracked via Git LFS:

```bash
git lfs install && git lfs pull
```

### Dependencies

| Group | Key Packages |
|-------|-------------|
| **Core** | pandas, numpy, pyarrow, joblib, tqdm |
| **Ingestion** | openmeteo-requests, requests-cache, retry-requests |
| **EDA** | scipy, statsmodels, prophet, matplotlib, seaborn |
| **ML** | scikit-learn, xgboost, lightgbm, catboost, imbalanced-learn, optuna, shap |
| **Visualization** | folium, plotly |

---

## Glossary

| Term | Description |
|------|-------------|
| **FWI** | Fire Weather Index — composite wildfire danger metric |
| **FFMC** | Fine Fuel Moisture Code — surface litter dryness |
| **DMC/DC** | Duff Moisture / Drought Code — subsurface drought indicators |
| **ISI/BUI** | Initial Spread Index / Buildup Index — fire spread and fuel availability |
| **VPD** | Vapor Pressure Deficit — atmospheric dryness (higher = more fire risk) |
| **FIRMS** | NASA Fire Information for Resource Management System |
| **MODIS/VIIRS** | Satellite sensors for thermal anomaly detection (~1 km / ~375 m resolution) |
| **PR-AUC** | Precision-Recall Area Under the Curve — primary metric for imbalanced classification |
| **SHAP** | SHapley Additive exPlanations — model interpretability method |
| **Isotonic calibration** | Non-parametric probability calibration for reliable risk scores |

---

## License

This project is intended for research and environmental monitoring purposes. Data sources are publicly available under their respective terms of use.
