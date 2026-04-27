# wildfire-risk-prediction
# Predictive Wildfire Risk Modeling — California

**Stack:** Python · XGBoost · Random Forest · LightGBM · Scikit-learn · Pandas

---

## Problem

California wildfires cause billions in damage and loss of life annually. Early identification of high-risk locations and time periods enables preemptive resource deployment — fire crews, equipment, and evacuation planning — before ignition occurs.

> **Goal:** Predict the probability of wildfire occurrence at a given latitude/longitude grid point in a given month, using satellite, climate, terrain, and vegetation data.

---

## Dataset

Five data sources merged into a single spatio-temporal feature matrix:

| Source | Features |
|---|---|
| NASA FIRMS satellite fire data | fire_count, avg_confidence, latitude, longitude |
| NASA POWER precipitation data | PRECTOTCORR, PRECTOTCORR_SUM, RH2M, WD10M, WS10M_RANGE |
| MODIS NDVI (vegetation index) | NDVI (rescaled from raw to 0-1) |
| NOAA temperature data | T2M, T2M_MAX, TS (surface temperature) |
| USGS terrain data | elevation, slope, aspect |

**Training period:** 2010 – 2023 (14 years)
**Prediction period:** 2024 – 2025

---

## Target Variable

```python
fire_occurred = 1 if fire_count > 1 AND avg_confidence >= 70%
fire_occurred = 0 otherwise
```

Confidence threshold of 70% filters out low-quality satellite detections. The fire_count > 1 requirement reduces false positives from single-pixel anomalies.

---

## Feature Engineering

**Spatio-temporal aggregation:** raw satellite rows grouped by latitude, longitude, year, and month. Multiple fire detections at the same location in the same month are aggregated into counts and averages.

**Cyclical month encoding:** month converted to sine and cosine components to capture seasonality without ordinal bias:
```python
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

**Fire risk index:** composite feature combining temperature, humidity, and precipitation:
```python
fire_risk_index = 0.4 × T2M_MAX + 0.3 × (100 - RH2M) + 0.3 × (1 / (PRECTOTCORR + 0.1))
```

---

## Models

Six models compared, with hyperparameter tuning on top two:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline, scaled features |
| Random Forest | Tuned via GridSearchCV |
| Gradient Boosting | Baseline |
| Support Vector Machine | Scaled features |
| XGBoost | Tuned via GridSearchCV, scale_pos_weight for imbalance |
| LightGBM | class_weight='balanced' |

**Ensembles:**
- **Stacking ensemble** (RF + XGBoost → Logistic Regression meta-learner)
- **Soft voting ensemble** (RF + XGBoost)

**Imbalance handling:** `scale_pos_weight` for XGBoost, `class_weight='balanced'` for tree-based models — fire events are rare relative to non-fire periods.

**Calibration:** `CalibratedClassifierCV` with sigmoid method applied to both RF and XGBoost so predicted probabilities reflect true fire likelihood.

---

---

## Output: Fire Risk Maps

The final output is a geospatial fire risk probability map by location and month. Each latitude/longitude grid point receives a calibrated probability score for 2024-2025.

Example: January 2025 fire risk map plots predicted probability across California using a Yellow-to-Red color scale — darker red = higher risk.

---

## Pipeline Overview

```
Step 1  ── Load and merge 5 data sources (fire, weather, NDVI, temperature, terrain)
Step 2  ── Filter to confidence > 50%, aggregate to lat/lon/year/month grid
Step 3  ── Define target variable (fire_occurred)
Step 4  ── Engineer features (fire risk index, cyclical month encoding)
Step 5  ── EDA (class distribution, correlation matrix, high-correlation pairs)
Step 6  ── Train/test split (70/30, stratified)
Step 7  ── Train and evaluate 6 baseline models
Step 8  ── Hyperparameter tuning (GridSearchCV on XGBoost and RF)
Step 9  ── Feature importance comparison (RF vs XGBoost)
Step 10 ── Probability calibration (CalibratedClassifierCV)
Step 11 ── Stacking and voting ensembles
Step 12 ── Generate fire risk probability map for 2024-2025
```

---

## Key Design Decisions

**Why aggregate to monthly grid?**
Raw satellite data has one row per fire detection per day. Aggregating to monthly lat/lon grid reduces noise, aligns with the temporal resolution of climate data, and creates a meaningful unit of analysis for risk prediction.

**Why cyclical encoding for month?**
Treating month as an integer (1-12) implies January and December are far apart. Sine/cosine encoding captures the cyclical nature — December and January are adjacent, and the peak fire season (summer) forms a smooth wave.

**Why calibrate probabilities?**
Tree ensemble probabilities are often poorly calibrated on imbalanced data. Without calibration, a 70% predicted probability might not actually correspond to 70% observed fire frequency. Sigmoid calibration corrects this so probabilities are meaningful for resource allocation decisions.

**Why stacking over simple voting?**
A stacking ensemble lets the meta-learner (LR) learn the optimal way to combine RF and XGBoost predictions, rather than averaging them equally. It captures when each model is right and wrong.

---

## Files

```
wildfire_risk_prediction.ipynb              main notebook
Data and Analysis for Wildfire project.docx project report
Group5_Presentation_AML.pptx               presentation slides
ca_wildfire_data.csv                        primary dataset
README.md                                   this file
```

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn openpyxl

# Place all data files in the same directory as the notebook:
# ca_wildfire_data.csv
# Precipitation_cali_data.csv
# Monthly_NDVI_California_2010_2025.csv
# Temperature_cali_monthly.xlsx
# California_Elevation_Slope_Aspect.csv

# Update file paths in the notebook from absolute paths to relative paths
# e.g. '/Users/yashaswireddy/Downloads/ca_wildfire_data.csv' -> 'ca_wildfire_data.csv'

# Run the notebook
jupyter notebook wildfire_risk_prediction.ipynb
```

> Note: GridSearchCV tuning on XGBoost and Random Forest is computationally intensive. Runtime is approximately 30-60 minutes on a standard laptop.

