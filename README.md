# 🏠 HouseML — House Price Prediction Pipeline

A Streamlit web application that walks you through a complete machine learning pipeline for predicting house prices — from raw CSV upload to a trained linear regression model and live price inference.

---

## Features

- **5-step guided pipeline** — Upload → Audit → Clean → Train → Predict
- **Interactive data audit** — detects missing values, duplicates, and outliers per column
- **Configurable cleaning** — choose imputation strategies, outlier handling, and optional log-transform on the target
- **Linear regression training** — with StandardScaler, OneHotEncoder, and cross-validation scoring
- **Visual diagnostics** — feature importance bar chart and actual vs. predicted scatter plot
- **Live price prediction** — enter house attributes and get an estimated price with a confidence range

---

## Requirements

- Python 3.8+
- pip packages:

```
streamlit
pandas
numpy
matplotlib
scikit-learn
```

Install all at once:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

---

## Running the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dataset Format

Upload a CSV file with house records. The app auto-detects numeric and categorical columns. It works best with a `price_usd` target column, but any numeric column can be chosen as the target during the Clean step.

### Expected / Supported Columns

| Column | Type | Description |
|---|---|---|
| `price_usd` | float | Sale price in USD (default target) |
| `sqft_living` | float | Interior living area (sq ft) |
| `sqft_lot` | float | Lot size (sq ft) |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | int | Number of bathrooms |
| `floors` | int | Number of floors |
| `has_basement` | bool (0/1) | Basement present |
| `basement_sqft` | int | Basement area (sq ft) |
| `garage_cars` | int | Garage capacity (cars) |
| `has_pool` | bool (0/1) | Pool present |
| `year_built` | int | Year of construction |
| `house_age_years` | int | Age of the house |
| `renovated` | bool (0/1) | Has been renovated |
| `condition_score` | int (1–5) | Physical condition rating |
| `quality_grade` | int (1–10) | Construction quality grade |
| `school_rating` | float | Nearby school rating (0–10) |
| `crime_rate_per_1000` | float | Local crime rate |
| `dist_city_center_km` | float | Distance to city center |
| `dist_school_km` | float | Distance to nearest school |
| `dist_hospital_km` | float | Distance to nearest hospital |
| `property_tax_rate_pct` | float | Annual property tax rate (%) |
| `hoa_monthly_usd` | int | Monthly HOA fee (USD) |
| `market_trend` | float | Market trend index (−1 to 1) |

Columns not listed here are still supported — numeric columns get scaled and categorical columns get one-hot encoded automatically.

---

## Pipeline Steps

### 1. Upload CSV
Drag and drop or browse for a `.csv` file. A preview of the first 8 rows is shown before proceeding.

### 2. Data Audit
Generates a per-column report showing data type, missing value counts, unique value counts, and IQR-based outlier counts. Rows with issues are highlighted.

### 3. Clean Data
Configure the cleaning pipeline:
- **Missing values** — median / mean / zero fill or row drop (numeric); mode / "Unknown" fill or row drop (categorical)
- **Outlier handling** — IQR clipping (1.5×), Z-score clipping (±3σ), or none
- **Target column** — select which column to predict
- **Log-transform** — apply `log1p` to the target for skewed price distributions (recommended)
- **Drop columns** — exclude irrelevant columns from training

### 4. Train Model
Fits a `LinearRegression` model inside a `scikit-learn` `Pipeline` with:
- `SimpleImputer` + `StandardScaler` for numeric features
- `SimpleImputer` + `OneHotEncoder` for categorical features
- 80/20 train-test split + 5-fold cross-validation

Displays R², RMSE, MAE, CV R², a top-10 feature importance chart, and an actual vs. predicted scatter plot.

### 5. Predict Price
Enter property details through a structured form (Size, Features, Condition, Location) and click **Predict Price** to get an estimated value with a ±RMSE confidence range.

---

## Project Structure

```
.
├── app.py          # Main Streamlit application
└── README.md       # This file
```

---

## Notes

- The model is **Linear Regression** — suitable as a baseline. For better accuracy on real-world data, consider gradient boosting models (XGBoost, LightGBM).
- Log-transforming the target is strongly recommended for right-skewed price distributions.
- All pipeline state is stored in `st.session_state`, so refreshing the page resets the session.
