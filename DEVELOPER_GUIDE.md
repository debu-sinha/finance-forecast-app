# Developer Guide

This guide answers common questions about working with the Finance Forecasting Platform, including customization, extending functionality, and production deployment considerations.

## Target Platform: Databricks Apps

**This application is designed to run as a Databricks App in production.** While it works locally for development, all features are optimized for Databricks Apps deployment.

### Platform Comparison

| Factor | Local Development | Databricks Apps |
|--------|-------------------|-----------------|
| **Compute** | Your machine (often more powerful) | **4 vCPU / 12GB RAM max** |
| **Authentication** | Manual token in `.env.local` | Auto-inherited from workspace |
| **SQL Access** | Need warehouse ID + token | WorkspaceClient auto-configures |
| **MLflow** | Points to Databricks via token | Native integration |
| **Unity Catalog** | Works via token | Native access |
| **Parallelism** | Can use more workers | Limited by 4 vCPU |

### Compute Considerations

**Databricks Apps currently has a hard limit of 4 vCPU / 12GB RAM.** This affects:
- Number of parallel training workers
- Hyperparameter grid search combinations
- Memory for large datasets

**Local development may actually be faster** for heavy training workloads. The configuration is designed to be easily adjustable between environments.

---

## Configuration Reference

All tunable parameters are controlled via environment variables. Set these in `.env.local` for local development or in `app.yaml` for Databricks Apps deployment.

### Parallelism & Performance

| Variable | Default | Description | Local Recommended | Databricks Apps |
|----------|---------|-------------|-------------------|-----------------|
| `MLFLOW_MAX_WORKERS` | `1` | Parallel threads for hyperparameter search | `4-8` | `1-2` |
| `PROPHET_MAX_COMBINATIONS` | `3` | Max hyperparameter combinations for Prophet | `9-12` | `3-6` |
| `ARIMA_MAX_COMBINATIONS` | `6` | Max hyperparameter combinations for ARIMA | `12-18` | `4-6` |
| `ETS_MAX_COMBINATIONS` | `4` | Max hyperparameter combinations for ETS | `8-12` | `4` |
| `SARIMAX_MAX_COMBINATIONS` | `8` | Max hyperparameter combinations for SARIMAX | `16-24` | `6-8` |
| `XGBOOST_MAX_COMBINATIONS` | `4` | Max hyperparameter combinations for XGBoost | `8-16` | `4` |

### Databricks Connection

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABRICKS_HOST` | *(required)* | Workspace URL. Use `inherit` on Databricks Apps for auto-config. |
| `DATABRICKS_TOKEN` | *(required locally)* | Personal access token. Not needed on Databricks Apps. |
| `DATABRICKS_SQL_WAREHOUSE_ID` | *(optional)* | SQL Warehouse ID for direct SQL queries |

### MLflow & Unity Catalog

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_EXPERIMENT_NAME` | `/Shared/finance-forecasting` | MLflow experiment path |
| `UC_CATALOG_NAME` | `main` | Unity Catalog catalog name |
| `UC_SCHEMA_NAME` | `default` | Unity Catalog schema name |
| `UC_MODEL_NAME` | `main.default.finance_forecast_model` | Full model path in Unity Catalog |
| `UC_MODEL_NAME_ONLY` | `finance_forecast_model` | Model name (without catalog/schema) |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Backend server port |

### Example: Local Development (High Performance)

Create `.env.local`:
```bash
# Databricks Connection
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_personal_access_token

# Performance - Use more resources locally
MLFLOW_MAX_WORKERS=4
PROPHET_MAX_COMBINATIONS=9
ARIMA_MAX_COMBINATIONS=12
ETS_MAX_COMBINATIONS=8
SARIMAX_MAX_COMBINATIONS=16
XGBOOST_MAX_COMBINATIONS=8

# MLflow
MLFLOW_EXPERIMENT_NAME=/Shared/finance-forecasting

# Optional: SQL Access
DATABRICKS_SQL_WAREHOUSE_ID=your_warehouse_id
```

### Example: Databricks Apps (Constrained - 4 vCPU / 12GB RAM)

Configure in `app.yaml`:
```yaml
command:
  - python
  - -m
  - uvicorn
  - backend.main:app
  - --host
  - 0.0.0.0
  - --port
  - "8000"

env:
  - name: DATABRICKS_HOST
    value: "inherit"  # Auto-inherit from workspace
  # No DATABRICKS_TOKEN needed - uses workspace auth

  # Performance - Conservative for 4 vCPU limit
  - name: MLFLOW_MAX_WORKERS
    value: "1"
  - name: PROPHET_MAX_COMBINATIONS
    value: "3"
  - name: ARIMA_MAX_COMBINATIONS
    value: "6"
  - name: ETS_MAX_COMBINATIONS
    value: "4"
  - name: SARIMAX_MAX_COMBINATIONS
    value: "8"
  - name: XGBOOST_MAX_COMBINATIONS
    value: "4"

  # MLflow
  - name: MLFLOW_EXPERIMENT_NAME
    value: "/Shared/finance-forecasting"

  # Unity Catalog
  - name: UC_CATALOG_NAME
    value: "main"
  - name: UC_SCHEMA_NAME
    value: "default"
```

### Tuning Guidelines

**For faster local development:**
```bash
# In .env.local - aggressive parallelism
MLFLOW_MAX_WORKERS=8
PROPHET_MAX_COMBINATIONS=12
```

**For large datasets on Databricks Apps:**
```yaml
# In app.yaml - conservative to avoid memory issues
- name: MLFLOW_MAX_WORKERS
  value: "1"
- name: PROPHET_MAX_COMBINATIONS
  value: "2"
```

**Memory-constrained scenarios:**
- Reduce `*_MAX_COMBINATIONS` to limit grid search
- Set `MLFLOW_MAX_WORKERS=1` for sequential processing
- Consider pre-aggregating data to reduce row count

---

## Hardware & Scaling Recommendations

### Platform Constraints

| Platform | vCPU | RAM | Max Parallel Workers | Recommended Data Size |
|----------|------|-----|---------------------|----------------------|
| **Databricks Apps** | 4 | 12GB | 1-2 | ≤50K rows per segment |
| **Local Dev (Laptop)** | 4-8 | 16GB | 2-4 | ≤100K rows per segment |
| **Local Dev (Workstation)** | 8-16 | 32GB+ | 4-8 | ≤500K rows per segment |

### Data Size Guidelines

| Data Size (rows) | Single Model Training | Batch Training (10 segments) | Notes |
|------------------|----------------------|------------------------------|-------|
| **≤10K** | ✅ Fast (<30s) | ✅ Fast (<5 min) | Ideal for prototyping |
| **10K-50K** | ✅ Normal (30s-2min) | ✅ Normal (5-20 min) | Recommended for production |
| **50K-100K** | ⚠️ Slow (2-5min) | ⚠️ Slow (20-50 min) | Consider aggregation |
| **100K-500K** | ⚠️ Very slow (5-15min) | ❌ May timeout | Pre-aggregate or sample |
| **>500K** | ❌ Not recommended | ❌ Not recommended | Use Spark-based solution |

### Batch Training Capacity

**Sequential Processing (Current Implementation):**
- Segments are processed one at a time for reliable progress tracking
- Each segment takes 30s-3min depending on data size and models selected
- **Recommended max segments per batch:** 20-50 segments

| Segments | Est. Time (50K rows/seg, 2 models) | MLflow Runs Created |
|----------|-----------------------------------|---------------------|
| 5 | ~5-10 min | 5 parent runs |
| 10 | ~10-20 min | 10 parent runs |
| 20 | ~20-40 min | 20 parent runs |
| 50 | ~50-100 min | 50 parent runs |

**Why Sequential?**
1. **Progress visibility** - Users see real-time progress per segment
2. **Error isolation** - One segment failure doesn't affect others
3. **Memory stability** - Prevents OOM on constrained environments
4. **MLflow consistency** - Ensures proper run logging

### Memory Usage by Model

| Model | Memory per 10K rows | Training Time | Scales With |
|-------|--------------------:|---------------|-------------|
| **Prophet** | ~200MB | Moderate | Data size, seasonality |
| **ARIMA** | ~100MB | Fast | Data size |
| **ETS** | ~100MB | Fast | Data size |
| **SARIMAX** | ~300MB | Slow | Data size, covariates |
| **XGBoost** | ~400MB | Moderate | Features, tree depth |

### Recommended Configurations

#### Databricks Apps (4 vCPU / 12GB RAM)
```yaml
env:
  - name: MLFLOW_MAX_WORKERS
    value: "1"
  - name: PROPHET_MAX_COMBINATIONS
    value: "3"
  - name: ARIMA_MAX_COMBINATIONS
    value: "4"
  - name: XGBOOST_MAX_COMBINATIONS
    value: "3"
```

**Best practices for Databricks Apps:**
- Limit to 2 models per training run (e.g., Prophet + XGBoost)
- Keep data under 50K rows per segment
- Batch training: max 20 segments recommended
- Use date range filters to reduce data size

#### Local Development (16GB+ RAM)
```bash
# .env.local
MLFLOW_MAX_WORKERS=4
PROPHET_MAX_COMBINATIONS=9
ARIMA_MAX_COMBINATIONS=12
XGBOOST_MAX_COMBINATIONS=8
```

**Best practices for local:**
- Can run all 5 models simultaneously
- Handle up to 100K rows comfortably
- Batch training: 50+ segments feasible
- Use for initial exploration before deploying

### Scaling Beyond Limits

If you need to process larger datasets or more segments:

1. **Pre-aggregate data**
   - Convert daily → weekly or monthly
   - Use `/api/aggregate` endpoint

2. **Sample data**
   - Use representative sample for hyperparameter tuning
   - Train final model on full data

3. **Parallelize externally**
   - Use Databricks Jobs to run multiple batch sessions
   - Each job processes a subset of segments

4. **Use Spark-based forecasting**
   - For >500K rows, consider Spark MLlib or distributed Prophet
   - Beyond scope of this app (designed for interactive use)

### Timeout Considerations

| Component | Default Timeout | Adjustable? |
|-----------|-----------------|-------------|
| Frontend fetch | 120s | Yes (in code) |
| Databricks Apps proxy | 300s | No (platform limit) |
| MLflow logging | None | N/A |
| Model training | None | N/A |

**If you hit timeouts:**
- Reduce data size or number of models
- Use CLI script for large batch jobs (no timeout)
- Split into multiple smaller batches

---

## Table of Contents

1. [Configuration Reference](#configuration-reference)
2. [Frequency & Data Questions](#frequency--data-questions)
3. [Covariate Configuration](#covariate-configuration)
4. [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
5. [Model Settings](#model-settings)
6. [Scaling & Performance](#scaling--performance)
7. [Data Integration](#data-integration)
8. [Production Deployment](#production-deployment)
9. [UI Improvements](#ui-improvements)
10. [Actuals vs Forecast Comparison](#actuals-vs-forecast-comparison)
11. [Adding New Models](#adding-new-models)

---

## Frequency & Data Questions

### Can the forecast be weekly vs. daily?

**Yes.** The platform supports three frequencies:
- `daily` - Day-level forecasts
- `weekly` - Week-level forecasts
- `monthly` - Month-level forecasts

Select the frequency in the UI dropdown or pass it in the API:

```json
{
  "frequency": "weekly",
  "horizon": 12
}
```

### Does forecast frequency need to match training data frequency?

**Yes.** The training data frequency should match the forecast frequency. If you have daily data but want weekly forecasts, you have two options:

#### Option 1: Use the Aggregation API ✅ IMPLEMENTED

The `/api/aggregate` endpoint converts data from higher to lower frequency automatically:

**Request:**
```json
POST /api/aggregate
{
  "data": [
    {"date": "2024-01-01", "sales": 100, "promo": 1},
    {"date": "2024-01-02", "sales": 120, "promo": 0},
    ...
  ],
  "time_col": "date",
  "target_col": "sales",
  "covariates": ["promo"],
  "source_frequency": "daily",
  "target_frequency": "weekly",
  "aggregation_method": "sum"
}
```

**Response:**
```json
{
  "data": [
    {"date": "2024-01-07", "sales": 840, "promo": 1}
  ],
  "original_rows": 7,
  "aggregated_rows": 1,
  "source_frequency": "daily",
  "target_frequency": "weekly",
  "aggregation_methods": {"sales": "sum", "promo": "max"}
}
```

**Aggregation behavior:**
- Target column: Uses specified method (`sum`, `mean`, or `last`)
- Binary covariates (0/1 values): Automatically uses `max` (1 if any day had 1)
- Continuous covariates: Automatically uses `mean`
- You can override with `covariate_aggregation`: `{"promo": "sum", "price": "mean"}`

#### Option 2: Pre-aggregate your data manually

Aggregate daily data to weekly before uploading:

```python
# Example: Aggregate daily to weekly
df['ds'] = pd.to_datetime(df['ds'])
weekly_df = df.resample('W', on='ds').agg({
    'y': 'sum',  # or 'mean' depending on your metric
    'covariate1': 'sum',
    'covariate2': 'mean'
}).reset_index()
```

**Recommendation:** Pre-aggregate for now. Training on daily and forecasting weekly would require careful handling of how covariates are aggregated (sum vs mean vs max).

---

### Will Prophet run without covariates?

**Yes, absolutely.** All models work with just the time series data (date + target). Covariates are optional enhancements.

When no covariates are provided:
- Prophet uses only trend + seasonality + holidays
- ARIMA/ETS use only the target series
- XGBoost uses lag features derived from the target

```json
{
  "data": [{"ds": "2023-01-01", "y": 1000}],
  "time_col": "ds",
  "target_col": "y",
  "covariates": [],  // Empty - no covariates
  "models": ["prophet"]
}
```

---

## Covariate Configuration

### When to use different regressor methods?

The `regressor_method` parameter controls how future covariate values are estimated when you don't provide them explicitly. This dropdown appears in the UI when you have covariates selected.

**UI Labels vs API Values:**

| UI Label | API Value | When to Use | Example |
|----------|-----------|-------------|---------|
| Mean of Last 12 Values | `mean` | Covariate is stationary/stable | Marketing spend that's consistent |
| Last Known Value | `last_value` | Recent values are most relevant | Economic indicators, prices |
| Linear Trend Projection | `linear_trend` | Covariate has a clear trend | Growing metrics, inflation |

#### `last_value` (Default)
Uses the most recent observed value:
```
Future value = last observed value
```
**Best for:** Prices, rates, metrics that don't change frequently

#### `mean`
Uses the mean of the last 12 historical values:
```
Future value = mean(last 12 values)
```
**Best for:** Promotional flags, holiday indicators, stable metrics

#### `linear_trend`
Extrapolates based on recent trend:
```
Future value = linear regression extrapolation
```
**Best for:** Growing/declining metrics, inflation adjustments

#### Best Practice: Provide actual future values

The most accurate approach is to provide known future covariate values:

```json
{
  "future_features": [
    {"ds": "2025-01-01", "Black Friday": 0, "marketing_spend": 50000},
    {"ds": "2025-01-08", "Black Friday": 0, "marketing_spend": 45000}
  ]
}
```

---

### When to change the Random Seed from 42?

**Short answer:** Almost never for production use. Change it only for specific testing purposes.

#### Keep seed at 42 when:
- You want reproducible results
- You're comparing model versions
- You're debugging or validating
- You're in production

#### Change the seed when:
- **Ensemble testing**: Train multiple models with different seeds and average predictions
- **Robustness testing**: Verify model isn't sensitive to initialization
- **A/B experiments**: Compare different random initializations

```python
# Example: Ensemble with multiple seeds
seeds = [42, 123, 456, 789, 1000]
predictions = []
for seed in seeds:
    result = train_model(data, random_seed=seed)
    predictions.append(result['forecast'])

# Average predictions for more robust forecast
ensemble_forecast = np.mean(predictions, axis=0)
```

#### What the seed affects:
- Prophet: MCMC sampling, cross-validation splits
- XGBoost: Feature subsampling, data sampling
- Train/test split randomization (if not time-based)

**Note:** Our train/test split is time-based (last N periods), so the seed mainly affects model internals.

---

## Preprocessing & Feature Engineering

### Design Philosophy

The preprocessing module (`backend/preprocessing.py`) adds **generic features** that improve forecasting for all algorithms. It follows these principles:

1. **User's promo columns are preserved as-is** - Binary indicators are already well-structured
2. **Only universally useful features are added** - Calendar, trend, and conditional YoY lags
3. **No redundant derived features** - Models learn directly from user's holiday indicators
4. **Conditional features based on data availability** - YoY lags only added if 1+ year of data

### Features Added

#### Calendar Features (Always Added)

| Feature | Description |
|---------|-------------|
| `day_of_week` | 0=Monday, 6=Sunday |
| `is_weekend` | 1 if Saturday/Sunday |
| `month` | 1-12 |
| `quarter` | 1-4 |
| `day_of_month` | 1-31 |
| `week_of_year` | 1-52 |

#### Trend Features (Always Added)

| Feature | Description |
|---------|-------------|
| `time_index` | Sequential index (0, 1, 2, ...) |
| `year` | Year from date |

#### YoY Lag Features (Conditional)

Only added if sufficient historical data exists:

| Frequency | Min Rows | Features Added |
|-----------|----------|----------------|
| Daily | 400+ | `lag_364`, `lag_364_avg` |
| Weekly | 60+ | `lag_52`, `lag_52_avg` |
| Monthly | 15+ | `lag_12`, `lag_12_avg` |

### How It's Applied Per Model

#### Prophet
```python
# In train_service.py
df = enhance_features_for_forecasting(df, date_col='ds', target_col='y',
                                       promo_cols=covariates, frequency=frequency)

# Calendar + trend + YoY features added as regressors (if they have valid values)
for col in derived_cols:
    if train_df[col].notna().any():
        model.add_regressor(col)
```

#### XGBoost
```python
# In models_training.py - create_xgboost_features()
# Calendar features
df['day_of_week'] = df['ds'].dt.dayofweek
df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)
df['month'] = df['ds'].dt.month

# Short-term lags (XGBoost-specific, not in preprocessing.py)
df['lag_1'] = df[target_col].shift(1)
df['lag_7'] = df[target_col].shift(7)
df['rolling_mean_7'] = df[target_col].rolling(window=7).mean()

# YoY lags (same as preprocessing)
df['lag_364'] = df[target_col].shift(364)
```

#### SARIMAX
```python
# In models_training.py - train_sarimax_model()
# Uses user covariates directly + auto-adds country holidays
country_holidays = holidays.country_holidays(country)
train_df['is_holiday'] = train_df['ds'].apply(lambda x: 1 if x in country_holidays else 0)
```

### User Promo Columns

Your binary holiday indicators are used directly without modification:

```csv
date,Black Friday,Christmas,Valentine's Day,Easter
2024-11-29,1,0,0,0
2024-11-30,1,0,0,0
2024-12-25,0,1,0,0
```

Each column becomes a separate regressor - models learn distinct coefficients for each holiday's effect.

### Customizing Preprocessing

Edit `backend/preprocessing.py` to modify behavior:

```python
# Change minimum data requirements for YoY lags
lag_config = {
    'daily': {'lag': 364, 'min_rows': 400},   # Increase to 500 for stricter requirement
    'weekly': {'lag': 52, 'min_rows': 60},
    'monthly': {'lag': 12, 'min_rows': 15},
}

# Add additional calendar features
df['is_month_end'] = dates.dt.is_month_end.astype(int)
df['is_quarter_end'] = dates.dt.is_quarter_end.astype(int)
```

### Reproducibility

All preprocessing code is logged to MLflow:

1. **Inline in training code**: `reproducibility/training_code.py`
2. **Full module**: `reproducibility/preprocessing.py`

```python
import mlflow

# Download and reproduce
preprocessing_path = mlflow.artifacts.download_artifacts(
    run_id="your_run_id",
    artifact_path="reproducibility/preprocessing.py"
)
exec(open(preprocessing_path).read())
df = enhance_features_for_forecasting(your_data, ...)
```

### Troubleshooting

#### YoY lag features not appearing?
Check if you have enough data:
```python
# Daily data needs 400+ rows for lag_364
if len(df) < 400:
    print("Insufficient data for YoY lags - they will be skipped")
```

#### Feature has all NaN values?
The system automatically filters out features with all-NaN values before training:
```python
# Only features with valid values are used
if train_df[col].notna().any():
    covariates.append(col)
```

#### Model fails with "all values are NaN"?
This typically means a covariate column exists but has no valid values in the training split. The preprocessing now automatically skips these columns and logs a warning.

---

## Model Settings

### How to switch Prophet from additive to multiplicative?

#### Via UI:
Select **"Multiplicative"** or **"Additive"** in the **Seasonality Mode** dropdown in the configuration panel before training. The dropdown is located below the Frequency/Horizon row.

#### Via API:
```json
{
  "seasonality_mode": "multiplicative"  // or "additive"
}
```

#### When to use each:

| Mode | When to Use | Example |
|------|-------------|---------|
| **Multiplicative** | Seasonality scales with trend | Sales that grow 20% in December regardless of base level |
| **Additive** | Seasonality is constant | Fixed +1000 units every December |

**Rule of thumb:** If your seasonal peaks get bigger as your overall values increase, use multiplicative.

### How to access more model settings?

Currently exposed settings in the UI:
- `frequency`: daily/weekly/monthly
- `horizon`: forecast periods
- `seasonality_mode`: additive/multiplicative
- `random_seed`: reproducibility seed
- `start_date` / `end_date`: training data date range
- `regressor_method`: how to handle future covariates (mean/last_value/linear_trend)

To expose additional Prophet parameters, modify `backend/models.py`:

```python
class TrainRequest(BaseModel):
    # Add new parameters
    changepoint_prior_scale: float = Field(default=0.05, description="Flexibility of trend changes (0.001-0.5)")
    seasonality_prior_scale: float = Field(default=10.0, description="Strength of seasonality (0.01-10)")
    yearly_seasonality: bool = Field(default=True, description="Enable yearly seasonality")
    weekly_seasonality: bool = Field(default=True, description="Enable weekly seasonality")
    daily_seasonality: bool = Field(default=False, description="Enable daily seasonality")
```

Then pass them through in `backend/train_service.py`:

```python
model = Prophet(
    seasonality_mode=seasonality_mode,
    changepoint_prior_scale=request.changepoint_prior_scale,
    seasonality_prior_scale=request.seasonality_prior_scale,
    yearly_seasonality=request.yearly_seasonality,
    # ...
)
```

---

## Scaling & Performance

### How to run multiple forecasts (different data cuts) faster?

Currently the UI processes one forecast at a time. Here are options to parallelize:

#### Option 1: Batch API Endpoint ✅ IMPLEMENTED

The batch training endpoint is now available at `/api/train-batch`. It processes multiple training requests in parallel.

**Request:**
```json
POST /api/train-batch
{
  "requests": [
    {
      "data": [...],
      "time_col": "ds",
      "target_col": "y",
      "horizon": 12,
      "filters": {"segment": "US"}
    },
    {
      "data": [...],
      "time_col": "ds",
      "target_col": "y",
      "horizon": 12,
      "filters": {"segment": "EU"}
    }
  ],
  "max_workers": 4
}
```

**Response:**
```json
{
  "total_requests": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "segment_id": "segment=US",
      "status": "success",
      "result": { /* TrainResponse */ },
      "filters": {"segment": "US"}
    },
    ...
  ]
}
```

**Note:** `max_workers` is automatically limited by the `MLFLOW_MAX_WORKERS` environment variable to prevent resource exhaustion on Databricks Apps (4 vCPU limit).

#### Option 2: UI Batch Training Modal ✅ IMPLEMENTED

The batch training feature is now integrated into the UI via the `BatchTraining` and `BatchComparison` components.

**How to Use:**
1. Upload and configure your data in the main application
2. Click the purple **"Batch Training"** button in the CONFIG step header
3. Select one or more segment columns (e.g., `region`, `product`, `channel`)
4. Preview the unique segment combinations and row counts
5. Click "Train N Segments" to start batch training
6. View results with MAPE statistics (min, max, mean, median)
7. Click "Done - Compare with Actuals" to proceed to batch comparison scorecard

**Key Components:**
- `components/BatchTraining.tsx` - Multi-select segment columns, progress tracking, results table
- `components/BatchComparison.tsx` - Upload actuals, generate scorecard, filter by status
- `services/databricksApi.ts` - `trainBatchOnBackend()` and `exportBatchResultsToCSV()` functions

**Features:**
- **Multi-column segmentation**: Select multiple columns to create segment combinations
- **Progress tracking**: Real-time progress bar with current segment indicator
- **MAPE statistics**: Min, max, mean, median across all segments
- **Status indicators**: Color-coded based on finance industry MAPE thresholds
- **Filter by status**: Focus on excellent, good, acceptable, review, or deviation segments
- **Export to CSV**: Download batch results and comparison scorecards

#### Option 3: CLI Script for Batch Processing ✅ IMPLEMENTED

The batch processing script is available at `scripts/batch_forecast.py`. It provides a full-featured CLI for parallel training with **multi-column segmentation support**.

**Basic Usage:**
```bash
# Single column segmentation
python scripts/batch_forecast.py --input data.csv --segment-cols region

# Multi-column segmentation (forecasts each unique combination)
python scripts/batch_forecast.py --input data.csv --segment-cols "region,product,channel"

# Forecast specific segments only (single-column mode)
python scripts/batch_forecast.py --input data.csv --segment-cols region --segments "US,EU,APAC"

# With custom configuration
python scripts/batch_forecast.py --input data.csv --segment-cols "category,store" \
    --time-col date --target-col revenue --horizon 12 --frequency monthly \
    --models "prophet,arima" --workers 4
```

**Multi-Column Segmentation Example:**

Given data like:
```
| region | product | channel | date       | revenue |
|--------|---------|---------|------------|---------|
| US     | Widget  | Online  | 2024-01-01 | 1000    |
| US     | Widget  | Retail  | 2024-01-01 | 800     |
| EU     | Gadget  | Online  | 2024-01-01 | 1200    |
```

Running `--segment-cols "region,product,channel"` will create separate forecasts for:
- `region=US | product=Widget | channel=Online`
- `region=US | product=Widget | channel=Retail`
- `region=EU | product=Gadget | channel=Online`
- ... (all unique combinations)

**Available Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | (required) | Input CSV file path |
| `--segment-cols`, `-s` | (required) | Comma-separated columns to segment by (e.g., "region,product") |
| `--output`, `-o` | `./batch_results` | Output directory |
| `--segments` | all | Specific segments (single-column mode only) |
| `--time-col` | `ds` | Time column name |
| `--target-col` | `y` | Target column name |
| `--covariates` | none | Comma-separated covariate columns |
| `--horizon` | `12` | Forecast periods |
| `--frequency` | `monthly` | daily/weekly/monthly |
| `--models` | `prophet` | Comma-separated model list |
| `--workers` | `4` | Parallel workers |
| `--api-url` | `http://localhost:8000` | Backend API URL |

**Output:**
- `batch_results_{timestamp}.json` - Detailed results with run IDs and filter metadata
- `batch_summary_{timestamp}.csv` - Summary with individual segment columns, metrics, and MAPE statistics

---

### Dynamic grouping fields from uploaded file ✅ FIXED

**Issue:** Grouping options showed columns from previously loaded file.

**Solution:** The `handleMainFileUpload` function now resets all column selections when a new file is uploaded:

```typescript
// In App.tsx handleMainFileUpload function
// Always reset column selections when new main file is uploaded
setTimeCol('');
setTargetCol('');
setGroupCols([]);
setCovariates([]);
setFilters({});
```

This ensures that when you load a new file with different headers, the UI properly resets and shows only columns from the new file.

---

## Data Integration

### Deployment Target: Databricks Apps

This application is designed to run as a **Databricks App** in production. This significantly simplifies authentication and SQL access:

| Concern | Local Development | Databricks Apps |
|---------|-------------------|-----------------|
| Authentication | Manual token in `.env.local` | Auto-inherited from workspace |
| SQL Access | Need warehouse ID + token | WorkspaceClient auto-configured |
| Permissions | Token must have all permissions | User's workspace permissions apply |
| Secrets | `.env.local` file | Databricks Secrets or app.yaml env vars |

### How to connect directly to SQL?

**Important:** SQL access is much simpler on Databricks Apps because authentication is automatic.

#### Step 1: Add SQL Warehouse ID to Configuration

**For local development** - add to `.env.local`:
```bash
DATABRICKS_SQL_WAREHOUSE_ID=your_warehouse_id_here
```

**For Databricks Apps** - add to `app.yaml`:
```yaml
env:
  - name: DATABRICKS_SQL_WAREHOUSE_ID
    value: "your_warehouse_id_here"
```

#### Step 2: Add SQL Query Endpoint

Add to `backend/main.py`:

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

@app.post("/api/query-data")
async def query_data(query: str, warehouse_id: str = None):
    """Execute SQL query and return data for forecasting

    Note: On Databricks Apps, WorkspaceClient() auto-configures authentication.
    Locally, it uses DATABRICKS_HOST and DATABRICKS_TOKEN from environment.
    """
    client = WorkspaceClient()

    # Use default warehouse or specified one
    warehouse_id = warehouse_id or os.getenv("DATABRICKS_SQL_WAREHOUSE_ID")

    if not warehouse_id:
        raise HTTPException(400, "SQL Warehouse ID not configured. Set DATABRICKS_SQL_WAREHOUSE_ID.")

    # Execute query
    response = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=query,
        wait_timeout="5m"
    )

    if response.status.state == StatementState.SUCCEEDED:
        # Convert to list of dicts
        columns = [col.name for col in response.manifest.schema.columns]
        data = [dict(zip(columns, row)) for row in response.result.data_array]
        return {"data": data, "columns": columns}
    else:
        raise HTTPException(500, f"Query failed: {response.status.error}")
```

#### Option 2: Saved Queries

Store commonly used queries and let users select:

```python
SAVED_QUERIES = {
    "weekly_volume_by_segment": """
        SELECT
            DATE_TRUNC('week', order_date) as ds,
            segment,
            SUM(volume) as y,
            MAX(is_promo) as promo
        FROM orders
        WHERE order_date >= DATEADD(year, -2, CURRENT_DATE)
        GROUP BY 1, 2
        ORDER BY 1, 2
    """,
    "daily_revenue": """
        SELECT
            order_date as ds,
            SUM(revenue) as y
        FROM orders
        GROUP BY 1
        ORDER BY 1
    """
}

@app.get("/api/saved-queries")
async def list_saved_queries():
    return {"queries": list(SAVED_QUERIES.keys())}

@app.post("/api/run-saved-query/{query_name}")
async def run_saved_query(query_name: str, params: dict = {}):
    query = SAVED_QUERIES.get(query_name)
    if not query:
        raise HTTPException(404, f"Query not found: {query_name}")

    # Substitute parameters
    for key, value in params.items():
        query = query.replace(f"{{{key}}}", str(value))

    return await query_data(query)
```

#### Frontend Integration

Add a "Load from SQL" button in the UI:

```typescript
const loadFromSQL = async () => {
  const response = await fetch('/api/run-saved-query/weekly_volume_by_segment');
  const { data } = await response.json();
  setData(data);
  setStep(AppStep.CONFIG);
};
```

---

## Production Deployment

### Can this run on Databricks instead of locally?

**Yes!** The app is designed for Databricks Apps deployment.

#### Deployment Steps:

1. **Build frontend:**
   ```bash
   npm run build
   ```

2. **Deploy to Databricks:**
   ```bash
   databricks bundle deploy
   ```

3. **Access via Databricks workspace:**
   Navigate to **Apps** → **finance-forecast-app**

#### What happens on Databricks:
- App runs on Databricks serverless compute
- Authentication is automatic (inherits workspace credentials)
- MLflow tracking goes to workspace MLflow
- Models register in Unity Catalog
- No local setup needed for end users

### Automatic data ingestion and model updates

#### Option 1: Databricks Workflow (Recommended)

Create a scheduled job that:
1. Queries latest data
2. Retrains models
3. Registers new model version

```python
# notebooks/scheduled_retrain.py
from databricks.sdk import WorkspaceClient
import requests
import json

def retrain_model():
    # 1. Query latest data
    client = WorkspaceClient()

    query = """
        SELECT date as ds, SUM(volume) as y, MAX(promo) as promo
        FROM orders
        WHERE date >= DATEADD(year, -2, CURRENT_DATE)
        GROUP BY date
        ORDER BY date
    """

    result = client.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=query
    )

    data = [dict(zip(columns, row)) for row in result.result.data_array]

    # 2. Call training API
    response = requests.post(
        f"{APP_URL}/api/train",
        json={
            "data": data,
            "time_col": "ds",
            "target_col": "y",
            "covariates": ["promo"],
            "horizon": 12,
            "frequency": "weekly",
            "models": ["prophet"]
        }
    )

    result = response.json()
    print(f"New model version: {result['run_id']}")

    # 3. Optionally auto-deploy
    # requests.post(f"{APP_URL}/api/deploy", json={...})

# Schedule this notebook to run every Monday at 6 AM
```

#### Option 2: Delta Live Tables Pipeline

For more sophisticated pipelines with data quality checks:

```python
import dlt
from pyspark.sql.functions import *

@dlt.table(
    comment="Weekly aggregated forecast input"
)
def forecast_input():
    return (
        spark.table("orders")
        .groupBy(date_trunc("week", "order_date").alias("ds"))
        .agg(
            sum("volume").alias("y"),
            max("is_promo").alias("promo")
        )
        .orderBy("ds")
    )

@dlt.table(
    comment="Forecast results"
)
def forecast_output():
    # Trigger model training and return results
    # This would call your training API
    pass
```

### Model Versioning

**Yes, automatic versioning!**

Every training run:
1. Creates a new MLflow run with unique ID
2. Logs all artifacts (data, model, code)
3. Can register to Unity Catalog with version number

```python
# View model versions
import mlflow
mlflow.set_registry_uri("databricks-uc")

client = mlflow.MlflowClient()
versions = client.search_model_versions("filter='name=\"main.default.finance_forecast_model\"'")

for v in versions:
    print(f"Version {v.version}: {v.creation_timestamp} - {v.current_stage}")
```

To compare versions:
```python
# Load specific version
model_v1 = mlflow.pyfunc.load_model("models:/main.default.finance_forecast_model/1")
model_v2 = mlflow.pyfunc.load_model("models:/main.default.finance_forecast_model/2")

# Compare predictions
pred_v1 = model_v1.predict(test_data)
pred_v2 = model_v2.predict(test_data)
```

---

## UI Improvements

### Covariate support is already indicated in the UI

The model selection section already shows which models support covariates:

- **Prophet**: Shows green badge "✓ Supports Covariates"
- **ARIMA**: Shows orange badge "Univariate Only"
- **Exponential Smoothing**: Shows orange badge "Univariate Only"
- **SARIMAX**: Shows green badge "✓ Supports Covariates"
- **XGBoost**: Shows green badge "✓ Supports Covariates"

This is implemented in the Models to Train & Compare section in `App.tsx` (lines ~1700-1750).

**Optional Enhancement:** Add a warning when covariates are selected but univariate models are chosen:

```typescript
{selectedModels.includes('arima') && covariates.length > 0 && (
  <div className="text-amber-600 text-xs mt-1">
    ⚠️ ARIMA ignores covariates - will use target series only
  </div>
)}
```

---

## Actuals vs Forecast Comparison

### How to automate weekly actuals comparison?

#### Option 1: SQL-Based Actuals Loading

Add an endpoint to load actuals from SQL:

```python
@app.post("/api/load-actuals")
async def load_actuals(
    query: str = None,
    start_date: str = None,
    end_date: str = None
):
    """Load actuals data from SQL for comparison"""

    if not query:
        # Default query for last week's actuals
        query = f"""
            SELECT
                date as ds,
                SUM(actual_volume) as actual
            FROM actuals_table
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY date
            ORDER BY date
        """

    # Execute and return
    result = execute_sql(query)
    return {"actuals": result}
```

#### Option 2: Automated Weekly Comparison Job

Create a scheduled notebook:

```python
# notebooks/weekly_forecast_scorecard.py

from datetime import datetime, timedelta
import pandas as pd

def generate_weekly_scorecard():
    # Get last week's date range
    today = datetime.now()
    last_monday = today - timedelta(days=today.weekday() + 7)
    last_sunday = last_monday + timedelta(days=6)

    # Load forecast made last Monday
    forecast_query = f"""
        SELECT * FROM forecast_log
        WHERE forecast_date = '{last_monday.strftime('%Y-%m-%d')}'
    """
    forecast_df = spark.sql(forecast_query).toPandas()

    # Load actual results
    actuals_query = f"""
        SELECT date as ds, SUM(volume) as actual
        FROM orders
        WHERE date BETWEEN '{last_monday}' AND '{last_sunday}'
        GROUP BY date
    """
    actuals_df = spark.sql(actuals_query).toPandas()

    # Compare
    comparison = forecast_df.merge(actuals_df, on='ds')
    comparison['error'] = comparison['actual'] - comparison['yhat']
    comparison['mape'] = abs(comparison['error'] / comparison['actual']) * 100

    # Calculate metrics
    overall_mape = comparison['mape'].mean()
    bias = comparison['error'].mean()

    # Log to tracking table
    spark.sql(f"""
        INSERT INTO forecast_scorecard VALUES (
            '{last_monday}',
            {overall_mape},
            {bias},
            '{comparison.to_json()}'
        )
    """)

    # Send alert if MAPE > threshold
    if overall_mape > 15:
        send_alert(f"Weekly MAPE ({overall_mape:.1f}%) exceeded threshold!")

    return comparison

# Run and display
scorecard = generate_weekly_scorecard()
display(scorecard)
```

#### Option 3: UI Enhancement for Easy Comparison

Add a "Quick Compare" feature:

```typescript
// Add to App.tsx
const quickCompareLastWeek = async () => {
  // Load actuals from SQL for the forecast period
  const response = await fetch('/api/load-actuals', {
    method: 'POST',
    body: JSON.stringify({
      start_date: forecastStartDate,
      end_date: forecastEndDate
    })
  });

  const { actuals } = await response.json();

  // Automatically populate comparison
  setActualsData(actuals);
  setShowComparison(true);
};

// Button in UI
<button onClick={quickCompareLastWeek}>
  📊 Compare with Last Week's Actuals
</button>
```

### Actuals Comparison UI Features ✅ IMPLEMENTED

The UI now includes several features for analyzing forecast accuracy:

#### Severity Filtering

Filter comparison results by forecast accuracy severity level:

| Severity | MAPE Range | Color | Use Case |
|----------|-----------|-------|----------|
| **Excellent** | ≤5% | Green | Highly accurate periods |
| **Good** | 5-10% | Blue | Acceptable accuracy |
| **Acceptable** | 10-15% | Yellow | Minor deviations |
| **Review** | 15-25% | Orange | Investigate these periods |
| **Deviation** | >25% | Red | Significant misses requiring attention |

**Usage:**
- Click severity buttons to toggle filters (multi-select supported)
- Shows count for each severity level
- "Clear filters" button to reset all filters
- Row count indicator shows filtered vs total periods

#### Duplicate Date Handling

When actuals data contains multiple rows for the same date (e.g., multiple transactions per day), values are automatically summed:

```typescript
// Values for duplicate dates are aggregated
const existingValue = actualsMap.get(dateKey) || 0;
actualsMap.set(dateKey, existingValue + Number(row[valueCol]));
```

A console warning is logged when duplicates are detected to help with debugging.

#### Filter Persistence Across Model Changes

When switching between models in the comparison view, the system maintains filter state:
- Uses `filteredActualsForComparison` state to store the filtered actuals data
- When model changes, comparison re-runs using the same filtered data
- Prevents loss of data filters when comparing different models

---

## Adding New Models

### How hard is it to add new models?

**Moderate difficulty.** The codebase is structured to make this straightforward.

#### Steps to add a new model (e.g., NeuralProphet):

##### 1. Add model type to types

In `types.ts`:
```typescript
export type ModelType = 'prophet' | 'arima' | 'exponential_smoothing' | 'sarimax' | 'xgboost' | 'neuralprophet';
```

In `backend/models.py`:
```python
models: List[str] = Field(
    default=["prophet"],
    description="Models: 'prophet', 'arima', 'exponential_smoothing', 'sarimax', 'xgboost', 'neuralprophet'"
)
```

##### 2. Create training function

In `backend/models_training.py`:

```python
def train_neuralprophet(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    covariates: List[str],
    horizon: int,
    frequency: str,
    test_size: int,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Train NeuralProphet model"""
    from neuralprophet import NeuralProphet

    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Prepare data
    prophet_df = df[[time_col, target_col]].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    # Train/test split
    train_df = prophet_df.iloc[:-test_size]
    test_df = prophet_df.iloc[-test_size:]

    # Hyperparameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'epochs': [50, 100],
        'batch_size': [32, 64],
        'n_changepoints': [5, 10],
        'yearly_seasonality': [True, False],
        'weekly_seasonality': [True, False],
    }

    best_model = None
    best_mape = float('inf')
    best_params = {}

    # Grid search
    for params in generate_param_combinations(param_grid):
        model = NeuralProphet(
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            n_changepoints=params['n_changepoints'],
            yearly_seasonality=params['yearly_seasonality'],
            weekly_seasonality=params['weekly_seasonality'],
        )

        # Add covariates
        for cov in covariates:
            model.add_future_regressor(cov)

        # Fit model
        model.fit(train_df, freq=freq_map[frequency])

        # Validate
        val_forecast = model.predict(test_df)
        mape = calculate_mape(test_df['y'], val_forecast['yhat1'])

        if mape < best_mape:
            best_mape = mape
            best_model = model
            best_params = params

    # Cross-validation
    cv_results = time_series_cross_validate(
        y=prophet_df['y'].values,
        model_fit_fn=lambda train: fit_neuralprophet(train, best_params),
        model_predict_fn=lambda model, test: model.predict(test)['yhat1'],
        n_splits=3
    )

    # Generate forecast
    future = model.make_future_dataframe(
        prophet_df,
        periods=horizon,
        n_historic_predictions=len(prophet_df)
    )
    forecast = model.predict(future)

    # Compute prediction intervals
    forecast_lower, forecast_upper = compute_prediction_intervals(
        y_train=train_df['y'].values,
        y_pred_train=model.predict(train_df)['yhat1'].values,
        forecast_values=forecast['yhat1'].tail(horizon).values
    )

    return {
        'model': model,
        'forecast': forecast,
        'validation': val_forecast,
        'metrics': {
            'rmse': rmse,
            'mape': best_mape,
            'r2': r2,
            'cv_mape': cv_results['mean_mape'],
            'cv_mape_std': cv_results['std_mape']
        },
        'best_params': best_params
    }
```

##### 3. Add to main.py router

In `backend/main.py`, add to the model training logic:

```python
if 'neuralprophet' in request.models:
    try:
        result = train_neuralprophet(
            df=df,
            time_col=request.time_col,
            target_col=request.target_col,
            covariates=request.covariates,
            horizon=request.horizon,
            frequency=request.frequency,
            test_size=test_size,
            random_seed=request.random_seed
        )
        # Log to MLflow, create response, etc.
        results.append(create_model_result('neuralprophet', result))
    except Exception as e:
        logger.error(f"NeuralProphet training failed: {e}")
```

##### 4. Add dependencies

In `requirements.txt`:
```
neuralprophet>=0.6.0
torch>=2.0.0
```

##### 5. Add pyfunc wrapper for serving

Create a model wrapper class similar to existing models:

```python
class NeuralProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, time_col, target_col, covariates, freq):
        self.model = model
        self.time_col = time_col
        self.target_col = target_col
        self.covariates = covariates
        self.freq = freq

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        if 'periods' in model_input.columns:
            periods = int(model_input['periods'].iloc[0])
            future = self.model.make_future_dataframe(
                self.model.history,
                periods=periods
            )
        else:
            future = model_input

        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat1', 'yhat1_lower', 'yhat1_upper']].rename(
            columns={'yhat1': 'yhat', 'yhat1_lower': 'yhat_lower', 'yhat1_upper': 'yhat_upper'}
        )
```

#### Estimated effort:
- Simple model (like NeuralProphet): 4-8 hours
- Complex model with custom serving logic: 1-2 days
- Model with new data requirements: 2-3 days

#### Tips for adding models:
1. Follow the existing pattern in `models_training.py`
2. Ensure CV and proper prediction intervals are implemented
3. Create an appropriate MLflow pyfunc wrapper
4. Add clear logging and error handling
5. Test locally before deploying

---

## Summary: Quick Reference

| Question | Short Answer |
|----------|--------------|
| Weekly vs daily forecast? | Yes, select in UI dropdown. Data frequency should match. |
| Prophet without covariates? | Yes, works fine - all models work without covariates. |
| When to change regressor method? | `last_value` (default) for recent-relevant, `mean` for stable metrics, `linear_trend` for trending |
| When to change seed? | Almost never. Only for ensemble testing or robustness checks. |
| Change seasonality mode? | Yes, select Multiplicative/Additive in UI dropdown. |
| Run multiple forecasts faster? | Add batch API endpoint or use CLI script (see Scaling section) |
| SQL integration? | Add `/api/query-data` endpoint - auth is automatic on Databricks Apps |
| Auto data ingestion? | Use Databricks scheduled workflow with model versioning |
| Dynamic grouping fields? | ✅ Fixed - columns reset automatically when new file is uploaded |
| Deploy to Databricks? | Yes, use `databricks bundle deploy` - no local setup for users |
| More model settings? | Some exposed in UI; add more via `TrainRequest` model |
| Clarify covariate support? | Already shown in UI - green "Supports Covariates" / orange "Univariate Only" badges |
| Automate actuals comparison? | Create weekly scorecard job (code provided above) |
| Filter comparison by severity? | ✅ Multi-select filter buttons for Excellent/Good/Acceptable/Review/Deviation |
| Add new models? | Moderate effort (4-8 hours), follow existing patterns in `models_training.py` |

---

## Implementation Priority (for Databricks Apps)

### Tier 1: Quick Wins ✅ COMPLETED
- ✅ Fix dynamic grouping fields bug - columns now reset on new file upload
- ✅ Add covariate warning for univariate models - warning shown when covariates selected with ARIMA/ETS
- ✅ CLI batch processing script - `scripts/batch_forecast.py` for parallel training

### Tier 2: Batch Processing ✅ COMPLETED
- ✅ Batch API endpoint `/api/train-batch` - parallel training of multiple segments
- ✅ Auto-aggregation `/api/aggregate` - convert daily data to weekly/monthly
- ✅ **Batch Training UI** - `components/BatchTraining.tsx` with multi-column segmentation, progress tracking, MAPE statistics
- ✅ **Batch Comparison Scorecard** - `components/BatchComparison.tsx` for comparing batch forecasts against actuals across all segments

### Tier 3: SQL Integration (3-5 days)
- SQL query endpoint (auth is automatic on Apps!)
- Saved queries for common data pulls
- Load actuals from SQL button

### Tier 4: Advanced UI (1 week)
- Multi-select groups for parallel training
- More Prophet parameters exposed
- User-configurable executive summary prompts

### Tier 5: Automation (separate infrastructure)
- Scheduled retrain workflows
- Weekly scorecard automation

---

## Recent Changes (December 2024)

### Completed Features

| Feature | Files Changed | Description |
|---------|---------------|-------------|
| **Simplified Preprocessing** | `preprocessing.py`, `train_service.py`, `models_training.py` | Generic calendar + trend features, conditional YoY lags |
| **Consistent Preprocessing** | `preprocessing.py`, `models_training.py` | All models use same preprocessing approach, no duplicates |
| **Conditional YoY Lags** | `preprocessing.py` | Only added when 1+ year of data exists |
| **User Covariates Preserved** | All model files | Binary holiday indicators used as-is without modification |
| **Reproducibility Logging** | `train_service.py` | Preprocessing code logged to MLflow artifacts |
| **Segment Exclusion** | `BatchTraining.tsx` | Click segments to exclude before training |
| **Real MAPE Calculation** | `BatchComparison.tsx` | Compares forecast values vs uploaded actuals |
| **Bias Metric** | `BatchComparison.tsx` | Shows under/over-forecast direction |
| **Holiday Calendar** | `models_training.py` | SARIMAX auto-adds country holidays |
| **Auto-open Comparison** | `App.tsx` | Opens comparison modal after batch completes |
| **localStorage Persistence** | `App.tsx` | Batch results survive page refresh |
| **Confirmation Dialog** | `BatchTraining.tsx` | Warns before losing unsaved results |
| **UX Improvements** | `App.tsx` | Data validation, metric tooltips, error messages |
| **Filter Bug Fix** | `App.tsx` | Empty "All (Aggregated)" filter now works |
| **MLflow Deprecation Fix** | `models_training.py`, `train_service.py` | `artifact_path` → `name` |

### Test Coverage Status

**Current:** ~15% (1 test file: `backend/test_future_features.py`)

**Needed Tests:**
- [ ] Backend endpoint tests (`/api/train`, `/api/train-batch`)
- [ ] Model training function tests (all 5 models)
- [ ] Frontend component tests (BatchTraining, BatchComparison)
- [ ] CSV parser edge cases
- [ ] Holiday calendar integration tests

---

**Questions?** Open an issue or reach out to the maintainers.
