# Databricks Finance Forecasting Platform

<div align="center">
  <img src="https://img.shields.io/badge/Databricks-Apps-FF3621?style=for-the-badge&logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Prophet-4285F4?style=for-the-badge" />
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge" />
</div>

---

## Quick Links

| I want to... | Go to... |
|-------------|----------|
| **Start forecasting now** | [Quick Start](#quick-start-recommended) |
| **Understand the architecture** | [Architecture Diagrams](ARCHITECTURE.md) |
| **Understand the modes (Simple vs Expert)** | [USER_GUIDE.md](USER_GUIDE.md#-understanding-the-modes) |
| **Learn step-by-step usage** | [USER_GUIDE.md](USER_GUIDE.md) |
| **Extend or deploy the platform** | [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) |
| **Add a new forecasting model** | [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#-adding-new-models) |
| **Troubleshoot an issue** | [USER_GUIDE.md](USER_GUIDE.md#-troubleshooting) |

---

## Overview

A comprehensive **prototype and reference implementation** for an AI-powered time series forecasting application built on Databricks. This accelerator demonstrates how to combine modern data science workflows (MLflow, Unity Catalog, Databricks Apps) with an intuitive notebook-style UI to make financial forecasting accessible to business users.

### Key Features

*   **Simple Mode (NEW!)**: Autopilot forecasting for finance users - upload data, get forecast. No ML knowledge required.
*   **AutoML Forecasting Paradigm**: Automatically trains and compares multiple model types (Prophet, ARIMA, Exponential Smoothing, SARIMAX, XGBoost) to find the best fit for your data.
*   **World-Class Holiday Handling (v1.3.0)**:
    - Multi-day holiday effect windows for Prophet (Thanksgiving -1 to +3 days, Christmas -7 to +1 days, etc.)
    - Daily proximity features (`days_to_christmas`, `days_since_thanksgiving`, `is_black_friday_window`)
    - Automatic Easter, Super Bowl, Mother's Day, Father's Day date calculations
*   **Smart ARIMA Model Selection**: Excludes degenerate random walk models (0,1,0) that produce flat forecasts
*   **Time Series Cross-Validation**: Expanding window CV for robust metric estimation (not just single holdout).
*   **Statistical Prediction Intervals**: Residual-based confidence bounds using t-distribution (not arbitrary ±10%).
*   **Parallel Hyperparameter Tuning**: Leverages multi-threading to optimize model parameters efficiently.
*   **AI-Powered Analysis**: Uses Databricks Foundation Models for dataset analysis and executive summaries.
*   **One-Click Deployment**: Deploy best-performing model to Databricks Model Serving for real-time inference.
*   **Interactive UI**: React-based frontend for data upload, configuration, and visualization.
*   **Forecast vs Actuals Comparison**: Compare predictions against actuals with finance industry MAPE thresholds.
*   **Batch Training**: Train forecasts for multiple segments (e.g., region × product × channel) in parallel with progress tracking and MAPE statistics.
*   **Batch Comparison Scorecard**: Compare batch forecasts against actuals across all segments with status breakdown.
*   **Full Reproducibility**: Logs exact training datasets, random seeds, and generates reproducible Python code.
*   **Enterprise Governance**: Full integration with Unity Catalog and MLflow experiment tracking.
*   **Export Capabilities**: Download comparison reports (CSV), batch results, scorecards, Excel with formulas, and executive summaries (Markdown).

---

## Project Structure

```
databricks-forecast-for-finance/
├── App.tsx                        # Main React application
├── index.tsx                      # React entry point
├── index.html                     # HTML template
├── types.ts                       # TypeScript type definitions
├── vite.config.ts                 # Vite build configuration
├── tsconfig.json                  # TypeScript configuration
├── package.json                   # Node.js dependencies
├── requirements.txt               # Python dependencies
│
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── schemas.py                 # Pydantic request/response models
│   ├── preprocessing.py           # Feature engineering for holiday/weekend forecasting
│   ├── ai_service.py              # AI analysis via Foundation Models
│   ├── deploy_service.py          # Model deployment to serving endpoints
│   ├── models/                    # Model implementations
│   │   ├── __init__.py
│   │   ├── prophet.py             # Prophet model training
│   │   ├── arima.py               # ARIMA/SARIMAX model training
│   │   ├── ets.py                 # Exponential Smoothing training
│   │   └── xgboost.py             # XGBoost model training
│   ├── simple_mode/               # Simple Mode - Autopilot forecasting
│   │   ├── __init__.py
│   │   ├── api.py                 # Simple mode API endpoints
│   │   ├── data_profiler.py       # Auto-detect frequency, columns, quality
│   │   ├── autopilot_config.py    # Generate optimal config
│   │   ├── forecast_explainer.py  # Excel-level transparency
│   │   └── excel_exporter.py      # Export with formulas
│   └── tests/                     # Backend tests
│       └── test_model_inference.py
│
├── components/
│   ├── ResultsChart.tsx           # Main forecast visualization
│   ├── EvaluationChart.tsx        # Model evaluation charts
│   ├── ForecastTable.tsx          # Tabular forecast display
│   ├── CovariateImpactChart.tsx   # Feature importance visualization
│   ├── TrainTestSplitViz.tsx      # Train/test split visualization
│   ├── NotebookCell.tsx           # Code display component
│   ├── BatchTraining.tsx          # Batch training modal
│   ├── BatchResultsViewer.tsx     # Batch results view with deployment
│   └── BatchComparison.tsx        # Batch comparison scorecard
│
├── services/
│   ├── analysisService.ts         # Frontend data analysis
│   └── databricksApi.ts           # Databricks API client
│
├── utils/
│   └── csvParser.ts               # CSV parsing utilities
│
├── scripts/
│   ├── batch_forecast.py          # CLI for parallel batch training
│   └── validate_mlflow_data.py    # MLflow artifact validation
│
├── setup-local.sh                 # One-time setup script
├── start-local.sh                 # Start development servers
├── restart-clean.sh               # Clean restart (kills old processes)
├── deploy-to-databricks.sh        # Deployment helper script
│
├── databricks.yml                 # Databricks Asset Bundle config
├── app.yaml                       # Databricks App configuration
├── databricks_notebook_template.py # Standalone Databricks notebook example
├── .env.example                   # Environment template (copy to .env.local)
├── .env.local                     # Local environment variables (not in git)
│
├── README.md                      # This file (overview & quick start)
├── USER_GUIDE.md                  # Step-by-step guide for users
└── DEVELOPER_GUIDE.md             # Technical docs for developers
```

---

## Supported Models

| Model | Covariates Support | Auto-Preprocessing | Cross-Validation | Best For |
|-------|-------------------|-------------------|------------------|----------|
| **Prophet** | Yes | Calendar + Trend + YoY lags | Time series CV (prophet.diagnostics) | Complex seasonality, holidays, trend changes |
| **ARIMA** | No | None (univariate) | Expanding window CV | Short-term forecasts, stationary data |
| **Exponential Smoothing (ETS)** | No | None (univariate) | Expanding window CV | Clear trends and seasonality patterns |
| **SARIMAX** | Yes | User covariates + country holidays | Expanding window CV | Seasonal patterns with external regressors |
| **XGBoost** | Yes | Calendar + Trend + YoY lags + short-term lags | Expanding window CV | Non-linear patterns, feature-rich datasets |

---

## Hardware & Scaling Quick Reference

| Environment | Max Data Size | Max Batch Segments | Parallel Workers |
|-------------|--------------|-------------------|------------------|
| **Databricks Apps** (4 vCPU/12GB) | 50K rows/segment | 20 segments | 2-4 |
| **Local Dev** (16GB RAM) | 100K rows/segment | 50 segments | 4-8 |

**Key Limits:**
- Databricks Apps has a hard limit of **4 vCPU / 12GB RAM**
- Batch training processes segments **in parallel** (default 4 workers)
- Each segment takes ~30s-3min depending on data size and models
- For larger workloads, use the CLI script or pre-aggregate data

> **📖 See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#hardware--scaling-recommendations)** for detailed hardware recommendations, memory usage by model, and scaling strategies.

---

## Local Development Guide

### Prerequisites
*   **Node.js** 18+
*   **Python** 3.10+
*   **Databricks CLI** (Optional for local run, Required for deployment)
*   **OpenMP** (Required for XGBoost on macOS): `brew install libomp`

### Quick Start (Recommended)

1.  **Setup Environment**:
    ```bash
    ./setup-local.sh
    ```

2.  **Configure Credentials**:
    Edit `.env.local` and add your Databricks credentials:
    ```bash
    # Backend Authentication
    DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
    DATABRICKS_TOKEN=dapi...

    # Frontend Authentication (required for AI features)
    VITE_DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
    VITE_DATABRICKS_TOKEN=dapi...

    # MLflow Experiment Path
    MLFLOW_EXPERIMENT_NAME=/Users/your.email@company.com/finance-forecasting
    ```

3.  **Run Application**:
    ```bash
    ./start-local.sh
    ```
    Access the app at `http://localhost:3000`.

4.  **Clean Restart** (if things get stuck):
    ```bash
    ./restart-clean.sh
    ```

### Manual Setup

#### 1. Create `.env.local`

```bash
# Backend Authentication
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi...

# Frontend Authentication (required for AI features)
VITE_DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
VITE_DATABRICKS_TOKEN=dapi...

# MLflow Experiment Path
MLFLOW_EXPERIMENT_NAME=/Users/your.email@company.com/finance-forecasting

# Unity Catalog Settings (optional - has defaults)
UC_CATALOG_NAME=main
UC_SCHEMA_NAME=default
UC_MODEL_NAME_ONLY=finance_forecast_model
UC_MODEL_NAME=main.default.finance_forecast_model

# Performance & Tuning (optional - has defaults)
MLFLOW_MAX_WORKERS=2           # Number of parallel training jobs
PROPHET_MAX_COMBINATIONS=3     # Grid search limit for Prophet
ARIMA_MAX_COMBINATIONS=6       # Grid search limit for ARIMA
ETS_MAX_COMBINATIONS=4         # Grid search limit for ETS
SARIMAX_MAX_COMBINATIONS=4     # Grid search limit for SARIMAX
XGBOOST_MAX_COMBINATIONS=4     # Grid search limit for XGBoost
```

#### 2. Install Dependencies

**Frontend:**
```bash
npm install
```

**Backend:**
```bash
pip install -r requirements.txt
```

**XGBoost on macOS** (if you encounter OpenMP errors):
```bash
brew install libomp
```

#### 3. Run Application

**Start Backend (FastAPI):**
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

**Start Frontend (React):**
```bash
npm run dev
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABRICKS_HOST` | Yes | - | Databricks workspace URL |
| `DATABRICKS_TOKEN` | Yes (local) | - | Personal access token (auto on Apps) |
| `VITE_DATABRICKS_HOST` | Yes (local) | - | Frontend AI Gateway access |
| `VITE_DATABRICKS_TOKEN` | Yes (local) | - | Frontend AI Gateway token |
| `MLFLOW_TRACKING_URI` | No | `databricks` | MLflow tracking server |
| `MLFLOW_EXPERIMENT_NAME` | Yes | - | Experiment path (e.g., `/Users/you@company.com/finance-forecasting`) |
| `UC_CATALOG_NAME` | No | `main` | Unity Catalog catalog |
| `UC_SCHEMA_NAME` | No | `default` | Unity Catalog schema |
| `UC_MODEL_NAME_ONLY` | No | `finance_forecast_model` | Model name (without catalog/schema) |
| `UC_MODEL_NAME` | No | `main.default.finance_forecast_model` | Full 3-level model name |
| `MLFLOW_MAX_WORKERS` | No | `1` | Parallel training workers |
| `PROPHET_MAX_COMBINATIONS` | No | `3` | Prophet hyperparameter grid size |
| `ARIMA_MAX_COMBINATIONS` | No | `6` | ARIMA grid size |
| `ETS_MAX_COMBINATIONS` | No | `4` | ETS grid size |
| `SARIMAX_MAX_COMBINATIONS` | No | `8` | SARIMAX grid size |
| `XGBOOST_MAX_COMBINATIONS` | No | `4` | XGBoost grid size |

> **Note:** See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for detailed configuration tuning between local development and Databricks Apps (4 vCPU/12GB RAM limit).

### Vite Configuration (vite.config.ts)

```typescript
export default defineConfig({
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  plugins: [react()],
  envPrefix: ['VITE_'],
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'chart-vendor': ['recharts'],
        }
      }
    }
  }
});
```

### TypeScript Configuration (tsconfig.json)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
    "moduleResolution": "bundler",
    "paths": { "@/*": ["./*"] },
    "skipLibCheck": true,
    "allowImportingTsExtensions": true,
    "noEmit": true
  }
}
```

### Databricks App Configuration (app.yaml)

```yaml
name: finance-forecast-app
description: Finance Forecasting Platform with MLflow and Model Serving

command:
  - /bin/bash
  - -c
  - |
    pip install -r requirements.txt && \
    python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4

env:
  - name: MLFLOW_TRACKING_URI
    value: "databricks"
  - name: MLFLOW_EXPERIMENT_NAME
    value: "/Users/${DATABRICKS_USER}/finance-forecasting"
  - name: MLFLOW_MAX_WORKERS
    value: "2"
  - name: PROPHET_MAX_COMBINATIONS
    value: "3"
  - name: UC_CATALOG_NAME
    value: "main"
  - name: UC_SCHEMA_NAME
    value: "default"
  - name: UC_MODEL_NAME
    value: "main.default.finance_forecast_model"

permissions:
  - workspace_access
  - cluster_access
  - model_serving_access
  - unity_catalog_access
```

### Databricks Asset Bundle (databricks.yml)

```yaml
bundle:
  name: finance-forecast-app

resources:
  apps:
    finance-forecast-app:
      name: finance-forecast-app
      description: "Finance Forecasting Platform"
      command:
        - /bin/bash
        - -c
        - |
          pip install -r requirements.txt && \
          python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4

      env:
        - name: MLFLOW_TRACKING_URI
          value: "databricks"
        - name: UC_CATALOG_NAME
          value: "main"
        # ... additional env vars

targets:
  dev:
    mode: development
    default: true
    workspace:
      host: ${workspace_host}

  prod:
    mode: production
    workspace:
      host: ${workspace_host}
```

---

## Automatic Feature Preprocessing

### How It Works

The platform automatically adds generic features that improve forecasting accuracy across all covariate-supporting models. **Your promo/holiday columns are used as-is** - the system only adds complementary calendar and trend features.

### Design Philosophy

1. **User's promo columns are preserved as-is** - Binary indicators (0/1) for holidays/promos are already well-structured
2. **Only universally useful features are added** - Calendar features, trend indicators, and YoY lags (when applicable)
3. **No redundant derived features** - Models learn holiday effects directly from your indicators

### Features Added (for Prophet, SARIMAX, XGBoost)

| Category | Features | Description |
|----------|----------|-------------|
| **Calendar** | `day_of_week`, `is_weekend`, `month`, `quarter`, `day_of_month`, `week_of_year` | Captures day/week/month patterns |
| **Trend** | `time_index`, `year` | Helps XGBoost capture trends (Prophet/SARIMAX handle internally) |
| **YoY Lags** | `lag_364`, `lag_364_avg` (daily) / `lag_52` (weekly) / `lag_12` (monthly) | Same-period-last-year patterns (only if 1+ year of data) |

### Conditional YoY Lag Features

YoY lag features are **only added if sufficient historical data exists**:

| Frequency | Required Data | Lag Feature |
|-----------|---------------|-------------|
| Daily | 400+ rows (~1.1 years) | `lag_364`, `lag_364_avg` |
| Weekly | 60+ rows (~1.2 years) | `lag_52`, `lag_52_avg` |
| Monthly | 15+ rows (~1.25 years) | `lag_12`, `lag_12_avg` |

If you have less data, these features are automatically skipped to avoid all-NaN columns.

### Models That Benefit

| Model | Preprocessing Applied | Notes |
|-------|----------------------|-------|
| **Prophet** | Calendar + Trend + YoY lags | All features added as regressors |
| **SARIMAX** | User covariates + country holidays | Auto-adds `is_holiday` indicator |
| **XGBoost** | Calendar + Trend + YoY lags + short-term lags | Also adds `lag_1`, `lag_7`, `rolling_mean_7` |
| **ARIMA** | None | Univariate model - cannot use features |
| **ETS** | None | Univariate model - cannot use features |

### Your Promo File Structure

The system works optimally with sparse promo files containing binary indicators:

```csv
date,Black Friday,Christmas,Valentine's Day,...
2024-11-29,1,0,0,...
2024-11-30,1,0,0,...
2024-12-25,0,1,0,...
```

Each holiday column is used directly by Prophet/SARIMAX/XGBoost - they learn separate coefficients for each holiday's effect.

### Reproducibility

All preprocessing code is logged to MLflow artifacts:
- `reproducibility/training_code.py` - Includes inline preprocessing functions
- `reproducibility/preprocessing.py` - Full preprocessing module source

---

## Model Validation & Metrics

### Time Series Cross-Validation

All models use **expanding window cross-validation** for robust metric estimation:

```
Split 1: [===Train===][Test]
Split 2: [=====Train=====][Test]
Split 3: [=======Train=======][Test]
```

**Why CV matters:**
- Single holdout validation can give misleading metrics due to random variation
- CV MAPE provides a more reliable estimate of out-of-sample performance
- Standard deviation shows metric stability across different time periods

**Metrics returned:**
- `mape`: Validation MAPE (single holdout)
- `cv_mape`: Cross-validation MAPE (averaged across folds)
- `cv_mape_std`: Standard deviation of CV MAPE

### Statistical Prediction Intervals

Confidence intervals are computed using residual standard deviation and t-distribution:

```
PI = forecast ± t_{α/2,n-1} × σ_residuals × √(1 + 1/n)
```

This replaces arbitrary ±10% bounds with **statistically valid intervals** based on actual model uncertainty.

### MAPE Thresholds (Finance Industry Best Practices)

| Category | MAPE Range | Description |
|----------|------------|-------------|
| Excellent | ≤5% | Industry gold standard |
| Good | 5-10% | Reliable for planning |
| Acceptable | 10-15% | Suitable with caution |
| Needs Review | 15-25% | Investigate root causes |
| Significant Deviation | >25% | Requires attention |

---

## AutoML & Hyperparameter Tuning

### How It Works

1.  **Parallel Training**: The system trains five distinct model types simultaneously.

2.  **Hyperparameter Tuning**:
    *   **Prophet**: Grid search over `changepoint_prior_scale`, `seasonality_prior_scale`, `seasonality_mode`, `growth`.
    *   **ARIMA**: Grid search over `p` (0-2), `d` (0-1), `q` (0-1) parameters.
    *   **ETS**: Grid search over `trend` and `seasonal` components.
    *   **SARIMAX**: Grid search over `(p,d,q)` and seasonal `(P,D,Q,s)` parameters.
    *   **XGBoost**: Grid search over `max_depth`, `n_estimators`, `learning_rate`.

3.  **Model Selection**:
    *   All models evaluated on holdout validation set with cross-validation
    *   Compares MAPE, RMSE, R², and CV MAPE
    *   Best model (lowest CV MAPE) automatically recommended for deployment

---

## Backend API Reference

### Simple Mode Endpoints (NEW!)

Simple Mode provides autopilot forecasting for finance users who want Excel-like simplicity with ML accuracy.

#### POST /api/simple/profile

Profile uploaded data without running forecast. Returns auto-detected settings so user can review before forecasting.

**Request:** Multipart form with CSV/Excel file

**Response:**
```json
{
  "success": true,
  "profile": {
    "frequency": "weekly",
    "date_column": "week_start",
    "target_column": "revenue",
    "history_months": 18.5,
    "data_quality_score": 92.5,
    "holiday_coverage_score": 45.0,
    "recommended_models": ["prophet", "xgboost"],
    "recommended_horizon": 12
  },
  "warnings": [
    {
      "level": "medium",
      "message": "Only 18 months of data. Holiday forecasts may be less accurate.",
      "recommendation": "Provide 2+ years of data for best holiday accuracy."
    }
  ]
}
```

#### POST /api/simple/forecast

Upload data and get forecast automatically. All configuration is auto-detected.

**Request:** Multipart form with CSV/Excel file, optional `horizon` query parameter

**Response:**
```json
{
  "success": true,
  "mode": "simple",
  "run_id": "abc123",
  "summary": "Forecast Summary: Total $1.2M, Trend: +5.2%, Confidence: High",
  "forecast": [102000, 105000, ...],
  "dates": ["2025-01-06", "2025-01-13", ...],
  "components": {
    "formula": "Forecast = Base + Trend + Seasonality + Holiday Effect",
    "periods": [...]
  },
  "confidence": {
    "level": "high",
    "score": 85,
    "mape": 4.2
  },
  "audit": {
    "run_id": "abc123",
    "data_hash": "7f83b165...",
    "reproducibility_token": "7f83b165:2c26b46b:1.0"
  },
  "excel_download_url": "/api/simple/export/abc123/excel"
}
```

#### GET /api/simple/export/{run_id}/excel

Download forecast as Excel file with multiple sheets:
- Summary (executive view)
- Forecast Detail (with Excel formulas showing Base + Trend + Seasonal + Holiday)
- Components breakdown
- Confidence factors
- Audit trail (for compliance)
- Raw data

#### GET /api/simple/export/{run_id}/csv

Download forecast as simple CSV file.

#### POST /api/simple/reproduce/{run_id}

Reproduce an exact previous forecast using stored configuration and data hash.

---

### Core Endpoints

#### POST /api/train

Train forecasting models on uploaded data.

**Request Body:**
```json
{
  "data": [{"ds": "2023-01-01", "y": 1000, "marketing_spend": 500}],
  "time_col": "ds",
  "target_col": "y",
  "covariates": ["marketing_spend"],
  "horizon": 12,
  "frequency": "monthly",
  "seasonality_mode": "multiplicative",
  "test_size": 100,
  "regressor_method": "mean",
  "models": ["prophet", "arima", "exponential_smoothing", "sarimax", "xgboost"],
  "catalog_name": "main",
  "schema_name": "default",
  "model_name": "finance_forecast_model",
  "country": "US",
  "random_seed": 42,
  "from_date": "2023-01-01",
  "to_date": "2024-12-31",
  "future_features": [{"ds": "2025-01-01", "marketing_spend": 600}]
}
```

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `data` | array | Yes | - | Time series data rows |
| `time_col` | string | Yes | - | Name of date/time column |
| `target_col` | string | Yes | - | Name of target column |
| `covariates` | array | No | `[]` | List of covariate column names |
| `horizon` | int | No | `12` | Number of periods to forecast |
| `frequency` | string | No | `monthly` | `daily`, `weekly`, or `monthly` |
| `seasonality_mode` | string | No | `multiplicative` | `additive` or `multiplicative` |
| `test_size` | int | No | auto | Size of validation set |
| `regressor_method` | string | No | `mean` | `mean`, `last_value`, or `linear_trend` |
| `models` | array | No | `["prophet"]` | Models to train |
| `random_seed` | int | No | `42` | Random seed for reproducibility |
| `from_date` | string | No | - | Filter data from this date |
| `to_date` | string | No | - | Filter data to this date |
| `future_features` | array | No | - | Future covariate values |

**Response:**
```json
{
  "models": [
    {
      "model_type": "prophet",
      "model_name": "Prophet_20241204_123456",
      "run_id": "207dc64a1c8b46e7b58c8e4d1cfd0d02",
      "metrics": {
        "rmse": "125.43",
        "mape": "8.32",
        "r2": "0.92",
        "cv_mape": "6.09",
        "cv_mape_std": "0.91"
      },
      "validation": [...],
      "forecast": [...],
      "covariate_impacts": [...],
      "is_best": true,
      "experiment_url": "https://...",
      "run_url": "https://..."
    }
  ],
  "best_model": "Prophet_20241204_123456",
  "artifact_uri": "dbfs:/databricks/mlflow/..."
}
```

#### POST /api/deploy

Deploy a trained model to a serving endpoint.

**Request:**
```json
{
  "model_name": "main.default.finance_forecast_model",
  "model_version": "1",
  "endpoint_name": "finance-forecast-endpoint",
  "workload_size": "Small",
  "scale_to_zero": true
}
```

#### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "Backend is operational",
  "databricks_connected": true,
  "mlflow_enabled": true
}
```

### Analysis Endpoints

*   **`POST /api/analyze`**: Analyze uploaded dataset structure
*   **`POST /api/insights`**: Generate natural language insights
*   **`POST /api/executive-summary`**: Generate executive summary with root cause analysis

### Batch & Data Processing

*   **`POST /api/train-batch`**: Train multiple forecasts in parallel (see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md))
*   **`POST /api/deploy-batch`**: Deploy all batch-trained models to a single serving endpoint with automatic segment routing
*   **`POST /api/aggregate`**: Convert daily data to weekly/monthly before training

### Endpoint Management

*   **`GET /api/endpoints`**: List all serving endpoints
*   **`GET /api/endpoints/{name}/status`**: Get endpoint status
*   **`DELETE /api/endpoints/{name}`**: Delete an endpoint

---

## MLflow Artifacts & Reproducibility

### Logged Datasets

| Artifact Path | Description |
|--------------|-------------|
| `datasets/raw/original_timeseries_data.csv` | Original uploaded data |
| `datasets/raw/promotions_future_features.csv` | Promotions file (if provided) |
| `datasets/processed/full_merged_data.csv` | Merged data with preprocessing features |
| `datasets/training/train.csv` | Training split |
| `datasets/training/eval.csv` | Evaluation/validation split |
| `datasets/inference/input.csv` | Sample input for endpoint |
| `datasets/inference/output.csv` | Forecast output |
| `reproducibility/training_code.py` | Reproducible training script (includes preprocessing) |
| `reproducibility/preprocessing.py` | Full preprocessing module source code |

### Logged Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `frequency` | Data frequency | `monthly` |
| `horizon` | Forecast periods | `12` |
| `time_column` | Time column name | `date` |
| `target_column` | Target column name | `sales` |
| `covariates` | List of covariates | `['Black Friday', 'Christmas']` |
| `random_seed` | Seed for reproducibility | `42` |
| `cv_mape` | Cross-validation MAPE | `6.09` |
| `cv_mape_std` | CV MAPE standard deviation | `0.91` |

### Validation Script

Use the validation script to verify MLflow artifacts:

```bash
# List recent experiments
python scripts/validate_mlflow_data.py --list-only

# Validate specific run
python scripts/validate_mlflow_data.py --run-id <run_id>

# Download artifacts locally
python scripts/validate_mlflow_data.py --run-id <run_id> --download
```

The script verifies:
- Date continuity (train → eval → forecast with no gaps)
- Random seed logging
- CV metrics presence
- Training code accuracy

### Batch Processing Script

Run forecasts for multiple data segments in parallel:

```bash
# Basic: Forecast all segments
python scripts/batch_forecast.py --input data.csv --segment-col BUSINESS_SEGMENT

# Specify segments and configuration
python scripts/batch_forecast.py --input data.csv --segment-col region \
    --segments "US,EU,APAC" --time-col date --target-col revenue \
    --horizon 12 --frequency monthly --workers 4

# Use multiple models
python scripts/batch_forecast.py --input data.csv --segment-col category \
    --models "prophet,arima,xgboost" --output ./results
```

Results are saved to the output directory as JSON (detailed) and CSV (summary).

---

## Deployment to Databricks

### Prerequisites
*   Databricks Workspace with **Unity Catalog** and **Model Serving** enabled
*   **Databricks Apps** feature enabled

### Deployment Steps

1.  **Install Databricks CLI**:
    ```bash
    curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
    databricks auth login
    ```

2.  **Build Frontend**:
    ```bash
    npm run build
    ```

3.  **Deploy App**:
    ```bash
    databricks bundle validate
    databricks bundle deploy
    ```

4.  **Access App**:
    Navigate to **Apps** in Databricks workspace and open `finance-forecast-app`.

---

## Model Serving API Usage

### Prophet (Supports Covariates)

#### Mode 1: Simple Forecast

```json
{
  "dataframe_records": [
    {"ds": "2025-01-01", "periods": 12}
  ]
}
```

#### Mode 2: Advanced Forecast (with covariates)

```json
{
  "dataframe_records": [
    {"ds": "2025-11-28", "Black Friday": 1, "Christmas": 0},
    {"ds": "2025-12-25", "Black Friday": 0, "Christmas": 1}
  ]
}
```

### ARIMA & ETS (Univariate)

```json
{
  "dataframe_records": [
    {"periods": 12, "start_date": "2025-01-01"}
  ]
}
```

### SARIMAX & XGBoost (with covariates)

```json
{
  "dataframe_records": [
    {
      "periods": 12,
      "start_date": "2025-01-01",
      "covariates": {
        "Black Friday": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "Christmas": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
      }
    }
  ]
}
```

---

## Forecast vs Actuals Comparison

### Features
*   **Flexible File Upload**: Upload actuals in various CSV formats with column selection
*   **Multi-Segment Data Filtering**: Filter by business segment, region, etc.
*   **Severity Filter**: Toggle buttons to filter comparison table by error severity (Excellent, Good, Acceptable, Review, Deviation)
*   **Duplicate Date Handling**: Automatically aggregates (sums) values when actuals file has multiple rows per date
*   **Context Indicators**: Day of week, weekend flags, active promos/holidays
*   **Root Cause Analysis**: AI-powered executive summary identifies potential causes
*   **Export Capability**: Download comparison report as CSV

### Severity Filtering

Filter the comparison table to focus on anomalous forecasts:

| Severity | MAPE Range | Button Color |
|----------|------------|--------------|
| Excellent | ≤5% | Green |
| Good | 5-10% | Blue |
| Acceptable | 10-15% | Yellow |
| Review | 15-25% | Orange |
| Significant Deviation | >25% | Red |

- Click severity buttons to filter (multi-select supported)
- Each button shows count of periods in that category
- "Clear filters" link resets to show all periods
- Buttons are disabled when count is zero

### Understanding Errors
*   **Under-forecast** (blue): Actual > Predicted
*   **Over-forecast** (purple): Actual < Predicted
*   **Bias**: Positive = systematic under-forecasting, Negative = systematic over-forecasting

---

## Batch Training & Comparison

### Overview

The Batch Training feature allows you to train forecasts for multiple segments (e.g., region × product × channel) in a single workflow. Each unique combination of segment values gets its own trained model.

### How to Use Batch Training

1. **Upload and Configure**: Load your data and configure the forecast settings (time column, target, covariates, models, etc.)

2. **Click "Batch Training"**: In the CONFIG step, click the purple "Batch Training" button in the header

3. **Select Segment Columns**: Choose one or more columns that define your segments (e.g., `region`, `product_line`, `channel`)
   - Excluded columns: time column, target column, and selected covariates

4. **Preview Segments**: View the unique segment combinations and row counts

5. **Train**: Click "Train N Segments" to start sequential training for all segments

6. **View Results**: See MAPE statistics (min, max, mean, median) and per-segment results with status indicators

7. **Deploy All Models**: Click "Deploy All Models" to deploy all segment models to a single serving endpoint
   - Creates a router model that automatically routes requests to the correct segment model
   - Each model is tested with its logged input_example before deployment
   - Single endpoint handles all segments - just include segment identifiers in your request

8. **Export or Compare**:
   - Export results to CSV
   - Click "Done - Compare with Actuals" to proceed to batch comparison

### Batch Comparison Scorecard

After batch training, compare forecasts against actuals across all segments:

1. **Upload Actuals**: Upload a CSV file with actuals data that includes the same segment columns

2. **Column Mapping**: Select the date and value columns from your actuals file

3. **Generate Scorecard**: View overall MAPE and status distribution across all segments

4. **Filter & Export**: Filter by status and export the scorecard to CSV

### MAPE Thresholds (Finance Industry Standards)

| Status | MAPE Range | Interpretation |
|--------|------------|----------------|
| Excellent | ≤5% | Industry gold standard |
| Good | 5-10% | Good forecast accuracy |
| Acceptable | 10-15% | Within acceptable range |
| Review | 15-25% | Needs investigation |
| Significant Deviation | >25% | Requires immediate attention |

### CLI Batch Training

For automated pipelines, use the batch training CLI script:

```bash
python scripts/batch_forecast.py \
  --data-path data/sales.csv \
  --segment-cols region,product_line \
  --time-col date \
  --target-col revenue \
  --horizon 12 \
  --frequency monthly \
  --models prophet,xgboost
```

---

## Dependencies

### Python (requirements.txt)

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
prophet>=1.1.4
statsmodels>=0.14.0
pmdarima>=2.0.3
xgboost>=2.0.0
mlflow>=2.9.0
databricks-sdk>=0.12.0
pydantic>=2.0.0
requests>=2.31.0
holidays>=0.35
```

### Node.js (package.json)

```json
{
  "dependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "recharts": "^2.15.0",
    "lucide-react": "^0.454.0",
    "papaparse": "^5.4.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.4",
    "typescript": "~5.6.2",
    "vite": "^6.0.1"
  }
}
```

---

## Troubleshooting

### Common Issues

1.  **"Invalid URL 'inherit/...'**: Run locally without `.env.local`. App expects real URLs locally but uses "inherit" on Databricks.

2.  **Model Training Fails**: Ensure enough history (at least 2 full seasonal cycles for ETS/Prophet).

3.  **Deployment Timeouts**: Model serving endpoints can take 5-10 minutes. Check Serving tab.

4.  **XGBoost OpenMP Error on macOS**:
    ```bash
    brew install libomp
    ```

5.  **Port Already in Use**:
    ```bash
    ./restart-clean.sh
    # or manually:
    lsof -ti:8000 | xargs kill -9
    lsof -ti:3000 | xargs kill -9
    ```

6.  **No Matching Dates in Actuals Comparison**: This was fixed! The issue was Prophet generating Sunday-based weeks while actual data uses Monday-based weeks. The system now auto-detects the day-of-week from your training data and generates forecast dates that align correctly.

7.  **MLflow Connection Error**: Verify DATABRICKS_HOST and DATABRICKS_TOKEN in `.env.local`.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 19 + TypeScript + Vite | Interactive UI |
| **Backend** | FastAPI + Uvicorn | High-performance API |
| **ML Engine** | Prophet, Statsmodels, pmdarima, XGBoost | Forecasting Algorithms |
| **Tracking** | MLflow | Experiment Management |
| **Registry** | Unity Catalog | Model Governance |
| **Platform** | Databricks Apps | Hosting & Compute |
| **AI** | Databricks Foundation Models | Natural Language Analysis |

---

## Recent Updates (December 2024)

### New Features
- **Batch Deployment with Router Model**: Deploy all batch-trained models to a single serving endpoint with automatic segment routing
- **Pre-deployment Model Inference Testing**: Models are tested with their logged input_example before deployment to ensure inference works correctly
- **Simplified Preprocessing**: Generic calendar and trend features that work with any promo file structure
- **Conditional YoY Lags**: Year-over-year lag features only added when sufficient data exists (1+ year)
- **User Promo Columns Preserved**: Binary holiday indicators used directly without modification
- **Batch Training with Segment Exclusion**: Click on segments to exclude them from training before running
- **Real Actual vs Forecast MAPE**: BatchComparison now calculates true MAPE from uploaded actuals (not just training MAPE)
- **Bias Calculation**: Shows forecast bias direction (under-forecast vs over-forecast) in comparison scorecard
- **Country Holiday Support**: All models (Prophet, SARIMAX, XGBoost) now use country-specific holiday calendars
- **Auto-open Comparison**: After batch training completes, comparison modal opens automatically
- **localStorage Persistence**: Batch results survive page refresh - no more losing 1-hour training sessions
- **Confirmation Dialogs**: Warning before closing batch training with unsaved results
- **UX Improvements**: Data validation, metric tooltips, improved error messages

### Bug Fixes
- **Fixed "No overlapping dates" error in forecast vs actuals comparison** - Prophet was generating Sunday-based weeks while actual data used Monday-based weeks, causing a 1-day mismatch. Now auto-detects day-of-week from training data.
- Fixed model serving deployment failures by adding `holidays` dependency to all model pip requirements
- Fixed Prophet model failures with short datasets (less than 1 year of data)
- Fixed empty chart bug when "All (Aggregated)" filter was selected
- Fixed MLflow `artifact_path` deprecation warning (now uses `name`)
- Dynamic grouping fields now reset properly when loading new files
- Removed duplicate preprocessing code across model files

### Infrastructure
- Enhanced model inference testing using logged MLflow input_example for consistency
- Refactored `backend/preprocessing.py` for consistency across all models
- Preprocessing code logged to MLflow for 100% reproducibility
- Improved MLflow experiment organization with batch_id grouping
- Added segment-level error isolation in batch training

---

## Pending Work / Future Enhancements

| Feature | Priority | Status |
|---------|----------|--------|
| **SQL Query Import** | Medium | Not started - currently CSV upload only |
| **Auto-ingestion Scheduling** | Low | Documented workaround using Databricks Workflows |
| **Executive Summary Customization** | Low | System prompt editable in code, not exposed in UI |
| **Test Coverage** | High | Only 1 test file exists (~15% coverage) |
| **NeuralProphet Model** | Low | Template documented in DEVELOPER_GUIDE |

---

**Built for Finance Teams using Databricks**

**Created by Debu Sinha**
