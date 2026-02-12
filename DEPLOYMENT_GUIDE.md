# Finance Forecasting Platform - Deployment Guide

## Quick Deploy (One Command)

```bash
# Deploy to e2-dogfood staging
./scripts/deploy.sh --profile e2-dogfood --target dev

# Deploy to production
./scripts/deploy.sh --profile e2-dogfood --target prod
```

## Manual Deployment Steps

If the automated script fails, follow these steps:

### 1. Prerequisites

```bash
# Ensure Databricks CLI is authenticated
databricks auth login --profile e2-dogfood

# Verify authentication
databricks auth profiles
```

### 2. Create Dedicated Training Cluster

```bash
# Create cluster via CLI
databricks clusters create --profile e2-dogfood --json '{
  "cluster_name": "forecast-training-dev",
  "spark_version": "15.4.x-scala2.12",
  "node_type_id": "i3.16xlarge",
  "num_workers": 0,
  "spark_conf": {
    "spark.databricks.cluster.profile": "singleNode",
    "spark.master": "local[*]",
    "spark.executor.memory": "200g",
    "spark.driver.memory": "200g"
  },
  "custom_tags": {
    "project": "finance-forecasting",
    "cost_center": "analytics"
  },
  "autotermination_minutes": 30
}'

# Note the cluster_id from the output
```

### 3. Upload Training Notebook

```bash
# Create workspace directory
databricks workspace mkdirs /Workspace/finance-forecasting/notebooks --profile e2-dogfood

# Upload notebook
databricks workspace import notebooks/train_models.py \
  /Workspace/finance-forecasting/notebooks/train_models \
  --format SOURCE --language PYTHON --overwrite --profile e2-dogfood
```

### 4. Set Up Lakebase PostgreSQL

1. Go to Databricks UI > SQL > Lakebase
2. Create a new Lakebase instance named `forecast`
3. Run the SQL schema from `sql/create_tables.sql`
4. Note the connection details (host, database, user)

### 5. Deploy Databricks App

```bash
# Validate bundle
databricks bundle validate -t dev --profile e2-dogfood

# Deploy with cluster ID
databricks bundle deploy -t dev --profile e2-dogfood \
  --var "dedicated_cluster_id=<YOUR_CLUSTER_ID>" \
  --var "lakebase_host=<YOUR_LAKEBASE_HOST>"
```

### 6. Create Training Job

```bash
databricks jobs create --profile e2-dogfood --json '{
  "name": "forecast-training-job-dev",
  "tasks": [{
    "task_key": "train_all_models",
    "existing_cluster_id": "<YOUR_CLUSTER_ID>",
    "notebook_task": {
      "notebook_path": "/Workspace/finance-forecasting/notebooks/train_models"
    },
    "timeout_seconds": 7200,
    "retry_on_timeout": true,
    "max_retries": 1
  }],
  "queue": {"enabled": true},
  "max_concurrent_runs": 10,
  "tags": {
    "project": "finance-forecasting",
    "type": "ml-training"
  }
}'
```

## Environment Variables

The app uses these environment variables (configured in `databricks.yml`):

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server | `databricks` |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment path | `/Users/<deployer>/finance-forecasting` |
| `UC_CATALOG_NAME` | Unity Catalog catalog | `main` |
| `UC_SCHEMA_NAME` | Unity Catalog schema | `default` |
| `LAKEBASE_HOST` | Lakebase PostgreSQL host | (set per environment) |
| `LAKEBASE_DATABASE` | Lakebase database name | `forecast` |
| `DEDICATED_CLUSTER_ID` | Training cluster ID | (set after cluster creation) |
| `ENABLE_CLUSTER_DELEGATION` | Enable job delegation | `true` |

## Verification

```bash
# Check app status
databricks apps get finance-forecast-app --profile e2-dogfood

# Check cluster status
databricks clusters get <CLUSTER_ID> --profile e2-dogfood

# Check job status
databricks jobs list --profile e2-dogfood | grep forecast

# View app logs
databricks apps logs finance-forecast-app --profile e2-dogfood
```

## Troubleshooting

### CLI Timeout Issues

If Databricks CLI commands timeout:
1. Check VPN connection
2. Verify profile authentication: `databricks auth env --profile e2-dogfood`
3. Try direct workspace URL access in browser

### Bundle Validation Errors

```bash
# Verbose validation
databricks bundle validate -t dev --profile e2-dogfood --debug
```

### App Deployment Issues

```bash
# Check deployment logs
databricks apps logs finance-forecast-app --profile e2-dogfood --tail 100
```

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     Databricks Workspace                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐    ┌──────────────────────────────────────┐ │
│  │ Databricks App│    │   Dedicated Cluster (64 vCPU)        │ │
│  │ (FastAPI)     │───▶│   - Prophet, ARIMA, XGBoost          │ │
│  │ 4 vCPU/12GB   │    │   - StatsForecast, Chronos           │ │
│  └───────┬───────┘    │   - Ensemble + Conformal             │ │
│          │            └──────────────────────────────────────┘ │
│          │                                                      │
│  ┌───────▼───────┐    ┌──────────────────────────────────────┐ │
│  │   Lakebase    │    │         MLflow + Unity Catalog       │ │
│  │  PostgreSQL   │    │   - Experiments & Model Registry     │ │
│  │  - Sessions   │    │   - Forecasts & Artifacts            │ │
│  │  - History    │    └──────────────────────────────────────┘ │
│  │  - Results    │                                              │
│  └───────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Files Created

| File | Purpose |
|------|---------|
| `scripts/deploy.sh` | One-click deployment script |
| `sql/create_tables.sql` | Lakebase PostgreSQL schema |
| `notebooks/train_models.py` | Training notebook for Jobs |
| `backend/services/lakebase_client.py` | PostgreSQL client |
| `backend/services/session_manager.py` | Session management |
| `backend/services/job_service.py` | Databricks Jobs API |
| `backend/services/history_service.py` | Execution history |
| `backend/models/statsforecast_models.py` | Fast statistical models |
| `backend/models/chronos_model.py` | Foundation model |
| `backend/models/ensemble.py` | Model ensembling |
| `backend/models/conformal.py` | Prediction intervals |
