# Release Notes

## v1.1.0 (2025-12-10)

### New Features

#### Batch Deployment with Router Model
- Deploy multiple models across segments with a single operation
- Router model automatically selects the best model per segment based on validation metrics
- Pre-deployment inference testing validates models before MLflow registration
- Batch comparison view for analyzing forecast vs actuals across all segments

#### Scalable Architecture Documentation
- Added `docs/SCALABLE_ARCHITECTURE.md` with design for serverless compute scaling
- Enables offloading model training to Databricks Serverless Jobs
- Supports 100+ parallel training jobs vs current 2-4 limit
- Includes cost analysis, migration path, and implementation checklist

### Bug Fixes

#### Fixed "No overlapping dates" Error in Forecast vs Actuals Comparison
- **Root Cause**: Prophet, ARIMA, ETS, SARIMAX, and XGBoost models were generating Sunday-based weekly dates (`freq='W'`) while actual business data uses Monday-based weeks
- **Impact**: 1-day mismatch caused zero overlapping dates between forecast and actuals, breaking comparison functionality
- **Solution**: Auto-detect day-of-week from training data and use appropriate pandas frequency code (`W-MON`, `W-TUE`, etc.)
- **Files Modified**:
  - `backend/models/prophet.py` - Inline detection in training and wrapper
  - `backend/models/arima.py` - Added `detect_weekly_freq_code()`, updated ARIMA and SARIMAX
  - `backend/models/ets.py` - Added `detect_weekly_freq_code()`, updated ETS
  - `backend/models/xgboost.py` - Added `detect_weekly_freq_code()`, updated XGBoost

#### Fixed SPA Catch-All Route
- Moved SPA catch-all route to end of `main.py` to prevent API route interception
- API endpoints now correctly return 404 instead of serving index.html

### Testing

#### New Test Suites
- `backend/tests/test_weekly_date_alignment.py` - 15 comprehensive tests covering:
  - Frequency code detection for all models
  - Forecast date generation on correct day-of-week
  - End-to-end forecast vs actuals comparison
  - Model wrapper date generation
- `backend/tests/test_batch_e2e.py` - Batch deployment end-to-end tests

### Architecture Changes

#### Model Code Refactoring
- Split monolithic `models_training.py` into separate model files:
  - `backend/models/prophet.py`
  - `backend/models/arima.py`
  - `backend/models/ets.py`
  - `backend/models/xgboost.py`
  - `backend/models/utils.py`
- Renamed `models.py` to `schemas.py` for clarity

### Infrastructure Notes

#### Databricks Apps Sizing
- **Medium**: 2 vCPU / 6GB RAM - Suitable for UI + orchestration only
- **Large**: 4 vCPU / 12GB RAM - Maximum available, supports 2-4 parallel model training jobs

#### Scaling Recommendations
For workloads exceeding 4 parallel jobs or 12GB RAM:
- Use Databricks Serverless Jobs for model training
- Keep App as lightweight orchestrator
- See `docs/SCALABLE_ARCHITECTURE.md` for implementation details

---

## v1.0.0 (Initial Release)

- Multi-model forecasting (Prophet, ARIMA, SARIMAX, ETS, XGBoost)
- Single and batch training modes
- MLflow integration for experiment tracking and model registry
- Interactive visualization with forecast vs actuals comparison
- Promotional/covariate support
- Data preprocessing and validation
