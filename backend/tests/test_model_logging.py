import pandas as pd
import numpy as np
import os
import sys
import logging
import shutil
import mlflow
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.getcwd())

# Import training functions
try:
    from backend.models.prophet import train_prophet_model
    from backend.models.arima import train_arima_model, train_sarimax_model
    from backend.models.ets import train_exponential_smoothing_model
    from backend.models.xgboost import train_xgboost_model
except ImportError as e:
    logger.error(f"Could not import backend modules. Run from project root. Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def setup_mlflow():
    """Setup temporary MLflow tracking"""
    tracking_uri = "file:///tmp/mlruns_test_models"
    if os.path.exists("/tmp/mlruns_test_models"):
        shutil.rmtree("/tmp/mlruns_test_models")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test_experiment")
    os.environ["MLFLOW_MAX_WORKERS"] = "1"
    return tracking_uri

def generate_synthetic_data(n_rows=100, freq='D'):
    """Generate synthetic time series data"""
    dates = pd.date_range(start='2024-01-01', periods=n_rows, freq=freq)
    # Trend + Seasonality + Noise
    y = np.linspace(0, 10, n_rows) + np.sin(np.linspace(0, 10, n_rows)) + np.random.normal(0, 0.1, n_rows)
    
    df = pd.DataFrame({
        'ds': dates,
        'y': y,
        'promo': np.random.choice([0, 1], size=n_rows)
    })
    return df

def test_prophet_logging():
    logger.info("Testing Prophet Logging & Inference...")
    df = generate_synthetic_data()
    # Prepare data for Prophet (expects list of dicts)
    data_records = df.to_dict('records')
    
    # Train
    run_id, model_uri, _, _, _, _, _ = train_prophet_model(
        data=data_records,
        time_col='ds',
        target_col='y',
        covariates=['promo'],
        horizon=10,
        frequency='daily'
    )
    
    # Load model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # Verify Signature (Advanced Mode)
    # Create input dataframe for inference
    future_dates = pd.date_range(start=df['ds'].max() + timedelta(days=1), periods=5, freq='D')
    input_df = pd.DataFrame({
        'ds': future_dates.strftime('%Y-%m-%d'),
        'promo': [0, 1, 0, 1, 0]
    })
    
    # Predict
    prediction = loaded_model.predict(input_df)
    
    # Assertions
    assert isinstance(prediction, pd.DataFrame)
    assert 'ds' in prediction.columns
    assert 'yhat' in prediction.columns
    assert len(prediction) == 5
    logger.info("✅ Prophet Test Passed")

def test_arima_logging():
    logger.info("Testing ARIMA Logging & Inference...")
    df = generate_synthetic_data()
    train_df = df.iloc[:-10]
    test_df = df.iloc[-10:]
    
    # Train
    run_id, model_uri, _, _, _, _, _ = train_arima_model(
        train_df=train_df,
        test_df=test_df,
        horizon=10,
        frequency='daily'
    )
    
    # Load model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # Verify Signature (Simple Mode)
    input_df = pd.DataFrame({
        'periods': [5],
        'start_date': [str(df['ds'].max().date())]
    })
    
    # Predict
    prediction = loaded_model.predict(input_df)
    
    # Assertions
    assert isinstance(prediction, pd.DataFrame)
    assert 'ds' in prediction.columns
    assert 'yhat' in prediction.columns
    assert len(prediction) == 5
    logger.info("✅ ARIMA Test Passed")

def test_ets_logging():
    logger.info("Testing ETS Logging & Inference...")
    df = generate_synthetic_data()
    train_df = df.iloc[:-10]
    test_df = df.iloc[-10:]
    
    # Train
    run_id, model_uri, _, _, _, _, _ = train_exponential_smoothing_model(
        train_df=train_df,
        test_df=test_df,
        horizon=10,
        frequency='daily'
    )
    
    # Load model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # Verify Signature (Simple Mode)
    input_df = pd.DataFrame({
        'periods': [5],
        'start_date': [str(df['ds'].max().date())]
    })
    
    # Predict
    prediction = loaded_model.predict(input_df)
    
    # Assertions
    assert isinstance(prediction, pd.DataFrame)
    assert 'ds' in prediction.columns
    assert 'yhat' in prediction.columns
    assert len(prediction) == 5
    logger.info("✅ ETS Test Passed")

def test_sarimax_logging():
    logger.info("Testing SARIMAX Logging & Inference...")
    df = generate_synthetic_data()
    train_df = df.iloc[:-10]
    test_df = df.iloc[-10:]
    
    # Train
    run_id, model_uri, _, _, _, _, _ = train_sarimax_model(
        train_df=train_df,
        test_df=test_df,
        horizon=10,
        frequency='daily',
        covariates=['promo']
    )
    
    # Load model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # Verify Signature (Simple Mode)
    input_df = pd.DataFrame({
        'periods': [5],
        'start_date': [str(df['ds'].max().date())]
    })
    
    # Predict
    prediction = loaded_model.predict(input_df)
    
    # Assertions
    assert isinstance(prediction, pd.DataFrame)
    assert 'ds' in prediction.columns
    assert 'yhat' in prediction.columns
    assert len(prediction) == 5
    logger.info("✅ SARIMAX Test Passed")

def test_xgboost_logging():
    logger.info("Testing XGBoost Logging & Inference...")
    df = generate_synthetic_data(n_rows=200) # More data for XGBoost lags
    train_df = df.iloc[:-20]
    test_df = df.iloc[-20:]
    
    # Train
    run_id, model_uri, _, _, _, _, _ = train_xgboost_model(
        train_df=train_df,
        test_df=test_df,
        horizon=10,
        frequency='daily',
        covariates=['promo']
    )
    
    # Load model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # Verify Signature (Simple Mode)
    input_df = pd.DataFrame({
        'periods': [5],
        'start_date': [str(df['ds'].max().date())]
    })
    
    # Predict
    prediction = loaded_model.predict(input_df)
    
    # Assertions
    assert isinstance(prediction, pd.DataFrame)
    assert 'ds' in prediction.columns
    assert 'yhat' in prediction.columns
    assert len(prediction) == 5
    logger.info("✅ XGBoost Test Passed")

if __name__ == "__main__":
    setup_mlflow()
    try:
        test_prophet_logging()
        test_arima_logging()
        test_sarimax_logging()
        test_ets_logging()
        test_xgboost_logging()
        logger.info("🎉 ALL MODEL TESTS PASSED")
    except Exception as e:
        logger.error(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
