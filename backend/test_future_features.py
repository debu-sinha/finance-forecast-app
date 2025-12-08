import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

try:
    from backend.models.prophet import train_prophet_model
except ImportError:
    logger.error("Could not import backend.train_service. Make sure you are running this from the project root.")
    sys.exit(1)

import argparse

def test_future_features_logic():
    parser = argparse.ArgumentParser(description='Test Future Features Logic')
    parser.add_argument('--main-data', type=str, help='Path to main data CSV')
    parser.add_argument('--feature-data', type=str, help='Path to features data CSV')
    parser.add_argument('--target-col', type=str, default='y', help='Target column name')
    parser.add_argument('--time-col', type=str, default='ds', help='Time column name')
    parser.add_argument('--covariates', type=str, help='Comma-separated list of covariates')
    
    args = parser.parse_args()
    
    logger.info("Starting Future Features Verification Test")
    
    if args.main_data and args.feature_data:
        logger.info(f"1️⃣ Loading custom data from:\n   Main: {args.main_data}\n   Features: {args.feature_data}")
        try:
            # Auto-detect separator (comma or tab)
            with open(args.main_data, 'r') as f:
                first_line = f.readline()
                main_sep = '\t' if '\t' in first_line else ','
            
            with open(args.feature_data, 'r') as f:
                first_line = f.readline()
                feature_sep = '\t' if '\t' in first_line else ','
            
            main_data = pd.read_csv(args.main_data, sep=main_sep)
            future_features = pd.read_csv(args.feature_data, sep=feature_sep)
            
            logger.info(f"   Loaded {len(main_data)} rows from main data")
            logger.info(f"   Loaded {len(future_features)} rows from features data")
            
            # Handle different time column names between files
            # Check if the time column exists in main_data
            if args.time_col not in main_data.columns:
                # Try to find a date-like column
                date_cols = [c for c in main_data.columns if 'date' in c.lower() or 'day' in c.lower() or 'time' in c.lower()]
                if date_cols:
                    logger.info(f"   Renaming '{date_cols[0]}' to '{args.time_col}' in main data")
                    main_data = main_data.rename(columns={date_cols[0]: args.time_col})
                else:
                    raise ValueError(f"Time column '{args.time_col}' not found in main data. Available columns: {main_data.columns.tolist()}")
            
            # Check if the time column exists in future_features
            if args.time_col not in future_features.columns:
                # Try to find a date-like column
                date_cols = [c for c in future_features.columns if 'date' in c.lower() or 'day' in c.lower() or 'time' in c.lower()]
                if date_cols:
                    logger.info(f"   Renaming '{date_cols[0]}' to '{args.time_col}' in features data")
                    future_features = future_features.rename(columns={date_cols[0]: args.time_col})
                else:
                    raise ValueError(f"Time column '{args.time_col}' not found in features data. Available columns: {future_features.columns.tolist()}")
            
            # Ensure date columns are datetime
            main_data[args.time_col] = pd.to_datetime(main_data[args.time_col])
            future_features[args.time_col] = pd.to_datetime(future_features[args.time_col])
            
            target_col = args.target_col
            time_col = args.time_col
            
            # Determine covariates
            if args.covariates:
                covariates = [c.strip() for c in args.covariates.split(',')]
            else:
                # Auto-detect: all columns in features except time and target
                covariates = [c for c in future_features.columns if c not in [time_col, target_col]]
                logger.info(f"   Auto-detected covariates: {covariates[:5]}...")
                
        except Exception as e:
            logger.error(f"Failed to load custom data: {e}")
            return False
    else:
        # 1. Setup Mock Data
        logger.info("1️⃣ Setting up mock data (Default)...")
        target_col = 'y'
        time_col = 'ds'
        covariates = ['promo']
        
        # Main Data (History): Jan 1 - Jan 10
        dates_hist = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        main_data = pd.DataFrame({
            'ds': dates_hist,
            'y': np.random.rand(len(dates_hist)) * 100,
            'promo': [0] * len(dates_hist) # No promo in history
        })
        
        # Future Features (Leakage Check): Jan 11 - Jan 15
        # INTENTIONALLY include 'y' to test leakage prevention
        dates_future = pd.date_range(start='2024-01-11', end='2024-01-15', freq='D')
        future_features = pd.DataFrame({
            'ds': dates_future,
            'y': [9999.99] * len(dates_future), # LEAKAGE VALUES
            'promo': [1] * len(dates_future)    # Future promos are active
        })
    
    # Convert to list of dicts as expected by the API
    # Handle NaN/Inf for JSON serialization if needed, though to_dict('records') usually handles it
    # But for API simulation we should be careful.
    # For this test, we pass directly to python function, so pandas objects in dict are fine.
    
    # IMPORTANT: The API expects string dates usually, but the internal function handles pandas timestamps too.
    # Let's convert dates to string to match API behavior more closely if using custom data
    main_data_api = main_data.copy()
    future_features_api = future_features.copy()
    
    main_data_api[time_col] = main_data_api[time_col].dt.strftime('%Y-%m-%d')
    future_features_api[time_col] = future_features_api[time_col].dt.strftime('%Y-%m-%d')
    
    data_records = main_data_api.to_dict('records')
    future_records = future_features_api.to_dict('records')
    
    logger.info(f"   History rows: {len(data_records)}")
    logger.info(f"   Future rows: {len(future_records)}")
    if target_col in future_features.columns:
        logger.info(f"   Future '{target_col}' value (potential leakage): {future_features[target_col].iloc[0]}")
    
    # 2. Run Training
    logger.info("2️⃣ Running train_prophet_model...")
    
    # We need to mock mlflow to avoid actual logging overhead/errors
    import mlflow
    mlflow.set_tracking_uri("file:///tmp/mlruns_test")
    
    try:
        train_prophet_model(
            data=data_records,
            time_col=time_col,
            target_col=target_col,
            covariates=covariates,
            horizon=90, # Default horizon
            frequency='daily',
            future_features=future_records
        )
    except Exception as e:
        # It might fail on registering model etc, but we care about the artifacts generated before that
        logger.warning(f"   Training finished with error (expected in test env): {e}")
        
    # 3. Verify Artifacts
    logger.info("3️⃣ Verifying artifacts...")
    
    artifact_path = "/tmp/prophet_future_data.csv"
    if not os.path.exists(artifact_path):
        logger.error(f"Artifact {artifact_path} was not created!")
        return False
        
    df_result = pd.read_csv(artifact_path)
    logger.info(f"   Read {len(df_result)} rows from {artifact_path}")
    logger.info(f"   Columns: {df_result.columns.tolist()}")
    
    # Check 1: Target Leakage
    if target_col in df_result.columns:
        logger.error(f"FAILED: '{target_col}' column found in future dataframe! Leakage detected.")
        return False
    else:
        logger.info(f"PASSED: '{target_col}' column correctly removed from future dataframe.")
        
    # Check 2: Feature Values
    # We'll check the first covariate
    if covariates:
        check_cov = covariates[0]
        if check_cov in df_result.columns:
            # Check the last few rows (future)
            last_rows = df_result.tail(5)
            logger.info(f"   Last 5 rows '{check_cov}' values: {last_rows[check_cov].tolist()}")
            
            # If we are using custom data, we can't strictly assert values without knowing them
            # But we can check they are not all 0 if we expect them to be non-zero
            # For now, just logging them is useful for the user
            logger.info(f"PASSED: '{check_cov}' exists in future dataframe.")
        else:
            logger.error(f"FAILED: '{check_cov}' missing from future dataframe.")
            return False
            
    logger.info("🎉 ALL TESTS PASSED!")
    return True

if __name__ == "__main__":
    success = test_future_features_logic()
    sys.exit(0 if success else 1)
