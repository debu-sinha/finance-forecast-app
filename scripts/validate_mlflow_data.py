#!/usr/bin/env python3
"""
Script to download and validate MLflow experiment data.
Run this script to verify that training data, forecasts, and actuals are aligned correctly.

Usage:
    python scripts/validate_mlflow_data.py [--experiment-name NAME] [--run-id RUN_ID]
"""

import os
import sys
import argparse
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_mlflow():
    """Configure MLflow connection"""
    # Try to get credentials from environment
    host = os.environ.get('DATABRICKS_HOST')
    token = os.environ.get('DATABRICKS_TOKEN')

    if host and token:
        mlflow.set_tracking_uri("databricks")
        print(f"Connected to Databricks MLflow at {host}")
    else:
        # Try local
        mlflow.set_tracking_uri("mlflow")
        print("Using local MLflow tracking")

    return MlflowClient()


def list_recent_experiments(client, limit=10):
    """List recent experiments"""
    experiments = client.search_experiments(order_by=["last_update_time DESC"])
    print(f"\n=== Recent Experiments (showing {min(limit, len(experiments))}) ===")
    for i, exp in enumerate(experiments[:limit]):
        print(f"  {i+1}. {exp.name} (ID: {exp.experiment_id})")
    return experiments


def list_recent_runs(client, experiment_name, limit=10):
    """List recent runs for an experiment"""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found")
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=limit
    )

    print(f"\n=== Recent Runs for '{experiment_name}' ===")
    for i, run in enumerate(runs):
        metrics = run.data.metrics
        mape = metrics.get('mape', 'N/A')
        rmse = metrics.get('rmse', 'N/A')
        model_type = run.data.params.get('model_type', 'unknown')
        print(f"  {i+1}. Run {run.info.run_id[:8]}... | {model_type} | MAPE: {mape} | RMSE: {rmse}")

    return runs


def download_and_validate_run(client, run_id, output_dir="./validation_output"):
    """Download artifacts from a run and validate the data"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Validating Run: {run_id} ===")

    run = client.get_run(run_id)
    print(f"Model Type: {run.data.params.get('model_type', 'unknown')}")
    print(f"MAPE: {run.data.metrics.get('mape', 'N/A')}")
    print(f"RMSE: {run.data.metrics.get('rmse', 'N/A')}")

    # Download artifacts
    artifact_path = client.download_artifacts(run_id, "", output_dir)
    print(f"\nArtifacts downloaded to: {artifact_path}")

    # List downloaded files
    print("\n--- Downloaded Files ---")
    for root, dirs, files in os.walk(artifact_path):
        level = root.replace(artifact_path, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = '  ' * (level + 1)
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            print(f"{subindent}{file} ({size:,} bytes)")

    # Validate datasets
    print("\n--- Data Validation ---")

    datasets_path = os.path.join(artifact_path, "datasets")
    if os.path.exists(datasets_path):
        # Check training data
        train_path = os.path.join(datasets_path, "training", "train.csv")
        eval_path = os.path.join(datasets_path, "training", "eval.csv")
        full_path = os.path.join(datasets_path, "processed", "full_merged_data.csv")
        input_path = os.path.join(datasets_path, "inference", "input.csv")
        output_path = os.path.join(datasets_path, "inference", "output.csv")

        validation_results = {}

        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            print(f"\n[Training Data] {train_path}")
            print(f"  Shape: {train_df.shape}")
            print(f"  Columns: {list(train_df.columns)}")
            if 'ds' in train_df.columns:
                train_df['ds'] = pd.to_datetime(train_df['ds'])
                print(f"  Date Range: {train_df['ds'].min()} to {train_df['ds'].max()}")
            if 'y' in train_df.columns:
                print(f"  Target Stats: min={train_df['y'].min():.2f}, max={train_df['y'].max():.2f}, mean={train_df['y'].mean():.2f}")
            validation_results['train'] = train_df

        if os.path.exists(eval_path):
            eval_df = pd.read_csv(eval_path)
            print(f"\n[Evaluation Data] {eval_path}")
            print(f"  Shape: {eval_df.shape}")
            if 'ds' in eval_df.columns:
                eval_df['ds'] = pd.to_datetime(eval_df['ds'])
                print(f"  Date Range: {eval_df['ds'].min()} to {eval_df['ds'].max()}")
            validation_results['eval'] = eval_df

        if os.path.exists(output_path):
            forecast_df = pd.read_csv(output_path)
            print(f"\n[Forecast Output] {output_path}")
            print(f"  Shape: {forecast_df.shape}")
            print(f"  Columns: {list(forecast_df.columns)}")
            if 'ds' in forecast_df.columns:
                forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
                print(f"  Date Range: {forecast_df['ds'].min()} to {forecast_df['ds'].max()}")
            if 'yhat' in forecast_df.columns:
                print(f"  Forecast Stats: min={forecast_df['yhat'].min():.2f}, max={forecast_df['yhat'].max():.2f}, mean={forecast_df['yhat'].mean():.2f}")
            validation_results['forecast'] = forecast_df

        # Check for date continuity
        print("\n--- Date Continuity Check ---")
        if 'train' in validation_results and 'forecast' in validation_results:
            train_max = validation_results['train']['ds'].max()
            forecast_min = validation_results['forecast']['ds'].min()

            print(f"  Last training date: {train_max}")
            print(f"  First forecast date: {forecast_min}")

            if forecast_min <= train_max:
                print(f"  WARNING: Forecast starts ON or BEFORE training data ends!")
                print(f"           This could indicate data leakage or incorrect date alignment.")
            else:
                gap_days = (forecast_min - train_max).days
                print(f"  Gap between training and forecast: {gap_days} days")
                if gap_days > 7:
                    print(f"  NOTE: Gap seems large. Verify this is expected.")

        return validation_results
    else:
        print(f"No datasets folder found in artifacts")
        return {}


def compare_with_actuals(forecast_df, actuals_path):
    """Compare forecast with actual values"""
    if not os.path.exists(actuals_path):
        print(f"Actuals file not found: {actuals_path}")
        return

    actuals_df = pd.read_csv(actuals_path)
    print(f"\n=== Comparing Forecast with Actuals ===")
    print(f"Actuals file: {actuals_path}")
    print(f"Actuals shape: {actuals_df.shape}")
    print(f"Actuals columns: {list(actuals_df.columns)}")

    # Try to find date and value columns
    date_col = None
    value_col = None

    for col in actuals_df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['date', 'ds', 'time', 'period']):
            date_col = col
        elif any(kw in col_lower for kw in ['value', 'actual', 'y', 'revenue', 'sales', 'amount']):
            value_col = col

    if not date_col or not value_col:
        print(f"Could not auto-detect date/value columns. Please specify.")
        print(f"Available columns: {list(actuals_df.columns)}")
        return

    print(f"Using date column: {date_col}")
    print(f"Using value column: {value_col}")

    # Parse dates
    actuals_df[date_col] = pd.to_datetime(actuals_df[date_col])
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    # Merge
    merged = pd.merge(
        forecast_df[['ds', 'yhat']],
        actuals_df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'actual'}),
        on='ds',
        how='inner'
    )

    if len(merged) == 0:
        print("\nNO MATCHING DATES FOUND!")
        print(f"Forecast date range: {forecast_df['ds'].min()} to {forecast_df['ds'].max()}")
        print(f"Actuals date range: {actuals_df[date_col].min()} to {actuals_df[date_col].max()}")
        return

    print(f"\nMatched {len(merged)} periods")

    # Calculate metrics
    merged['error'] = merged['actual'] - merged['yhat']
    merged['abs_error'] = merged['error'].abs()
    merged['pct_error'] = (merged['error'] / merged['actual']) * 100
    merged['mape'] = merged['pct_error'].abs()

    print(f"\n--- Comparison Results ---")
    print(f"Overall MAPE: {merged['mape'].mean():.2f}%")
    print(f"Overall RMSE: {(merged['error']**2).mean()**0.5:.2f}")
    print(f"Overall Bias: {merged['error'].mean():.2f}")

    print(f"\n--- Sample Comparisons (first 5) ---")
    print(merged[['ds', 'yhat', 'actual', 'error', 'mape']].head().to_string())

    # Show worst predictions
    print(f"\n--- Worst Predictions (top 5 by MAPE) ---")
    worst = merged.nlargest(5, 'mape')
    print(worst[['ds', 'yhat', 'actual', 'error', 'mape']].to_string())


def main():
    parser = argparse.ArgumentParser(description='Validate MLflow experiment data')
    parser.add_argument('--experiment-name', '-e',
                       default=os.environ.get('MLFLOW_EXPERIMENT_NAME', '/Users/debu.sinha@databricks.com/finance-forecasting'),
                       help='MLflow experiment name')
    parser.add_argument('--run-id', '-r', help='Specific run ID to validate')
    parser.add_argument('--actuals', '-a', help='Path to actuals CSV file for comparison')
    parser.add_argument('--output-dir', '-o', default='./validation_output', help='Output directory for downloaded artifacts')
    parser.add_argument('--list-only', '-l', action='store_true', help='Only list experiments and runs')

    args = parser.parse_args()

    # Setup
    client = setup_mlflow()

    if args.list_only:
        list_recent_experiments(client)
        if args.experiment_name:
            list_recent_runs(client, args.experiment_name)
        return

    # If run ID specified, validate that run
    if args.run_id:
        results = download_and_validate_run(client, args.run_id, args.output_dir)

        if args.actuals and 'forecast' in results:
            compare_with_actuals(results['forecast'], args.actuals)
    else:
        # List runs and ask for selection
        runs = list_recent_runs(client, args.experiment_name)
        if not runs:
            return

        print("\nEnter run number to validate (or 'q' to quit): ", end='')
        choice = input().strip()

        if choice.lower() == 'q':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                results = download_and_validate_run(client, runs[idx].info.run_id, args.output_dir)

                if args.actuals and 'forecast' in results:
                    compare_with_actuals(results['forecast'], args.actuals)
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")


if __name__ == "__main__":
    main()
