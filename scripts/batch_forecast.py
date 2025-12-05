#!/usr/bin/env python3
"""
Batch Forecasting Script
=========================

Run forecasts for multiple data segments in parallel using the backend API.

Usage:
    python scripts/batch_forecast.py --input data.csv --segment-cols REGION,PRODUCT --output results/

Examples:
    # Single column segmentation
    python scripts/batch_forecast.py --input sales_data.csv --segment-cols region

    # Multi-column segmentation (creates unique combinations)
    python scripts/batch_forecast.py --input sales_data.csv --segment-cols "region,product,channel"

    # Forecast specific segment combinations only
    python scripts/batch_forecast.py --input sales_data.csv --segment-cols region --segments "US,EU,APAC"

    # With custom configuration
    python scripts/batch_forecast.py --input data.csv --segment-cols "category,store" \
        --time-col date --target-col revenue --horizon 12 --frequency monthly

Requirements:
    - Backend server running on localhost:8000 (or specify --api-url)
    - pip install pandas requests tqdm
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import requests

# Optional: progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df


def train_segment(
    segment_id: str,
    segment_data: pd.DataFrame,
    config: dict,
    api_url: str,
    filters: Optional[Dict[str, Any]] = None
) -> dict:
    """Train a forecast model for a single segment."""

    payload = {
        "data": segment_data.to_dict('records'),
        "time_col": config['time_col'],
        "target_col": config['target_col'],
        "covariates": config.get('covariates', []),
        "horizon": config.get('horizon', 12),
        "frequency": config.get('frequency', 'monthly'),
        "seasonality_mode": config.get('seasonality_mode', 'multiplicative'),
        "models": config.get('models', ['prophet']),
        "random_seed": config.get('random_seed', 42),
        "filters": filters,  # Store segment metadata
    }

    try:
        response = requests.post(
            f"{api_url}/api/train",
            json=payload,
            timeout=config.get('timeout', 600)
        )
        response.raise_for_status()
        result = response.json()

        return {
            "segment": segment_id,
            "filters": filters,
            "status": "success",
            "best_model": result.get('best_model'),
            "models": [
                {
                    "model_type": m.get('model_type'),
                    "mape": m.get('metrics', {}).get('mape'),
                    "cv_mape": m.get('metrics', {}).get('cv_mape'),
                    "run_id": m.get('run_id'),
                }
                for m in result.get('models', [])
            ],
            "run_id": result.get('models', [{}])[0].get('run_id') if result.get('models') else None,
        }

    except requests.exceptions.Timeout:
        return {
            "segment": segment_id,
            "filters": filters,
            "status": "error",
            "error": "Request timed out"
        }
    except requests.exceptions.RequestException as e:
        return {
            "segment": segment_id,
            "filters": filters,
            "status": "error",
            "error": str(e)
        }
    except Exception as e:
        return {
            "segment": segment_id,
            "filters": filters,
            "status": "error",
            "error": str(e)
        }


def run_batch_forecast(
    df: pd.DataFrame,
    segment_cols: List[str],
    config: dict,
    api_url: str,
    max_workers: int = 4,
    segments: Optional[List[str]] = None
) -> list:
    """
    Run forecasts for multiple segments in parallel.

    Supports both single-column and multi-column segmentation.
    For multi-column, creates forecasts for each unique combination.

    Args:
        df: Input DataFrame
        segment_cols: List of column names to segment by (e.g., ['region', 'product'])
        config: Training configuration dict
        api_url: Backend API URL
        max_workers: Number of parallel workers
        segments: Optional list of specific segments to process (only for single-column)

    Returns:
        List of result dicts with segment info and metrics
    """

    # Validate all segment columns exist
    missing_cols = [col for col in segment_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Segment columns not found in data: {missing_cols}. Available: {list(df.columns)}")

    # Get unique segment combinations
    if len(segment_cols) == 1:
        # Single column - backward compatible behavior
        col = segment_cols[0]
        all_segments = df[col].unique().tolist()

        if segments:
            # Filter to specified segments
            segments_to_process = [s for s in segments if s in all_segments]
            missing = set(segments) - set(all_segments)
            if missing:
                print(f"Warning: Segments not found in data: {missing}")
        else:
            segments_to_process = all_segments

        # Convert to list of filter dicts for consistency
        segment_filters = [{col: seg} for seg in segments_to_process]
    else:
        # Multi-column - get unique combinations
        unique_combinations = df[segment_cols].drop_duplicates()
        segment_filters = unique_combinations.to_dict('records')

        if segments:
            print(f"Warning: --segments filter is ignored for multi-column segmentation")

    # Create segment IDs
    def make_segment_id(filters: Dict[str, Any]) -> str:
        """Create a readable segment ID from filter dict."""
        return " | ".join(f"{k}={v}" for k, v in filters.items())

    print(f"\nSegmentation columns: {segment_cols}")
    print(f"Found {len(segment_filters)} unique segment combinations")
    print(f"Processing with {max_workers} parallel workers...")

    # Show first few segments
    if len(segment_filters) <= 10:
        for f in segment_filters:
            print(f"  - {make_segment_id(f)}")
    else:
        for f in segment_filters[:5]:
            print(f"  - {make_segment_id(f)}")
        print(f"  ... and {len(segment_filters) - 5} more")
    print()

    results = []

    # Create progress wrapper
    if HAS_TQDM:
        progress = tqdm(total=len(segment_filters), desc="Training")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for filters in segment_filters:
            # Filter data for this segment combination
            mask = pd.Series(True, index=df.index)
            for col, val in filters.items():
                mask &= (df[col] == val)
            segment_data = df[mask].copy()

            segment_id = make_segment_id(filters)

            # Skip empty segments
            if len(segment_data) == 0:
                print(f"Warning: No data for segment {segment_id}, skipping...")
                continue

            future = executor.submit(
                train_segment,
                segment_id,
                segment_data,
                config,
                api_url,
                filters
            )
            futures[future] = (segment_id, filters)

        for future in as_completed(futures):
            segment_id, filters = futures[future]
            try:
                result = future.result()
                results.append(result)

                status_icon = "+" if result['status'] == 'success' else "X"
                if result['status'] == 'success':
                    mape = result.get('models', [{}])[0].get('mape', 'N/A')
                    print(f"[{status_icon}] {segment_id}: MAPE={mape}")
                else:
                    print(f"[{status_icon}] {segment_id}: {result.get('error', 'Unknown error')}")

            except Exception as e:
                results.append({
                    "segment": segment_id,
                    "filters": filters,
                    "status": "error",
                    "error": str(e)
                })
                print(f"[X] {segment_id}: {e}")

            if HAS_TQDM:
                progress.update(1)

    if HAS_TQDM:
        progress.close()

    return results


def save_results(results: list, output_dir: str, timestamp: str, segment_cols: List[str]):
    """Save batch results to files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed JSON results
    json_file = output_path / f"batch_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {json_file}")

    # Save summary CSV with filter columns expanded
    summary_data = []
    for r in results:
        row = {
            "segment": r['segment'],
            "status": r['status'],
            "best_model": r.get('best_model', ''),
            "run_id": r.get('run_id', ''),
            "error": r.get('error', ''),
        }

        # Add individual filter columns
        if r.get('filters'):
            for col in segment_cols:
                row[col] = r['filters'].get(col, '')

        # Add metrics from first (best) model
        if r.get('models'):
            row['mape'] = r['models'][0].get('mape', '')
            row['cv_mape'] = r['models'][0].get('cv_mape', '')

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Reorder columns: segment cols first, then metrics, then other info
    col_order = segment_cols + ['status', 'best_model', 'mape', 'cv_mape', 'run_id', 'error', 'segment']
    col_order = [c for c in col_order if c in summary_df.columns]
    summary_df = summary_df[col_order]

    csv_file = output_path / f"batch_summary_{timestamp}.csv"
    summary_df.to_csv(csv_file, index=False)
    print(f"Summary saved to: {csv_file}")

    # Print summary stats
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    print(f"\nSummary: {success_count} succeeded, {error_count} failed")

    # Print MAPE stats for successful runs
    if success_count > 0:
        mapes = []
        for r in results:
            if r['status'] == 'success' and r.get('models'):
                try:
                    mape = float(r['models'][0].get('mape', 0))
                    mapes.append(mape)
                except (ValueError, TypeError):
                    pass

        if mapes:
            print(f"\nMAPE Statistics:")
            print(f"  Min:    {min(mapes):.2f}%")
            print(f"  Max:    {max(mapes):.2f}%")
            print(f"  Mean:   {sum(mapes)/len(mapes):.2f}%")
            print(f"  Median: {sorted(mapes)[len(mapes)//2]:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run batch forecasts for multiple data segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--segment-cols', '-s',
        required=True,
        help='Comma-separated column names to segment by (e.g., "region" or "region,product,channel")'
    )

    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        default='./batch_results',
        help='Output directory for results (default: ./batch_results)'
    )
    parser.add_argument(
        '--segments',
        help='Comma-separated list of specific segments to process (only for single-column segmentation)'
    )
    parser.add_argument(
        '--time-col',
        default='ds',
        help='Time column name (default: ds)'
    )
    parser.add_argument(
        '--target-col',
        default='y',
        help='Target column name (default: y)'
    )
    parser.add_argument(
        '--covariates',
        help='Comma-separated list of covariate columns'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=12,
        help='Forecast horizon periods (default: 12)'
    )
    parser.add_argument(
        '--frequency',
        choices=['daily', 'weekly', 'monthly'],
        default='monthly',
        help='Data frequency (default: monthly)'
    )
    parser.add_argument(
        '--models',
        default='prophet',
        help='Comma-separated list of models to train (default: prophet)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--api-url',
        default='http://localhost:8000',
        help='Backend API URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Request timeout in seconds (default: 600)'
    )

    # Keep backward compatibility with old --segment-col argument
    parser.add_argument(
        '--segment-col',
        help=argparse.SUPPRESS  # Hidden, for backward compatibility
    )

    args = parser.parse_args()

    # Handle backward compatibility: --segment-col -> --segment-cols
    if args.segment_col and not args.segment_cols:
        args.segment_cols = args.segment_col

    # Load data
    try:
        df = load_data(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Parse segment columns
    segment_cols = [col.strip() for col in args.segment_cols.split(',')]

    # Validate segment columns exist
    missing_cols = [col for col in segment_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Segment column(s) not found in data: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Parse optional lists
    segments = args.segments.split(',') if args.segments else None
    covariates = args.covariates.split(',') if args.covariates else []
    models = args.models.split(',')

    # Build config
    config = {
        'time_col': args.time_col,
        'target_col': args.target_col,
        'covariates': covariates,
        'horizon': args.horizon,
        'frequency': args.frequency,
        'models': models,
        'timeout': args.timeout,
        'random_seed': 42,
    }

    print(f"\nConfiguration:")
    print(f"  Segment columns: {segment_cols}")
    print(f"  Time column: {config['time_col']}")
    print(f"  Target column: {config['target_col']}")
    print(f"  Covariates: {config['covariates'] or 'None'}")
    print(f"  Horizon: {config['horizon']} periods")
    print(f"  Frequency: {config['frequency']}")
    print(f"  Models: {config['models']}")
    print(f"  API URL: {args.api_url}")

    # Check API is reachable
    try:
        health = requests.get(f"{args.api_url}/api/health", timeout=5)
        if health.status_code != 200:
            print(f"\nWarning: API health check failed (status {health.status_code})")
    except requests.exceptions.RequestException:
        print(f"\nError: Cannot reach API at {args.api_url}")
        print("Make sure the backend is running: ./start-local.sh")
        sys.exit(1)

    # Run batch forecast
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = run_batch_forecast(
        df=df,
        segment_cols=segment_cols,
        config=config,
        api_url=args.api_url,
        max_workers=args.workers,
        segments=segments
    )

    # Save results
    save_results(results, args.output, timestamp, segment_cols)

    print("\nDone!")


if __name__ == "__main__":
    main()
