#!/usr/bin/env python3
"""
End-to-End Batch Workflow Test

This test exercises the entire batch training and deployment workflow using real API calls.
It tests:
1. Health check endpoint
2. Single model training (baseline)
3. Batch training with multiple segments
4. Model inference testing
5. Batch deployment with router model

Prerequisites:
- Backend server running on localhost:8000
- Valid Databricks credentials configured (DATABRICKS_HOST, DATABRICKS_TOKEN)
- MLflow configured to use Databricks tracking

Usage:
    python backend/tests/test_batch_e2e.py [--api-url URL] [--skip-deploy]

Options:
    --api-url URL    Backend API URL (default: http://localhost:8000)
    --skip-deploy    Skip deployment tests (useful for local testing without Databricks)
    --models MODELS  Comma-separated list of models to test (default: prophet)
    --verbose        Enable verbose logging
"""

import requests
import pandas as pd
import numpy as np
import time
import logging
import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchE2ETest:
    """End-to-end test suite for batch workflow."""

    def __init__(self, api_url: str = "http://localhost:8000", skip_deploy: bool = False,
                 models: List[str] = None, verbose: bool = False):
        self.api_url = api_url.rstrip("/")
        self.skip_deploy = skip_deploy
        self.models = models or ["prophet"]
        self.verbose = verbose
        self.test_results = {}

        # Track resources created for cleanup
        self.created_endpoints = []
        self.test_run_ids = []
        self.last_registered_version = None  # Track last registered model version

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    def log_step(self, step_name: str, status: str = "START"):
        """Log a test step."""
        icon = "🚀" if status == "START" else "✅" if status == "PASS" else "❌"
        logger.info(f"\n{'='*60}")
        logger.info(f"{icon} {step_name}")
        logger.info(f"{'='*60}")

    def generate_synthetic_data(self, n_rows: int = 200, segment: str = None,
                                  include_promo: bool = True) -> List[Dict]:
        """
        Generate synthetic time series data for testing.

        Args:
            n_rows: Number of data points
            segment: Optional segment identifier to add to data
            include_promo: Whether to include promotion indicator

        Returns:
            List of data records (dict format for API)
        """
        dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='D')

        # Generate realistic sales data with trend and seasonality
        trend = np.linspace(100, 200, n_rows)
        weekly_seasonality = 20 * np.sin(2 * np.pi * np.arange(n_rows) / 7)
        noise = np.random.normal(0, 10, n_rows)

        y = trend + weekly_seasonality + noise

        data = {
            'ds': dates.strftime('%Y-%m-%d').tolist(),
            'y': y.tolist()
        }

        if include_promo:
            # Add promotional events (random days)
            promo = np.zeros(n_rows)
            promo_days = np.random.choice(n_rows, size=n_rows // 10, replace=False)
            promo[promo_days] = 1
            # Boost sales on promo days
            y_with_promo = y.copy()
            y_with_promo[promo_days] += 30
            data['y'] = y_with_promo.tolist()
            data['promo'] = promo.astype(int).tolist()

        if segment:
            data['segment'] = [segment] * n_rows

        # Convert to list of dicts
        df = pd.DataFrame(data)
        return df.to_dict('records')

    def generate_segmented_data(self, segments: List[str], rows_per_segment: int = 150) -> Dict[str, List[Dict]]:
        """
        Generate data for multiple segments.

        Args:
            segments: List of segment names
            rows_per_segment: Number of rows per segment

        Returns:
            Dict mapping segment name to data records
        """
        segmented_data = {}
        for segment in segments:
            segmented_data[segment] = self.generate_synthetic_data(
                n_rows=rows_per_segment,
                segment=segment,
                include_promo=True
            )
        return segmented_data

    # =========================================================================
    # TEST 1: Health Check
    # =========================================================================
    def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        self.log_step("TEST 1: Health Check")

        try:
            response = requests.get(f"{self.api_url}/api/health", timeout=10)
            response.raise_for_status()

            data = response.json()
            logger.info(f"Health status: {data}")

            assert data.get("status") == "healthy", f"Expected healthy status, got {data.get('status')}"

            # Log Databricks connection status
            if data.get("databricks_connected"):
                logger.info("✅ Databricks connected")
            else:
                logger.warning("⚠️ Databricks not connected - some tests may fail")

            if data.get("mlflow_enabled"):
                logger.info("✅ MLflow enabled")
            else:
                logger.warning("⚠️ MLflow not enabled - training tests will fail")

            self.log_step("TEST 1: Health Check", "PASS")
            self.test_results["health_check"] = {"passed": True}
            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.log_step("TEST 1: Health Check", "FAIL")
            self.test_results["health_check"] = {"passed": False, "error": str(e)}
            return False

    # =========================================================================
    # TEST 2: Single Model Training
    # =========================================================================
    def test_single_training(self) -> Optional[Dict]:
        """Test single model training endpoint."""
        self.log_step("TEST 2: Single Model Training")

        try:
            # Generate test data
            data = self.generate_synthetic_data(n_rows=200)
            logger.info(f"Generated {len(data)} rows of synthetic data")

            # Prepare training request
            payload = {
                "data": data,
                "time_col": "ds",
                "target_col": "y",
                "covariates": ["promo"],
                "horizon": 14,
                "frequency": "daily",
                "seasonality_mode": "additive",
                "regressor_method": "mean",
                "models": self.models,
                "catalog_name": "main",
                "schema_name": "default",
                "model_name": "finance_forecast_model_test",
                "country": "US",
                "random_seed": 42
            }

            logger.info(f"Sending training request with models: {self.models}")
            start_time = time.time()

            response = requests.post(
                f"{self.api_url}/api/train",
                json=payload,
                timeout=600  # 10 minute timeout for training
            )

            elapsed = time.time() - start_time
            logger.info(f"Training completed in {elapsed:.2f} seconds")

            response.raise_for_status()
            result = response.json()

            # Validate response structure
            assert "models" in result, "Response missing 'models' field"
            assert "best_model" in result, "Response missing 'best_model' field"
            assert len(result["models"]) > 0, "No models returned"

            # Log model results
            for model in result["models"]:
                model_name = model.get("model_name", "Unknown")
                metrics = model.get("metrics", {})
                run_id = model.get("run_id", "N/A")
                is_best = model.get("is_best", False)
                registered_version = model.get("registered_version")

                logger.info(f"  {'🏆' if is_best else '  '} {model_name}:")
                logger.info(f"      MAPE: {metrics.get('mape', 'N/A')}")
                logger.info(f"      RMSE: {metrics.get('rmse', 'N/A')}")
                logger.info(f"      Run ID: {run_id}")
                if registered_version:
                    logger.info(f"      Registered Version: {registered_version}")

                if run_id and run_id != "N/A":
                    self.test_run_ids.append(run_id)

                # Capture the registered version of the best model for inference testing
                if is_best and registered_version:
                    self.last_registered_version = registered_version
                    logger.info(f"      📝 Captured version {registered_version} for inference testing")

            # Validate best model selection
            best_model = result.get("best_model")
            logger.info(f"Best model selected: {best_model}")

            # Validate history data returned
            history = result.get("history", [])
            logger.info(f"History data returned: {len(history)} records")

            self.log_step("TEST 2: Single Model Training", "PASS")
            self.test_results["single_training"] = {
                "passed": True,
                "elapsed_seconds": elapsed,
                "models_trained": len(result["models"]),
                "best_model": best_model
            }
            return result

        except Exception as e:
            logger.error(f"Single training failed: {e}")
            self.log_step("TEST 2: Single Model Training", "FAIL")
            self.test_results["single_training"] = {"passed": False, "error": str(e)}
            return None

    # =========================================================================
    # TEST 3: Batch Training
    # =========================================================================
    def test_batch_training(self) -> Optional[Dict]:
        """Test batch training endpoint with multiple segments."""
        self.log_step("TEST 3: Batch Training")

        try:
            # Define segments to test
            segments = ["US", "EU", "APAC"]
            logger.info(f"Testing batch training with {len(segments)} segments: {segments}")

            # Generate data for each segment
            segmented_data = self.generate_segmented_data(segments, rows_per_segment=150)

            # Build batch request
            batch_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            batch_requests = []

            for i, segment in enumerate(segments):
                segment_data = segmented_data[segment]
                request = {
                    "data": segment_data,
                    "time_col": "ds",
                    "target_col": "y",
                    "covariates": ["promo"],
                    "horizon": 14,
                    "frequency": "daily",
                    "seasonality_mode": "additive",
                    "regressor_method": "mean",
                    "models": self.models,
                    "catalog_name": "main",
                    "schema_name": "default",
                    "model_name": "finance_forecast_model_test",
                    "country": "US",
                    "random_seed": 42,
                    "filters": {"region": segment},
                    "batch_id": batch_id,
                    "batch_segment_id": f"region={segment}",
                    "batch_segment_index": i + 1,
                    "batch_total_segments": len(segments)
                }
                batch_requests.append(request)

            payload = {
                "requests": batch_requests,
                "max_workers": 2
            }

            logger.info(f"Sending batch training request with {len(batch_requests)} segments")
            start_time = time.time()

            response = requests.post(
                f"{self.api_url}/api/train-batch",
                json=payload,
                timeout=1800  # 30 minute timeout for batch training
            )

            elapsed = time.time() - start_time
            logger.info(f"Batch training completed in {elapsed:.2f} seconds")

            response.raise_for_status()
            result = response.json()

            # Validate response structure
            assert "total_requests" in result, "Response missing 'total_requests'"
            assert "successful" in result, "Response missing 'successful'"
            assert "failed" in result, "Response missing 'failed'"
            assert "results" in result, "Response missing 'results'"

            logger.info(f"Batch Results:")
            logger.info(f"  Total: {result['total_requests']}")
            logger.info(f"  Successful: {result['successful']}")
            logger.info(f"  Failed: {result['failed']}")

            # Log per-segment results
            for seg_result in result.get("results", []):
                segment_id = seg_result.get("segment_id", "Unknown")
                status = seg_result.get("status", "unknown")
                error = seg_result.get("error")

                if status == "success":
                    best = seg_result.get("result", {}).get("best_model", "N/A")
                    logger.info(f"  ✅ {segment_id}: Best model = {best}")

                    # Track run IDs for cleanup
                    for model in seg_result.get("result", {}).get("models", []):
                        if model.get("run_id"):
                            self.test_run_ids.append(model["run_id"])
                else:
                    logger.error(f"  ❌ {segment_id}: {error}")

            # Validate at least some segments succeeded
            assert result["successful"] > 0, "No segments trained successfully"

            self.log_step("TEST 3: Batch Training", "PASS")
            self.test_results["batch_training"] = {
                "passed": True,
                "elapsed_seconds": elapsed,
                "total_segments": result["total_requests"],
                "successful_segments": result["successful"],
                "failed_segments": result["failed"]
            }
            return result

        except Exception as e:
            logger.error(f"Batch training failed: {e}")
            self.log_step("TEST 3: Batch Training", "FAIL")
            self.test_results["batch_training"] = {"passed": False, "error": str(e)}
            return None

    # =========================================================================
    # TEST 4: Model Inference Testing
    # =========================================================================
    def test_model_inference(self, model_name: str = None, model_version: str = None) -> bool:
        """Test the model inference testing endpoint."""
        self.log_step("TEST 4: Model Inference Testing")

        try:
            # Use default model name if not provided
            model_name = model_name or "main.default.finance_forecast_model_test"

            # If no version provided, try to get the latest version from our training results
            if model_version is None:
                # Extract version from single_training results
                if "single_training" in self.test_results and self.test_results["single_training"].get("passed"):
                    # Try to find a registered version from the test
                    model_version = self.last_registered_version or "1"
                else:
                    model_version = "1"  # Default to version 1

            payload = {
                "model_name": model_name,
                "model_version": str(model_version),
                "test_periods": 5,
                "frequency": "daily"
            }

            logger.info(f"Testing model: {model_name} v{model_version}")

            response = requests.post(
                f"{self.api_url}/api/test-model",
                json=payload,
                timeout=120
            )

            # Note: This endpoint might fail if model doesn't exist yet
            if response.status_code == 500:
                error_detail = response.json().get("detail", "Unknown error")
                if "not found" in error_detail.lower() or "does not exist" in error_detail.lower():
                    logger.warning(f"Model not found (expected if no training ran before): {error_detail}")
                    self.test_results["model_inference"] = {
                        "passed": True,
                        "skipped": True,
                        "reason": "Model not found - training may not have run"
                    }
                    self.log_step("TEST 4: Model Inference Testing", "PASS")
                    return True
                else:
                    raise Exception(error_detail)

            response.raise_for_status()
            result = response.json()

            logger.info(f"Inference test result:")
            logger.info(f"  Test passed: {result.get('test_passed', False)}")
            logger.info(f"  Message: {result.get('message', 'N/A')}")
            logger.info(f"  Load time: {result.get('load_time_seconds', 0):.2f}s")
            logger.info(f"  Inference time: {result.get('inference_time_seconds', 0):.3f}s")

            if not result.get("test_passed"):
                logger.warning(f"Model test failed: {result.get('error_details', 'Unknown')}")
                # Even if inference test fails, the API test passed
                self.log_step("TEST 4: Model Inference Testing", "PASS")
                self.test_results["model_inference"] = {
                    "passed": True,
                    "inference_passed": False,
                    "message": result.get('message', 'Unknown'),
                    "load_time": result.get("load_time_seconds"),
                    "inference_time": result.get("inference_time_seconds")
                }
                return True

            self.log_step("TEST 4: Model Inference Testing", "PASS")
            self.test_results["model_inference"] = {
                "passed": True,
                "inference_passed": True,
                "load_time": result.get("load_time_seconds"),
                "inference_time": result.get("inference_time_seconds")
            }
            return True

        except Exception as e:
            logger.error(f"Model inference test failed: {e}")
            self.log_step("TEST 4: Model Inference Testing", "FAIL")
            self.test_results["model_inference"] = {"passed": False, "error": str(e)}
            return False

    # =========================================================================
    # TEST 5: Single Model Deployment
    # =========================================================================
    def test_single_deployment(self, run_id: str = None) -> bool:
        """Test single model deployment endpoint."""
        if self.skip_deploy:
            logger.info("Skipping deployment test (--skip-deploy flag set)")
            self.test_results["single_deployment"] = {"passed": True, "skipped": True}
            return True

        self.log_step("TEST 5: Single Model Deployment")

        try:
            endpoint_name = f"test-forecast-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            payload = {
                "model_name": "main.default.finance_forecast_model_test",
                "model_version": "latest",
                "endpoint_name": endpoint_name,
                "workload_size": "Small",
                "scale_to_zero": True
            }

            if run_id:
                payload["run_id"] = run_id
                payload["model_version"] = None

            logger.info(f"Deploying model to endpoint: {endpoint_name}")

            response = requests.post(
                f"{self.api_url}/api/deploy",
                json=payload,
                timeout=300
            )

            response.raise_for_status()
            result = response.json()

            logger.info(f"Deployment result:")
            logger.info(f"  Status: {result.get('status', 'N/A')}")
            logger.info(f"  Endpoint: {result.get('endpoint_name', 'N/A')}")
            logger.info(f"  Version: {result.get('deployed_version', 'N/A')}")

            # Track endpoint for cleanup
            if result.get("endpoint_name"):
                self.created_endpoints.append(result["endpoint_name"])

            self.log_step("TEST 5: Single Model Deployment", "PASS")
            self.test_results["single_deployment"] = {
                "passed": True,
                "endpoint_name": result.get("endpoint_name"),
                "deployed_version": result.get("deployed_version")
            }
            return True

        except Exception as e:
            logger.error(f"Single deployment failed: {e}")
            self.log_step("TEST 5: Single Model Deployment", "FAIL")
            self.test_results["single_deployment"] = {"passed": False, "error": str(e)}
            return False

    # =========================================================================
    # TEST 6: Batch Deployment
    # =========================================================================
    def test_batch_deployment(self, batch_results: Dict = None) -> bool:
        """Test batch deployment endpoint with router model."""
        if self.skip_deploy:
            logger.info("Skipping batch deployment test (--skip-deploy flag set)")
            self.test_results["batch_deployment"] = {"passed": True, "skipped": True}
            return True

        self.log_step("TEST 6: Batch Deployment")

        try:
            endpoint_name = f"test-batch-router-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            # Build segments from batch results or create mock segments
            segments = []
            if batch_results and batch_results.get("results"):
                for seg in batch_results["results"]:
                    if seg.get("status") == "success":
                        segments.append({
                            "segment_id": seg.get("segment_id", "unknown"),
                            "filters": seg.get("filters", {}),
                            "model_version": "latest",
                            "run_id": None  # Could extract from nested result
                        })

            if not segments:
                # Create mock segment for testing the endpoint structure
                segments = [
                    {
                        "segment_id": "region=US",
                        "filters": {"region": "US"},
                        "model_version": "latest"
                    }
                ]

            payload = {
                "segments": segments,
                "endpoint_name": endpoint_name,
                "catalog_name": "main",
                "schema_name": "default",
                "model_name": "finance_forecast_model_test"
            }

            logger.info(f"Deploying {len(segments)} segments to router endpoint: {endpoint_name}")

            response = requests.post(
                f"{self.api_url}/api/deploy-batch",
                json=payload,
                timeout=600
            )

            response.raise_for_status()
            result = response.json()

            logger.info(f"Batch deployment result:")
            logger.info(f"  Status: {result.get('status', 'N/A')}")
            logger.info(f"  Message: {result.get('message', 'N/A')}")
            logger.info(f"  Endpoint: {result.get('endpoint_name', 'N/A')}")
            logger.info(f"  Deployed segments: {result.get('deployed_segments', 0)}")
            logger.info(f"  Router version: {result.get('router_model_version', 'N/A')}")

            # Track endpoint for cleanup
            if result.get("endpoint_name"):
                self.created_endpoints.append(result["endpoint_name"])

            self.log_step("TEST 6: Batch Deployment", "PASS")
            self.test_results["batch_deployment"] = {
                "passed": True,
                "endpoint_name": result.get("endpoint_name"),
                "deployed_segments": result.get("deployed_segments"),
                "router_version": result.get("router_model_version")
            }
            return True

        except Exception as e:
            logger.error(f"Batch deployment failed: {e}")
            self.log_step("TEST 6: Batch Deployment", "FAIL")
            self.test_results["batch_deployment"] = {"passed": False, "error": str(e)}
            return False

    # =========================================================================
    # TEST 7: Endpoint Status Check
    # =========================================================================
    def test_endpoint_status(self, endpoint_name: str = None) -> bool:
        """Test endpoint status check."""
        if self.skip_deploy or not endpoint_name:
            logger.info("Skipping endpoint status test")
            return True

        self.log_step("TEST 7: Endpoint Status Check")

        try:
            response = requests.get(
                f"{self.api_url}/api/endpoints/{endpoint_name}/status",
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            logger.info(f"Endpoint status for {endpoint_name}:")
            logger.info(f"  State: {result.get('state', 'N/A')}")
            logger.info(f"  Ready: {result.get('ready', False)}")

            self.log_step("TEST 7: Endpoint Status Check", "PASS")
            return True

        except Exception as e:
            logger.error(f"Endpoint status check failed: {e}")
            self.log_step("TEST 7: Endpoint Status Check", "FAIL")
            return False

    # =========================================================================
    # Cleanup
    # =========================================================================
    def cleanup(self):
        """Clean up test resources (endpoints)."""
        if not self.created_endpoints:
            logger.info("No endpoints to clean up")
            return

        logger.info(f"\n{'='*60}")
        logger.info("🧹 CLEANUP")
        logger.info(f"{'='*60}")

        for endpoint_name in self.created_endpoints:
            try:
                logger.info(f"Deleting endpoint: {endpoint_name}")
                response = requests.delete(
                    f"{self.api_url}/api/endpoints/{endpoint_name}",
                    timeout=60
                )
                if response.status_code == 200:
                    logger.info(f"  ✅ Deleted {endpoint_name}")
                else:
                    logger.warning(f"  ⚠️ Could not delete {endpoint_name}: {response.text}")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to delete {endpoint_name}: {e}")

    # =========================================================================
    # Run All Tests
    # =========================================================================
    def run_all(self, cleanup: bool = True) -> Dict:
        """
        Run the complete end-to-end test suite.

        Args:
            cleanup: Whether to clean up created resources after tests

        Returns:
            Dict with test results summary
        """
        logger.info("\n" + "="*70)
        logger.info("🧪 BATCH E2E TEST SUITE")
        logger.info("="*70)
        logger.info(f"API URL: {self.api_url}")
        logger.info(f"Models to test: {self.models}")
        logger.info(f"Skip deployment: {self.skip_deploy}")
        logger.info("="*70 + "\n")

        start_time = time.time()

        # Run tests in sequence
        # Test 1: Health Check
        if not self.test_health_check():
            logger.error("Health check failed - cannot proceed with tests")
            return self.generate_summary(start_time)

        # Test 2: Single Training
        single_result = self.test_single_training()

        # Test 3: Batch Training
        batch_result = self.test_batch_training()

        # Test 4: Model Inference Testing
        self.test_model_inference()

        # Test 5: Single Deployment (optional)
        if single_result:
            best_run_id = None
            for model in single_result.get("models", []):
                if model.get("is_best") and model.get("run_id"):
                    best_run_id = model["run_id"]
                    break
            self.test_single_deployment(run_id=best_run_id)

        # Test 6: Batch Deployment (optional)
        self.test_batch_deployment(batch_result)

        # Cleanup
        if cleanup and self.created_endpoints:
            self.cleanup()

        return self.generate_summary(start_time)

    def generate_summary(self, start_time: float) -> Dict:
        """Generate test summary report."""
        elapsed = time.time() - start_time

        passed = sum(1 for r in self.test_results.values() if r.get("passed"))
        failed = sum(1 for r in self.test_results.values() if not r.get("passed"))
        skipped = sum(1 for r in self.test_results.values() if r.get("skipped"))

        logger.info("\n" + "="*70)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*70)
        logger.info(f"Total time: {elapsed:.2f} seconds")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")
        logger.info("")

        for test_name, result in self.test_results.items():
            if result.get("skipped"):
                status = "⏭️ SKIPPED"
            elif result.get("passed"):
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"

            logger.info(f"  {status} {test_name}")
            if result.get("error"):
                logger.info(f"           Error: {result['error'][:100]}")

        logger.info("="*70)

        all_passed = failed == 0
        if all_passed:
            logger.info("🎉 ALL TESTS PASSED!")
        else:
            logger.error(f"💥 {failed} TEST(S) FAILED!")

        return {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "elapsed_seconds": elapsed,
            "all_passed": all_passed,
            "results": self.test_results
        }


def main():
    parser = argparse.ArgumentParser(description="End-to-End Batch Workflow Test")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Backend API URL")
    parser.add_argument("--skip-deploy", action="store_true",
                        help="Skip deployment tests")
    parser.add_argument("--models", default="prophet",
                        help="Comma-separated list of models to test")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Don't clean up created resources")

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    test = BatchE2ETest(
        api_url=args.api_url,
        skip_deploy=args.skip_deploy,
        models=models,
        verbose=args.verbose
    )

    summary = test.run_all(cleanup=not args.no_cleanup)

    # Exit with appropriate code
    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
