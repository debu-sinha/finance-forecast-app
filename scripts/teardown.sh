#!/bin/bash
# =============================================================================
# Finance Forecasting Platform - Teardown Script
# =============================================================================
# Removes all deployed resources:
#   - Databricks App
#   - Training Job
#   - Dedicated Cluster
#   - (Optional) Lakebase tables
#
# Usage:
#   ./scripts/teardown.sh [dev|prod] [--include-data]
#
# Options:
#   --include-data    Also delete Lakebase database and data
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TARGET="${1:-dev}"
INCLUDE_DATA=false
PROFILE="${DATABRICKS_PROFILE:-DEFAULT}"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --include-data)
            INCLUDE_DATA=true
            shift
            ;;
    esac
done

APP_NAME="finance-forecast-app"
CLUSTER_NAME="forecast-training-cluster-${TARGET}"
JOB_NAME="forecast-training-job-${TARGET}"

echo ""
echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║     Finance Forecasting Platform - Teardown                  ║${NC}"
echo -e "${YELLOW}║     Target: ${TARGET}                                              ║${NC}"
echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Confirm
echo -e "${YELLOW}WARNING: This will delete the following resources:${NC}"
echo "  - Databricks App: $APP_NAME"
echo "  - Training Job: $JOB_NAME"
echo "  - Training Cluster: $CLUSTER_NAME"
if [ "$INCLUDE_DATA" = true ]; then
    echo "  - Lakebase Database: forecast_${TARGET}"
fi
echo ""
read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Teardown cancelled."
    exit 0
fi

# Delete App
echo -e "${BLUE}[INFO]${NC} Deleting Databricks App..."
databricks apps delete "$APP_NAME" -p "$PROFILE" 2>/dev/null || echo "  App not found or already deleted"

# Delete Job
echo -e "${BLUE}[INFO]${NC} Deleting Training Job..."
JOB_ID=$(databricks jobs list -p "$PROFILE" --output json 2>/dev/null | \
    jq -r ".jobs[] | select(.settings.name == \"$JOB_NAME\") | .job_id" || echo "")
if [ -n "$JOB_ID" ]; then
    databricks jobs delete "$JOB_ID" -p "$PROFILE"
    echo "  Job $JOB_ID deleted"
else
    echo "  Job not found"
fi

# Delete Cluster
echo -e "${BLUE}[INFO]${NC} Deleting Training Cluster..."
CLUSTER_ID=$(databricks clusters list -p "$PROFILE" --output json 2>/dev/null | \
    jq -r ".clusters[] | select(.cluster_name == \"$CLUSTER_NAME\") | .cluster_id" || echo "")
if [ -n "$CLUSTER_ID" ]; then
    databricks clusters permanent-delete "$CLUSTER_ID" -p "$PROFILE"
    echo "  Cluster $CLUSTER_ID deleted"
else
    echo "  Cluster not found"
fi

# Delete Lakebase data (if requested)
if [ "$INCLUDE_DATA" = true ]; then
    echo -e "${BLUE}[INFO]${NC} Deleting Lakebase database..."
    echo "  Note: Lakebase database deletion requires manual intervention"
    echo "  Run: DROP DATABASE IF EXISTS lakebase.forecast_${TARGET} CASCADE"
fi

# Clean up local config
CONFIG_FILE="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/.deployment-config-${TARGET}.json"
if [ -f "$CONFIG_FILE" ]; then
    rm "$CONFIG_FILE"
    echo -e "${BLUE}[INFO]${NC} Removed local config: $CONFIG_FILE"
fi

echo ""
echo -e "${GREEN}Teardown complete!${NC}"
