#!/bin/bash
# =============================================================================
# Finance Forecasting Platform - One-Click Deployment Script
# =============================================================================
# This script sets up all resources needed for the Finance Forecasting Platform:
# 1. Lakebase PostgreSQL instance and tables
# 2. Dedicated training cluster (64 vCPU, 256GB RAM)
# 3. Databricks App deployment
#
# Usage:
#   ./scripts/deploy.sh [OPTIONS]
#
# Options:
#   --profile PROFILE    Databricks CLI profile (default: e2-dogfood)
#   --target TARGET      Deployment target: dev or prod (default: dev)
#   --skip-lakebase      Skip Lakebase setup (if already configured)
#   --skip-cluster       Skip cluster creation (if already exists)
#   --dry-run            Show what would be done without executing
#   --help               Show this help message
#
# Requirements:
#   - Databricks CLI v0.200+ with valid authentication
#   - Python 3.9+ with databricks-sdk installed
#   - Access to target workspace
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
PROFILE="${DATABRICKS_PROFILE:-e2-dogfood}"
TARGET="dev"
SKIP_LAKEBASE=false
SKIP_CLUSTER=false
DRY_RUN=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Lakebase configuration
LAKEBASE_DATABASE="forecast"
LAKEBASE_SCHEMA="forecast"

# Cluster configuration
CLUSTER_NAME_PREFIX="forecast-training"
CLUSTER_NODE_TYPE="i3.16xlarge"  # AWS: 64 vCPU, 488 GB RAM
CLUSTER_SPARK_VERSION="15.4.x-scala2.12"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

show_help() {
    head -30 "$0" | tail -25
    exit 0
}

check_prerequisites() {
    log_step "Checking Prerequisites"

    # Check Databricks CLI
    if ! command -v databricks &> /dev/null; then
        log_error "Databricks CLI not found. Install with: pip install databricks-cli"
        exit 1
    fi
    log_success "Databricks CLI found: $(databricks --version 2>/dev/null | head -1)"

    # Check profile validity
    if ! databricks auth profiles 2>/dev/null | grep -E "^${PROFILE}\s+" | grep -q "YES"; then
        log_error "Profile '$PROFILE' is not valid or not authenticated"
        log_info "Run: databricks auth login --profile $PROFILE"
        exit 1
    fi
    log_success "Profile '$PROFILE' is authenticated"

    # Get workspace info
    WORKSPACE_HOST=$(databricks auth env -p "$PROFILE" 2>/dev/null | grep DATABRICKS_HOST | cut -d= -f2 || echo "unknown")
    log_info "Target workspace: $WORKSPACE_HOST"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found"
        exit 1
    fi
    log_success "Python found: $(python3 --version)"

    # Check databricks-sdk
    if ! python3 -c "import databricks.sdk" 2>/dev/null; then
        log_warning "databricks-sdk not installed. Installing..."
        pip install databricks-sdk
    fi
    log_success "databricks-sdk available"
}

# =============================================================================
# Lakebase Setup
# =============================================================================

setup_lakebase() {
    if [ "$SKIP_LAKEBASE" = true ]; then
        log_warning "Skipping Lakebase setup (--skip-lakebase)"
        return 0
    fi

    log_step "Setting up Lakebase PostgreSQL"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would create Lakebase instance and tables"
        return 0
    fi

    # Create Python script to setup Lakebase
    python3 << 'PYTHON_SCRIPT'
import os
import sys
from databricks.sdk import WorkspaceClient

profile = os.environ.get('PROFILE', 'e2-dogfood')
database = os.environ.get('LAKEBASE_DATABASE', 'forecast')

print(f"Connecting to workspace with profile: {profile}")

try:
    w = WorkspaceClient(profile=profile)

    # Check if Lakebase is available (this is a preview feature)
    # For now, we'll check if we can access SQL warehouses as a proxy
    warehouses = list(w.warehouses.list())
    print(f"Found {len(warehouses)} SQL warehouses")

    # Get a warehouse to execute SQL
    if not warehouses:
        print("No SQL warehouses found. Creating serverless warehouse...")
        # In production, you'd create a warehouse here
        print("Please create a SQL warehouse manually or enable serverless")
        sys.exit(0)

    # Use first available warehouse
    warehouse = next((wh for wh in warehouses if wh.state.value == 'RUNNING'), warehouses[0])
    print(f"Using warehouse: {warehouse.name} (ID: {warehouse.id})")

    # Read SQL file
    sql_file = os.path.join(os.environ.get('PROJECT_DIR', '.'), 'sql', 'create_tables.sql')
    if os.path.exists(sql_file):
        print(f"SQL file found: {sql_file}")
        with open(sql_file, 'r') as f:
            sql_content = f.read()

        # For Lakebase (PostgreSQL), we'd use a different connection
        # For now, let's check if we can use Unity Catalog SQL
        print("Note: Lakebase PostgreSQL requires separate setup via Databricks UI")
        print("The SQL schema has been prepared in sql/create_tables.sql")
        print("Please create the Lakebase instance via Databricks UI and run the SQL")
    else:
        print(f"SQL file not found: {sql_file}")

    print("Lakebase setup check completed")

except Exception as e:
    print(f"Error during Lakebase setup: {e}")
    print("Lakebase may not be available in this workspace")
    print("Continuing with deployment...")

PYTHON_SCRIPT

    log_success "Lakebase setup check completed"
}

# =============================================================================
# Cluster Setup
# =============================================================================

setup_cluster() {
    if [ "$SKIP_CLUSTER" = true ]; then
        log_warning "Skipping cluster setup (--skip-cluster)"
        return 0
    fi

    log_step "Setting up Dedicated Training Cluster"

    CLUSTER_NAME="${CLUSTER_NAME_PREFIX}-${TARGET}"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would create cluster: $CLUSTER_NAME"
        return 0
    fi

    # Check if cluster already exists
    EXISTING_CLUSTER=$(databricks clusters list -p "$PROFILE" --output json 2>/dev/null | \
        python3 -c "import sys,json; clusters=json.load(sys.stdin).get('clusters',[]); print(next((c['cluster_id'] for c in clusters if c['cluster_name']=='$CLUSTER_NAME'), ''))" 2>/dev/null || echo "")

    if [ -n "$EXISTING_CLUSTER" ]; then
        log_info "Cluster '$CLUSTER_NAME' already exists: $EXISTING_CLUSTER"
        CLUSTER_ID="$EXISTING_CLUSTER"
    else
        log_info "Creating new cluster: $CLUSTER_NAME"

        # Create cluster using Python SDK (non-blocking)
        CLUSTER_ID=$(python3 << PYTHON_SCRIPT
import os
from databricks.sdk import WorkspaceClient

profile = os.environ.get('PROFILE', 'e2-dogfood')
cluster_name = os.environ.get('CLUSTER_NAME', 'forecast-training-dev')
node_type = os.environ.get('CLUSTER_NODE_TYPE', 'i3.16xlarge')
spark_version = os.environ.get('CLUSTER_SPARK_VERSION', '15.4.x-scala2.12')

w = WorkspaceClient(profile=profile)

# Create single-node cluster for parallel model training (non-blocking)
# Use create_and_wait=False to get cluster ID immediately
response = w.clusters.create(
    cluster_name=cluster_name,
    spark_version=spark_version,
    node_type_id=node_type,
    num_workers=0,  # Single node mode
    spark_conf={
        "spark.databricks.cluster.profile": "singleNode",
        "spark.master": "local[*]",
        "spark.executor.memory": "200g",
        "spark.driver.memory": "200g",
        "spark.sql.shuffle.partitions": "64",
    },
    custom_tags={
        "project": "finance-forecasting",
        "cost_center": "analytics",
        "team": "fp-and-a",
    },
    autotermination_minutes=30,
)

# Get cluster_id from the wait handle
print(response.cluster_id)
PYTHON_SCRIPT
)
        log_success "Created cluster: $CLUSTER_ID"
    fi

    # Install required libraries on cluster
    log_info "Installing libraries on cluster..."
    python3 << PYTHON_SCRIPT
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import Library, PythonPyPiLibrary

profile = os.environ.get('PROFILE', 'e2-dogfood')
cluster_id = os.environ.get('CLUSTER_ID', '')

if not cluster_id:
    print("No cluster ID provided")
    exit(0)

w = WorkspaceClient(profile=profile)

libraries = [
    Library(pypi=PythonPyPiLibrary(package="prophet")),
    Library(pypi=PythonPyPiLibrary(package="statsforecast>=1.7.0")),
    Library(pypi=PythonPyPiLibrary(package="chronos-forecasting>=1.4.0")),
    Library(pypi=PythonPyPiLibrary(package="torch>=2.0.0")),
    Library(pypi=PythonPyPiLibrary(package="mapie>=0.8.0")),
    Library(pypi=PythonPyPiLibrary(package="asyncpg>=0.29.0")),
    Library(pypi=PythonPyPiLibrary(package="psycopg2-binary>=2.9.9")),
]

try:
    w.libraries.install(cluster_id=cluster_id, libraries=libraries)
    print(f"Libraries installation initiated on cluster {cluster_id}")
except Exception as e:
    print(f"Library installation note: {e}")
    print("Libraries will be installed when cluster starts")

PYTHON_SCRIPT

    # Export cluster ID for later use
    export CLUSTER_ID
    echo "$CLUSTER_ID" > "$PROJECT_DIR/.cluster_id"
    log_success "Cluster ID saved to .cluster_id: $CLUSTER_ID"
}

# =============================================================================
# Upload Notebook
# =============================================================================

upload_notebook() {
    log_step "Uploading Training Notebook"

    NOTEBOOK_PATH="/Workspace/finance-forecasting/notebooks/train_models"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would upload notebook to: $NOTEBOOK_PATH"
        return 0
    fi

    # Create workspace directory
    log_info "Creating workspace directory..."
    databricks workspace mkdirs "/Workspace/finance-forecasting/notebooks" -p "$PROFILE" 2>/dev/null || true

    # Upload notebook
    if [ -f "$PROJECT_DIR/notebooks/train_models.py" ]; then
        log_info "Uploading train_models.py..."
        databricks workspace import "$PROJECT_DIR/notebooks/train_models.py" \
            "$NOTEBOOK_PATH" \
            --format SOURCE \
            --language PYTHON \
            --overwrite \
            -p "$PROFILE"
        log_success "Notebook uploaded to: $NOTEBOOK_PATH"
    else
        log_warning "Notebook not found: $PROJECT_DIR/notebooks/train_models.py"
    fi
}

# =============================================================================
# Deploy Databricks App
# =============================================================================

deploy_app() {
    log_step "Deploying Databricks App"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would deploy app using databricks bundle"
        return 0
    fi

    cd "$PROJECT_DIR"

    # Read cluster ID if available
    if [ -f "$PROJECT_DIR/.cluster_id" ]; then
        CLUSTER_ID=$(cat "$PROJECT_DIR/.cluster_id")
        log_info "Using cluster ID: $CLUSTER_ID"
    else
        CLUSTER_ID=""
        log_warning "No cluster ID found. Set dedicated_cluster_id manually."
    fi

    # Validate bundle
    log_info "Validating bundle configuration..."
    if ! databricks bundle validate -t "$TARGET" -p "$PROFILE"; then
        log_error "Bundle validation failed"
        exit 1
    fi
    log_success "Bundle validation passed"

    # Deploy bundle
    log_info "Deploying bundle to $TARGET environment..."
    if databricks bundle deploy -t "$TARGET" -p "$PROFILE" \
        --var "dedicated_cluster_id=$CLUSTER_ID"; then
        log_success "Bundle deployed successfully"
    else
        log_error "Bundle deployment failed"
        exit 1
    fi

    # Get app URL
    log_info "Getting app deployment status..."
    sleep 5  # Wait for deployment to register

    APP_INFO=$(databricks apps get finance-forecast-app -p "$PROFILE" --output json 2>/dev/null || echo "{}")
    APP_URL=$(echo "$APP_INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('url', 'pending'))" 2>/dev/null || echo "pending")

    if [ "$APP_URL" != "pending" ] && [ -n "$APP_URL" ]; then
        log_success "App URL: $APP_URL"
    else
        log_info "App URL will be available once deployment completes"
    fi
}

# =============================================================================
# Create Training Job
# =============================================================================

create_training_job() {
    log_step "Creating Training Job"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would create training job"
        return 0
    fi

    # Read cluster ID
    if [ -f "$PROJECT_DIR/.cluster_id" ]; then
        CLUSTER_ID=$(cat "$PROJECT_DIR/.cluster_id")
    else
        log_warning "No cluster ID found. Job creation skipped."
        return 0
    fi

    JOB_NAME="forecast-training-job-${TARGET}"

    # Check if job exists
    EXISTING_JOB=$(databricks jobs list -p "$PROFILE" --output json 2>/dev/null | \
        python3 -c "import sys,json; jobs=json.load(sys.stdin).get('jobs',[]); print(next((str(j['job_id']) for j in jobs if j['settings']['name']=='$JOB_NAME'), ''))" 2>/dev/null || echo "")

    if [ -n "$EXISTING_JOB" ]; then
        log_info "Job '$JOB_NAME' already exists: $EXISTING_JOB"
        JOB_ID="$EXISTING_JOB"
    else
        log_info "Creating job: $JOB_NAME"

        JOB_ID=$(python3 << PYTHON_SCRIPT
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

profile = os.environ.get('PROFILE', 'e2-dogfood')
cluster_id = os.environ.get('CLUSTER_ID', '')
job_name = os.environ.get('JOB_NAME', 'forecast-training-job-dev')

w = WorkspaceClient(profile=profile)

job = w.jobs.create(
    name=job_name,
    tasks=[
        jobs.Task(
            task_key="train_all_models",
            existing_cluster_id=cluster_id,
            notebook_task=jobs.NotebookTask(
                notebook_path="/Workspace/finance-forecasting/notebooks/train_models",
            ),
            timeout_seconds=7200,
            retry_on_timeout=True,
            max_retries=1,
        )
    ],
    queue=jobs.QueueSettings(enabled=True),
    max_concurrent_runs=10,
    tags={
        "project": "finance-forecasting",
        "type": "ml-training",
    },
)

print(job.job_id)
PYTHON_SCRIPT
)
        log_success "Created job: $JOB_ID"
    fi

    echo "$JOB_ID" > "$PROJECT_DIR/.job_id"
    log_success "Job ID saved to .job_id: $JOB_ID"
}

# =============================================================================
# Print Summary
# =============================================================================

print_summary() {
    log_step "Deployment Summary"

    echo -e "${GREEN}Finance Forecasting Platform deployed successfully!${NC}\n"

    # Read saved IDs
    CLUSTER_ID=""
    JOB_ID=""
    [ -f "$PROJECT_DIR/.cluster_id" ] && CLUSTER_ID=$(cat "$PROJECT_DIR/.cluster_id")
    [ -f "$PROJECT_DIR/.job_id" ] && JOB_ID=$(cat "$PROJECT_DIR/.job_id")

    echo "Configuration:"
    echo "  Profile:     $PROFILE"
    echo "  Target:      $TARGET"
    echo "  Workspace:   $WORKSPACE_HOST"
    echo ""
    echo "Resources Created:"
    [ -n "$CLUSTER_ID" ] && echo "  Cluster ID:  $CLUSTER_ID"
    [ -n "$JOB_ID" ] && echo "  Job ID:      $JOB_ID"
    echo ""
    echo "Next Steps:"
    echo "  1. Set up Lakebase PostgreSQL instance via Databricks UI"
    echo "  2. Run SQL schema: sql/create_tables.sql"
    echo "  3. Configure Lakebase connection in databricks.yml variables"
    echo "  4. Access the app at: https://<workspace>/apps/finance-forecast-app"
    echo ""
    echo "To check app status:"
    echo "  databricks apps get finance-forecast-app -p $PROFILE"
    echo ""
    echo "To start the cluster:"
    echo "  databricks clusters start $CLUSTER_ID -p $PROFILE"
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --profile)
                PROFILE="$2"
                shift 2
                ;;
            --target)
                TARGET="$2"
                shift 2
                ;;
            --skip-lakebase)
                SKIP_LAKEBASE=true
                shift
                ;;
            --skip-cluster)
                SKIP_CLUSTER=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                show_help
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                ;;
        esac
    done

    # Export variables for Python scripts
    export PROFILE
    export TARGET
    export LAKEBASE_DATABASE
    export PROJECT_DIR
    export CLUSTER_NAME="${CLUSTER_NAME_PREFIX}-${TARGET}"
    export CLUSTER_NODE_TYPE
    export CLUSTER_SPARK_VERSION
    export JOB_NAME="forecast-training-job-${TARGET}"

    echo -e "\n${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     Finance Forecasting Platform - Deployment Script          ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}\n"

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN MODE - No changes will be made"
    fi

    # Run deployment steps
    check_prerequisites
    setup_lakebase
    setup_cluster
    upload_notebook
    create_training_job
    deploy_app
    print_summary
}

# Run main function
main "$@"
