#!/bin/bash
# =============================================================================
# Quick Setup Script - Validates environment and shows deployment options
# =============================================================================
# Usage: ./scripts/setup.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Finance Forecasting Platform - Quick Setup               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Databricks CLI
if command -v databricks &> /dev/null; then
    echo "  ✓ Databricks CLI installed"

    # List available profiles
    echo ""
    echo "Available Databricks profiles:"
    databricks auth profiles 2>/dev/null || echo "  (run 'databricks auth login' to configure)"
else
    echo "  ✗ Databricks CLI not found"
    echo "    Install: pip install databricks-cli"
fi

# Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "  ✓ $PYTHON_VERSION"
else
    echo "  ✗ Python 3 not found"
fi

# Node.js (optional for frontend)
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "  ✓ Node.js $NODE_VERSION"
else
    echo "  ○ Node.js not found (optional - needed for frontend build)"
fi

# jq
if command -v jq &> /dev/null; then
    echo "  ✓ jq installed"
else
    echo "  ✗ jq not found"
    echo "    Install: brew install jq (macOS) or apt-get install jq (Linux)"
fi

echo ""
echo "Deployment Options:"
echo ""
echo "  1. Full Deployment (recommended for first time):"
echo "     ./scripts/deploy.sh dev"
echo ""
echo "  2. Production Deployment:"
echo "     ./scripts/deploy.sh prod"
echo ""
echo "  3. Use specific Databricks profile:"
echo "     DATABRICKS_PROFILE=myprofile ./scripts/deploy.sh dev"
echo ""
echo "  4. Teardown (remove all resources):"
echo "     ./scripts/teardown.sh dev"
echo ""
echo "Environment Variables (optional):"
echo "  DATABRICKS_PROFILE  - Databricks CLI profile to use"
echo "  UC_CATALOG_NAME     - Unity Catalog name (default: main)"
echo "  UC_SCHEMA_NAME      - Unity Catalog schema (default: default)"
echo ""
