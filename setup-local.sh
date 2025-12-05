#!/bin/bash

# Setup script for local development environment
# Run this once to set up your development environment

set -e

echo "======================================"
echo "Finance Forecasting - Setup"
echo "======================================"
echo ""

# Detect Python command (python3 or python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python is not installed"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

# Detect pip command (pip3 or pip, or python -m pip)
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    # Fallback to python -m pip
    PIP_CMD="$PYTHON_CMD -m pip"
fi

# Check Python version
echo "🐍 Checking Python version..."
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')

if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10+ is required. Found: $PYTHON_VERSION"
    echo "Please install Python 3.10 or higher"
    exit 1
fi
echo "✅ Python $PYTHON_VERSION (using: $PYTHON_CMD)"
echo "✅ Pip (using: $PIP_CMD)"

# Check Node version
echo "📦 Checking Node.js version..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    echo "Please install Node.js 18+ from https://nodejs.org/"
    exit 1
fi
NODE_VERSION=$(node --version)
echo "✅ Node.js $NODE_VERSION"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed"
    exit 1
fi
NPM_VERSION=$(npm --version)
echo "✅ npm $NPM_VERSION"

# Check for OpenMP on macOS (required for XGBoost)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Checking OpenMP for XGBoost (macOS)..."
    if [ -f "/opt/homebrew/opt/libomp/lib/libomp.dylib" ] || [ -f "/usr/local/opt/libomp/lib/libomp.dylib" ]; then
        echo "✅ OpenMP is installed"
    else
        echo "⚠️  OpenMP not found. XGBoost requires it on macOS."
        echo "   Install with: brew install libomp"
        echo "   Continuing setup, but XGBoost may fail without OpenMP."
    fi
fi

echo ""
echo "📦 Installing Python dependencies..."
$PIP_CMD install -r requirements.txt

echo ""
echo "📦 Installing Node.js dependencies..."
npm install

echo ""
echo "⚙️  Setting up environment files..."
if [ ! -f ".env.local" ]; then
    cat > .env.local << 'EOF'
# Backend Authentication
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_databricks_token_here

# Frontend Authentication (required for AI features)
VITE_DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
VITE_DATABRICKS_TOKEN=your_databricks_token_here

# MLflow Experiment Path
MLFLOW_EXPERIMENT_NAME=/Users/your.email@company.com/finance-forecasting

# Python environment
PYTHONUNBUFFERED=1
EOF
    echo "✅ Created .env.local with template"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env.local and add your credentials:"
    echo "   - DATABRICKS_HOST (your workspace URL)"
    echo "   - DATABRICKS_TOKEN (from User Settings > Access Tokens)"
    echo ""
    echo "   That's it! No Gateway setup needed."
else
    echo "✅ .env.local already exists"
fi

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env.local and add your credentials"
echo "  2. Run: ./start-local.sh"
echo ""
echo "For deployment to Databricks:"
echo "  See DEPLOYMENT.md for detailed instructions"
echo ""

