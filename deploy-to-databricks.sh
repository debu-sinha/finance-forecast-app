#!/bin/bash
set -e

echo "🚀 Deploying Finance Forecasting App to Databricks (DAB)"
echo "========================================================"

# Step 1: Clean and Build
echo ""
echo "🧹 Step 1: Cleaning and Building frontend..."
rm -rf dist
find . -type d -name "__pycache__" -exec rm -rf {} +

npm install
npm run build

if [ ! -d "dist" ]; then
    echo "❌ Error: dist/ directory not found after build"
    exit 1
fi

echo "✅ Frontend built successfully"

# Step 2: Deploy Bundle
echo ""
echo "📤 Step 2: Deploying Bundle..."

if ! command -v databricks &> /dev/null; then
    echo "❌ Databricks CLI not found. Please install it."
    exit 1
fi

databricks bundle deploy

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Run your app:"
echo "   databricks bundle run finance-forecast-app"
echo ""
