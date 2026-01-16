#!/bin/bash

# Start script for local development
# This script starts both the backend and frontend in development mode

set -e

echo "======================================"
echo "Finance Forecasting Platform - Local"
echo "======================================"
echo ""

# Detect Python command (python3 or python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python is not installed"
    exit 1
fi

# Detect pip command (pip3 or pip, or python -m pip)
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    PIP_CMD="$PYTHON_CMD -m pip"
fi

# Check for .env.local file
if [ ! -f ".env.local" ]; then
    echo "⚠️  .env.local not found!"
    echo "Please run ./setup-local.sh first."
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env.local | xargs)

# Check for Databricks environment variables (optional - fallback mode works without them)
DATABRICKS_MODE="full"
if [ -z "$DATABRICKS_HOST" ] || [ "$DATABRICKS_HOST" = "https://your-workspace.cloud.databricks.com" ]; then
    DATABRICKS_MODE="fallback"
    echo "⚠️  DATABRICKS_HOST not configured - running in fallback mode"
fi

if [ -z "$DATABRICKS_TOKEN" ] || [ "$DATABRICKS_TOKEN" = "your_databricks_token_here" ]; then
    DATABRICKS_MODE="fallback"
    echo "⚠️  DATABRICKS_TOKEN not configured - running in fallback mode"
fi

if [ "$DATABRICKS_MODE" = "fallback" ]; then
    echo ""
    echo "📌 Fallback mode enabled:"
    echo "   - MLflow will use local SQLite database"
    echo "   - AI Thinker will provide basic analysis (Opus 4.5 unavailable)"
    echo "   - All other features work normally"
    echo ""
    echo "   To enable full features, set in .env.local:"
    echo "   - DATABRICKS_HOST=https://your-workspace.cloud.databricks.com"
    echo "   - DATABRICKS_TOKEN=your_token"
    echo "   - OPUS_ENDPOINT_NAME=your-opus-endpoint (optional)"
    echo ""
else
    echo "✅ Databricks credentials configured"
    if [ -n "$OPUS_ENDPOINT_NAME" ]; then
        echo "✅ AI Thinker endpoint: $OPUS_ENDPOINT_NAME"
    else
        echo "ℹ️  AI Thinker will use default endpoint: opus-4-5-thinker"
    fi
fi

echo "✅ Environment variables loaded"
echo ""

# Check if Python packages are installed
if ! $PYTHON_CMD -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing Python dependencies..."
    $PIP_CMD install -r requirements.txt
fi

# Check if Node modules are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node dependencies..."
    npm install
fi

echo ""
echo "🚀 Starting services..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM

# Start backend
echo "🐍 Starting Python backend on port 8000..."
$PYTHON_CMD -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "⚛️  Starting React frontend on port 3000..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "======================================"
echo "✅ Application is running!"
echo "======================================"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🧠 AI Thinker: http://localhost:8000/api/thinker/status"
echo ""
if [ "$DATABRICKS_MODE" = "fallback" ]; then
    echo "⚠️  Running in fallback mode (no Databricks)"
fi
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
