#!/bin/bash

# Clean restart script - kills old processes and starts fresh

echo "======================================"
echo "Finance Forecasting - Clean Restart"
echo "======================================"
echo ""

# Kill any existing processes
echo "🧹 Cleaning up old processes..."

# Kill uvicorn (backend)
pkill -f "uvicorn backend.main:app" 2>/dev/null && echo "  ✓ Killed old backend" || echo "  • No old backend found"

# Kill npm dev (frontend)  
pkill -f "npm run dev" 2>/dev/null && echo "  ✓ Killed old frontend" || echo "  • No old frontend found"

# Kill vite
pkill -f "vite" 2>/dev/null && echo "  ✓ Killed old vite" || echo "  • No old vite found"

# Kill any process on ports 8000, 3000
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "  ✓ Freed port 8000" || echo "  • Port 8000 already free"
lsof -ti:3000 | xargs kill -9 2>/dev/null && echo "  ✓ Freed port 3000" || echo "  • Port 3000 already free"

# Wait for ports to be released
echo ""
echo "⏳ Waiting for ports to be released..."
sleep 3

# Clean up temp files to free file handles
echo ""
echo "🗑️  Cleaning up temp files..."
rm -f /tmp/*.csv /tmp/*.pkl /tmp/*.json 2>/dev/null || true
rm -rf /tmp/tmp* 2>/dev/null || true
echo "  ✓ Temp files cleaned"

# Increase file descriptor limit (helps with long-running MLflow training)
echo ""
echo "📈 Increasing file descriptor limits..."
ulimit -n 4096 2>/dev/null && echo "  ✓ File limit set to 4096" || echo "  • Could not increase file limit (may require sudo)"

# Load environment variables
if [ ! -f ".env.local" ]; then
    echo "❌ .env.local not found! Run ./setup-local.sh first"
    exit 1
fi

export $(grep -v '^#' .env.local | xargs)
echo "✅ Environment variables loaded"

# Detect Python command (python3 or python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python is not installed"
    exit 1
fi

# Check required variables
if [ -z "$DATABRICKS_HOST" ]; then
    echo "❌ DATABRICKS_HOST not set in .env.local"
    exit 1
fi

if [ -z "$DATABRICKS_TOKEN" ]; then
    echo "❌ DATABRICKS_TOKEN not set in .env.local"
    exit 1
fi

echo ""
echo "🚀 Starting services..."
echo ""

# Start backend with environment
echo "🐍 Starting Python backend on port 8000..."
cd "$(dirname "$0")"
$PYTHON_CMD -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✅ Backend is ready on port 8000"
else
    echo "❌ Backend failed to start. Check backend.log"
    exit 1
fi

# Start frontend
echo "⚛️  Starting React frontend on port 3000..."
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!

echo ""
echo "======================================"
echo "✅ Application is running!"
echo "======================================"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Logs:"
echo "   Backend:  tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "Press Ctrl+C to stop (or run: pkill -f uvicorn && pkill -f vite)"
echo ""

# Keep script running
wait

