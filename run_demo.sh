#!/bin/bash
set -e

mkdir -p results logs data

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$NVIDIA_API_KEY" ]; then
    echo "ERROR: NVIDIA_API_KEY not set. Copy .env.example to .env and add your key."
    exit 1
fi

echo "Starting MetaJudge AI backend..."
uvicorn fastapi_app:app --reload --port 8000 --log-level warning &
BACKEND_PID=$!
sleep 3

echo ""
echo "Health check:"
curl -s http://localhost:8000/health | python3 -m json.tool
echo ""
echo "Open demo/index.html in your browser"
echo "API docs: http://localhost:8000/docs"
echo "Press Ctrl+C to stop"

trap "kill $BACKEND_PID 2>/dev/null" EXIT
wait
