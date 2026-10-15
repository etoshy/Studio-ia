#!/usr/bin/env bash
# Para a API e os workers.
ROOT="$(cd "$(dirname "$0")" && pwd)"
pkill -f "$ROOT/workers/worker_images.py" 2>/dev/null
pkill -f "$ROOT/workers/worker_tts.py" 2>/dev/null
pkill -f "uvicorn app.main:app" 2>/dev/null
echo "parado"
