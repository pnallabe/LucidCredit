#!/bin/bash
# Start LucidCredit backend on port 8090
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
ENV_FILE="$SCRIPT_DIR/.env"

cd "$BACKEND_DIR"

# Load env
set -a
source "$ENV_FILE"
set +a

exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload
