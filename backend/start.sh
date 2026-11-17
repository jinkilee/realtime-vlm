#!/usr/bin/env bash
# backend container: uvicorn (auto-reload on backend/app or configs/ changes when backend.reload=true).
# Waiting for the serving container is handled by docker-compose (depends_on: service_healthy).
set -euo pipefail
cd /workspace/backend
eval "$(python3 -m app.config shell)"
RELOAD_ARGS=()
if [[ "${BACKEND_RELOAD}" == "1" ]]; then
  RELOAD_ARGS=(--reload --reload-dir /workspace/backend/app --reload-dir /workspace/configs)
fi
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${BACKEND_PORT}" "${RELOAD_ARGS[@]}"
