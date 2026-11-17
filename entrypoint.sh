#!/usr/bin/env bash
# 1) start the serving backend (vllm | triton)  2) wait until healthy  3) start the app
set -euo pipefail
cd /workspace

log() { echo "[entrypoint] $*"; }

case "${SERVING}" in
  vllm)   ./serving/vllm.sh & ;;
  triton) log "SERVING=triton is not implemented yet (planned for P6)"; exit 1 ;;
  *)      log "unknown SERVING=${SERVING} (expected vllm|triton)"; exit 1 ;;
esac
SERVING_PID=$!
trap 'kill -TERM ${SERVING_PID} 2>/dev/null; wait ${SERVING_PID}' INT TERM

log "waiting for ${SERVING} on :${SERVING_PORT} (first run downloads the model) ..."
until curl -sf "http://127.0.0.1:${SERVING_PORT}/health" > /dev/null; do
  if ! kill -0 "${SERVING_PID}" 2>/dev/null; then
    log "serving process exited during startup"; exit 1
  fi
  sleep 3
done
log "${SERVING} is ready: model=${MODEL_ID} served-as=${SERVED_MODEL_NAME}"

if [[ -f app/main.py ]]; then
  log "starting app on :${APP_PORT}"
  python -m uvicorn app.main:app --host 0.0.0.0 --port "${APP_PORT}" &
  APP_PID=$!
  trap 'kill -TERM ${APP_PID} ${SERVING_PID} 2>/dev/null; wait' INT TERM
  wait -n "${APP_PID}" "${SERVING_PID}"
else
  log "app not present yet — serving only"
  wait "${SERVING_PID}"
fi
