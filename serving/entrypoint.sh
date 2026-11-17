#!/usr/bin/env bash
# serving container: load configs/config.yaml -> start the selected backend (vllm | triton).
# Readiness is checked by the docker-compose healthcheck (GET /health).
set -euo pipefail
cd /workspace

eval "$(python3 serving/gen_env.py)"
echo "[serving] SERVING=${SERVING} MODEL_ID=${MODEL_ID} mm_processor_kwargs=${MM_PROCESSOR_KWARGS}"

case "${SERVING}" in
  vllm)   exec bash serving/vllm.sh ;;
  triton) echo "[serving] SERVING=triton is not implemented yet (planned for P6)"; exit 1 ;;
  *)      echo "[serving] unknown SERVING=${SERVING} (expected vllm|triton)"; exit 1 ;;
esac
