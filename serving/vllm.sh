#!/usr/bin/env bash
# vLLM OpenAI-compatible server. All knobs come from env (see Dockerfile defaults).
exec vllm serve "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${SERVING_PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --limit-mm-per-prompt "{\"image\": {\"count\": ${MAX_IMAGES}, \"width\": ${MAX_IMAGE_SIZE}, \"height\": ${MAX_IMAGE_SIZE}}, \"video\": 0}" \
  ${VLLM_EXTRA_ARGS:-}
