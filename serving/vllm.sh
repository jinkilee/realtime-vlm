#!/usr/bin/env bash
# vLLM OpenAI-compatible server. Values come from configs/config.yaml (exported by entrypoint.sh).
exec vllm serve "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${SERVING_PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
  --mm-processor-kwargs "${MM_PROCESSOR_KWARGS}" \
  ${VLLM_EXTRA_ARGS:-}
