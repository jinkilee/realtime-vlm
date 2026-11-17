# realtime-vlm runtime image
#   - vLLM (OpenAI-compatible server) for the VLM
#   - FastAPI app (stream decoding + inference client) — added in P2
#   - frontend build stage — added in P4
ARG VLLM_VERSION=v0.33.0
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh ./
COPY serving/ ./serving/
COPY app/ ./app/
RUN chmod +x entrypoint.sh serving/*.sh

# Defaults (override with -e at docker run)
ENV SERVING=vllm \
    MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
    SERVED_MODEL_NAME=vlm \
    SERVING_PORT=9000 \
    APP_PORT=8000 \
    GPU_MEM_UTIL=0.78 \
    MAX_MODEL_LEN=8192 \
    MAX_NUM_SEQS=4 \
    MAX_IMAGES=4 \
    MAX_IMAGE_SIZE=448

EXPOSE 8000 9000
ENTRYPOINT ["/workspace/entrypoint.sh"]
