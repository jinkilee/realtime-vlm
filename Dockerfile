# realtime-vlm runtime image: vLLM + system/python dependencies only.
# Source code (app/, serving/, configs/, examples/, entrypoint.sh) is NOT baked in;
# the project directory is mounted at /workspace by run_docker.sh.
ARG VLLM_VERSION=v0.33.0
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
EXPOSE 8000 9000
ENTRYPOINT ["bash", "/workspace/entrypoint.sh"]
