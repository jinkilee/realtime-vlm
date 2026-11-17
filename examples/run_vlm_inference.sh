#!/usr/bin/env bash
# Run examples/run_vlm_inference.py inside the running realtime-vlm container.
# Paths are relative to the project root (mounted at /workspace). Usage: sh examples/run_vlm_inference.sh --help
docker exec -i -u "$(id -u):$(id -g)" -w /workspace realtime-vlm python3 examples/run_vlm_inference.py "$@"
