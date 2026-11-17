#!/usr/bin/env bash
# Run run_vlm_inference.py inside the backend container. Paths are relative to the project root.
# Usage: sh backend/examples/run_vlm_inference.sh --help
docker exec -i -w /workspace realtime-vlm-backend python3 backend/examples/run_vlm_inference.py "$@"
