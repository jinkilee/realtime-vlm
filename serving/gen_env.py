"""Print `export KEY=VALUE` lines for the serving scripts from configs/config.yaml.

    eval "$(python3 serving/gen_env.py)"
Env overrides: SERVING -> serving.backend, MODEL_ID -> serving.model_id
"""
from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RVLM_ROOT", Path(__file__).resolve().parent.parent))
CONFIG = Path(os.environ.get("RVLM_CONFIG", ROOT / "configs" / "config.yaml"))


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    s, p = cfg["serving"], cfg["preprocess"]
    backend = os.environ.get("SERVING") or s["backend"]
    model = os.environ.get("MODEL_ID") or s["model_id"]
    if not Path(model).is_absolute() and (ROOT / model).exists():
        model = str(ROOT / model)
    max_side = int(p["max_side"])
    max_pixels = max_side * max_side
    env = {
        "SERVING": backend,
        "MODEL_ID": model,
        "SERVED_MODEL_NAME": s["served_model_name"],
        "SERVING_PORT": s["port"],
        "GPU_MEM_UTIL": s["gpu_memory_utilization"],
        "MAX_MODEL_LEN": s["max_model_len"],
        "MAX_NUM_SEQS": s["max_num_seqs"],
        "SCHEDULING_POLICY": s.get("scheduling_policy", "fcfs"),
        # vLLM memory profiling assumes images of this size
        "LIMIT_MM_PER_PROMPT": json.dumps(
            {"image": {"count": s["max_images_per_request"], "width": max_side, "height": max_side},
             "video": 0}),
        # hard cap in the HF Qwen2/3-VL image processor (size.* are pixel counts)
        "MM_PROCESSOR_KWARGS": json.dumps(
            {"size": {"longest_edge": max_pixels, "shortest_edge": min(65536, max_pixels)}}),
        "HF_HUB_OFFLINE": "1" if s.get("offline") else "0",
        "VLLM_EXTRA_ARGS": s.get("extra_args") or "",
    }
    print("\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in env.items()))


if __name__ == "__main__":
    main()
