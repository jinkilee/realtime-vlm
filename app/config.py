"""Config loader: configs/config.yaml + a few env overrides.

CLI (used by shell scripts):
    python -m app.config shell     # print `export KEY=VALUE` lines for serving scripts
"""
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("RVLM_CONFIG", ROOT / "configs" / "config.yaml"))

ENV_OVERRIDES = {
    "SERVING": ("serving", "backend"),
    "MODEL_ID": ("serving", "model_id"),
}


def load(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for env, (section, key) in ENV_OVERRIDES.items():
        if os.environ.get(env):
            cfg[section][key] = os.environ[env]
    mid = cfg["serving"]["model_id"]
    if not Path(mid).is_absolute() and (ROOT / mid).exists():
        cfg["serving"]["model_id"] = str(ROOT / mid)
    if cfg["inference"]["num_frames"] > cfg["serving"]["max_images_per_request"]:
        raise ValueError("inference.num_frames must be <= serving.max_images_per_request")
    return cfg


def serving_url(cfg: dict, host: str = "127.0.0.1") -> str:
    return f"http://{host}:{cfg['serving']['port']}"


def shell_exports(cfg: dict) -> str:
    s, p = cfg["serving"], cfg["preprocess"]
    max_side = int(p["max_side"])
    max_pixels = max_side * max_side
    env = {
        "SERVING": s["backend"],
        "MODEL_ID": s["model_id"],
        "SERVED_MODEL_NAME": s["served_model_name"],
        "SERVING_PORT": s["port"],
        "APP_PORT": cfg["app"]["port"],
        "GPU_MEM_UTIL": s["gpu_memory_utilization"],
        "MAX_MODEL_LEN": s["max_model_len"],
        "MAX_NUM_SEQS": s["max_num_seqs"],
        "LIMIT_MM_PER_PROMPT": json.dumps(
            {"image": {"count": s["max_images_per_request"], "width": max_side, "height": max_side},
             "video": 0}),
        # HF Qwen2/3-VL image processor: size.longest_edge / shortest_edge are pixel counts
        "MM_PROCESSOR_KWARGS": json.dumps(
            {"size": {"longest_edge": max_pixels, "shortest_edge": min(65536, max_pixels)}}),
        "HF_HUB_OFFLINE": "1" if s.get("offline") else "0",
        "VLLM_EXTRA_ARGS": s.get("extra_args") or "",
    }
    return "\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in env.items())


if __name__ == "__main__":
    if sys.argv[1:] == ["shell"]:
        print(shell_exports(load()))
    else:
        print(json.dumps(load(), indent=2, ensure_ascii=False))
