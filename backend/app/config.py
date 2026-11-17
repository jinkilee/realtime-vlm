"""Config loader: configs/config.yaml + a few env overrides.

Project root (ROOT) is the directory holding configs/, videos/, data/ — /workspace in the container.

CLI:
    python -m app.config shell     # `export KEY=VALUE` lines for backend/start.sh
    python -m app.config           # dump the effective config
"""
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RVLM_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_PATH = Path(os.environ.get("RVLM_CONFIG", ROOT / "configs" / "config.yaml"))

ENV_OVERRIDES = {
    "SERVING": ("serving", "backend"),
    "MODEL_ID": ("serving", "model_id"),
    "SERVING_HOST": ("serving", "host"),
}


def load(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for env, (section, key) in ENV_OVERRIDES.items():
        if os.environ.get(env):
            cfg[section][key] = os.environ[env]
    if cfg["inference"]["window_sec"] > cfg["stream"]["buffer_sec"]:
        raise ValueError("inference.window_sec must be <= stream.buffer_sec")
    if cfg["inference"]["num_frames"] > cfg["serving"]["max_images_per_request"]:
        raise ValueError("inference.num_frames must be <= serving.max_images_per_request")
    return cfg


def serving_url(cfg: dict) -> str:
    return f"http://{cfg['serving'].get('host', '127.0.0.1')}:{cfg['serving']['port']}"


def shell_exports(cfg: dict) -> str:
    b = cfg["backend"]
    env = {"BACKEND_PORT": b["port"], "BACKEND_RELOAD": "1" if b.get("reload") else "0"}
    return "\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in env.items())


if __name__ == "__main__":
    if sys.argv[1:] == ["shell"]:
        print(shell_exports(load()))
    else:
        print(json.dumps(load(), indent=2, ensure_ascii=False))
