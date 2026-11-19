#!/usr/bin/env bash
# Load scenario: replace all streams with N video streams using one prompt, warm up, then collect
# over the same window:
#   - vLLM /metrics (server side: queue / prefill / decode, GPU util)    -> <label>.json
#   - WebSocket stats (client side)                                      -> <label>.ws.txt
#   - backend stage timing + per-thread CPU (GET /api/profile)           -> <label>.profile.json
#   - py-spy (PYSPY=1, default): all threads with native (C) frames      -> <label>.pyspy.json (speedscope)
#                                and GIL holders only                    -> <label>.gil.txt (collapsed stacks)
# Then prints a combined report (examples/profile_report.py).
# Run on the host from the project root. WARNING: removes every current stream.
#
#   sh backend/examples/metrics_scenario.sh <label> <n_streams> <prompt> [interval_ms] [sec]
#   sh backend/examples/metrics_scenario.sh 4s-fall 4 "넘어지는 사람" 1000 60
#   PYSPY=0 sh backend/examples/metrics_scenario.sh ...    # skip py-spy
#   MJPEG=1 ...                       # also pull every stream's MJPEG preview (like the UI does)
#   VIDEO_DIR=videos/samples-1080p    # same clips at 1080p (default videos/samples: 320x240)
#
# py-spy view: open <label>.pyspy.json at https://www.speedscope.app (file is parsed in the browser).
# py-spy needs ptrace: it runs via `docker exec --privileged -u root` (installed on first use).
set -eu
LABEL=$1; N=$2; PROMPT=$3; INTERVAL=${4:-1000}; SEC=${5:-60}; WARMUP=${WARMUP:-10}; PYSPY=${PYSPY:-1}
MJPEG=${MJPEG:-0}; VIDEO_DIR=${VIDEO_DIR:-videos/samples}
API=${API:-http://127.0.0.1:8001}
C=realtime-vlm-backend
OUT=data/metrics
VIDEOS="urfd-fall-01-rgb.mp4 urfd-fall-02-rgb.mp4 urfd-fall-05-rgb.mp4 urfd-adl-01-rgb.mp4"

for id in $(curl -sf "$API/api/streams" | python3 -c 'import json,sys;[print(s["id"]) for s in json.load(sys.stdin)]'); do
  curl -sf -X DELETE "$API/api/streams/$id" >/dev/null
done
i=0; IDS=""
for v in $VIDEOS; do
  [ "$i" -ge "$N" ] && break
  body=$(python3 -c 'import json,sys;print(json.dumps({"url":sys.argv[4]+"/"+sys.argv[1],"prompt":sys.argv[2],"interval_ms":int(sys.argv[3])}))' "$v" "$PROMPT" "$INTERVAL" "$VIDEO_DIR")
  IDS="$IDS $(curl -sf -X POST "$API/api/streams" -H 'content-type: application/json' -d "$body" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')"
  i=$((i + 1))
done
echo "[$LABEL] $N streams from $VIDEO_DIR, prompt=\"$PROMPT\", interval ${INTERVAL}ms, mjpeg=$MJPEG; warm-up ${WARMUP}s"
mkdir -p "$OUT"

if [ "$PYSPY" != 0 ]; then
  docker exec "$C" sh -c 'command -v py-spy >/dev/null' \
    || docker exec -u root "$C" pip install -q --root-user-action=ignore py-spy
  # uvicorn --reload: PID 1 is the reloader, the app is its multiprocessing child
  # (match on the parent pid: this shell's own cmdline would also contain "spawn_main")
  APP_PID=$(docker exec "$C" python3 -c 'import glob
for d in glob.glob("/proc/[0-9]*"):
    try:
        ppid = open(d + "/stat").read().rsplit(")", 1)[1].split()[1]
        cmd = open(d + "/cmdline").read()
    except OSError:
        continue
    if ppid == "1" and "spawn_main" in cmd:
        print(d[6:]); break')
  [ -n "$APP_PID" ] || APP_PID=1
fi
if [ "$MJPEG" != 0 ]; then  # preview viewers for warm-up + window (connections close by themselves)
  for id in $IDS; do curl -s -m $((WARMUP + SEC + 5)) "$API/api/streams/$id/mjpeg" -o /dev/null & done
fi
sleep "$WARMUP"

curl -sf -X POST "$API/api/profile?enabled=true&reset=true" >/dev/null
if [ "$PYSPY" != 0 ]; then
  # --native pauses the process ~50x/s to unwind C stacks (small overhead); --gil is non-blocking
  docker exec --privileged -u root "$C" py-spy record --pid "$APP_PID" -d "$SEC" -r 50 --threads --native \
    -f speedscope -o "/workspace/$OUT/$LABEL.pyspy.json" >"$OUT/$LABEL.pyspy.log" 2>&1 &
  docker exec --privileged -u root "$C" py-spy record --pid "$APP_PID" -d "$SEC" -r 100 --gil --nonblocking \
    -f raw -o "/workspace/$OUT/$LABEL.gil.txt" >"$OUT/$LABEL.gil.log" 2>&1 &
fi
docker exec "$C" python3 examples/ws_monitor.py --sec "$SEC" > "$OUT/$LABEL.ws.txt" &
python3 backend/examples/vllm_metrics.py --sec "$SEC" --label "$LABEL" --out "$OUT/$LABEL.json" > "$OUT/$LABEL.vllm.txt"
wait
curl -sf "$API/api/profile" > "$OUT/$LABEL.profile.json"
curl -sf -X POST "$API/api/profile?enabled=false&reset=false" >/dev/null
# py-spy ran as root: give the files to the host user
docker exec -u root "$C" sh -c "chown $(id -u):$(id -g) /workspace/$OUT/$LABEL.* 2>/dev/null || true"

cat "$OUT/$LABEL.vllm.txt"
echo "-- client side (WebSocket, includes backend preprocessing + HTTP)"
cat "$OUT/$LABEL.ws.txt"
python3 backend/examples/profile_report.py "$OUT/$LABEL"
