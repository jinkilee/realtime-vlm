#!/usr/bin/env bash
# Test RTSP camera: serves an mp4 (looped) at rtsp://127.0.0.1:8554/<name> INSIDE the realtime-vlm
# container's network namespace (mediamtx shares it), so the app can use that URL directly.
#   sh tools/rtsp_sim.sh videos/samples/urfd-fall-01-rgb.mp4 cam1     # start
#   sh tools/rtsp_sim.sh stop                                         # stop all
set -e
if [ "$1" = "stop" ]; then
  docker exec realtime-vlm pkill -f "ffmpeg.*rtsp://127.0.0.1:8554" || true
  docker rm -f rtsp-sim >/dev/null 2>&1 || true
  echo "stopped"; exit 0
fi
VIDEO=${1:?usage: rtsp_sim.sh <video path inside project> [name] | stop}
NAME=${2:-cam1}
docker ps -q --filter name=^rtsp-sim$ | grep -q . || docker run -d --rm --name rtsp-sim --network container:realtime-vlm bluenviron/mediamtx:latest >/dev/null
sleep 1
docker exec -d -w /workspace realtime-vlm ffmpeg -nostdin -loglevel error -re -stream_loop -1 -i "$VIDEO" -c:v libx264 -preset veryfast -tune zerolatency -g 30 -an -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:8554/$NAME"
echo "rtsp://127.0.0.1:8554/$NAME  <- $VIDEO"
