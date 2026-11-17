#!/usr/bin/env bash
# Test RTSP camera on the compose network: mediamtx (rtsp-sim) + an ffmpeg publisher looping an mp4.
# The backend reaches it at rtsp://rtsp-sim:8554/<name>  (enter that URL in the UI).
#   sh tools/rtsp_sim.sh videos/samples/urfd-fall-01-rgb.mp4 cam1    # start (repeat with other names)
#   sh tools/rtsp_sim.sh stop                                        # stop all
set -e
cd "$(dirname "$0")/.."
NET=realtime-vlm
if [ "$1" = "stop" ]; then
  docker ps -aq --filter name=^rtsp-pub- | xargs -r docker rm -f >/dev/null
  docker rm -f rtsp-sim >/dev/null 2>&1 || true
  echo "stopped"; exit 0
fi
VIDEO=${1:?usage: rtsp_sim.sh <video path under videos/> [name] | stop}
NAME=${2:-cam1}
docker ps -q --filter name=^rtsp-sim$ | grep -q . || \
  docker run -d --rm --name rtsp-sim --network $NET bluenviron/mediamtx:latest >/dev/null
sleep 1
docker rm -f rtsp-pub-$NAME >/dev/null 2>&1 || true
docker run -d --rm --name rtsp-pub-$NAME --network $NET -v "$(pwd)/videos:/workspace/videos:ro" -w /workspace \
  realtime-vlm-backend:latest ffmpeg -nostdin -loglevel error -re -stream_loop -1 -i "$VIDEO" \
  -c:v libx264 -preset veryfast -tune zerolatency -g 30 -an -f rtsp -rtsp_transport tcp "rtsp://rtsp-sim:8554/$NAME" >/dev/null
echo "rtsp://rtsp-sim:8554/$NAME  <- $VIDEO"
