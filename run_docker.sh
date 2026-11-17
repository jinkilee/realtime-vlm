#mkdir -p videos data && docker run --rm -itd --name realtime-vlm --gpus all --ipc=host -p 127.0.0.1:8000:8000 -p 127.0.0.1:9000:9000 -v $(pwd):/workspace -v $HOME/.cache/huggingface:/root/.cache/huggingface -e SERVING -e MODEL_ID realtime-vlm:latest

docker run --rm -itd --name realtime-vlm --gpus all --ipc=host -p 127.0.0.1:8000:8000 -p 127.0.0.1:9000:9000 -v $(pwd):/workspace -e SERVING -e MODEL_ID realtime-vlm:latest
