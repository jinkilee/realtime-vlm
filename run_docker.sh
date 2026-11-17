mkdir -p videos data && HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d --renew-anon-volumes
