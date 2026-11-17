docker run --rm -u $(id -u):$(id -g) -e npm_config_cache=/tmp/.npm -v $(cd $(dirname $0) && pwd):/app -w /app node:22-slim sh -c "npm install --no-audit --no-fund && npm run build"
