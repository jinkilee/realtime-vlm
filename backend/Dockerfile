# backend: FastAPI app (stream decoding, inference client, events). No GPU needed.
# backend/, configs/, videos/, data/ are mounted at /workspace (see docker-compose.yml).
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/backend
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

# ---- dev tools / shell + vim settings (all users: backend/frontend run as host UID) -------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends vim net-tools \
 && rm -rf /var/lib/apt/lists/*
RUN echo "alias old='cd \${OLDPWD}'" >> /etc/bash.bashrc && \
    echo "alias fn='find ./ -name'" >> /etc/bash.bashrc && \
    echo "alias wd='grep -R'" >> /etc/bash.bashrc && \
    echo "alias vi='vim'" >> /etc/bash.bashrc && \
    printf '%s\n' \
      'set number' \
      'syntax on' \
      'set nocp' \
      'set ts=4' \
      'set shiftwidth=4' \
      'set expandtab' \
      'autocmd FileType yaml setlocal ts=2 sts=2 sw=2 expandtab' \
      >> /etc/vim/vimrc.local

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace/backend
EXPOSE 8000
CMD ["bash", "/workspace/backend/start.sh"]
