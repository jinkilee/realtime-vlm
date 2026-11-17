# frontend: Vite dev server with HMR. Source is mounted at /app; node_modules stays in the image
# (anonymous volume in docker-compose.yml), so only package.json changes need a rebuild.
FROM node:22-slim

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

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund && chmod -R a+rX node_modules
EXPOSE 5173
CMD ["npm", "run", "dev"]
