# ALE Kasm Desktop Image
#
# Kasm XFCE desktop + Python 3.12 + CUA Computer Server + Playwright.
# Agents (claude, droid, cursor-agent, openclaw, etc.) are NOT baked in;
# each agent's deployer.install() handles its own binary at runtime.
#
# Build (from repo root — context is the repo root so the cua MCP bridge
# source at ale_run/agents/_assets/cua_mcp_server/ can be prebaked):
#   docker build -f ale_run/environments/images/ale_kasm/Dockerfile -t agentslastexam/ale-kasm:latest .
#
# Build for x86 from ARM host:
#   docker buildx build --platform linux/amd64 -f ale_run/environments/images/ale_kasm/Dockerfile -t agentslastexam/ale-kasm:latest .
#
# Base image: Kasm core Ubuntu 22.04 (Jammy) with XFCE, KasmVNC, PulseAudio.

FROM kasmweb/core-ubuntu-jammy:1.17.0
USER root

ENV HOME=/home/kasm-default-profile
ENV STARTUPDIR=/dockerstartup
ENV INST_SCRIPTS=$STARTUPDIR/install
WORKDIR $HOME

# ============================================================
# 1. System packages: Python 3.12, build tools, desktop utils, common CLI tools
# ============================================================
RUN apt-get update && \
    apt install -y wget build-essential libncursesw5-dev libssl-dev \
    libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.12 python3.12-dev python3.12-tk \
    gnome-screenshot wmctrl ffmpeg socat xclip curl \
    unzip zip p7zip-full jq && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ============================================================
# 1b. Node.js 22 LTS + npm
# ============================================================
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ============================================================
# 2. Firefox (via Kasm's own install script)
# ============================================================
ENV DEBIAN_FRONTEND=noninteractive \
    INST_DIR=$STARTUPDIR/install
COPY ale_run/environments/images/ale_kasm/src/ $INST_DIR
RUN bash ${INST_DIR}/ubuntu/install/firefox/install_firefox.sh

# ============================================================
# 3. Set Python 3.12 as default + install pip
# ============================================================
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# ============================================================
# 4. KasmVNC: disable SSL (container runs behind docker network)
# ============================================================
RUN sed -i 's/require_ssl: true/require_ssl: false/g' /usr/share/kasmvnc/kasmvnc_defaults.yaml
RUN sed -i 's/-sslOnly//g' /dockerstartup/vnc_startup.sh

# ============================================================
# 5. kasm-user: password + sudo
# ============================================================
RUN echo "kasm-user:password" | chpasswd
RUN usermod -aG sudo kasm-user
RUN echo "kasm-user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ============================================================
# 6. Home directory permissions
# ============================================================
RUN chown 1000:0 $HOME
RUN $STARTUPDIR/set_user_permission.sh $HOME
ENV HOME=/home/kasm-user
WORKDIR $HOME
RUN mkdir -p $HOME $HOME/.config/gcloud && chown -R 1000:0 $HOME

# ============================================================
# 6b. Task data root (owned by kasm-user for GCS staging)
# ============================================================
RUN mkdir -p /media/user/data/ale-data && chown -R 1000:0 /media/user

# ============================================================
# 6c. Prebake the cua MCP bridge into mcp_server_dir (speed only).
#
# This is purely an optimization so MCP-capable deployers skip the
# runtime copy + `npm install` (see _bootstrap.ensure_cua_mcp_server,
# which fast-skips when src/index.js + node_modules/@modelcontextprotocol/sdk
# are already present). The dynamic install remains the correctness
# guarantee on thinner images; this layer just makes the common case fast.
# Path matches ale_kasm.Image.mcp_server_dir.
# ============================================================
COPY ale_run/agents/_assets/cua_mcp_server/ /home/kasm-user/cua_mcp_server/
RUN cd /home/kasm-user/cua_mcp_server && npm install --production && \
    chown -R 1000:0 /home/kasm-user/cua_mcp_server

# ============================================================
# 7. Python tooling: uv + CUA packages + Playwright
# ============================================================
RUN python3.12 -m pip install uv

RUN uv pip install --system cua-computer-server "cua-agent[all]" pyyaml

RUN uv pip install --system playwright && \
    python3.12 -m playwright install --with-deps firefox

# ============================================================
# 7b. Task eval-side Python deps (system /usr/bin/python3)
# ------------------------------------------------------------
# The evaluate()/score scripts of the baked-in tasks import these at eval time
# and run under /usr/bin/python3 (no venv). The set is the union of
# tasks/pyproject.toml and a static AST scan of every cpu-free-ubuntu task's
# imports (which surfaced pydicom + pymedphys that pyproject had missed).
# torch is already present in the base image, so it is satisfied here.
# ============================================================
COPY tasks/pyproject.toml /tmp/ale_task_deps/pyproject.toml
RUN uv pip install --system \
      $(python3.12 -c "import tomllib,sys; print(' '.join(tomllib.load(open('/tmp/ale_task_deps/pyproject.toml','rb'))['project']['dependencies']))")

# ============================================================
# 8. Google Cloud SDK (gcloud + gsutil, for gs:// task data staging)
# ============================================================
RUN curl -fsSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz \
      | tar -xz -C /opt && \
    /opt/google-cloud-sdk/install.sh --quiet --path-update=false && \
    ln -sf /opt/google-cloud-sdk/bin/gcloud  /usr/local/bin/gcloud && \
    ln -sf /opt/google-cloud-sdk/bin/gsutil  /usr/local/bin/gsutil && \
    mkdir -p /home/kasm-user/.config/gcloud && \
    chown -R 1000:0 /home/kasm-user/.config/gcloud

# ============================================================
# 9. Cache cleanup
# ============================================================
RUN rm -rf /home/kasm-user/.cache/uv /root/.cache/uv
RUN chown -R 1000:0 /home/kasm-user/.cache 2>/dev/null || true

# ============================================================
# 9. Startup script: upgrade CUA packages + start computer_server
# ============================================================
RUN echo -e "#!/bin/bash\nsudo uv pip install --upgrade --system cua-computer-server \"cua-agent[all]\"\n/usr/bin/python3 -m computer_server" > $STARTUPDIR/custom_startup.sh \
    && chmod +x $STARTUPDIR/custom_startup.sh

USER 1000
