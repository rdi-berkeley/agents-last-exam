# ALE Kasm Desktop Image
#
# Kasm XFCE desktop + Python 3.12 + CUA Computer Server + Playwright.
# Agents (claude, droid, cursor-agent, openclaw, etc.) are NOT baked in;
# each agent's deployer.install() handles its own binary at runtime.
#
# Build:
#   docker build -t agentslastexam/ale-kasm:latest docker/kasm/
#
# Build for x86 from ARM host:
#   docker buildx build --platform linux/amd64 -t agentslastexam/ale-kasm:latest docker/kasm/
#
# Base image: Kasm core Ubuntu 22.04 (Jammy) with XFCE, KasmVNC, PulseAudio.
# Ships Node 22 + npm 10 in the base layer (kasm-recorder user).

FROM kasmweb/core-ubuntu-jammy:1.17.0
USER root

ENV HOME=/home/kasm-default-profile
ENV STARTUPDIR=/dockerstartup
ENV INST_SCRIPTS=$STARTUPDIR/install
WORKDIR $HOME

# ============================================================
# 1. System packages: Python 3.12, build tools, desktop utils
# ============================================================
RUN apt-get update && \
    apt install -y wget build-essential libncursesw5-dev libssl-dev \
    libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.12 python3.12-dev python3.12-tk \
    gnome-screenshot wmctrl ffmpeg socat xclip curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ============================================================
# 2. Firefox (via Kasm's own install script)
# ============================================================
ENV DEBIAN_FRONTEND=noninteractive \
    INST_DIR=$STARTUPDIR/install
COPY src/ $INST_DIR
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
RUN mkdir -p $HOME && chown -R 1000:0 $HOME

# ============================================================
# 7. Python tooling: uv + CUA packages + Playwright
# ============================================================
RUN python3.12 -m pip install uv

RUN uv pip install --system cua-computer-server "cua-agent[all]"

RUN uv pip install --system playwright && \
    python3.12 -m playwright install --with-deps firefox

# ============================================================
# 8. Cache cleanup
# ============================================================
RUN rm -rf /home/kasm-user/.cache/uv /root/.cache/uv
RUN chown -R 1000:0 /home/kasm-user/.cache 2>/dev/null || true

# ============================================================
# 9. Startup script: upgrade CUA packages + start computer_server
# ============================================================
RUN echo -e "#!/bin/bash\nsudo uv pip install --upgrade --system cua-computer-server \"cua-agent[all]\"\n/usr/bin/python3 -m computer_server" > $STARTUPDIR/custom_startup.sh \
    && chmod +x $STARTUPDIR/custom_startup.sh

USER 1000
