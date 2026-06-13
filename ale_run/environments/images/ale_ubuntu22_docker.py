"""``ale-ubuntu22-docker`` — the ale-ubuntu22 sandbox as a local Docker image.

Built by exporting the ``ale-ubuntu22`` GCE VM's rootfs into a container image,
so the userspace is byte-for-byte identical: every VM-side absolute path and the
cua-server port (5000) match ``ale-ubuntu22``. It is a separate image (separate
container ref, run on the docker provider) — the fields below mirror
``ale_ubuntu22.py`` because the contents are the same.
"""
from __future__ import annotations

from . import Image


IMAGE = Image(
    name="ale-ubuntu22-docker",
    os="linux",

    # sandbox-side paths (identical to ale-ubuntu22 — same exported rootfs)
    work_dir_base="/home/user/.ale",
    task_data_root="/media/user/data/agenthle",
    node="/usr/local/bin/node",
    python="/opt/ale-run/.venv/bin/python",
    mcp_server_dir="/home/user/cua_mcp_server",

    # provisioning defaults
    default_machine_type="e2-standard-4",

    # Container ref the docker provider boots. Hub-qualified so a user WITHOUT a
    # local copy auto-pulls it via `docker run` (an unqualified name would resolve
    # to docker.io/library/... and fail). Locally-built images should be tagged
    # with this same ref so the local copy is used instead of pulling.
    docker_image="agentslastexam/ale-ubuntu22-docker:latest",

    # cua-server port (same as the GCE image)
    cua_server_port=5000,
)
