"""``ale-kasm`` — Docker-based Ubuntu 22.04 via trycua/cua-ubuntu.

Lightweight alternative to the GCE-backed ``ale-ubuntu22`` for
cpu-free-ubuntu tasks.  Runs cua-server on port 8000 (vs 5000 on GCE).
Base: trycua/cua-ubuntu (Kasm XFCE desktop + KasmVNC + cua-server).
System Python 3.12 at /usr/bin/python3 — no dedicated venv needed since
all evaluate-side packages are pre-installed in the image.

User: kasm-user (uid 1000).
"""
from __future__ import annotations

from . import Image


IMAGE = Image(
    name="ale-kasm",
    os="linux",

    # sandbox-side paths
    work_dir_base="/home/kasm-user/.ale",
    task_data_root="/media/user/data/agenthle",
    node="/usr/bin/node",
    python="/usr/bin/python3",
    mcp_server_dir="/home/kasm-user/cua_mcp_server",

    # no GCE machine type — Docker containers are sized by the host
    default_machine_type="",

    # published container image the docker provider boots. :latest now has
    # the task eval packages baked into /usr/bin/python3 (scipy/sklearn/skimage/
    # opencv/h5py/rasterio/netCDF4/pydicom/pymedphys/torch/...).
    docker_image="agentslastexam/ale-kasm:latest",

    # cua-computer-server's package default port on this image (vs 5000 on GCE)
    cua_server_port=8000,
)
