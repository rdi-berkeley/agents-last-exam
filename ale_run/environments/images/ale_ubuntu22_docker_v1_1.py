from dataclasses import replace

from .ale_ubuntu22_docker import IMAGE as BASE_IMAGE


IMAGE = replace(
    BASE_IMAGE,
    name="ale-ubuntu22-docker-v1-1",
    docker_image=(
        "agentslastexam/ale-ubuntu22-docker:v1.1@"
        "sha256:b0804013b6490eec77d735e3faf256b8d2ac06952061d524b66848f756c64f34"
    ),
)
