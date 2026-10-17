from dataclasses import replace

from .ale_ubuntu22_docker import IMAGE as BASE_IMAGE


IMAGE = replace(
    BASE_IMAGE,
    name="ale-ubuntu22-docker-v1-1",
    docker_image=(
        "agentslastexam/ale-ubuntu22-docker:v1.1@"
        "sha256:0b223c5042bedfa78f37f17881f6b410aa3d745310b667e5dff62637363c5085"
    ),
)
