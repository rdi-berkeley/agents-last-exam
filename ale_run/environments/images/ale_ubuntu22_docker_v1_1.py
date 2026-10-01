from dataclasses import replace

from .ale_ubuntu22_docker import IMAGE as BASE_IMAGE


IMAGE = replace(
    BASE_IMAGE,
    name="ale-ubuntu22-docker-v1-1",
    docker_image=(
        "agentslastexam/ale-ubuntu22-docker:v1.1@"
        "sha256:0b4d5173e0be38234d97a3b193a70ebbcd766a0983476d4d7c86539d3cb033be"
    ),
)
