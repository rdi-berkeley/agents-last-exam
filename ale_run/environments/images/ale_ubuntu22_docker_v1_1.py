from dataclasses import replace

from .ale_ubuntu22_docker import IMAGE as BASE_IMAGE


IMAGE = replace(
    BASE_IMAGE,
    name="ale-ubuntu22-docker-v1-1",
    docker_image="agentslastexam/ale-ubuntu22-docker:v1.1",
)
