from dataclasses import replace

from .ale_ubuntu22 import IMAGE as BASE_IMAGE


IMAGE = replace(BASE_IMAGE, name="ale-ubuntu22-v1-1")
