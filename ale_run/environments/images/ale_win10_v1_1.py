from dataclasses import replace

from .ale_win10 import IMAGE as BASE_IMAGE


IMAGE = replace(BASE_IMAGE, name="ale-win10-v1-1")
