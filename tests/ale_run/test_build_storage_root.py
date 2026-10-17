import os
from pathlib import Path
import subprocess
import sys


def test_builder_accepts_configurable_persistent_root(tmp_path):
    script = Path(__file__).resolve().parents[2] / "ale_run/environments/images/ale_ubuntu22_docker"
    environment = dict(os.environ, ALE_IMAGE_BUILD_ROOT=str(tmp_path), PYTHONPATH=str(script))
    output = subprocess.check_output(
        [sys.executable, "-c", "import local_build; print(local_build.PERSISTENT_ROOT)"],
        env=environment,
        text=True,
    )
    assert output.strip() == str(tmp_path.resolve())


def test_builder_default_is_user_relative():
    script = Path(__file__).resolve().parents[2] / "ale_run/environments/images/ale_ubuntu22_docker"
    environment = dict(os.environ, PYTHONPATH=str(script))
    environment.pop("ALE_IMAGE_BUILD_ROOT", None)
    output = subprocess.check_output(
        [sys.executable, "-c", "import local_build; print(local_build.PERSISTENT_ROOT)"],
        env=environment,
        text=True,
    )
    assert output.strip() == str((Path.home() / "ale-overall").resolve())
