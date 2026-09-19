from pathlib import Path
import subprocess

import pytest


DATA = (
    Path(__file__).resolve().parents[2]
    / "task-data-hf/extracted/physical_sciences/mose2_bse_absorption_soc/base"
)


@pytest.fixture
def software():
    directory = DATA / "software"
    if not directory.exists():
        pytest.skip("MoSe2 release data is not installed")
    return directory


def test_installer_selects_task_local_spinor_converter(software):
    script = software / "install_software.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    content = script.read_text()
    assert '[pw2bgw.x]="$TASK_ROOT/software/libexec/pw2bgw-spinor.x"' in content
    assert '[pw.x]="$ENV_PREFIX/bin/pw.x"' in content
    assert '[sigma.cplx.x]="$BGW_ROOT/Sigma/sigma.cplx.x"' in content


def test_runtime_documentation_matches_converter_contract(software):
    readme = (software / "README.md").read_text()
    assert "serial executable" in readme
    assert "vxcg_flag = .true." in readme
    assert "vxc_flag = .false." in readme
    assert "dont_use_vxcdat" in readme
    assert "relative to the QE `outdir`" in readme


def test_task_prompt_lists_spinor_converter():
    from tasks.physical_sciences.mose2_bse_absorption_soc.main import CONFIG

    assert f"{CONFIG.software_bin_dir}/pw2bgw.x" in CONFIG.task_description
    assert "serial, spinor-capable" in CONFIG.task_description
