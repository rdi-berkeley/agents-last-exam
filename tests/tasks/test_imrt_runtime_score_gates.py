import json
from types import SimpleNamespace

import pytest

from tasks.health_medicine.prostate_imrt_matrad_reproduction import main


@pytest.mark.parametrize(
    "total,rubric,rc,missing,score_exists,expected",
    [
        (70, True, "0", False, True, 1),
        (69, True, "0", False, True, 0),
        (100, False, "0", False, True, 0),
        (100, True, "1", False, True, 0),
        (100, True, "0", True, True, 0),
        (100, True, "0", False, False, 0),
    ],
)
async def test_runtime_repair_keeps_clinical_and_completion_gates(
    monkeypatch, total, rubric, rc, missing, score_exists, expected
):
    async def runtime(*args, **kwargs):
        return {"ok": True, "prefix": "/mamba/envs/rtplan", "micromamba": "/mamba/bin/micromamba"}

    async def command(*args, **kwargs):
        return None

    class Session:
        interface = SimpleNamespace(create_dir=command)

        async def file_exists(self, path):
            if path.endswith(".dcm"):
                return not missing
            if path.endswith("score.json"):
                return score_exists
            return True

        async def directory_exists(self, path):
            return False

        async def write_file(self, *args):
            pass

        async def read_file(self, path):
            if path.endswith(".score_rc"):
                return rc
            if path.endswith("score.json"):
                return json.dumps({"total_score": total, "pass": rubric})
            return ""

    monkeypatch.setattr(main, "_ensure_python_runtime", runtime)
    monkeypatch.setattr(main, "_run_command", command)
    task = SimpleNamespace(
        metadata={
            "remote_output_dir": "/output",
            "reference_dir": "/reference",
            "eval_tmp_dir": "/eval",
            "matrad_wrapper": "/software/run_matrad.sh",
        }
    )
    assert await main.evaluate(task, Session()) == [expected]


async def test_runtime_uses_staged_wrapper_without_host_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://host-only.invalid:9999")
    monkeypatch.setenv("ALE_TASK_PIP_PROXY", "http://other-host.invalid:9999")
    commands, files = [], {}

    async def mkdir(path):
        pass

    class Session:
        interface = SimpleNamespace(create_dir=mkdir)

        async def write_file(self, path, text):
            files[path] = text

        async def read_file(self, path):
            return '{"ok":true,"installed":false}'

        async def run_command(self, command, *, check=True):
            commands.append(command)
            return {"return_code": 0, "stdout": "", "stderr": ""}

    cfg = SimpleNamespace(
        metadata={
            "eval_tmp_dir": "/task with space/eval",
            "reference_dir": "/task with space/reference",
            "matrad_wrapper": "/task with space/software/run_matrad.sh",
        }
    )
    result = await main._ensure_python_runtime(cfg, Session(), phase="evaluation")
    assert result["ok"]
    assert commands == [
        (
            "bash '/task with space/software/run_matrad.sh' python -I "
            "'/task with space/eval/ensure_runtime.py' --report "
            "'/task with space/eval/runtime_evaluation.json' --reference "
            "'/task with space/reference/RTDOSE_reference.dcm'"
        )
    ]
    assert "host-only" not in commands[0] and "other-host" not in commands[0]
    assert "--install" not in commands[0]
    assert files["/task with space/eval/ensure_runtime.py"] == main._read_script(
        "ensure_runtime.py"
    )


def test_default_preflight_never_installs_or_locks_environment(monkeypatch):
    from tasks.health_medicine.prostate_imrt_matrad_reproduction.scripts import ensure_runtime

    monkeypatch.setattr(
        ensure_runtime, "fresh_probe", lambda ref: {"ok": False, "error": "missing QA package"}
    )

    def unexpected_process(*args, **kwargs):
        pytest.fail("Default preflight must not install packages")

    monkeypatch.setattr(ensure_runtime.subprocess, "run", unexpected_process)
    with pytest.raises(RuntimeError, match="--install"):
        ensure_runtime.ensure_runtime()


def test_preflight_reports_ready_without_installing(monkeypatch):
    from tasks.health_medicine.prostate_imrt_matrad_reproduction.scripts import ensure_runtime

    monkeypatch.setattr(
        ensure_runtime, "fresh_probe", lambda ref: {"ok": True, "reference_checked": bool(ref)}
    )
    assert ensure_runtime.ensure_runtime("reference.dcm") == {
        "ok": True,
        "reference_checked": True,
        "installed": False,
    }


def test_bad_reference_never_triggers_dependency_reinstallation(monkeypatch, tmp_path):
    from tasks.health_medicine.prostate_imrt_matrad_reproduction.scripts import ensure_runtime

    monkeypatch.setattr(ensure_runtime.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        ensure_runtime,
        "fresh_probe",
        lambda ref: (
            {"ok": False, "error": "Reference dose contains non-finite values"}
            if ref
            else {"ok": True}
        ),
    )

    def unexpected_install(*args, **kwargs):
        pytest.fail("A broken reference cannot be repaired by package installation")

    monkeypatch.setattr(ensure_runtime.subprocess, "run", unexpected_install)
    with pytest.raises(RuntimeError, match="non-finite"):
        ensure_runtime.ensure_runtime("broken.dcm", install=True)


def test_explicit_provisioning_is_rechecked(monkeypatch, tmp_path):
    from tasks.health_medicine.prostate_imrt_matrad_reproduction.scripts import ensure_runtime

    monkeypatch.setattr(ensure_runtime.sys, "prefix", str(tmp_path))
    reports = iter([{"ok": False, "error": "missing package"}, {"ok": True}])
    monkeypatch.setattr(ensure_runtime, "fresh_probe", lambda ref: next(reports))
    commands = []
    monkeypatch.setattr(ensure_runtime.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    assert ensure_runtime.ensure_runtime(install=True) == {"ok": True, "installed": True}
    assert len(commands) == 2
    assert "ensurepip" in commands[0]
    assert "pip" in commands[1] and "pydicom==3.0.1" in commands[1]
