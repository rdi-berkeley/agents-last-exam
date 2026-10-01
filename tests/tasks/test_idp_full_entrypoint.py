import asyncio
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import traceback
from types import SimpleNamespace
import uuid

import pytest

from tasks.life_sciences.idp_ensemble_scoring import main as task


SCENARIOS = [
    "reference",
    "equivalent",
    "missing",
    "wrong_cell",
    "extra_column",
    "invalid_reference",
]
REFERENCE_SHA256 = "8c4133b7bab4d50781642485d88cc1f754df5466084c15c4870363e6759b46f3"


@pytest.fixture
def evidence():
    value = os.environ.get("IDP_SCIENTIFIC_EVIDENCE")
    if not value:
        pytest.skip("Set IDP_SCIENTIFIC_EVIDENCE for authentic full-entrypoint replay")
    return Path(value)


def scenario_data(evidence, scenario):
    reference = (evidence / "guest/full-pool-v1/Final_Output.csv").read_bytes()
    assert hashlib.sha256(reference).hexdigest() == REFERENCE_SHA256
    candidate = reference
    if scenario in {"equivalent", "wrong_cell", "extra_column"}:
        rows = list(csv.DictReader(io.StringIO(reference.decode())))
        columns = list(rows[0])
        if scenario == "equivalent":
            columns.reverse()
            rows.reverse()
            for row in rows:
                for column in columns:
                    if column != "Method":
                        row[column] = format(float(row[column]), ".17e")
        elif scenario == "wrong_cell":
            rows[0]["CS"] = "0.9"
        else:
            columns.append("Unexpected")
            for row in rows:
                row["Unexpected"] = "0.5"
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
        candidate = ("\ufeff" + stream.getvalue()).encode()
    elif scenario in {"missing", "invalid_reference"}:
        candidate = None
        if scenario == "invalid_reference":
            reference = b"Method,Total\nModel1,NaN\n"
    return reference, candidate


class LocalSession:
    def __init__(self, root):
        self.root = root
        self.interface = self
        self.executions = []

    async def file_exists(self, path):
        return Path(path).is_file()

    async def directory_exists(self, path):
        return Path(path).is_dir()

    async def create_dir(self, path):
        assert Path(path).is_relative_to(self.root)
        Path(path).mkdir(parents=True, exist_ok=True)

    async def write_file(self, path, content):
        assert Path(path).is_relative_to(self.root)
        Path(path).write_text(content)

    async def run_command(self, value, timeout=None, check=False):
        result = subprocess.run(
            shlex.split(value), capture_output=True, text=True, timeout=timeout, check=check
        )
        self.executions.append(
            dict(
                command=value,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        return self.executions[-1]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_full_local_entrypoint_executes_uploaded_verifier(
    tmp_path, monkeypatch, evidence, scenario
):
    reference, candidate = scenario_data(evidence, scenario)
    reference_file = tmp_path / "reference.csv"
    output_file = tmp_path / "Final_Output.csv"
    reference_file.write_bytes(reference)
    if candidate is not None:
        output_file.write_bytes(candidate)
    monkeypatch.setattr(task, "EVAL_TMP_DIR", str(tmp_path / "eval"))
    config = SimpleNamespace(
        metadata=dict(
            reference_dir=str(tmp_path),
            reference_file=str(reference_file),
            output_file=str(output_file),
        )
    )
    session = LocalSession(tmp_path)
    if scenario == "invalid_reference":
        with pytest.raises(RuntimeError, match="IDP evaluator failed"):
            asyncio.run(task.evaluate(config, session))
    else:
        expected = (
            1.0
            if scenario in {"reference", "equivalent"}
            else 0.95
            if scenario == "wrong_cell"
            else 0.0
        )
        assert asyncio.run(task.evaluate(config, session)) == [expected]
    assert len(session.executions) == 1
    assert (tmp_path / "eval/verify_output.py").read_bytes() == (
        Path(task.__file__).parent / "scripts/verify_output.py"
    ).read_bytes()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_full_retained_guest_entrypoint(tmp_path, monkeypatch, evidence, scenario):
    port = os.environ.get("IDP_ENTRYPOINT_GUEST_PORT")
    if not port:
        pytest.skip("Set IDP_ENTRYPOINT_GUEST_PORT for the authorized retained guest")
    assert port == "37874"
    helper = Path(
        "/home/allennie/ale-overall/task-rerun-gpt56-sol-20260903/repair-validation-20260905/guest_api.py"
    )
    spec = importlib.util.spec_from_file_location("idp_entrypoint_guest_api", helper)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    root = "/home/user/.ale-audit/idp-scientific-20260910/entrypoint-replays/" + tmp_path.name
    response = api.command(int(port), "mkdir -p " + shlex.quote(root))
    assert response["return_code"] == 0
    reference, candidate = scenario_data(evidence, scenario)
    for name, content in [("reference.csv", reference), ("Final_Output.csv", candidate)]:
        if content is not None:
            local = tmp_path / name
            local.write_bytes(content)
            api.upload(int(port), local, root + "/" + name)

    class GuestSession:
        def __init__(self):
            self.interface = self
            self.executions = []

        async def file_exists(self, path):
            return api.command(int(port), "test -f " + shlex.quote(path))["return_code"] == 0

        async def directory_exists(self, path):
            return api.command(int(port), "test -d " + shlex.quote(path))["return_code"] == 0

        async def create_dir(self, path):
            assert path.startswith(root + "/")
            result = api.command(int(port), "mkdir -p " + shlex.quote(path))
            assert result["return_code"] == 0

        async def write_file(self, path, content):
            assert path.startswith(root + "/")
            local = tmp_path / "uploaded-verifier.py"
            local.write_text(content)
            api.upload(int(port), local, path)

        async def run_command(self, value, timeout=None, check=False):
            result = api.command(int(port), value, timeout=(timeout or 60) + 10)
            self.executions.append(dict(command=value, **result))
            return result

    monkeypatch.setattr(task, "EVAL_TMP_DIR", root + "/eval")
    config = SimpleNamespace(
        metadata=dict(
            reference_dir=root,
            reference_file=root + "/reference.csv",
            output_file=root + "/Final_Output.csv",
        )
    )
    session = GuestSession()
    if scenario == "invalid_reference":
        with pytest.raises(RuntimeError, match="IDP evaluator failed"):
            asyncio.run(task.evaluate(config, session))
    else:
        expected = (
            1.0
            if scenario in {"reference", "equivalent"}
            else 0.95
            if scenario == "wrong_cell"
            else 0.0
        )
        assert asyncio.run(task.evaluate(config, session)) == [expected]
    assert len(session.executions) == 1
    (evidence / ("entrypoint-guest-" + scenario + ".json")).write_text(
        json.dumps(session.executions, indent=2) + "\n"
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_production_desktop_session_full_entrypoint(monkeypatch, evidence, scenario):
    from cua_bench.computers.remote import RemoteDesktopSession

    port = os.environ.get("IDP_ENTRYPOINT_GUEST_PORT")
    if not port:
        pytest.skip("Set IDP_ENTRYPOINT_GUEST_PORT for the authorized retained guest")
    assert port == "37874"
    root = "/home/user/.ale-audit/idp-scientific-20260910/production-entrypoint/" + uuid.uuid4().hex
    monkeypatch.setattr(task, "EVAL_TMP_DIR", root + "/eval")
    reference, candidate = scenario_data(evidence, scenario)
    config = SimpleNamespace(
        metadata=dict(
            reference_dir=root,
            reference_file=root + "/reference.csv",
            output_file=root + "/Final_Output.csv",
        )
    )
    source = Path(inspect.getfile(RemoteDesktopSession))
    receipt = dict(
        scenario=scenario,
        metadata=config.metadata,
        session_class="cua_bench.computers.remote.RemoteDesktopSession",
        session_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        run_command_signature=str(inspect.signature(RemoteDesktopSession.run_command)),
        main_sha256=hashlib.sha256(Path(task.__file__).read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(reference).hexdigest(),
        method_replacement=False,
        transport_adapter=False,
    )

    async def run():
        session = RemoteDesktopSession(api_url="http://127.0.0.1:" + port, os_type="linux")
        try:
            await session.start()
            assert session._client_only_mode
            await session.interface.create_dir(root)
            await session.write_bytes(config.metadata["reference_file"], reference)
            if candidate is not None:
                await session.write_bytes(config.metadata["output_file"], candidate)
            try:
                receipt["score"] = await task.evaluate(config, session)
            except Exception as error:
                receipt["exception"] = type(error).__name__
                receipt["message"] = str(error)
                receipt["traceback"] = traceback.format_exc()
                if scenario != "invalid_reference" or not isinstance(error, RuntimeError):
                    raise
            else:
                assert scenario != "invalid_reference"
                expected = (
                    1.0
                    if scenario in {"reference", "equivalent"}
                    else 0.95
                    if scenario == "wrong_cell"
                    else 0.0
                )
                assert receipt["score"] == [expected]
            assert await session.read_bytes(config.metadata["reference_file"]) == reference
            if candidate is not None:
                assert await session.read_bytes(config.metadata["output_file"]) == candidate
            receipt["passed"] = True
        finally:
            await session.close()
            (evidence / ("entrypoint-production-" + scenario + ".json")).write_text(
                json.dumps(receipt, indent=2) + "\n"
            )

    asyncio.run(run())
