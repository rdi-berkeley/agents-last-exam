import asyncio
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile
import time
import uuid

from tasks.utils.evaluation import JudgeInfrastructureError


ROOT = Path(__file__).parent
RUNTIME_WRAPPER = "/usr/local/bin/ale-migration-env"


async def checked_command(session, command):
    script = (
        "import json, subprocess, sys\n"
        "try:\n"
        "    result = subprocess.run([sys.argv[1], 'sh', '-c', sys.argv[2]], "
        "capture_output=True, text=True, errors='replace', timeout=300)\n"
        "except OSError as error:\n"
        "    print(json.dumps({'exit_code': 127, 'stdout': '', "
        "'stderr': 'Required migration runtime wrapper unavailable: ' "
        "+ sys.argv[1] + ': ' + str(error)}))\n"
        "else:\n"
        "    print(json.dumps({'exit_code': result.returncode, "
        "'stdout': result.stdout, 'stderr': result.stderr}))\n"
    )
    try:
        transport = await asyncio.wait_for(
            session.run_command(
                shlex.join(["python3", "-c", script, RUNTIME_WRAPPER, command]), check=False
            ),
            timeout=320,
        )
        result = json.loads(transport.get("stdout", ""))
        if type(result.get("exit_code")) is not int or not isinstance(result.get("stdout"), str):
            raise ValueError("Missing command exit receipt")
    except (TimeoutError, ValueError, AttributeError) as error:
        raise JudgeInfrastructureError("Migration command lacks an execution receipt") from error
    if result["exit_code"]:
        raise JudgeInfrastructureError(
            f"Migration command failed with exit {result['exit_code']}: {result.get('stderr', '')}"
        )
    return result


async def run_evaluator(session, command, evidence, *, timeout=7200, poll_interval=5):
    status = shlex.quote(evidence + "/worker.exit")
    log = shlex.quote(evidence + "/worker.log")
    worker = f"({command})" + f'; result=$?; printf "%s\\n" "$result" > {status}'
    launch = (
        f"mkdir -p {shlex.quote(evidence)} && "
        f"if mkdir {shlex.quote(evidence + '/worker-lock')} 2>/dev/null; then "
        f"nohup sh -c {shlex.quote(worker)} > {log} 2>&1 < /dev/null & fi"
    )
    await checked_command(session, launch)
    deadline = time.monotonic() + timeout + 60
    while time.monotonic() < deadline:
        result = await checked_command(session, f"if test -f {status}; then cat {status}; fi")
        value = result.get("stdout", "").strip()
        if value:
            try:
                code = int(value)
            except ValueError as exc:
                raise JudgeInfrastructureError("Invalid native evaluator status") from exc
            return {"return_code": code}
        await asyncio.sleep(poll_interval)
    raise JudgeInfrastructureError("Native migration evaluator did not finish within its budget")


async def stage(meta, session, *, evaluating=False):
    registry = json.loads((ROOT / "assets/sources.json").read_text())
    entry = registry[meta["variant_name"]]
    archive = ROOT / "assets" / entry["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != entry["sha256"]:
        raise JudgeInfrastructureError("Prepared source archive hash mismatch")
    destination = (
        f"{meta['task_dir']}/.migration-eval-{uuid.uuid4().hex}"
        if evaluating
        else meta["input_dir"]
    )
    await checked_command(
        session,
        shlex.join(["mkdir", "-p", destination, meta["software_dir"], meta["remote_output_dir"]]),
    )
    archive_path = f"{destination}/source.tar.gz"
    await session.write_bytes(archive_path, archive.read_bytes())
    await checked_command(session, shlex.join(["tar", "-xzf", archive_path, "-C", destination]))
    software = destination + "/tools" if evaluating else meta["software_dir"]
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as packed:
        for path in sorted((ROOT / "scripts").iterdir()):
            if path.is_file() and path.name != "score_audio_remote.py":
                packed.add(path, arcname=path.name)
        packed.add(ROOT / "rubric.txt", arcname="rubric.txt")
    await session.write_bytes(f"{destination}/tools.tar.gz", payload.getvalue())
    commands = [
        shlex.join(["mkdir", "-p", software]),
        shlex.join(["tar", "-xzf", f"{destination}/tools.tar.gz", "-C", software]),
        shlex.join(["python3", f"{software}/rebind.py", f"{destination}/project/project.ardour"]),
        shlex.join(["rm", f"{destination}/source.tar.gz", f"{destination}/tools.tar.gz"]),
    ]
    for command in commands:
        await checked_command(session, command)
    return destination, software
