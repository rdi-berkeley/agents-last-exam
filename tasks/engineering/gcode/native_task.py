"""Stage public CAM helpers and evaluate saved projects through native software."""

import asyncio
import json
import math
import shlex
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).parent
PUBLIC_SCRIPTS = (
    "cam_adapter.py",
    "enable_adaptive.FCMacro",
    "machine_setup.py",
    "native_pipeline.py",
    "native_profiles.py",
    "native_project.py",
    "native_safety.py",
    "native_simulation.py",
    "native_stock.py",
    "nc_motion.py",
    "nc_replay.py",
    "open_cam.FCMacro",
    "open_cam.py",
    "preview_stock.FCMacro",
    "preview_stock.py",
    "tool_library.py",
)


class EvaluationUnavailableError(RuntimeError):
    """The installed environment could not establish a machining score."""


async def runtime_configuration(metadata, session):
    runtime = json.loads(await session.read_file(metadata["cam_runtime_config"]))
    return {
        **{
            name: runtime[name]
            for name in (
                "freecad",
                "interpreter_runtime",
                "collision_python",
                "stock_binary",
            )
        },
        "source_document": metadata["input_dir"] + "/blank.FCStd",
        "library_root": metadata["input_dir"] + "/tool-library",
    }


async def stage_native_task(metadata, session):
    software = metadata["software_dir"]
    await session.run_command(
        shlex.join(
            [
                "mkdir",
                "-p",
                software,
                metadata["remote_output_dir"],
            ]
        )
    )
    config = await runtime_configuration(metadata, session)
    for name in (*PUBLIC_SCRIPTS, "check_runtime.py"):
        await session.write_file(software + "/" + name, (ROOT / "scripts" / name).read_text())
    await session.write_file(software + "/preview.json", json.dumps(config))
    result = await session.run_command(
        shlex.join(
            [
                config["collision_python"],
                software + "/check_runtime.py",
                software + "/preview.json",
                metadata["input_dir"] + "/geometry/machining-region.json",
            ]
        ),
        check=False,
    )
    try:
        checked = json.loads(result.get("stdout", ""))
        ready = checked.get("runtime_ready") is True and checked.get("original_tools", 0) > 0
    except (ValueError, TypeError, AttributeError):
        ready = False
    if result.get("return_code", 1) or not ready:
        raise EvaluationUnavailableError(
            "CAM runtime or converted public input is incomplete: " + result.get("stderr", "")
        )


async def evaluate_native_task(metadata, session):
    work = metadata["task_dir"] + "/.evaluation/" + uuid.uuid4().hex
    scripts = work + "/scripts"
    await session.run_command(shlex.join(["mkdir", "-p", scripts]))
    config = await runtime_configuration(metadata, session)
    for name in (*PUBLIC_SCRIPTS, "native_geometry.py", "score_native.py"):
        await session.write_file(scripts + "/" + name, (ROOT / "scripts" / name).read_text())
    request = {
        "project": metadata["output_project"],
        "reference": metadata["reference_dir"] + "/reference_sim.stl",
        "region": metadata["input_dir"] + "/geometry/machining-region.json",
        "runtime": config,
    }
    await session.write_file(work + "/request.json", json.dumps(request))
    command = [config["collision_python"], scripts + "/score_native.py", work + "/request.json"]
    launch = (
        "import json,os,subprocess; "
        f"log=open({work + '/worker.log'!r},'w'); "
        f"process=subprocess.Popen({command!r},stdin=subprocess.DEVNULL,stdout=log,"
        "stderr=subprocess.STDOUT,start_new_session=True,"
        "env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}); "
        "print(json.dumps({'pid':process.pid}))"
    )
    launched = await session.run_command(shlex.join(["python3", "-c", launch]))
    pid = int(json.loads(launched["stdout"])["pid"])
    deadline = time.monotonic() + 6900
    try:
        while time.monotonic() < deadline:
            probe = (
                "import json,os; from pathlib import Path; "
                f"terminal=Path({work + '/terminal.json'!r}); "
                f"state=Path('/proc/{pid}/stat'); "
                "print(terminal.read_text() if terminal.is_file() else json.dumps("
                "{'status':'running' if state.is_file() and "
                "state.read_text().rsplit(')',1)[1].split()[0]!='Z' else 'worker_missing'}))"
            )
            observed = await session.run_command(shlex.join(["python3", "-c", probe]))
            result = json.loads(observed["stdout"])
            if result["status"] == "completed":
                score = float(result["score"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise EvaluationUnavailableError("Native evaluator returned an invalid score")
                return [score]
            if result["status"] != "running":
                raise EvaluationUnavailableError(
                    f"CAM evaluation unavailable; inspect {work}: {result.get('error', result['status'])}"
                )
            await asyncio.sleep(5)
        raise EvaluationUnavailableError(f"CAM evaluation exceeded its deadline: {work}")
    finally:
        cleanup = (
            "import os,signal\n"
            f"try: os.killpg({pid},signal.SIGTERM)\n"
            "except ProcessLookupError: pass\n"
        )
        await session.run_command(shlex.join(["python3", "-c", cleanup]), check=False)
