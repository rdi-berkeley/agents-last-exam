import base64
import io
from pathlib import Path
import shlex
import tarfile
import json

from .native_commands import NativeRoadEnvironmentError, run_native_command


ROOT = Path(__file__).parent
SCRIPT_NAMES = (
    "native_adapter.py", "terrain_sampling.py", "road_compat.py", "entry.FCMacro",
    "open_road.FCMacro", "export_metrics.FCMacro", "refresh_profile.FCMacro", "prepare_runtime.py",
)


async def stage_native_task(meta, session):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted((ROOT / "assets").iterdir()):
            if path.is_file():
                archive.add(path, arcname=f"input/{path.name}")
        for name in SCRIPT_NAMES:
            archive.add(ROOT / "scripts" / name, arcname=f"software/{name}")
    task_dir = meta["task_dir"]
    packed = f"{task_dir}/.native-setup.tar.gz"
    await run_native_command(session, shlex.join(["mkdir", "-p", task_dir, meta["remote_output_dir"]]), seconds=30)
    payload = buffer.getvalue()
    for offset in range(0, len(payload), 32768):
        encoded = base64.b64encode(payload[offset:offset + 32768]).decode()
        command = "import base64; " + f"open({packed!r}, {'wb' if offset == 0 else 'ab'!r}).write(base64.b64decode({encoded!r}))"
        await run_native_command(session, shlex.join(["python3", "-c", command]), seconds=30)
    await run_native_command(session, shlex.join(["tar", "-xzf", packed, "-C", task_dir]), seconds=30)
    installed = await run_native_command(session, shlex.join(["python3", f"{meta['software_dir']}/prepare_runtime.py", meta["software_dir"]]), seconds=30)
    try:
        receipt = json.loads(installed["stdout"])
        if not receipt.get("runtime") or not receipt.get("launcher") or receipt.get("source_files_modified") is not False:
            raise ValueError("Incomplete runtime preparation receipt")
    except (ValueError, TypeError, AttributeError) as error:
        raise NativeRoadEnvironmentError("Native Road setup lacks a runtime preparation receipt") from error
