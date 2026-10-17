"""Bound task commands without extending the DesktopSession protocol."""

import asyncio
import base64
import json
import shlex
from pathlib import Path

from .verification import EvaluationUnavailableError


async def upload_files(session, root, files):
    for name, content in files.items():
        path = str(Path(root) / name)
        for offset in range(0, max(1, len(content)), 32768):
            payload = base64.b64encode(content[offset : offset + 32768]).decode()
            mode = "wb" if offset == 0 else "ab"
            script = (
                "import base64; from pathlib import Path; "
                f"path=Path({path!r}); path.parent.mkdir(parents=True,exist_ok=True); "
                f"path.open({mode!r}).write(base64.b64decode({payload!r}))"
            )
            result = await run_command(
                session, shlex.join(["python3", "-c", script]), timeout=30, check=False
            )
            if result.get("return_code", 1):
                raise EvaluationUnavailableError(
                    f"Native staging failed for {name}: {result.get('stderr')}"
                )


async def run_command(session, command, *, timeout, check=True):
    script = """import json,subprocess,sys
try:
 result=subprocess.run(sys.argv[1],shell=True,capture_output=True,text=True,errors='replace',timeout=float(sys.argv[2]))
 payload={'return_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
except subprocess.TimeoutExpired:
 payload={'return_code':124,'stdout':'','stderr':'Task command exceeded its remote deadline'}
print(json.dumps(payload))
"""
    try:
        transport = await asyncio.wait_for(
            session.run_command(
                shlex.join(["python3", "-c", script, command, str(timeout)]), check=False
            ),
            timeout=timeout + 5,
        )
    except TimeoutError as error:
        raise EvaluationUnavailableError(
            f"Support remote command exceeded {timeout}s; inspect retained native receipts"
        ) from error
    try:
        result = json.loads(transport["stdout"])
        if (
            type(result["return_code"]) is not int
            or not isinstance(result["stdout"], str)
            or not isinstance(result["stderr"], str)
        ):
            raise ValueError("Invalid command receipt")
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationUnavailableError("Support command transport has no exit receipt") from error
    if result["return_code"] == 124 or (check and result["return_code"]):
        raise EvaluationUnavailableError(
            f"Support remote command exited {result['return_code']}: {result['stderr']}"
        )
    return result
