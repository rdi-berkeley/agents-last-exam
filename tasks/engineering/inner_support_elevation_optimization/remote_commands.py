"""Bound task commands without extending the DesktopSession protocol."""

import asyncio
import json
import shlex

from .verification import EvaluationUnavailableError


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
