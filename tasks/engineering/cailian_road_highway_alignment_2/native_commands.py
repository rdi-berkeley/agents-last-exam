import asyncio
import json
import shlex


class NativeRoadEnvironmentError(RuntimeError):
    pass


async def run_native_command(session, command, *, seconds=320):
    script = """import json,subprocess,sys
result=subprocess.run(sys.argv[1],shell=True,capture_output=True,text=True,errors='replace')
print(json.dumps({'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}))
"""
    try:
        transport = await asyncio.wait_for(
            session.run_command(shlex.join(["python3", "-c", script, command]), check=False),
            timeout=seconds,
        )
        receipt = json.loads(transport.get("stdout", ""))
        if type(receipt.get("exit_code")) is not int or receipt["exit_code"] != 0:
            raise NativeRoadEnvironmentError(f"Native Road command failed: {receipt}")
        if not isinstance(receipt.get("stdout"), str):
            raise ValueError("Missing command output")
        return {"return_code": 0, "stdout": receipt["stdout"], "stderr": receipt.get("stderr", "")}
    except (TimeoutError, ValueError, AttributeError) as error:
        raise NativeRoadEnvironmentError("Native Road command lacks a successful execution receipt") from error
