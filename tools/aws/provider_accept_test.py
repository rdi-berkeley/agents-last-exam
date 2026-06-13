"""AwsProvider acceptance test: acquire -> handle I/O -> release.

Exercises the real AwsProvider code (run-instances, IP poll, wait_cua_ready,
SandboxHandle I/O, terminate) against environment_aws.yaml's cpu-free-ubuntu
snapshot. Does NOT need cua_bench (only open_session does)."""
import asyncio, os, sys
from pathlib import Path
from ale_run.orchestration.config_loader import _build_environment_from_path
from ale_run.orchestration.factory import build_provider
from ale_run.base_interface import SandboxSpec

async def main():
    spec_env = _build_environment_from_path("configs/environments/environment_aws.yaml", base_dir=Path("."))[0]
    prov = build_provider(spec_env.provider_specs["aws"])
    spec = SandboxSpec(snapshot="cpu-free-ubuntu", os="linux", task_id="accept-test", harness="smoke", model_tag="none")
    print("acquiring (launches EC2, waits cua)...", flush=True)
    sb = await prov.acquire(spec)
    try:
        print(f"ACQUIRED id={sb.id} endpoint={sb.endpoint} os={sb.os}", flush=True)
        print(f"paths: work={sb.work_dir_base} data={sb.task_data_root} node={sb.node} py={sb.python}", flush=True)
        r = await sb.run_command("echo hello-from-provider && uname -r && id -un", timeout=30)
        print(f"run_command rc={r.returncode} stdout={r.stdout.strip()!r} stderr={r.stderr.strip()!r}", flush=True)
        # file round-trip
        await sb.write_file("/tmp/ale_accept.txt", "roundtrip-ok\n")
        back = await sb.read_text("/tmp/ale_accept.txt")
        print(f"file roundtrip: {back.strip()!r} exists={await sb.exists('/tmp/ale_accept.txt')}", flush=True)
        print("PROVIDER_ACCEPT_OK" if (r.returncode==0 and back.strip()=="roundtrip-ok") else "PROVIDER_ACCEPT_FAIL", flush=True)
    finally:
        print(f"releasing (terminating {sb.id})...", flush=True)
        await prov.release(sb, mode="delete")
        print("RELEASED", flush=True)

asyncio.run(main())
