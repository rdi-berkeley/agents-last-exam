"""End-to-end repro: hostile Linux listing entries must not escape gather.

Drives the REAL _list_dir_sync parser and the REAL gather_dir/pull_to_host
consumers with only the cua HTTP transport faked (the same seam the
existing test suite mocks). Demonstrates the pre-fix escape (for the
historical record) and asserts the fix: every hostile entry variant is
rejected and nothing lands outside the destination root.

Run:  .venv/bin/python tests/ale_run/repro/repro_sandbox_listing_traversal.py
Exit 0 when the fix holds (hostile entries rejected, no escape, benign
gather still works).
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from ale_run.agents.dummy.config import DummyConfig  # noqa: E402
from ale_run.base_interface import SandboxHandle  # noqa: E402
from ale_run.base_interface import sandbox as sandbox_interface  # noqa: E402
from ale_run.executors import sandbox as sandbox_executor  # noqa: E402
from ale_run.executors.sandbox import SandboxExecutor  # noqa: E402


def linux_sandbox() -> SandboxHandle:
    return SandboxHandle(
        id="repro-traversal",
        endpoint="http://invalid.invalid",
        os="linux",
        work_dir_base="/home/user/.ale",
        task_data_root="/data",
        node="node",
        python="python",
        mcp_server_dir="cua_mcp_server",
    )


async def main() -> int:
    sandbox = linux_sandbox()
    hostile_variants = [
        r"..\..\escaped_root\pwned_by_backslash.txt",
        r"nested\..\..\escaped_root\pwned_mixed.txt",
        "../escaped_root/pwned_native.txt",
        "nested/../../escaped_root/pwned_normpath.txt",
    ]
    outcomes = []
    for rel in hostile_variants:
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "gather_dst"
            dst.mkdir()
            payload = f"{rel}\tf\t21\n"
            transport = Mock(
                return_value={"success": True, "return_code": 0, "stdout": payload}
            )
            sandbox_interface._post_cmd = transport
            download = AsyncMock(return_value=True)
            sandbox_executor._download_with_retry = download
            executor = SandboxExecutor(
                config=DummyConfig(), work_dir=sandbox.work_dir_base, sandbox=sandbox
            )
            report = await executor.gather_dir(src=sandbox.work_dir_base, dst=dst)
            escaped = any(
                p.is_file()
                for p in Path(td).rglob("*")
                if "escaped_root" in str(p)
            )
            rejected = bool(report.error)
            outcomes.append((rel, rejected, escaped))
            status = "REJECTED" if rejected and not escaped else "ESCAPED"
            print(f"  [{status}] {rel!r}")

    # newline-forgery variant: a filename containing \n injects a second
    # listing line whose columns are attacker-controlled
    forged = 'benign.txt\n..\\..\\escaped_root\\pwned_by_newline.txt\tf\t21'
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "gather_dst"
        dst.mkdir()
        payload = f"real.txt\tf\t5\n{forged}\n"
        transport = Mock(
            return_value={"success": True, "return_code": 0, "stdout": payload}
        )
        sandbox_interface._post_cmd = transport
        download = AsyncMock(return_value=True)
        sandbox_executor._download_with_retry = download
        executor = SandboxExecutor(
            config=DummyConfig(), work_dir=sandbox.work_dir_base, sandbox=sandbox
        )
        report = await executor.gather_dir(src=sandbox.work_dir_base, dst=dst)
        escaped = any("escaped_root" in str(p) for p in Path(td).rglob("*"))
        outcomes.append(("<newline-forged>", bool(report.error), escaped))
        status = "REJECTED" if report.error and not escaped else "ESCAPED"
        print(f"  [{status}] newline-forged listing line")

    # benign gather still works
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "gather_dst"
        dst.mkdir()
        payload = "output/results.json\tf\t11\nlogs/trace.jsonl\td\t0\n"
        transport = Mock(
            return_value={"success": True, "return_code": 0, "stdout": payload}
        )
        sandbox_interface._post_cmd = transport
        download = AsyncMock(return_value=True)
        sandbox_executor._download_with_retry = download
        executor = SandboxExecutor(
            config=DummyConfig(), work_dir=sandbox.work_dir_base, sandbox=sandbox
        )
        report = await executor.gather_dir(src=sandbox.work_dir_base, dst=dst)
        benign_ok = report.error is None and download.await_count == 1
        outcomes.append(("<benign>", not benign_ok, False))
        print(f"  [{'OK' if benign_ok else 'BROKEN'}] benign listing still gathers")

    all_rejected = all(r and not e for rel, r, e in outcomes[:-1])
    print(
        "RESULT:",
        "HARDENED - every hostile listing entry rejected; benign gather unaffected"
        if all_rejected and benign_ok
        else "VULNERABLE - at least one hostile entry escaped the destination root",
    )
    return 0 if all_rejected and benign_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
