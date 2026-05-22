'''
Reverse a bundled Windows crackme and make it appear registered, probably by using IDA or x64dbg to recover the logic and patching the registry or executable.
Credit to: Juanru Li @ SJTU
'''
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

logger = logging.getLogger(__name__)


REGISTRY_PATH = (
    r"Registry::HKEY_CURRENT_USER\Software\Classes\VirtualStore\MACHINE\SOFTWARE\WOW6432Node"
    r"\Stefan Pettersson\YourTris"
)


def generate_password(name: str) -> str:
    tmp = "".join(str(ord(ch) * (idx + 1) + idx) for idx, ch in enumerate(name))
    n = len(name)
    start_idx = max(0, n - 7)
    end_idx = max(start_idx + 6, n - 1)
    return tmp[start_idx:end_idx]


def last_json(stdout: str) -> Dict[str, Any]:
    for line in reversed([x.strip() for x in str(stdout).splitlines() if x.strip()]):
        try:
            return json.loads(line)
        except Exception:
            continue
    return {}


def trim_text(value: Any, limit: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


@dataclass
class TaskConfig(GeneralTaskConfig):
    VARIANT_NAME: str = "base"
    DOMAIN_NAME: str = "computing_math"

    TASK_NAME: str = "tris_crackme"

    @property
    def task_dir(self) -> str:
        return fr"{self.REMOTE_ROOT_DIR}\{self.DOMAIN_NAME}\{self.TASK_NAME}\{self.VARIANT_NAME}"

    @property
    def exe_path(self) -> str:
        return fr"{self.task_dir}\input\tris.exe"

    @property
    def task_description(self) -> str:
        return f"""
Goal:
Solve the registration crackme and make the bundled Windows program appear registered.

Target program:
- Executable: {self.exe_path}

Requirements:
1. Launch and analyze the program if needed.
2. Make the registration succeed. You can use any method to achieve this, such as reverse engineering, patching, or registry manipulation. Make sure that after restarting the program, it still shows as registered.

Verification:
- Score is 1.0 only if the program shows as registered when launched.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "exe_path": self.exe_path,
                "registry_path": REGISTRY_PATH,
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    out_dir = task_cfg.metadata["remote_output_dir"]
    registry_path = task_cfg.metadata["registry_path"]

    logger.info("[tris_crackme] evaluate() begin")
    logger.info("[tris_crackme] reading registry from %s", registry_path)

    query_cmd = (
        "powershell -NoProfile -ExecutionPolicy Bypass -Command "
        f"\"$path = '{registry_path}'; "
        "if (Test-Path -LiteralPath $path) { "
        "$item = Get-ItemProperty -LiteralPath $path; "
        "[pscustomobject]@{"
        "exists=$true; "
        "RegName=([string]$item.RegName); "
        "RegCode=([string]$item.RegCode)"
        "} | ConvertTo-Json -Compress "
        "} else { "
        "[pscustomobject]@{exists=$false; RegName=$null; RegCode=$null} | ConvertTo-Json -Compress "
        "}\""
    )

    result: Dict[str, Any] = {}
    try:
        cmd_result = await session.run_command(query_cmd, check=False)
        result = last_json(cmd_result.get("stdout", ""))
        logger.info(
            "[tris_crackme] query rc=%s stdout=%s stderr=%s parsed=%s",
            cmd_result.get("return_code"),
            trim_text(cmd_result.get("stdout")),
            trim_text(cmd_result.get("stderr")),
            result,
        )
    except Exception as e:
        logger.warning("Registry query failed: %s", e)

    reg_name = str(result.get("RegName") or "")
    reg_code = str(result.get("RegCode") or "")
    expected_code = generate_password(reg_name) if reg_name else ""
    passed = bool(result.get("exists")) and bool(reg_name) and reg_code == expected_code

    logger.info(
        "[tris_crackme] exists=%s reg_name=%r reg_code=%r expected_code=%r passed=%s",
        result.get("exists"),
        reg_name,
        reg_code,
        expected_code,
        passed,
    )

    report = {
        "registry_path": registry_path,
        "observed": result,
        "expected_reg_code": expected_code,
        "passed": passed,
    }

    try:
        await session.write_file(
            os.path.join(out_dir, "autograde_report.json"),
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        logger.info("[tris_crackme] wrote autograde_report.json")
    except Exception:
        pass

    return [1.0 if passed else 0.0]
