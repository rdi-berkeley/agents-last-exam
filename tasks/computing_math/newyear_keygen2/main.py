'''
2025 New Year Keygen Challenge
Credit to: 52pojie
'''
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

logger = logging.getLogger(__name__)

DELTA = 0xB979379E
MASK32 = 0xFFFFFFFF
K1 = (0x1231F787, 0x9ACD6D9A, 0xD7851B65, 0x473457C1)
K2 = (0xC52D865C, 0x10778A6E, 0xB728E994, 0x1746382E)
K3 = (0x779375B2, 0xEFCB8541, 0x7459F437, 0x090D1E5D)
V2_PREFIX = 0x04040404
SALT = bytes.fromhex("3532706F6A6965203230323520E788B1E9A39EE79A84E78CAB00")
DEFAULT_UID_TEXT = os.environ.get("NEWYEAR_KEYGEN2_UID", "20252025")


def u32(value: int) -> int:
    return value & MASK32


def encrypt(block: tuple[int, int], key: tuple[int, int, int, int]) -> tuple[int, int]:
    v0, v1 = block[1], block[0]
    sum_ = 0
    k0, k1, k2, k3 = key
    for _ in range(12):
        sum_ = u32(sum_ + DELTA)
        v1 = u32(v1 + (((v0 << 4) + k0) ^ (v0 + sum_) ^ ((v0 >> 5) + k1)))
        v0 = u32(v0 + (((v1 << 4) + k2) ^ (v1 + sum_) ^ ((v1 >> 5) + k3)))
    return (v0, v1)


def block_to_hex(block: tuple[int, int]) -> str:
    return "".join(value.to_bytes(4, "little").hex() for value in block)


def leftrotate(value: int, shift: int) -> int:
    value &= MASK32
    return ((value << shift) | (value >> (32 - shift))) & MASK32


def custom_md5_digest(data: bytes) -> bytes:
    shifts = [7, 12, 17, 22] * 4 + [5, 9, 14, 20] * 4 + [4, 11, 16, 23] * 4 + [6, 10, 15, 21] * 4
    table = [int(abs(__import__("math").sin(i + 1)) * (1 << 32)) & MASK32 for i in range(64)]
    a0, b0, c0, d0 = 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476
    msg = bytearray(data)
    bit_len = (8 * len(msg)) & 0xFFFFFFFFFFFFFFFF
    msg.append(0x80)
    while len(msg) % 64 != 56:
        msg.append(0)
    bit_len = (bit_len + 2) & 0xFFFFFFFFFFFFFFFF
    msg += bit_len.to_bytes(8, "little")
    for offset in range(0, len(msg), 64):
        chunk = msg[offset : offset + 64]
        words = [int.from_bytes(chunk[i : i + 4], "little") for i in range(0, 64, 4)]
        a, b, c, d = a0, b0, c0, d0
        for i in range(64):
            if i < 16:
                f = (b & c) | ((~b) & d)
                g = i
            elif i < 32:
                f = (d & b) | ((~d) & c)
                g = (5 * i + 1) % 16
            elif i < 48:
                f = b ^ c ^ d
                g = (3 * i + 5) % 16
            else:
                f = c ^ (b | (~d))
                g = (7 * i) % 16
            f = (f + a + table[i] + words[g]) & MASK32
            a, d, c, b = d, c, b, (b + leftrotate(f, shifts[i])) & MASK32
        a0 = (a0 + a) & MASK32
        b0 = (b0 + b) & MASK32
        c0 = (c0 + c) & MASK32
        d0 = (d0 + d) & MASK32
    return b"".join(value.to_bytes(4, "little") for value in (a0, b0, c0, d0))


def derive_magic_value(uid: int, timestamp_utc: int) -> int:
    prefix = timestamp_utc.to_bytes(8, "little") + uid.to_bytes(8, "little")
    digest = custom_md5_digest(SALT + prefix * 10)
    return digest[15] | (digest[11] << 8) | (digest[7] << 16) | (digest[3] << 24)


def generate_password(uid: int, timestamp_utc: int) -> str:
    part0 = encrypt((0, timestamp_utc), K1)
    part1 = encrypt((0, uid), K2)
    part2 = encrypt((V2_PREFIX, derive_magic_value(uid, timestamp_utc)), K3)
    raw = block_to_hex(part0) + block_to_hex(part1) + block_to_hex(part2)
    return f"flag{{{raw}}}"


def current_utc_halfhour_slot(now_ts: int | None = None) -> int:
    if now_ts is None:
        now_ts = int(time.time())
    return now_ts - (now_ts % 1800)


def build_task_description(exe_path: str, output_path: str, uid_text: str) -> str:
    return f"""
Provided UID:
- `{uid_text}`

Goal:
Reverse the bundled Windows crackme and compute the correct password for the provided UID.

Inputs:
- Executable to analyze: {exe_path}

Output:
- Save EXACTLY one UTF-8 text file to: {output_path}
- The file must contain exactly one line in the form `flag{{...}}`
- Do not add commentary, labels, or extra blank lines

Important:
- The valid password depends on the current UTC time.
- Use the most recent UTC half-hour boundary, i.e. the current UTC timestamp rounded down to a multiple of 1800 seconds.

Verification:
- Score is 1.0 only if the output matches the password for this UID using the evaluation-time UTC half-hour slot.
"""


@dataclass
class TaskConfig(GeneralTaskConfig):
    VARIANT_NAME: str = "base"
    DOMAIN_NAME: str = "computing_math"

    TASK_NAME: str = "newyear_keygen2"
    OUTPUT_FILENAME: str = "key.txt"

    @property
    def task_dir(self) -> str:
        return fr"{self.REMOTE_ROOT_DIR}\{self.DOMAIN_NAME}\{self.TASK_NAME}\{self.VARIANT_NAME}"

    @property
    def exe_path(self) -> str:
        return fr"{self.task_dir}\input\crackme.exe"

    @property
    def output_path(self) -> str:
        return fr"{self.remote_output_dir}\{self.OUTPUT_FILENAME}"

    @property
    def task_description(self) -> str:
        return f"""
Goal:
Reverse the bundled Windows crackme and compute the correct password for the provided UID.

Inputs:
- Executable to analyze: {self.exe_path}

Output:
- Save EXACTLY one UTF-8 text file to: {self.output_path}
- The file must contain exactly one line in the form `flag{{...}}`
- Do not add commentary, labels, or extra blank lines
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "exe_path": self.exe_path,
                "output_path": self.output_path,
                "output_filename": self.OUTPUT_FILENAME,
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    uid_text = DEFAULT_UID_TEXT
    description = build_task_description(config.exe_path, config.output_path, uid_text)
    metadata = config.to_metadata()
    metadata["uid_text"] = uid_text
    return [
        cb.Task(
            description=description,
            metadata=metadata,
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
    output_path = task_cfg.metadata["output_path"]
    uid_text = task_cfg.metadata["uid_text"]

    logger.info("[newyear_keygen2] evaluate() begin")
    logger.info("[newyear_keygen2] output_path=%s", output_path)
    logger.info("[newyear_keygen2] uid_text=%s", uid_text)

    try:
        output_text = await session.read_file(output_path)
    except Exception as e:
        logger.warning("[newyear_keygen2] failed to read required files: %s", e)
        try:
            await session.write_file(
                os.path.join(out_dir, "autograde_report.json"),
                json.dumps({"passed": False, "error": str(e)}, ensure_ascii=False, indent=2),
            )
        except Exception:
            pass
        return [0.0]

    try:
        uid_value = int(uid_text)
    except Exception:
        uid_value = None

    output_lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    candidate = output_lines[0].lower() if len(output_lines) == 1 else ""
    slot = current_utc_halfhour_slot()
    expected = generate_password(uid_value, slot) if uid_value is not None else ""
    passed = bool(candidate) and candidate == expected

    report: Dict[str, Any] = {
        "passed": passed,
        "uid": uid_text,
        "uid_parse_ok": uid_value is not None,
        "output_line_count": len(output_lines),
        "candidate": candidate,
        "expected": expected,
        "slot_utc": slot,
    }

    try:
        await session.write_file(
            os.path.join(out_dir, "autograde_report.json"),
            json.dumps(report, ensure_ascii=False, indent=2),
        )
    except Exception:
        pass

    logger.info(
        "[newyear_keygen2] uid=%s slot=%s candidate=%s expected=%s passed=%s",
        uid_text,
        slot,
        candidate,
        expected,
        passed,
    )
    return [1.0 if passed else 0.0]
