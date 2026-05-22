"""AgentHLE task: go_game_reconstruction_1."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import DATA_ROOT, LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
VERIFY_SCRIPT_PATH = SCRIPTS_DIR / "verify_sgf.py"

def _load_verify_module():
    spec = importlib.util.spec_from_file_location(
        "go_game_reconstruction_1_verify_sgf", VERIFY_SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load verifier module from {VERIFY_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFY_MODULE = _load_verify_module()


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: Optional[float] = None,
    check: bool = False,
) -> dict[str, Any]:
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


def _parse_json_stdout(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty stdout")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"could not parse JSON from stdout: {text[:400]}")


class GoGameReconstructionConfig(LinuxTaskConfig):
    """Configuration for the Sabaki GUI reconstruction benchmark."""

    VARIANT_NAME: str = "base"

    def __init__(self) -> None:
        super().__init__(
            DOMAIN_NAME="computing_math", TASK_NAME="go_game_reconstruction_1",
            VARIANT_NAME="base",
            OS_TYPE="linux",
            REMOTE_ROOT_DIR=os.environ.get("REMOTE_ROOT_DIR", "/media/user/data/agenthle"),
        )

    @property
    def board_image(self) -> str:
        return f"{self.input_dir}/input-board-position.png"

    @property
    def sabaki_appimage(self) -> str:
        return f"{self.software_dir}/sabaki-v0.52.2-linux-x64.AppImage"

    @property
    def preferred_output_name(self) -> str:
        return "reconstructed_game.sgf"

    @property
    def preferred_output_path(self) -> str:
        return f"{self.remote_output_dir}/{self.preferred_output_name}"

    @property
    def reference_sgf(self) -> str:
        return f"{self.reference_dir}/ground-truth.sgf"

    @property
    def task_description(self) -> str:
        return f"""\
You are reconstructing a professional 19x19 Go game in Sabaki on Ubuntu.

## Your Task
Use the final board image plus the known metadata to reconstruct the game move
by move inside Sabaki, then export the reconstructed game as SGF.

## Visible Inputs
- Final board image: `{self.board_image}`
- Sabaki AppImage: `{self.sabaki_appimage}`

## Known Constraints
- Total moves: `168`
- Rules: `Chinese rules`
- Result: `White wins by resignation`
- Move 1: `B at R4`
- Move 2: `W at Q16`
- Move 3: `B at C4`
- Move 4: `W at C16`
- Move 5: `B at E3`

## What You Must Do
1. Launch Sabaki from the staged AppImage, for example:
   `"{self.sabaki_appimage}" --no-sandbox`
2. Use Sabaki's GUI to enter the game move by move.
3. Do not use a Go engine or a web browser.
4. Export the SGF into `{self.remote_output_dir}`.
5. Prefer the exact filename `{self.preferred_output_name}`.

Only files under `{self.remote_output_dir}` count as your output.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_name": self.VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "remote_output_dir": self.remote_output_dir,
                "board_image": self.board_image,
                "sabaki_appimage": self.sabaki_appimage,
                "preferred_output_name": self.preferred_output_name,
                "preferred_output_path": self.preferred_output_path,
                "reference_dir": self.reference_dir,
                "reference_gcs_prefix": "gs://ale-data-all/computing_math/go_game_reconstruction_1/base/reference",
                "reference_sgf": self.reference_sgf,
                "canonical_gcs_root": "gs://ale-data-all/computing_math/go_game_reconstruction_1/base/",
            }
        )
        return metadata


config = GoGameReconstructionConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": config.OS_TYPE},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _choose_candidate_path(meta: dict[str, Any], session: cb.DesktopSession) -> Optional[str]:
    preferred = meta["preferred_output_path"]
    if await session.exists(preferred):
        return preferred

    result = await _run_command(
        session,
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import json\n"
        f"root = Path({meta['remote_output_dir']!r})\n"
        "files = sorted(str(path) for path in root.glob('*.sgf') if path.is_file())\n"
        "print(json.dumps(files))\n"
        "PY",
        timeout=30.0,
        check=False,
    )
    if result.get("return_code", 1) != 0:
        logger.warning("candidate enumeration failed: %s", (result.get("stderr") or "")[:300])
        return None

    try:
        sgf_paths = _parse_json_stdout(result.get("stdout", ""))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("candidate enumeration parse failed: %s", exc)
        return None

    if not isinstance(sgf_paths, list):
        return None
    if len(sgf_paths) == 1:
        return sgf_paths[0]
    if not sgf_paths:
        return None

    fallbacks = [
        f"{meta['remote_output_dir']}/ground-truth.sgf",
        f"{meta['remote_output_dir']}/blank_19x19.sgf",
    ]
    for path in fallbacks:
        if path in sgf_paths:
            return path
    logger.warning("ambiguous candidate outputs in %s: %s", meta["remote_output_dir"], sgf_paths)
    return None


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score a candidate SGF locally against the evaluator-owned hidden reference SGF."""

    meta = task_cfg.metadata
    candidate_path = await _choose_candidate_path(meta, session)
    if not candidate_path:
        logger.warning("no candidate SGF found in %s", meta["remote_output_dir"])
        return [0.0]

    try:
        candidate_bytes = await session.read_bytes(candidate_path)
    except Exception as exc:
        logger.error("failed reading candidate SGF from VM: %s", exc)
        return [0.0]

    with tempfile.TemporaryDirectory(prefix="go_game_reconstruction_1_eval_") as tmpdir:
        tmp_root = Path(tmpdir)
        candidate_local = tmp_root / "candidate.sgf"
        reference_local = tmp_root / "reference.sgf"
        candidate_local.write_bytes(candidate_bytes)
        try:
            reference_local.write_bytes(await session.read_bytes(meta["reference_sgf"]))
        except Exception as exc:
            logger.error("failed reading staged reference SGF from VM: %s", exc)
            return [0.0]
        try:
            report = VERIFY_MODULE.score(candidate_local, reference_local)
        except Exception as exc:
            logger.error("local SGF evaluation failed: %s", exc)
            return [0.0]

    logger.info(
        "Scored candidate %s against hidden reference: %s/%s checkpoints",
        candidate_path,
        report.get("checkpoints_passed"),
        report.get("checkpoints_total"),
    )
    return [float(report.get("score", 0.0))]
