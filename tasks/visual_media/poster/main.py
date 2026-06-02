"""Poster recreation and editing task family."""

import asyncio
import io
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.utils.evaluation import llm_vision_yes_no_judge, resolve_llm_judge_model

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
EVALUATOR_PACKAGES = [
    "Pillow",
    "numpy<2",
    "opencv-python-headless<4.8",
    "scikit-image",
    "easyocr==1.7.2",
    "torch==2.5.1",
]
VARIANTS = [
    "text_only",
    "change_images",
    "add_and_fix_components",
    "change_qr_code",
    "final_exam",
]


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def _parse_json_from_command(result: dict, *, label: str) -> dict:
    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""
    if stderr.strip():
        logger.info("%s stderr: %s", label, stderr.strip())

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    last_open = stdout.rfind("{")
    if last_open != -1:
        candidate = stdout[last_open:].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse JSON from {label}: {result}")


def _run_host_verifier(script_name: str, args: list[str], *, label: str) -> dict:
    command = ["uv", "run", "--no-project"]
    for package in EVALUATOR_PACKAGES:
        command.extend(["--with", package])
    command.extend(["python", str(SCRIPTS_DIR / script_name), *args])
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = {"stdout": result.stdout, "stderr": result.stderr, "return_code": result.returncode}
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed on evaluator host: {payload}")
    return payload


def _check_host_verifier_runtime() -> None:
    command = ["uv", "run", "--no-project"]
    for package in EVALUATOR_PACKAGES:
        command.extend(["--with", package])
    command.extend(
        [
            "python",
            "-c",
            ("import cv2, easyocr; " "assert cv2.__version__.startswith('4.7.'), cv2.__version__"),
        ]
    )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            "poster evaluator host runtime is unavailable: "
            + json.dumps(
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                },
                ensure_ascii=True,
            )
        )


@dataclass
class PosterTaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "poster"
    VARIANT_NAME: str = ""

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def assets_dir(self) -> str:
        return rf"{self.input_dir}\assets"

    @property
    def original_poster(self) -> str:
        return rf"{self.input_dir}\original_poster.png"

    @property
    def edit_request(self) -> str:
        return rf"{self.input_dir}\edit_request.txt"

    @property
    def output_poster(self) -> str:
        return rf"{self.remote_output_dir}\edited_poster.png"

    @property
    def reference_poster(self) -> str:
        return rf"{self.reference_dir}\edited_poster.png"

    @property
    def output_test_pos_dir(self) -> str:
        return rf"{self.task_dir}\output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return rf"{self.task_dir}\output_test_neg"

    @property
    def task_description(self) -> str:
        return f"""\
You are a graphic designer.

## Your Task
1. Inspect the poster image at `{self.original_poster}`.
2. Read the edit request at `{self.edit_request}`.
3. Recreate the poster as faithfully as possible.
4. Apply the requested edits while preserving unrelated content.
5. Export the final PNG to `{self.output_poster}`.

## Input Files
- Reference poster: `{self.original_poster}`
- Edit request: `{self.edit_request}`
- Optional assets directory: `{self.assets_dir}`

## Output
- Save the final image exactly to `{self.output_poster}`

## Notes
- Use any suitable tool available on the VM.
- Keep the output resolution comparable to the original poster.
- Preserve content that is not mentioned in the edit request.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.pop("software_dir", None)
        metadata.update(
            {
                "input_dir": self.input_dir,
                "assets_dir": self.assets_dir,
                "original_poster": self.original_poster,
                "edit_request": self.edit_request,
                "output_poster": self.output_poster,
                "reference_poster": self.reference_poster,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "eval_model": resolve_llm_judge_model(default="gpt-5.2"),
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=PosterTaskConfig(VARIANT_NAME=tag).task_description,
            metadata=PosterTaskConfig(VARIANT_NAME=tag).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
        for tag in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    import io as _io

    meta = task_cfg.metadata
    output_poster = meta["output_poster"]
    reference_poster = meta["reference_poster"]
    original_poster = meta["original_poster"]
    edit_request_path = meta["edit_request"]

    if not (await session.file_exists(output_poster) or await session.directory_exists(output_poster)):
        logger.warning("Missing output file: %s", output_poster)
        return [0.0]
    if not (await session.file_exists(reference_poster) or await session.directory_exists(reference_poster)):
        logger.warning("Missing reference file: %s", reference_poster)
        return [0.0]

    await asyncio.to_thread(_check_host_verifier_runtime)
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="poster_eval_") as tmpdir:
        tmp_root = Path(tmpdir)
        agent_path = tmp_root / "agent.png"
        reference_path = tmp_root / "reference.png"
        original_path = tmp_root / "original.png"
        agent_bytes = await session.read_bytes(output_poster)
        reference_bytes = await session.read_bytes(reference_poster)
        original_bytes = await session.read_bytes(original_poster)
        agent_path.write_bytes(agent_bytes)
        reference_path.write_bytes(reference_bytes)
        original_path.write_bytes(original_bytes)

        verify_result = await asyncio.to_thread(
            _run_host_verifier,
            "verify_poster.py",
            [
                "--agent",
                str(agent_path),
                "--reference",
                str(reference_path),
                "--original",
                str(original_path),
            ],
            label="verify_poster",
        )
        try:
            verify_payload = _parse_json_from_command(verify_result, label="verify_poster")
        except Exception:
            logger.error("Failed to parse verify output: %s", verify_result)
            return [0.0]

        if not verify_payload.get("gate_passed"):
            logger.info("Poster gate failed: %s", verify_payload.get("gate_reason"))
            return [0.0]

        ssim_score = float(verify_payload.get("ssim_score", 0.0))
        ocr_score = float(verify_payload.get("ocr_score", 0.0))

        regions_result = await asyncio.to_thread(
            _run_host_verifier,
            "detect_edit_regions.py",
            [
                "--original",
                str(original_path),
                "--reference",
                str(reference_path),
            ],
            label="detect_edit_regions",
        )
        try:
            regions_payload = _parse_json_from_command(regions_result, label="detect_edit_regions")
            regions = regions_payload.get("regions", [])
        except Exception:
            logger.error("Failed to parse edit-region output: %s", regions_result)
            regions = []

        agent_img = Image.open(_io.BytesIO(agent_bytes)).convert("RGB")
        reference_img = Image.open(_io.BytesIO(reference_bytes)).convert("RGB")

    try:
        if agent_img.size != reference_img.size:
            agent_img = agent_img.resize(reference_img.size, Image.Resampling.LANCZOS)
    except Exception:
        logger.exception("Failed to load local poster images")
        return [0.0]

    edit_request_text = ""
    if (await session.file_exists(edit_request_path) or await session.directory_exists(edit_request_path)):
        edit_request_text = (await session.read_bytes(edit_request_path)).decode(
            "utf-8", errors="replace"
        )

    yes_count = 0
    for idx, region in enumerate(regions, start=1):
        x = int(region["x"])
        y = int(region["y"])
        w = int(region["w"])
        h = int(region["h"])
        crop_box = (x, y, x + w, y + h)

        agent_crop = agent_img.crop(crop_box)
        reference_crop = reference_img.crop(crop_box)

        agent_buf = io.BytesIO()
        reference_buf = io.BytesIO()
        agent_crop.save(agent_buf, format="PNG")
        reference_crop.save(reference_buf, format="PNG")

        judge_prompt = (
            "You are checking whether a requested poster edit was applied correctly.\n\n"
            f"Full edit request:\n{edit_request_text}\n\n"
            f"Compare the two crops for region {idx} of {len(regions)}.\n"
            "The first crop is the agent output. The second crop is the hidden reference.\n"
            "Answer YES only if the text, layout, and visual content match closely enough.\n"
            "Answer NO if the crop is wrong, missing, or noticeably different."
        )
        judge_result = await llm_vision_yes_no_judge(
            prompt=judge_prompt,
            image_bytes=agent_buf.getvalue(),
            reference_image_bytes=reference_buf.getvalue(),
            model=meta.get("eval_model"),
            max_tokens=10,
            return_details=True,
        )
        yes_count += int(float(judge_result.get("score", 0.0)) >= 0.5)

    vlm_score = yes_count / len(regions) if regions else 0.0
    auto_score = min(ssim_score, ocr_score)
    final_score = max(0.0, min(1.0, min(auto_score, vlm_score)))
    return [final_score]
