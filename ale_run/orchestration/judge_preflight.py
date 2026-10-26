"""Probe shared LLM judges before provisioning any selected run units."""

from __future__ import annotations

import ast
import asyncio
import io
import logging
from pathlib import Path
from typing import Iterable

from .experiment_spec import RunUnit

logger = logging.getLogger(__name__)

_TEXT_ONLY_TASKS = frozenset({
    "business_finance/digital_marketing_ab_test_analysis_1",
    "engineering/humanoid_wbc_policy_evaluation",
})
_TASKS_ROOT = Path(__file__).resolve().parents[2] / "tasks"


def judge_requirements(task_paths: Iterable[str]) -> dict[str, list[str]]:
    """Discover task-local shared judge imports without executing task modules."""
    required: dict[str, list[str]] = {}
    for task_path in sorted(set(task_paths)):
        directory = _TASKS_ROOT / task_path
        sources = list(directory.glob("*.py")) + list((directory / "scripts").rglob("*.py"))
        uses_judge = False
        for source in sources:
            if "evidence" in source.relative_to(directory).parts:
                continue
            tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
            if any(
                isinstance(node, ast.ImportFrom)
                and node.module == "tasks.utils.evaluation"
                and any(
                    alias.name.startswith("llm_") or alias.name == "compare_screenshots_game"
                    for alias in node.names
                )
                for node in ast.walk(tree)
            ):
                uses_judge = True
                break
        if uses_judge:
            required.setdefault("text", []).append(task_path)
            if task_path not in _TEXT_ONLY_TASKS:
                required.setdefault("vision", []).append(task_path)
    return required


async def check_judges(units: Iterable[RunUnit]) -> None:
    required = judge_requirements(unit.task_path for unit in units)
    if not required:
        return

    from tasks.utils.evaluation import (
        JudgeInfrastructureError,
        build_vision_content,
        llm_multimodal_json,
    )

    async def probe(capability: str) -> None:
        if capability == "vision":
            from PIL import Image

            stream = io.BytesIO()
            Image.new("RGB", (64, 64), (255, 0, 0)).save(stream, format="PNG")
            content = build_vision_content(
                'What is the dominant color in this image? Return JSON {"color":"color name"}.',
                [stream.getvalue()],
            )
        else:
            content = [{"type": "text", "text": 'Return JSON {"sum":7+8}, evaluating the sum.'}]
        try:
            result = await asyncio.wait_for(
                llm_multimodal_json(content=content, max_tokens=128), timeout=150,
            )
            valid = (
                str(result.get("color", "")).strip().lower() == "red"
                if capability == "vision"
                else result.get("sum") == 15
            )
            if not valid:
                raise JudgeInfrastructureError("Judge probe returned an incorrect answer")
        except Exception as error:
            raise JudgeInfrastructureError(
                f"Judge {capability} preflight failed ({type(error).__name__}); "
                "no sandboxes were started. Check JUDGE_BASE_URL, JUDGE_API_KEY, "
                "JUDGE_MODEL and retry the experiment."
            ) from None
        logger.info("Judge %s preflight passed for %d selected task(s)", capability, len(required[capability]))

    await asyncio.gather(*(probe(capability) for capability in required))
