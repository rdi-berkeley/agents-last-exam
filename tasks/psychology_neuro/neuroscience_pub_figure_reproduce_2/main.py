"""Neuroscience publication figure reproduction task (Task 2).

The agent must recreate a neuroscience experimental schematic in Adobe
Illustrator, producing both an editable .ai file and an exported PDF that
match the reference in scientific content and visual structure.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.utils.evaluation import (
    EvaluationContext,
    llm_vision_binary_checklist_judge,
    llm_vision_yes_no_judge,
)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent / "scripts"

# Minimum meaningful file sizes (bytes)
MIN_PDF_SIZE = 2048
MIN_AI_SIZE = 4096


@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "psychology_neuro"
    TASK_NAME: str = "neuroscience_pub_figure_reproduce_2"
    VARIANT_NAME: str = "base"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def input_reference_pdf(self) -> str:
        return rf"{self.input_dir}\reference_schematic.pdf"

    @property
    def input_template_ai(self) -> str:
        return rf"{self.input_dir}\template_mouse.ai"

    @property
    def input_caption(self) -> str:
        return rf"{self.input_dir}\caption.txt"

    @property
    def output_pdf(self) -> str:
        return rf"{self.remote_output_dir}\schematic_reproduce.pdf"

    @property
    def output_ai(self) -> str:
        return rf"{self.remote_output_dir}\schematic_reproduce.ai"

    @property
    def reference_expert_pdf(self) -> str:
        return rf"{self.reference_dir}\expert_reproduction.pdf"

    @property
    def reference_expert_ai(self) -> str:
        return rf"{self.reference_dir}\expert_reproduction.ai"

    @property
    def reference_hashes(self) -> str:
        return rf"{self.reference_dir}\source_hashes.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are a neuroscience researcher who needs to recreate a publication-quality \
experimental schematic in Adobe Illustrator.

## Your Task

1. Launch Adobe Illustrator using the batch file at:
   `{self.software_dir}\\launch_illustrator.bat`

2. Examine the reference schematic at `{self.input_reference_pdf}` and the \
technical caption at `{self.input_caption}` to understand the required content.

3. Open the template mouse component at `{self.input_template_ai}` — use it \
as a starting element for your reproduction.

4. Recreate the schematic as a genuine Illustrator drawing. The figure should \
depict a freely-moving electrophysiology recording setup with these key elements:
   - A mouse inside a rectangular open-field arena
   - A head-mounted implant with a tether
   - An overhead commutator on a support frame
   - The arena elevated above a base platform
   - A beneath-arena camera for behavioral tracking
   - All required labels and annotations

5. Apply consistent styling: appropriate line weights, colors, and fonts.

6. Save the final editable Illustrator file to:
   `{self.remote_output_dir}\\schematic_reproduce.ai`

7. Export the final figure as a high-resolution PDF to:
   `{self.remote_output_dir}\\schematic_reproduce.pdf`

## Important Notes
- Do NOT simply copy or paste the reference PDF into the output.
- The output must be a genuine vector Illustrator drawing, not a raster paste.
- Match the reference in scientific content, visual clarity, and layout.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_dir": self.input_dir,
                "input_reference_pdf": self.input_reference_pdf,
                "input_template_ai": self.input_template_ai,
                "input_caption": self.input_caption,
                "output_pdf": self.output_pdf,
                "output_ai": self.output_ai,
                "reference_expert_pdf": self.reference_expert_pdf,
                "reference_expert_ai": self.reference_expert_ai,
                "reference_hashes": self.reference_hashes,
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
            computer={
                "provider": "computer",
                "setup_config": {"os_type": config.OS_TYPE},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _render_pdf_to_png(pdf_bytes: bytes, dpi: int = 200) -> bytes:
    """Render the first page of a PDF to PNG bytes using PyMuPDF."""
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def _check_ai_illustrator_markers(data: bytes) -> dict:
    """Check whether an .ai file has genuine Adobe Illustrator PDF-backing.

    Returns a dict with boolean checks for key Illustrator markers.
    """
    header = data[:4096].decode("latin-1", errors="replace")
    body_sample = data[: min(len(data), 65536)].decode("latin-1", errors="replace")

    checks = {
        "has_pdf_header": header.startswith("%PDF"),
        "has_illustrator_creator": "Illustrator" in body_sample,
        "has_ai_private_data": "AIPrivateData" in body_sample or "AIPDFPrivateData" in body_sample,
        "has_optional_content": "/OCProperties" in body_sample or "/OC " in body_sample,
    }
    checks["passes"] = checks["has_pdf_header"] and (
        checks["has_illustrator_creator"] or checks["has_ai_private_data"]
    )
    return checks


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = f"neuroscience_pub_figure_reproduce_2::{meta.get('variant_name', 'base')}"

    async with EvaluationContext(task_tag=tag, mode="custom") as ctx:
        try:
            score = await _run_evaluation(meta, session, ctx)
            ctx.add_score(score)
            ctx.finalize()
            return [score]
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            ctx.log_error(identifier="top_level", error=exc)
            ctx.finalize()
            return [0.0]


async def _run_evaluation(
    meta: dict,
    session: cb.DesktopSession,
    ctx: EvaluationContext,
) -> float:
    """Core evaluation logic. Returns a score in [0.0, 1.0]."""

    # ------------------------------------------------------------------
    # Gate 1: Output file existence and minimum size
    # ------------------------------------------------------------------
    output_pdf_path = meta["output_pdf"]
    output_ai_path = meta["output_ai"]

    try:
        output_pdf_bytes = await session.read_bytes(output_pdf_path)
    except Exception:
        logger.error("Output PDF not found: %s", output_pdf_path)
        ctx.log_evaluation(identifier="gate_pdf_exists", score=0.0, error="Output PDF missing")
        return 0.0

    try:
        output_ai_bytes = await session.read_bytes(output_ai_path)
    except Exception:
        logger.error("Output AI not found: %s", output_ai_path)
        ctx.log_evaluation(identifier="gate_ai_exists", score=0.0, error="Output AI missing")
        return 0.0

    if len(output_pdf_bytes) < MIN_PDF_SIZE:
        ctx.log_evaluation(
            identifier="gate_pdf_size",
            score=0.0,
            error=f"PDF too small: {len(output_pdf_bytes)} bytes",
        )
        return 0.0

    if len(output_ai_bytes) < MIN_AI_SIZE:
        ctx.log_evaluation(
            identifier="gate_ai_size",
            score=0.0,
            error=f"AI too small: {len(output_ai_bytes)} bytes",
        )
        return 0.0

    ctx.log_evaluation(identifier="gate_files_present", score=1.0)

    # ------------------------------------------------------------------
    # Gate 2: Direct-copy detection via hash comparison
    # ------------------------------------------------------------------
    output_pdf_hash = _sha256(output_pdf_bytes)
    output_ai_hash = _sha256(output_ai_bytes)

    # Read input files for hash comparison
    try:
        input_ref_pdf_bytes = await session.read_bytes(meta["input_reference_pdf"])
        input_ref_pdf_hash = _sha256(input_ref_pdf_bytes)
    except Exception:
        input_ref_pdf_hash = None

    try:
        input_template_ai_bytes = await session.read_bytes(meta["input_template_ai"])
        input_template_ai_hash = _sha256(input_template_ai_bytes)
    except Exception:
        input_template_ai_hash = None

    if input_ref_pdf_hash and output_pdf_hash == input_ref_pdf_hash:
        ctx.log_evaluation(
            identifier="gate_copy_pdf",
            score=0.0,
            error="Output PDF is an exact copy of input reference",
        )
        return 0.0

    if input_template_ai_hash and output_ai_hash == input_template_ai_hash:
        ctx.log_evaluation(
            identifier="gate_copy_ai",
            score=0.0,
            error="Output AI is an exact copy of input template",
        )
        return 0.0

    if output_ai_hash == output_pdf_hash:
        ctx.log_evaluation(
            identifier="gate_ai_equals_pdf",
            score=0.0,
            error="Output AI hash matches output PDF (just renamed)",
        )
        return 0.0

    # Also compare against source hashes from reference/ if available
    try:
        hashes_bytes = await session.read_bytes(meta["reference_hashes"])
        source_hashes = json.loads(hashes_bytes.decode("utf-8"))
        for section in ("input_hashes",):
            section_data = source_hashes.get(section, {})
            if not isinstance(section_data, dict):
                continue
            for key, ref_hash in section_data.items():
                if output_pdf_hash == ref_hash:
                    ctx.log_evaluation(
                        identifier=f"gate_hash_match_{key}",
                        score=0.0,
                        error=f"Output PDF hash matches source hash: {key}",
                    )
                    return 0.0
                if output_ai_hash == ref_hash:
                    ctx.log_evaluation(
                        identifier=f"gate_hash_match_{key}",
                        score=0.0,
                        error=f"Output AI hash matches source hash: {key}",
                    )
                    return 0.0
    except Exception:
        logger.warning("Could not read source_hashes.json; skipping hash gate")

    ctx.log_evaluation(identifier="gate_copy_detection", score=1.0)

    # ------------------------------------------------------------------
    # Gate 3: AI artifact validation (Illustrator markers)
    # ------------------------------------------------------------------
    ai_checks = _check_ai_illustrator_markers(output_ai_bytes)
    if not ai_checks["passes"]:
        ctx.log_evaluation(
            identifier="gate_ai_artifact",
            score=0.0,
            error=f"AI file lacks Illustrator markers: {ai_checks}",
        )
        return 0.0

    ctx.log_evaluation(identifier="gate_ai_artifact", score=1.0)

    # ------------------------------------------------------------------
    # Gate 4: Output PDF must be renderable
    # ------------------------------------------------------------------
    try:
        output_png = _render_pdf_to_png(output_pdf_bytes)
    except Exception as exc:
        ctx.log_evaluation(
            identifier="gate_pdf_render", score=0.0, error=f"Cannot render output PDF: {exc}"
        )
        return 0.0

    ctx.log_evaluation(identifier="gate_pdf_render", score=1.0)

    # ------------------------------------------------------------------
    # Gate 5: Visual input-copy check — ensure output is not visually
    #         identical to the visible input reference
    # ------------------------------------------------------------------
    try:
        input_png = _render_pdf_to_png(input_ref_pdf_bytes)
        copy_check = await llm_vision_yes_no_judge(
            prompt=(
                "Compare these two images. The first is a submitted reproduction "
                "and the second is the original reference. Are they pixel-for-pixel "
                "identical or essentially the same image with no meaningful "
                "differences? Answer YES if they are identical copies, NO if they "
                "differ in any visible way."
            ),
            image_bytes=output_png,
            reference_image_bytes=input_png,
            max_tokens=10,
            eval_context=ctx,
            identifier="gate_visual_copy_check",
        )
        if copy_check.get("score", 0.0) >= 1.0:
            logger.error("Output appears to be a visual copy of the input reference")
            return 0.0
    except Exception as exc:
        logger.warning("Visual copy check failed, continuing: %s", exc)

    # ------------------------------------------------------------------
    # Scoring: Read expert reference and caption for comparison
    # ------------------------------------------------------------------
    try:
        ref_pdf_bytes = await session.read_bytes(meta["reference_expert_pdf"])
        ref_png = _render_pdf_to_png(ref_pdf_bytes)
    except Exception as exc:
        logger.error("Cannot read/render expert reference PDF: %s", exc)
        ctx.log_evaluation(identifier="ref_read_fail", score=0.0, error=str(exc))
        return 0.0

    try:
        caption_bytes = await session.read_bytes(meta["input_caption"])
        caption_text = caption_bytes.decode("utf-8", errors="replace")
    except Exception:
        caption_text = ""

    # ------------------------------------------------------------------
    # Score component 1: Structural/visual similarity checklist (40%)
    # ------------------------------------------------------------------
    structural_result = await llm_vision_binary_checklist_judge(
        prompt_intro=(
            "You are comparing a submitted neuroscience schematic reproduction "
            "(first image) against an expert reference reproduction (second image). "
            "Judge whether the submitted figure preserves the overall structure "
            "and layout of the reference."
        ),
        checklist_items=[
            (
                "layout_match",
                "Does the overall spatial layout of the submitted "
                "figure match the reference (similar arrangement of major elements)?",
            ),
            (
                "scale_proportion",
                "Are the major elements (arena, mouse, frame, "
                "camera) at roughly similar scales and proportions as the reference?",
            ),
            (
                "labels_present",
                "Does the submitted figure include text labels "
                "and annotations comparable to those in the reference?",
            ),
            (
                "line_style",
                "Does the submitted figure use clean, consistent "
                "line weights and styling similar to the reference?",
            ),
            (
                "publication_quality",
                "Does the submitted figure look publication-"
                "ready with sufficient visual clarity and polish?",
            ),
        ],
        image_bytes=output_png,
        reference_image_bytes=ref_png,
        max_tokens=512,
        eval_context=ctx,
        identifier="structural_similarity",
    )
    structural_score = structural_result.get("score", 0.0)

    # ------------------------------------------------------------------
    # Score component 2: Scientific content checklist (40%)
    # ------------------------------------------------------------------
    semantic_result = await llm_vision_binary_checklist_judge(
        prompt_intro=(
            "You are evaluating a submitted neuroscience experimental schematic "
            "(first image) for scientific content accuracy. Compare it against "
            "the expert reference (second image) and the following caption "
            "describing required content:\n\n"
            f'"""\n{caption_text}\n"""\n\n'
            "Judge whether the submitted figure includes the essential "
            "scientific elements."
        ),
        checklist_items=[
            (
                "mouse_in_arena",
                "Is there a mouse depicted inside a rectangular " "open-field arena?",
            ),
            (
                "head_implant_tether",
                "Does the figure show a head-mounted implant "
                "on the mouse connected by a tether?",
            ),
            (
                "commutator_frame",
                "Is there an overhead commutator mounted on a " "support frame above the arena?",
            ),
            (
                "elevated_arena",
                "Is the arena shown elevated above a base " "platform or ground level?",
            ),
            (
                "camera_below",
                "Is there a camera depicted beneath or below the " "arena for behavioral tracking?",
            ),
            (
                "correct_connections",
                "Are the connections between elements "
                "(tether from mouse to commutator, camera aimed at arena) "
                "logically correct?",
            ),
        ],
        image_bytes=output_png,
        reference_image_bytes=ref_png,
        max_tokens=512,
        eval_context=ctx,
        identifier="semantic_content",
    )
    semantic_score = semantic_result.get("score", 0.0)

    # ------------------------------------------------------------------
    # Score component 3: Overall scientific equivalence (20%)
    # ------------------------------------------------------------------
    equivalence_result = await llm_vision_yes_no_judge(
        prompt=(
            "Compare these two neuroscience experimental schematics. The first "
            "is a submitted reproduction and the second is an expert reference. "
            "Do both figures communicate the same main scientific setup and "
            "experimental concept? Answer YES if the core scientific message is "
            "preserved, NO if essential elements are missing or incorrect."
        ),
        image_bytes=output_png,
        reference_image_bytes=ref_png,
        max_tokens=10,
        eval_context=ctx,
        identifier="scientific_equivalence",
    )
    equivalence_score = equivalence_result.get("score", 0.0)

    # ------------------------------------------------------------------
    # Combine scores
    # ------------------------------------------------------------------
    final_score = 0.4 * structural_score + 0.4 * semantic_score + 0.2 * equivalence_score

    logger.info(
        "Final score: %.3f (structural=%.3f, semantic=%.3f, equivalence=%.3f)",
        final_score,
        structural_score,
        semantic_score,
        equivalence_score,
    )
    return final_score
