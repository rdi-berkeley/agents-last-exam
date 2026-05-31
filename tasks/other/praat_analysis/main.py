"""praat_analysis — Vowel formant analysis in Praat.

The agent performs a complete vowel formant analysis workflow: annotate vowel
boundaries in TextGrids, extract F1/F2 formants, and generate a publication-
quality vowel formant chart.

Evaluation: three checkpoints — TextGrid annotation accuracy (0.35),
formant extraction accuracy (0.40), and vowel plot quality via LLM vision
judge (0.25).
"""

import csv
import io
import logging
import re
import sys
from dataclasses import dataclass

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.utils.evaluation import EvaluationContext, llm_vision_json_judge

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)


def _decode_textgrid_bytes(raw: bytes) -> str:
    """Decode TextGrid bytes with BOM-aware handling.

    Praat writes TextGrids as UTF-16 BE (with 0xFE 0xFF BOM) whenever any
    label contains a non-ASCII character (e.g. `ɛ`, `ʌ`, `æ`); ASCII-only
    TextGrids are written as plain UTF-8.
    """
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be")
    return raw.decode("utf-8")


W_TEXTGRID = 0.35
W_FORMANT = 0.40
W_PLOT = 0.25


# ---------------------------------------------------------------------------
# Variant spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VariantSpec:
    """Describes the per-variant WAV/TextGrid/vowel layout.

    - base hides vowel identity behind numeric WAV filenames; the agent must
      listen and identify each vowel.
    - pos-2 / pos-3 deliver WAVs already named after their vowel; the agent
      skips identification and goes straight to annotation + measurement.
    """

    variant_name: str
    vowels: tuple[str, ...]
    wav_stems: tuple[str, ...]
    vowel_identity_hidden: bool

    def stem_to_vowel(self) -> dict[str, str]:
        return dict(zip(self.wav_stems, self.vowels, strict=True))


VARIANT_SPECS: dict[str, VariantSpec] = {
    "base": VariantSpec(
        variant_name="base",
        vowels=("a", "e", "i", "o", "u"),
        wav_stems=("1", "2", "3", "4", "5"),
        vowel_identity_hidden=True,
    ),
    "pos-2": VariantSpec(
        variant_name="pos-2",
        vowels=("a", "e", "ɛ", "i", "o", "u", "ʌ"),
        wav_stems=("a", "e", "ɛ", "i", "o", "u", "ʌ"),
        vowel_identity_hidden=False,
    ),
    "pos-3": VariantSpec(
        variant_name="pos-3",
        vowels=("æ", "e", "i", "o", "ʌ"),
        wav_stems=("æ", "e", "i", "o", "ʌ"),
        vowel_identity_hidden=False,
    ),
}

VARIANTS = [(name,) for name in VARIANT_SPECS]


# ---------------------------------------------------------------------------
# TextGrid parser
# ---------------------------------------------------------------------------
def parse_textgrid(text: str) -> list[tuple[float, float, str]]:
    """Parse a Praat TextGrid (ooTextFile format).

    Returns a list of (xmin, xmax, label) for all intervals in the first
    IntervalTier.
    """
    intervals: list[tuple[float, float, str]] = []
    lines = text.splitlines()
    i = 0
    # Find the first IntervalTier's intervals
    while i < len(lines):
        if "intervals [" in lines[i] and "size" not in lines[i]:
            # Read xmin, xmax, text for this interval
            xmin = xmax = 0.0
            label = ""
            for j in range(i + 1, min(i + 4, len(lines))):
                line = lines[j].strip()
                if line.startswith("xmin"):
                    xmin = float(line.split("=")[1].strip())
                elif line.startswith("xmax"):
                    xmax = float(line.split("=")[1].strip())
                elif line.startswith("text"):
                    match = re.search(r'"(.*?)"', line)
                    label = match.group(1) if match else ""
            intervals.append((xmin, xmax, label))
        i += 1
    return intervals


def vowel_intervals(
    intervals: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Filter to non-empty (vowel) intervals."""
    return [(xmin, xmax, label) for xmin, xmax, label in intervals if label.strip()]


# ---------------------------------------------------------------------------
# Formant CSV parser
# ---------------------------------------------------------------------------
def parse_formant_csv(text: str) -> list[tuple[str, int, int]]:
    """Parse vowel-formants.txt (Seg,F1,F2).

    Returns list of (seg, f1, f2) tuples.
    """
    rows: list[tuple[str, int, int]] = []
    # Strip BOM if present
    text = text.lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        seg = row["Seg"].strip()
        f1 = int(round(float(row["F1"])))
        f2 = int(round(float(row["F2"])))
        rows.append((seg, f1, f2))
    return rows


# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "other"

    TASK_NAME: str = "praat_analysis"
    VARIANT_NAME: str = "base"
    OS_TYPE: str = "windows"

    @property
    def spec(self) -> VariantSpec:
        return VARIANT_SPECS[self.VARIANT_NAME]

    @property
    def reference_textgrid_dir(self) -> str:
        return rf"{self.reference_dir}\textgrid"

    @property
    def reference_formants_dir(self) -> str:
        return rf"{self.reference_dir}\formants"

    @property
    def reference_plot_dir(self) -> str:
        return rf"{self.reference_dir}\plot"

    @property
    def task_description(self) -> str:
        spec = self.spec
        wav_list = ", ".join(f"`{s}.wav`" for s in spec.wav_stems)
        tg_list = ", ".join(f"`{s}.TextGrid`" for s in spec.wav_stems)
        formant_list = ", ".join(f"`formant-{v}.txt`" for v in spec.vowels)
        vowel_letters = ", ".join(spec.vowels)
        n_files = len(spec.wav_stems)

        if spec.vowel_identity_hidden:
            identity_block = (
                "- The vowel identity is NOT given — the agent must listen and "
                "determine which vowel each file contains.\n"
                f"- Possible vowels: {vowel_letters}."
            )
            annotate_hint = (
                "   - Determine the vowel identity by listening and inspecting "
                "the waveform / spectrogram\n"
                f"   - Possible labels: {vowel_letters}"
            )
        else:
            identity_block = (
                "- Each WAV filename already encodes the vowel identity "
                "(e.g. `a.wav` contains the vowel `a`).\n"
                f"- Vowels in this variant: {vowel_letters}."
            )
            annotate_hint = (
                "   - Use the filename stem as the vowel label for every "
                "non-silence interval in that file"
            )

        return f"""\
You are a phonetician performing a vowel formant analysis using Praat.

## Your Task
Perform a complete vowel formant analysis workflow using Praat, starting from
raw audio recordings and producing a publication-quality vowel formant chart.

Variant: `{self.VARIANT_NAME}`

## Input Files
- Located at: `{self.input_dir}`
- {n_files} WAV files: {wav_list}
- Each file contains 3 repetitions of the same vowel separated by silence
{identity_block}

## Steps

1. **Open Praat** on this computer.
   - If a task-local launcher is available, check `{self.software_dir}`

2. **Load all {n_files} WAV files** from the input directory into Praat.

3. **Create TextGrid annotations** for each WAV file:
   - Create a new TextGrid with a single IntervalTier named "vowel"
   - Inspect the waveform and spectrogram to identify each vowel token
{annotate_hint}
   - Annotate each vowel interval with its label
   - Leave silence intervals with empty text
   - Save each TextGrid to the output directory using the same filename stem
     (e.g. `{spec.wav_stems[0]}.TextGrid`)

4. **Extract formants** for each annotated vowel interval:
   - Use `To Formant (burg)` or equivalent Praat analysis
   - Divide each vowel interval into 10 equal chunks
   - Extract mean F1 and F2 per chunk
   - Save per-vowel formant tables as {formant_list}
     with tab-separated columns: File_name, Intv_id, Seg, t, t_m, F1, F2, F3, F4

5. **Compile combined CSV**:
   - Create `vowel-formants.txt` with header `Seg,F1,F2`
   - Include all extracted F1/F2 values (integer-rounded Hz) for all vowels

6. **Generate vowel formant scatter plot**:
   - X-axis: F2 (Hz), reversed (high to low, left to right)
   - Y-axis: F1 (Hz), reversed (high to low, top to bottom)
   - Each vowel type in a distinct color
   - Mean position of each vowel labeled with the vowel letter

7. **Save the plot** as `vowel-formants.pdf` in the output directory.

## Output
Save all results to: `{self.output_dir}`
- TextGrids: {tg_list}
- Per-vowel formant tables: {formant_list}
- `vowel-formants.txt` (combined CSV)
- `vowel-formants.pdf` (scatter plot)
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        spec = self.spec
        metadata.update(
            {
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "reference_textgrid_dir": self.reference_textgrid_dir,
                "reference_formants_dir": self.reference_formants_dir,
                "reference_plot_dir": self.reference_plot_dir,
                "task_name": self.TASK_NAME,
                "variant_name": self.VARIANT_NAME,
                "vowels": list(spec.vowels),
                "wav_stems": list(spec.wav_stems),
            }
        )
        return metadata


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
@cb.tasks_config(split="train")
def load():
    tasks = []
    for (tag,) in VARIANTS:
        cfg = TaskConfig(VARIANT_NAME=tag)
        tasks.append(
            cb.Task(
                description=cfg.task_description,
                metadata=cfg.to_metadata(),
                computer={
                    "provider": "computer",
                    "setup_config": {"os_type": cfg.OS_TYPE},
                },
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


# ---------------------------------------------------------------------------
# PDF to PNG conversion
# ---------------------------------------------------------------------------
def _pdf_to_png(pdf_bytes: bytes) -> bytes:
    """Convert the first page of a PDF to PNG bytes for LLM vision judge."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes
    except ImportError:
        logger.warning("PyMuPDF not available, sending raw PDF bytes")
        return pdf_bytes


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def _score_textgrids(
    agent_textgrids: dict[str, str],
    ref_textgrids: dict[str, str],
    wav_stems: tuple[str, ...] | list[str],
) -> float:
    """Checkpoint 1: compare TextGrid annotations.

    TextGrids are keyed by WAV stem. For each vowel interval: label must match
    exactly, and boundaries must be within 5% of the reference vowel duration.
    """
    total = 0
    passed = 0

    for stem in wav_stems:
        ref_text = ref_textgrids.get(stem)
        agent_text = agent_textgrids.get(stem)
        if not ref_text or not agent_text:
            total += 3
            continue

        ref_intervals = vowel_intervals(parse_textgrid(ref_text))
        agent_intervals = vowel_intervals(parse_textgrid(agent_text))

        for idx, ref_iv in enumerate(ref_intervals):
            total += 1
            ref_xmin, ref_xmax, ref_label = ref_iv
            ref_dur = ref_xmax - ref_xmin
            tolerance = 0.05 * ref_dur

            if idx >= len(agent_intervals):
                continue

            ag_xmin, ag_xmax, ag_label = agent_intervals[idx]

            if ag_label.strip().lower() != ref_label.strip().lower():
                continue

            if abs(ag_xmin - ref_xmin) <= tolerance and abs(ag_xmax - ref_xmax) <= tolerance:
                passed += 1

    return passed / total if total > 0 else 0.0


def _score_formants(
    agent_rows: list[tuple[str, int, int]],
    ref_rows: list[tuple[str, int, int]],
) -> float:
    """Checkpoint 2: compare formant values.

    Match rows by vowel label and order. F1 and F2 must each be within 5%
    of reference.
    """
    if not ref_rows:
        return 0.0

    # Group by vowel, preserving order
    agent_by_vowel: dict[str, list[tuple[int, int]]] = {}
    ref_by_vowel: dict[str, list[tuple[int, int]]] = {}

    for seg, f1, f2 in agent_rows:
        agent_by_vowel.setdefault(seg, []).append((f1, f2))
    for seg, f1, f2 in ref_rows:
        ref_by_vowel.setdefault(seg, []).append((f1, f2))

    total = 0
    passed = 0

    for vowel, ref_vals in ref_by_vowel.items():
        agent_vals = agent_by_vowel.get(vowel, [])
        for idx, (ref_f1, ref_f2) in enumerate(ref_vals):
            total += 1
            if idx >= len(agent_vals):
                continue
            ag_f1, ag_f2 = agent_vals[idx]

            f1_ok = abs(ag_f1 - ref_f1) <= 0.05 * abs(ref_f1) if ref_f1 != 0 else ag_f1 == 0
            f2_ok = abs(ag_f2 - ref_f2) <= 0.05 * abs(ref_f2) if ref_f2 != 0 else ag_f2 == 0

            if f1_ok and f2_ok:
                passed += 1

    return passed / total if total > 0 else 0.0


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score the agent's output across three checkpoints."""
    try:
        meta = task_cfg.metadata
        tag = meta["variant_name"]
        output_dir = meta["output_dir"]
        ref_textgrid_dir = meta["reference_textgrid_dir"]
        ref_formants_dir = meta["reference_formants_dir"]
        wav_stems: list[str] = list(meta["wav_stems"])
        vowels: list[str] = list(meta["vowels"])
        n_expected_tg = len(wav_stems)

        if not (await session.file_exists(output_dir) or await session.directory_exists(output_dir)):
            logger.warning(f"Output directory not found: {output_dir}")
            return [0.0]

        output_files = await session.list_dir(output_dir)

        async with EvaluationContext(
            task_tag=tag,
            mode="custom",
            output_dir=None,
            target_path=output_dir,
        ) as ctx:

            # ---------------------------------------------------------------
            # Gate 1: at least one TextGrid per input WAV
            # ---------------------------------------------------------------
            tg_files = [f for f in output_files if f.endswith(".TextGrid")]
            if len(tg_files) < n_expected_tg:
                logger.warning(
                    f"Only {len(tg_files)} TextGrid files — gate fail " f"(need {n_expected_tg})"
                )
                ctx.log_evaluation(
                    identifier="gate_textgrid",
                    score=0.0,
                    error=f"Only {len(tg_files)} TextGrid files (need {n_expected_tg})",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]
            ctx.log_evaluation(identifier="gate_textgrid", score=1.0)

            # ---------------------------------------------------------------
            # Gate 2: vowel-formants.txt with >= 50 rows
            # ---------------------------------------------------------------
            if "vowel-formants.txt" not in output_files:
                logger.warning("vowel-formants.txt missing — gate fail")
                ctx.log_evaluation(
                    identifier="gate_formants",
                    score=0.0,
                    error="vowel-formants.txt not found",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            agent_csv_path = rf"{output_dir}\vowel-formants.txt"
            agent_csv_text = await session.read_file(agent_csv_path)
            agent_formant_rows = parse_formant_csv(agent_csv_text)

            if len(agent_formant_rows) < 50:
                logger.warning(
                    f"vowel-formants.txt has {len(agent_formant_rows)} rows "
                    f"(need >=50) — gate fail"
                )
                ctx.log_evaluation(
                    identifier="gate_formants",
                    score=0.0,
                    error=f"Only {len(agent_formant_rows)} data rows (need 50)",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]
            ctx.log_evaluation(identifier="gate_formants", score=1.0)

            # ---------------------------------------------------------------
            # Gate 3: vowel-formants.pdf exists
            # ---------------------------------------------------------------
            if "vowel-formants.pdf" not in output_files:
                logger.warning("vowel-formants.pdf missing — gate fail")
                ctx.log_evaluation(
                    identifier="gate_plot",
                    score=0.0,
                    error="vowel-formants.pdf not found",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]
            ctx.log_evaluation(identifier="gate_plot", score=1.0)

            # ---------------------------------------------------------------
            # Checkpoint 1: TextGrid annotation (0.35)
            # ---------------------------------------------------------------
            agent_tg: dict[str, str] = {}
            ref_tg: dict[str, str] = {}

            for stem in wav_stems:
                tg_name = f"{stem}.TextGrid"
                agent_tg_path = rf"{output_dir}\{tg_name}"
                ref_tg_path = rf"{ref_textgrid_dir}\{tg_name}"

                if (await session.file_exists(agent_tg_path) or await session.directory_exists(agent_tg_path)):
                    agent_tg[stem] = _decode_textgrid_bytes(await session.read_bytes(agent_tg_path))
                if (await session.file_exists(ref_tg_path) or await session.directory_exists(ref_tg_path)):
                    ref_tg[stem] = _decode_textgrid_bytes(await session.read_bytes(ref_tg_path))

            ckpt1_raw = _score_textgrids(agent_tg, ref_tg, wav_stems)
            ckpt1_score = W_TEXTGRID * ckpt1_raw
            ctx.add_score(ckpt1_score)
            ctx.log_evaluation(
                identifier="ckpt1_textgrid",
                score=ckpt1_score,
                raw_score=round(ckpt1_raw, 4),
            )
            logger.info(f"Ckpt 1 TextGrid: {ckpt1_raw:.4f} " f"(weighted {ckpt1_score:.4f})")

            # ---------------------------------------------------------------
            # Checkpoint 2: Formant extraction (0.40)
            # ---------------------------------------------------------------
            ref_csv_path = rf"{ref_formants_dir}\vowel-formants.txt"
            ref_csv_text = await session.read_file(ref_csv_path)
            ref_formant_rows = parse_formant_csv(ref_csv_text)

            ckpt2_raw = _score_formants(agent_formant_rows, ref_formant_rows)
            ckpt2_score = W_FORMANT * ckpt2_raw
            ctx.add_score(ckpt2_score)
            ctx.log_evaluation(
                identifier="ckpt2_formants",
                score=ckpt2_score,
                raw_score=round(ckpt2_raw, 4),
                agent_rows=len(agent_formant_rows),
                ref_rows=len(ref_formant_rows),
            )
            logger.info(f"Ckpt 2 Formants: {ckpt2_raw:.4f} " f"(weighted {ckpt2_score:.4f})")

            # ---------------------------------------------------------------
            # Checkpoint 3: Vowel plot (0.25) — LLM vision judge
            # ---------------------------------------------------------------
            plot_path = rf"{output_dir}\vowel-formants.pdf"
            plot_bytes = await session.read_bytes(plot_path)

            # Convert PDF to PNG for the LLM vision judge
            plot_image_bytes = _pdf_to_png(plot_bytes)

            vowels_joined = ", ".join(vowels)
            plot_prompt = f"""\
You are evaluating a vowel formant chart (scatter plot).

Examine this image and check EACH of these 4 criteria independently:

1. **all_vowels_present**: Are all {len(vowels)} vowels ({vowels_joined}) \
represented as data points in the plot?
2. **vowels_labeled**: Is each vowel type labeled with its letter/symbol \
({vowels_joined}) somewhere on or near its data cluster?
3. **visually_differentiated**: Are the vowel types visually differentiated \
by distinct colors, markers, or both?
4. **axes_labeled**: Are the axes labeled as F1 and F2 (in any order)?

Return a JSON object with exactly these keys, each with value true or false:
{{"all_vowels_present": ..., "vowels_labeled": ..., \
"visually_differentiated": ..., "axes_labeled": ...}}
"""
            try:
                judge_result = await llm_vision_json_judge(
                    prompt=plot_prompt,
                    image_bytes_list=[plot_image_bytes],
                    max_tokens=200,
                    temperature=0,
                )
                criteria_met = sum(
                    1
                    for key in [
                        "all_vowels_present",
                        "vowels_labeled",
                        "visually_differentiated",
                        "axes_labeled",
                    ]
                    if judge_result.get(key, False) is True
                )
                ckpt3_raw = criteria_met / 4.0
            except Exception as e:
                logger.warning(f"LLM vision judge failed: {e}")
                ckpt3_raw = 0.0
                judge_result = {"error": str(e)}

            ckpt3_score = W_PLOT * ckpt3_raw
            ctx.add_score(ckpt3_score)
            ctx.log_evaluation(
                identifier="ckpt3_plot",
                score=ckpt3_score,
                raw_score=round(ckpt3_raw, 4),
                judge_result=judge_result,
            )
            logger.info(f"Ckpt 3 Plot: {ckpt3_raw:.4f} " f"(weighted {ckpt3_score:.4f})")

            # ---------------------------------------------------------------
            # Final score
            # ---------------------------------------------------------------
            ctx.finalize(num_output_files=len(output_files))
            total = ctx.total_score
            logger.info(
                f"Final score: {total:.4f} "
                f"(textgrid={ckpt1_score:.3f} formants={ckpt2_score:.3f} "
                f"plot={ckpt3_score:.3f})"
            )
            return [total]

    except Exception as e:
        logger.error(f"Evaluation error: {e}")

    return [0.0]
