"""Genome-browser SVG task for K562 regulatory loci."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.life_sciences.tp53_locus_variant_histone_browser_svg.scripts.score_svg import (
    score_svg_bytes,
)
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

OUTPUT_FILE = "output.svg"
VCF_NAME = "ENCFF960SSF.vcf.gz"
BIGWIG_NAME = "wgEncodeBroadHistoneK562H3k27acStdSig.bigWig"

VARIANTS = [
    ("base", "TP53", "chr17", 7571651, 7590910),
    ("variant_1", "GATA1", "chrX", 48644981, 48652717),
    ("variant_2", "CDKN1A", "chr6", 36644313, 36655116),
    ("variant_3", "NFE2", "chr12", 54685890, 54694821),
    ("variant_4", "BCL11A", "chr2", 60678301, 60780633),
]


class GenomeBrowserSvgConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "life_sciences"
    TASK_NAME: str = "tp53_locus_variant_histone_browser_svg"

    def __init__(
        self,
        *,
        VARIANT_NAME: str = "base",
        GENE: str = "TP53",
        CHROM: str = "chr17",
        START: int = 7571651,
        END: int = 7590910,
    ) -> None:
        super().__init__(
            DOMAIN_NAME=self.DOMAIN_NAME,
            TASK_NAME=self.TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )
        self.GENE = GENE
        self.CHROM = CHROM
        self.START = START
        self.END = END

    @property
    def task_config_path(self) -> str:
        return f"{self.input_dir}/task_config.json"

    @property
    def vcf_path(self) -> str:
        return f"{self.input_dir}/{VCF_NAME}"

    @property
    def bigwig_path(self) -> str:
        return f"{self.input_dir}/{BIGWIG_NAME}"

    @property
    def output_svg_path(self) -> str:
        return f"{self.remote_output_dir}/{OUTPUT_FILE}"

    @property
    def reference_json_path(self) -> str:
        return f"{self.reference_dir}/expected_view.json"

    @property
    def launcher_path(self) -> str:
        return f"{self.software_dir}/open_ucsc_browser.sh"

    @property
    def task_description(self) -> str:
        locus = f"{self.CHROM}:{self.START}-{self.END}"
        return f"""You are a bioinformatics analyst working on a K562 regulatory genomics visualization task.

Task directory:
- `{self.task_dir}`

Goal:
- Create a genome-browser SVG view for the hg19 {self.GENE} locus: `{locus}`.
- The view must include both the provided K562 structural variant VCF track and the K562 H3K27ac BigWig signal track at the same time.

Input files:
- Variant calls: `{self.vcf_path}`
- H3K27ac signal: `{self.bigwig_path}`
- Machine-readable task config: `{self.task_config_path}`

Available software:
- You may use the UCSC Genome Browser, IGV, WashU Epigenome Browser, or another browser/tool capable of loading VCF and BigWig tracks on hg19 and exporting SVG.
- A helper that opens the UCSC hg19 locus page is available with:
  `bash {self.launcher_path}`

Output:
- Save exactly one SVG file to `{self.output_svg_path}`.
- The SVG must be valid XML with an `<svg>` root.
- The SVG should visibly show the hg19 locus labels, the VCF/variant track, and the K562 H3K27ac signal track.
- Do not save the final answer under any other filename.

Signal track serialization:
- Encode the actual full-window H3K27ac signal as one x-monotone `<polyline>` or straight-segment `<path>` (M/L/H/V commands, absolute or relative, with optional final Z), a filled `<polygon>` following that contour and closing to a horizontal baseline, or aligned `<rect>` bars sharing a horizontal baseline and the same explicit `stroke` value (`none` is suitable). Bar fill colors may vary. Do not split one signal contour across disconnected paths.
- Choose sampling resolution and plotting scale that preserve the input signal: at least 12 contour samples or 12 bars, spanning at least 400 units horizontally and 30 vertically, with at least 8 signal heights distinguishable at 0.1-unit precision. These are SVG user units after element/group transforms, not genomic coordinates. Do not invent or alter signal values to meet these presentation requirements.
- Store finite, unitless numeric geometry directly in `points`, `d`, or rect `x`/`y`/`width`/`height` attributes under the root `<svg>` or nested `<g>` elements. Numeric SVG `transform` attributes are allowed if the resulting contour remains x-monotone and bars remain axis-aligned. Bake nested SVG viewport mappings into the signal coordinates; a root `viewBox` is allowed.
- Make the signal self-contained and visible using presentation attributes or literal inline `style` declarations on the shapes or their groups. Convert signal encodings that depend on Bezier/arc commands (C/S/Q/T/A), `<use>`/`<symbol>`/`<defs>`, raster images, CSS stylesheets/classes/variables, scripts, filters, masks, clipping, or occlusion into the actual geometry described above. Do not rely on those features to construct or hide part of the signal contour.
- These serialization restrictions apply only to the H3K27ac signal and its presentation, not to unrelated text, fonts, labels, or other SVG content. Browser/tool choice and GUI workflow remain unrestricted; you may post-process an export to serialize the signal without changing its scientific content.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "gene": self.GENE,
                "chrom": self.CHROM,
                "start": self.START,
                "end": self.END,
                "task_config_path": self.task_config_path,
                "vcf_path": self.vcf_path,
                "bigwig_path": self.bigwig_path,
                "output_svg_path": self.output_svg_path,
                "reference_json_path": self.reference_json_path,
                "launcher_path": self.launcher_path,
            }
        )
        return metadata


def _cfg_for_variant(spec: tuple[str, str, str, int, int]) -> GenomeBrowserSvgConfig:
    name, gene, chrom, start, end = spec
    return GenomeBrowserSvgConfig(
        VARIANT_NAME=name,
        GENE=gene,
        CHROM=chrom,
        START=start,
        END=end,
    )


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=_cfg_for_variant(spec).task_description,
            metadata=_cfg_for_variant(spec).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
        for spec in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        output_bytes = await session.read_bytes(meta["output_svg_path"])
    except Exception as exc:
        logger.error("Failed to read %s: %s", meta["output_svg_path"], exc)
        return [0.0]

    try:
        reference_bytes = await session.read_bytes(meta["reference_json_path"])
        reference = json.loads(reference_bytes.decode("utf-8"))
    except Exception as exc:
        logger.error("Failed to read reference %s: %s", meta["reference_json_path"], exc)
        return [0.0]

    result = score_svg_bytes(output_bytes, reference)
    logger.info(
        "[%s] final_score=%.4f checks=%s notes=%s",
        meta["variant_name"],
        result.score,
        result.checks,
        result.notes,
    )
    return [result.score]


if __name__ == "__main__":
    for task in load():
        print(task.description)
