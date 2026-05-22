"""

Expert Note:
`ar_full_300` evaluates full-flow extraction on a mid-size corpus.

What is truly hard in this benchmark:
- File coverage degrades with scale; missed PDFs propagate to many missing rows.
- Extraction consistency across heterogeneous reports is difficult.
- Rule violations (wrongly included/excluded rows) are common.

Why this matters:
Mid-scale performance is a strong predictor of production readiness.

Scale Reality:
- Task scope includes 300 report files.
- Many files are hundreds of pages with inconsistent table and heading structures.
- Evidence lookup is effectively needle-in-a-haystack document search before row extraction.
"""

import logging
from dataclasses import dataclass
from pathlib import PureWindowsPath

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from utils.finance_evaluation import (verify_dataset_samples_remote,
                                      verify_files_remote, win_join)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)


def _uses_fixture_only_scoring(output_dir: str) -> bool:
    return PureWindowsPath(output_dir).name.lower() in {"output_test_pos", "output_test_neg"}


@dataclass
class TaskConfig(GeneralTaskConfig):
    VARIANT_NAME: str = "base"
    DOMAIN_NAME: str = "business_finance"

    TASK_NAME: str = "ar_full_300"

    @property
    def file_list_url(self) -> str:
        return win_join(self.task_dir, "input", "file_list.txt")

    @property
    def download_url(self) -> str:
        return win_join(self.remote_output_dir, "downloads")

    @property
    def task_description(self) -> str:
        return f"""
Task Type:
- Full pipeline (download + parse + structure output), with file verification.

Goal:
- Download all reports listed in {self.file_list_url} from East Money (东方财富).
- Extract core technical personnel data.
- Output one consolidated Excel file.

Input Files:
1) {self.file_list_url}
   - UTF-8 text file, one PDF filename per line.
   - Example line: `688001_某公司_2023年年报.pdf`

Software Provided:
- Browser launcher: `software\\open_chrome.bat`
  - Full path on this VM: `{self.task_dir}\\software\\open_chrome.bat`
  - Opens the task-local Chrome entry point on a stable blank page for East Money downloads.
- Python launcher: `software\\python.bat`
  - Full path on this VM: `{self.task_dir}\\software\\python.bat`
  - Runs the preinstalled Python runtime with UTF-8 mode enabled.

Output File:
1) {self.remote_output_dir}\\final_dataset.xlsx
   - One sheet.
   - One row per person.
   - Required columns (exact names):
     识别码, 证券代码, 股票简称, 姓名, 性别, 年龄, 国籍, 最高学历, 简历,
     2019薪酬, 2020薪酬, 2021薪酬, 2022薪酬, 2023薪酬, 2024薪酬,
     2019持股, 2020持股, 2021持股, 2022持股, 2023持股, 2024持股.

Data Rules:
- `识别码` (Identifier) format:
  - Build as: `股票简称 + 姓名` (no separator).
  - Use `股票简称` for the company part (do NOT use `证券代码`).
  - Example: if company is `华兴源创` and person is `曹振宇`, then `识别码` is `华兴源创曹振宇`.
  - Example: if company is `天准科技` and person is `徐一华`, then `识别码` is `天准科技徐一华`.
- Try to fill each required field from evidence as completely as possible.
- Use NaN only when the value cannot be reliably found.
- For numeric output values, keep at least 3 decimal places (0.001 precision) when applicable.
- `最高学历` code mapping:
  - 1 = PhD
  - 2 = Master
  - 3 = Bachelor
  - 4 = Associate
  - 5 = Other / Undisclosed
- Output format requirement for `最高学历`:
  - Use numeric code values only (1/2/3/4/5).
  - `1` and `1.0` are treated as the same numeric value.
  - Text labels like `博士` / `硕士` are not accepted in this column.

Scoring:
- Full score requires all conditions below to be satisfied:
  1) Every required PDF in `file_list.txt` is correctly downloaded.
  2) `final_dataset.xlsx` matches the required schema exactly.
  3) Extracted values are accurate on hidden verification samples.
"""

    def to_metadata(self) -> dict:
        md = super().to_metadata()
        md.update({"file_list_url": self.file_list_url, "download_url": self.download_url})
        return md


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


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    outdir = task_cfg.metadata["remote_output_dir"]
    refdir = task_cfg.metadata["reference_dir"]
    try:
        data_score = await verify_dataset_samples_remote(session, outdir, refdir)
        if _uses_fixture_only_scoring(outdir):
            final = data_score / 50.0
            logger.info("Final: %.4f (fixture-only sample replay, data=%.2f/50)", final, data_score)
            return [final]

        file_score = await verify_files_remote(session, outdir, refdir)
        final = (file_score + data_score) / 100.0
        logger.info("Final: %.4f (file=%.2f/50, data=%.2f/50)", final, file_score, data_score)
        return [final]
    except Exception as e:
        logger.warning("Evaluation error: %s", e)
        return [0.0]
