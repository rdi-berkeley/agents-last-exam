"""Chinese e-discovery evidence-index task."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

logger = logging.getLogger(__name__)

TASK_DIR_LOCAL = Path(__file__).resolve().parent
EXTRACT_SCRIPT = (TASK_DIR_LOCAL / "scripts" / "score_evidence_index.py").read_text(
    encoding="utf-8"
)

RUBRIC = """\
证据目录 Evaluation Rubric (总分 20 分)

1. 证据分组合理 (4分)
证据目录应按证明事项分组，至少包括：事实劳动关系组、事故发生/工伤构成组、医疗伤情及费用组、工伤认定/鉴定/仲裁程序组、已付款或赔偿协商组。
- 分组完整且逻辑清楚 → 4分
- 缺少1个关键分组 → 3分
- 分组较混乱但大体能看出证明事项 → 2分
- 仅按页码或证据出现顺序罗列，未按证明事项分组 → 1分
- 完全无分组 → 0分

2. 证据目录栏目完整 (3分)
证据目录应包含：分组、序号、证据名称、证据来源、页码、证明目的、备注。
- 栏目完整 → 3分
- 缺少1个非关键栏目 → 2分
- 缺少"证明目的"或"页码"等关键栏目 → 最高1分
- 没有形成表格化证据目录 → 0分

3. 核心证据覆盖充分 (4分)
应覆盖本案核心证据：证据1-9、11-15（共14项）。
- 覆盖12项以上且无重大遗漏 → 4分
- 覆盖9-11项 → 3分
- 覆盖6-8项 → 2分
- 覆盖不足6项 → 1分
- 大量遗漏工伤认定、鉴定结论、仲裁裁决、事故证据等核心材料 → 0分

4. 事实劳动关系证明目的准确 (3分)
对证据1-5、10、14的证明目的应写明：证明周某进入项目施工现场，接受青岚机电或其项目管理人员管理，从事安装工作，存在考勤、派工、工资或生活费支付等事实，从而证明双方存在事实劳动关系。
- 证明目的具体、准确 → 3分
- 只写"证明劳动关系"但未说明具体证明内容 → 2分
- 把社保、个税缺失错误写成否定劳动关系的核心依据 → 1分
- 未设置劳动关系证明目的 → 0分

5. 事故发生与工伤构成证明目的准确 (3分)
对证据4-8、12的证明目的应写明：证明周某于2025年5月14日在宁州云仓项目B区B2线执行安装支架工作时受伤，事故发生在工作时间、工作地点、因工作原因，并已被认定为工伤。
- 证明目的具体、准确 → 3分
- 只笼统写"证明受伤事实" → 2分
- 错误写成"叉车撞击事实已经确定"但未说明证据限制 → 1分
- 未设置事故发生或工伤构成证明目的 → 0分

6. 医疗伤情、费用和伤残等级证明目的准确 (2分)
对证据8、9、13的证明目的应写明：证明周某左踝骨折、住院治疗、医疗费用支出、部分费用已垫付、劳动功能障碍等级为九级且无生活自理障碍。
- 三类证明目的均准确 → 2分
- 遗漏费用或伤残等级之一 → 1分
- 未列医疗伤情或费用证明目的 → 0分

7. 程序材料证明目的准确 (2分)
对证据11-14的证明目的应写明：证明周某已启动并完成工伤认定程序，取得工伤认定决定和劳动能力鉴定结论，并已经过劳动仲裁前置程序，具备进一步起诉的程序基础。
- 证明目的完整 → 2分
- 只列工伤认定或鉴定，遗漏仲裁前置 → 1分
- 完全未列程序证明目的 → 0分

注意: 原来 rubric 中第8项(已付款/协商/抵扣 1分) 和第9项(弱相关/烟雾弹 1分) 合并入上述评分, 不再单独列出。LLM judge 应综合考量, 总分仍为20分。如果证据16被正确标记为弱相关/背景材料而非核心证据, 在总分中酌情加1分; 如果已付款和协商事项在相关证据的证明目的中有准确描述, 酌情加1分。
"""

LLM_JUDGE_PROMPT = """\
You are an expert legal document evaluator. Score the following Chinese evidence index workbook against the provided rubric.

## Rubric
{rubric}

## Workbook Content (extracted from agent's output)
Headers: {headers}

Data rows:
{rows_text}

## Instructions
1. Evaluate the workbook against each rubric dimension.
2. For each dimension, provide a brief justification and a score.
3. Output ONLY a JSON object with this exact structure (no markdown, no extra text):

{{
  "dimensions": {{
    "grouping": {{"score": <0-4>, "reason": "<brief justification>"}},
    "columns": {{"score": <0-3>, "reason": "<brief justification>"}},
    "core_coverage": {{"score": <0-4>, "reason": "<brief justification>"}},
    "labor_relation": {{"score": <0-3>, "reason": "<brief justification>"}},
    "accident_injury": {{"score": <0-3>, "reason": "<brief justification>"}},
    "medical_disability": {{"score": <0-2>, "reason": "<brief justification>"}},
    "procedure": {{"score": <0-2>, "reason": "<brief justification>"}}
  }},
  "bonus_weak_evidence": <0 or 1>,
  "bonus_payment_negotiation": <0 or 1>,
  "total_raw": <sum of all scores>,
  "total_capped": <min(total_raw, 20)>,
  "final_score": <1.0 if total_capped >= 19, else total_capped / 20.0>
}}
"""


def _load_openai_key() -> str:
    try:  # load secret/eval_time/*.env so the OpenAI judge key is present
        from tasks.utils.evaluation import load_eval_env

        load_eval_env()
    except Exception:
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip().strip("\"'")
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY"):
                _, _, val = line.partition("=")
                return val.strip().strip("\"'")
    raise RuntimeError("OPENAI_API_KEY not found in environment or .env")


async def _llm_judge(workbook_content: dict) -> dict:
    """Call OpenAI GPT-4o to score the workbook content."""
    import openai

    headers = workbook_content.get("headers", [])
    rows = workbook_content.get("rows", [])

    rows_text_parts = []
    for i, row in enumerate(rows, 1):
        parts = [f"  {k}: {v}" for k, v in row.items() if v]
        rows_text_parts.append(f"[Row {i}]\n" + "\n".join(parts))
    rows_text = "\n\n".join(rows_text_parts)

    prompt = LLM_JUDGE_PROMPT.format(
        rubric=RUBRIC,
        headers=json.dumps(headers, ensure_ascii=False),
        rows_text=rows_text,
    )

    client = openai.OpenAI(api_key=_load_openai_key())
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000,
    )

    raw_text = response.choices[0].message.content.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(raw_text)


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "legal"
    TASK_NAME: str = "e_discovery_chinese"
    VARIANT_NAME: str = "base"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/证据目录.xlsx"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

## Task Directory
`{self.task_dir}`

## Visible Inputs
- Client statement: `{self.input_dir}/Input.docx`
- Evidence images: `{self.input_dir}/Evidence/`
- File manifest: `{self.input_dir}/manifest.json`
- Detailed task instructions: `{self.input_dir}/task_instructions.md`

## Optional Local Software
- LibreOffice entry point: `{self.task_dir}/software/libreoffice`
- Tesseract OCR entry point: `{self.task_dir}/software/tesseract`
- Python entry point: `{self.task_dir}/software/python`

## Your Task
Review the Chinese client statement and the 16 Chinese evidence images, then
create a Chinese evidence index workbook.

Use the input facts only. Do not use external legal databases, commercial
Chinese-law platforms, or online OCR services.

Save your final workbook as:
`{self.output_file}`

The workbook must contain an evidence-index table with columns for grouping,
serial number, evidence name, evidence form/type, evidence source, page number,
proof purpose, and remarks.

Do not write outputs outside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update({"output_file": self.output_file})
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


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    eval_dir = f"/tmp/agenthle_eval/{meta['task_name']}"
    extract_script = f"{eval_dir}/score_evidence_index.py"
    try:
        await session.interface.create_dir(eval_dir)
        await session.write_file(extract_script, EXTRACT_SCRIPT)
        result = await session.run_command(
            f'python "{extract_script}" --output "{meta["remote_output_dir"]}"'
        )
        stdout = result.get("stdout", "")
        if result.get("return_code", 1) != 0:
            logger.warning("extract script failed: %s", result.get("stderr", ""))
            return [0.0]

        workbook_content = json.loads(stdout)
        if not workbook_content.get("ok"):
            logger.warning("workbook extraction failed: %s", workbook_content.get("error"))
            return [0.0]

        judge_result = await _llm_judge(workbook_content)
        logger.info("LLM judge result: %s", json.dumps(judge_result, ensure_ascii=False))
        score = float(judge_result.get("final_score", 0.0))
        return [score]
    except Exception as exc:
        logger.error("evaluation failed: %s", exc)
        return [0.0]


if __name__ == "__main__":
    print(config.task_description)
