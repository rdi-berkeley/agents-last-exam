"""Shared evaluators for finance tasks."""

from __future__ import annotations

import json
import logging
import math
from io import BytesIO
from pathlib import PureWindowsPath
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

METRICS_OUTPUT_FILENAME = "final_metrics.xlsx"
METRICS_REFERENCE_FILENAME = "expected_metrics.xlsx"
METRICS_KEY_COLUMN = "证券代码"
NUMERIC_TOLERANCE = 0.01
CELL_PENALTY = 20
MAX_SCORE = 100.0

DATASET_OUTPUT_FILENAME = "final_dataset.xlsx"
DATASET_KEY_COLUMN = "识别码"
EVAL_TMP_ROOT = r"C:\Users\User\AppData\Local\Temp\agenthle_eval"


def win_join(*parts: str) -> str:
    return str(PureWindowsPath(*parts))


def _normalize_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return str(value).strip()
    return str(value).strip()


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not pd.isna(value)


def _cells_match(expected: Any, actual: Any) -> bool:
    if pd.isna(expected) and pd.isna(actual):
        return True
    if _is_numeric(expected) and _is_numeric(actual):
        return abs(float(expected) - float(actual)) < NUMERIC_TOLERANCE
    return str(expected).strip() == str(actual).strip()


def _read_workbook(data: bytes, label: str) -> pd.DataFrame:
    if not data:
        raise ValueError(f"{label} workbook is empty")
    return pd.read_excel(BytesIO(data))


def _score_metrics_workbook(actual_df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
    if list(actual_df.columns) != list(expected_df.columns):
        logger.warning(
            "finance eval header mismatch: actual=%s expected=%s",
            list(actual_df.columns),
            list(expected_df.columns),
        )
        return 0.0
    if METRICS_KEY_COLUMN not in actual_df.columns:
        logger.warning("finance eval missing key column %s", METRICS_KEY_COLUMN)
        return 0.0

    expected = expected_df.copy()
    actual = actual_df.copy()
    expected["_row_key"] = expected[METRICS_KEY_COLUMN].map(_normalize_key)
    actual["_row_key"] = actual[METRICS_KEY_COLUMN].map(_normalize_key)

    if not expected["_row_key"].all() or not actual["_row_key"].all():
        logger.warning("finance eval encountered empty row keys")
        return 0.0
    if expected["_row_key"].duplicated().any() or actual["_row_key"].duplicated().any():
        logger.warning("finance eval encountered duplicate row keys")
        return 0.0

    actual_rows = actual.set_index("_row_key")
    wrong_cells = 0
    checked_cells = 0

    for _, expected_row in expected.iterrows():
        row_key = expected_row["_row_key"]
        if row_key not in actual_rows.index:
            wrong_cells += len(expected_df.columns)
            checked_cells += len(expected_df.columns)
            continue

        actual_row = actual_rows.loc[row_key]
        for column in expected_df.columns:
            checked_cells += 1
            if not _cells_match(expected_row[column], actual_row[column]):
                wrong_cells += 1

    logger.info("finance eval checked %s cells with %s mismatches", checked_cells, wrong_cells)
    return max(0.0, MAX_SCORE - wrong_cells * CELL_PENALTY)


def _last_json(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in str(stdout).splitlines() if line.strip()]):
        try:
            return json.loads(line)
        except Exception:
            continue
    return {}


async def _read_json_remote(session, path: str) -> Any:
    return json.loads(await session.read_file(path))


async def verify_metrics_table_remote(session, output_dir: str, reference_dir: str) -> float:
    output_path = win_join(output_dir, METRICS_OUTPUT_FILENAME)
    reference_path = win_join(reference_dir, METRICS_REFERENCE_FILENAME)

    if not (await session.file_exists(output_path) or await session.directory_exists(output_path)):
        logger.warning("finance eval missing output workbook: %s", output_path)
        return 0.0
    if not (await session.file_exists(reference_path) or await session.directory_exists(reference_path)):
        logger.warning("finance eval missing reference workbook: %s", reference_path)
        return 0.0

    try:
        actual_bytes = await session.read_bytes(output_path)
        expected_bytes = await session.read_bytes(reference_path)
        actual_df = _read_workbook(actual_bytes, "actual")
        expected_df = _read_workbook(expected_bytes, "reference")
    except Exception as exc:
        logger.warning("finance eval failed to load workbooks: %s", exc)
        return 0.0

    try:
        return _score_metrics_workbook(actual_df=actual_df, expected_df=expected_df)
    except Exception as exc:
        logger.warning("finance eval failed to score workbooks: %s", exc)
        return 0.0


async def verify_files_remote(
    session,
    output_dir: str,
    reference_dir: str,
    task_name: str = "finance_task",
) -> float:
    """Return the 50-point file score for ar_full-style tasks."""

    manifest = win_join(reference_dir, "file_manifest.json")
    downloads = win_join(output_dir, "downloads")
    eval_tmp_dir = win_join(EVAL_TMP_ROOT, task_name)
    ps1_path = win_join(eval_tmp_dir, "verify_files_md5.ps1")

    ps1 = r"""param(
  [Parameter(Mandatory=$true)][string]$ManifestPath,
  [Parameter(Mandatory=$true)][string]$DownloadsDir
)

$max = 20
$throttle = [Math]::Min(8, [Environment]::ProcessorCount)
if ($throttle -lt 1) { $throttle = 1 }

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$m = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json

$total = ($m.PSObject.Properties | Measure-Object).Count
$correct = 0

$missing = 0; $size_bad = 0; $hash_bad = 0; $err = 0
$missing_ex = @(); $size_ex = @(); $hash_ex = @(); $err_ex = @()
$todo = New-Object System.Collections.Generic.List[object]

foreach ($p in $m.PSObject.Properties) {
  $fn = $p.Name
  $exp = $p.Value
  $fp = Join-Path $DownloadsDir $fn

  if (!(Test-Path -LiteralPath $fp)) {
    $missing++
    if ($missing_ex.Count -lt $max) { $missing_ex += $fn }
    continue
  }

  try {
    $size = (Get-Item -LiteralPath $fp).Length
    $emin = [double]$exp.size * 0.95
    $emax = [double]$exp.size * 1.05
    if ($size -lt $emin -or $size -gt $emax) {
      $size_bad++
      if ($size_ex.Count -lt $max) { $size_ex += $fn }
      continue
    }

    $todo.Add([pscustomobject]@{ fn=$fn; fp=$fp; exp_hash=($exp.hash.ToLower()) })
  } catch {
    $err++
    if ($err_ex.Count -lt $max) { $err_ex += $fn }
    continue
  }
}

$pool = [RunspaceFactory]::CreateRunspacePool(1, $throttle)
$pool.Open()

$tasks = New-Object System.Collections.Generic.List[object]

$scriptBlock = {
  param($path)
  try {
    (Get-FileHash -LiteralPath $path -Algorithm MD5).Hash.ToLower()
  } catch {
    ""
  }
}

foreach ($item in $todo) {
  $ps = [PowerShell]::Create()
  $ps.RunspacePool = $pool
  [void]$ps.AddScript($scriptBlock).AddArgument($item.fp)
  $handle = $ps.BeginInvoke()
  $tasks.Add([pscustomobject]@{
    ps=$ps; handle=$handle; fn=$item.fn; exp_hash=$item.exp_hash
  })
}

foreach ($t in $tasks) {
  try {
    $res = $t.ps.EndInvoke($t.handle)
    $t.ps.Dispose()
    $hash = ""
    if ($res -is [System.Array] -and $res.Length -gt 0) { $hash = [string]$res[0] }
    elseif ($res) { $hash = [string]$res }

    if ([string]::IsNullOrEmpty($hash)) {
      $err++
      if ($err_ex.Count -lt $max) { $err_ex += $t.fn }
      continue
    }

    if ($hash -eq $t.exp_hash) {
      $correct++
    } else {
      $hash_bad++
      if ($hash_ex.Count -lt $max) { $hash_ex += $t.fn }
    }
  } catch {
    $err++
    if ($err_ex.Count -lt $max) { $err_ex += $t.fn }
    continue
  }
}

$pool.Close()
$pool.Dispose()
$sw.Stop()

@{
  total=$total; correct=$correct; throttle=$throttle; hashed_count=$todo.Count; elapsed_sec=[Math]::Round($sw.Elapsed.TotalSeconds, 3);
  missing_count=$missing; size_mismatch_count=$size_bad; hash_mismatch_count=$hash_bad; error_count=$err;
  missing_examples=$missing_ex; size_mismatch_examples=$size_ex; hash_mismatch_examples=$hash_ex; error_examples=$err_ex
} | ConvertTo-Json -Compress
"""

    try:
        await session.interface.create_dir(eval_tmp_dir)
        await session.write_file(ps1_path, ps1)
        result = await session.run_command(
            (
                f'powershell -NoProfile -ExecutionPolicy Bypass -File "{ps1_path}" '
                f'-ManifestPath "{manifest}" -DownloadsDir "{downloads}"'
            ),
            check=False,
        )
        stats = _last_json(result.get("stdout", ""))
        if not stats:
            logger.warning("file MD5 check produced no parsable JSON")
            logger.info("raw stdout (first 500 chars): %s", str(result.get("stdout", ""))[:500])
            logger.info("raw stderr (first 500 chars): %s", str(result.get("stderr", ""))[:500])
            return 0.0

        total = int(stats.get("total", 0))
        correct = int(stats.get("correct", 0))
        wrong = max(0, total - correct)
        score = float(max(0, 50 - 5 * wrong))

        logger.info(
            "file check: total=%s correct=%s wrong=%s score=%.2f/50",
            total,
            correct,
            wrong,
            score,
        )
        logger.info(
            "file failures: missing=%s size=%s hash=%s error=%s",
            stats.get("missing_count", 0),
            stats.get("size_mismatch_count", 0),
            stats.get("hash_mismatch_count", 0),
            stats.get("error_count", 0),
        )
        for key, label in [
            ("missing_examples", "Missing examples"),
            ("size_mismatch_examples", "Size mismatch examples"),
            ("hash_mismatch_examples", "Hash mismatch examples"),
            ("error_examples", "Error examples"),
        ]:
            examples = stats.get(key, [])
            if isinstance(examples, list) and examples:
                logger.warning("%s (up to 20): %s", label, ", ".join(map(str, examples)))

        return score
    except Exception as exc:
        logger.warning("MD5 verification failed: %s", exc)
        return 0.0


async def verify_dataset_samples_remote(session, output_dir: str, reference_dir: str) -> float:
    """Return the 50-point sample score for ar_full-style tasks."""

    samples_path = win_join(reference_dir, "data_samples.json")
    table_path = win_join(output_dir, DATASET_OUTPUT_FILENAME)

    try:
        samples = await _read_json_remote(session, samples_path)
    except Exception as exc:
        logger.warning("failed to load data_samples.json: %s", exc)
        return 0.0

    if not (await session.file_exists(table_path) or await session.directory_exists(table_path)):
        logger.warning("missing output workbook: %s", table_path)
        return 0.0

    try:
        df = pd.read_excel(BytesIO(await session.read_bytes(table_path)))
    except Exception as exc:
        logger.warning("failed to read %s: %s", DATASET_OUTPUT_FILENAME, exc)
        return 0.0

    if DATASET_KEY_COLUMN not in df.columns:
        logger.warning("missing required column: %s", DATASET_KEY_COLUMN)
        return 0.0

    try:
        df_idx = df.set_index(DATASET_KEY_COLUMN, drop=False)
    except Exception:
        df_idx = df

    correct = 0
    total = len(samples)
    mismatches: list[dict[str, Any]] = []

    for sample in samples:
        try:
            row_id = sample["row_id"]
            column = sample["column"]
            expected = sample["value"]
            match_type = sample.get("match_type", "exact")

            row = df_idx.loc[row_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            actual = row[column] if column in row.index else None

            if pd.isna(actual) and pd.isna(expected):
                ok = True
            elif pd.isna(actual) or pd.isna(expected):
                ok = False
            elif match_type == "contains":
                ok = str(expected).replace(" ", "") in str(actual).replace(" ", "")
            else:
                try:
                    ok = abs(float(actual) - float(expected)) < NUMERIC_TOLERANCE
                except Exception:
                    ok = str(actual).strip() == str(expected).strip()

            if ok:
                correct += 1
            elif len(mismatches) < 20:
                mismatches.append(
                    {
                        "row_id": row_id,
                        "column": column,
                        "match_type": match_type,
                        "expected": expected,
                        "actual": actual,
                    }
                )
        except Exception as exc:
            if len(mismatches) < 20:
                mismatches.append({"error": str(exc), "sample": sample})

    wrong = total - correct
    score = float(max(0, 50 - 5 * wrong))
    logger.info("data check: total=%s correct=%s wrong=%s score=%.2f/50", total, correct, wrong, score)
    if mismatches:
        logger.warning("data mismatch examples (up to 20):")
        for mismatch in mismatches:
            logger.warning("%s", mismatch)
    return score
