import json
import logging
import math
import os
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

logger = logging.getLogger(__name__)
RFC3339_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_object(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_c2_servers(servers: Iterable[dict[str, Any]]) -> list[tuple[str, int, str, str]]:
    normalized = []
    for server in servers:
        normalized.append(
            (
                _normalize_string(server.get("ip")),
                int(server.get("port")),
                _normalize_string(server.get("protocol")).lower(),
                _normalize_string(server.get("first_seen")),
            )
        )
    return sorted(normalized)


def _normalize_infection_chain(steps: Iterable[dict[str, Any]]) -> list[tuple[int, str, str, str, str | None]]:
    normalized = []
    for step in steps:
        normalized.append(
            (
                int(step.get("step")),
                _normalize_string(step.get("timestamp")),
                _normalize_string(step.get("src_ip")),
                _normalize_string(step.get("dst_ip")),
                _normalize_optional_string(step.get("url_or_domain")),
            )
        )
    return normalized


def _normalize_ioc_list(values: Iterable[Any]) -> list[str]:
    return sorted({_normalize_string(value) for value in values if _normalize_string(value)})


def _score_equal(actual: Any, expected: Any) -> float:
    return 1.0 if actual == expected else 0.0


def _has_required_top_level_keys(report: dict[str, Any]) -> tuple[bool, list[str]]:
    required = [
        "malware_family",
        "compromised_host",
        "initial_vector",
        "c2_servers",
        "infection_chain",
        "exfiltration",
        "iocs",
    ]
    missing = [key for key in required if key not in report]
    return (len(missing) == 0, missing)


def _type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _is_rfc3339_datetime(value: str) -> bool:
    if not RFC3339_DATETIME_RE.match(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []

    expected_type = schema.get("type")
    if expected_type is not None:
        allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_type_matches(value, item) for item in allowed_types):
            return [f"{path}: expected {allowed_types}, got {type(value).__name__}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} not in enum {schema['enum']!r}")

    if schema.get("format") == "date-time":
        if not isinstance(value, str):
            errors.append(f"{path}: date-time format requires string value")
        elif not _is_rfc3339_datetime(value):
            errors.append(f"{path}: invalid date-time value {value!r}")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        for key, subschema in properties.items():
            if key in value:
                errors.extend(_validate_schema(value[key], subschema, f"{path}.{key}"))
        return errors

    if isinstance(value, list):
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(value):
                errors.extend(_validate_schema(item, item_schema, f"{path}[{idx}]"))
        return errors

    return errors


def _load_hidden_reference(reference_text: str) -> dict[str, Any]:
    return json.loads(reference_text)


@dataclass
class TaskConfig(GeneralTaskConfig):
    VARIANT_NAME: str = "base"
    DOMAIN_NAME: str = "computing_math"
    TASK_NAME: str = "pcap_enterprise_triage_01"
    OUTPUT_FILENAME: str = "report.json"
    OS_TYPE: str = "windows"

    @property
    def task_dir(self) -> str:
        return rf"E:\agenthle\{self.DOMAIN_NAME}\{self.TASK_NAME}\{self.VARIANT_NAME}"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def output_path(self) -> str:
        return rf"{self.remote_output_dir}\{self.OUTPUT_FILENAME}"

    @property
    def runtime_output_dir(self) -> str:
        return self.remote_output_dir

    @property
    def reference_path(self) -> str:
        return rf"{self.reference_dir}\reference_report.json"

    @property
    def schema_path(self) -> str:
        return rf"{self.input_dir}\output_schema.json"

    @property
    def pcap_path(self) -> str:
        return rf"{self.input_dir}\capture_enhanced.pcap"

    @property
    def task_description(self) -> str:
        return f"""\
You are a cybersecurity analyst working on a Windows VM.

## Your Task
Use Wireshark to analyze the staged enterprise packet capture and produce a structured triage report.

## Input Files
- PCAP: `{self.pcap_path}`
- Output schema: `{self.schema_path}`
- Software notes: `{self.software_dir}\\README.txt`

## Software
- Launch Wireshark from: `C:\\Program Files\\Wireshark\\Wireshark.exe`

## What You Must Do
1. Identify the compromised internal host.
2. Reconstruct the infection chain chronologically.
3. Recover the initial vector and malicious delivery URL.
4. Identify the C2 server information.
5. Extract the required IOC sets.
6. Save one JSON report that follows the visible schema.

## Output Requirements
- Save exactly one file to: `{self.output_path}`
- Filename must be `report.json`
- The JSON must follow the schema in `output_schema.json`
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_dir": self.input_dir,
                "output_path": self.output_path,
                "runtime_output_dir": self.runtime_output_dir,
                "reference_gcs_prefix": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/reference"
                ),
                "reference_path": self.reference_path,
                "schema_path": self.schema_path,
                "pcap_path": self.pcap_path,
                "output_filename": self.OUTPUT_FILENAME,
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
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _score_report(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    actual_initial = actual.get("initial_vector", {})
    expected_initial = expected.get("initial_vector", {})
    actual_exfil = actual.get("exfiltration", {})
    expected_exfil = expected.get("exfiltration", {})
    actual_iocs = actual.get("iocs", {})
    expected_iocs = expected.get("iocs", {})

    component_scores = {
        "malware_family": 0.15
        * _score_equal(
            _normalize_string(actual.get("malware_family")).lower(),
            _normalize_string(expected.get("malware_family")).lower(),
        ),
        "compromised_host": 0.20
        * _score_equal(
            _normalize_string(actual.get("compromised_host")),
            _normalize_string(expected.get("compromised_host")),
        ),
        "initial_vector": 0.15
        * _score_equal(
            _normalize_object(
                {
                    "type": _normalize_string(actual_initial.get("type")).lower(),
                    "url": _normalize_string(actual_initial.get("url")),
                    "source_ip": _normalize_string(actual_initial.get("source_ip")),
                    "timestamp": _normalize_string(actual_initial.get("timestamp")),
                }
            ),
            _normalize_object(
                {
                    "type": _normalize_string(expected_initial.get("type")).lower(),
                    "url": _normalize_string(expected_initial.get("url")),
                    "source_ip": _normalize_string(expected_initial.get("source_ip")),
                    "timestamp": _normalize_string(expected_initial.get("timestamp")),
                }
            ),
        ),
        "c2_servers": 0.15
        * _score_equal(
            _normalize_c2_servers(actual.get("c2_servers", [])),
            _normalize_c2_servers(expected.get("c2_servers", [])),
        ),
        "infection_chain": 0.20
        * _score_equal(
            _normalize_infection_chain(actual.get("infection_chain", [])),
            _normalize_infection_chain(expected.get("infection_chain", [])),
        ),
        "exfiltration": 0.05
        * _score_equal(
            _normalize_object(
                {
                    "detected": bool(actual_exfil.get("detected")),
                    "dst_ip": _normalize_optional_string(actual_exfil.get("dst_ip")),
                    "method": _normalize_optional_string(actual_exfil.get("method")),
                    "data_size_bytes": int(actual_exfil.get("data_size_bytes", 0)),
                }
            ),
            _normalize_object(
                {
                    "detected": bool(expected_exfil.get("detected")),
                    "dst_ip": _normalize_optional_string(expected_exfil.get("dst_ip")),
                    "method": _normalize_optional_string(expected_exfil.get("method")),
                    "data_size_bytes": int(expected_exfil.get("data_size_bytes", 0)),
                }
            ),
        ),
        "iocs": 0.10
        * _score_equal(
            _normalize_object(
                {
                    "malicious_ips": _normalize_ioc_list(actual_iocs.get("malicious_ips", [])),
                    "malicious_domains": _normalize_ioc_list(actual_iocs.get("malicious_domains", [])),
                    "malicious_urls": _normalize_ioc_list(actual_iocs.get("malicious_urls", [])),
                    "file_hashes": _normalize_ioc_list(actual_iocs.get("file_hashes", [])),
                }
            ),
            _normalize_object(
                {
                    "malicious_ips": _normalize_ioc_list(expected_iocs.get("malicious_ips", [])),
                    "malicious_domains": _normalize_ioc_list(expected_iocs.get("malicious_domains", [])),
                    "malicious_urls": _normalize_ioc_list(expected_iocs.get("malicious_urls", [])),
                    "file_hashes": _normalize_ioc_list(expected_iocs.get("file_hashes", [])),
                }
            ),
        ),
    }
    score = sum(component_scores.values())
    passed = math.isclose(score, 1.0, rel_tol=0.0, abs_tol=1e-9)
    return {
        "score": score,
        "passed": passed,
        "component_scores": component_scores,
    }


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    output_path = task_cfg.metadata["output_path"]
    schema_path = task_cfg.metadata["schema_path"]
    runtime_out_dir = task_cfg.metadata["runtime_output_dir"]

    logger.info("[pcap_enterprise_triage_01] evaluate() begin")
    logger.info("[pcap_enterprise_triage_01] output_path=%s", output_path)

    report: Dict[str, Any] = {
        "output_path": output_path,
        "passed": False,
        "score": 0.0,
    }

    try:
        output_text = await session.read_file(output_path)
    except Exception as e:
        report["error"] = f"failed to read output: {e}"
        await session.write_file(
            os.path.join(runtime_out_dir, "autograde_report.json"),
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return [0.0]

    try:
        schema_text = await session.read_file(schema_path)
        schema = json.loads(schema_text)
    except Exception as e:
        report["error"] = f"failed to read schema: {e}"
        await session.write_file(
            os.path.join(runtime_out_dir, "autograde_report.json"),
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return [0.0]

    try:
        actual = json.loads(output_text)
    except json.JSONDecodeError as e:
        report["error"] = f"invalid json: {e}"
        await session.write_file(
            os.path.join(runtime_out_dir, "autograde_report.json"),
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return [0.0]

    try:
        expected = _load_hidden_reference(await session.read_file(task_cfg.metadata["reference_path"]))
    except Exception as e:
        report["error"] = f"failed to load staged hidden reference: {e}"
        await session.write_file(
            os.path.join(runtime_out_dir, "autograde_report.json"),
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return [0.0]

    has_required, missing = _has_required_top_level_keys(actual)
    report["missing_required_top_level_keys"] = missing
    if not has_required:
        await session.write_file(
            os.path.join(runtime_out_dir, "autograde_report.json"),
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return [0.0]

    schema_errors = _validate_schema(actual, schema)
    if schema_errors:
        report["schema_errors"] = schema_errors[:50]
        await session.write_file(
            os.path.join(runtime_out_dir, "autograde_report.json"),
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return [0.0]

    score_report = _score_report(actual, expected)
    report.update(score_report)

    await session.write_file(
        os.path.join(runtime_out_dir, "autograde_report.json"),
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    logger.info("[pcap_enterprise_triage_01] score=%s passed=%s", report["score"], report["passed"])
    return [float(report["score"])]
