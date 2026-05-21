"""Task definition loader.

Ported from simprun/task_loader.py. Reads ``main.py`` + ``task_card.json``
under each task dir and produces normalised task metadata. ``TaskDataSpec``
itself lives in :mod:`ale_run.base_interface`.

Known coupling: this module imports ``parse_gce_machine_type`` from
``environments/machine_types`` to convert the GCE machine type string
from ``task_card.json`` into vCPU/memory ints. That's a helper crossing
the tasks→environments boundary; a future refactor should let the
orchestrator parse that string instead, but for now the leak stays.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict

from ..base_interface import TaskDataSpec
from ..environments.machine_types import parse_gce_machine_type

__all__ = ["TaskDataSpec", "TaskLoader"]

logger = logging.getLogger(__name__)

# Task-local bare-name imports declared by the Stage 2 hard rule.
_TASK_LOCAL_MODULE_NAMES = (
    "score_outputs",
    "verify_outputs",
)

_TASK_IMPORT_LOCK = threading.Lock()


# ======================================================================
# TaskLoader
# ======================================================================


class TaskLoader:
    def __init__(self, task_path: str):
        self.task_path = Path(task_path).resolve()
        self.main_py = self.task_path / "main.py"
        if not self.main_py.exists():
            raise FileNotFoundError(f"main.py not found at {self.main_py}")
        self._module = None

    def _load_task_variant(self, variant_index: int = 0) -> Any | None:
        module = self._load_module()
        load_fn = getattr(module, "load", None)
        if load_fn is None or not callable(load_fn):
            return None
        tasks = load_fn() or []
        if tasks and len(tasks) > variant_index:
            return tasks[variant_index]
        return None

    def _load_module(self):
        if self._module is not None:
            return self._module

        rel_parts = self.task_path.parts
        try:
            tasks_idx = rel_parts.index("tasks")
            unique_suffix = "_".join(rel_parts[tasks_idx + 1 :])
        except ValueError:
            unique_suffix = self.task_path.name
        module_name = f"_task_{unique_suffix}"

        with _TASK_IMPORT_LOCK:
            scripts_dir = str(self.task_path / "scripts")
            task_dir = str(self.task_path)

            for mod_name in _TASK_LOCAL_MODULE_NAMES:
                sys.modules.pop(mod_name, None)

            added_paths = []
            for p in (scripts_dir, task_dir):
                if os.path.isdir(p):
                    try:
                        sys.path.remove(p)
                    except ValueError:
                        pass
                    sys.path.insert(0, p)
                    added_paths.append(p)

            modules_before = set(sys.modules.keys())

            if os.path.isdir(scripts_dir):
                for mod_name in _TASK_LOCAL_MODULE_NAMES:
                    src = Path(scripts_dir) / f"{mod_name}.py"
                    if not src.is_file():
                        continue
                    sub_spec = importlib.util.spec_from_file_location(mod_name, str(src))
                    if sub_spec is None or sub_spec.loader is None:
                        continue
                    sub_module = importlib.util.module_from_spec(sub_spec)
                    sys.modules[mod_name] = sub_module
                    sub_spec.loader.exec_module(sub_module)

            try:
                spec = importlib.util.spec_from_file_location(module_name, str(self.main_py))
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot create module spec for {self.main_py}")

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                self._module = module
                return module
            finally:
                for mod_name in _TASK_LOCAL_MODULE_NAMES:
                    sys.modules.pop(mod_name, None)
                _task_prefixes = tuple(os.path.abspath(p) + os.sep for p in added_paths)
                for k in set(sys.modules.keys()) - modules_before:
                    if k == module_name:
                        continue
                    mod = sys.modules.get(k)
                    origin = getattr(getattr(mod, "__spec__", None), "origin", None) or ""
                    f = getattr(mod, "__file__", None) or ""
                    src = os.path.abspath(origin or f) if (origin or f) else ""
                    if src and any(src.startswith(tp) for tp in _task_prefixes):
                        del sys.modules[k]
                for p in added_paths:
                    try:
                        sys.path.remove(p)
                    except ValueError:
                        pass

    def load(self, variant_index: int = 0) -> Dict[str, Any]:
        module = self._load_module()
        config = getattr(module, "config", None)

        try:
            task = self._load_task_variant(variant_index=variant_index)
            if task is not None:
                description = getattr(task, "description", "")
                metadata = getattr(task, "metadata", {}) or {}
                computer = getattr(task, "computer", {}) or {}
                logger.info("Loaded task config via load() function")
                task_data = self._extract_task_data(metadata=metadata, config=config)
                return self._enrich_with_task_card(
                    {
                        "description": description,
                        "metadata": metadata,
                        "computer": computer,
                        "os_type": self._extract_os_type(task=task, config=config),
                        "task_data": task_data,
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to call load(): {e}")

        if config is not None and hasattr(config, "task_description"):
            description = config.task_description
            metadata = config.to_metadata() if hasattr(config, "to_metadata") else {}
            logger.info("Loaded task config from module-level 'config' object")
            task_data = self._extract_task_data(metadata=metadata, config=config)
            return self._enrich_with_task_card(
                {
                    "description": description,
                    "metadata": metadata,
                    "os_type": self._extract_os_type(config=config),
                    "task_data": task_data,
                }
            )

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and attr_name.endswith("Config")
                and attr_name != "GeneralTaskConfig"
                and hasattr(obj, "task_description")
            ):
                try:
                    instance = obj()
                    description = instance.task_description
                    metadata = instance.to_metadata() if hasattr(instance, "to_metadata") else {}
                    logger.info(f"Loaded task config from class {attr_name}")
                    task_data = self._extract_task_data(metadata=metadata, config=instance)
                    return self._enrich_with_task_card(
                        {
                            "description": description,
                            "metadata": metadata,
                            "os_type": self._extract_os_type(config=instance),
                            "task_data": task_data,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to instantiate {attr_name}: {e}")

        raise RuntimeError(
            f"Could not extract task config from {self.main_py}. "
            f"Expected a module-level 'config' object, a *Config class, "
            f"or a load() function."
        )

    def _enrich_with_task_card(self, task_info: dict) -> dict:
        card = self._load_task_card()
        vm_cfg = card.get("vm", {})
        if not vm_cfg and (card.get("snapshot") or card.get("vm_category")):
            vm_cfg = {
                "snapshot": card.get("snapshot") or card.get("vm_category"),
                "machineType": card.get("machineType"),
                "timeout": card.get("timeout"),
            }
        if vm_cfg:
            task_info["image_category"] = vm_cfg.get("snapshot")
            task_info["snapshot_name"] = vm_cfg.get("snapshot")
            if "timeout" in vm_cfg:
                task_info["timeout_s"] = self._parse_task_timeout(vm_cfg["timeout"])
            raw_mt = vm_cfg.get("machineType")
            task_info["machine_type"] = raw_mt
            if raw_mt is not None:
                shape = parse_gce_machine_type(raw_mt)
                if shape is None:
                    raise ValueError(
                        f"task_card.json for {self.task_path} has unparseable "
                        f"vm.machineType={raw_mt!r}; expected a standard GCE "
                        "machine type like 'n2-highmem-16' or "
                        "'n2-custom-8-16384'"
                    )
                task_info["vcpus"] = shape.vcpus
                task_info["memory_gb"] = shape.memory_gb
        return task_info

    def _parse_task_timeout(self, raw_timeout: Any) -> int:
        if isinstance(raw_timeout, bool):
            raise ValueError(
                f"task_card.json for {self.task_path} has invalid vm.timeout={raw_timeout!r}; "
                "expected positive integer seconds"
            )
        try:
            if isinstance(raw_timeout, float) and not raw_timeout.is_integer():
                raise ValueError
            timeout_s = int(raw_timeout)
        except (TypeError, ValueError):
            raise ValueError(
                f"task_card.json for {self.task_path} has invalid vm.timeout={raw_timeout!r}; "
                "expected positive integer seconds"
            ) from None
        if timeout_s <= 0:
            raise ValueError(
                f"task_card.json for {self.task_path} has invalid vm.timeout={raw_timeout!r}; "
                "expected positive integer seconds"
            )
        return timeout_s

    def build_task_cfg(self, variant_index: int = 0) -> Any:
        module = self._load_module()

        try:
            task = self._load_task_variant(variant_index=variant_index)
            if task is not None:
                if not hasattr(task, "metadata") or getattr(task, "metadata") is None:
                    setattr(task, "metadata", {})
                return task
        except Exception as e:
            logger.warning(f"Failed to resolve task object via load(): {e}")

        config = getattr(module, "config", None)
        if config is not None:
            if not hasattr(config, "metadata"):
                if hasattr(config, "to_metadata"):
                    config.metadata = config.to_metadata()
                else:
                    config.metadata = {}
            return config

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and attr_name.endswith("Config")
                and attr_name != "GeneralTaskConfig"
                and hasattr(obj, "task_description")
            ):
                instance = obj()
                if not hasattr(instance, "metadata"):
                    if hasattr(instance, "to_metadata"):
                        instance.metadata = instance.to_metadata()
                    else:
                        instance.metadata = {}
                return instance

        raise RuntimeError(
            f"Could not build task cfg from {self.main_py}. "
            f"Expected a module-level 'config' object, a *Config class, "
            f"or a load() function."
        )

    def _load_task_card(self) -> dict:
        card_path = self.task_path / "task_card.json"
        if not card_path.exists():
            return {}
        try:
            with open(card_path, encoding="utf-8") as f:
                return json.load(f) or {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read task_card.json at %s: %s", card_path, e)
            return {}

    def _extract_os_type(self, *, task: Any | None = None, config: Any | None = None) -> str:
        computer = getattr(task, "computer", None)
        if isinstance(computer, dict):
            setup_config = computer.get("setup_config") or {}
            os_type = setup_config.get("os_type")
            if isinstance(os_type, str) and os_type:
                return os_type

        for candidate in (config,):
            if candidate is None:
                continue
            for attr_name in ("OS_TYPE", "os_type"):
                value = getattr(candidate, attr_name, None)
                if isinstance(value, str) and value:
                    return value

        logger.warning("Task %s did not expose os_type; defaulting to windows", self.task_path)
        return "windows"

    def _extract_task_data(
        self, *, metadata: dict[str, Any], config: Any | None = None
    ) -> TaskDataSpec:
        explicit_requires = metadata.get("requires_task_data")
        if (
            explicit_requires is None
            and config is not None
            and hasattr(config, "REQUIRES_TASK_DATA")
        ):
            explicit_requires = getattr(config, "REQUIRES_TASK_DATA")
        if explicit_requires is None:
            explicit_requires = any(
                metadata.get(key) for key in ("input_dir", "software_dir", "reference_dir")
            )

        requires_task_data = bool(explicit_requires)
        if not requires_task_data:
            return TaskDataSpec(requires_task_data=False)

        domain_name = str(metadata.get("domain_name") or "").strip()
        task_name = str(metadata.get("task_name") or "").strip()
        variant_name = str(metadata.get("variant_name") or "").strip()
        if not domain_name or not task_name or not variant_name:
            raise RuntimeError(
                f"Task {self.task_path} requires task data but metadata is missing "
                f"domain_name/task_name/variant_name "
                f"(got domain_name={domain_name!r}, task_name={task_name!r}, variant_name={variant_name!r})"
            )

        return TaskDataSpec(
            requires_task_data=True,
            domain_name=domain_name,
            task_name=task_name,
            variant_name=variant_name,
            source_relpath=f"{domain_name}/{task_name}/{variant_name}",
            input_dir=metadata.get("input_dir"),
            software_dir=metadata.get("software_dir"),
            reference_dir=metadata.get("reference_dir"),
            reference_gcs_prefix=metadata.get("reference_gcs_prefix"),
            remote_output_dir=metadata.get("remote_output_dir"),
            eval_gcs_prefix=metadata.get("eval_gcs_prefix"),
            eval_dir=metadata.get("eval_dir"),
        )

    def get_setup_fn(self):
        module = self._load_module()
        return getattr(module, "start", None)

    def get_evaluate_fn(self):
        module = self._load_module()
        return getattr(module, "evaluate", None)
