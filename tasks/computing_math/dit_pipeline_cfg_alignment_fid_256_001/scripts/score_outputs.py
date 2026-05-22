"""Hidden scorer for dit_pipeline_cfg_alignment_fid_256_001."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
import types
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


@dataclass
class CaseResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ScoreResult:
    score: float
    passed: bool
    failures: list[str] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cases"] = [asdict(case) for case in self.cases]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScoreResult":
        return cls(
            score=float(payload["score"]),
            passed=bool(payload["passed"]),
            failures=list(payload.get("failures", [])),
            cases=[CaseResult(**case) for case in payload.get("cases", [])],
        )


class _AutoencoderKL:
    def __init__(self) -> None:
        self.config = SimpleNamespace(scaling_factor=0.5)

    def decode(self, latents: torch.Tensor) -> SimpleNamespace:
        sample = torch.stack(
            [
                latents[:, 0],
                latents[:, 1] * 0.5,
                latents[:, 3] if latents.shape[1] > 3 else latents[:, 0] * 0.0,
            ],
            dim=1,
        )
        return SimpleNamespace(sample=sample)


class _KarrasDiffusionSchedulers:
    pass


class _DiffusionPipeline:
    def __init__(self) -> None:
        self._device = torch.device("cpu")

    def register_modules(self, **modules: Any) -> None:
        for key, value in modules.items():
            setattr(self, key, value)

    @property
    def _execution_device(self) -> torch.device:
        return self._device

    def progress_bar(self, iterable):
        return iterable

    def maybe_free_model_hooks(self) -> None:
        return None

    def numpy_to_pil(self, samples: np.ndarray) -> np.ndarray:
        return samples

    def to(self, device: str | torch.device):
        self._device = torch.device(device)
        return self


@dataclass
class _ImagePipelineOutput:
    images: Any


def _randn_tensor(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


@contextmanager
def _diffusers_stubs():
    module_names = [
        "diffusers",
        "diffusers.models",
        "diffusers.schedulers",
        "diffusers.utils",
        "diffusers.utils.torch_utils",
        "diffusers.pipelines",
        "diffusers.pipelines.pipeline_utils",
    ]
    saved = {name: sys.modules.get(name) for name in module_names}

    diffusers = types.ModuleType("diffusers")
    models = types.ModuleType("diffusers.models")
    schedulers = types.ModuleType("diffusers.schedulers")
    utils = types.ModuleType("diffusers.utils")
    torch_utils = types.ModuleType("diffusers.utils.torch_utils")
    pipelines = types.ModuleType("diffusers.pipelines")
    pipeline_utils = types.ModuleType("diffusers.pipelines.pipeline_utils")

    models.AutoencoderKL = _AutoencoderKL
    schedulers.KarrasDiffusionSchedulers = _KarrasDiffusionSchedulers
    utils.is_torch_xla_available = lambda: False
    torch_utils.randn_tensor = _randn_tensor
    pipeline_utils.DiffusionPipeline = _DiffusionPipeline
    pipeline_utils.ImagePipelineOutput = _ImagePipelineOutput

    diffusers.models = models
    diffusers.schedulers = schedulers
    diffusers.utils = utils
    diffusers.pipelines = pipelines
    utils.torch_utils = torch_utils
    pipelines.pipeline_utils = pipeline_utils

    injected = {
        "diffusers": diffusers,
        "diffusers.models": models,
        "diffusers.schedulers": schedulers,
        "diffusers.utils": utils,
        "diffusers.utils.torch_utils": torch_utils,
        "diffusers.pipelines": pipelines,
        "diffusers.pipelines.pipeline_utils": pipeline_utils,
    }

    try:
        sys.modules.update(injected)
        yield
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _load_module_from_text(module_name: str, text: str, root: Path):
    path = root / f"{module_name}.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)


class _DummyTransformer:
    def __init__(self, *, sample_size: int = 2, in_channels: int = 4, out_channels: int = 8) -> None:
        self.config = SimpleNamespace(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.dtype = torch.float32

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        *,
        timestep: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> SimpleNamespace:
        batch, channels, _, _ = latent_model_input.shape
        class_term = class_labels.float().reshape(batch, 1, 1, 1)
        time_term = timestep.float().reshape(batch, 1, 1, 1)
        primary = latent_model_input * 0.5 + class_term * 0.1 + time_term * 0.01
        extra = torch.stack(
            [
                latent_model_input[:, 0] + 1.0,
                latent_model_input[:, 1] + 2.0,
                latent_model_input[:, 2] + 3.0,
                latent_model_input[:, 3] + 4.0,
            ],
            dim=1,
        )
        sample = torch.cat([primary, extra], dim=1)
        return SimpleNamespace(sample=sample)


class _DummyScheduler:
    def __init__(self, *, variance_type: str, has_scale_model_input: bool) -> None:
        self.variance_type = variance_type
        if has_scale_model_input:
            self.scale_model_input = self._scale_model_input

    def _scale_model_input(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return latents + 0.05

    def set_timesteps(self, _: int) -> None:
        self.timesteps = [torch.tensor(2), torch.tensor(1)]

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        latent_model_input: torch.Tensor,
    ) -> SimpleNamespace:
        del timestep
        update = model_output[:, : latent_model_input.shape[1]]
        return SimpleNamespace(prev_sample=latent_model_input - 0.1 * update)


def _instantiate_pipeline(module: Any, *, variance_type: str, has_scale_model_input: bool):
    transformer = _DummyTransformer()
    vae = _AutoencoderKL()
    scheduler = _DummyScheduler(
        variance_type=variance_type,
        has_scale_model_input=has_scale_model_input,
    )
    return module.DiTPipeline(transformer=transformer, vae=vae, scheduler=scheduler)


def _run_behavior_case(
    candidate_module: Any,
    reference_module: Any,
    *,
    name: str,
    cfg_on_3_channels: bool,
    guidance_scale: float,
    variance_type: str,
    has_scale_model_input: bool,
) -> CaseResult:
    try:
        candidate = _instantiate_pipeline(
            candidate_module,
            variance_type=variance_type,
            has_scale_model_input=has_scale_model_input,
        )
        reference = _instantiate_pipeline(
            reference_module,
            variance_type=variance_type,
            has_scale_model_input=has_scale_model_input,
        )

        candidate_generator = torch.Generator(device="cpu").manual_seed(123)
        reference_generator = torch.Generator(device="cpu").manual_seed(123)

        candidate_output = candidate(
            class_labels=[1, 7],
            guidance_scale=guidance_scale,
            cfg_on_3_channels=cfg_on_3_channels,
            generator=candidate_generator,
            num_inference_steps=2,
            output_type="np",
        )
        reference_output = reference(
            class_labels=[1, 7],
            guidance_scale=guidance_scale,
            cfg_on_3_channels=cfg_on_3_channels,
            generator=reference_generator,
            num_inference_steps=2,
            output_type="np",
        )

        candidate_images = np.asarray(candidate_output.images)
        reference_images = np.asarray(reference_output.images)

        if candidate_images.shape != reference_images.shape:
            return CaseResult(
                name=name,
                passed=False,
                detail=f"shape mismatch: {candidate_images.shape} vs {reference_images.shape}",
            )
        if not np.allclose(candidate_images, reference_images, atol=1e-6):
            delta = float(np.max(np.abs(candidate_images - reference_images)))
            return CaseResult(
                name=name,
                passed=False,
                detail=f"numeric mismatch, max_abs_delta={delta}",
            )
        return CaseResult(name=name, passed=True, detail="matched reference behavior")
    except Exception as exc:  # noqa: BLE001
        return CaseResult(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}")


def _score_submission_text_in_process(submission_text: str, reference_text: str) -> ScoreResult:
    failures: list[str] = []
    cases: list[CaseResult] = []

    with tempfile.TemporaryDirectory(prefix="dit_cfg_score_") as temp_dir:
        root = Path(temp_dir)
        with _diffusers_stubs():
            try:
                candidate_module = _load_module_from_text("candidate_pipeline_dit", submission_text, root)
            except Exception as exc:  # noqa: BLE001
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    failures=[f"candidate import failed: {type(exc).__name__}: {exc}"],
                    cases=[],
                )

            try:
                reference_module = _load_module_from_text("reference_pipeline_dit", reference_text, root)
            except Exception as exc:  # noqa: BLE001
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    failures=[f"reference import failed: {type(exc).__name__}: {exc}"],
                    cases=[],
                )

            if not hasattr(candidate_module, "DiTPipeline"):
                failures.append("candidate module does not define DiTPipeline")
            if not hasattr(reference_module, "DiTPipeline"):
                failures.append("reference module does not define DiTPipeline")

            if not failures:
                signature = inspect.signature(candidate_module.DiTPipeline.__call__)
                if "cfg_on_3_channels" not in signature.parameters:
                    failures.append("candidate __call__ is missing cfg_on_3_channels")

            if failures:
                return ScoreResult(score=0.0, passed=False, failures=failures, cases=cases)

            default = signature.parameters["cfg_on_3_channels"].default
            if default is False:
                cases.append(CaseResult(
                    name="cfg_on_3_channels_default",
                    passed=True,
                    detail="default is False as expected",
                ))
            else:
                cases.append(CaseResult(
                    name="cfg_on_3_channels_default",
                    passed=False,
                    detail=f"default is {default!r}, expected False",
                ))

            cases.extend(
                [
                    _run_behavior_case(
                        candidate_module,
                        reference_module,
                        name="cfg_on_3_channels_learned_range",
                        cfg_on_3_channels=True,
                        guidance_scale=1.5,
                        variance_type="learned_range",
                        has_scale_model_input=True,
                    ),
                    _run_behavior_case(
                        candidate_module,
                        reference_module,
                        name="cfg_all_channels_learned_range",
                        cfg_on_3_channels=False,
                        guidance_scale=1.5,
                        variance_type="learned_range",
                        has_scale_model_input=True,
                    ),
                    _run_behavior_case(
                        candidate_module,
                        reference_module,
                        name="cfg_on_3_channels_without_scale_model_input",
                        cfg_on_3_channels=True,
                        guidance_scale=1.5,
                        variance_type="learned_range",
                        has_scale_model_input=False,
                    ),
                    _run_behavior_case(
                        candidate_module,
                        reference_module,
                        name="fixed_variance_path",
                        cfg_on_3_channels=True,
                        guidance_scale=1.5,
                        variance_type="fixed_small",
                        has_scale_model_input=True,
                    ),
                ]
            )

    failures.extend(case.detail for case in cases if not case.passed)
    passed = not failures
    score = sum(1 for c in cases if c.passed) / len(cases) if cases else 0.0
    return ScoreResult(score=score, passed=passed, failures=failures, cases=cases)


def score_submission_text(submission_text: str, reference_text: str) -> ScoreResult:
    with tempfile.TemporaryDirectory(prefix="dit_cfg_parent_") as temp_dir:
        root = Path(temp_dir)
        submission_path = root / "submission_pipeline_dit.py"
        reference_path = root / "reference_pipeline_dit.py"
        submission_path.write_text(submission_text, encoding="utf-8")
        reference_path.write_text(reference_text, encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                "--submission-file",
                str(submission_path),
                "--reference-file",
                str(reference_path),
                "--json-only",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode not in {0, 1}:
            return ScoreResult(
                score=0.0,
                passed=False,
                failures=[
                    f"scorer worker failed: returncode={result.returncode}, stderr={result.stderr.strip()}"
                ],
                cases=[],
            )
        try:
            return ScoreResult.from_dict(json.loads(result.stdout))
        except json.JSONDecodeError as exc:
            return ScoreResult(
                score=0.0,
                passed=False,
                failures=[f"could not parse scorer worker output: {exc}: {result.stdout!r}"],
                cases=[],
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-file", required=True)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    submission_text = Path(args.submission_file).read_text(encoding="utf-8")
    reference_text = Path(args.reference_file).read_text(encoding="utf-8")
    if args.worker:
        result = _score_submission_text_in_process(submission_text, reference_text)
    else:
        result = score_submission_text(submission_text, reference_text)
    payload = result.to_dict()
    if args.json_only:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
