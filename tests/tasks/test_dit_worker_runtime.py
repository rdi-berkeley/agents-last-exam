from __future__ import annotations

import subprocess

import pytest

from tasks.computing_math.dit_pipeline_cfg_alignment_fid_256_001.scripts import score_outputs


def test_worker_preserves_explicit_pythonpath(monkeypatch, tmp_path):
    (tmp_path / "dit_worker_probe.py").write_text("VALUE = 42\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    candidate = "import dit_worker_probe\nassert dit_worker_probe.VALUE == 42\n"
    result = score_outputs.score_submission_text(candidate, candidate)
    assert result.score == 0.0
    assert result.failures == [
        "candidate module does not define DiTPipeline",
        "reference module does not define DiTPipeline",
    ]


def test_worker_does_not_export_mutated_host_import_paths(monkeypatch, tmp_path):
    (tmp_path / "host_only_probe.py").write_text("VALUE = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    result = score_outputs.score_submission_text("import host_only_probe", "")
    assert result.score == 0.0
    assert "ModuleNotFoundError" in result.failures[0]


def test_empty_worker_output_is_an_infrastructure_error(monkeypatch):
    def failed_worker(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, "", "No module named 'torch'")

    monkeypatch.setattr(score_outputs.subprocess, "run", failed_worker)
    with pytest.raises(RuntimeError, match="No module named 'torch'"):
        score_outputs.score_submission_text("", "")


def test_nonstandard_worker_exit_is_an_infrastructure_error(monkeypatch):
    def killed_worker(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], -9, "", "")

    monkeypatch.setattr(score_outputs.subprocess, "run", killed_worker)
    with pytest.raises(RuntimeError, match="returncode=-9"):
        score_outputs.score_submission_text("", "")


def test_candidate_import_failure_remains_a_scored_failure():
    result = score_outputs.score_submission_text("raise ValueError('bad submission')", "")
    assert result.score == 0.0
    assert result.failures == ["candidate import failed: ValueError: bad submission"]


def test_candidate_system_exit_is_a_scored_failure():
    result = score_outputs.score_submission_text("raise SystemExit(7)", "")
    assert result.score == 0.0
    assert result.failures == ["candidate import failed: SystemExit: 7"]


def test_worker_timeout_is_bounded():
    with pytest.raises(subprocess.TimeoutExpired):
        score_outputs.score_submission_text("while True: pass", "", worker_timeout=0.2)


PIPELINE = """import torch
from diffusers import AutoencoderKL, DiTTransformer2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
class DiTPipeline(DiffusionPipeline):
    def __init__(self, transformer, vae, scheduler):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)
    def __call__(self, class_labels, guidance_scale, cfg_on_3_channels=False,
                 generator=None, num_inference_steps=2, output_type="np"):
        assert self.scheduler.config.variance_type == self.scheduler.config["variance_type"]
        latents = torch.randn((2,4,2,2), generator=generator)
        guided_channels = 3 if cfg_on_3_channels else 4
        latents[:, :guided_channels] *= guidance_scale
        self.scheduler.set_timesteps(num_inference_steps)
        for timestep in self.scheduler.timesteps:
            if hasattr(self.scheduler, "scale_model_input"):
                latents = self.scheduler.scale_model_input(latents, timestep)
            noise = self.transformer(latents, timestep=timestep.expand(2),
                                     class_labels=torch.tensor(class_labels)).sample
            if self.scheduler.config.variance_type not in ("learned", "learned_range"):
                noise = noise[:, :4]
            latents = self.scheduler.step(noise, timestep, latents).prev_sample
        images = self.vae.decode(latents).sample.permute(0,2,3,1).numpy()
        return ImagePipelineOutput(images=images)
"""


@pytest.mark.parametrize(
    "defect", [None, "ignored_cfg", "wrong_default", "shape", "nonfinite", "system_exit"]
)
def test_interface_compatibility_preserves_behavioral_checks(defect):
    candidate = PIPELINE
    if defect == "ignored_cfg":
        candidate = candidate.replace("3 if cfg_on_3_channels else 4", "4")
    elif defect == "wrong_default":
        candidate = candidate.replace("cfg_on_3_channels=False", "cfg_on_3_channels=True")
    elif defect == "shape":
        candidate = candidate.replace("images=images)", "images=images[:1])")
    elif defect == "nonfinite":
        candidate = candidate.replace("images=images)", 'images=images * float("nan"))')
    elif defect == "system_exit":
        candidate = candidate.replace(
            "assert self.scheduler.config.variance_type",
            "raise SystemExit(7)\n        assert self.scheduler.config.variance_type",
        )
    result = score_outputs._score_submission_text_in_process(candidate, PIPELINE)
    assert result.passed == (defect is None)
    assert (result.score == 1) == (defect is None)
    assert len(result.cases) == 5
