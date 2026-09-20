import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tasks.computing_math.dit_pipeline_cfg_alignment_fid_256_001.scripts import score_outputs


DATA = (
    Path(__file__).resolve().parents[2]
    / "task-data-hf/extracted/computing_math/dit_pipeline_cfg_alignment_fid_256_001/base"
)


@pytest.fixture
def reference_text():
    reference = DATA / "reference/pipeline_dit.py"
    if not reference.exists():
        pytest.skip("DIT release data is not installed")
    return reference.read_text()


@pytest.mark.parametrize("model_name", ["AutoencoderKL", "DiTTransformer2DModel"])
def test_diffusers_top_level_model_exports_match_nested(model_name, tmp_path):
    with score_outputs._diffusers_stubs():
        module = score_outputs._load_module_from_text(
            "dit_export_contract",
            f"from diffusers import {model_name} as top_level\n"
            f"from diffusers.models import {model_name} as nested\n",
            tmp_path,
        )
        assert module.top_level is module.nested


@pytest.mark.parametrize("scaled_step_state", [False, True])
def test_top_level_model_imports_preserve_numerical_scores(reference_text, scaled_step_state):
    nested_import = "from diffusers.models import AutoencoderKL, DiTTransformer2DModel"
    candidate = reference_text.replace("from diffusers.models import AutoencoderKL", nested_import)
    candidate = candidate.replace("        transformer,", "        transformer: DiTTransformer2DModel,")
    assert candidate != reference_text
    if scaled_step_state:
        original = "self.scheduler.step(model_output, t, latent_model_input)"
        assert candidate.count(original) == 1
        candidate = candidate.replace(original, "self.scheduler.step(model_output, t, model_input)")

    expected = score_outputs.score_submission_text(candidate, reference_text)
    assert expected.score == (0.6 if scaled_step_state else 1.0), expected.to_dict()
    assert len(expected.cases) == 5
    assert candidate.count(nested_import) == 1
    candidate = candidate.replace(
        nested_import, "from diffusers import AutoencoderKL, DiTTransformer2DModel"
    )

    result = score_outputs.score_submission_text(candidate, reference_text)

    assert result.to_dict() == expected.to_dict()


class TracingTransformer:
    config = SimpleNamespace(sample_size=2, in_channels=4, out_channels=8)
    dtype = torch.float32

    def __init__(self):
        self.calls = []

    def __call__(self, model_input, *, timestep, class_labels):
        signal = model_input * 0.25 + class_labels.reshape(-1, 1, 1, 1) * 0.001
        signal += timestep.reshape(-1, 1, 1, 1) * 0.001
        variance = model_input + 2.0 + class_labels.reshape(-1, 1, 1, 1) * 0.002
        output = torch.cat([signal, variance], dim=1)
        self.calls.append((model_input.clone(), output.clone()))
        return SimpleNamespace(sample=output)


class TracingScheduler:
    def __init__(self, mode, variance_type):
        self.variance_type = variance_type
        self.config = SimpleNamespace(variance_type=variance_type)
        self.calls = []
        if mode != "absent":
            factor, offset = {"identity": (1.0, 0.0), "offset": (1.0, 0.05), "scale": (0.5, 0.0)}[
                mode
            ]
            self.scale_model_input = lambda sample, timestep: sample * factor + offset

    def set_timesteps(self, count):
        assert count == 2
        self.timesteps = [torch.tensor(2), torch.tensor(1)]

    def step(self, model_output, timestep, sample):
        self.calls.append((sample.clone(), model_output.clone()))
        return SimpleNamespace(prev_sample=sample - 0.1 * model_output[:, :4])


class TracingVae:
    config = SimpleNamespace(scaling_factor=0.5)

    def decode(self, latents):
        self.latents = latents.clone()
        return SimpleNamespace(sample=latents[:, :3])


def assert_scheduler_contract(
    reference_text, tmp_path, monkeypatch, mode, guidance_scale, cfg_on_3_channels, variance_type
):
    initial = torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2) / 100 - 0.2
    transformer = TracingTransformer()
    scheduler = TracingScheduler(mode, variance_type)
    vae = TracingVae()
    with score_outputs._diffusers_stubs():
        module = score_outputs._load_module_from_text(
            "dit_state_contract", reference_text, tmp_path
        )
        monkeypatch.setattr(module, "randn_tensor", lambda **kwargs: initial.clone())
        pipeline = module.DiTPipeline(transformer=transformer, scheduler=scheduler, vae=vae)
        pipeline(
            class_labels=[1, 7],
            guidance_scale=guidance_scale,
            cfg_on_3_channels=cfg_on_3_channels,
            num_inference_steps=2,
            output_type="np",
        )

    assert len(transformer.calls) == len(scheduler.calls) == 2
    expected_state = initial
    for model_call, scheduler_call in zip(transformer.calls, scheduler.calls, strict=True):
        guided = guidance_scale > 1
        raw_sample = torch.cat([expected_state, expected_state]) if guided else expected_state
        expected_model_input = raw_sample
        if mode == "offset":
            expected_model_input = raw_sample + 0.05
        elif mode == "scale":
            expected_model_input = raw_sample * 0.5
        torch.testing.assert_close(model_call[0], expected_model_input)
        torch.testing.assert_close(scheduler_call[0], raw_sample)

        expected_prediction = model_call[1].clone()
        if guided:
            channel_count = 3 if cfg_on_3_channels else 4
            conditional = expected_prediction[:2, :channel_count].clone()
            unconditional = expected_prediction[2:, :channel_count].clone()
            combined = unconditional + guidance_scale * (conditional - unconditional)
            expected_prediction[:2, :channel_count] = combined
            expected_prediction[2:, :channel_count] = combined
        if variance_type == "fixed_small":
            expected_prediction = expected_prediction[:, :4]
        torch.testing.assert_close(scheduler_call[1], expected_prediction)
        expected_state = raw_sample - 0.1 * expected_prediction[:, :4]
        if guided:
            expected_state = expected_state[:2]
    torch.testing.assert_close(vae.latents, expected_state / vae.config.scaling_factor)


@pytest.mark.parametrize("mode", ["absent", "identity", "offset", "scale"])
@pytest.mark.parametrize("guidance_scale", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("cfg_on_3_channels", [False, True])
@pytest.mark.parametrize("variance_type", ["fixed_small", "learned", "learned_range"])
def test_reference_obeys_scheduler_state_and_guidance_contract(
    reference_text, tmp_path, monkeypatch, mode, guidance_scale, cfg_on_3_channels, variance_type
):
    assert_scheduler_contract(
        reference_text,
        tmp_path,
        monkeypatch,
        mode,
        guidance_scale,
        cfg_on_3_channels,
        variance_type,
    )


@pytest.mark.parametrize("mode", ["offset", "scale"])
@pytest.mark.parametrize(
    "original,replacement,cfg_on_3_channels,variance_type",
    [
        (
            "model_input = self.scheduler.scale_model_input(latent_model_input, t)",
            "model_input = latent_model_input",
            True,
            "learned_range",
        ),
        (
            "model_input = self.scheduler.scale_model_input(latent_model_input, t)",
            "self.scheduler.scale_model_input(latent_model_input, t)",
            True,
            "learned_range",
        ),
        (
            "model_input = self.scheduler.scale_model_input(latent_model_input, t)",
            "model_input = self.scheduler.scale_model_input(self.scheduler.scale_model_input(latent_model_input, t), t)",
            True,
            "learned_range",
        ),
        (
            "self.scheduler.step(model_output, t, latent_model_input)",
            "self.scheduler.step(model_output, t, model_input)",
            True,
            "learned_range",
        ),
        (
            "model_output = noise_pred",
            "model_output = noise_pred[:, :latent_channels]",
            True,
            "learned_range",
        ),
        (
            "cfg_channels = 3 if cfg_on_3_channels else latent_channels",
            "cfg_channels = noise_pred.shape[1]",
            False,
            "learned_range",
        ),
        (
            "cfg_channels = 3 if cfg_on_3_channels else latent_channels",
            "cfg_channels = latent_channels",
            True,
            "learned_range",
        ),
        (
            "cfg_channels = 3 if cfg_on_3_channels else latent_channels",
            "cfg_channels = 3",
            False,
            "learned_range",
        ),
        (
            "model_output, _ = torch.split(noise_pred, latent_channels, dim=1)",
            "model_output = noise_pred",
            True,
            "fixed_small",
        ),
    ],
)
def test_internal_contract_rejects_scaling_guidance_and_variance_mutations(
    reference_text,
    tmp_path,
    monkeypatch,
    mode,
    original,
    replacement,
    cfg_on_3_channels,
    variance_type,
):
    assert reference_text.count(original) == 1
    mutant = reference_text.replace(original, replacement)
    with pytest.raises(AssertionError):
        assert_scheduler_contract(
            mutant, tmp_path, monkeypatch, mode, 1.5, cfg_on_3_channels, variance_type
        )


def test_reference_manifest_matches_installed_bytes(reference_text):
    manifest = json.loads((DATA / "reference/stage1_manifest.json").read_text())
    assert (
        manifest["canonical_files"]["reference/pipeline_dit.py"]
        == hashlib.sha256(reference_text.encode()).hexdigest()
    )


def test_scaled_step_state_is_still_rejected(reference_text):
    candidate = reference_text.replace(
        "self.scheduler.step(model_output, t, latent_model_input)",
        "self.scheduler.step(model_output, t, model_input)",
    )
    assert candidate != reference_text
    result = score_outputs.score_submission_text(candidate, reference_text)
    assert not result.passed
    assert result.score < 1.0
    assert len(result.cases) == 5


def test_guidance_faults_still_lose_original_score(reference_text):
    candidate = reference_text.replace(
        "cfg_channels = 3 if cfg_on_3_channels else latent_channels",
        "cfg_channels = latent_channels",
    )
    assert candidate != reference_text
    result = score_outputs.score_submission_text(candidate, reference_text)
    assert not result.passed
    assert result.score < 1.0
    assert len(result.cases) == 5
