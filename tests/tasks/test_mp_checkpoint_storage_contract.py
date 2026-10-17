from __future__ import annotations

import json
import os
import re
import runpy
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

TASK_PATH = Path("computing_math/mp_checkpoint_consolidation_v2/base")
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "task-data-hf/extracted"
RAW = Path(os.environ.get("ALE_COMPUTING_RAW_DATA", DEFAULT_DATA)) / TASK_PATH
REPAIRED = Path(os.environ.get("ALE_COMPUTING_REPAIR_DATA", DEFAULT_DATA)) / TASK_PATH
pytestmark = pytest.mark.skipif(
    not (RAW / "reference/expected_model.safetensors").is_file()
    or not (REPAIRED / "input/framework/config.py").is_file(),
    reason="Requires the checkpoint data and repaired public storage configuration",
)


def test_declared_storage_conventions_match_every_affected_raw_shard():
    configuration = runpy.run_path(str(REPAIRED / "input/framework/config.py"))["CheckpointConfig"]()
    splitter_type = runpy.run_path(str(RAW / "input/framework/parallel.py"))["TensorParallelSplitter"]
    model = json.loads((RAW / "input/reference_model/config.json").read_text())
    reference = load_file(str(RAW / "reference/expected_model.safetensors"))
    shards = {}
    for path in (RAW / "input/checkpoints").glob("*.pt"):
        match = re.fullmatch(r"shard_tp(\d+)_pp(\d+)_ep(\d+)\.pt", path.name)
        assert match is not None
        shards[tuple(map(int, match.groups()))] = torch.load(path, map_location="cpu", weights_only=True)
    tensor_size = max(rank[0] for rank in shards) + 1
    expert_size = max(rank[2] for rank in shards) + 1
    layer_counts = {
        pipeline_rank: 1 + max(int(key.split(".")[2]) for key in state if key.startswith("transformer.layers."))
        for (tensor_rank, pipeline_rank, expert_rank), state in shards.items()
        if tensor_rank == expert_rank == 0
    }
    splitter = splitter_type(tensor_size, model["num_attention_heads"], model["head_dim"])
    counts = dict.fromkeys(("dense_down", "moe_down", "dense_gate_up", "moe_gate_up"), 0)
    for (tensor_rank, pipeline_rank, expert_rank), state in shards.items():
        for key, stored in state.items():
            match = re.fullmatch(
                r"transformer.layers.(\d+)\.(mlp|moe.experts.(\d+))\.(down_proj|gate_up_proj).weight", key
            )
            if match is None:
                continue
            local_layer, branch, local_expert, projection = match.groups()
            layer = sum(layer_counts[index] for index in range(pipeline_rank)) + int(local_layer)
            family = "dense" if local_expert is None else "moe"
            branch = "mlp" if local_expert is None else f"moe.experts.{int(local_expert) * expert_size + expert_rank}"
            prefix = f"layers.{layer}.{branch}"
            if projection == "down_proj":
                logical = splitter.split_down_proj(reference[prefix + ".down_proj.weight"], tensor_rank)
                expected = logical.t() if configuration.transpose_down_proj else logical
                assert not torch.equal(stored, logical)
                category = family + "_down"
            else:
                gate = reference[prefix + ".gate_proj.weight"]
                up = reference[prefix + ".up_proj.weight"]
                logical = splitter.split_gate_up_fused(gate, up, tensor_rank)
                if family == "moe" and configuration.moe_gate_up_row_order == "up_gate":
                    expected = splitter.split_gate_up_fused(up, gate, tensor_rank)
                    assert not torch.equal(stored, logical)
                else:
                    expected = logical
                category = family + "_gate_up"
            assert torch.equal(stored, expected), key
            counts[category] += 1
    assert counts == {"dense_down": 16, "moe_down": 32, "dense_gate_up": 16, "moe_gate_up": 32}


def test_repaired_instructions_reference_existing_inputs_without_answer_logits():
    instructions = (REPAIRED / "input/task_instructions.md").read_text()
    assert "machine_learning/" not in instructions
    assert "logits.pt" not in instructions
    assert "input_ids.pt" not in instructions
    guest_root = "/media/user/data/agenthle/" + TASK_PATH.as_posix()
    paths = re.findall(r"`(" + re.escape(guest_root) + r"/input[^`]*)`", instructions)
    assert len(paths) >= 8
    for path in paths:
        relative = path.removeprefix(guest_root + "/")
        assert (RAW / relative).exists() or (REPAIRED / relative).exists(), relative
