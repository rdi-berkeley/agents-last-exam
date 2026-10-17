import hashlib
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "tasks/engineering/inner_support_elevation_optimization/scripts"
)


@pytest.fixture
def checkpoint(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    restart = tmp_path / "restart"
    restart.mkdir()
    state = b"retained constitutive state"
    (restart / "latest.rest").write_bytes(state)
    review = {"input_hashes": {"family.json": "unchanged"}}
    metadata = {
        **review,
        "label": "stage_5_dewater_1",
        "native_state_sha256": hashlib.sha256(state).hexdigest(),
        "receipt": {"elapsed_s": 1725.2},
    }
    (restart / "latest.json").write_text(json.dumps(metadata))
    (tmp_path / "receipt.json").write_text("old receipt")
    (tmp_path / "native_stdout.log").write_text("old log")
    return tmp_path, review, restart / "latest", state


def test_same_history_resume_archives_logs_without_copying_or_mutating_native_state(checkpoint):
    from native_replay_case import prepare_output

    output, review, restart, state = checkpoint
    receipt = prepare_output(output, review, restart, True)
    assert receipt["checkpoint_label"] == "stage_5_dewater_1"
    assert (output / "before-continuation-receipt.json").read_text() == "old receipt"
    assert (output / "before-continuation-native_stdout.log").read_text() == "old log"
    assert restart.with_suffix(".rest").read_bytes() == state
    assert len(list(output.rglob("*.rest"))) == 1
    with pytest.raises(ValueError, match="already has"):
        prepare_output(output, review, restart, True)


def test_changed_checkpoint_is_rejected_before_old_receipts_are_moved(checkpoint):
    from native_replay_case import prepare_output

    output, review, restart, _ = checkpoint
    restart.with_suffix(".rest").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_output(output, review, restart, True)
    assert (output / "receipt.json").read_text() == "old receipt"


def test_other_family_or_other_result_checkpoint_cannot_resume_in_place(checkpoint):
    from native_replay_case import prepare_output

    output, review, restart, _ = checkpoint
    with pytest.raises(ValueError, match="frozen native inputs"):
        prepare_output(output, {"input_hashes": {"family.json": "different"}}, restart, True)
    with pytest.raises(ValueError, match="this result"):
        prepare_output(output, review, output / "other/latest", True)
