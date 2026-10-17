import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tasks.visual_media.butterfly_flap_animation import main as butterfly


def test_butterfly_contract_is_software_independent():
    folder = Path(butterfly.__file__).parent
    card = json.loads((folder / "task_card.json").read_text())
    spec = (folder / "task_spec.md").read_text()
    description = butterfly.ButterflyFlapConfig().task_description
    for text in (description, spec, json.dumps(card)):
        assert "After Effects" not in text
        assert "launch_after_effects" not in text
    for text in (description, spec, card["taskPrompt"]):
        assert "4 complete" in text
        assert "30 fps" in text
        assert "4.0 to 5.5" in text
        assert "H.264" in text
    assert card["vm"]["snapshot"] == "cpu-free"
    assert card["vm"]["machineType"] == "n2-standard-8"


@pytest.mark.asyncio
async def test_setup_replaces_baked_software_requirement(monkeypatch):
    setup = AsyncMock()
    monkeypatch.setattr(butterfly, "_setup", setup)
    session = SimpleNamespace(write_file=AsyncMock())
    task = SimpleNamespace(metadata={"task_spec": r"C:\task\input\task_spec.md"})
    await butterfly.start(task, session)
    setup.assert_awaited_once_with(task, session)
    session.write_file.assert_awaited_once_with(
        task.metadata["task_spec"],
        (Path(butterfly.__file__).parent / "task_spec.md").read_text(encoding="utf-8"),
    )
