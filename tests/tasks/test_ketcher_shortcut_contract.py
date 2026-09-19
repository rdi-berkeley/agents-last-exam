from __future__ import annotations

import configparser
import json
from pathlib import Path, PureWindowsPath

import pytest

from tasks.physical_sciences.ketcher_smiles_reproduction.main import (
    VARIANTS,
    KetcherSmilesTaskConfig,
)


ROOT = Path(__file__).resolve().parents[2]


def test_ketcher_prompt_card_and_metadata_name_url_shortcut():
    variant, molecule_id, ketcher_url = VARIANTS[0]
    config = KetcherSmilesTaskConfig(
        VARIANT_NAME=variant, MOLECULE_ID=molecule_id, KETCHER_URL=ketcher_url
    )
    card = json.loads(config.task_card_path.read_text())["taskPrompt"]

    assert PureWindowsPath(config.software_shortcut).name == "Ketcher.url"
    assert config.to_metadata()["software_shortcut"] == config.software_shortcut
    assert config.task_description.replace(config.task_dir, variant).strip() == card.strip()
    for prompt in (config.task_description, card):
        assert "Ketcher.url" in prompt
        assert "Ketcher.lnk" not in prompt
        assert ketcher_url in prompt


def test_ketcher_shortcut_matches_staged_internet_shortcut():
    variant, molecule_id, ketcher_url = VARIANTS[0]
    config = KetcherSmilesTaskConfig(
        VARIANT_NAME=variant, MOLECULE_ID=molecule_id, KETCHER_URL=ketcher_url
    )
    software = (
        ROOT
        / "task-data-hf/extracted/physical_sciences/ketcher_smiles_reproduction"
        / variant
        / "software"
    )
    if not software.exists():
        pytest.skip("Ketcher release data is not installed")
    shortcut = software / PureWindowsPath(config.software_shortcut).name
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(shortcut.read_text(encoding="utf-8-sig"))

    assert parser["InternetShortcut"]["URL"] == ketcher_url
