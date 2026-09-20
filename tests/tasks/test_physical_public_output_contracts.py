from __future__ import annotations

import configparser
import json
from pathlib import Path, PureWindowsPath

import pytest

from tasks.physical_sciences._shared.materials_science import _common as scorer
from tasks.physical_sciences.egt710_table1_smiles_extraction.main import (
    CHEMINFO_URL,
    EGT710TaskConfig,
)
from tasks.physical_sciences.silicon_bse_absorption.main import SiliconBSEAbsorptionConfig


ROOT = Path(__file__).resolve().parents[2]


def test_egt710_prompt_card_and_metadata_name_url_shortcut():
    config = EGT710TaskConfig(VARIANT_NAME="table1")
    card = json.loads(config.task_card_path.read_text())["taskPrompt"]

    assert PureWindowsPath(config.software_shortcut).name == "ChemInfo.url"
    assert config.to_metadata()["software_shortcut"] == config.software_shortcut
    assert config.task_description.replace(config.task_dir, "table1").strip() == card.strip()
    for prompt in (config.task_description, card):
        assert "ChemInfo.url" in prompt
        assert "ChemInfo.lnk" not in prompt
        assert CHEMINFO_URL in prompt


def test_egt710_shortcut_matches_staged_internet_shortcut():
    config = EGT710TaskConfig(VARIANT_NAME="table1")
    software = (
        ROOT
        / "task-data-hf/extracted/physical_sciences/egt710_table1_smiles_extraction"
        / "table1/software"
    )
    if not software.exists():
        pytest.skip("EGT710 release data is not installed")
    shortcut = software / PureWindowsPath(config.software_shortcut).name
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(shortcut.read_text(encoding="utf-8-sig"))

    assert parser["InternetShortcut"]["URL"] == CHEMINFO_URL


def test_silicon_prompt_and_card_disclose_existing_numeric_contract():
    config = SiliconBSEAbsorptionConfig()
    card = json.loads(config.task_card_path.read_text())["taskPrompt"]

    assert config.task_description.replace(config.task_dir, "base") == card
    for prompt in (config.task_description, card):
        for required in (
            "All seven `.dat` files",
            "nonempty UTF-8, whitespace-separated tables of finite numeric values",
            "Blank lines and lines starting with `#` after optional whitespace are allowed",
            "QE `&plot ... /` headers, are not accepted",
            "at least seven columns",
            "(1) numeric field, conventionally spin, ignored by the scorer",
            "(2) band index",
            "(3-5) Cartesian kx, ky, kz",
            "(6) mean-field energy in eV",
            "(7) GW quasiparticle energy in eV",
            "Additional numeric columns are allowed",
            "band indices <= 4 as valence and >= 5 as conduction",
        ):
            assert required in prompt
        for filename in scorer.SILICON_BSE_ABSORPTION_SPEC.required_files:
            assert f"/output/{filename}`" in prompt


@pytest.mark.parametrize("filename", scorer.SILICON_BSE_ABSORPTION_SPEC.numeric_files)
def test_documented_numeric_format_accepts_bom_comments_and_whitespace(filename):
    table = b"\xef\xbb\xbf  # column names\n\n1\t2.5  -3e-2\n  # another comment\n"

    assert scorer._parse_numeric_table(table, filename) == [[1.0, 2.5, -0.03]]


@pytest.mark.parametrize(
    ("table", "reason"),
    [
        (b" &plot nbnd= 16, nks= 8 /\n", "line 1 is not purely numeric"),
        (b"spin band kx ky kz E_DFT E_GW\n", "line 1 is not purely numeric"),
        (b"1 2 nan\n", "contains a non-finite value"),
        (b"1 2 inf\n", "contains a non-finite value"),
        (b"# no data\n\n", "no numeric data rows found"),
    ],
)
def test_documented_numeric_rejections_remain_unchanged(table, reason):
    with pytest.raises(ValueError, match=reason):
        scorer._parse_numeric_table(table, "bandstructure.dat")


@pytest.mark.parametrize("extra_columns", [[], [123.0, -456.0]])
def test_documented_band_columns_extract_gaps_and_topology(extra_columns):
    rows = [
        [99.0, 4.0, 0.0, 0.0, 0.0, 1.0, 2.0] + extra_columns,
        [42.0, 5.0, 0.8, 0.0, 0.0, 1.6, 3.1] + extra_columns,
    ]

    metrics = scorer._silicon_band_metrics(rows, "bandstructure.dat")

    assert metrics == pytest.approx(
        {
            "dft_gap_ev": 0.6,
            "gw_gap_ev": 1.1,
            "dft_vbm_gamma_distance": 0.0,
            "gw_vbm_gamma_distance": 0.0,
            "dft_cbm_delta_distance": 0.0,
            "gw_cbm_delta_distance": 0.0,
        }
    )


def test_band_rows_still_require_at_least_seven_columns():
    with pytest.raises(ValueError, match="needs at least seven columns"):
        scorer._silicon_band_metrics([[1, 4, 0, 0, 0, 1]], "bandstructure.dat")


@pytest.mark.parametrize("band", [4, 5])
def test_band_rows_still_require_both_valence_and_conduction(band):
    with pytest.raises(ValueError, match="could not identify silicon valence/conduction"):
        scorer._silicon_band_metrics([[1, band, 0, 0, 0, 1, 2]], "bandstructure.dat")


def test_qe_header_still_returns_original_format_gate_zero():
    spec = scorer.SILICON_BSE_ABSORPTION_SPEC
    reference = {filename: b"1 2\n" for filename in spec.numeric_files}
    reference.update({filename: scorer.PNG_SIGNATURE for filename in spec.png_files})
    candidate = dict(reference)
    candidate["bandstructure.dat"] = b" &plot nbnd=  16, nks=     8 /\n"

    result = scorer.score_file_payloads(
        spec, agent_payloads=candidate, reference_payloads=reference
    )

    assert result == {
        "score": 0.0,
        "passed": False,
        "failures": ["bandstructure.dat: line 1 is not purely numeric"],
    }
