import hashlib
import json
from pathlib import Path

import pytest

from tasks.life_sciences.amber_minimization_script_prep_instance_1.scripts import (
    verify_submission as minimization,
)
from tasks.life_sciences.amber_three_stage_mmgbsa_workflow_instance_1.scripts import (
    verify_submission as mmgbsa,
)


LEAP = f"""source leaprc.protein.ff14SB
set default PBradii mbondi3
protein = loadpdb complex_structure.pdb
saveamberparm protein {minimization.SYSTEM_BASENAME}.prmtop {minimization.SYSTEM_BASENAME}.inpcrd
savepdb protein {minimization.SYSTEM_BASENAME}_fixed.pdb
quit
"""


@pytest.mark.parametrize("property_name", ["PBRadii", "PBradii"])
def test_minimization_native_radii_property_spellings(property_name):
    assert minimization._check_leap(LEAP.replace("PBradii", property_name)) == []


@pytest.mark.parametrize(
    "old,new",
    [
        ("mbondi3", "mbondi2"),
        ("mbondi3", "MBONDI3"),
        ("mbondi3", "mbondi3 extra"),
        ("mbondi3", ""),
        ("PBradii", "PBradiii"),
        ("set default", "# set default"),
        ("set default", "set Default"),
        ("mbondi3", "mbondi3\nset default PBRadii mbondi2"),
        ("ff14SB", "ff19SB"),
        ("complex_structure.pdb", "Complex_structure.pdb"),
        ("saveamberparm protein", "saveamberparm Protein"),
    ],
)
def test_minimization_radii_fix_preserves_scientific_and_case_checks(old, new):
    assert minimization._check_leap(LEAP.replace(old, new))


def test_actual_r04_mmgbsa_artifacts_unchanged():
    evidence_path = (
        Path(__file__).resolve().parents[3]
        / "release-staging/ale-task-fixes-20260913/post-validation-20260918"
        / "task-reviews-r04/last-linux-observed/mmgbsa-observed.json"
    )
    if not evidence_path.exists():
        pytest.skip("r04 MMGBSA artifacts are not staged")
    evidence = json.loads(evidence_path.read_bytes())
    payloads = {}
    for record in [evidence["run"], evidence["trajectory"], evidence["reference"], *evidence["artifacts"]]:
        source = Path(record["path"])
        payloads[source] = source.read_bytes()
        assert hashlib.sha256(payloads[source]).hexdigest() == record["sha256"]
    bundle = {
        Path(record["path"]).name: payloads[Path(record["path"])].decode()
        for record in evidence["artifacts"]
    }
    assert "set default PBradii mbondi3" in bundle["submit_min.sh"]
    result = mmgbsa.evaluate_output_bundle(
        bundle,
        present_files=sorted(bundle),
        hidden_reference_text=payloads[Path(evidence["reference"]["path"])].decode(),
    )
    assert result["passed"], result
    assert result["score"] == 1.0
    assert result["delta_total"] == -116.3475
    assert result["hidden_delta_total"] == -116.3296
    assert json.loads(payloads[Path(evidence["run"]["path"])])["score"] == 0
    assert {source: source.read_bytes() for source in payloads} == payloads
