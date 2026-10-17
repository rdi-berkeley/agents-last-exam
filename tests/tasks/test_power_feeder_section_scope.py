import json

from test_power_feeder_contract import PARAMS, TASK, builder, fixture_xml, validate_independently


def test_empty_sections_require_normal_source_connection(tmp_path):
    path = fixture_xml(
        tmp_path,
        extra=[
            ("Breaker", "empty_branch", ("n4", "energized_empty"), {}),
            (
                "Breaker",
                "open_boundary",
                ("energized_empty", "isolated_empty"),
                {"Switch.normalOpen": "true"},
            ),
        ],
    )
    model = builder.build_model(builder.read_cim(path))
    result, evidence = builder.calculate(model, PARAMS)
    validate_independently(path, PARAMS, result, evidence)
    assert model["sections"]["energized_empty"] == {"energized_empty"}
    assert "isolated_empty" not in model["graph"]
    for table in ("fault_rows", "scheduled_rows"):
        names = {row["name"] for row in result[table]}
        assert "energized_empty" in names
        assert "isolated_empty" not in names
        assert "n0" in names
    boundary = next(row for row in result["device_fault_rows"] if row["name"] == "open_boundary")
    assert boundary["section"] == "energized_empty"
    assert boundary["device_count"] == 1
    assert boundary["affected_users"] == 0


def test_public_contract_and_card_share_section_scope():
    contract = (TASK / "scientific_contract.md").read_text()
    card = json.loads((TASK / "task_card.json").read_text())
    assert contract.strip() in card["taskPrompt"]
    assert "the source-connected normal-state conducting graph only" in contract
    assert "including open ties" in contract
    assert "connectivity-node-only regions" in contract
