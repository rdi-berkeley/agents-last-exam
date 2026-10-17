import asyncio
from collections import Counter, defaultdict, deque
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks/engineering/power_10kv_feeder_reliability_001"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
CIM = "{http://iec.ch/TC57/2016/CIM-schema-cim17#}"
SPEC = importlib.util.spec_from_file_location(
    "power_reference_v2", TASK / "scripts/build_reference_v2.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
SPEC = importlib.util.spec_from_file_location(
    "power_scorer", TASK / "scripts/verify_reliability_indices.py"
)
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)

PARAMS = {
    "line_fault": {
        "lambda_overhead": 0.1,
        "lambda_cable": 0.2,
        "t_manual_isolation_h": 0.5,
        "t_auto_isolation_h": 0.1,
        "t_repair_h": 4,
    },
    "scheduled_outage": {
        "lambda_scheduled_overhead": 0.03,
        "lambda_scheduled_cable": 0.04,
        "t_scheduled_overhead_h": 5,
        "t_scheduled_cable_h": 6,
    },
    "device_fault": {
        "lambda_breaker": 0.01,
        "lambda_transformer": 0.02,
        "t_repair_breaker_h": 2,
        "t_repair_transformer_h": 3,
    },
}


def fixture_xml(directory, *, tie_open=True, extra=()):
    records = [
        (
            "Feeder",
            "feeder",
            (),
            {"Feeder.IsCurrentFeeder": "true", "Feeder.NormalEnergizingSubstation": "station"},
        ),
        ("Substation", "station", (), {}),
        ("BaseVoltage", "voltage", (), {}),
        ("BaseVoltage", "low_voltage", (), {}),
        ("BusbarSection", "bus", ("n0",), {"Equipment.EquipmentContainer": "station"}),
        ("Breaker", "head", ("n0", "n1"), {"Equipment.EquipmentContainer": "station"}),
        ("ACLineSegment", "trunk", ("n1", "n2"), {"IdentifiedObject.Length": "1000"}),
        ("Breaker", "branch_a", ("n2", "n3"), {}),
        (
            "ACLineSegment",
            "cable",
            ("n3", "n4"),
            {"IdentifiedObject.Length": "2000", "PowerSystemResource.PSRType": "PD_20100000"},
        ),
        ("Breaker", "branch_b", ("n2", "n5"), {}),
        ("ACLineSegment", "branch_line", ("n5", "n6"), {"IdentifiedObject.Length": "500"}),
        ("Breaker", "tie", ("n4", "n6"), {"Switch.normalOpen": str(tie_open).lower()}),
        ("PowerTransformer", "load_root", ("n2", "shared_lv"), {}),
        ("PowerTransformer", "load_a", ("n4", "shared_lv"), {}),
        ("PowerTransformer", "load_b", ("n6", "shared_lv"), {}),
        *extra,
    ]
    root = ET.Element(RDF + "RDF")
    nodes = set()
    for kind, identifier, endpoints, custom in records:
        element = ET.SubElement(root, CIM + kind, {RDF + "ID": identifier})
        attrs = {"IdentifiedObject.name": "duplicate-name"}
        if endpoints:
            attrs.update(
                {
                    "Equipment.EquipmentContainer": "feeder",
                    "ConductingEquipment.BaseVoltage": "voltage",
                }
            )
        if kind in {"Breaker", "Disconnector", "Fuse"}:
            attrs.update({"Switch.normalOpen": "false", "Switch.open": "true"})
        if kind == "ACLineSegment":
            attrs["PowerSystemResource.PSRType"] = "PD_10100000"
        attrs.update(custom)
        for field, value in attrs.items():
            child = ET.SubElement(element, CIM + field)
            if field.endswith(
                (
                    ".EquipmentContainer",
                    ".BaseVoltage",
                    ".PSRType",
                    ".NormalEnergizingSubstation",
                    ".EquipmentBelongtoFeeder",
                )
            ):
                child.set(RDF + "resource", "#" + value)
            else:
                child.text = value
        for index, node in enumerate(endpoints):
            terminal_id = f"{identifier}_terminal_{index}"
            terminal = ET.SubElement(root, CIM + "Terminal", {RDF + "ID": terminal_id})
            ET.SubElement(
                terminal, CIM + "Terminal.ConductingEquipment", {RDF + "resource": "#" + identifier}
            )
            ET.SubElement(
                terminal, CIM + "Terminal.ConnectivityNode", {RDF + "resource": "#" + node}
            )
            nodes.add(node)
            if kind == "PowerTransformer":
                end = ET.SubElement(
                    root, CIM + "PowerTransformerEnd", {RDF + "ID": f"{identifier}_end_{index}"}
                )
                for field, value in {
                    "PowerTransformerEnd.PowerTransformer": identifier,
                    "TransformerEnd.BaseVoltage": "voltage" if index == 0 else "low_voltage",
                    "TransformerEnd.Terminal": terminal_id,
                }.items():
                    ET.SubElement(end, CIM + field, {RDF + "resource": "#" + value})
    for node in sorted(nodes):
        ET.SubElement(root, CIM + "ConnectivityNode", {RDF + "ID": node})
    path = directory / "synthetic.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def independent_oracle(xml_path, parameters):
    """Reconstruct node-only multiedges and sum decimal equipment/load events."""
    elements = {element.attrib[RDF + "ID"]: element for element in ET.parse(xml_path).getroot()}

    def prop(identifier, field):
        for child in elements[identifier]:
            if child.tag.split("}")[-1] == field:
                return child.attrib.get(RDF + "resource", child.text or "").strip().lstrip("#")
        return ""

    kinds = {key: element.tag.split("}")[-1] for key, element in elements.items()}
    (current,) = [key for key in elements if prop(key, "Feeder.IsCurrentFeeder") in ("true", "1")]
    assigned = {
        key
        for key in elements
        if current
        in (
            prop(key, "Equipment.EquipmentContainer"),
            prop(key, "Equipment.EquipmentBelongtoFeeder"),
        )
    }
    lines = {key for key in assigned if kinds[key] == "ACLineSegment"}
    transformers = {key for key in assigned if kinds[key] == "PowerTransformer"}
    (voltage,) = {prop(key, "ConductingEquipment.BaseVoltage") for key in lines}
    station = prop(current, "Feeder.NormalEnergizingSubstation")
    terminals = defaultdict(list)
    by_terminal = {}
    for key in elements:
        if kinds[key] == "Terminal":
            node = prop(key, "Terminal.ConnectivityNode")
            terminals[prop(key, "Terminal.ConductingEquipment")].append(node)
            by_terminal[key] = node
    equipment = set()
    for key in elements:
        if kinds[key] not in {
            "Breaker",
            "Fuse",
            "Disconnector",
            "ACLineSegment",
            "Junction",
            "BusbarSection",
        }:
            continue
        ancestors = {key}
        pending = [key]
        while pending:
            ancestor = pending.pop()
            for field in (
                "Equipment.EquipmentContainer",
                "VoltageLevel.Substation",
                "Bay.VoltageLevel",
            ):
                parent = prop(ancestor, field)
                if parent and parent not in ancestors:
                    ancestors.add(parent)
                    pending.append(parent)
        if key in assigned or (
            station in ancestors and prop(key, "ConductingEquipment.BaseVoltage") == voltage
        ):
            equipment.add(key)
    head_terminal = prop(current, "Feeder.NormalHeadTerminal")
    if head_terminal:
        source = by_terminal[head_terminal]
    else:
        (bus,) = [key for key in equipment - assigned if kinds[key] == "BusbarSection"]
        source = terminals[bus][0]
    load_node = {}
    for key in elements:
        if (
            kinds[key] == "PowerTransformerEnd"
            and prop(key, "TransformerEnd.BaseVoltage") == voltage
        ):
            transformer = prop(key, "PowerTransformerEnd.PowerTransformer")
            if transformer in transformers:
                assert transformer not in load_node
                load_node[transformer] = by_terminal[prop(key, "TransformerEnd.Terminal")]
    assert set(load_node) == transformers
    open_switches = {
        key
        for key in equipment
        if kinds[key] in {"Breaker", "Fuse", "Disconnector"}
        and prop(key, "Switch.normalOpen") in ("true", "1")
    }
    breakers = {key for key in equipment if kinds[key] == "Breaker"}
    edges = [
        (key, first, second)
        for key in equipment - open_switches
        for first in terminals[key]
        for second in terminals[key]
        if first != second
    ]
    adjacency = defaultdict(list)
    for key, first, second in edges:
        adjacency[first].append((second, key))

    def reachable(blocked=frozenset(), failed=None):
        visited = {source} - blocked
        pending = deque(visited)
        while pending:
            for other, key in adjacency[pending.popleft()]:
                if key != failed and other not in blocked and other not in visited:
                    visited.add(other)
                    pending.append(other)
        return visited

    live_nodes = reachable()
    assert all(node in live_nodes for node in load_node.values())
    parents = {node: node for node in live_nodes}

    def representative(node):
        while parents[node] != node:
            node = parents[node]
        return node

    for key, first, second in edges:
        if key not in breakers and first in live_nodes and second in live_nodes:
            first_root, second_root = representative(first), representative(second)
            parents[max(first_root, second_root)] = min(first_root, second_root)
    groups = defaultdict(set)
    for node in live_nodes:
        groups[representative(node)].add(node)
    owner = {node: name for name, members in groups.items() for node in members}
    graph = defaultdict(set)
    for key in breakers - open_switches:
        first, second = terminals[key]
        graph[owner[first]].add(owner[second])
        graph[owner[second]].add(owner[first])
    distances = {owner[source]: 0}
    pending = deque(distances)
    while pending:
        name = pending.popleft()
        for other in graph[name]:
            if other not in distances:
                distances[other] = distances[name] + 1
                pending.append(other)
    section_lines = defaultdict(list)
    for key in sorted(lines):
        endpoints = {owner[node] for node in terminals[key]}
        (name,) = endpoints
        section_lines[name].append(key)
    section_loads = defaultdict(list)
    for key, node in load_node.items():
        section_loads[owner[node]].append(key)
    affected = {
        name: sorted(key for key, node in load_node.items() if node not in reachable(members))
        for name, members in groups.items()
    }
    breaker_affected = {
        key: sorted(load for load, node in load_node.items() if node not in reachable(failed=key))
        for key in breakers
    }
    section_breakers = {}
    for key in breakers:
        choices = {owner[node] for node in terminals[key] if node in live_nodes}
        section_breakers[key] = sorted(choices, key=lambda name: (-distances[name], name))[0]
    params = json.loads(json.dumps(parameters), parse_float=Decimal, parse_int=Decimal)
    zero = Decimal(0)
    fault_time = params["line_fault"]["t_manual_isolation_h"] + params["line_fault"]["t_repair_h"]
    breaker_time = (
        params["line_fault"]["t_manual_isolation_h"] + params["device_fault"]["t_repair_breaker_h"]
    )
    totals = {suffix: [zero, zero] for suffix in "FDS"}
    events = {suffix: {key: [zero, zero] for key in transformers} for suffix in "FDS"}
    expected = {
        name: {"length_km": zero, "lambda_i": zero, "scheduled_rate": zero, "scheduled_hours": zero}
        for name in groups
    }
    known_lengths = defaultdict(lambda: zero)
    missing = []
    missing_terms = {}
    for key in sorted(lines):
        name = owner[terminals[key][0]]
        raw = prop(key, "IdentifiedObject.Length")
        length = Decimal(raw) / 1000 if raw else zero
        if not raw:
            missing.append(key)
        kind = {"PD_10100000": "overhead", "PD_20100000": "cable"}[
            prop(key, "PowerSystemResource.PSRType")
        ]
        if not raw:
            unit_events = [zero, zero, zero, zero]
            for load in affected[name]:
                unit_events[0] += params["line_fault"][f"lambda_{kind}"]
                unit_events[1] += params["line_fault"][f"lambda_{kind}"] * fault_time
                unit_events[2] += params["scheduled_outage"][f"lambda_scheduled_{kind}"]
                unit_events[3] += (
                    params["scheduled_outage"][f"lambda_scheduled_{kind}"]
                    * params["scheduled_outage"][f"t_scheduled_{kind}_h"]
                )
            missing_terms[key] = {
                "line_type": kind,
                **{
                    field: value / len(transformers)
                    for field, value in zip(
                        (
                            "SAIFI_F_per_km",
                            "SAIDI_F_h_per_km",
                            "SAIFI_S_per_km",
                            "SAIDI_S_h_per_km",
                        ),
                        unit_events,
                    )
                },
            }
        known_lengths[kind] += length
        rate = length * params["line_fault"][f"lambda_{kind}"]
        scheduled_rate = length * params["scheduled_outage"][f"lambda_scheduled_{kind}"]
        scheduled_hours = scheduled_rate * params["scheduled_outage"][f"t_scheduled_{kind}_h"]
        expected[name]["length_km"] += length
        expected[name]["lambda_i"] += rate
        expected[name]["scheduled_rate"] += scheduled_rate
        expected[name]["scheduled_hours"] += scheduled_hours
        for load in affected[name]:
            events["F"][load][0] += rate
            events["F"][load][1] += rate * fault_time
            events["S"][load][0] += scheduled_rate
            events["S"][load][1] += scheduled_hours
    for key in breakers:
        for load in breaker_affected[key]:
            events["D"][load][0] += params["device_fault"]["lambda_breaker"]
            events["D"][load][1] += params["device_fault"]["lambda_breaker"] * breaker_time
    for load in transformers:
        events["D"][load][0] += params["device_fault"]["lambda_transformer"]
        events["D"][load][1] += (
            params["device_fault"]["lambda_transformer"]
            * params["device_fault"]["t_repair_transformer_h"]
        )
    for suffix in "FDS":
        totals[suffix] = [
            sum((values[index] for values in events[suffix].values()), zero) / len(transformers)
            for index in (0, 1)
        ]
    return {
        "feeder": prop(current, "IdentifiedObject.name"),
        "N_T": len(transformers),
        "groups": dict(groups),
        "section_lines": dict(section_lines),
        "section_loads": {name: sorted(values) for name, values in section_loads.items()},
        "affected": affected,
        "breaker_affected": breaker_affected,
        "breaker_owner": section_breakers,
        "expected": expected,
        "totals": totals,
        "events": events,
        "known_lengths": {kind: known_lengths[kind] for kind in ("overhead", "cable")},
        "missing": missing,
        "missing_terms": missing_terms,
        "open_breakers": sorted(breakers & open_switches),
    }


def validate_independently(xml_path, params, result, evidence):
    audit = independent_oracle(xml_path, params)
    assert result["N_T"] == audit["N_T"]
    assert result["feeder"] == audit["feeder"]
    assert set(evidence["sections"]) == set(audit["groups"])
    for name, section in evidence["sections"].items():
        assert section["lines"] == audit["section_lines"].get(name, [])
        assert section["local_loads"] == audit["section_loads"].get(name, [])
        assert section["affected_loads"] == audit["affected"][name]
    assert Counter(
        key for section in evidence["sections"].values() for key in section["lines"]
    ) == Counter(key for members in audit["section_lines"].values() for key in members)
    assert Counter(
        key for section in evidence["sections"].values() for key in section["local_loads"]
    ) == Counter(key for members in audit["section_loads"].values() for key in members)
    assert len(result["fault_rows"]) == len(result["scheduled_rows"]) == len(audit["groups"])
    for row in result["fault_rows"]:
        expected = audit["expected"][row["name"]]
        count = len(audit["affected"][row["name"]])
        assert row["perm_users"] == count
        assert row["length_km"] == pytest.approx(float(expected["length_km"]), rel=1e-12, abs=1e-12)
        assert row["lambda_i"] == pytest.approx(float(expected["lambda_i"]), rel=1e-12, abs=1e-12)
        assert row["t_iso_h"] == params["line_fault"]["t_manual_isolation_h"]
        assert row["r_i_h"] == pytest.approx(row["t_iso_h"] + params["line_fault"]["t_repair_h"])
        assert row["lambda_N"] == pytest.approx(
            float(expected["lambda_i"]) * count, rel=1e-12, abs=1e-12
        )
        assert row["lambda_N_r"] == pytest.approx(
            row["lambda_N"] * row["r_i_h"], rel=1e-12, abs=1e-12
        )
    for row in result["scheduled_rows"]:
        expected = audit["expected"][row["name"]]
        count = len(audit["affected"][row["name"]])
        assert row["users"] == count
        assert row["length_km"] == pytest.approx(float(expected["length_km"]), rel=1e-12, abs=1e-12)
        assert row["lambda_N"] == pytest.approx(
            float(expected["scheduled_rate"]) * count, rel=1e-12, abs=1e-12
        )
        assert row["lambda_N_r"] == pytest.approx(
            float(expected["scheduled_hours"]) * count, rel=1e-12, abs=1e-12
        )
    transformer_rows = {
        row["name"]: row for row in result["device_fault_rows"] if row["type"] == "transformer"
    }
    switch_rows = {
        row["name"]: row for row in result["device_fault_rows"] if row["type"] == "switch"
    }
    assert set(transformer_rows) == set(audit["section_loads"])
    assert set(switch_rows) == set(audit["breaker_affected"])
    assert len(result["device_fault_rows"]) == len(transformer_rows) + len(switch_rows)
    assert sum(row["device_count"] for row in transformer_rows.values()) == result["N_T"]
    for name, row in transformer_rows.items():
        assert row["device_count"] == len(audit["section_loads"][name])
        assert row["affected_users"] == 1
    for key, row in switch_rows.items():
        assert row["device_count"] == 1
        assert row["section"] == audit["breaker_owner"][key]
        assert row["affected_users"] == len(audit["breaker_affected"][key])
        assert evidence["breakers"][key]["affected_loads"] == audit["breaker_affected"][key]
    for row in result["device_fault_rows"]:
        kind = "breaker" if row["type"] == "switch" else "transformer"
        duration = params["device_fault"][f"t_repair_{kind}_h"] + (
            params["line_fault"]["t_manual_isolation_h"] if kind == "breaker" else 0
        )
        assert row["t_repair_h"] == pytest.approx(duration)
        assert row["lambda_per_device"] == params["device_fault"][f"lambda_{kind}"]
        assert row["lambda_N"] == pytest.approx(
            row["device_count"] * row["affected_users"] * row["lambda_per_device"],
            rel=1e-12,
            abs=1e-12,
        )
        assert row["lambda_N_r"] == pytest.approx(row["lambda_N"] * duration, rel=1e-12, abs=1e-12)
    for suffix in "FDS":
        assert result[f"SAIFI_{suffix}"] == pytest.approx(
            float(audit["totals"][suffix][0]), rel=1e-12
        )
        assert result[f"SAIDI_{suffix}_h"] == pytest.approx(
            float(audit["totals"][suffix][1]), rel=1e-12
        )
        assert result[f"SAIDI_{suffix}_min"] == pytest.approx(
            60 * result[f"SAIDI_{suffix}_h"], rel=1e-12
        )
    frequency = float(sum(pair[0] for pair in audit["totals"].values()))
    duration = float(sum(pair[1] for pair in audit["totals"].values()))
    assert result["SAIFI"] == pytest.approx(frequency, rel=1e-12)
    assert result["SAIDI_h"] == pytest.approx(duration, rel=1e-12)
    caidi = duration / frequency if frequency else 0
    assert result["CAIDI_h"] == pytest.approx(caidi, rel=1e-12)
    assert result["SAIDI_min"] == pytest.approx(60 * duration, rel=1e-12)
    assert result["CAIDI_min"] == pytest.approx(60 * caidi, rel=1e-12)
    assert result["ASAI"] == pytest.approx(1 - duration / 8760, abs=1e-14)
    assert evidence["missing_length_ids"] == audit["missing"]
    quality = result["data_quality"]
    assert quality["assessment_scope"] == "recorded_line_exposure_subtotal"
    assert quality["missing_length_line_ids"] == audit["missing"]
    assert quality["missing_length_count"] == len(audit["missing"])
    assert quality["line_exposure_complete"] is (not audit["missing"])
    assert quality["full_feeder_point_estimate_available"] is not any(
        value > 0
        for row in audit["missing_terms"].values()
        for field, value in row.items()
        if field != "line_type"
    )
    assert quality["recorded_length_km_by_type"] == pytest.approx(
        {kind: float(value) for kind, value in audit["known_lengths"].items()}, rel=1e-12, abs=1e-12
    )
    terms = {row["line_id"]: row for row in result["missing_exposure_terms"]}
    assert len(terms) == len(result["missing_exposure_terms"]) == len(audit["missing"])
    assert set(terms) == set(audit["missing"])
    for identifier, row in terms.items():
        expected = audit["missing_terms"][identifier]
        assert row["line_type"] == expected["line_type"]
        for field, value in expected.items():
            if field != "line_type":
                assert row[field] == pytest.approx(float(value), rel=1e-12, abs=1e-12)
    assert not scorer._scientific_contract_issues(result, result)
    return audit


@pytest.mark.parametrize("tie_open", [True, False])
def test_branched_tie_and_shared_low_voltage(tmp_path, tie_open):
    path = fixture_xml(tmp_path, tie_open=tie_open)
    model = builder.build_model(builder.read_cim(path))
    result, evidence = builder.calculate(model, PARAMS)
    validate_independently(path, PARAMS, result, evidence)
    assert len(result["fault_rows"]) == 4
    assert result["SAIFI_F"] == pytest.approx(0.75 / 3)
    assert result["SAIDI_F_h"] == pytest.approx(0.75 * 4.5 / 3)
    assert result["SAIFI_S"] == pytest.approx(0.185 / 3)
    assert result["SAIDI_S_h"] == pytest.approx(1.005 / 3)
    assert result["SAIFI_D"] == pytest.approx((0.11 if tie_open else 0.09) / 3)
    assert evidence["breakers"]["head"]["affected_loads"] == ["load_a", "load_b", "load_root"]
    assert evidence["breakers"]["tie"]["affected_loads"] == []
    assert evidence["breakers"]["branch_a"]["affected_loads"] == (["load_a"] if tie_open else [])
    assert evidence["breakers"]["tie"]["section"] == "n3"


def test_downstream_scheduled_users_and_missing_exposure(tmp_path):
    path = fixture_xml(
        tmp_path,
        extra=[
            ("Breaker", "tail_breaker", ("n4", "n8"), {}),
            ("ACLineSegment", "unknown_span", ("n8", "n9"), {"IdentifiedObject.Length": "  "}),
            ("PowerTransformer", "tail_load", ("n9", "tail_lv"), {}),
        ],
    )
    result, evidence = builder.calculate(builder.build_model(builder.read_cim(path)), PARAMS)
    validate_independently(path, PARAMS, result, evidence)
    assert evidence["sections"]["n3"]["local_loads"] == ["load_a"]
    assert evidence["sections"]["n3"]["affected_loads"] == ["load_a", "tail_load"]
    assert evidence["missing_length_ids"] == ["unknown_span"]
    assert next(row for row in result["scheduled_rows"] if row["name"] == "n3")["users"] == 2
    assert next(row for row in result["fault_rows"] if row["name"] == "n8")["length_km"] == 0


@pytest.mark.parametrize("kind", ["PowerTransformer", "ACLineSegment"])
def test_disconnected_data_rejected(tmp_path, kind):
    path = fixture_xml(
        tmp_path,
        extra=[(kind, "orphan", ("island_a", "island_b"), {"IdentifiedObject.Length": "100"})],
    )
    with pytest.raises(ValueError, match="Disconnected assigned"):
        builder.build_model(builder.read_cim(path))


@pytest.mark.parametrize("kind", ["Disconnector", "Fuse"])
def test_switch_normal_state_precedes_actual_and_stays_inside_section(tmp_path, kind):
    path = fixture_xml(
        tmp_path,
        extra=[
            (kind, "tap", ("n4", "n7"), {}),
            ("PowerTransformer", "tap_load", ("n7", "tap_lv"), {}),
        ],
    )
    objects = builder.read_cim(path)
    result, evidence = builder.calculate(builder.build_model(objects), PARAMS)
    validate_independently(path, PARAMS, result, evidence)
    assert "tap_load" in evidence["sections"]["n3"]["local_loads"]
    assert len(result["fault_rows"]) == 4
    objects["tap"]["attrs"]["Switch.normalOpen"] = "true"
    with pytest.raises(ValueError, match="Disconnected assigned"):
        builder.build_model(objects)


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_invalid_lengths_rejected(tmp_path, value):
    objects = builder.read_cim(fixture_xml(tmp_path))
    objects["trunk"]["attrs"]["IdentifiedObject.Length"] = value
    with pytest.raises(ValueError, match="Invalid line length"):
        builder.build_model(objects)


def test_normal_state_required_and_explicit_head_supported(tmp_path):
    objects = builder.read_cim(fixture_xml(tmp_path))
    objects["feeder"]["attrs"]["Feeder.NormalHeadTerminal"] = "head_terminal_0"
    result, _ = builder.calculate(builder.build_model(objects), PARAMS)
    assert result["N_T"] == 3
    objects["branch_a"]["attrs"].pop("Switch.normalOpen")
    with pytest.raises(ValueError, match="Switch.normalOpen"):
        builder.build_model(objects)


def test_parallel_breakers_and_internal_breaker_no_double_count(tmp_path):
    path = fixture_xml(
        tmp_path,
        extra=[
            ("Breaker", "parallel", ("n2", "n3"), {}),
            ("Breaker", "internal", ("n3", "n4"), {}),
        ],
    )
    result, evidence = builder.calculate(builder.build_model(builder.read_cim(path)), PARAMS)
    validate_independently(path, PARAMS, result, evidence)
    assert evidence["breakers"]["branch_a"]["affected_loads"] == []
    assert evidence["breakers"]["parallel"]["affected_loads"] == []
    assert evidence["breakers"]["internal"]["affected_loads"] == []


def test_order_and_display_names_do_not_change_reference(tmp_path):
    objects = builder.read_cim(fixture_xml(tmp_path))
    first, _ = builder.calculate(builder.build_model(objects), PARAMS)
    for identifier, obj in objects.items():
        if obj["kind"] != "Feeder":
            obj["attrs"]["IdentifiedObject.name"] = "changed-" + identifier
    second, _ = builder.calculate(
        builder.build_model(dict(reversed(list(objects.items())))), PARAMS
    )
    assert first == second


def test_zero_rates_and_mixed_type_section(tmp_path):
    path = fixture_xml(
        tmp_path,
        extra=[
            ("ACLineSegment", "mixed", ("n4", "n7"), {"IdentifiedObject.Length": "1000"}),
        ],
    )
    model = builder.build_model(builder.read_cim(path))
    result, evidence = builder.calculate(model, PARAMS)
    validate_independently(path, PARAMS, result, evidence)
    row = next(row for row in result["fault_rows"] if row["name"] == "n3")
    assert row["length_km"] == 3
    assert row["lambda_i"] == pytest.approx(0.5)
    params = json.loads(json.dumps(PARAMS))
    for values in params.values():
        for key in values:
            if key.startswith("lambda_"):
                values[key] = 0
    result, evidence = builder.calculate(model, params)
    validate_independently(path, params, result, evidence)
    assert result["SAIFI"] == result["SAIDI_h"] == result["CAIDI_h"] == 0
    assert result["ASAI"] == 1


@pytest.mark.parametrize("defect", ["source", "transformer_end", "line_type"])
def test_ambiguous_or_unsupported_input_fails(tmp_path, defect):
    objects = builder.read_cim(fixture_xml(tmp_path))
    if defect == "source":
        objects["bus"]["kind"] = "Junction"
        message = "requires one"
    elif defect == "transformer_end":
        objects["load_a_end_1"]["attrs"]["TransformerEnd.BaseVoltage"] = "voltage"
        message = "one feeder-voltage"
    else:
        objects["cable"]["attrs"]["PowerSystemResource.PSRType"] = "unknown"
        message = "Unsupported line type"
    with pytest.raises(ValueError, match=message):
        builder.build_model(objects)


def test_true_input_conservation_and_independent_formulas():
    inputs = Path(
        os.environ.get(
            "POWER_FEEDER_INPUT_DIR",
            ROOT
            / "task-data-hf/extracted/engineering/power_10kv_feeder_reliability_001/base/input",
        )
    )
    if not (inputs / "gis.null.xml").exists():
        pytest.skip("Set POWER_FEEDER_INPUT_DIR to the original input archive")
    params = json.loads((inputs / "params.json").read_text())
    result, evidence = builder.calculate(
        builder.build_model(builder.read_cim(inputs / "gis.null.xml")), params
    )
    audit = validate_independently(inputs / "gis.null.xml", params, result, evidence)
    assert audit["N_T"] == 117
    assert sum(map(len, audit["section_lines"].values())) == 1232
    assert len(audit["breaker_affected"]) == 51
    assert len(audit["open_breakers"]) == 2
    assert audit["known_lengths"] == {"overhead": Decimal("71.81"), "cable": Decimal("0.2")}
    assert len(audit["missing"]) == 34
    assert len(audit["groups"]) == 50
    reference_path = os.environ.get("POWER_FEEDER_REFERENCE")
    if reference_path:
        assert json.loads(Path(reference_path).read_text()) == result
    report_path = os.environ.get("POWER_FEEDER_AUDIT_REPORT")
    if report_path:
        Path(report_path).write_text(
            json.dumps(
                audit,
                indent=2,
                sort_keys=True,
                default=lambda value: sorted(value) if isinstance(value, set) else str(value),
            )
            + "\n"
        )


def test_scorer_weights_tolerances_and_one_to_one_matching(tmp_path):
    result, _ = builder.calculate(
        builder.build_model(builder.read_cim(fixture_xml(tmp_path))), PARAMS
    )
    assert scorer.REL_TOL == 0.05
    assert scorer.ASAI_ABS_TOL == 1e-4
    for table in scorer.TABLE_NAMES:
        rows = list(reversed(json.loads(json.dumps(result[table]))))
        for index, row in enumerate(rows):
            row["section"] = f"renamed-{index}"
        issues = []
        correct, total = scorer._score_table(rows, result[table], table, issues)
        assert correct == total == sum(map(len, rows))
        assert not issues
        rows.append(dict(rows[0]))
        correct, total = scorer._score_table(rows, result[table], table, [])
        assert correct < total
        assert total == sum(map(len, rows))
    assert scorer._same_number(1.04, 1, field="SAIFI")
    assert not scorer._same_number(1.06, 1, field="SAIFI")
    assert not scorer._same_number(float("nan"), 1, field="SAIFI")
    assert not scorer._same_number(1e-8, 0, field="SAIFI")


def test_public_contract_and_card_match():
    contract = (TASK / "scientific_contract.md").read_text()
    card = json.loads((TASK / "task_card.json").read_text())
    assert contract in card["taskPrompt"]
    assert builder.VERSION in contract
    assert "staged reliability reference data" not in card["taskPrompt"]
    assert "ANON_" not in contract
    assert "113" not in contract and "117" not in contract
    assert card["referenceFiles"][0]["name"] == f"reliability_indices.{builder.VERSION}.json"


def test_runtime_requires_versioned_reference_and_executes_scorer(tmp_path, monkeypatch):
    from tasks.engineering.power_10kv_feeder_reliability_001 import main as task_main

    result, _ = builder.calculate(
        builder.build_model(builder.read_cim(fixture_xml(tmp_path))), PARAMS
    )
    assert set(result) == set(task_main.EXPECTED_TOP_LEVEL_KEYS)
    assert (TASK / "scientific_contract.md").read_text() in task_main.config.task_description
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    output = tmp_path / "reliability_indices.json"
    output.write_text(json.dumps(result))
    (reference_dir / "reliability_indices.json").write_text(json.dumps(result))
    config = SimpleNamespace(
        metadata={
            "variant_name": "base",
            "output_file": str(output),
            "reference_dir": str(reference_dir),
        }
    )
    monkeypatch.setattr(task_main, "EVAL_TMP_DIR", str(tmp_path / "evaluator"))

    class LocalSession:
        def __init__(self):
            self.interface = self
            self.commands = []

        async def file_exists(self, path):
            return Path(path).is_file()

        async def directory_exists(self, path):
            return Path(path).is_dir()

        async def create_dir(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

        async def write_file(self, path, text):
            Path(path).write_text(text)

        async def run_command(self, command, timeout=None, check=False):
            self.commands.append(command)
            process = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout, check=check
            )
            return {
                "return_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }

    session = LocalSession()
    with pytest.raises(RuntimeError, match="Controlled reference"):
        asyncio.run(task_main.evaluate(config, session))
    assert not session.commands
    (reference_dir / task_main.REFERENCE_FILENAME).write_text(json.dumps(result))
    assert asyncio.run(task_main.evaluate(config, session)) == [1.0]
    assert len(session.commands) == 1


@pytest.fixture
def evaluation_case(tmp_path, monkeypatch):
    from tasks.engineering.power_10kv_feeder_reliability_001 import main as task_main

    result, _ = builder.calculate(
        builder.build_model(builder.read_cim(fixture_xml(tmp_path))), PARAMS
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference = reference_dir / task_main.REFERENCE_FILENAME
    reference.write_text(json.dumps(result))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output = output_dir / "reliability_indices.json"
    output.write_text(json.dumps(result))
    config = SimpleNamespace(
        metadata={
            "variant_name": "base",
            "output_file": str(output),
            "reference_dir": str(reference_dir),
        }
    )
    monkeypatch.setattr(task_main, "EVAL_TMP_DIR", str(tmp_path / "verifier"))

    class CheckOnlySession:
        def __init__(self):
            self.interface = self
            self.commands = []
            self.probes = []
            self.results = []

        async def file_exists(self, path):
            self.probes.append(path)
            return Path(path).is_file()

        async def create_dir(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

        async def write_file(self, path, text):
            Path(path).write_text(text)

        async def run_command(self, command, *, check=True):
            self.commands.append((command, check))
            process = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=check
            )
            result = {
                "return_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
            self.results.append(result)
            return result

    return SimpleNamespace(
        main=task_main,
        config=config,
        session=CheckOnlySession(),
        output=output,
        reference=reference,
        result=result,
    )


@pytest.mark.parametrize(
    "reference_defect",
    [
        "missing",
        "directory",
        "bad_json",
        "nonobject",
        "empty_object",
        "missing_scalar",
        "bad_row",
        "bad_affine",
        "nonfinite",
    ],
)
@pytest.mark.parametrize("candidate_present", [False, True])
def test_evaluate_reference_errors_precede_candidate(
    evaluation_case, reference_defect, candidate_present
):
    case = evaluation_case
    if not candidate_present:
        case.output.unlink()
    if reference_defect == "missing":
        case.reference.unlink()
    elif reference_defect == "directory":
        case.reference.unlink()
        case.reference.mkdir()
    elif reference_defect == "bad_json":
        case.reference.write_text("{")
    elif reference_defect == "nonobject":
        case.reference.write_text("[]")
    elif reference_defect == "empty_object":
        case.reference.write_text("{}")
    else:
        ref = json.loads(case.reference.read_text())
        if reference_defect == "missing_scalar":
            del ref["SAIFI"]
        elif reference_defect == "bad_row":
            ref["fault_rows"][0] = {}
        elif reference_defect == "bad_affine":
            ref["missing_exposure_terms"] = [{}]
        else:
            ref["SAIFI"] = float("nan")
        case.reference.write_text(json.dumps(ref))
    with pytest.raises(RuntimeError, match="Controlled reference|Verifier command failed"):
        asyncio.run(case.main.evaluate(case.config, case.session))
    assert case.session.probes == [str(case.reference)]
    if case.session.results:
        response = json.loads(case.session.results[0]["stdout"])
        assert response["status"] == "evaluation_error"
        assert response["reason"].startswith("reference_error:")
        assert case.session.results[0]["return_code"] != 0


@pytest.mark.parametrize(
    "candidate",
    ["missing", "directory", "bad_json", "invalid_utf8", "array", "null", "number", "empty_object"],
)
def test_evaluate_malformed_candidate_is_normal_zero(evaluation_case, candidate):
    case = evaluation_case
    if candidate == "missing":
        case.output.unlink()
    elif candidate == "directory":
        case.output.unlink()
        case.output.mkdir()
    elif candidate == "invalid_utf8":
        case.output.write_bytes(b"\xff")
    else:
        case.output.write_text(
            {"bad_json": "{", "array": "[]", "null": "null", "number": "42", "empty_object": "{}"}[
                candidate
            ]
        )
    assert asyncio.run(case.main.evaluate(case.config, case.session)) == [0.0]
    assert case.session.probes == [str(case.reference)]
    assert case.session.commands[0][1] is False
    response = json.loads(case.session.results[0]["stdout"])
    assert response["status"] == ("ok" if candidate == "empty_object" else "candidate_error")
    assert response["score"] == 0 and response["passed"] is False
    assert case.session.results[0]["return_code"] == 0


@pytest.mark.parametrize(
    "failure",
    [
        "nonzero_empty",
        "nonzero_valid_json",
        "invalid_json",
        "empty_stdout",
        "nonobject_json",
        "missing_schema",
        "reported_runtime_error",
        "nan_score",
        "out_of_range_score",
        "boolean_score",
    ],
)
def test_evaluate_verifier_failures_are_errors(evaluation_case, monkeypatch, failure):
    case = evaluation_case
    payload = {"status": "ok", "score": 1, "passed": True, "reason": "ok", "issues": []}
    result = {"return_code": 0, "stdout": json.dumps(payload), "stderr": ""}
    if failure == "nonzero_empty":
        result.update(return_code=1, stdout="", stderr="program failed")
    elif failure == "nonzero_valid_json":
        result["return_code"] = 7
    elif failure == "invalid_json":
        result["stdout"] = "not JSON"
    elif failure == "empty_stdout":
        result["stdout"] = ""
    elif failure == "nonobject_json":
        result["stdout"] = "[]"
    elif failure == "missing_schema":
        result["stdout"] = '{"score": 0}'
    else:
        if failure == "reported_runtime_error":
            payload.update(status="evaluation_error", score=0, passed=False)
        else:
            payload["score"] = {
                "nan_score": float("nan"),
                "out_of_range_score": 2,
                "boolean_score": False,
            }[failure]
        result["stdout"] = json.dumps(payload)

    async def broken_command(command, *, check=True):
        return result

    monkeypatch.setattr(case.session, "run_command", broken_command)
    with pytest.raises(RuntimeError):
        asyncio.run(case.main.evaluate(case.config, case.session))


@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError, TypeError])
def test_evaluate_transport_or_internal_typeerror_is_not_retried(
    evaluation_case, monkeypatch, error_type
):
    case = evaluation_case
    calls = []

    async def broken_command(command, *, timeout=None, check=True):
        calls.append((timeout, check))
        raise error_type("verifier transport or program failure")

    monkeypatch.setattr(case.session, "run_command", broken_command)
    with pytest.raises(error_type, match="transport or program failure"):
        asyncio.run(case.main.evaluate(case.config, case.session))
    assert calls == [(300.0, False)]


def test_evaluate_installed_remote_run_command_check_fallback(evaluation_case):
    from cua_bench.computers.remote import RemoteDesktopSession

    case = evaluation_case
    case.session.shell_command = case.session.run_command
    case.session.run_command = RemoteDesktopSession.run_command.__get__(case.session)
    assert asyncio.run(case.main.evaluate(case.config, case.session)) == [1.0]
    assert len(case.session.commands) == 1
    assert case.session.commands[0][1] is False
    assert json.loads(case.session.results[0]["stdout"])["status"] == "ok"


def test_verifier_program_failure_exits_nonzero(evaluation_case, monkeypatch, capsys):
    case = evaluation_case

    def broken_table(*args, **kwargs):
        raise RuntimeError("scorer program failure")

    monkeypatch.setattr(scorer, "_score_table", broken_table)
    monkeypatch.setattr(scorer, "EXPECTED_INPUT_MD5S", {})
    monkeypatch.setattr(
        scorer.sys, "argv", ["verify", "--agent", str(case.output), "--ref", str(case.reference)]
    )
    assert scorer.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "evaluation_error"
    assert "scorer program failure" in payload["reason"]


def test_evaluate_uploaded_verifier_program_crash(evaluation_case, monkeypatch):
    case = evaluation_case
    monkeypatch.setattr(
        case.main, "_read_script", lambda name: "raise RuntimeError('verifier crashed')\n"
    )
    with pytest.raises(RuntimeError, match="Verifier command failed"):
        asyncio.run(case.main.evaluate(case.config, case.session))
    assert case.session.results[0]["return_code"] != 0
    assert "verifier crashed" in case.session.results[0]["stderr"]


@pytest.fixture
def affine_case(tmp_path):
    path = fixture_xml(
        tmp_path,
        extra=[
            ("Breaker", "tail_breaker", ("n6", "n8"), {}),
            ("ACLineSegment", "unknown_tail", ("n8", "n9"), {}),
            ("PowerTransformer", "tail_load", ("n9", "tail_lv"), {}),
        ],
    )
    tree = ET.parse(path)
    for element in tree.getroot():
        if element.get(RDF + "ID") in {"trunk", "cable"}:
            element.find(CIM + "IdentifiedObject.Length").text = "  "
    tree.write(path, encoding="utf-8", xml_declaration=True)
    result, evidence = builder.calculate(builder.build_model(builder.read_cim(path)), PARAMS)
    validate_independently(path, PARAMS, result, evidence)
    return path, result, evidence


@pytest.mark.parametrize(
    "lengths_km",
    [
        {"trunk": 0.025, "cable": 2.5, "unknown_tail": 0.07},
        {"trunk": 0.12, "cable": 0.03, "unknown_tail": 1.7},
        {"trunk": 2.0, "cable": 0.8, "unknown_tail": 0.001},
        {"trunk": 0.001, "cable": 0.001, "unknown_tail": 0.001},
    ],
)
@pytest.mark.parametrize("tie_closed", [False, True])
def test_affine_completion_for_synthetic_positive_lengths(affine_case, lengths_km, tie_closed):
    path, _, _ = affine_case
    tree = ET.parse(path)
    if tie_closed:
        for element in tree.getroot():
            if element.get(RDF + "ID") == "tie":
                element.find(CIM + "Switch.normalOpen").text = "false"
        tree.write(path, encoding="utf-8", xml_declaration=True)
    partial, evidence = builder.calculate(builder.build_model(builder.read_cim(path)), PARAMS)
    validate_independently(path, PARAMS, partial, evidence)
    assert not partial["data_quality"]["line_exposure_complete"]
    assert not partial["data_quality"]["full_feeder_point_estimate_available"]
    terms = {row["line_id"]: row for row in partial["missing_exposure_terms"]}
    assert terms["trunk"]["SAIFI_F_per_km"] == pytest.approx(0.1)
    assert terms["cable"]["SAIFI_F_per_km"] == pytest.approx(0.05)
    assert terms["unknown_tail"]["SAIFI_F_per_km"] == pytest.approx(0.025)
    for element in tree.getroot():
        identifier = element.get(RDF + "ID")
        if identifier in lengths_km:
            length = element.find(CIM + "IdentifiedObject.Length")
            if length is None:
                length = ET.SubElement(element, CIM + "IdentifiedObject.Length")
            length.text = str(Decimal(str(lengths_km[identifier])) * 1000)
    complete_path = path.with_name("synthetic-complete.xml")
    tree.write(complete_path, encoding="utf-8", xml_declaration=True)
    complete, complete_evidence = builder.calculate(
        builder.build_model(builder.read_cim(complete_path)), PARAMS
    )
    oracle = validate_independently(complete_path, PARAMS, complete, complete_evidence)
    assert complete["data_quality"]["line_exposure_complete"] is True
    assert complete["data_quality"]["full_feeder_point_estimate_available"] is True
    assert complete["data_quality"]["missing_length_line_ids"] == []
    assert complete["missing_exposure_terms"] == []
    components = {}
    for suffix in "FS":
        frequency = partial[f"SAIFI_{suffix}"] + sum(
            row[f"SAIFI_{suffix}_per_km"] * lengths_km[row["line_id"]]
            for row in partial["missing_exposure_terms"]
        )
        duration = partial[f"SAIDI_{suffix}_h"] + sum(
            row[f"SAIDI_{suffix}_h_per_km"] * lengths_km[row["line_id"]]
            for row in partial["missing_exposure_terms"]
        )
        assert frequency == pytest.approx(float(oracle["totals"][suffix][0]), rel=1e-12)
        assert duration == pytest.approx(float(oracle["totals"][suffix][1]), rel=1e-12)
        assert complete[f"SAIDI_{suffix}_min"] == pytest.approx(60 * duration, rel=1e-12)
        components[suffix] = (frequency, duration)
    assert complete["SAIFI_D"] == partial["SAIFI_D"]
    assert complete["SAIDI_D_h"] == partial["SAIDI_D_h"]
    frequency = partial["SAIFI_D"] + sum(pair[0] for pair in components.values())
    duration = partial["SAIDI_D_h"] + sum(pair[1] for pair in components.values())
    assert complete["SAIFI"] == pytest.approx(frequency, rel=1e-12)
    assert complete["SAIDI_h"] == pytest.approx(duration, rel=1e-12)
    assert complete["CAIDI_h"] == pytest.approx(duration / frequency, rel=1e-12)
    assert complete["CAIDI_min"] == pytest.approx(60 * duration / frequency, rel=1e-12)
    assert complete["ASAI"] == pytest.approx(1 - duration / 8760, abs=1e-14)
    assert scorer.score_submission(complete, complete)["passed"]


def test_zero_missing_exposure_does_not_mean_measured(affine_case):
    path, _, _ = affine_case
    params = json.loads(json.dumps(PARAMS))
    for values in params.values():
        for key in values:
            if key.startswith("lambda_"):
                values[key] = 0
    result, evidence = builder.calculate(builder.build_model(builder.read_cim(path)), params)
    validate_independently(path, params, result, evidence)
    assert result["data_quality"]["line_exposure_complete"] is False
    assert result["data_quality"]["full_feeder_point_estimate_available"] is True
    assert len(result["missing_exposure_terms"]) == 3
    assert result["CAIDI_h"] == 0
    assert result["ASAI"] == 1
    assert scorer.score_submission(result, result)["passed"]


def test_all_measured_scope_gate_and_recorded_zero(tmp_path):
    objects = builder.read_cim(fixture_xml(tmp_path))
    objects["cable"]["attrs"]["IdentifiedObject.Length"] = "0"
    result, _ = builder.calculate(builder.build_model(objects), PARAMS)
    assert result["missing_exposure_terms"] == []
    assert result["data_quality"]["missing_length_count"] == 0
    assert result["data_quality"]["recorded_length_km_by_type"]["cable"] == 0
    assert scorer.score_submission(result, result)["passed"]
    for field in ("line_exposure_complete", "full_feeder_point_estimate_available"):
        candidate = json.loads(json.dumps(result))
        candidate["data_quality"][field] = False
        assert scorer.score_submission(candidate, result)["score"] == 0


def test_unknown_length_with_no_affected_loads_still_has_term(tmp_path):
    path = fixture_xml(
        tmp_path,
        extra=[
            ("Breaker", "unloaded_branch", ("n4", "n8"), {}),
            ("ACLineSegment", "unknown_unloaded_line", ("n8", "n9"), {}),
        ],
    )
    result, evidence = builder.calculate(builder.build_model(builder.read_cim(path)), PARAMS)
    validate_independently(path, PARAMS, result, evidence)
    assert result["data_quality"]["line_exposure_complete"] is False
    assert result["data_quality"]["full_feeder_point_estimate_available"] is True
    (row,) = result["missing_exposure_terms"]
    assert row["line_id"] == "unknown_unloaded_line"
    assert all(row[field] == 0 for field in scorer.COEFFICIENT_FIELDS)
    candidate = json.loads(json.dumps(result))
    candidate["missing_exposure_terms"] = []
    assert scorer.score_submission(candidate, result)["score"] == 0


@pytest.mark.parametrize("field", sorted(scorer.QUALITY_FIELDS))
def test_every_data_quality_field_is_required(affine_case, field):
    _, reference, _ = affine_case
    candidate = json.loads(json.dumps(reference))
    del candidate["data_quality"][field]
    score = scorer.score_submission(candidate, reference)
    assert score["score"] == 0
    assert not score["scientific_contract"]["passed"]
    assert score["detail"]["leaf_score"] == 1


@pytest.mark.parametrize(
    "defect",
    [
        "missing_quality",
        "full_scope",
        "complete_claim",
        "point_estimate_claim",
        "string_boolean",
        "missing_id",
        "extra_id",
        "duplicate_id",
        "nonstring_id",
        "count",
        "fractional_count",
        "boolean_count",
        "length_unit",
        "length_type_missing",
        "missing_terms",
        "terms_object",
        "term_missing",
        "term_extra",
        "term_duplicate",
        "term_wrong_id",
        "term_wrong_type",
        "coefficient_missing",
        "coefficient_wrong",
        "coefficient_nan",
        "coefficient_infinity",
        "coefficient_negative",
        "coefficient_boolean",
        "coefficient_string",
        "imputed_length",
        "extra_quality_field",
    ],
)
def test_scientific_gate_rejects_incomplete_or_fabricated_model(affine_case, defect):
    _, reference, _ = affine_case
    candidate = json.loads(json.dumps(reference))
    quality = candidate["data_quality"]
    identifiers = quality["missing_length_line_ids"]
    terms = candidate["missing_exposure_terms"]
    if defect == "missing_quality":
        del candidate["data_quality"]
    elif defect == "full_scope":
        quality["assessment_scope"] = "full_feeder_estimate"
    elif defect == "complete_claim":
        quality["line_exposure_complete"] = True
    elif defect == "point_estimate_claim":
        quality["full_feeder_point_estimate_available"] = True
    elif defect == "string_boolean":
        quality["line_exposure_complete"] = "false"
    elif defect == "missing_id":
        identifiers.pop()
    elif defect == "extra_id":
        identifiers.append("not-a-missing-line")
    elif defect == "duplicate_id":
        identifiers[-1] = identifiers[0]
    elif defect == "nonstring_id":
        identifiers[0] = None
    elif defect == "count":
        quality["missing_length_count"] -= 1
    elif defect == "fractional_count":
        quality["missing_length_count"] += 0.01
    elif defect == "boolean_count":
        quality["missing_length_count"] = True
    elif defect == "length_unit":
        quality["recorded_length_km_by_type"]["overhead"] *= 1000
    elif defect == "length_type_missing":
        del quality["recorded_length_km_by_type"]["cable"]
    elif defect == "missing_terms":
        del candidate["missing_exposure_terms"]
    elif defect == "terms_object":
        candidate["missing_exposure_terms"] = {}
    elif defect == "term_missing":
        terms.pop()
    elif defect == "term_extra":
        terms.append({**terms[0], "line_id": "not-a-missing-line"})
    elif defect == "term_duplicate":
        terms[-1] = dict(terms[0])
    elif defect == "term_wrong_id":
        terms[0]["line_id"] = "not-a-missing-line"
    elif defect == "term_wrong_type":
        terms[0]["line_type"] = "overhead" if terms[0]["line_type"] == "cable" else "cable"
    elif defect == "coefficient_missing":
        del terms[0]["SAIFI_F_per_km"]
    elif defect.startswith("coefficient_"):
        terms[0]["SAIFI_F_per_km"] = {
            "coefficient_wrong": terms[0]["SAIFI_F_per_km"] * 2,
            "coefficient_nan": float("nan"),
            "coefficient_infinity": float("inf"),
            "coefficient_negative": -1,
            "coefficient_boolean": True,
            "coefficient_string": "0.05",
        }[defect]
    elif defect == "imputed_length":
        terms[0]["length_km"] = 0.06
    elif defect == "extra_quality_field":
        quality["imputed"] = True
    else:
        raise AssertionError(defect)
    score = scorer.score_submission(candidate, reference)
    assert score["score"] == 0
    assert not score["passed"]
    assert not score["scientific_contract"]["passed"]
    assert score["detail"]["correct"] == score["detail"]["total"]
    assert score["detail"]["leaf_score"] == 1


def test_affine_order_formatting_and_tolerances(affine_case):
    _, reference, _ = affine_case
    candidate = json.loads(json.dumps(reference, sort_keys=True))
    candidate["data_quality"]["missing_length_line_ids"] = [
        f" {identifier} "
        for identifier in reversed(candidate["data_quality"]["missing_length_line_ids"])
    ]
    candidate["data_quality"]["assessment_scope"] = " recorded_line_exposure_subtotal "
    candidate["data_quality"]["missing_length_count"] = 3.0
    candidate["missing_exposure_terms"].reverse()
    for row in candidate["missing_exposure_terms"]:
        row["line_id"] = f" {row['line_id']} "
        row["line_type"] = f" {row['line_type']} "
    score = scorer.score_submission(candidate, reference)
    expected_leaves = 17 + sum(len(row) for table in scorer.TABLE_NAMES for row in reference[table])
    assert score["passed"] and score["score"] == 1
    assert score["detail"]["total"] == expected_leaves
    candidate["missing_exposure_terms"][0]["SAIFI_F_per_km"] *= 1.04
    assert scorer.score_submission(candidate, reference)["passed"]
    candidate["missing_exposure_terms"][0]["SAIFI_F_per_km"] *= 1.1
    assert scorer.score_submission(candidate, reference)["score"] == 0


def test_affine_cli_gate_cannot_return_full_score(affine_case, tmp_path, monkeypatch, capsys):
    _, reference, _ = affine_case
    candidate = json.loads(json.dumps(reference))
    candidate.pop("missing_exposure_terms")
    agent_path = tmp_path / "candidate.json"
    ref_path = tmp_path / "reference.json"
    agent_path.write_text(json.dumps(candidate))
    ref_path.write_text(json.dumps(reference))
    monkeypatch.setattr(
        scorer.sys, "argv", ["verify", "--agent", str(agent_path), "--ref", str(ref_path)]
    )
    monkeypatch.setattr(scorer, "EXPECTED_INPUT_MD5S", {})
    assert scorer.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["score"] == 0
    assert payload["detail"]["leaf_score"] == 1
    assert not payload["scientific_contract"]["passed"]


def test_partial_only_reference_cannot_be_used(affine_case):
    _, reference, _ = affine_case
    stale = {key: value for key, value in reference.items() if key not in scorer.SCIENTIFIC_FIELDS}
    with pytest.raises(ValueError, match="affine scientific contract"):
        scorer.score_submission(reference, stale)
