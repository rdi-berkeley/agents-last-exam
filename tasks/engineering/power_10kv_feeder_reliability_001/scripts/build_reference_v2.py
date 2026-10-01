"""Build the benchmark-owned normal-state affine reliability model from CIM.

Requires networkx. This maintenance utility is not staged as solver input.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import networkx as nx


VERSION = "normal-affine-v2.1.0"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
SWITCHES = {"Breaker", "Disconnector", "Fuse"}
CONDUCTORS = SWITCHES | {"ACLineSegment", "BusbarSection", "Junction"}
LINE_TYPES = {"PD_10100000": "overhead", "PD_20100000": "cable"}


def read_cim(path: Path) -> dict:
    objects = {}
    for element in ET.parse(path).getroot():
        identifier = element.get(RDF + "ID")
        if not identifier or identifier in objects:
            raise ValueError("CIM objects require unique rdf:ID")
        objects[identifier] = {
            "kind": element.tag.rsplit("}", 1)[-1],
            "attrs": {
                child.tag.rsplit("}", 1)[-1]: child.get(RDF + "resource", child.text or "")
                .strip()
                .removeprefix("#")
                for child in element
            },
        }
    return objects


def normal_open(obj: dict) -> bool:
    value = obj["attrs"].get("Switch.normalOpen", "").lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError("Missing or invalid Switch.normalOpen")
    return value in {"true", "1"}


def build_model(objects: dict) -> dict:
    feeders = [
        key
        for key, obj in objects.items()
        if obj["kind"] == "Feeder"
        and obj["attrs"].get("Feeder.IsCurrentFeeder", "").lower() in {"true", "1"}
    ]
    if len(feeders) != 1:
        raise ValueError("Expected one current feeder")
    feeder = feeders[0]
    assigned = {
        key
        for key, obj in objects.items()
        if feeder
        in (
            obj["attrs"].get("Equipment.EquipmentContainer"),
            obj["attrs"].get("Equipment.EquipmentBelongtoFeeder"),
        )
    }
    lines = {key for key in assigned if objects[key]["kind"] == "ACLineSegment"}
    loads = {key for key in assigned if objects[key]["kind"] == "PowerTransformer"}
    voltages = {objects[key]["attrs"]["ConductingEquipment.BaseVoltage"] for key in lines}
    if len(voltages) != 1 or not loads:
        raise ValueError("Expected one feeder line voltage and positive transformer count")
    voltage = voltages.pop()
    station = objects[feeder]["attrs"].get("Feeder.NormalEnergizingSubstation")
    contained = {station} if station else set()
    while True:
        expanded = contained | {
            key
            for key, obj in objects.items()
            if any(
                obj["attrs"].get(field) in contained
                for field in (
                    "Equipment.EquipmentContainer",
                    "VoltageLevel.Substation",
                    "Bay.VoltageLevel",
                )
            )
        }
        if expanded == contained:
            break
        contained = expanded
    station_equipment = {
        key
        for key in contained
        if key in objects
        and objects[key]["kind"] in CONDUCTORS
        and objects[key]["attrs"].get("ConductingEquipment.BaseVoltage") == voltage
    }
    equipment = {key for key in assigned if objects[key]["kind"] in CONDUCTORS}
    equipment |= station_equipment
    terminals = defaultdict(list)
    terminal_nodes = {}
    for key, obj in objects.items():
        if obj["kind"] == "Terminal":
            attrs = obj["attrs"]
            node = attrs["Terminal.ConnectivityNode"]
            if node not in objects or objects[node]["kind"] != "ConnectivityNode":
                raise ValueError(f"Invalid connectivity node on terminal {key}")
            terminals[attrs["Terminal.ConductingEquipment"]].append(node)
            terminal_nodes[key] = node
    graph = nx.Graph()
    opened = set()
    for key in sorted(equipment):
        obj = objects[key]
        nodes = terminals[key]
        if not nodes or (obj["kind"] in SWITCHES | {"ACLineSegment"} and len(nodes) != 2):
            raise ValueError(f"Invalid terminal count for {key}")
        graph.add_nodes_from(nodes)
        if obj["kind"] in SWITCHES and normal_open(obj):
            opened.add(key)
            continue
        graph.add_edges_from((key, node) for node in nodes)
    primary = defaultdict(list)
    for obj in objects.values():
        if obj["kind"] == "PowerTransformerEnd":
            attrs = obj["attrs"]
            if attrs.get("TransformerEnd.BaseVoltage") == voltage:
                primary[attrs["PowerTransformerEnd.PowerTransformer"]].append(
                    terminal_nodes[attrs["TransformerEnd.Terminal"]]
                )
    load_nodes = {}
    for key in sorted(loads):
        if len(primary[key]) != 1:
            raise ValueError(f"Expected one feeder-voltage transformer end for {key}")
        load_nodes[key] = primary[key][0]
        graph.add_edge(key, load_nodes[key])
    head = objects[feeder]["attrs"].get("Feeder.NormalHeadTerminal")
    if head:
        source = terminal_nodes[head]
    else:
        busbars = [key for key in station_equipment if objects[key]["kind"] == "BusbarSection"]
        if len(busbars) != 1:
            raise ValueError("Missing head terminal requires one energizing-station voltage busbar")
        source = busbars[0]
    if source not in graph:
        raise ValueError("Source absent from normal-state graph")
    energized = set(nx.node_connected_component(graph, source))
    disconnected = (loads | lines) - energized
    if disconnected:
        raise ValueError(f"Disconnected assigned lines/transformers: {sorted(disconnected)}")
    breakers = {key for key in equipment if objects[key]["kind"] == "Breaker"}
    if any(not set(terminals[key]) & energized for key in breakers):
        raise ValueError("Breaker has no energized endpoint")
    graph = graph.subgraph(energized).copy()
    interior = graph.copy()
    interior.remove_nodes_from(breakers)
    sections = {}
    owner = {}
    for members in nx.connected_components(interior):
        nodes = {key for key in members if objects[key]["kind"] == "ConnectivityNode"}
        if not nodes:
            raise ValueError("Section without a connectivity node")
        name = min(nodes)
        sections[name] = set(members)
        owner.update({key: name for key in members})
    section_graph = nx.Graph()
    section_graph.add_nodes_from(sections)
    for key in breakers - opened:
        endpoints = {owner[node] for node in terminals[key]}
        if len(endpoints) == 2:
            section_graph.add_edge(*sorted(endpoints))
    depth = nx.single_source_shortest_path_length(section_graph, owner[source])
    breaker_owner = {}
    for key in sorted(breakers):
        endpoints = {owner[node] for node in terminals[key] if node in owner}
        breaker_owner[key] = min(endpoints, key=lambda name: (-depth[name], name))
    lengths = {}
    missing = []
    for key in sorted(lines):
        attrs = objects[key]["attrs"]
        psr_type = attrs.get("PowerSystemResource.PSRType")
        if psr_type not in LINE_TYPES:
            raise ValueError(f"Unsupported line type for {key}: {psr_type}")
        kind = LINE_TYPES[psr_type]
        raw = attrs.get("IdentifiedObject.Length", "").strip()
        length = float(raw) / 1000 if raw else 0.0
        if not math.isfinite(length) or length < 0:
            raise ValueError(f"Invalid line length for {key}")
        lengths[key] = (kind, length)
        if not raw:
            missing.append(key)
    return {
        "feeder": objects[feeder]["attrs"]["IdentifiedObject.name"],
        "graph": graph,
        "source": source,
        "sections": sections,
        "owner": owner,
        "lines": lines,
        "loads": loads,
        "lengths": lengths,
        "missing": missing,
        "breakers": breakers,
        "opened": opened,
        "breaker_owner": breaker_owner,
        "load_nodes": load_nodes,
    }


def interrupted_loads(model: dict, removed: set) -> set:
    graph = nx.subgraph_view(model["graph"], filter_node=lambda node: node not in removed)
    reachable = (
        set(nx.node_connected_component(graph, model["source"]))
        if model["source"] in graph
        else set()
    )
    return model["loads"] - reachable


def calculate(model: dict, params: dict) -> tuple[dict, dict]:
    fault = params["line_fault"]
    scheduled = params["scheduled_outage"]
    device = params["device_fault"]
    isolation = fault["t_manual_isolation_h"]
    result = {"feeder": model["feeder"], "N_T": len(model["loads"])}
    result.update({key: [] for key in ("fault_rows", "device_fault_rows", "scheduled_rows")})
    evidence = {"version": VERSION, "source": model["source"], "sections": {}, "breakers": {}}
    for name, members in sorted(model["sections"].items()):
        affected = interrupted_loads(model, members)
        local = members & model["loads"]
        lines = members & model["lines"]
        lengths = {
            kind: math.fsum(
                model["lengths"][key][1] for key in lines if model["lengths"][key][0] == kind
            )
            for kind in ("overhead", "cable")
        }
        rate = math.fsum(lengths[kind] * fault[f"lambda_{kind}"] for kind in lengths)
        weighted_rate = rate * len(affected)
        duration = fault["t_repair_h"] + isolation
        result["fault_rows"].append(
            {
                "section": name,
                "name": name,
                "length_km": math.fsum(lengths.values()),
                "lambda_i": rate,
                "perm_users": len(affected),
                "t_iso_h": isolation,
                "r_i_h": duration,
                "lambda_N": weighted_rate,
                "lambda_N_r": weighted_rate * duration,
            }
        )
        result["scheduled_rows"].append(
            {
                "section": name,
                "name": name,
                "length_km": math.fsum(lengths.values()),
                "users": len(affected),
                "lambda_N": len(affected)
                * math.fsum(
                    lengths[kind] * scheduled[f"lambda_scheduled_{kind}"] for kind in lengths
                ),
                "lambda_N_r": len(affected)
                * math.fsum(
                    lengths[kind]
                    * scheduled[f"lambda_scheduled_{kind}"]
                    * scheduled[f"t_scheduled_{kind}_h"]
                    for kind in lengths
                ),
            }
        )
        if local:
            weighted_rate = len(local) * device["lambda_transformer"]
            result["device_fault_rows"].append(
                {
                    "type": "transformer",
                    "section": name,
                    "name": name,
                    "device_count": len(local),
                    "lambda_per_device": device["lambda_transformer"],
                    "affected_users": 1,
                    "t_repair_h": device["t_repair_transformer_h"],
                    "lambda_N": weighted_rate,
                    "lambda_N_r": weighted_rate * device["t_repair_transformer_h"],
                }
            )
        evidence["sections"][name] = {
            "members": sorted(members),
            "lines": sorted(lines),
            "local_loads": sorted(local),
            "affected_loads": sorted(affected),
            "lengths_km": lengths,
        }
    for key in sorted(model["breakers"]):
        affected = interrupted_loads(model, {key})
        duration = device["t_repair_breaker_h"] + isolation
        weighted_rate = device["lambda_breaker"] * len(affected)
        result["device_fault_rows"].append(
            {
                "type": "switch",
                "section": model["breaker_owner"][key],
                "name": key,
                "device_count": 1,
                "lambda_per_device": device["lambda_breaker"],
                "affected_users": len(affected),
                "t_repair_h": duration,
                "lambda_N": weighted_rate,
                "lambda_N_r": weighted_rate * duration,
            }
        )
        evidence["breakers"][key] = {
            "section": model["breaker_owner"][key],
            "open": key in model["opened"],
            "affected_loads": sorted(affected),
        }
    for suffix, table in (("F", "fault_rows"), ("D", "device_fault_rows"), ("S", "scheduled_rows")):
        result[f"SAIFI_{suffix}"] = (
            math.fsum(row["lambda_N"] for row in result[table]) / result["N_T"]
        )
        result[f"SAIDI_{suffix}_h"] = (
            math.fsum(row["lambda_N_r"] for row in result[table]) / result["N_T"]
        )
        result[f"SAIDI_{suffix}_min"] = 60 * result[f"SAIDI_{suffix}_h"]
    result["SAIFI"] = math.fsum(result[f"SAIFI_{suffix}"] for suffix in "FDS")
    result["SAIDI_h"] = math.fsum(result[f"SAIDI_{suffix}_h"] for suffix in "FDS")
    result["SAIDI_min"] = result["SAIDI_h"] * 60
    result["CAIDI_h"] = result["SAIDI_h"] / result["SAIFI"] if result["SAIFI"] else 0.0
    result["CAIDI_min"] = result["CAIDI_h"] * 60
    result["ASAI"] = 1 - result["SAIDI_h"] / 8760
    evidence["missing_length_ids"] = model["missing"]
    evidence["conservation"] = {
        "transformers": sum(
            len(section["local_loads"]) for section in evidence["sections"].values()
        ),
        "lines": sum(len(section["lines"]) for section in evidence["sections"].values()),
        "breakers": len(evidence["breakers"]),
        "lengths_km": {
            kind: math.fsum(
                length for line_kind, length in model["lengths"].values() if line_kind == kind
            )
            for kind in ("overhead", "cable")
        },
    }
    result["missing_exposure_terms"] = []
    for key in model["missing"]:
        kind = model["lengths"][key][0]
        affected_count = len(evidence["sections"][model["owner"][key]]["affected_loads"])
        fault_coefficient = affected_count * fault[f"lambda_{kind}"] / result["N_T"]
        scheduled_coefficient = (
            affected_count * scheduled[f"lambda_scheduled_{kind}"] / result["N_T"]
        )
        result["missing_exposure_terms"].append(
            {
                "line_id": key,
                "line_type": kind,
                "SAIFI_F_per_km": fault_coefficient,
                "SAIDI_F_h_per_km": fault_coefficient * (isolation + fault["t_repair_h"]),
                "SAIFI_S_per_km": scheduled_coefficient,
                "SAIDI_S_h_per_km": scheduled_coefficient * scheduled[f"t_scheduled_{kind}_h"],
            }
        )
    result["data_quality"] = {
        "assessment_scope": "recorded_line_exposure_subtotal",
        "line_exposure_complete": not model["missing"],
        "missing_length_line_ids": list(model["missing"]),
        "missing_length_count": len(model["missing"]),
        "recorded_length_km_by_type": dict(evidence["conservation"]["lengths_km"]),
        "full_feeder_point_estimate_available": not any(
            row[field] > 0
            for row in result["missing_exposure_terms"]
            for field in (
                "SAIFI_F_per_km",
                "SAIDI_F_h_per_km",
                "SAIFI_S_per_km",
                "SAIDI_S_h_per_km",
            )
        ),
    }
    return result, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    objects = read_cim(args.input / "gis.null.xml")
    result, evidence = calculate(
        build_model(objects), json.loads((args.input / "params.json").read_text())
    )
    evidence["sha256"] = {
        name: hashlib.sha256((args.input / name).read_bytes()).hexdigest()
        for name in ("gis.null.xml", "gis.null.svg", "params.json")
    }
    evidence["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for path, data in ((args.output, result), (args.evidence, evidence)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "version": VERSION,
                "conservation": evidence["conservation"],
                "rows": {
                    key: len(value) for key, value in result.items() if isinstance(value, list)
                },
            }
        )
    )


if __name__ == "__main__":
    main()
