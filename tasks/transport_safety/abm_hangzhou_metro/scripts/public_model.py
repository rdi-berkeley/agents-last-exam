"""Executable evaluator for the public Hangzhou discrete-event contract v2."""

from __future__ import annotations

import csv
import hashlib
import heapq
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

CONTRACT_VERSION = "hangzhou-discrete-event-v2"


@dataclass(frozen=True)
class Service:
    stations: tuple[str, ...]
    running: tuple[int, ...]


@dataclass(frozen=True)
class Route:
    legs: tuple[tuple[str, int, int], ...]
    minutes: int


@dataclass
class Simulation:
    trips: list[tuple[str, str, str, int, int]]
    ends: list[int | None]
    transfers: list[int]
    diagnostics: dict[str, int]


def clock_minutes(value: str) -> int:
    hours, minutes = map(int, value.split(":"))
    if hours < 0 or not 0 <= minutes < 60:
        raise ValueError(f"invalid operation time: {value}")
    return 60 * hours + minutes


def haversine_meters(left, right) -> float:
    lon_left, lat_left = map(math.radians, left)
    lon_right, lat_right = map(math.radians, right)
    chord = (
        math.sin((lat_right - lat_left) / 2) ** 2
        + math.cos(lat_left) * math.cos(lat_right) * math.sin((lon_right - lon_left) / 2) ** 2
    )
    return 2 * 6371008.8 * math.asin(math.sqrt(min(1.0, max(0.0, chord))))


def chainage(point, coordinates) -> float:
    best = (math.inf, math.inf)
    distance = 0.0
    for left, right in itertools.pairwise(coordinates):
        delta_lon, delta_lat = right[0] - left[0], right[1] - left[1]
        squared = delta_lon**2 + delta_lat**2
        fraction = (
            max(
                0.0,
                min(
                    1.0,
                    ((point[0] - left[0]) * delta_lon + (point[1] - left[1]) * delta_lat) / squared,
                ),
            )
            if squared
            else 0.0
        )
        error = (point[0] - left[0] - fraction * delta_lon) ** 2 + (
            point[1] - left[1] - fraction * delta_lat
        ) ** 2
        segment = haversine_meters(left, right)
        best = min(best, (error, distance + fraction * segment))
        distance += segment
    if not math.isfinite(best[1]):
        raise ValueError("line geometry has no segments")
    return best[1]


def load_network(input_dir: Path):
    if nx.__version__ != "3.6.1":
        raise RuntimeError("public route tie order requires NetworkX 3.6.1")
    params = json.loads((input_dir / "network_config/operation_parameters.json").read_text())
    positive = (
        "metro_speed_kmh",
        "train_capacity",
        "default_headway_minutes",
        "simulation_step_minutes",
        "station_stop_time_minutes",
        "transfer_time_minutes",
    )
    if any(not math.isfinite(params[key]) or params[key] <= 0 for key in positive):
        raise ValueError("operation parameters must be finite and positive")
    if params["simulation_step_minutes"] != 1:
        raise ValueError("v2 requires the staged one-minute simulation step")
    for key in (
        "train_capacity",
        "station_stop_time_minutes",
        "transfer_time_minutes",
        "station_entry_flow_limit_per_minute",
    ):
        if type(params[key]) is not int or params[key] < 0:
            raise ValueError(f"invalid integer operation parameter: {key}")
    if not math.isfinite(params["block_length_meters"]) or params["block_length_meters"] < 0:
        raise ValueError("invalid block length")
    route_params = params["route_choice"]
    if type(route_params["k_shortest_paths"]) is not int or route_params["k_shortest_paths"] <= 0:
        raise ValueError("invalid route count")
    for key in ("logit_epsilon", "logit_beta_transfer_times", "logit_beta_travel_time"):
        if not math.isfinite(route_params[key]):
            raise ValueError("non-finite route utility")
    lines = json.loads((input_dir / "gis/hangzhou_lines.json").read_text())
    stations = json.loads((input_dir / "gis/hangzhou_stations.json").read_text())
    geometries = {
        feature["properties"]["linename"]: feature["geometry"]["coordinates"]
        for feature in lines["features"]
    }
    points = {
        (feature["properties"]["linename"], feature["properties"]["stationnames"]): feature[
            "geometry"
        ]["coordinates"]
        for feature in stations["features"]
    }
    sequences = defaultdict(list)
    names = {}
    with (input_dir / "network_config/station_sequence.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            service = row["line_id"]
            names[service] = f"{row['line_name_cn']}({row['direction']})"
            sequences[service].append((int(row["station_order"]), row["station_name"]))
    services = {}
    edge_data = {}
    for service, rows in sorted(sequences.items()):
        rows.sort()
        if [order for order, _ in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"invalid station order: {service}")
        ordered = tuple(station for _, station in rows)
        if len(ordered) < 2 or len(set(ordered)) != len(ordered):
            raise ValueError(f"invalid station sequence: {service}")
        geometry = geometries[names[service]]
        positions = [chainage(points[(names[service], station)], geometry) for station in ordered]
        if any(right <= left for left, right in itertools.pairwise(positions)):
            raise ValueError(f"non-increasing GIS chainage: {service}")
        running = tuple(
            max(1, math.ceil((right - left) / (params["metro_speed_kmh"] * 1000 / 60)))
            for left, right in itertools.pairwise(positions)
        )
        services[service] = Service(ordered, running)
        for index, (origin, destination) in enumerate(itertools.pairwise(ordered)):
            edge_data[(service, index)] = (
                origin,
                destination,
                {
                    "weight": running[index] + params["station_stop_time_minutes"],
                    "service": service,
                    "index": index,
                },
            )
    graph = nx.DiGraph()
    graph.add_nodes_from(
        sorted({station for service in services.values() for station in service.stations})
    )
    graph.add_nodes_from(sorted(edge_data))
    for edge_node, (origin, destination, attributes) in sorted(edge_data.items()):
        graph.add_edge(origin, edge_node, **attributes)
        graph.add_edge(edge_node, destination, weight=0)
    if not services or not nx.is_strongly_connected(graph):
        raise ValueError("disconnected or empty service network")
    return params, services, graph


def route_options(graph, origin: str, destination: str, params) -> tuple[Route, ...]:
    if origin == destination:
        return (Route((), 0),)
    paths = nx.shortest_simple_paths(graph, origin, destination, weight="weight")
    routes = []
    for path in itertools.islice(paths, params["route_choice"]["k_shortest_paths"]):
        legs = []
        total = 0
        for left, right in itertools.pairwise(path):
            edge = graph[left][right]
            if "service" not in edge:
                continue
            service, index = edge["service"], edge["index"]
            total += edge["weight"]
            if legs and legs[-1][0] == service and legs[-1][2] == index:
                legs[-1] = (service, legs[-1][1], index + 1)
            else:
                legs.append((service, index, index + 1))
        total -= len(legs) * params["station_stop_time_minutes"]
        total += (len(legs) - 1) * params["transfer_time_minutes"]
        routes.append(Route(tuple(legs), total))
    return tuple(routes)


def choose_route(options: tuple[Route, ...], params, seed: int, row_index: int) -> Route:
    route_params = params["route_choice"]
    utilities = [
        route_params["logit_beta_travel_time"] * route.minutes
        + route_params["logit_beta_transfer_times"] * max(0, len(route.legs) - 1)
        for route in options
    ]
    maximum = max(utilities)
    weights = [math.exp(utility - maximum) for utility in utilities]
    digest = hashlib.sha256(f"{CONTRACT_VERSION}:{seed}:{row_index}".encode("ascii")).digest()
    uniform = int.from_bytes(digest[:8], "big") / 2**64
    threshold = uniform * sum(weights)
    cumulative = 0.0
    for route, weight in zip(options, weights, strict=True):
        cumulative += weight
        if threshold < cumulative:
            return route
    return options[-1]


def departures(params) -> list[int]:
    start = clock_minutes(params["operation_hours"]["start"])
    end = clock_minutes(params["operation_hours"]["end"])
    if start >= end:
        raise ValueError("operation end must be later than start on the extended day")
    periods = [
        (clock_minutes(period["start"]), clock_minutes(period["end"]), period["headway_minutes"])
        for period in params["peak_periods"]
    ]
    if any(
        not math.isfinite(headway) or headway <= 0 or right <= left
        for left, right, headway in periods
    ):
        raise ValueError("invalid headway period")
    nominal = float(start)
    result = []
    while nominal <= end:
        result.append(math.ceil(nominal))
        headway = next(
            (headway for left, right, headway in periods if left <= nominal < right),
            params["default_headway_minutes"],
        )
        nominal += headway
    clearance = max(
        1, math.ceil(params["block_length_meters"] / (params["metro_speed_kmh"] * 1000 / 60))
    )
    if any(
        right - left < params["station_stop_time_minutes"] + clearance
        for left, right in itertools.pairwise(result)
    ):
        raise ValueError("timetable violates the public dwell-plus-block clearance bound")
    return result


def simulate(input_dir: Path, seed: int) -> Simulation:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    params, services, graph = load_network(input_dir)
    timetable = departures(params)
    trips = []
    with (input_dir / "data/afc_hangzhou.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["card_id", "o_station", "d_station", "start_time", "end_time"]:
            raise ValueError("invalid AFC schema")
        for row in reader:
            trip = (
                row["card_id"],
                row["o_station"],
                row["d_station"],
                int(row["start_time"]),
                int(row["end_time"]),
            )
            if trip[1] not in graph or trip[2] not in graph or min(trip[3:]) < 0:
                raise ValueError("invalid AFC station or time")
            trips.append(trip)
    if not trips or len({trip[:4] for trip in trips}) != len(trips):
        raise ValueError("empty AFC or duplicate trip identities")
    route_cache = {}
    selected = []
    entrants = defaultdict(list)
    for row_index, trip in enumerate(trips):
        od = trip[1:3]
        if od not in route_cache:
            route_cache[od] = route_options(graph, *od, params)
        selected.append(choose_route(route_cache[od], params, seed, row_index))
        entrants[trip[1]].append(row_index)
    ends = [None] * len(trips)
    transfers = [max(0, len(route.legs) - 1) for route in selected]
    next_leg = [0] * len(trips)
    queues = defaultdict(list)
    admitted = 0
    for positions in entrants.values():
        positions.sort(key=lambda position: (trips[position][3], position))
        minute, used = timetable[0], 0
        for position in positions:
            if trips[position][3] > minute:
                minute, used = trips[position][3], 0
            limit = params["station_entry_flow_limit_per_minute"]
            if limit and used >= limit:
                minute, used = minute + 1, 0
            used += 1
            if minute > clock_minutes(params["operation_hours"]["end"]):
                continue
            admitted += 1
            if selected[position].legs:
                service, board, _ = selected[position].legs[0]
                queues[(service, board)].append((minute + 1, position))
            else:
                ends[position] = minute + 2
    for queue in queues.values():
        heapq.heapify(queue)
    events = []
    for service, definition in services.items():
        for train, departure in enumerate(timetable):
            minute = departure
            events.append((minute, 1, service, train, 0))
            for station, running in enumerate(definition.running, 1):
                minute += running
                events.append((minute, 0, service, train, station))
                if station < len(definition.stations) - 1:
                    minute += params["station_stop_time_minutes"]
                    events.append((minute, 1, service, train, station))
    onboard = defaultdict(lambda: defaultdict(list))
    loads = defaultdict(int)
    peak_load = 0
    boardings = 0
    alightings = 0
    for minute, phase, service, train, station in sorted(events):
        train_key = (service, train)
        if phase == 0:
            passengers = onboard[train_key].pop(station, ())
            loads[train_key] -= len(passengers)
            alightings += len(passengers)
            for position in passengers:
                next_leg[position] += 1
                if next_leg[position] == len(selected[position].legs):
                    ends[position] = minute + 1
                else:
                    next_service, board, _ = selected[position].legs[next_leg[position]]
                    heapq.heappush(
                        queues[(next_service, board)],
                        (minute + params["transfer_time_minutes"], position),
                    )
        else:
            queue = queues[(service, station)]
            while queue and queue[0][0] <= minute and loads[train_key] < params["train_capacity"]:
                _, position = heapq.heappop(queue)
                _, _, alight = selected[position].legs[next_leg[position]]
                onboard[train_key][alight].append(position)
                loads[train_key] += 1
                boardings += 1
            peak_load = max(peak_load, loads[train_key])
    completed = sum(end is not None for end in ends)
    queued = sum(len(queue) for queue in queues.values())
    if any(loads.values()) or boardings != alightings or completed + queued != admitted:
        raise RuntimeError("passenger conservation failure in evaluator")
    return Simulation(
        trips,
        ends,
        transfers,
        {
            "demand": len(trips),
            "admitted": admitted,
            "completed": completed,
            "unserved": len(trips) - completed,
            "queued_at_close": queued,
            "not_admitted": len(trips) - admitted,
            "boardings": boardings,
            "alightings": alightings,
            "peak_train_load": peak_load,
            "capacity": params["train_capacity"],
            "od_pairs": len(route_cache),
            "directed_services": len(services),
            "trains": len(services) * len(timetable),
        },
    )
