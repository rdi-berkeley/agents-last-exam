import csv
import hashlib

import FreeCAD
import MeshPart
import numpy as np

from terrain_sampling import NativeTerrainSampler


def refresh_profile(frame):
    from freecad.road.geometry.profile.profile import Profile

    alignment = frame.getParentGroup().getParentGroup()
    model = alignment.Model
    fingerprint = hashlib.sha256(repr((
        [element.to_dict() for element in model.get_elements()],
        model.coordinate_system.to_dict(),
        [(terrain.Name, terrain.Points, terrain.Faces) for terrain in frame.Terrains],
    )).encode()).hexdigest()
    if getattr(frame.Proxy, "_cailian_signature", None) == fingerprint:
        return
    alignment.Proxy.execute(alignment)
    length = model.get_length()
    start = model.get_sta_start()
    end = start + length
    stations = set(np.arange(start, end, 0.25).tolist() + [end])
    stations.update(np.arange(start, end, 20).tolist())
    for terrain in frame.Terrains:
        if terrain.Operations:
            raise ValueError("Use the supplied unchanged source terrain")
        for edge in alignment.Shape.Edges:
            parameters = [edge.FirstParameter, edge.LastParameter] + MeshPart.findSectionParameters(edge, terrain.Mesh, FreeCAD.Vector(0, 0, 1))
            for parameter in parameters:
                if edge.FirstParameter <= parameter <= edge.LastParameter:
                    point = edge.valueAt(parameter).sub(alignment.Placement.Base).multiply(0.001)
                    station, offset = model.get_station_offset((point.x, point.y), input_system="current")
                    if station is not None and start <= station <= end and abs(offset) < 1e-5:
                        stations.add(station)
    ordered = sorted(stations)
    ordered = [station for index, station in enumerate(ordered) if index == 0 or station - ordered[index - 1] > 1e-8]
    profiles = []
    for terrain in frame.Terrains:
        sampler = NativeTerrainSampler(terrain)
        data = []
        for station in ordered:
            coordinates = model.coordinate_system.transform_from_system(model.get_point_at_station(station))
            height = sampler.elevation(coordinates[1], coordinates[0])
            data.append({"pvi": {"station": station, "elevation": height}})
        profiles.append(Profile(terrain.Label, "Raw existing ground from native Points/Faces", data))
    frame.Proxy._cailian_signature = fingerprint
    model.profiles.surface_profiles = profiles
    alignment.Model = model


def install(read_only=False):
    from freecad.road.objects.profile_frame import ProfileFrame
    from freecad.road.objects.alignment import Alignment
    from freecad.road.objects.geo_object import GeoObject
    from freecad.road.viewproviders.view_group import ViewProviderGroup

    if getattr(ProfileFrame, "_cailian_installed", False):
        return
    original_execute = ProfileFrame.execute
    original_alignment_changed = Alignment.onChanged

    def on_changed(proxy, obj, prop):
        if obj.Document.Restoring:
            return
        GeoObject.onChanged(proxy, obj, prop)
        if prop == "Terrains":
            obj.touch()

    def execute(proxy, obj):
        if obj.Document.Restoring:
            return
        group = obj.getParentGroup()
        alignment = group.getParentGroup() if group else None
        if alignment is None or not alignment.Model:
            return
        if not read_only and obj.Terrains:
            refresh_profile(obj)
        original_execute(proxy, obj)

    def alignment_changed(proxy, obj, prop):
        original_alignment_changed(proxy, obj, prop)
        if prop == "Model" and not obj.Document.Restoring and hasattr(obj, "Group"):
            for group in obj.Group:
                if getattr(group.Proxy, "Type", None) == "Road::Profiles":
                    for frame in group.Group:
                        frame.touch()

    def dump_icon(proxy):
        return getattr(proxy, "icon", getattr(proxy, "Icon", ""))

    def load_icon(proxy, state):
        proxy.icon = state or ""
        proxy.Icon = proxy.icon

    ProfileFrame.onChanged = on_changed
    ProfileFrame.execute = execute
    ProfileFrame._cailian_installed = True
    Alignment.onChanged = alignment_changed
    ViewProviderGroup.dumps = dump_icon
    ViewProviderGroup.loads = load_icon
    ViewProviderGroup.getIcon = dump_icon


def export_metrics(alignment, destination):
    model = alignment.Model
    start = model.get_sta_start()
    end = start + model.get_length()
    profiles = model.get_profiles().surface_profiles
    if len(profiles) != 1:
        raise ValueError("Select one associated existing-ground profile")
    with open(destination, "w") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["Station", "X", "Y", "Z"])
        for station in np.arange(start, end, 20).tolist() + [end]:
            northing, easting = model.coordinate_system.transform_from_system(model.get_point_at_station(station))
            height = profiles[0].get_elevation_at_station(station)
            writer.writerow([f"{value:.9f}" for value in (station, northing, easting, height)])
