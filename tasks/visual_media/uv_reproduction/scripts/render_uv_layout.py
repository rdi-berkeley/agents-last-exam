"""Export a UV wireframe from the face-referenced coordinates of an OBJ."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


def read_uv_faces(path: Path) -> tuple[list[tuple[float, float]], list[list[int]]]:
    coordinates: list[tuple[float, float]] = []
    faces: list[list[int]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            if fields[0] == "vt":
                if len(fields) < 3:
                    raise ValueError(f"{path}:{line_number}: incomplete UV coordinate")
                coordinate = (float(fields[1]), float(fields[2]))
                if not all(math.isfinite(value) for value in coordinate):
                    raise ValueError(f"{path}:{line_number}: nonfinite UV coordinate")
                coordinates.append(coordinate)
            elif fields[0] == "f":
                face = []
                for corner in fields[1:]:
                    indices = corner.split("/")
                    if len(indices) < 2 or not indices[1]:
                        raise ValueError(f"{path}:{line_number}: face has no UV index")
                    index = int(indices[1])
                    if index == 0:
                        raise ValueError(f"{path}:{line_number}: zero UV index")
                    index = index - 1 if index > 0 else len(coordinates) + index
                    if not 0 <= index < len(coordinates):
                        raise ValueError(f"{path}:{line_number}: UV index out of range")
                    face.append(index)
                if len(face) < 3:
                    raise ValueError(f"{path}:{line_number}: face has fewer than three corners")
                faces.append(face)
    if not faces:
        raise ValueError(f"{path}: no UV-bearing faces")
    return coordinates, faces


def render_uv_layout(path: Path, size: int = 1024) -> Image.Image:
    coordinates, faces = read_uv_faces(path)
    if size < 2:
        raise ValueError("size must be at least two pixels")
    used = {index for face in faces for index in face}
    if any(not 0 <= value <= 1 for index in used for value in coordinates[index]):
        raise ValueError("UV guide requires face coordinates within the unit square")
    image = Image.new("RGB", (size, size), (16, 18, 24))
    draw = ImageDraw.Draw(image)
    for grid_index in range(33):
        position = round(grid_index * (size - 1) / 32)
        draw.line([(position, 0), (position, size - 1)], fill=(50, 52, 60))
        draw.line([(0, position), (size - 1, position)], fill=(50, 52, 60))
    for face in faces:
        points = [
            (
                round(coordinates[index][0] * (size - 1)),
                round((1 - coordinates[index][1]) * (size - 1)),
            )
            for index in face
        ]
        draw.line(points + points[:1], fill=(255, 201, 56), width=1)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    image = render_uv_layout(args.obj)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        image.save(stream, format="PNG")
    print(
        json.dumps(
            {
                "source_obj": str(args.obj.resolve()),
                "source_sha256": hashlib.sha256(args.obj.read_bytes()).hexdigest(),
                "output": str(args.output.resolve()),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "size": list(image.size),
                "coordinates": "x=round(u*1023), y=round((1-v)*1023); all face edges",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
