const COORD_MAX = 1000;
const VALID_SPACES = new Set(["normalized", "pixel"]);

export function parseCoordinateSpace(value) {
  const coordinateSpace = (value || "normalized").trim().toLowerCase();
  if (!VALID_SPACES.has(coordinateSpace)) {
    throw new Error(
      `Invalid CUA_COORDINATE_SPACE=${JSON.stringify(coordinateSpace)}; ` +
      'expected "normalized" or "pixel".'
    );
  }
  return coordinateSpace;
}

export function coordinateDescription(coordinateSpace) {
  return coordinateSpace === "pixel"
    ? "(x, y) coordinates in screenshot pixels."
    : "(x, y) coordinates normalized to [0, 1000].";
}

export function toAbsoluteCoordinate(coordinate, coordinateSpace, screen) {
  if (coordinateSpace === "pixel") {
    return { x: Math.round(coordinate[0]), y: Math.round(coordinate[1]) };
  }
  return {
    x: Math.round((coordinate[0] / COORD_MAX) * screen.width),
    y: Math.round((coordinate[1] / COORD_MAX) * screen.height),
  };
}

export function fromAbsoluteCoordinate(absX, absY, coordinateSpace, screen) {
  if (coordinateSpace === "pixel") {
    return [Math.round(absX), Math.round(absY)];
  }
  return [
    Math.round((absX / screen.width) * COORD_MAX),
    Math.round((absY / screen.height) * COORD_MAX),
  ];
}
