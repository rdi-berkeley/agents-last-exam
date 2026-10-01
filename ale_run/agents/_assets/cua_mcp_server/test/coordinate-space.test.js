import assert from "node:assert/strict";
import test from "node:test";

import {
  coordinateDescription,
  fromAbsoluteCoordinate,
  parseCoordinateSpace,
  toAbsoluteCoordinate,
} from "../src/coordinate-space.js";

const SCREEN = { width: 1024, height: 768 };

test("normalized remains the default coordinate space", () => {
  assert.equal(parseCoordinateSpace(undefined), "normalized");
  assert.deepEqual(
    toAbsoluteCoordinate([500, 500], "normalized", SCREEN),
    { x: 512, y: 384 }
  );
});

test("pixel coordinates pass through without scaling", () => {
  assert.equal(parseCoordinateSpace("PIXEL"), "pixel");
  assert.deepEqual(
    toAbsoluteCoordinate([688, 320], "pixel"),
    { x: 688, y: 320 }
  );
  assert.deepEqual(
    fromAbsoluteCoordinate(688, 320, "pixel"),
    [688, 320]
  );
  assert.match(coordinateDescription("pixel"), /screenshot pixels/);
});

test("normalized cursor coordinates retain the existing contract", () => {
  assert.deepEqual(
    fromAbsoluteCoordinate(512, 384, "normalized", SCREEN),
    [500, 500]
  );
});

test("unknown coordinate spaces fail fast", () => {
  assert.throws(() => parseCoordinateSpace("auto"), /expected.*normalized.*pixel/);
});
