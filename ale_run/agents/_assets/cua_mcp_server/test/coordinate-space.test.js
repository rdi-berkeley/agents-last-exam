import assert from "node:assert/strict";
import test from "node:test";
import { coordinateDescription, fromAbsoluteCoordinate, parseCoordinateSpace, toAbsoluteCoordinate } from "../src/coordinate-space.js";

const screen = { width: 1024, height: 768 };

test("normalized is the bridge default; pixel mode is explicit", () => {
  assert.equal(parseCoordinateSpace(undefined), "normalized");
  assert.equal(parseCoordinateSpace(" PIXEL "), "pixel");
  assert.throws(() => parseCoordinateSpace("auto"), /CUA_COORDINATE_SPACE/);
  assert.match(coordinateDescription("pixel"), /screenshot pixels/);
});

test("normalized coordinates scale to the current screen and stay inside it", () => {
  assert.deepEqual(toAbsoluteCoordinate([500, 500], "normalized", screen), { x: 512, y: 384 });
  assert.deepEqual(toAbsoluteCoordinate([1000, 1000], "normalized", screen), { x: 1023, y: 767 });
  assert.deepEqual(toAbsoluteCoordinate([500, 500], "normalized", { width: 1920, height: 1080 }), { x: 960, y: 540 });
  assert.deepEqual(fromAbsoluteCoordinate(512, 384, "normalized", screen), [500, 500]);
});

test("pixel coordinates and cursor positions are not rescaled", () => {
  assert.deepEqual(toAbsoluteCoordinate([688.2, 320.4], "pixel"), { x: 688, y: 320 });
  assert.deepEqual(fromAbsoluteCoordinate(688, 320, "pixel"), [688, 320]);
});
